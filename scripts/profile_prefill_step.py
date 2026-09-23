#!/usr/bin/env python3
"""Profile one agent-shaped prefill (small fresh suffix on a long cached prefix).

WARNING: maintenance window only. On 2026-09-23 /start_profile on the live
GB10 serve killed VllmWorker-0 (likely unified-memory exhaustion from the
profiler buffers; ~6 GB free) and every request failed until a restart. Do
not unpack the resulting trace (~84 MB gz) on the GB10 host either.

Needs the serve booted with the torch profiler armed, e.g. in EXTRA_ARGS:
  --profiler-config.profiler torch
  --profiler-config.torch_profiler_dir /root/.cache/vllm/profiles
  --profiler-config.active_iterations 8
The profiler is idle until /start_profile, but starting it can kill the worker.

Steps: (1) warm a CTX-token prefix (not profiled), (2) /start_profile, send the
same prefix plus FRESH new tokens, /stop_profile, (3) summarize the newest
trace in PROFILE_DIR: top GPU kernels, GPU busy vs wall, and the largest GPU
idle gaps (host-bound time). Rank 1's trace is written on the worker node.

  scripts/profile_prefill_step.py [--ctx 40000] [--fresh 2500] [--summarize-only]
Env: API_KEY / VLLM_API_KEY, GLM53_URL (default http://127.0.0.1:8888),
     GLM53_MODEL (default GLM-5.3-Flash),
     PROFILE_DIR (default ~/.cache/vllm-glm53-flash/profiles).
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import time
import urllib.request
from pathlib import Path

URL = os.environ.get("GLM53_URL", "http://127.0.0.1:8888")
MODEL = os.environ.get("GLM53_MODEL", "GLM-5.3-Flash")
PROFILE_DIR = Path(os.environ.get(
    "PROFILE_DIR", str(Path.home() / ".cache/vllm-glm53-flash/profiles")))


def _headers():
    key = os.environ.get("API_KEY") or os.environ.get("VLLM_API_KEY")
    h = {"Content-Type": "application/json"}
    if key:
        h["Authorization"] = f"Bearer {key}"
    return h


def _post(path, body=None, timeout=1800):
    req = urllib.request.Request(URL + path, data=json.dumps(body or {}).encode(),
                                 headers=_headers(), method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw) if raw.strip() else {}


def _text(n_tokens, tag):
    # ~1.3 tokens per line item below; distinct tag keeps prefixes unique per run.
    line = "Entry {i}: node N{m} checksum CK-{i:06d} temp {t} C fan {f} pct ok. "
    n = max(1, int(n_tokens / 14))
    return f"[{tag}] " + "".join(
        line.format(i=i, m=i % 7, t=40 + i * 7 % 23, f=30 + i * 13 % 60) for i in range(n))


def _chat(content, max_tokens):
    t0 = time.monotonic()
    d = _post("/v1/chat/completions", {
        "model": MODEL, "max_tokens": max_tokens, "temperature": 0,
        "messages": [{"role": "user", "content": content}]})
    return d.get("usage", {}), time.monotonic() - t0


def run(ctx, fresh):
    tag = f"prof-{int(time.time())}"
    prefix = _text(ctx, tag)
    usage, dt = _chat(prefix + "\nReply OK.", 1)
    print(f"warm prefix: prompt_tokens={usage.get('prompt_tokens')} in {dt:.1f}s")
    before = set(PROFILE_DIR.rglob("*.json*")) if PROFILE_DIR.exists() else set()
    _post("/start_profile")
    try:
        usage, dt = _chat(prefix + "\n" + _text(fresh, tag + "-fresh") + "\nReply OK.", 16)
    finally:
        _post("/stop_profile")
    print(f"profiled turn: prompt_tokens={usage.get('prompt_tokens')} "
          f"cached={usage.get('prompt_tokens_details') or {}} wall={dt:.2f}s")
    for _ in range(60):  # traces are flushed asynchronously
        new = set(PROFILE_DIR.rglob("*.json*")) - before if PROFILE_DIR.exists() else set()
        if new:
            return sorted(new, key=lambda p: p.stat().st_mtime)
        time.sleep(2)
    raise SystemExit(f"no new trace under {PROFILE_DIR}")


def _load(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        return json.load(f)


def summarize(path, top=15, gap_ms=5.0):
    events = _load(path)
    events = events.get("traceEvents", events) if isinstance(events, dict) else events
    kernels = [e for e in events if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    cpu_ops = [e for e in events if e.get("ph") == "X" and e.get("cat") in ("cpu_op", "user_annotation", "python_function")]
    if not kernels:
        print(f"{path}: no GPU kernel events")
        return
    by_name = collections.Counter()
    count = collections.Counter()
    for k in kernels:
        by_name[k["name"][:90]] += k["dur"]
        count[k["name"][:90]] += 1
    start = min(k["ts"] for k in kernels)
    end = max(k["ts"] + k["dur"] for k in kernels)
    # Union of kernel intervals = GPU busy time (streams may overlap).
    busy, cur_s, cur_e = 0.0, None, None
    gaps = []
    for k in sorted(kernels, key=lambda e: e["ts"]):
        s, e = k["ts"], k["ts"] + k["dur"]
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                busy += cur_e - cur_s
                if (s - cur_e) / 1000 >= gap_ms:
                    gaps.append(((s - cur_e) / 1000, cur_e))
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    busy += cur_e - cur_s
    wall = end - start
    print(f"\n== {path.name}")
    print(f"GPU span {wall / 1e3:.1f} ms, busy {busy / 1e3:.1f} ms ({100 * busy / wall:.0f}%), "
          f"idle {(wall - busy) / 1e3:.1f} ms; kernels {len(kernels)}")
    print(f"top {top} GPU kernels by total time:")
    for name, dur in by_name.most_common(top):
        print(f"  {dur / 1e3:9.1f} ms  x{count[name]:<6d} {name}")
    gaps.sort(reverse=True)
    print(f"GPU idle gaps >= {gap_ms} ms: {len(gaps)}, total {sum(g for g, _ in gaps):.1f} ms")
    for g, at in gaps[:8]:
        near = [c["name"][:60] for c in cpu_ops if c["ts"] <= at <= c["ts"] + c["dur"]]
        print(f"  {g:7.1f} ms gap; host inside: {', '.join(near[-3:]) or '-'}")
    if cpu_ops:
        host = collections.Counter()
        for c in cpu_ops:
            host[c["name"][:70]] += c["dur"]
        print("top host ops (inclusive):")
        for name, dur in host.most_common(8):
            print(f"  {dur / 1e3:9.1f} ms  {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=40000)
    ap.add_argument("--fresh", type=int, default=2500)
    ap.add_argument("--summarize-only", nargs="*", type=Path)
    a = ap.parse_args()
    paths = a.summarize_only if a.summarize_only is not None else run(a.ctx, a.fresh)
    if not paths:
        paths = sorted(PROFILE_DIR.rglob("*.json*"), key=lambda p: p.stat().st_mtime)[-1:]
    for p in paths:
        summarize(p)


if __name__ == "__main__":
    main()
