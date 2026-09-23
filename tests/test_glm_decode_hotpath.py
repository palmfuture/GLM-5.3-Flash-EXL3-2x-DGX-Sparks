#!/usr/bin/env python3
"""CPU checks for overlay/patch_glm_decode_hotpath.py (vLLM #55736 backport).

Applies the overlay to a copy of the five target files, checks it is
idempotent and fails closed on anchor drift, and checks the `token_stride`
helper it installs against the tensor layouts Glm5NextKDA really passes
(column slices of the fused projection), using a stride-only fake tensor so
no torch is needed.

Source tree: GLM53_VLLM_SRC_ROOT (a vllm package root holding the files as
left by the earlier overlays), else the image's installed package.
"""
from __future__ import annotations

import ast
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PATCH = next(p for p in (HERE / "patch_glm_decode_hotpath.py",
                         HERE.parent / "overlay" / "patch_glm_decode_hotpath.py")
             if p.is_file())
spec = importlib.util.spec_from_file_location("glm53_decode_hotpath", PATCH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

SRC = Path(os.environ.get("GLM53_VLLM_SRC_ROOT",
                          "/usr/local/lib/python3.12/dist-packages/vllm"))
FILES = sorted({rel for rel, *_ in mod.HUNKS})


class FakeTensor:
    def __init__(self, shape, stride):
        self.shape, self._stride = tuple(shape), tuple(stride)

    def stride(self):
        return self._stride

    def dim(self):
        return len(self.shape)


def load_token_stride(root: Path):
    path = root / "third_party/flash_linear_attention/ops/fused_recurrent.py"
    src = path.read_text()
    fn = next(n for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name == "token_stride")
    ns = {"torch": None}
    exec(ast.get_source_segment(src, fn), ns)
    return ns["token_stride"]


def run(root: Path) -> subprocess.CompletedProcess:
    env = {**os.environ, "GLM53_VLLM_ROOT": str(root)}
    return subprocess.run([sys.executable, str(PATCH)], env=env,
                          capture_output=True, text=True)


def main() -> int:
    missing = [rel for rel in FILES if not (SRC / rel).is_file()]
    if missing:
        raise SystemExit(f"missing under {SRC}: {missing}")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "vllm"
        for rel in FILES:
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(SRC / rel, root / rel)
        first = run(root)
        assert first.returncode == 0, first.stdout + first.stderr
        patched = {rel: (root / rel).read_text() for rel in FILES}
        for rel, text in patched.items():
            compile(text, rel, "exec")
        for rel, label, old, new in mod.HUNKS:
            assert patched[rel].count(new) == 1, label
        # Restart: a second run verifies without rewriting.
        second = run(root)
        assert second.returncode == 0 and "verified" in second.stdout, second.stdout
        assert all((root / rel).read_text() == patched[rel] for rel in FILES)
        # The duplicate router GEMM is gone; the runner-held gate stays.
        model = patched["models/glm5next/nvidia/model.py"]
        assert "router_logits, _ = self.gate(hidden_states)" not in model
        assert "gate=self.gate," in model
        # Drift fails closed and leaves the file untouched.
        rel, label, old, new = mod.HUNKS[0]
        drifted = patched[rel].replace(new, new.replace("torch.bmm", "torch.bmm_x"), 1)
        (root / rel).write_text(drifted)
        bad = run(root)
        assert bad.returncode != 0 and "drifted" in (bad.stdout + bad.stderr)
        assert (root / rel).read_text() == drifted

        token_stride = load_token_stride(root)
        H, D, T = 16, 128, 9
        P = H * D
        W = 3 * P + H + 2 * D  # in_proj_qkvbfg_a: qkv | beta | f_a | g_a
        # q/k/v: column slices of qkv reshaped to (1, T, H, D).
        assert token_stride(FakeTensor((1, T, H, D), (T * W, W, D, 1))) == W
        # beta: column slice of the projection, unsqueezed to (1, T, H).
        assert token_stride(FakeTensor((1, T, H), (T * W, W, 1))) == W
        # Contiguous callers are unchanged.
        assert token_stride(FakeTensor((1, T, H, D), (T * P, P, D, 1))) == P
        # Layouts the kernel cannot address fail loudly.
        for shape, stride in (((1, T, H, D), (T * W, W, 1, H)),        # heads strided
                              ((1, T, H, D), (T * P, P // 2, D, 1)),   # tokens overlap
                              ((2, T, H, D), (4 * T * P, P, D, 1))):   # batch not dense
            try:
                token_stride(FakeTensor(shape, stride))
            except AssertionError:
                continue
            raise AssertionError(f"accepted unaddressable layout {shape} {stride}")
    print("glm decode hot-path overlay OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
