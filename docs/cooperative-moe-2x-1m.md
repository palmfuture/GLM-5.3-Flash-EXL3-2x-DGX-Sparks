# Cooperative decode MoE on this 2× kit at 1M (2026-09-18)

Geometry 1 measured on `work/stable-e3` at `MAX_MODEL_LEN=1048576`, not the
850k/14 GiB configuration upstream published. Everything else is this branch's
serving config: adaptive-k `ema` set `2,4,7` with `GLM53_ADAPTIVE_K_BATCH=max`,
`GLM53_DENSE_FP8=dense,kda`, E3 grouped prefill including the gateup fuse from
`700b2fc`, DFlash2 k=7 draft TP2, `MAX_NUM_SEQS=4`, KV pinned at 1.05×.

## Rebuilt artifacts and why two digests were repinned

`extensions/cooperative_moe/` is byte-identical to upstream, and the build used
upstream's own pin `02aef45cd681b960a00afcd0749a4ab99e6c1bfe` with nvcc 13.0.88.
Two of the three pinned digests still had to be replaced for a local copy:

| Pin | Upstream | Here | Why |
|---|---|---|---|
| `ADAPTER_SHA` (`runtime.py`) | `9427f6a6…` | same | unmodified |
| `BINARY_SHA` (`cooperative_moe.so`) | `aa3fe5e9…` | `68a302db…` | built in this fork's image, not `miaai-lab:exl3` |
| `STOCK_SHA` (`overlay/exl3.py`) | `fe07cf3c…` | `b629333f…` | this branch's exl3.py carries the E3 gateup fuse |

`runtime.py` pins the binary a second time at load, so its own digest moves with
the repin and `ADAPTER_SHA` follows. Both repins live in staged copies under the
run directory and the vLLM cache; `extensions/` and `overlay/exl3.py` in the tree
are untouched, so a future merge does not conflict. The digest pin is a guard
against running an unreviewed binary, and it was replaced by the review it asks
for — identical sources, upstream's pin, then the packaged GPU gate as the real
numerical check.

## GPU gate, both nodes

`test_cuda_integration.py`, 48 checks, `status: pass`, exit 0 on head and worker.

- `capture_rows` covered `[1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 20, 24, 32]`,
  which includes the live adaptive-k rows 3 and 5 that upstream listed as
  outstanding.
- Rows 40 fell back to E3 grouped as designed (`peak_rel` 1.1e-07).
- `max_rel_l2` 0.00267 against a 0.05 limit.
- Two recorded peak violations, identical on both nodes to six digits:
  `eager-mut-vs-stock rows=10 scale=0.6` at 0.003606 and
  `eager-vs-stock rows=8 conc=True limit=0.5` at 0.003252, both marginally over
  the 0.003 unit-scale screen. Upstream recorded the same ~0.0036 at capture
  size 10 and scale 0.6.
- Not bit-exact, as documented: 2,610,801 post-BF16 elements differ.

## Serving

Stock is one pass on this branch immediately before selecting the overlay;
cooperative is two passes after. `tests/bench_decode.py`, 5 runs per workload,
temperature 0, thinking off.

| Workload | Stock ms/step | Coop ms/step | Δ | Stock tok/s | Coop tok/s |
|---|---:|---:|---:|---:|---:|
| Prose ×1 | 96.73 | 87.42 / 86.30 | **−9.6 %** | 31.24 | 36.07 / 35.84 |
| Structured ×1 | 110.09 | 99.36 / 99.22 | **−9.8 %** | 69.28 | 77.40 / 77.65 |
| Coding ×1 | 111.10 | 97.98 / 98.52 | **−11.6 %** | 47.18 | 50.46 / 49.92 |
| Prose ×2 aggregate | 66.05 | 57.52 / 56.40 | **−13.5 %** | 45.74 | 51.86 / 52.99 |

Read `ms/step`, not `tok/s`. Three stock passes measured prose at 95.42 / 96.49 /
96.73 ms/step, a ±0.7 % spread, so a 9.6 % move is roughly fourteen times the
run-to-run noise. `tok/s` mixes the kernel with acceptance and acceptance is not
reproducible here.

**Structured acceptance did not move**: `accepted_per_step` 6.712 and
`accept_ratio` 0.9588 in the stock pass and in both cooperative passes, to four
decimals. The whole +11.8 % on structured is the kernel. Upstream measured
accept/step falling 6–13 % on their arms; that did not reproduce here. Prose and
coding acceptance both moved, in opposite directions and inside their usual
jitter (prose 2.008 → 2.15, coding 4.258 → 3.973), so neither is a finding at
this sample size.

Prose ×1 landed at 36.07 / 35.84, which is where upstream's published sparkDash
TP2 table sits (36.1 stream).

## Other gates

- `scripts/spec-accept-gate.sh`: PASS on both cooperative passes.
- Coherence probes: `coherent: true`, `any_nan: false`.
- `tests/eval_coop_quality.py`: `pass: true`, six probes, no NaN. Upstream's gate
  reports `quality_eval_verified: false`; this closes it for this configuration.
- Host headroom: no measurable cost. Idle `MemAvailable` was 5.19 GiB head /
  7.13 GiB worker with the overlay against 4.21 / 6.87 stock, both inside the
  4.2–5.6 GiB band this kit shows across boots.

## Still outstanding

Bounded memcheck/racecheck, prolonged production burn-in, and a matched stock
pass on the same boot rather than the preceding one.

## Rollback

Comment out `EXL3_OVERLAY_HOST` in `.env` and restart. `overlay/exl3.py`, the
image and the launcher defaults are never modified by this procedure.
