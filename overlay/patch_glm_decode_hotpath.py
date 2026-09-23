#!/usr/bin/env python3
"""Backport vLLM #55736: GLM-5.3-Flash decode hot-path cleanups.

Upstream (merged 2026-09-10, in v0.30.0) found three per-step wastes while
profiling GLM-5.3-Flash; all three are bit-identical and all three exist in
this image (base 487ecf187, before the PR):

  * Glm5NextMoE computed the router logits and passed them to MoERunner, which
    holds the gate (FusedMoEFactory(gate=...)) and recomputes them. The
    model-level GEMM ran on all 42 MoE layers every step for nothing; pass a
    placeholder like DeepseekV2MoE does.
  * fused_recurrent_kda made q/k/v/beta contiguous (4 copies per KDA layer per
    step, 34 layers). The recurrent kernel now takes explicit token strides so
    the column slices of the fused conv/projection outputs are read in place;
    `token_stride` asserts the layouts it can address.
  * MLA wrote the absorbed MQA query (N, B, L) then transposed it; it now
    writes token-major (B, N, L) directly, and the SM120 sparse backend skips
    the zero-width RoPE concat for NoPE models (the rope_pad is then the only
    copy). Upstream only changed the generic FlashInfer sparse backend; this
    serve runs FLASHINFER_MLA_SPARSE_SM120, so that hunk is adapted to it.

Upstream measured +1.8% decode at c=1 and -2.5% TTFT (4x GB300, TP4, FP8).
Hunks were adapted to this image's paths (the KDA ops live in
vllm/third_party/flash_linear_attention/ops/) and anchored against the files
as left by every earlier overlay, so this runs last in GLM53_OVERLAY_ORDER.

Idempotent: each hunk is applied when its old text occurs exactly once,
accepted when its new text already does, and anything else fails closed.
GLM53_VLLM_ROOT overrides the vLLM package root (tests).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

MARK = "[glm53-decode-hotpath-55736]"
ROOT = Path(os.environ.get(
    "GLM53_VLLM_ROOT", "/usr/local/lib/python3.12/dist-packages/vllm"))

# (path relative to the vllm package, label, old, new)
HUNKS = (
    (
        'model_executor/layers/attention/mla_attention.py',
        'mla-token-major-query',
        '                if self.q_pad_num_heads is not None:\n                    mqa_ql_nope = mqa_q_nope.new_empty((self.q_pad_num_heads, B, L))\n                    mqa_ql_nope.resize_((N, B, L))\n                else:\n                    mqa_ql_nope = mqa_q_nope.new_empty((N, B, L))\n\n                # Multiply (N, B, P) x (N, P, L) -> (N, B, L)\n                torch.bmm(mqa_q_nope, W_UK_T, out=mqa_ql_nope)\n\n                # Convert from (N, B, L) to (B, N, L)\n                mqa_ql_nope = mqa_ql_nope.transpose(0, 1)\n\n            if fp8_attention and self.impl.supports_quant_query_input:\n                assert mqa_ql_nope.shape[0] == mqa_q_pe.shape[0]\n',
        '                if self.q_pad_num_heads is not None:\n                    mqa_ql_nope = mqa_q_nope.new_empty((self.q_pad_num_heads, B, L))\n                    mqa_ql_nope.resize_((N, B, L))\n                    # Multiply (N, B, P) x (N, P, L) -> (N, B, L)\n                    torch.bmm(mqa_q_nope, W_UK_T, out=mqa_ql_nope)\n                    # Convert from (N, B, L) to (B, N, L)\n                    mqa_ql_nope = mqa_ql_nope.transpose(0, 1)\n                else:\n                    # Write the (N, B, L) bmm result straight into a\n                    # token-major (B, N, L) buffer so the MQA query is already\n                    # contiguous; a NoPE model (qk_rope_head_dim == 0) then\n                    # needs no concat at all.\n                    mqa_ql_nope = mqa_q_nope.new_empty((B, N, L))\n                    torch.bmm(mqa_q_nope, W_UK_T, out=mqa_ql_nope.transpose(0, 1))\n\n            if fp8_attention and self.impl.supports_quant_query_input:\n                assert mqa_ql_nope.shape[0] == mqa_q_pe.shape[0]\n',
    ),
    (
        'models/glm5next/nvidia/model.py',
        'moe-no-duplicate-router',
        "        if self.is_sequence_parallel and not already_sequence_parallel:\n            hidden_states = sequence_parallel_chunk(hidden_states)\n\n        # The router is always external (self.gate); main's MoERunner expects\n        # pre-computed router_logits, so compute them here unconditionally.\n        router_logits, _ = self.gate(hidden_states)\n        final_hidden_states = self.experts(\n            hidden_states=hidden_states, router_logits=router_logits\n        )\n\n        if self.is_sequence_parallel and not already_sequence_parallel:\n",
        '        if self.is_sequence_parallel and not already_sequence_parallel:\n            hidden_states = sequence_parallel_chunk(hidden_states)\n\n        # MoERunner holds the gate (passed to FusedMoEFactory) and computes\n        # the router logits itself, so nothing is precomputed here (matches\n        # DeepseekV2MoE; `router_logits` is a placeholder).\n        final_hidden_states = self.experts(\n            hidden_states=hidden_states, router_logits=hidden_states\n        )\n\n        if self.is_sequence_parallel and not already_sequence_parallel:\n',
    ),
    (
        'third_party/flash_linear_attention/ops/fused_recurrent.py',
        'kda-token-stride-helper',
        'from .op import exp, log\n\n\n@triton.heuristics(\n    {\n        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,\n',
        'from .op import exp, log\n\n\ndef token_stride(x: torch.Tensor) -> int:\n    """Token stride (elements) of a ``[B, T, H, D]`` or ``[B, T, H]`` tensor.\n\n    The recurrent kernel walks tokens with this stride and addresses heads\n    densely inside a token, so each token\'s ``[H, D]`` (or ``[H]``) block must\n    be contiguous, tokens must not overlap, and with ``B > 1`` sequence ``n``\n    must start at token ``n * T`` (dense batch). Column slices of a wider\n    per-token projection buffer satisfy this and are consumed in place.\n    """\n    st = x.stride()\n    assert x.dim() in (3, 4) and st[-1] == 1, (x.shape, st)\n    assert x.dim() == 3 or st[2] == x.shape[3], (x.shape, st)\n    assert st[1] >= x.shape[2] * (x.shape[3] if x.dim() == 4 else 1), (x.shape, st)\n    assert x.shape[0] == 1 or st[0] == x.shape[1] * st[1], (x.shape, st)\n    return st[1]\n\n\n@triton.heuristics(\n    {\n        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,\n',
    ),
    (
        'third_party/flash_linear_attention/ops/fused_recurrent.py',
        'kda-kernel-stride-args',
        '    stride_final_state_token: tl.constexpr,\n    stride_indices_seq: tl.constexpr,\n    stride_indices_tok: tl.constexpr,\n    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state\n    INPLACE_FINAL_STATE: tl.constexpr,  # whether to store final state inplace\n    IS_BETA_HEADWISE: tl.constexpr,  # whether beta is headwise vector or scalar,\n',
        '    stride_final_state_token: tl.constexpr,\n    stride_indices_seq: tl.constexpr,\n    stride_indices_tok: tl.constexpr,\n    # Token strides of q/k/v/beta (elements), see `token_stride`.\n    stride_q_t,\n    stride_k_t,\n    stride_v_t,\n    stride_beta_t,\n    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state\n    INPLACE_FINAL_STATE: tl.constexpr,  # whether to store final state inplace\n    IS_BETA_HEADWISE: tl.constexpr,  # whether beta is headwise vector or scalar,\n',
    ),
    (
        'third_party/flash_linear_attention/ops/fused_recurrent.py',
        'kda-kernel-stride-pointers',
        '    o_k = i_k * BK + tl.arange(0, BK)\n    o_v = i_v * BV + tl.arange(0, BV)\n\n    p_q = q + (bos * H + i_h) * K + o_k\n    p_k = k + (bos * H + i_h) * K + o_k\n    p_v = v + (bos * HV + i_hv) * V + o_v\n    if IS_BETA_HEADWISE:\n        p_beta = beta + (bos * HV + i_hv) * V + o_v\n    else:\n        p_beta = beta + bos * HV + i_hv\n\n    if not IS_KDA:\n        p_g = g + bos * HV + i_hv\n',
        '    o_k = i_k * BK + tl.arange(0, BK)\n    o_v = i_v * BV + tl.arange(0, BV)\n\n    p_q = q + bos * stride_q_t + i_h * K + o_k\n    p_k = k + bos * stride_k_t + i_h * K + o_k\n    p_v = v + bos * stride_v_t + i_hv * V + o_v\n    if IS_BETA_HEADWISE:\n        p_beta = beta + bos * stride_beta_t + i_hv * V + o_v\n    else:\n        p_beta = beta + bos * stride_beta_t + i_hv\n\n    if not IS_KDA:\n        p_g = g + bos * HV + i_hv\n',
    ),
    (
        'third_party/flash_linear_attention/ops/fused_recurrent.py',
        'kda-kernel-stride-advance',
        '            p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]\n            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)\n\n        p_q += H * K\n        p_k += H * K\n        p_o += HV * V\n        p_v += HV * V\n        if not IS_KDA:\n            p_g += HV\n        else:\n            p_gk += HV * K\n        p_beta += HV * (V if IS_BETA_HEADWISE else 1)\n\n\ndef fused_recurrent_gated_delta_rule_fwd(\n',
        '            p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]\n            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)\n\n        p_q += stride_q_t\n        p_k += stride_k_t\n        p_o += HV * V\n        p_v += stride_v_t\n        if not IS_KDA:\n            p_g += HV\n        else:\n            p_gk += HV * K\n        p_beta += stride_beta_t\n\n\ndef fused_recurrent_gated_delta_rule_fwd(\n',
    ),
    (
        'third_party/flash_linear_attention/ops/fused_recurrent.py',
        'gdn-fwd-pass-strides',
        '        stride_final_state_token=stride_final_state_token,\n        stride_indices_seq=stride_indices_seq,\n        stride_indices_tok=stride_indices_tok,\n        IS_BETA_HEADWISE=beta.ndim == v.ndim,\n        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,\n        INPLACE_FINAL_STATE=inplace_final_state,\n',
        '        stride_final_state_token=stride_final_state_token,\n        stride_indices_seq=stride_indices_seq,\n        stride_indices_tok=stride_indices_tok,\n        stride_q_t=token_stride(q),\n        stride_k_t=token_stride(k),\n        stride_v_t=token_stride(v),\n        stride_beta_t=token_stride(beta),\n        IS_BETA_HEADWISE=beta.ndim == v.ndim,\n        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,\n        INPLACE_FINAL_STATE=inplace_final_state,\n',
    ),
    (
        'third_party/flash_linear_attention/ops/kda.py',
        'kda-import-token-stride',
        '\nfrom .chunk_delta_h import chunk_gated_delta_rule_fwd_h\nfrom .cumsum import chunk_local_cumsum\nfrom .fused_recurrent import fused_recurrent_gated_delta_rule_fwd_kernel\nfrom .index import prepare_chunk_indices\nfrom .l2norm import l2norm_fwd\nfrom .op import exp2, log\n',
        '\nfrom .chunk_delta_h import chunk_gated_delta_rule_fwd_h\nfrom .cumsum import chunk_local_cumsum\nfrom .fused_recurrent import (\n    fused_recurrent_gated_delta_rule_fwd_kernel,\n    token_stride,\n)\nfrom .index import prepare_chunk_indices\nfrom .l2norm import l2norm_fwd\nfrom .op import exp2, log\n',
    ),
    (
        'third_party/flash_linear_attention/ops/kda.py',
        'kda-out-dense',
        '        g_bias = g_bias.reshape(-1).contiguous()\n\n    if out is None:\n        o = torch.empty_like(k)\n    else:\n        # Caller-provided output buffer; must be layout-compatible with the\n        # tensor the kernel indexes (contiguous, same shape/dtype as k).\n',
        '        g_bias = g_bias.reshape(-1).contiguous()\n\n    if out is None:\n        o = torch.empty(k.shape, dtype=k.dtype, device=k.device)\n    else:\n        # Caller-provided output buffer; must be layout-compatible with the\n        # tensor the kernel indexes (contiguous, same shape/dtype as k).\n',
    ),
    (
        'third_party/flash_linear_attention/ops/kda.py',
        'kda-fwd-pass-strides',
        '        stride_final_state_token=stride_final_state_token,\n        stride_indices_seq=stride_indices_seq,\n        stride_indices_tok=stride_indices_tok,\n        IS_BETA_HEADWISE=beta.ndim == v.ndim,\n        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,\n        INPLACE_FINAL_STATE=inplace_final_state,\n',
        '        stride_final_state_token=stride_final_state_token,\n        stride_indices_seq=stride_indices_seq,\n        stride_indices_tok=stride_indices_tok,\n        stride_q_t=token_stride(q),\n        stride_k_t=token_stride(k),\n        stride_v_t=token_stride(v),\n        stride_beta_t=token_stride(beta),\n        IS_BETA_HEADWISE=beta.ndim == v.ndim,\n        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,\n        INPLACE_FINAL_STATE=inplace_final_state,\n',
    ),
    (
        'third_party/flash_linear_attention/ops/kda.py',
        'kda-no-contiguous-copies',
        '    if scale is None:\n        scale = k.shape[-1] ** -0.5\n\n    o, final_state = fused_recurrent_kda_fwd(\n        q=q.contiguous(),\n        k=k.contiguous(),\n        v=v.contiguous(),\n        g=g.contiguous(),\n        beta=beta.contiguous(),\n        scale=scale,\n        initial_state=initial_state,\n        inplace_final_state=inplace_final_state,\n',
        '    if scale is None:\n        scale = k.shape[-1] ** -0.5\n\n    # q/k/v/beta are consumed in place with an explicit token stride, so\n    # column slices of the fused projection buffer need no copy; layouts the\n    # kernel cannot address fail loudly in `token_stride`.\n    o, final_state = fused_recurrent_kda_fwd(\n        q=q,\n        k=k,\n        v=v,\n        g=g.contiguous(),\n        beta=beta,\n        scale=scale,\n        initial_state=initial_state,\n        inplace_final_state=inplace_final_state,\n',
    ),
    (
        'v1/attention/backends/mla/flashinfer_mla_sparse_sm120.py',
        'sm120-skip-nope-concat',
        '        layer: AttentionLayer,\n    ) -> tuple[torch.Tensor, torch.Tensor | None]:\n        if isinstance(q, tuple):\n            q = torch.cat(q, dim=-1)\n        if self.rope_pad:\n            q = torch.nn.functional.pad(q, (0, self.rope_pad))\n\n',
        '        layer: AttentionLayer,\n    ) -> tuple[torch.Tensor, torch.Tensor | None]:\n        if isinstance(q, tuple):\n            ql_nope, q_pe = q\n            if q_pe.shape[-1] == 0 and ql_nope.is_contiguous():\n                # NoPE: skip the zero-width concat (slow CatArrayBatchedCopy);\n                # the rope_pad below is then the only copy.\n                q = ql_nope\n            else:\n                q = torch.cat(q, dim=-1)\n        if self.rope_pad:\n            q = torch.nn.functional.pad(q, (0, self.rope_pad))\n\n',
    ),
)


def apply(root: Path = ROOT) -> list[str]:
    texts: dict[str, str] = {}
    changed: list[str] = []
    for rel, label, old, new in HUNKS:
        path = root / rel
        if rel not in texts:
            if not path.is_file():
                raise SystemExit(f"{MARK} missing {path}")
            texts[rel] = path.read_text()
        text = texts[rel]
        n_old, n_new = text.count(old), text.count(new)
        if n_new == 1 and n_old == 0:
            continue
        if n_old != 1:
            raise SystemExit(
                f"{MARK} {rel}: {label} anchor drifted (old={n_old}, new={n_new})")
        texts[rel] = text.replace(old, new, 1)
        changed.append(f"{rel}:{label}")
    for rel, text in texts.items():
        compile(text, str(root / rel), "exec")
    for rel in {c.split(":", 1)[0] for c in changed}:
        (root / rel).write_text(texts[rel])
    return changed


def main() -> int:
    changed = apply()
    if changed:
        print(f"{MARK} patched {len(changed)} hunks in "
              f"{len({c.split(':', 1)[0] for c in changed})} files")
    else:
        print(f"{MARK} already present - verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
