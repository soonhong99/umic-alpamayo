"""Fused rotary embedding (P2 half): one kernel per tensor, fp32 in registers.

Measured motivation (results/260610_m3_ve): HF's apply_rotary_pos_emb_vision
does q.float()/k.float(), rotate_half via torch.cat((-x2, x1)), four muls,
two adds and two casts back — ~26 GB of VE traffic at fp32. The fused
kernel reads q once (bf16), gathers the rotate-half partner by index
instead of materialising a cat, does the math in fp32 registers, and
writes bf16 once. Bit-equivalent math (same fp32 internal precision).
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception as _exc:  # noqa: BLE001
    HAS_TRITON = False
    logger.warning("Triton unavailable (%s); rope falls back", _exc)


if HAS_TRITON:

    @triton.jit
    def _rope_kernel(
        x_ptr, cos_ptr, sin_ptr, out_ptr,
        H, D, HALF,
        stride_xs, stride_xh, stride_xd,
        stride_os, stride_oh, stride_od,
        stride_cs,
        BLOCK_H: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        s = tl.program_id(0)
        offs_h = tl.arange(0, BLOCK_H)
        offs_d = tl.arange(0, BLOCK_D)
        mask = (offs_h[:, None] < H) & (offs_d[None, :] < D)

        base = s * stride_xs + offs_h[:, None] * stride_xh + offs_d[None, :] * stride_xd
        x = tl.load(x_ptr + base, mask=mask, other=0.0).to(tl.float32)

        # rotate_half partner: first half pairs with +HALF (negated),
        # second half pairs with -HALF — no cat materialisation.
        partner = tl.where(offs_d < HALF, offs_d + HALF, offs_d - HALF)
        pbase = s * stride_xs + offs_h[:, None] * stride_xh + partner[None, :] * stride_xd
        x_rot = tl.load(x_ptr + pbase, mask=mask, other=0.0).to(tl.float32)
        sign = tl.where(offs_d < HALF, -1.0, 1.0)

        cmask = offs_d < D
        cos = tl.load(cos_ptr + s * stride_cs + offs_d, mask=cmask, other=0.0).to(tl.float32)
        sin = tl.load(sin_ptr + s * stride_cs + offs_d, mask=cmask, other=0.0).to(tl.float32)

        out = x * cos[None, :] + sign[None, :] * x_rot * sin[None, :]
        obase = s * stride_os + offs_h[:, None] * stride_oh + offs_d[None, :] * stride_od
        tl.store(out_ptr + obase, out.to(out_ptr.dtype.element_ty), mask=mask)

    def _apply_one(x: torch.Tensor, cos2d: torch.Tensor, sin2d: torch.Tensor,
                   s_dim: int, h_dim: int) -> torch.Tensor:
        """Rotate one tensor laid out with seq dim `s_dim`, head dim `h_dim`."""
        S, H, D = x.shape[s_dim], x.shape[h_dim], x.shape[-1]
        out = torch.empty_like(x)
        _rope_kernel[(S,)](
            x, cos2d, sin2d, out,
            H, D, D // 2,
            x.stride(s_dim), x.stride(h_dim), x.stride(-1),
            out.stride(s_dim), out.stride(h_dim), out.stride(-1),
            cos2d.stride(0),
            BLOCK_H=triton.next_power_of_2(H),
            BLOCK_D=triton.next_power_of_2(D),
        )
        return out

    def apply_rotary_vision_triton(q: torch.Tensor, k: torch.Tensor,
                                   cos: torch.Tensor, sin: torch.Tensor):
        """Drop-in for HF apply_rotary_pos_emb_vision (q/k: [S, H, D])."""
        cos2d = cos.contiguous()
        sin2d = sin.contiguous()
        return (_apply_one(q, cos2d, sin2d, 0, 1),
                _apply_one(k, cos2d, sin2d, 0, 1))

    def make_text_rope(fallback):
        """Build a drop-in for HF apply_rotary_pos_emb (q/k: [B, H, S, D]).

        Replaces the eager rotate_half/cat/mul/add chain (~8 kernels,
        ~8.8 GB in prefill) with one launch per tensor. Only the B==1,
        unsqueeze_dim==1 fast path is taken; anything else falls back to
        the original function — coverage never blocks correctness.
        """

        def apply_rotary_text_triton(q, k, cos, sin, position_ids=None,
                                     unsqueeze_dim=1):
            if (unsqueeze_dim != 1 or q.shape[0] != 1 or not q.is_cuda
                    or cos.dim() != 3):
                return fallback(q, k, cos, sin, position_ids, unsqueeze_dim)
            cos2d = cos[0].contiguous()
            sin2d = sin[0].contiguous()
            # [1, H, S, D] -> view [H, S, D]; kernel walks s_dim=1, h_dim=0
            q_out = _apply_one(q[0], cos2d, sin2d, 1, 0).unsqueeze(0)
            k_out = _apply_one(k[0], cos2d, sin2d, 1, 0).unsqueeze(0)
            return q_out, k_out

        return apply_rotary_text_triton
