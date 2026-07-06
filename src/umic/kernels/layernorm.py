"""Fused LayerNorm: one kernel for eager's fp32 LN + cast/copy chain.

Measured motivation (results/260610_m3_ve): the ViT runs LayerNorm with
an fp32-upcast chain (vectorized_layer_norm fp32 + bf16 casts + fp32
residual adds) — the cluster is ~33 GB of VE's 98 GB. Same pathology the
fused RMSNorm killed in the LM (M1 step 4), different norm.

Reads bf16 rows once, mean/var in fp32 registers, writes bf16 once.
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
    logger.warning("Triton unavailable (%s); layernorm falls back", _exc)


if HAS_TRITON:

    @triton.jit
    def _layernorm_kernel(
        x_ptr, w_ptr, b_ptr, out_ptr,
        N,
        stride_xm, stride_om,
        eps,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        mask = cols < N

        x = tl.load(x_ptr + row * stride_xm + cols, mask=mask, other=0.0)
        x_f32 = x.to(tl.float32)
        mean = tl.sum(x_f32, axis=0) / N
        diff = tl.where(mask, x_f32 - mean, 0.0)
        var = tl.sum(diff * diff, axis=0) / N
        rstd = 1.0 / tl.sqrt(var + eps)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = diff * rstd * w + b
        tl.store(out_ptr + row * stride_om + cols,
                 y.to(out_ptr.dtype.element_ty), mask=mask)

    def layernorm_triton(x: torch.Tensor, weight: torch.Tensor,
                         bias: torch.Tensor, eps: float) -> torch.Tensor:
        """LayerNorm over the last dim; fp32 internal math, x.dtype out."""
        shape = x.shape
        x2d = x.reshape(-1, shape[-1]).contiguous()
        M, N = x2d.shape
        out = torch.empty_like(x2d)
        block_n = triton.next_power_of_2(N)
        num_warps = 8 if block_n >= 2048 else 4
        _layernorm_kernel[(M,)](
            x2d, weight, bias, out, N,
            x2d.stride(0), out.stride(0), eps,
            BLOCK_N=block_n, num_warps=num_warps,
        )
        return out.reshape(shape)

    @triton.jit
    def _add_layernorm_kernel(
        x_ptr, res_ptr, w_ptr, b_ptr, normed_ptr, summed_ptr,
        N,
        stride_xm, stride_rm, stride_nm, stride_sm,
        eps,
        BLOCK_N: tl.constexpr,
    ):
        # Fuses `summed = x + residual` and `layernorm(summed)` into one
        # pass -- same motivation as rmsnorm.py's _add_rmsnorm_kernel, ViT
        # pre-norm edition (measured 260706: eliminates the separate
        # residual-add kernel VE's Qwen3VLVisionBlock leaves between
        # attn-out and norm2, and (applied cross-block) between one
        # block's mlp-out and the next block's norm1).
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_ptr + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(res_ptr + row * stride_rm + cols, mask=mask, other=0.0).to(tl.float32)
        s = x + r
        tl.store(summed_ptr + row * stride_sm + cols,
                 s.to(summed_ptr.dtype.element_ty), mask=mask)
        mean = tl.sum(s, axis=0) / N
        diff = tl.where(mask, s - mean, 0.0)
        var = tl.sum(diff * diff, axis=0) / N
        rstd = 1.0 / tl.sqrt(var + eps)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = diff * rstd * w + b
        tl.store(normed_ptr + row * stride_nm + cols,
                 y.to(normed_ptr.dtype.element_ty), mask=mask)

    def add_layernorm_triton(x: torch.Tensor, residual: torch.Tensor,
                             weight: torch.Tensor, bias: torch.Tensor,
                             eps: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (layernorm(x+residual), x+residual) -- fused residual norm.

        Mirrors rmsnorm.py's add_rmsnorm_triton, LayerNorm edition: the
        sum is computed in fp32 registers (slightly more accurate than
        eager's bf16 add) and written back in x.dtype as the next residual.
        """
        shape = x.shape
        x2d = x.reshape(-1, shape[-1]).contiguous()
        r2d = residual.reshape(-1, shape[-1]).contiguous()
        M, N = x2d.shape
        normed = torch.empty_like(x2d)
        summed = torch.empty_like(x2d)
        block_n = triton.next_power_of_2(N)
        num_warps = 8 if block_n >= 2048 else 4
        _add_layernorm_kernel[(M,)](
            x2d, r2d, weight, bias, normed, summed, N,
            x2d.stride(0), r2d.stride(0), normed.stride(0), summed.stride(0),
            eps, BLOCK_N=block_n, num_warps=num_warps,
        )
        return normed.reshape(shape), summed.reshape(shape)
