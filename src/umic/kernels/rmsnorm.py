"""Fused RMSNorm: one kernel instead of eager's 5+ (pow/mean/mul/cast/copy).

Measured motivation (260610_ew_breakdown): eager RMSNorm on [3086, 4096]
bf16 explodes into fp32 pow (101 MB) + mean (51) + binary mul (101) +
bf16 cast (76) + copy (76) ≈ 400 MB per norm call vs the 50.6 MB ideal
(read x + write y once). ~50 GB of the remaining prefill traffic.

The fused kernel reads bf16 rows once, accumulates mean(x^2) in fp32
registers, scales, multiplies by weight, writes bf16 once. Numerics match
the HF reference (fp32 internal math).
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
    logger.warning("Triton unavailable (%s); rmsnorm falls back", _exc)


if HAS_TRITON:

    @triton.jit
    def _rmsnorm_kernel(
        x_ptr, w_ptr, out_ptr,
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
        var = tl.sum(x_f32 * x_f32, axis=0) / N
        rstd = 1.0 / tl.sqrt(var + eps)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = x_f32 * rstd * w
        tl.store(out_ptr + row * stride_om + cols,
                 y.to(out_ptr.dtype.element_ty), mask=mask)

    def rmsnorm_triton(x: torch.Tensor, weight: torch.Tensor,
                       eps: float) -> torch.Tensor:
        """RMSNorm over the last dim; fp32 internal math, output in x.dtype.

        Args:
            x: [..., N] activations (any leading shape, contiguous last dim).
            weight: [N] scale.
            eps: variance epsilon.
        """
        shape = x.shape
        x2d = x.reshape(-1, shape[-1]).contiguous()
        M, N = x2d.shape
        out = torch.empty_like(x2d)
        block_n = triton.next_power_of_2(N)
        num_warps = 8 if block_n >= 2048 else 4
        _rmsnorm_kernel[(M,)](
            x2d, weight, out, N,
            x2d.stride(0), out.stride(0), eps,
            BLOCK_N=block_n, num_warps=num_warps,
        )
        return out.reshape(shape)

    @triton.jit
    def _add_rmsnorm_kernel(
        x_ptr, res_ptr, w_ptr, normed_ptr, summed_ptr,
        N,
        stride_xm, stride_rm, stride_nm, stride_sm,
        eps,
        BLOCK_N: tl.constexpr,
    ):
        # Fuses `summed = x + residual` and `rmsnorm(summed)` into one pass:
        # the residual stream sum is materialized once (for the next add),
        # the normalized output once, and the sum is shared in-register with
        # the variance — eager runs these as two kernels (an elementwise add
        # round-tripping its result, then the norm re-reading it).
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_ptr + row * stride_xm + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(res_ptr + row * stride_rm + cols, mask=mask, other=0.0).to(tl.float32)
        s = x + r
        tl.store(summed_ptr + row * stride_sm + cols,
                 s.to(summed_ptr.dtype.element_ty), mask=mask)
        var = tl.sum(s * s, axis=0) / N
        rstd = 1.0 / tl.sqrt(var + eps)
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = s * rstd * w
        tl.store(normed_ptr + row * stride_nm + cols,
                 y.to(normed_ptr.dtype.element_ty), mask=mask)

    def add_rmsnorm_triton(x: torch.Tensor, residual: torch.Tensor,
                           weight: torch.Tensor, eps: float
                           ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (rmsnorm(x+residual)*w, x+residual) — fused residual norm.

        Mirrors rmsnorm_triton numerics (fp32 internal). The sum is computed
        in fp32 (slightly more accurate than eager's bf16 add) and written
        back in x.dtype as the next residual.

        Args:
            x: [..., N] sublayer output.
            residual: [..., N] residual stream (same shape as x).
            weight: [N] norm scale.
            eps: variance epsilon.

        Returns:
            (normed, summed): both [..., N] in x.dtype.
        """
        shape = x.shape
        x2d = x.reshape(-1, shape[-1]).contiguous()
        r2d = residual.reshape(-1, shape[-1]).contiguous()
        M, N = x2d.shape
        normed = torch.empty_like(x2d)
        summed = torch.empty_like(x2d)
        block_n = triton.next_power_of_2(N)
        num_warps = 8 if block_n >= 2048 else 4
        _add_rmsnorm_kernel[(M,)](
            x2d, r2d, weight, normed, summed, N,
            x2d.stride(0), r2d.stride(0), normed.stride(0), summed.stride(0),
            eps, BLOCK_N=block_n, num_warps=num_warps,
        )
        return normed.reshape(shape), summed.reshape(shape)
