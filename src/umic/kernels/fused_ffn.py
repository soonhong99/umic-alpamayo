"""P5 gate_silu_mul: silu(x @ Wg) * (x @ Wu) in one Triton kernel.

The eager path launches 4 kernels and round-trips two [seq, ffn_dim]
intermediates through DRAM (~272 MB per prefill layer). Here both GEMMs
share one K-loop — x tiles are loaded once and the SiLU*mul epilogue runs
in registers, so only the final product is written.

Written as a direct @triton.jit kernel on purpose: the torch.compile ->
Inductor -> Triton path is broken on Thor (2026-05-28), but that failure
is in Inductor's codegen API usage, not necessarily the Triton runtime.
M0 verifies the runtime itself on SM 11.0.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except Exception as _exc:  # noqa: BLE001 — record and fall back
    HAS_TRITON = False
    logger.warning("Triton unavailable (%s); gate_silu_mul uses eager fallback", _exc)


def gate_silu_mul_eager(x: torch.Tensor, w_gate: torch.Tensor,
                        w_up: torch.Tensor) -> torch.Tensor:
    """Reference implementation (also the always-available fallback).

    Args:
        x: Activations [M, K].
        w_gate: Gate projection weight [K, N].
        w_up: Up projection weight [K, N].

    Returns:
        silu(x @ w_gate) * (x @ w_up), shape [M, N].
    """
    return torch.nn.functional.silu(x @ w_gate) * (x @ w_up)


if HAS_TRITON:

    @triton.jit
    def _gate_silu_mul_kernel(
        x_ptr, wg_ptr, wu_ptr, out_ptr,
        M, N, K,
        stride_xm, stride_xk,
        stride_wk, stride_wn,
        stride_om, stride_on,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        wg_ptrs = wg_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
        wu_ptrs = wu_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

        acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in range(0, K, BLOCK_K):
            k_mask = (offs_k[None, :] + k) < K
            x_tile = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & k_mask, other=0.0)
            km_mask = (offs_k[:, None] + k) < K
            n_mask = offs_n[None, :] < N
            wg_tile = tl.load(wg_ptrs, mask=km_mask & n_mask, other=0.0)
            wu_tile = tl.load(wu_ptrs, mask=km_mask & n_mask, other=0.0)
            # One x tile feeds both GEMMs — this is the fusion.
            # (3-arg dot: accumulation fused into MMA; += form is ~2x slower)
            acc_g = tl.dot(x_tile, wg_tile, acc_g)
            acc_u = tl.dot(x_tile, wu_tile, acc_u)
            x_ptrs += BLOCK_K * stride_xk
            wg_ptrs += BLOCK_K * stride_wk
            wu_ptrs += BLOCK_K * stride_wk

        # Epilogue in registers: silu(gate) * up. Intermediates never hit DRAM.
        out = acc_g * tl.sigmoid(acc_g) * acc_u

        out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(out_ptrs, out.to(out_ptr.dtype.element_ty), mask=mask)

    def gate_silu_mul_triton(x: torch.Tensor, w_gate: torch.Tensor,
                             w_up: torch.Tensor,
                             block_m: int = 128, block_n: int = 128,
                             block_k: int = 64, num_warps: int = 8,
                             num_stages: int = 4) -> torch.Tensor:
        """Fused launch.

        Defaults are the best config from the 2026-06-10 on-Thor sweep at
        the prefill shape [3086,4096]x[4096,11008]: 6.00 ms (eager chain
        5.80 ms, but with +272 MB DRAM round-trip per call).
        """
        M, K = x.shape
        K2, N = w_gate.shape
        assert K == K2 and w_up.shape == (K, N)
        out = torch.empty((M, N), device=x.device, dtype=x.dtype)
        grid = (triton.cdiv(M, block_m), triton.cdiv(N, block_n))
        _gate_silu_mul_kernel[grid](
            x, w_gate, w_up, out,
            M, N, K,
            x.stride(0), x.stride(1),
            w_gate.stride(0), w_gate.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
            num_warps=num_warps, num_stages=num_stages,
        )
        return out


def gate_silu_mul(x: torch.Tensor, w_gate: torch.Tensor,
                  w_up: torch.Tensor) -> torch.Tensor:
    """Best available implementation for the P5 motif."""
    if HAS_TRITON and x.is_cuda:
        return gate_silu_mul_triton(x, w_gate, w_up)
    return gate_silu_mul_eager(x, w_gate, w_up)
