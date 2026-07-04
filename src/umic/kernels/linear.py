"""Plain Triton linear (x @ W^T [+ bias]) for projections where stock
cuBLAS is DRAM-inefficient on Thor SM 11.0.

Measured motivation (results/260610_m1_prefill/260610_gemm_breakdown):
down_proj moves 788 MB/launch vs 183 MB theoretical (4.3x, L2 61.7%),
q/o_proj 278 MB vs 84 MB (3.3x). k/v_proj launches are already at theory
(1.0x) and must NOT be replaced — replacement is per-site, measurement-
guided, not blanket.

Same tiling skeleton as fused_ffn (the config that hit L2 94% there).
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
    logger.warning("Triton unavailable (%s); linear_triton falls back", _exc)


if HAS_TRITON:

    @triton.jit
    def _linear_kernel(
        x_ptr, w_ptr, bias_ptr, out_ptr,
        M, N, K,
        stride_xm, stride_xk,
        stride_wk, stride_wn,
        stride_om, stride_on,
        HAS_BIAS: tl.constexpr, ACT_GELU: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP: tl.constexpr,
    ):
        # L2-aware grouped CTA ordering (the single change that took the
        # down_proj shape from 11.6 ms to 3.3 ms vs cuBLAS 5.2 — see
        # results/260611_gemm_v2): consecutive programs walk GROUP rows
        # of one column band so x/w tiles stay hot in L2.
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP
        group_size_m = min(num_pid_m - first_pid_m, GROUP)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in range(0, K, BLOCK_K):
            x_tile = tl.load(
                x_ptrs,
                mask=(offs_m[:, None] < M) & ((offs_k[None, :] + k) < K), other=0.0)
            w_tile = tl.load(
                w_ptrs,
                mask=((offs_k[:, None] + k) < K) & (offs_n[None, :] < N), other=0.0)
            # 3-arg dot fuses accumulation into the MMA pipeline; the
            # `acc += tl.dot(...)` form measured 2.3x slower (8.12 vs
            # 3.56 ms at the down shape) — separate FADD chain.
            acc = tl.dot(x_tile, w_tile, acc)
            x_ptrs += BLOCK_K * stride_xk
            w_ptrs += BLOCK_K * stride_wk

        if HAS_BIAS:
            b = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
            acc += b[None, :].to(tl.float32)

        if ACT_GELU:
            # exact (erf) GELU epilogue in registers — kills the separate
            # GeluCUDAKernelImpl launch and its activation round-trip
            acc = 0.5 * acc * (1.0 + tl.math.erf(acc * 0.7071067811865476))

        out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=mask)

    # Site-measured best configs keyed by (K, N) — locked-clock sweep
    # 2026-06-11 (scripts/260611_splitk_lab.py). Every entry beat cuBLAS
    # on Thor; unknown shapes use DEFAULT_CFG (may not win — callers add
    # sites only after measuring).
    #   (bm, bn, bk, group, warps, stages)
    # NOTE on objective: configs must be picked by BYTES + time, not
    # micro-bench time alone. The 2026-06-11 time-only picks blew up
    # in-model DRAM traffic (q/o 96->179 MB, down 787->1117 MB) because
    # an empty-L2 micro-bench hides Thor's 32 MB L2 ceiling. GROUP >=
    # num_pid_m reproduces the original 2D row-major ordering whose
    # measured traffic was 1.14x theory.
    # Keys are REAL in-model shapes, captured by forensics — never assumed.
    # (The intermediate dim is 12288; an assumed 11008 key missed the
    # lookup and routed down_proj to a catastrophic default. See
    # results/260611_down_gemm_findings.)
    # Configs must pass the STABILITY bench (3x30 iters, <1% variance) —
    # one-shot sweep bests can catch transient fast states and lie
    # (a "3.56 ms" sweep best re-measured at 8.1 ms stable).
    _CFGS = {
        (12288, 4096): (128, 128, 64, 4, 8, 5),   # down: stable 4.94 vs cuBLAS 5.45
        (4096, 4096): (128, 128, 64, 32, 8, 4),   # q/o: byte-optimal legacy order
    }
    # Conservative default: legacy ordering (G huge = plain row-major),
    # never the aggressive grouping — unknown shapes must not explode.
    DEFAULT_CFG = (128, 128, 64, 32, 8, 4)

    def linear_triton(x: torch.Tensor, weight: torch.Tensor,
                      bias: torch.Tensor | None = None,
                      act: str | None = None) -> torch.Tensor:
        """Compute x @ weight.T (+bias, +gelu) with site-tuned configs.

        Args:
            x: [M, K] activations.
            weight: [N, K] — nn.Linear layout; transposed strides are
                passed to the kernel, no copy.
            bias: Optional [N].
            act: None or "gelu" (erf) epilogue.
        """
        M, K = x.shape
        N = weight.shape[0]
        bm, bn, bk, grp, nw, ns = _CFGS.get((K, N), DEFAULT_CFG)
        out = torch.empty((M, N), device=x.device, dtype=x.dtype)
        grid = (triton.cdiv(M, bm) * triton.cdiv(N, bn),)
        _linear_kernel[grid](
            x, weight, bias if bias is not None else x, out,
            M, N, K,
            x.stride(0), x.stride(1),
            weight.stride(1), weight.stride(0),  # transposed view strides
            out.stride(0), out.stride(1),
            HAS_BIAS=bias is not None, ACT_GELU=act == "gelu",
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, GROUP=grp,
            num_warps=nw, num_stages=ns,
        )
        return out
