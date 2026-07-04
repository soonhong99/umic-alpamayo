"""Environment check + kernel smoke test — runs WITHOUT Alpamayo.

Three sections, in order:
  1. environment  : python / torch / CUDA / device / Triton / clocks
  2. alpamayo     : package importable? model in the local HF cache?
                    (informational — kernels are validated either way)
  3. kernel smoke : every UMIC Triton kernel vs its eager reference on
                    random tensors at real pipeline shapes (correctness
                    + per-kernel ms), so the engine is verified even on
                    a board that has no Alpamayo checkout yet.

Exit code 0 = environment usable and all available kernels correct.

Usage:  python scripts/check_env.py
"""

from __future__ import annotations

import platform
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

FAIL = 0


def status(ok: bool, label: str, detail: str = "") -> None:
    global FAIL
    mark = "[OK]  " if ok else "[FAIL]"
    if not ok:
        FAIL += 1
    print(f"  {mark} {label:<34} {detail}")


def info(label: str, detail: str) -> None:
    print(f"  [--]   {label:<34} {detail}")


def section_environment() -> "torch":
    print("\n== 1. environment ==")
    print(f"  python  {sys.version.split()[0]}  ({platform.machine()})")
    try:
        import torch
    except ImportError as e:
        status(False, "torch import", str(e))
        sys.exit(1)
    status(True, "torch", torch.__version__)
    status(torch.cuda.is_available(), "CUDA available")
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        status(True, "device",
               f"{torch.cuda.get_device_name(0)} (SM {cap[0]}.{cap[1]})")
        status(torch.cuda.is_bf16_supported(), "bf16 supported")
    try:
        import triton
        status(True, "triton", triton.__version__)
    except ImportError:
        status(False, "triton import",
               "fused kernels unavailable — everything falls back to eager")

    # jetson_clocks state (measurement rule 1: locked clocks mandatory)
    from umic.bench import gpu_clock_state
    state = gpu_clock_state()
    if state is None:
        info("GPU devfreq", "not readable (non-Jetson host?)")
    else:
        cur, mx = state
        locked = cur >= mx
        status(locked, "GPU clock locked",
               f"{cur / 1e6:.0f} / {mx / 1e6:.0f} MHz"
               + ("" if locked else "  -> run: sudo jetson_clocks"))
    return torch


def section_alpamayo() -> None:
    print("\n== 2. alpamayo (informational) ==")
    try:
        import alpamayo1_5  # noqa: F401
        info("alpamayo1_5 package", "importable")
    except ImportError:
        info("alpamayo1_5 package",
             "NOT importable — run_pipeline.py needs the alpamayo venv "
             "(see README section 6)")
        return
    cache = Path.home() / ".cache/huggingface/hub/models--nvidia--Alpamayo-1.5-10B"
    info("model HF cache", "present" if cache.exists()
         else "missing — first run will need HF download access")


def _bench(fn, iters: int = 20) -> float:
    import torch
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def section_kernels(torch) -> None:
    print("\n== 3. kernel smoke (random tensors, pipeline shapes) ==")
    if not torch.cuda.is_available():
        status(False, "kernel smoke", "no CUDA device")
        return
    from umic.kernels import fused_ffn, layernorm, linear, rmsnorm
    if not fused_ffn.HAS_TRITON:
        status(False, "kernel smoke", "Triton missing — skipped")
        return

    torch.manual_seed(0)
    dev, dt = "cuda", torch.bfloat16
    M, H, I = 3086, 4096, 12288      # Alpamayo LM prefill shapes

    x = torch.randn(M, H, device=dev, dtype=dt)
    wg = torch.randn(H, I, device=dev, dtype=dt) * 0.02
    wu = torch.randn(H, I, device=dev, dtype=dt) * 0.02

    def rel_err(a, b):
        return ((a.float() - b.float()).norm() / b.float().norm()).item()

    # P5 gate_silu_mul
    ref = fused_ffn.gate_silu_mul_eager(x, wg, wu)
    out = fused_ffn.gate_silu_mul_triton(x, wg, wu)
    e = rel_err(out, ref)
    ms = _bench(lambda: fused_ffn.gate_silu_mul_triton(x, wg, wu))
    status(e < 2e-2, "gate_silu_mul (P5)", f"rel_err {e:.1e}, {ms:.2f} ms")

    # linear (q_proj shape)
    w = torch.randn(H, H, device=dev, dtype=dt) * 0.02
    ref = torch.nn.functional.linear(x, w)
    out = linear.linear_triton(x, w)
    e = rel_err(out, ref)
    ms = _bench(lambda: linear.linear_triton(x, w))
    status(e < 2e-2, "linear (q/o proj)", f"rel_err {e:.1e}, {ms:.2f} ms")

    # rmsnorm
    wn = torch.randn(H, device=dev, dtype=dt)
    ref = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True)
                                   + 1e-6)).to(dt) * wn
    out = rmsnorm.rmsnorm_triton(x, wn, 1e-6)
    e = rel_err(out, ref)
    ms = _bench(lambda: rmsnorm.rmsnorm_triton(x, wn, 1e-6))
    status(e < 2e-2, "rmsnorm", f"rel_err {e:.1e}, {ms:.3f} ms")

    # add + rmsnorm (residual fusion)
    res = torch.randn_like(x)
    summed = (x.float() + res.float())
    ref_h = (summed * torch.rsqrt(summed.pow(2).mean(-1, keepdim=True)
                                  + 1e-6)).to(dt) * wn
    out_h, out_res = rmsnorm.add_rmsnorm_triton(x, res, wn, 1e-6)
    e = max(rel_err(out_h, ref_h), rel_err(out_res, summed.to(dt)))
    ms = _bench(lambda: rmsnorm.add_rmsnorm_triton(x, res, wn, 1e-6))
    status(e < 2e-2, "add_rmsnorm (residual)", f"rel_err {e:.1e}, {ms:.3f} ms")

    # layernorm (VE shape: 1152 wide)
    W = 1152
    xv = torch.randn(4096, W, device=dev, dtype=dt)
    g = torch.randn(W, device=dev, dtype=dt)
    b = torch.randn(W, device=dev, dtype=dt)
    ref = torch.nn.functional.layer_norm(xv.float(), (W,), g.float(),
                                         b.float(), 1e-6).to(dt)
    out = layernorm.layernorm_triton(xv, g, b, 1e-6)
    e = rel_err(out, ref)
    ms = _bench(lambda: layernorm.layernorm_triton(xv, g, b, 1e-6))
    status(e < 2e-2, "layernorm (VE)", f"rel_err {e:.1e}, {ms:.3f} ms")


def main() -> None:
    torch = section_environment()
    section_alpamayo()
    section_kernels(torch)
    print(f"\n{'ALL CHECKS PASSED' if FAIL == 0 else f'{FAIL} CHECK(S) FAILED'}")
    sys.exit(0 if FAIL == 0 else 1)


if __name__ == "__main__":
    main()
