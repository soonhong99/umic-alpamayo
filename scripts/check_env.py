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
import re
import subprocess
import sys
import sysconfig
import time
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

FAIL = 0
# Copy-pasteable commands that would clear each [FAIL], collected as they are
# detected and printed together at the end. The point is that one run of this
# script tells you everything you have to do, instead of one thing per attempt.
FIXES: list[tuple[str, str]] = []


def status(ok: bool, label: str, detail: str = "", fix: str | None = None) -> None:
    global FAIL
    mark = "[OK]  " if ok else "[FAIL]"
    if not ok:
        FAIL += 1
        if fix:
            FIXES.append((label, fix))
    print(f"  {mark} {label:<34} {detail}")


def _git(repo: Path, *args: str) -> str | None:
    """git stdout, or None on any failure (missing git, not a repo, timeout)."""
    try:
        proc = subprocess.run(["git", "-C", str(repo), *args],
                              capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def expected_alpamayo_commit() -> str | None:
    """Read the pinned commit out of configs/expected_thor.yaml.

    Deliberately a regex, not PyYAML: a missing PyYAML is itself one of the
    setup failures this script exists to report, so it must not need it.
    """
    try:
        text = (REPO_ROOT / "configs/expected_thor.yaml").read_text()
    except OSError:
        return None
    m = re.search(r"^alpamayo1_5_commit:\s*([0-9a-f]{40})\s*$", text, re.M)
    return m.group(1) if m else None


def info(label: str, detail: str) -> None:
    print(f"  [--]   {label:<34} {detail}")


def section_environment() -> types.ModuleType:
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
               "fused kernels unavailable — everything falls back to eager",
               fix="python3 -m pip install triton==3.7.1")

    # Triton compiles a C shim at first kernel launch, so it needs the CPython
    # development headers at RUNTIME -- not just at install time. Missing ones
    # surface as `fatal error: Python.h: No such file or directory` from inside
    # a kernel call, long after `import triton` succeeded, which is a confusing
    # place to meet them. Check up front instead.
    header = Path(sysconfig.get_paths()["include"]) / "Python.h"
    ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    status(header.exists(), "Python.h (Triton JIT needs it)",
           str(header) if header.exists() else f"missing at {header}",
           fix=f"sudo apt install python{ver}-dev      # or python3-dev")

    # jetson_clocks state (measurement rule 1: locked clocks mandatory)
    from umic.bench import gpu_clock_state
    state = gpu_clock_state()
    if state is None:
        info("GPU devfreq", "not readable (non-Jetson host?)")
    else:
        cur, mx = state
        locked = cur >= mx
        status(locked, "GPU clock locked",
               f"{cur / 1e6:.0f} / {mx / 1e6:.0f} MHz",
               fix="sudo jetson_clocks")
    return torch


def section_alpamayo() -> None:
    print("\n== 2. alpamayo ==")
    expected = expected_alpamayo_commit()
    try:
        import alpamayo1_5  # noqa: F401
    except ImportError:
        info("alpamayo1_5 package",
             "NOT importable — run_pipeline.py needs the alpamayo venv "
             "(see README section 6)")
        return
    info("alpamayo1_5 package", "importable")

    # Which revision, not just "is it there". alpamayo1_5 is a source checkout,
    # so nothing pins it; a fresh clone lands on main, which is months ahead of
    # what every number in configs/expected_thor.yaml was measured against.
    # UMIC matches the model structurally and newer revisions also default
    # attention to flash_attention_2, which cannot dispatch on SM 11.0.
    location = getattr(alpamayo1_5, "__file__", None)
    toplevel = _git(Path(location).resolve().parent, "rev-parse", "--show-toplevel") if location else None
    head = _git(Path(toplevel), "rev-parse", "HEAD") if toplevel else None
    if expected is None:
        info("alpamayo1_5 commit", "no pin found in configs/expected_thor.yaml")
    elif head is None:
        # Installed as a package, or no git. An unknown is not a failure.
        info("alpamayo1_5 commit",
             f"not determinable (not a git checkout?) — expected {expected[:12]}")
    else:
        status(head == expected, "alpamayo1_5 commit",
               f"{head[:12]}" + ("" if head == expected else f" — expected {expected[:12]}"),
               fix=f"git -C {toplevel} fetch origin && git -C {toplevel} checkout {expected}")

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
    if FAIL == 0:
        print("\nALL CHECKS PASSED")
        sys.exit(0)

    print(f"\n{FAIL} CHECK(S) FAILED")
    if FIXES:
        print("\nRun these, then re-run this script (or `bash scripts/run_all.sh`):")
        print("-" * 64)
        for label, fix in FIXES:
            print(f"  # {label}")
            print(f"  {fix}\n")
        print("-" * 64)
        print("Nothing here is run for you: checking out a different commit in")
        print("someone else's repository can discard uncommitted work, and apt")
        print("needs a password. Read them before pasting.")
    sys.exit(1)


if __name__ == "__main__":
    main()
