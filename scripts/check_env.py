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

import os
import platform
import re
import tempfile
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
FIXES: list[tuple[str, str, bool]] = []   # (label, command, safe to run for you)
# status()/info() write here, never to `print`'s default. The kernel smoke
# redirects file descriptors 1 and 2 to swallow Triton's PTX dump, and these
# lines have to survive that.
OUT = sys.stdout


def status(ok: bool, label: str, detail: str = "", fix: str | None = None,
           auto: bool = False) -> None:
    global FAIL
    mark = "[OK]  " if ok else "[FAIL]"
    if not ok:
        FAIL += 1
        if fix:
            FIXES.append((label, fix, auto))
    print(f"  {mark} {label:<34} {detail}", file=OUT)


def _git(repo: Path, *args: str) -> str | None:
    """git stdout, or None on any failure (missing git, not a repo, timeout)."""
    try:
        proc = subprocess.run(["git", "-C", str(repo), *args],
                              capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def reference_torch() -> str | None:
    """torch version the latency ranges were measured against, from the config.

    Informational, never a [FAIL]: a newer torch is a legitimate setup (it is
    what removed the need to hand-patch PyTorch for Thor at all). It just means
    the ranges below were not measured on your stack.
    """
    try:
        text = (REPO_ROOT / "configs/expected_thor.yaml").read_text()
    except OSError:
        return None
    m = re.search(r"^reference_torch:\s*\"?([0-9][^\"\s]*)\"?\s*$", text, re.M)
    return m.group(1) if m else None


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
    print(f"  [--]   {label:<34} {detail}", file=OUT)


def section_environment() -> types.ModuleType:
    print("\n== 1. environment ==")
    print(f"  python  {sys.version.split()[0]}  ({platform.machine()})")
    try:
        import torch
    except ImportError as e:
        status(False, "torch import", str(e))
        sys.exit(1)
    status(True, "torch", torch.__version__)
    ref = reference_torch()
    if ref and not torch.__version__.startswith(ref):
        info("torch != measurement anchor",
             f"anchor {ref}; every latency range in configs/expected_thor.yaml "
             f"was measured on it")
        info("", "torch 2.11 checked on Thor 2026-07-27: UMIC still wins "
                 "(-26.3% vs -27.1%). Verdicts may shift; the gain holds.")
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
               fix="python3 -m pip install triton==3.7.1", auto=True)

    # torch 2.11+ ships its own Triton (3.6.0), and that build cannot compile for
    # this device: `ptxas-blackwell fatal: Value 'sm_110a' is not defined for
    # option 'gpu-name'`. It surfaces as a codegen error inside the kernel smoke
    # test below, which does not look like a version problem. Say so here.
    # Verified on Thor 2026-07-27: torch 2.11 + triton 3.7.1 passes everything.
    try:
        import triton as _t
        if tuple(int(x) for x in _t.__version__.split(".")[:2]) < (3, 7):
            # Where the bad Triton lives decides the fix, and getting this wrong
            # matters: torch>=2.11 wheels bundle their own Triton, and when that
            # torch is on PYTHONPATH its Triton shadows site-packages. Installing
            # a newer one with pip then changes nothing, because PYTHONPATH is
            # searched first. Say so instead of "fixing" it uselessly.
            shadowing = "site-packages" not in _t.__file__
            where = str(Path(_t.__file__).parent.parent)
            if shadowing:
                status(False, "triton too old for SM 11.0",
                       f"{_t.__version__} cannot emit sm_110a -- shadowing from {where}",
                       fix=f"# {where} comes before site-packages on your PYTHONPATH.\n"
                           f"  #   mv {where}/triton {where}/_triton-disabled\n"
                           f"  # then a Triton >= 3.7.1 in the venv is used instead.\n"
                           f"  # (torch's own inductor does not need it for this benchmark)")
            else:
                status(False, "triton too old for SM 11.0",
                       f"{_t.__version__} cannot emit sm_110a",
                       fix="python3 -m pip install 'triton>=3.7.1'", auto=True)
    except ImportError:
        pass

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
    print("\n== 3. kernel smoke (random tensors, pipeline shapes) ==", file=OUT)
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
    # A Triton codegen failure dumps its entire PTX listing -- about 1,600 lines
    # here -- and it writes at the file-descriptor level, so redirect_stderr
    # does not catch it. Point fds 1 and 2 at a file for the duration, keep the
    # real stdout for status lines, and report the file instead of the wall.
    global OUT
    log_path = Path(tempfile.gettempdir()) / "umic_kernel_output.log"
    # Flush before duplicating: anything still sitting in Python's stdout buffer
    # would otherwise be written out *after* fd 1 has been pointed at the sink,
    # i.e. sections 1 and 2 would silently land in the compiler log instead of
    # on screen.
    sys.stdout.flush()
    sys.stderr.flush()
    real_out = os.fdopen(os.dup(1), "w", buffering=1)
    saved_out, saved_err = os.dup(1), os.dup(2)
    OUT = real_out
    try:
        with open(log_path, "w") as sink:
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
            try:
                section_kernels(torch)
            finally:
                sys.stdout.flush()
                sys.stderr.flush()
                os.dup2(saved_out, 1)
                os.dup2(saved_err, 2)
        noise = log_path.read_text().splitlines()
        if len(noise) > 5:
            print(f"  [--]   compiler output                 {len(noise)} lines -> {log_path}",
                  file=real_out)
    except Exception as exc:  # noqa: BLE001
        # A Triton codegen failure raises out of the smoke test. Letting it
        # propagate kills the script before the fix summary prints -- i.e. it
        # hides exactly the advice the reader needs. Report and carry on.
        first = str(exc).strip().splitlines()
        # No fix attached: whatever went wrong here, the triton check above has
        # already attached the right one if this is the sm_110a codegen failure.
        # Guessing a second, possibly wrong command would just add noise.
        status(False, "kernel smoke", first[-1][:90] if first else type(exc).__name__)
        print(f"         full compiler output: {log_path}", file=real_out)
    finally:
        OUT = sys.stdout
        for fd in (saved_out, saved_err):
            try:
                os.close(fd)
            except OSError:
                pass
    if FAIL == 0:
        print("\nALL CHECKS PASSED")
        sys.exit(0)

    print(f"\n{FAIL} CHECK(S) FAILED")

    # Repair what is safe to repair, then start over once so the run can just
    # continue. Only pinned pip installs of packages that are already present
    # qualify -- they resolve nothing and are undone by installing the old pin.
    auto = [(label, cmd) for label, cmd, is_auto in FIXES if is_auto]
    retried = os.environ.get("UMIC_CHECK_ENV_RETRIED") == "1"
    if auto and not retried and "--no-fix" not in sys.argv:
        print("\nRepairing (pass --no-fix to only report):")
        print("-" * 64)
        ok_all = True
        for label, cmd in dict.fromkeys(auto):
            print(f"  # {label}\n  {cmd}")
            proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=900)
            if proc.returncode == 0:
                print("    OK")
            else:
                ok_all = False
                tail = (proc.stderr or proc.stdout).strip().splitlines()
                print(f"    FAILED: {tail[-1][:120] if tail else proc.returncode}")
        print("-" * 64)
        if ok_all:
            print("Re-checking with the repaired environment...\n")
            os.environ["UMIC_CHECK_ENV_RETRIED"] = "1"
            # execv replaces this process image without flushing Python's
            # buffers. When stdout is a file or a pipe (i.e. block-buffered --
            # `bash run_all.sh > log`, CI, nohup) everything printed above,
            # including the list of repairs just made, is silently discarded.
            sys.stdout.flush()
            sys.stderr.flush()
            os.execv(sys.executable, [sys.executable, *sys.argv])

    manual = [(label, cmd) for label, cmd, is_auto in FIXES if not is_auto]
    if manual:
        print("\nRun these yourself, then re-run (or `bash scripts/run_all.sh`):")
        print("-" * 64)
        for label, cmd in manual:
            print(f"  # {label}\n  {cmd}\n")
        print("-" * 64)
        print("These are not run for you: `git checkout` in someone else's")
        print("repository can discard uncommitted work, and apt needs a password.")
    sys.exit(1)


if __name__ == "__main__":
    main()
