"""Run the Alpamayo pipeline with per-stage latency report and judgement.

The main entry point of this repo. Loads Alpamayo once, runs the full
4-stage pipeline (VE -> LM Prefill -> Decode -> Flow), and prints each
stage's wall time in ms next to the expected range for this board
(configs/expected_thor.yaml) with an OK / FAST / SLOW verdict.

Usage (on Thor, inside the alpamayo venv, clocks locked):

    python scripts/run_pipeline.py --mode umic            # UMIC (default)
    python scripts/run_pipeline.py --mode eager           # unmodified model
    python scripts/run_pipeline.py --mode both --runs 3   # A/B in one process

`--mode both` runs eager first, then applies UMIC to the same loaded
model — one model load (~3-4 min) instead of two.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("umic.run")


def check_clocks() -> None:
    """Warn loudly if the GPU governor is not locked (measurement rule 1)."""
    from umic.bench import gpu_clock_state

    state = gpu_clock_state()
    if state is None:
        log.warning("could not read GPU devfreq — clock state unknown")
        return
    cur, mx = state
    if cur < mx:
        log.warning("*" * 64)
        log.warning("GPU clock %d < max %d: DVFS governor is active!", cur, mx)
        log.warning("memory-bound stages will NOT ramp clocks; numbers will")
        log.warning("be ~30%% slow. Run:  sudo jetson_clocks   and retry.")
        log.warning("*" * 64)
    else:
        log.info("GPU clock locked at %.0f MHz — OK", cur / 1e6)


def load_expected() -> dict:
    import yaml

    with open(REPO_ROOT / "configs" / "expected_thor.yaml") as f:
        return yaml.safe_load(f)


def measure(model, inputs, bench, expected: dict, label: str,
            warmup: int, runs: int) -> list[dict]:
    """Warmup + N measured runs; per-run table; returns run dicts."""
    sep = bench.PhaseSeparator()
    hooks = bench.register_hooks(model, sep)
    try:
        for i in range(warmup):
            r = bench.run_inference(model, inputs, sep)
            log.info("[%s warmup %d] wall %.0f ms (%d steps)",
                     label, i + 1, r["wall_ms"], r["decode_n_steps"])
        results = []
        for i in range(runs):
            r = bench.run_inference(model, inputs, sep)
            results.append(r)
            rows = bench.judge(r, expected.get(label, {}))
            print(bench.format_table(
                rows, f"{label} run {i + 1}/{runs} "
                      f"({r['decode_n_steps']} decode steps)"))
        return results
    finally:
        for h in hooks:
            h.remove()


def summarize(results: list[dict], label: str) -> dict:
    """Median-of-runs summary (robust to one outlier run)."""
    keys = ["VE_ms", "LM_Prefill_ms", "Decode_total_ms",
            "decode_step_ss_ms", "Flow_ms", "wall_ms"]
    med = {k: round(statistics.median(r[k] for r in results), 1)
           for k in keys}
    med["label"] = label
    med["n_runs"] = len(results)
    return med


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--mode", choices=["umic", "eager", "both"], default="umic")
    p.add_argument("--runs", type=int, default=3, help="measured runs per mode")
    p.add_argument("--warmup", type=int, default=2,
                   help="warmup runs (steady state needs >= 2 at locked clocks)")
    p.add_argument("--clip-id", default=None, help="dataset clip id override")
    p.add_argument("--adaptive-flow", action="store_true",
                   help="opt-in approximate flow (NFE6, ~4 cm deviation)")
    p.add_argument("--output", default=None,
                   help="result JSON path (default results/run_<ts>.json)")
    args = p.parse_args()

    from umic import bench

    check_clocks()
    expected = load_expected()

    model = bench.load_model()
    inputs = bench.load_inputs(model, args.clip_id or bench.DEFAULT_CLIP_ID)

    out: dict = {"mode": args.mode, "runs": args.runs, "warmup": args.warmup,
                 "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    if args.mode in ("eager", "both"):
        out["eager"] = measure(model, inputs, bench, expected, "eager",
                               args.warmup, args.runs)
        out["eager_summary"] = summarize(out["eager"], "eager")

    if args.mode in ("umic", "both"):
        import umic
        cfg = umic.UmicConfig(adaptive_flow=args.adaptive_flow)
        out["umic_report"] = umic.apply(model, cfg)
        out["umic"] = measure(model, inputs, bench, expected, "umic",
                              args.warmup, args.runs)
        out["umic_summary"] = summarize(out["umic"], "umic")

    print("\n" + "=" * 55)
    for key in ("eager_summary", "umic_summary"):
        if key in out:
            s = out[key]
            print(f"{s['label']:>6} median: VE {s['VE_ms']:.0f} | "
                  f"Prefill {s['LM_Prefill_ms']:.0f} | "
                  f"Decode {s['decode_step_ss_ms']:.1f}/step | "
                  f"Flow {s['Flow_ms']:.0f} | wall {s['wall_ms']:.0f} ms")
    if "eager_summary" in out and "umic_summary" in out:
        e, u = out["eager_summary"], out["umic_summary"]
        # decode step count varies run-to-run (sampling), so compare walls
        # normalized to a common step count — otherwise a 16-step eager vs
        # a 19-step UMIC run understates the gain.
        n_ref = expected.get("conditions", {}).get("reference_decode_steps", 16)

        def norm_wall(s: dict) -> float:
            return (s["VE_ms"] + s["LM_Prefill_ms"]
                    + n_ref * s["decode_step_ss_ms"] + s["Flow_ms"])

        ew, uw = norm_wall(e), norm_wall(u)
        print(f"UMIC vs eager wall ({n_ref}-step normalized): "
              f"{(uw / ew - 1) * 100:+.1f}%  ({ew:.0f} -> {uw:.0f} ms; "
              f"official reference: -29.8%)")
        out["normalized_comparison"] = {
            "steps": n_ref, "eager_wall_ms": round(ew, 1),
            "umic_wall_ms": round(uw, 1),
            "improvement_pct": round((uw / ew - 1) * 100, 1)}

    out_path = Path(args.output) if args.output else \
        REPO_ROOT / "results" / f"run_{time.strftime('%y%m%d_%H%M%S')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    log.info("results saved: %s", out_path)


if __name__ == "__main__":
    main()
