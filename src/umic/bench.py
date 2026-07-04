"""Per-stage timing harness for the Alpamayo 4-stage pipeline.

Splits one inference into VE / LM Prefill / Decode (per-step) / Flow with
CUDA-event timers driven by forward hooks — no model-source change. Ported
from the profiling harness that produced every number in this repo
(260609_ncu_full_bandwidth.py), with the Thor-specific paths removed and
an expected-range judgement added.

Stage boundaries:
    VE      opens at the first vlm forward (seq > 1), closes when the LM
            trunk first sees seq > 1 (prefill start).
    Prefill closes at the LM trunk post-hook.
    Decode  opens at the first vlm forward with seq == 1; each step is
            timed; the stage closes right before the first flow ODE step.
    Flow    action_in_proj pre-hook -> diffusion.sample return.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import torch

logger = logging.getLogger(__name__)


class CUDATimer:
    """CUDA-event pair; ms() synchronizes lazily on first read."""

    def __init__(self) -> None:
        self.reset()

    def start(self) -> None:
        self._start.record()
        self._ms = None

    def stop(self) -> None:
        self._end.record()

    def ms(self) -> float:
        if self._ms is None:
            torch.cuda.synchronize()
            self._ms = self._start.elapsed_time(self._end)
        return self._ms

    def reset(self) -> None:
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)
        self._ms = None


class PhaseSeparator:
    """Stage state machine fed by the hooks in register_hooks()."""

    def __init__(self) -> None:
        self.t_ve = CUDATimer()
        self.t_prefill = CUDATimer()
        self.t_decode = CUDATimer()
        self.t_flow = CUDATimer()
        self._t_step = CUDATimer()
        self.t_decode_steps: list[float] = []
        self.reset()

    def reset(self) -> None:
        self.state = "idle"
        self.decode_step = 0
        self.decode_open = False
        self.ode_step = 0
        for t in (self.t_ve, self.t_prefill, self.t_decode, self.t_flow,
                  self._t_step):
            t.reset()
        self.t_decode_steps.clear()

    @staticmethod
    def _seq(args: tuple, kwargs: dict) -> int | None:
        for src in [kwargs.get("input_ids"), kwargs.get("inputs_embeds"),
                    kwargs.get("hidden_states"),
                    *(a for a in args if isinstance(a, torch.Tensor))]:
            if isinstance(src, torch.Tensor):
                if src.ndim == 2:
                    return int(src.shape[-1])
                if src.ndim == 3:
                    return int(src.shape[1])
        return None

    # -- vlm hooks -------------------------------------------------------
    def on_vlm_pre(self, module, args, kwargs) -> None:
        seq = self._seq(args, kwargs)
        if seq is None:
            return
        if seq > 1 and self.state == "idle":
            self.t_ve.start()
            self.state = "vision"
        elif seq == 1:
            if self.state == "post_prefill":
                self.t_decode.start()
                self.decode_open = True
                self.state = "decode"
                self.decode_step = 1
            elif self.state == "decode":
                self.decode_step += 1
            if self.state == "decode":
                self._t_step.start()

    def on_vlm_post(self, module, args, output) -> None:
        if self.state == "decode":
            self._t_step.stop()
            self.t_decode_steps.append(self._t_step.ms())
            self._t_step.reset()

    # -- LM trunk hooks (VE / prefill boundary) ---------------------------
    def on_lm_pre(self, module, args, kwargs) -> None:
        seq = self._seq(args, kwargs)
        if seq is not None and seq > 1 and self.state == "vision":
            self.t_ve.stop()
            self.t_prefill.start()
            self.state = "lm_prefill"

    def on_lm_post(self, module, args, output) -> None:
        if self.state == "lm_prefill":
            self.t_prefill.stop()
            self.state = "post_prefill"

    def close_decode(self) -> None:
        if self.decode_open:
            self.t_decode.stop()
            self.decode_open = False


class _SampleWrapHandle:
    """remove() for the diffusion.sample timing wrapper."""

    def __init__(self, mod, orig) -> None:
        self._mod, self._orig = mod, orig

    def remove(self) -> None:
        self._mod.sample = self._orig


def register_hooks(model, sep: PhaseSeparator) -> list:
    """Attach stage-boundary hooks; returns handles (call .remove())."""
    from umic.optimize import find_lm_module

    hooks: list = []
    vlm = getattr(model, "vlm", None)
    if vlm is None:
        raise RuntimeError("model has no .vlm — not an Alpamayo-family model")

    hooks.append(vlm.register_forward_pre_hook(sep.on_vlm_pre, with_kwargs=True))
    hooks.append(vlm.register_forward_hook(sep.on_vlm_post))

    lm = find_lm_module(model)
    if lm is not None:
        hooks.append(lm.register_forward_pre_hook(sep.on_lm_pre, with_kwargs=True))
        hooks.append(lm.register_forward_hook(sep.on_lm_post))
    else:
        logger.warning("LM trunk not found — VE/Prefill will not separate")

    def _flow_start(m, a, kw):
        if sep.ode_step == 0:
            sep.close_decode()
            sep.t_flow.start()
        sep.ode_step += 1

    if hasattr(model, "action_in_proj"):
        hooks.append(model.action_in_proj.register_forward_pre_hook(
            _flow_start, with_kwargs=True))
    elif hasattr(model, "expert"):
        hooks.append(model.expert.register_forward_pre_hook(
            _flow_start, with_kwargs=True))
    else:
        logger.warning("no flow entry module found — Flow will not be timed")

    diffusion = getattr(model, "diffusion", None)
    if diffusion is not None and hasattr(diffusion, "sample"):
        orig_sample = diffusion.sample

        def _timed_sample(*args: Any, **kwargs: Any):
            out = orig_sample(*args, **kwargs)
            if sep.ode_step > 0:
                sep.t_flow.stop()
            return out

        diffusion.sample = _timed_sample
        hooks.append(_SampleWrapHandle(diffusion, orig_sample))

    return hooks


def _safe_ms(timer: CUDATimer, name: str) -> float:
    try:
        return timer.ms()
    except Exception as exc:  # noqa: BLE001 — one dead timer must not kill the run
        logger.warning("%s timer failed: %s", name, exc)
        return 0.0


def collect_timing(sep: PhaseSeparator) -> dict:
    """Read all timers into a flat result dict (ms)."""
    ve = _safe_ms(sep.t_ve, "VE")
    prefill = _safe_ms(sep.t_prefill, "Prefill")
    flow = _safe_ms(sep.t_flow, "Flow")
    steps = sep.t_decode_steps.copy()
    n = len(steps)
    try:
        decode_total = sep.t_decode.ms()
    except Exception:  # noqa: BLE001
        decode_total = sum(steps)
    ss = steps[3:] if n > 3 else steps
    return {
        "VE_ms": round(ve, 1),
        "LM_Prefill_ms": round(prefill, 1),
        "Decode_total_ms": round(decode_total, 1),
        "decode_n_steps": n,
        "decode_step_mean_ms": round(sum(steps) / n, 2) if n else 0.0,
        "decode_step_ss_ms": round(sum(ss) / len(ss), 2) if ss else 0.0,
        "Flow_ms": round(flow, 1),
        "wall_ms": round(ve + prefill + decode_total + flow, 1),
    }


def run_inference(model, model_inputs: dict, sep: PhaseSeparator) -> dict:
    """One timed inference (same call signature as deployment)."""
    sep.reset()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs, top_p=0.98, temperature=0.6,
            num_traj_samples=1, return_extra=True)
    torch.cuda.synchronize()
    return collect_timing(sep)


# --------------------------------------------------------------------------
# Alpamayo loading helpers (the only Alpamayo-specific code in this module)
# --------------------------------------------------------------------------

DEFAULT_MODEL_ID = "nvidia/Alpamayo-1.5-10B"
DEFAULT_CLIP_ID = "030c760c-ae38-49aa-9ad8-f5650a545d26"


def load_model(model_id: str = DEFAULT_MODEL_ID):
    """Load Alpamayo from the local HF cache (~3-4 min for 22 GB).

    NOTE: Alpamayo1_5.from_pretrained only accepts an HF repo id (a local
    absolute path raises HFValidationError); local_files_only keeps it
    offline.
    """
    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

    logger.info("loading %s (first load ~3-4 min)...", model_id)
    t0 = time.time()
    model = Alpamayo1_5.from_pretrained(
        model_id, dtype=torch.bfloat16, local_files_only=True,
    ).cuda().eval()
    gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
    logger.info("model loaded: %.1f s, %.2f GB weights", time.time() - t0, gb)
    return model


def load_inputs(model, clip_id: str = DEFAULT_CLIP_ID, t0_us: int = 5_100_000) -> dict:
    """Load one dataset clip and convert it to model inputs (on cuda)."""
    from alpamayo1_5 import helper
    from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset

    data = load_physical_aiavdataset(clip_id, t0_us=t0_us)
    messages = helper.create_message(
        frames=data["image_frames"].flatten(0, 1),
        camera_indices=data["camera_indices"])
    processor = helper.get_processor(model.tokenizer)
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        continue_final_message=True, return_dict=True, return_tensors="pt")
    model_inputs = helper.to_device(
        {"tokenized_data": inputs,
         "ego_history_xyz": data["ego_history_xyz"],
         "ego_history_rot": data["ego_history_rot"]}, "cuda")
    logger.info("inputs ready: %d prompt tokens", inputs["input_ids"].shape[-1])
    return model_inputs


# --------------------------------------------------------------------------
# Expected-range judgement
# --------------------------------------------------------------------------

STAGE_KEYS = [
    ("VE", "VE_ms"),
    ("LM Prefill", "LM_Prefill_ms"),
    ("Decode/step (SS)", "decode_step_ss_ms"),
    ("Flow", "Flow_ms"),
    ("Wall total", "wall_ms"),
]


def judge(result: dict, expected: dict) -> list[tuple[str, float, str, str]]:
    """Compare one run against expected [lo, hi] ranges.

    Args:
        result: collect_timing() output.
        expected: mapping result-key -> [lo, hi] in ms.

    Returns:
        Rows (stage label, measured ms, expected string, verdict) where
        verdict is OK / FAST / SLOW / n-a.
    """
    rows = []
    for label, key in STAGE_KEYS:
        val = result.get(key, 0.0)
        rng = expected.get(key)
        if not rng:
            rows.append((label, val, "-", "n/a"))
            continue
        lo, hi = rng
        verdict = "OK" if lo <= val <= hi else ("FAST" if val < lo else "SLOW")
        rows.append((label, val, f"{lo:.0f}-{hi:.0f}", verdict))
    return rows


def format_table(rows: list[tuple[str, float, str, str]], title: str) -> str:
    """Render judge() rows as an aligned console table."""
    out = [f"\n=== {title} ===",
           f"{'stage':<18} {'measured':>10}   {'expected':>11}   verdict",
           "-" * 55]
    for label, val, rng, verdict in rows:
        mark = {"OK": "[OK]  ", "FAST": "[FAST]", "SLOW": "[SLOW]",
                "n/a": "      "}[verdict]
        out.append(f"{label:<18} {val:>8.1f} ms   {rng:>11}   {mark}")
    return "\n".join(out)
