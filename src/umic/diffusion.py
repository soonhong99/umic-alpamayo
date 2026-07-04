"""Adaptive-step flow matching: skip the redundant middle ODE steps.

Measured on Thor (results/260614_flow_adaptive): the flow-matching velocity
field is U-shaped — consecutive-step cosine similarity is lower at the first/
last steps (~0.994) and higher in the middle (~0.998), i.e. the velocity
changes most at the endpoints and is nearly constant through the middle.
(FlashDrive, Li et al. 2026, observed the same on discrete GPUs; we verified
it transfers to the iGPU before exploiting it.)

We therefore evaluate the network fresh only at the first `n_front` and last
`n_back` ODE steps and REUSE the last fresh velocity for the middle steps,
cutting NFE (network forward evals) with near-lossless trajectories. Thor A/B
(flow ODE time, deviation from full 10-step):
  NFE6 {0,1,2,7,8,9}: 412->248 ms (-40%), 4.2 cm   <- adopted default
  NFE5 {0,1,2,8,9}  : 412->207 ms (-50%), 8.4 cm
  NFE4 {0,1,8,9}    : 412->166 ms (-60%), 12.4 cm

Structural injection only (wrap diffusion.sample); model/weights untouched.
This is an *approximation* (unlike UMIC's bit-identical fusions) — it trades a
few cm of trajectory deviation for speed, so it is opt-in and gated on ADE.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


class AdaptiveFlowCache:
    """Wraps a flow step_fn: compute fresh at `fresh` step indices, else reuse.

    The Euler loop calls this once per ODE step. At a cached step we skip the
    expert forward entirely (the speedup) and return the last fresh velocity.
    A fresh closure is created per sample() call so the step counter resets.
    """

    def __init__(self, step_fn, fresh: set[int]):
        self.step_fn = step_fn
        self.fresh = fresh
        self.i = 0
        self.cached: torch.Tensor | None = None
        self.nfe = 0

    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.i in self.fresh or self.cached is None:
            self.cached = self.step_fn(x=x, t=t)
            self.nfe += 1
        self.i += 1
        return self.cached


def fuse_adaptive_flow(model: torch.nn.Module,
                       n_front: int = 3, n_back: int = 3) -> None:
    """Skip middle ODE steps in flow matching (keep first n_front / last n_back).

    Args:
        model: the Alpamayo model (uses model.diffusion).
        n_front: number of fresh network evals at the start of the ODE.
        n_back: number of fresh network evals at the end of the ODE.

    The fresh-step set is computed from diffusion.num_inference_steps at install
    time. Default 3/3 -> NFE6 on the standard 10-step schedule (adopted, 4 cm).
    """
    diffusion = model.diffusion
    total = getattr(diffusion, "num_inference_steps", 10)
    fresh = set(range(n_front)) | set(range(max(0, total - n_back), total))
    orig_sample = diffusion.sample

    def patched(*a, **kw):
        if kw.get("step_fn") is not None:
            kw["step_fn"] = AdaptiveFlowCache(kw["step_fn"], fresh)
        if kw.get("unguided_step_fn") is not None:
            kw["unguided_step_fn"] = AdaptiveFlowCache(kw["unguided_step_fn"], fresh)
        return orig_sample(*a, **kw)

    diffusion.sample = patched
    logger.info("fuse_adaptive_flow: NFE=%d fresh steps %s (of %d)",
                len(fresh), sorted(fresh), total)
