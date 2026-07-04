"""Decode-step CUDA Graphs: eliminate the measured dispatch-idle bubble.

ncu/nsys measurement (260611 report): during autoregressive decode the
GPU is fully idle ~10.6% (~160 ms per inference) waiting for the CPU to
launch kernels. This is the llm.npu bubble as it actually manifests on
Thor (stage transitions measured 0.6%; unified memory has no transfer
to hide — see results/260610_m3_schedule).

Design: ONE GRAPH PER KV LENGTH (vLLM-style), PERSISTENT ACROSS
INFERENCES. The padded-mask single-graph variant measured +23 ms/step
(4D mask disables enable_gqa -> repeat_kv 26 MB x2 x36/step, and
demotes flash to mem-efficient). Exact-length graphs keep
attention_mask=None -> flash + enable_gqa, identical kernels to eager.

Persistence contract (10 Hz continuous inference): KV buffers, static
input tensors and the rope-delta tensor live in the runner; each new
inference copies its prefill KV / rope_deltas INTO them, so previously
captured graphs (which bake buffer addresses) stay valid. First
inference captures ~19 graphs; every later inference is pure replay.
Sampling stays eager outside the graph.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from transformers.cache_utils import DynamicCache

from umic.cache import InplaceKVCache

logger = logging.getLogger(__name__)


class DecodeGraphRunner:
    """Wraps a VLM's forward; replays per-KV-length graphs for decode."""

    def __init__(self, vlm: torch.nn.Module, max_new_tokens: int = 96,
                 warmup_steps: int = 2) -> None:
        self.vlm = vlm
        self.orig_forward = vlm.forward
        self.max_new = max_new_tokens
        self.warmup_steps = warmup_steps
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.outs: dict[int, Any] = {}
        self.pool = None
        self.cache: InplaceKVCache | None = None   # persistent buffers
        self.base_len: int | None = None
        self.in_decode = False
        self.step = 0
        self.s_ids = self.s_pos = self.s_rope = None
        self.const_kwargs: dict[str, Any] = {}

        vlm.forward = self.__call__  # instance attr; module hooks still fire
        orig_generate = vlm.generate

        def _generate(*a: Any, **kw: Any):
            out = orig_generate(*a, **kw)
            self.finish()
            return out

        vlm.generate = _generate
        logger.info("DecodeGraphRunner installed on %s", type(vlm).__name__)

    # ------------------------------------------------------------------

    def finish(self) -> None:
        """generate() returned: hand the cache to flow in dynamic mode."""
        if self.cache is not None:
            self.cache.disable_graph_mode()
        self.in_decode = False
        self.step = 0

    def _begin_inference(self, kwargs: dict[str, Any]) -> bool:
        pkv = kwargs.get("past_key_values")
        if not (isinstance(pkv, DynamicCache) and getattr(pkv, "layers", None)
                and all(getattr(l, "keys", None) is not None for l in pkv.layers)):
            return False
        mask = kwargs["attention_mask"]
        if not bool((mask == 1).all().item()):
            logger.warning("DecodeGraphRunner: padded input mask — bypassing")
            return False

        base = pkv.get_seq_length()
        if self.cache is not None and base == self.base_len \
                and pkv is not self.cache:
            # reuse persistent buffers: copy the new prefill KV in —
            # captured graphs keep their baked addresses valid
            for dst, src in zip(self.cache.layers, pkv.layers):
                dst._k_buf[:, :, :base, :].copy_(src.keys)
                dst._v_buf[:, :, :base, :].copy_(src.values)
            self.cache.set_pos(base)
        elif pkv is not self.cache:
            if self.cache is not None:
                logger.info("DecodeGraphRunner: base_len changed %s -> %d, "
                            "rebuilding graphs", self.base_len, base)
                self.graphs.clear()
                self.outs.clear()
                self.pool = None
            self.cache = InplaceKVCache.from_dynamic(pkv, extra_tokens=self.max_new)
            self.base_len = base

        dev = kwargs["input_ids"].device
        rope_deltas = self.vlm.model.rope_deltas
        if self.s_ids is None:
            self.s_ids = torch.zeros(1, 1, dtype=torch.long, device=dev)
            self.s_pos = torch.zeros(1, dtype=torch.long, device=dev)
            self.s_rope = torch.zeros_like(rope_deltas.reshape(-1))
        self.s_rope.copy_(rope_deltas.reshape(-1))
        self.cache.enable_graph_mode(self.s_pos)
        self.const_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in ("input_ids", "attention_mask", "cache_position",
                         "past_key_values", "position_ids")}
        self.in_decode = True
        return True

    def _static_call(self):
        # position_ids supplied explicitly: the model's own derivation
        # branches on `cache_position[0] == 0` (CPU sync, capture-illegal).
        delta = (self.s_pos + self.s_rope).reshape(1, 1)
        position_ids = delta.unsqueeze(0).expand(3, 1, 1)
        return self.orig_forward(
            input_ids=self.s_ids, attention_mask=None,
            cache_position=self.s_pos, past_key_values=self.cache,
            position_ids=position_ids,
            **self.const_kwargs)

    # ------------------------------------------------------------------

    def __call__(self, *args: Any, **kwargs: Any):
        ids = kwargs.get("input_ids")
        if (args or ids is None or ids.shape != (1, 1)
                or kwargs.get("cache_position") is None
                or kwargs.get("attention_mask") is None):
            return self.orig_forward(*args, **kwargs)

        if not self.in_decode and not self._begin_inference(kwargs):
            return self.orig_forward(*args, **kwargs)

        self.s_ids.copy_(kwargs["input_ids"])
        self.s_pos.copy_(kwargs["cache_position"])
        pos = int(kwargs["cache_position"][0].item())
        kv_len = pos + 1
        self.cache.set_graph_len(kv_len)

        if not self.graphs and self.step < self.warmup_steps:
            # global lazy-init warmup, only ever before the first capture
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                out = self._static_call()
            torch.cuda.current_stream().wait_stream(s)
        elif kv_len not in self.graphs:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=self.pool,
                                  capture_error_mode="relaxed"):
                self.outs[kv_len] = self._static_call()
            if self.pool is None:
                self.pool = g.pool()
            g.replay()
            self.graphs[kv_len] = g
            out = self.outs[kv_len]
        else:
            self.graphs[kv_len].replay()
            out = self.outs[kv_len]

        self.step += 1
        self.cache.set_pos(kv_len)
        return out


def fuse_decode_graph(vlm: torch.nn.Module, max_new_tokens: int = 96) -> DecodeGraphRunner:
    """Install the per-length decode CUDA Graph runner on the VLM."""
    return DecodeGraphRunner(vlm, max_new_tokens=max_new_tokens)


class FlowStepGraph:
    """Capture one flow ODE step (fixed-shape expert forward), replay it.

    Why flow is simpler than decode: every ODE step runs step_fn on the
    SAME shapes — 32 action tokens attending to a fixed-length prefix KV
    that the step appends to [base:base+32] and crops straight back, so
    the buffer addresses the kernels touch are identical each step (the
    prefix [0:base] is never overwritten; only [base:base+32] is rewritten
    from the new x). Only x and t change *value*. One graph therefore
    replays for all steps — no per-length graphs, no index_copy_/pos_t.

    Re-captured per inference (each inference builds a fresh prompt cache
    with new buffer addresses); a 10 Hz-persistent variant would copy new
    prefill KV into persistent buffers like DecodeGraphRunner. Capture
    failure falls back to eager so correctness never breaks.
    """

    def __init__(self, step_fn, warmup: int = 2) -> None:
        self.step_fn = step_fn
        self.warmup = warmup
        self.n = 0
        self.graph: torch.cuda.CUDAGraph | None = None
        self.xs: torch.Tensor | None = None
        self.ts: torch.Tensor | None = None
        self.out: torch.Tensor | None = None
        self.failed = False

    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.failed or (self.graph is None and self.n < self.warmup):
            self.n += 1
            return self.step_fn(x=x, t=t)
        if self.graph is None:
            try:
                self.xs = x.clone()
                self.ts = t.clone()
                # warm the capture stream (CUDA Graph best practice)
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    self.step_fn(x=self.xs, t=self.ts)
                torch.cuda.current_stream().wait_stream(s)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, capture_error_mode="relaxed"):
                    self.out = self.step_fn(x=self.xs, t=self.ts)
                self.graph = g
                logger.info("FlowStepGraph captured (step %d)", self.n)
            except Exception as exc:  # noqa: BLE001 — never break correctness
                logger.warning("FlowStepGraph capture failed (%s); eager fallback", exc)
                self.failed = True
                return self.step_fn(x=x, t=t)
        self.xs.copy_(x)
        self.ts.copy_(t)
        self.graph.replay()
        return self.out.clone()


def fuse_flow_graph(model: torch.nn.Module, warmup: int = 2) -> None:
    """Wrap diffusion.sample so each ODE step_fn replays from a CUDA Graph.

    Targets the dispatch-idle bubble of flow's many small expert kernels.
    Structural injection (no model source change): the step closures the
    model passes to `sample` are wrapped; everything else is unchanged.
    """
    diffusion = model.diffusion
    orig_sample = diffusion.sample

    def patched_sample(*args, **kwargs):
        if "step_fn" in kwargs and kwargs["step_fn"] is not None:
            kwargs["step_fn"] = FlowStepGraph(kwargs["step_fn"], warmup)
        if kwargs.get("unguided_step_fn") is not None:
            kwargs["unguided_step_fn"] = FlowStepGraph(kwargs["unguided_step_fn"], warmup)
        return orig_sample(*args, **kwargs)

    diffusion.sample = patched_sample
    logger.info("fuse_flow_graph: flow ODE step graph runner installed")
