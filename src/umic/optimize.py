"""One-call UMIC application: `umic.apply(model)`.

Wires every *adopted* UMIC optimization into a loaded Alpamayo-family
model, in the exact configuration that produced the official benchmark
(2026-06-11, locked clocks: eager 3,846 ms -> UMIC 2,701 ms, -29.8%,
output-equivalence gate PASS at trajectory ADE 3.8 mm).

All matching is structural (duck-typed), never class-based, so the same
call works on model versions that keep the motifs (gate/up/down SiLU MLP,
pre-norm decoder layer, RMSNorm/LayerNorm, HF qwen3_vl RoPE). Anything
that does not match is silently left on the eager path — applying UMIC
to a model where nothing matches is a no-op, not an error.

Usage:

    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5
    import umic

    model = Alpamayo1_5.from_pretrained(...).cuda().eval()
    report = umic.apply(model)          # full adopted set
    report = umic.apply(model, umic.UmicConfig(decode_graph=False))
"""

from __future__ import annotations

import dataclasses
import logging

from torch import nn

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class UmicConfig:
    """Which optimizations to apply. Defaults = the official-benchmark set.

    Attributes:
        mlp_fusion: P5 gate_silu_mul fusion on the LM trunk MLPs.
        linear_fusion: q/o projections -> Triton linear (measurement-
            selected sites; k/v are at theory and stay on cuBLAS).
        norm_fusion: fused RMSNorm on LM + flow expert.
        add_rmsnorm_fusion: attn-residual add folded into post-norm
            (LM + expert decoder layers).
        text_rope_fusion: fused text RoPE (LM prefill/decode + expert).
        vision_fusion: VE fused LayerNorm + vision RoPE + bf16 residual +
            patch-embed conv->linear + packed varlen attention + the
            grid_thw-constants-cache/residual-add pipeline fusion
            (260706 VE investigation, -27.04% combined on the VE stage).
        flow_cache: InplaceKVCache for the flow append-then-crop loop.
        decode_cache: InplaceKVCache for autoregressive decode (only
            used when decode_graph is off; the graph runner owns the
            cache conversion itself).
        decode_graph: per-KV-length decode CUDA Graphs, persistent
            across inferences (10 Hz operation: capture once, replay).
        adaptive_flow: OPT-IN approximation — skip middle flow ODE steps
            (NFE6: flow -40% at ~4 cm trajectory deviation). Off by
            default because it is not bit-equivalent.
        max_new_tokens: decode-graph KV headroom (>= generate() budget).
    """

    mlp_fusion: bool = True
    linear_fusion: bool = True
    norm_fusion: bool = True
    add_rmsnorm_fusion: bool = True
    text_rope_fusion: bool = True
    vision_fusion: bool = True
    flow_cache: bool = True
    decode_cache: bool = True
    decode_graph: bool = True
    adaptive_flow: bool = False
    max_new_tokens: int = 96


def find_lm_module(model: nn.Module) -> nn.Module | None:
    """Locate the LM trunk under model.vlm (duck-typed, version-agnostic)."""
    vlm = getattr(model, "vlm", None)
    if vlm is None:
        return None
    for attr in ("language_model", "model"):
        cand = getattr(vlm, attr, None)
        if cand is None:
            continue
        if hasattr(cand, "layers"):
            return cand
        sub = getattr(cand, "model", None)
        if sub is not None and hasattr(sub, "layers"):
            return cand
    return None


def apply(model: nn.Module, config: UmicConfig | None = None) -> dict:
    """Patch all configured UMIC optimizations into `model`, in place.

    Args:
        model: A loaded Alpamayo-family model (weights untouched).
        config: Optional UmicConfig; defaults to the full adopted set.

    Returns:
        Report dict: per-optimization match counts / booleans. A count
        of 0 means the motif was not found (that fusion is a no-op).
    """
    from umic import integrate

    cfg = config or UmicConfig()
    report: dict[str, object] = {}

    lm = find_lm_module(model)
    if lm is None:
        logger.warning("umic.apply: no LM trunk found under model.vlm — "
                       "LM fusions skipped")

    if lm is not None:
        if cfg.mlp_fusion:
            report["lm_mlp_fused"] = integrate.fuse_mlps(lm)
        if cfg.linear_fusion:
            report["lm_linear_fused"] = integrate.fuse_linears(lm)
        if cfg.norm_fusion:
            report["lm_rmsnorm_fused"] = integrate.fuse_rmsnorms(lm)
        if cfg.add_rmsnorm_fusion:
            report["lm_add_rmsnorm_fused"] = integrate.fuse_add_rmsnorm(lm)

    if cfg.text_rope_fusion:
        report["text_rope_fused"] = integrate.fuse_text_rope()

    visual = getattr(getattr(model, "vlm", None), "visual", None)
    if cfg.vision_fusion and visual is not None:
        report["ve_layernorm_fused"] = integrate.fuse_layernorms(visual)
        report["ve_rope_fused"] = integrate.fuse_vision_rope()
        report["ve_bf16_residual"] = integrate.fuse_bf16_residual(visual)
        # ViT qkv->Triton and fc1+GELU stay OFF: measured losers at the
        # ViT shapes (see results/260610_m3_ve in the research repo).
        report["ve_patch_embed_fused"] = integrate.fuse_patch_embed_linear(visual)
        report["ve_attn_varlen_fused"] = integrate.fuse_vision_attention_varlen(visual)
        report["ve_pipeline_fused"] = integrate.fuse_vision_encoder_pipeline(visual)
        # ve_pipeline_fused replaces visual.forward wholesale (constants
        # cache + residual-add/LayerNorm fusion, intra- and cross-block);
        # it calls patch_embed/attn as ordinary submodule calls, so it
        # composes correctly with the two lines above whether or not
        # they matched anything (260706 VE investigation, -27.04%
        # combined on top of the layernorm/rope/bf16 fusions above).

    expert = getattr(model, "expert", None)
    if cfg.flow_cache and expert is not None:
        integrate.fuse_repeat_cache(expert)
        report["flow_cache_hook"] = True
        if cfg.norm_fusion:
            report["expert_rmsnorm_fused"] = integrate.fuse_rmsnorms(expert)
        if cfg.add_rmsnorm_fusion:
            report["expert_add_rmsnorm_fused"] = integrate.fuse_add_rmsnorm(expert)

    vlm = getattr(model, "vlm", None)
    if vlm is not None:
        if cfg.decode_graph:
            from umic.graph import fuse_decode_graph
            fuse_decode_graph(vlm, max_new_tokens=cfg.max_new_tokens)
            report["decode_graph"] = True
        elif cfg.decode_cache:
            integrate.fuse_repeat_cache(vlm, extra_tokens_hint=128)
            report["decode_cache_hook"] = True

    if cfg.adaptive_flow:
        from umic.diffusion import fuse_adaptive_flow
        fuse_adaptive_flow(model)
        report["adaptive_flow"] = True

    logger.info("umic.apply report: %s", report)
    return report
