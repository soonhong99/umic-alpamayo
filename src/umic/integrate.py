"""Module-level fusion injection — zero model modification.

Instead of rewriting checkpoints or model source, UMIC swaps the *forward*
of matching submodules at load time. The match is structural (duck-typed),
not class-based, so it survives model version changes: any module exposing
`gate_proj` / `up_proj` / `down_proj` Linears with a SiLU-family activation
is a P5 candidate — Qwen2, Qwen3, Llama, and whatever Alpamayo 2.0 ships,
as long as the motif is present.

Weights are shared (no copy, no quantization, no value change); only the
execution schedule of the same math changes.
"""

from __future__ import annotations

import logging

import torch
from torch import nn

from umic.kernels.fused_ffn import gate_silu_mul

logger = logging.getLogger(__name__)

_SILU_NAMES = ("silu", "swish")

# Below this row count the motif runs as GEMV (decode, seq=1) where cuBLAS
# is optimal and the eager intermediates are KB-scale — fusion would only
# hurt. Regime-aware dispatch, design doc 원칙 2.
FUSE_MIN_ROWS = 64


def _is_p5_mlp(module: nn.Module) -> bool:
    """Structural match for the gate/up/down SiLU MLP motif (pattern P5)."""
    for attr in ("gate_proj", "up_proj", "down_proj"):
        sub = getattr(module, attr, None)
        if not isinstance(sub, nn.Linear) or sub.bias is not None:
            return False
    act = getattr(module, "act_fn", None) or getattr(module, "act", None)
    act_name = type(act).__name__.lower() if act is not None else ""
    return any(s in act_name for s in _SILU_NAMES)


def _fused_mlp_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """P5-fused replacement: silu(x@Wg)*(x@Wu) in one kernel, then down."""
    shape = x.shape
    x2d = x.reshape(-1, shape[-1])
    if x2d.shape[0] < FUSE_MIN_ROWS:
        h = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(h)
    # nn.Linear stores weight as [out, in]; the kernel takes [in, out]
    # strides, so .t() is a free view — no copy, weights untouched.
    h = gate_silu_mul(x2d, self.gate_proj.weight.t(), self.up_proj.weight.t())
    out = self.down_proj(h)
    return out.reshape(*shape[:-1], out.shape[-1])


def fuse_mlps(model: nn.Module, dry_run: bool = False) -> int:
    """Swap the forward of every P5-matching MLP in `model`.

    Args:
        model: Any nn.Module tree (unmodified checkpoint).
        dry_run: If True, only count matches without patching.

    Returns:
        Number of modules matched (and patched unless dry_run).
    """
    count = 0
    for name, module in model.named_modules():
        if _is_p5_mlp(module):
            count += 1
            if not dry_run:
                module.forward = _fused_mlp_forward.__get__(module)
                logger.info("P5 fused: %s", name)
    logger.info("fuse_mlps: %d module(s) %s", count,
                "matched (dry run)" if dry_run else "patched")
    return count


# Projection sites measured DRAM-inefficient under cuBLAS on Thor SM 11.0
# (results/260610_m1_prefill). k_proj/v_proj measured AT theory — excluded.
# down_proj re-included 2026-06-11: the GROUP-swizzle kernel beats cuBLAS
# there (3.32 vs 5.24 ms at locked clocks) — no split-K needed; the
# earlier loss was missing L2-aware CTA ordering, not K size.
# down_proj FINAL VERDICT 2026-06-11 (full saga in
# results/260611_down_gemm_findings.md): the real shape is K=12288 (not
# 11008 — wrong key caused the 17 ms catastrophe), the stable-bench
# winner (G4, 4.94 vs cuBLAS 5.45 ms isolated) still loses ~+30 ms in
# e2e prefill (in-model penalty ~+1 ms/launch, unresolved). cuBLAS keeps
# the site until the in-model penalty is understood.
INEFFICIENT_LINEAR_NAMES = ("q_proj", "o_proj")

# ViT sites that beat cuBLAS with the same kernel (qkv 1.23 vs 2.22 ms);
# fc1 stays on cuBLAS (nvjet wins that shape, 1.16 vs 1.71 ms).
VIT_LINEAR_NAMES = ("qkv",)


def _patched_linear_forward(self: nn.Linear, x: torch.Tensor) -> torch.Tensor:
    """Route large-M calls to the Triton linear; GEMV stays on cuBLAS."""
    from umic.kernels.linear import linear_triton

    shape = x.shape
    x2d = x.reshape(-1, shape[-1])
    if x2d.shape[0] < FUSE_MIN_ROWS:
        return nn.functional.linear(x, self.weight, self.bias)
    out = linear_triton(x2d, self.weight, self.bias)
    return out.reshape(*shape[:-1], out.shape[-1])


def fuse_linears(model: nn.Module,
                 names: tuple[str, ...] = INEFFICIENT_LINEAR_NAMES,
                 dry_run: bool = False) -> int:
    """Swap forward of nn.Linear submodules whose attribute name matches.

    Site selection is measurement-guided: only projections whose ncu
    per-launch traffic exceeds theory get replaced (k/v_proj are already
    at theory and stay on cuBLAS). Weights untouched.

    Returns:
        Number of Linear modules matched (and patched unless dry_run).
    """
    count = 0
    for name, module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf in names and isinstance(module, nn.Linear):
            count += 1
            if not dry_run:
                module.forward = _patched_linear_forward.__get__(module)
    logger.info("fuse_linears(%s): %d module(s) %s", names, count,
                "matched (dry run)" if dry_run else "patched")
    return count


def _patched_rmsnorm_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Single fused kernel replacing the eager pow/mean/mul/cast chain."""
    from umic.kernels.rmsnorm import rmsnorm_triton

    eps = getattr(self, "variance_epsilon", None)
    if eps is None:
        eps = getattr(self, "eps", 1e-6)
    return rmsnorm_triton(x, self.weight, eps)


def fuse_rmsnorms(model: nn.Module, dry_run: bool = False) -> int:
    """Swap forward of every RMSNorm-family module (duck-typed).

    Match: class name contains "rmsnorm" + has a 1-D `weight`. Covers HF
    Qwen2/Qwen3/Llama RMSNorm variants regardless of module path. Small
    rows are fine — the kernel is one launch either way, strictly fewer
    than eager's 5+, so no min-rows dispatch is needed.

    Returns:
        Number of modules matched (and patched unless dry_run).
    """
    count = 0
    for name, module in model.named_modules():
        if "rmsnorm" not in type(module).__name__.lower():
            continue
        w = getattr(module, "weight", None)
        if not isinstance(w, torch.Tensor) or w.dim() != 1:
            continue
        count += 1
        if not dry_run:
            module.forward = _patched_rmsnorm_forward.__get__(module)
    logger.info("fuse_rmsnorms: %d module(s) %s", count,
                "matched (dry run)" if dry_run else "patched")
    return count


def _is_residual_decoder_layer(module: nn.Module) -> bool:
    """Structural match for the pre-norm attn+MLP decoder block (Qwen/Llama).

    Needs the two norms, an attention and an MLP submodule — the motif whose
    forward does `residual + attn` then `post_norm`, and `residual + mlp`.
    """
    return all(hasattr(module, a) for a in
               ("input_layernorm", "post_attention_layernorm", "self_attn", "mlp"))


def _fused_residual_layer_forward(self, hidden_states, position_embeddings,
                                  attention_mask=None, position_ids=None,
                                  past_key_values=None, use_cache=False,
                                  cache_position=None, **kwargs):
    """Decoder layer forward with the attn-residual add fused into post-norm.

    `residual + attn_out` (a standalone elementwise kernel round-tripping a
    [M,H] tensor) is folded into the following RMSNorm: one kernel produces
    both the normalized MLP input and the summed residual stream. The MLP
    residual add stays eager (its consumer is the *next* layer's input norm
    — a cross-layer fusion left for later). Numerics mirror the eager path
    with the sum taken in fp32.
    """
    from umic.kernels.rmsnorm import add_rmsnorm_triton

    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states, _ = self.self_attn(
        hidden_states=hidden_states, attention_mask=attention_mask,
        position_ids=position_ids, past_key_values=past_key_values,
        use_cache=use_cache, cache_position=cache_position,
        position_embeddings=position_embeddings, **kwargs)
    eps = getattr(self.post_attention_layernorm, "variance_epsilon", None)
    if eps is None:
        eps = getattr(self.post_attention_layernorm, "eps", 1e-6)
    hidden_states, residual = add_rmsnorm_triton(
        hidden_states, residual, self.post_attention_layernorm.weight, eps)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states
    return hidden_states


def fuse_add_rmsnorm(model: nn.Module, dry_run: bool = False) -> int:
    """Fuse the attention-residual add into post_attention_layernorm.

    Patches every pre-norm decoder layer's forward (structural match, so it
    covers the LM trunk and the flow Action Expert with one pass). Falls
    back to nothing for modules that don't match — accuracy never breaks.

    Returns:
        Number of decoder layers patched (unless dry_run).
    """
    count = 0
    for name, module in model.named_modules():
        if not _is_residual_decoder_layer(module):
            continue
        count += 1
        if not dry_run:
            module.forward = _fused_residual_layer_forward.__get__(module)
    logger.info("fuse_add_rmsnorm: %d decoder layer(s) %s", count,
                "matched (dry run)" if dry_run else "patched")
    return count


def _patched_layernorm_forward(self: nn.LayerNorm, x: torch.Tensor) -> torch.Tensor:
    """Single fused kernel replacing eager's fp32 LN + cast chain."""
    from umic.kernels.layernorm import layernorm_triton

    return layernorm_triton(x, self.weight, self.bias, self.eps)


def fuse_layernorms(model: nn.Module, dry_run: bool = False) -> int:
    """Swap forward of nn.LayerNorm modules (last-dim, affine with bias).

    Same pathology as the LM's RMSNorm chain, ViT edition (measured
    ~33 GB of VE traffic). One launch always beats eager's chain, so no
    min-rows dispatch.
    """
    count = 0
    for name, module in model.named_modules():
        if (isinstance(module, nn.LayerNorm)
                and len(module.normalized_shape) == 1
                and module.weight is not None and module.bias is not None):
            count += 1
            if not dry_run:
                module.forward = _patched_layernorm_forward.__get__(module)
    logger.info("fuse_layernorms: %d module(s) %s", count,
                "matched (dry run)" if dry_run else "patched")
    return count


# fc1/fc2 attribute aliases seen across ViT families (HF Qwen-VL, CLIP, …)
_FC1_NAMES = ("fc1", "linear_fc1", "up_proj")
_FC2_NAMES = ("fc2", "linear_fc2", "down_proj")


def _is_gelu_mlp(module: nn.Module) -> tuple[nn.Linear, nn.Linear] | None:
    """Structural match for the fc1 -> GELU -> fc2 motif (no gate)."""
    fc1 = next((getattr(module, n) for n in _FC1_NAMES
                if isinstance(getattr(module, n, None), nn.Linear)), None)
    fc2 = next((getattr(module, n) for n in _FC2_NAMES
                if isinstance(getattr(module, n, None), nn.Linear)), None)
    if fc1 is None or fc2 is None or hasattr(module, "gate_proj"):
        return None
    act = getattr(module, "act_fn", None) or getattr(module, "act", None) \
        or getattr(module, "activation_fn", None)
    name = type(act).__name__.lower() if act is not None else ""
    if "gelu" not in name and getattr(act, "__name__", "") != "gelu":
        return None
    return fc1, fc2


def _fused_gelu_mlp_forward(self: nn.Module, x: torch.Tensor) -> torch.Tensor:
    from umic.kernels.linear import linear_triton

    fc1, fc2 = self._umic_fc  # type: ignore[attr-defined]
    shape = x.shape
    x2d = x.reshape(-1, shape[-1])
    if x2d.shape[0] < FUSE_MIN_ROWS:
        return fc2(self.act_fn(fc1(x)) if hasattr(self, "act_fn")
                   else self.act(fc1(x)))
    h = linear_triton(x2d, fc1.weight, fc1.bias, act="gelu")
    out = nn.functional.linear(h, fc2.weight, fc2.bias)
    return out.reshape(*shape[:-1], out.shape[-1])


def fuse_gelu_mlps(model: nn.Module, dry_run: bool = False) -> int:
    """Fuse fc1+GELU into one kernel for GELU-MLP modules (ViT motif)."""
    count = 0
    for name, module in model.named_modules():
        pair = _is_gelu_mlp(module)
        if pair is None:
            continue
        count += 1
        if not dry_run:
            module._umic_fc = pair
            module.forward = _fused_gelu_mlp_forward.__get__(module)
    logger.info("fuse_gelu_mlps: %d module(s) %s", count,
                "matched (dry run)" if dry_run else "patched")
    return count


def fuse_vision_rope() -> bool:
    """Replace HF's fp32 apply_rotary_pos_emb_vision with the fused kernel.

    Library-function runtime injection (no model/source file change):
    the eager version costs ~26 GB in VE via q.float()/cat/mul/add/cast
    chains; the kernel does identical fp32 math in registers.
    """
    try:
        import transformers.models.qwen3_vl.modeling_qwen3_vl as mod
    except ImportError:
        logger.warning("fuse_vision_rope: qwen3_vl module not found")
        return False
    from umic.kernels.rope import HAS_TRITON, apply_rotary_vision_triton
    if not HAS_TRITON:
        return False
    mod.apply_rotary_pos_emb_vision = apply_rotary_vision_triton
    logger.info("fuse_vision_rope: apply_rotary_pos_emb_vision -> fused kernel")
    return True


def fuse_text_rope() -> bool:
    """Replace HF apply_rotary_pos_emb (text path) with the fused kernel.

    The qwen3_vl module-level function serves the LM (prefill + decode)
    AND the flow Action Expert (Qwen3VLTextModel) — one patch, three
    stages. N>1 / nonstandard layouts fall back to the original.
    """
    try:
        import transformers.models.qwen3_vl.modeling_qwen3_vl as mod
    except ImportError:
        logger.warning("fuse_text_rope: qwen3_vl module not found")
        return False
    from umic.kernels.rope import HAS_TRITON, make_text_rope
    if not HAS_TRITON:
        return False
    if getattr(mod.apply_rotary_pos_emb, "_umic_fused", False):
        return True
    fused = make_text_rope(mod.apply_rotary_pos_emb)
    fused._umic_fused = True  # type: ignore[attr-defined]
    mod.apply_rotary_pos_emb = fused
    logger.info("fuse_text_rope: apply_rotary_pos_emb -> fused kernel")
    return True


def fuse_bf16_residual(visual: nn.Module, dtype: torch.dtype = torch.bfloat16) -> bool:
    """Demote the ViT residual stream back to bf16 at the first block.

    Measured root cause (results/260610_m3_ve): fp32 pos-embeds promote
    `hidden + pos_embeds` to fp32, so every block then runs fp32 residual
    adds plus bf16 casts at each Linear boundary (~25 GB of VE traffic).
    All compute already runs in bf16 under autocast — the fp32 stream
    only buys cast/copy traffic. One cast at block 0 restores bf16
    end-to-end. Output equivalence is gated by 260610_ve_bf16_probe.py.
    """
    blocks = getattr(visual, "blocks", None)
    if not blocks:
        logger.warning("fuse_bf16_residual: no .blocks on %s", type(visual).__name__)
        return False

    def _pre(module, args, kwargs):
        hs = args[0] if args else kwargs.get("hidden_states")
        if hs is None or hs.dtype == dtype:
            return None
        if args:
            return (hs.to(dtype),) + args[1:], kwargs
        kwargs["hidden_states"] = hs.to(dtype)
        return args, kwargs

    blocks[0].register_forward_pre_hook(_pre, with_kwargs=True)
    logger.info("fuse_bf16_residual: cast hook on %s.blocks[0]",
                type(visual).__name__)
    return True


def fuse_repeat_cache(repeat_module: nn.Module, extra_tokens_hint: int = 64) -> None:
    """Eliminate per-step KV cat-copies in append-then-crop repeat loops.

    Registers a forward pre-hook on `repeat_module` (e.g. the flow Action
    Expert): the first time a plain DynamicCache arrives as
    past_key_values, it is converted once to InplaceKVCache; on every
    call the substitute's write pointer is re-synced to the caller's
    cache length, which makes the caller's external `.crop()` redundant
    without touching model source.
    """
    from transformers.cache_utils import DynamicCache

    from umic.cache import InplaceKVCache

    state: dict[str, object] = {"orig_id": None, "converted": None}

    def _pre_hook(module, args, kwargs):
        pkv = kwargs.get("past_key_values")
        if pkv is None or isinstance(pkv, InplaceKVCache):
            return None
        if not isinstance(pkv, DynamicCache) or not getattr(pkv, "layers", None):
            return None
        # generate() pre-creates the cache with lazy (keys=None) layers
        # before prefill — only convert once every layer holds real KV.
        if any(getattr(lyr, "keys", None) is None for lyr in pkv.layers):
            return None
        n_new = 0
        emb = kwargs.get("inputs_embeds")
        if emb is not None:
            n_new = emb.shape[1]
        if state["orig_id"] != id(pkv):
            state["orig_id"] = id(pkv)
            state["converted"] = InplaceKVCache.from_dynamic(
                pkv, extra_tokens=max(n_new, extra_tokens_hint))
        conv: InplaceKVCache = state["converted"]  # type: ignore[assignment]
        conv.crop(pkv.get_seq_length())  # external crop() emulation, O(1)
        kwargs["past_key_values"] = conv
        return args, kwargs

    repeat_module.register_forward_pre_hook(_pre_hook, with_kwargs=True)
    logger.info("fuse_repeat_cache: hook installed on %s",
                type(repeat_module).__name__)


def unfuse_mlps(model: nn.Module) -> int:
    """Restore original forwards (delete instance overrides)."""
    count = 0
    for _, module in model.named_modules():
        if "forward" in module.__dict__:
            del module.__dict__["forward"]
            count += 1
    return count
