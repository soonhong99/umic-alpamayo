"""In-place KV cache: kills the per-step cat-copy in repeated-stage loops.

Measured motivation (results/260610_m2_flow): Alpamayo's flow step_fn
appends expert tokens to the shared VLM KV cache and crops it back every
ODE step. DynamicLayer.update() concatenates — the whole 3,094-token
prefix (36 layers) is copied twice per step. Across 2 step_fns x 10 ODE
steps that is the cat/copy cluster (~28 GB) dominating flow DRAM waste.

InplaceLayer pre-allocates once, writes new tokens in place, and crop()
just moves the write pointer. Built against transformers >= 4.56 cache
API (Cache.layers / DynamicLayer); generalizes the project's
AppendOnlyCache-C (decode, 2026-05-31) to any append-then-crop loop.
"""

from __future__ import annotations

import logging

import torch
from transformers.cache_utils import DynamicCache, DynamicLayer

logger = logging.getLogger(__name__)


class InplaceLayer(DynamicLayer):
    """One layer's KV in a pre-allocated buffer; update is copy-free.

    NOTE: returned keys/values are strided prefix views of the buffer
    (no .contiguous() — that would re-copy the prefix and defeat the
    point; flow q_len≈32 attention handles strided KV).
    """

    def __init__(self, k_buf: torch.Tensor, v_buf: torch.Tensor,
                 base_len: int) -> None:
        super().__init__()
        self._k_buf = k_buf
        self._v_buf = v_buf
        self._pos = base_len
        self.keys = k_buf[:, :, :base_len, :]
        self.values = v_buf[:, :, :base_len, :]
        # CUDA-Graph mode (set by DecodeGraphRunner): single-token writes
        # go through index_copy_ with a device position tensor (replay-
        # safe), and the FULL buffer is returned so every replay sees a
        # static shape. Validity of the padded tail is handled by the
        # runner's padded 2D attention mask.
        self.graph_mode = False
        self._pos_t: torch.Tensor | None = None
        self._graph_len = 0  # exact KV length baked into the current graph

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor,
               cache_kwargs=None):
        n = key_states.shape[2]
        if self.graph_mode and n == 1 and self._pos_t is not None:
            # Replay-safe write at a device index; return the EXACT-length
            # prefix CONTIGUOUS (locked-clock A/B 2026-06-11: contiguous
            # KV lets flash/GEMV reach 93% BW = 79 ms/step; strided views
            # cap at ~96 ms. The copy runs inside the graph, alloc comes
            # from the graph pool.)
            self._k_buf.index_copy_(2, self._pos_t, key_states)
            self._v_buf.index_copy_(2, self._pos_t, value_states)
            self.keys = self._k_buf[:, :, :self._graph_len, :].contiguous()
            self.values = self._v_buf[:, :, :self._graph_len, :].contiguous()
            return self.keys, self.values
        pos = self._pos
        if pos + n > self._k_buf.shape[2]:
            raise RuntimeError(
                f"InplaceLayer overflow: pos {pos} + new {n} > "
                f"buffer {self._k_buf.shape[2]}")
        self._k_buf[:, :, pos:pos + n, :] = key_states
        self._v_buf[:, :, pos:pos + n, :] = value_states
        self._pos = pos + n
        self.keys = self._k_buf[:, :, :self._pos, :]
        self.values = self._v_buf[:, :, :self._pos, :]
        if n == 1:
            # Locked-clock A/B (2026-06-11): contiguous KV reaches 93% BW
            # (79 ms/step standalone); strided views cap lower. Earlier
            # "views win" call was made under the default DVFS governor —
            # a hidden variable, conclusion withdrawn.
            return self.keys.contiguous(), self.values.contiguous()
        return self.keys, self.values

    def get_seq_length(self, cache_position=None) -> int:
        return self._pos

    def crop(self, max_length: int) -> None:
        """O(1): move the write pointer back; no slicing copies."""
        if max_length < 0:
            max_length = self._pos + max_length
        self._pos = min(self._pos, max_length)
        self.keys = self._k_buf[:, :, :self._pos, :]
        self.values = self._v_buf[:, :, :self._pos, :]


class InplaceKVCache(DynamicCache):
    """DynamicCache whose layers are pre-allocated InplaceLayers.

    Inheriting DynamicCache keeps HF mask/back-end selection identical.
    """

    def __init__(self, layers: list[InplaceLayer]) -> None:
        super().__init__()
        self.layers = layers

    @classmethod
    def from_dynamic(cls, cache: DynamicCache, extra_tokens: int,
                     margin: int = 8) -> "InplaceKVCache":
        """One-time conversion: pre-allocate and copy the existing prefix.

        Costs one prefix copy total; every subsequent append-then-crop
        cycle is then copy-free (vs one full prefix copy per layer per
        step under DynamicLayer.update).
        """
        layers: list[InplaceLayer] = []
        alloc = 0
        for lyr in cache.layers:
            k, v = lyr.keys, lyr.values
            b, h, base_len, d = k.shape
            kb = torch.empty(b, h, base_len + extra_tokens + margin, d,
                             device=k.device, dtype=k.dtype)
            vb = torch.empty_like(kb)
            kb[:, :, :base_len, :] = k
            vb[:, :, :base_len, :] = v
            layers.append(InplaceLayer(kb, vb, base_len))
            alloc += kb.numel() * kb.element_size() * 2
        logger.info("InplaceKVCache: %d layers converted, base_len=%d, "
                    "alloc=%.0f MB", len(layers),
                    layers[0]._pos if layers else 0, alloc / 1e6)
        return cls(layers)

    def crop(self, max_length: int) -> None:
        for lyr in self.layers:
            lyr.crop(max_length)

    # --- CUDA-Graph mode plumbing (driven by umic.graph) ---------------

    def enable_graph_mode(self, pos_t: torch.Tensor) -> None:
        """Static-shape decode: index_copy_ writes, full-buffer reads."""
        for lyr in self.layers:
            lyr.graph_mode = True
            lyr._pos_t = pos_t

    def disable_graph_mode(self) -> None:
        """Back to dynamic semantics (flow append-then-crop etc.)."""
        for lyr in self.layers:
            lyr.graph_mode = False
            lyr.keys = lyr._k_buf[:, :, :lyr._pos, :]
            lyr.values = lyr._v_buf[:, :, :lyr._pos, :]

    def set_pos(self, n: int) -> None:
        """Sync the python write pointer after graph replays."""
        for lyr in self.layers:
            lyr._pos = n

    def set_graph_len(self, n: int) -> None:
        """Exact KV length for the graph being captured/replayed."""
        for lyr in self.layers:
            lyr._graph_len = n

    def get_seq_length(self, layer_idx: int = 0, cache_position=None) -> int:
        return self.layers[layer_idx].get_seq_length()
