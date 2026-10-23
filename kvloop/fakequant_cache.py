"""A Hugging Face DynamicCache whose stored K/V go through a compression scheme.

Works with any model whose cache layers expose `update(key, value, ...)` and
`is_sliding` (transformers >= 5). Layers are selected by type, so a Gemma 4
cache can compress only sliding layers, only global layers, or all of them.

Residual schemes (KIVI) keep a raw buffer per layer: incoming tokens are
appended raw, and whenever a complete group of `residual` raw tokens exists it
is quantised, both in the stored cache and in the tensors returned for the
current forward pass. This reproduces KIVI's "recent tokens stay full
precision" behaviour during both chunked prefill and token-by-token decode.
"""

from __future__ import annotations

import types

import torch

from kvloop.quant import Scheme, get_scheme


def _selected(layer, layers: str) -> bool:
    if layers == "all":
        return True
    if layers == "none":
        return False
    sliding = bool(getattr(layer, "is_sliding", False))
    return sliding if layers == "sliding" else not sliding


class _ResidualState:
    def __init__(self):
        self.k = None  # raw tokens not yet quantised (tail of the stored cache)
        self.v = None


def _wrap_plain(layer, scheme: Scheme):
    orig = layer.update

    def update(self, key_states, value_states, *args, **kwargs):
        return orig(scheme.key(key_states), scheme.value(value_states), *args, **kwargs)

    layer.update = types.MethodType(update, layer)


def _wrap_residual(layer, scheme: Scheme):
    orig = layer.update
    G = scheme.residual
    st = _ResidualState()

    def process(buf_old, new, fn):
        """Return (fixed_tail_for_stored, appended_block, new_raw_buffer)."""
        r = 0 if buf_old is None else buf_old.shape[-2]
        buf = new if buf_old is None else torch.cat([buf_old, new], dim=-2)
        m = (buf.shape[-2] // G) * G
        if m == 0:
            return None, new, buf
        q = fn(buf[..., :m, :]) if m else buf[..., :0, :]
        merged = torch.cat([q, buf[..., m:, :]], dim=-2)
        fixed_tail = merged[..., :r, :] if r else None  # replaces raw tail already stored
        return fixed_tail, merged[..., r:, :], buf[..., m:, :]

    def update(self, key_states, value_states, *args, **kwargs):
        k_tail, k_new, st.k = process(st.k, key_states, scheme.key)
        v_tail, v_new, st.v = process(st.v, value_states, scheme.value)
        if k_tail is not None and getattr(self, "is_initialized", False) and self.keys.numel():
            r = k_tail.shape[-2]
            self.keys[..., -r:, :] = k_tail
            self.values[..., -r:, :] = v_tail
        return orig(k_new, v_new, *args, **kwargs)

    layer.update = types.MethodType(update, layer)


def make_fakequant_cache(model_config, scheme_name: str, layers: str = "all"):
    """Build a DynamicCache for `model_config` with compression applied to `layers`."""
    from transformers import DynamicCache

    cache = DynamicCache(config=model_config)
    scheme = get_scheme(scheme_name)
    if scheme.is_identity or layers == "none":
        return cache
    n_wrapped = 0
    for layer in cache.layers:
        if not _selected(layer, layers):
            continue
        (_wrap_residual if scheme.residual else _wrap_plain)(layer, scheme)
        n_wrapped += 1
    cache.kvloop_info = {"scheme": scheme_name, "layers": layers, "wrapped": n_wrapped}
    return cache
