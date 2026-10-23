"""Fake quantisation (quantise -> dequantise) for KV tensors.

All functions take and return tensors shaped [batch, kv_heads, seq, head_dim]
in the model dtype. What comes back is what an attention kernel would read
from a cache stored in the compressed format.

Schemes mirror the formats studied in KVShuttle (Tiwari, 2026) and the KV
formats available in vLLM:
  fp8_e4m3   per-tensor scale 1.0 (vLLM `--kv-cache-dtype fp8`, uncalibrated)
  fp8_e5m2   same, E5M2
  int8_tok   symmetric INT8, one scale per (token, head)
  int4_tok   asymmetric INT4, one scale/zero per (token, head)  (vLLM >= 0.25)
  int4_g32   asymmetric INT4, groups of 32 along head_dim
  kivi2/4    KIVI: keys per-channel over 32-token groups, values per-token
             groups of 32, most recent <32 tokens kept in full precision
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

FP8_MAX = {"e4m3": 448.0, "e5m2": 57344.0}


def fp8(x: torch.Tensor, fmt: str = "e4m3", scale: float | None = 1.0) -> torch.Tensor:
    """FP8 round-trip. scale=None uses a dynamic per-tensor amax scale."""
    dt = torch.float8_e4m3fn if fmt == "e4m3" else torch.float8_e5m2
    xf = x.float()
    if scale is None:
        amax = xf.abs().amax().clamp(min=1e-12)
        s = amax / FP8_MAX[fmt]
    else:
        s = torch.tensor(scale, dtype=torch.float32, device=x.device)
    y = (xf / s).clamp(-FP8_MAX[fmt], FP8_MAX[fmt]).to(dt).float() * s
    return y.to(x.dtype)


def _grouped(x: torch.Tensor, group: int, axis: int) -> tuple[torch.Tensor, Callable]:
    """Move `axis` last and split it into groups of `group` elements."""
    xt = x.movedim(axis, -1)
    n = xt.shape[-1]
    if n % group:
        raise ValueError(f"axis length {n} not divisible by group {group}")
    shape = xt.shape
    g = xt.reshape(*shape[:-1], n // group, group)

    def restore(y: torch.Tensor) -> torch.Tensor:
        return y.reshape(shape).movedim(-1, axis)

    return g, restore


def int_sym(x: torch.Tensor, bits: int, group: int | None = None, axis: int = -1) -> torch.Tensor:
    """Symmetric integer round-trip with one scale per group along `axis`."""
    group = group or x.shape[axis]
    g, restore = _grouped(x.float(), group, axis)
    qmax = 2 ** (bits - 1) - 1
    s = (g.abs().amax(-1, keepdim=True) / qmax).clamp(min=1e-12)
    y = (g / s).round().clamp(-qmax, qmax) * s
    return restore(y).to(x.dtype)


def int_asym(x: torch.Tensor, bits: int, group: int | None = None, axis: int = -1) -> torch.Tensor:
    """Asymmetric (min/max) integer round-trip with one scale+zero per group along `axis`."""
    group = group or x.shape[axis]
    g, restore = _grouped(x.float(), group, axis)
    qmax = 2**bits - 1
    lo = g.amin(-1, keepdim=True)
    hi = g.amax(-1, keepdim=True)
    s = ((hi - lo) / qmax).clamp(min=1e-12)
    y = ((g - lo) / s).round().clamp(0, qmax) * s + lo
    return restore(y).to(x.dtype)


@dataclass(frozen=True)
class Scheme:
    """A KV compression scheme.

    key/value: functions applied to a block of tokens [B, H, S, D].
    residual: if > 0, tokens are only quantised in complete groups of this many
        tokens (KIVI); the trailing partial group stays in full precision.
    bits_per_element: storage cost incl. scale/zero overhead (for kvmath).
    """

    name: str
    key: Callable[[torch.Tensor], torch.Tensor]
    value: Callable[[torch.Tensor], torch.Tensor]
    residual: int = 0
    bits_per_element: float = 16.0

    @property
    def is_identity(self) -> bool:
        return self.name == "bf16"


def _ident(x):
    return x


def _kivi_key(bits, group=32):
    # per-channel: statistics over the token axis (axis=-2), groups of `group` tokens
    return lambda x: int_asym(x, bits, group=group, axis=-2)


def _per_token(bits, group=32):
    # per-token: groups of `group` channels along head_dim (whole head if smaller)
    return lambda x: int_asym(x, bits, group=min(group, x.shape[-1]), axis=-1)


def _overhead(group: int, bits: float) -> float:
    # one fp16 scale (+ one fp16 zero for asymmetric) per group
    return bits + 32.0 / group


SCHEMES: dict[str, Scheme] = {
    "bf16": Scheme("bf16", _ident, _ident, 0, 16.0),
    "fp8_e4m3": Scheme("fp8_e4m3", lambda x: fp8(x, "e4m3", 1.0), lambda x: fp8(x, "e4m3", 1.0), 0, 8.0),
    "fp8_e4m3_dyn": Scheme("fp8_e4m3_dyn", lambda x: fp8(x, "e4m3", None), lambda x: fp8(x, "e4m3", None), 0, 8.0),
    "fp8_e5m2": Scheme("fp8_e5m2", lambda x: fp8(x, "e5m2", 1.0), lambda x: fp8(x, "e5m2", 1.0), 0, 8.0),
    "int8_tok": Scheme("int8_tok", lambda x: int_sym(x, 8), lambda x: int_sym(x, 8), 0, 8.0 + 16.0 / 256),
    "int4_tok": Scheme("int4_tok", lambda x: int_asym(x, 4), lambda x: int_asym(x, 4), 0, _overhead(256, 4)),
    "int4_g32": Scheme("int4_g32", _per_token(4), _per_token(4), 0, _overhead(32, 4)),
    "kivi4": Scheme("kivi4", _kivi_key(4), _per_token(4), 32, _overhead(32, 4)),
    "kivi2": Scheme("kivi2", _kivi_key(2), _per_token(2), 32, _overhead(32, 2)),
}


def get_scheme(name: str) -> Scheme:
    if name not in SCHEMES:
        raise KeyError(f"unknown scheme {name!r}; available: {sorted(SCHEMES)}")
    return SCHEMES[name]
