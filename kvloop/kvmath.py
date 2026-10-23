"""Analytic KV-cache memory model for hybrid sliding/global attention.

Answers: for a given arm, how many bytes does one agent's KV cache take at
context length C, and how many agents fit in the KV pool at once?

Sliding layers only keep `sliding_window` tokens; global layers keep all C.
Global K=V layers may store K once (`store_kv_once=True`) or twice, depending
on the serving engine; vLLM's generic attention path stores K and V, so the
default is False. Check against the "GPU KV cache size: N tokens" line in the
vLLM log, which `kvloop probe` records.
"""

from __future__ import annotations

from dataclasses import dataclass

from kvloop.arch import AttentionLayout
from kvloop.arms import Arm
from kvloop.quant import get_scheme


@dataclass
class KVBreakdown:
    arm: str
    context: int
    sliding_bytes: float
    global_bytes: float

    @property
    def total_bytes(self) -> float:
        return self.sliding_bytes + self.global_bytes

    @property
    def global_share(self) -> float:
        return self.global_bytes / self.total_bytes if self.total_bytes else 0.0


def _bits(arm: Arm, kind: str) -> float:
    compressed = arm.layers == "all" or arm.layers == kind
    if not compressed or arm.layers == "none":
        return 16.0
    return get_scheme(arm.scheme).bits_per_element


def kv_bytes(layout: AttentionLayout, arm: Arm, context: int, store_kv_once: bool = False) -> KVBreakdown:
    """KV bytes for ONE sequence of `context` tokens (all GPUs combined)."""
    s_tok = min(context, layout.sliding_window) if layout.sliding_window else context
    n_s = len(layout.indices("sliding"))
    n_g = len(layout.indices("global"))
    s_elems = 2 * layout.sliding_kv_heads * layout.sliding_head_dim
    g_mult = 1 if (layout.global_k_eq_v and store_kv_once) else 2
    g_elems = g_mult * layout.global_kv_heads * layout.global_head_dim
    sb = n_s * s_tok * s_elems * _bits(arm, "sliding") / 8
    gb = n_g * context * g_elems * _bits(arm, "global") / 8
    return KVBreakdown(arm.name, context, sb, gb)


def concurrency(
    layout: AttentionLayout,
    arm: Arm,
    context: int,
    kv_pool_bytes: float,
    store_kv_once: bool = False,
) -> float:
    """How many sequences of `context` tokens fit in a KV pool of `kv_pool_bytes`."""
    return kv_pool_bytes / kv_bytes(layout, arm, context, store_kv_once).total_bytes


def estimate_kv_pool(
    total_gpu_bytes: float = 4 * 24e9,
    gpu_util: float = 0.80,
    weight_bytes: float = 20e9,
    overhead_bytes: float = 4 * 1.5e9,
) -> float:
    """Rough KV pool size; replace with the number vLLM logs at start-up."""
    return total_gpu_bytes * gpu_util - weight_bytes - overhead_bytes


def table(layout: AttentionLayout, arms: list[Arm], contexts: list[int], kv_pool_bytes: float,
          store_kv_once: bool = False) -> list[dict]:
    rows = []
    for arm in arms:
        for c in contexts:
            b = kv_bytes(layout, arm, c, store_kv_once)
            rows.append({
                "arm": arm.name,
                "context": c,
                "kv_gib_per_seq": b.total_bytes / 2**30,
                "global_share": b.global_share,
                "max_concurrent": kv_pool_bytes / b.total_bytes,
            })
    return rows
