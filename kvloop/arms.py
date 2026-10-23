"""Experimental arms: one KV-cache precision setting per arm.

Each arm can have two realisations:
  * `vllm`   - flags for the real serving stack (closed-loop runs). Only FP8 is
               available in the scorer's vLLM 0.19.1, with per-layer skipping.
  * `replay` - a fake-quantisation scheme for the teacher-forced replay engine
               (open-loop runs), which can also express INT8/INT4/KIVI.

`layers` says which attention layers are compressed: all, sliding or global.
The rest stay in the model dtype (BF16).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from kvloop.arch import AttentionLayout

LAYER_SETS = ("all", "sliding", "global", "none")


@dataclass
class Arm:
    name: str
    scheme: str  # replay scheme name, see kvloop.quant.SCHEMES
    layers: str = "all"  # which layers are compressed
    vllm_kv_cache_dtype: str | None = None  # None -> not servable on the scorer's vLLM
    description: str = ""
    extra_vllm_args: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.layers not in LAYER_SETS:
            raise ValueError(f"arm {self.name}: layers must be one of {LAYER_SETS}")

    @property
    def closed_loop_capable(self) -> bool:
        return self.vllm_kv_cache_dtype is not None

    def vllm_args(self, layout: AttentionLayout) -> list[str]:
        """vLLM CLI flags that realise this arm (empty list for the baseline)."""
        if not self.closed_loop_capable:
            raise ValueError(f"arm {self.name!r} has no vLLM realisation")
        args: list[str] = []
        if self.vllm_kv_cache_dtype != "auto" and self.layers != "none":
            args += ["--kv-cache-dtype", self.vllm_kv_cache_dtype]
            if self.layers == "global":
                # compress global layers only -> keep sliding layers in model dtype
                args += ["--kv-cache-dtype-skip-layers", "sliding_window"]
            elif self.layers == "sliding":
                # compress sliding layers only -> skip every global layer by index
                args += ["--kv-cache-dtype-skip-layers", *map(str, layout.indices("global"))]
        return args + list(self.extra_vllm_args)


DEFAULT_ARMS: list[Arm] = [
    Arm("bf16", "bf16", "none", "auto", "Baseline: KV cache in model dtype"),
    Arm("fp8_all", "fp8_e4m3", "all", "fp8", "FP8 E4M3, uncalibrated scale=1, every layer"),
    Arm("fp8_global", "fp8_e4m3", "global", "fp8", "FP8 on global layers only (vLLM skip sliding_window)"),
    Arm("fp8_sliding", "fp8_e4m3", "sliding", "fp8", "FP8 on sliding layers only (skip global indices)"),
    Arm("int8_all", "int8_tok", "all", None, "INT8 symmetric per token-head (replay only)"),
    Arm("int4_all", "int4_tok", "all", None, "INT4 asymmetric per token-head, as in vLLM>=0.25 (replay only)"),
    Arm("int4_global", "int4_tok", "global", None, "INT4 on global layers only (replay only)"),
    Arm("int4_sliding", "int4_tok", "sliding", None, "INT4 on sliding layers only (replay only)"),
    Arm("kivi2_all", "kivi2", "all", None, "KIVI 2-bit: per-channel K, per-token V, 32-token residual (replay only)"),
]


def load_arms(path: str | Path | None = None) -> list[Arm]:
    if path is None:
        return list(DEFAULT_ARMS)
    data = yaml.safe_load(Path(path).read_text())
    return [Arm(**a) for a in data["arms"]]


def get_arm(name: str, arms: list[Arm] | None = None) -> Arm:
    for a in arms or DEFAULT_ARMS:
        if a.name == name:
            return a
    raise KeyError(f"unknown arm {name!r}")
