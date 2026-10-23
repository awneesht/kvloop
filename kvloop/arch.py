"""Attention-layout description for hybrid sliding/global models (Gemma 4).

Gemma 4 31B interleaves 5 sliding-window layers with 1 global layer (60 layers:
50 sliding, 10 global). Sliding layers: 16 KV heads, head_dim 256, window 1024.
Global layers: 4 KV heads, head_dim 512, and K = V (no v_proj).

Defaults below come from public descriptions of the 31B config; prefer
`AttentionLayout.from_config_json` on the real config.json when it is available.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

SLIDING = "sliding_attention"
GLOBAL = "full_attention"


@dataclass
class AttentionLayout:
    layer_types: list[str]
    sliding_kv_heads: int
    sliding_head_dim: int
    global_kv_heads: int
    global_head_dim: int
    sliding_window: int
    global_k_eq_v: bool = True
    num_kv_shared_layers: int = 0
    name: str = "custom"
    notes: list[str] = field(default_factory=list)

    @property
    def num_layers(self) -> int:
        return len(self.layer_types)

    def indices(self, kind: str) -> list[int]:
        """Layer indices of a kind: 'sliding', 'global' or 'all'."""
        if kind == "all":
            return list(range(self.num_layers))
        target = {"sliding": SLIDING, "global": GLOBAL}[kind]
        return [i for i, t in enumerate(self.layer_types) if t == target]

    # ------------------------------------------------------------------ builders
    @classmethod
    def gemma4_31b(cls) -> "AttentionLayout":
        types = ([SLIDING] * 5 + [GLOBAL]) * 10
        return cls(
            layer_types=types,
            sliding_kv_heads=16,
            sliding_head_dim=256,
            global_kv_heads=4,
            global_head_dim=512,
            sliding_window=1024,
            global_k_eq_v=True,
            name="gemma-4-31b",
            notes=["defaults from public config descriptions; verify against config.json"],
        )

    @classmethod
    def from_config_json(cls, path: str | Path) -> "AttentionLayout":
        """Build from a Hugging Face config.json (handles a nested text_config)."""
        cfg = json.loads(Path(path).read_text())
        return cls.from_config_dict(cfg)

    @classmethod
    def from_config_dict(cls, cfg: dict) -> "AttentionLayout":
        text = cfg.get("text_config", cfg)
        types = text.get("layer_types")
        n = text["num_hidden_layers"]
        if not types:
            raise ValueError("config has no layer_types; cannot tell sliding from global layers")
        types = list(types)[:n]
        kv = text.get("num_key_value_heads")
        hd = text.get("head_dim")
        g_kv = text.get("num_global_key_value_heads") or kv
        g_hd = text.get("global_head_dim") or hd
        return cls(
            layer_types=types,
            sliding_kv_heads=kv,
            sliding_head_dim=hd,
            global_kv_heads=g_kv,
            global_head_dim=g_hd,
            sliding_window=text.get("sliding_window") or 0,
            global_k_eq_v=bool(text.get("attention_k_eq_v", False)),
            num_kv_shared_layers=int(text.get("num_kv_shared_layers") or 0),
            name=cfg.get("_name_or_path", "from-config"),
        )
