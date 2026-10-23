"""Aggregate closed-loop runs and open-loop replay into paper tables and figures.

Expected run layout (written by scripts/run_arm.sh):

    runs/<arm>/<run_id>/requests.jsonl   proxy log
    runs/<arm>/<run_id>/results/         harness output (pass/fail per task)
    runs/<arm>/<run_id>/meta.json        arm, flags, timings, KV pool size
    runs/<arm>/<run_id>/vllm.log
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from kvloop.results import load_results
from kvloop.stats import bootstrap_ci, mcnemar_exact, wilson
from kvloop.trajectories import behaviour, group, match_tasks, read_jsonl

KV_TOKENS_RE = re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens")
CONC_RE = re.compile(r"Maximum concurrency for ([\d,]+) tokens per request:\s*([\d.]+)x")


def parse_vllm_log(path: Path) -> dict:
    if not path.exists():
        return {}
    txt = path.read_text(errors="replace")
    out = {}
    m = KV_TOKENS_RE.findall(txt)
    if m:
        out["kv_cache_tokens"] = int(m[-1].replace(",", ""))
    m = CONC_RE.findall(txt)
    if m:
        out["max_concurrency_ctx"] = int(m[-1][0].replace(",", ""))
        out["max_concurrency"] = float(m[-1][1])
    return out


def load_closed(runs_root: str | Path, tasks: list[dict] | None = None) -> pd.DataFrame:
    """One row per (arm, run, trajectory) with behaviour metrics and pass/fail."""
    rows = []
    for run_dir in sorted(Path(runs_root).glob("*/*")):
        log = run_dir / "requests.jsonl"
        if not log.exists():
            continue
        arm, run = run_dir.parent.name, run_dir.name
        meta = json.loads((run_dir / "meta.json").read_text()) if (run_dir / "meta.json").exists() else {}
        meta.update(parse_vllm_log(run_dir / "vllm.log"))
        res = load_results(run_dir / "results") if (run_dir / "results").exists() else {}
        trajs = group(read_jsonl(log))
        if tasks:
            match_tasks(trajs, tasks)
        for tr in trajs:
            b = behaviour(tr)
            b.update(arm=arm, run=run, resolved=res.get(tr.task_id) if tr.task_id else None,
                     kv_cache_tokens=meta.get("kv_cache_tokens"), run_wall_s=meta.get("wall_s"))
            rows.append(b)
    return pd.DataFrame(rows)


def closed_summary(df: pd.DataFrame, baseline: str = "bf16") -> pd.DataFrame:
    out = []
    base = df[df.arm == baseline]
    base_task = base.dropna(subset=["resolved"]).groupby("task_id").resolved.mean()
    for arm, g in df.groupby("arm"):
        r = g.dropna(subset=["resolved"])
        k, n = int(r.resolved.sum()), len(r)
        lo, hi = wilson(k, n)
        row = {
            "arm": arm, "runs": g.run.nunique(), "trajectories": len(g), "resolved": k, "scored": n,
            "resolve_rate": k / n if n else np.nan, "ci_lo": lo, "ci_hi": hi,
            "malformed_per_traj": g.malformed.mean(), "repeated_calls": g.repeated_calls.mean(),
            "max_spin": g.max_spin.mean(), "turns": g.turns.mean(),
            "completion_tokens": g.completion_tokens.mean(), "max_prompt_tokens": g.max_prompt_tokens.mean(),
            "context_errors": g.context_errors.sum(), "kv_cache_tokens": g.kv_cache_tokens.dropna().median()
            if g.kv_cache_tokens.notna().any() else np.nan,
        }
        walls = g.groupby("run").run_wall_s.first().dropna()
        if len(walls) and n:
            row["resolved_per_gpu_hour"] = k / (walls.sum() / 3600 * 4)  # 4 GPUs
        if arm != baseline and len(base_task):
            arm_task = r.groupby("task_id").resolved.mean()
            common = base_task.index.intersection(arm_task.index)
            if len(common):
                d = (arm_task[common] - base_task[common]).values
                row["delta_vs_base"] = float(d.mean())
                row["delta_ci_lo"], row["delta_ci_hi"] = bootstrap_ci(d)
                bb, aa = base_task[common] >= 0.5, arm_task[common] >= 0.5
                row["mcnemar_p"] = mcnemar_exact(int((bb & ~aa).sum()), int((~bb & aa).sum()))
        out.append(row)
    return pd.DataFrame(out).sort_values("arm")


def open_summary(replay: pd.DataFrame) -> pd.DataFrame:
    """Per arm and segment: drift metrics averaged over turns, plus turn-level divergence."""
    g = replay.groupby(["arm", "segment"])
    s = g.agg(turns=("turn", "size"), tokens=("n", "sum"), d_nll=("d_nll_mean", "mean"),
              kl=("kl_mean", "mean"), flip_rate=("flip_rate", "mean"),
              turn_divergence=("any_flip", "mean")).reset_index()
    return s


def drift_by_context(replay: pd.DataFrame, bins=(0, 4096, 8192, 16384, 24576, 32768, 10**9)) -> pd.DataFrame:
    r = replay[replay.segment == "all"].copy()
    r["ctx_bin"] = pd.cut(r.prompt_len, bins=list(bins), right=False)
    return r.groupby(["arm", "ctx_bin"], observed=True).agg(
        turns=("turn", "size"), d_nll=("d_nll_mean", "mean"), flip_rate=("flip_rate", "mean"),
        turn_divergence=("any_flip", "mean")).reset_index()


def compounding_model(p_turn: float, turns: float) -> float:
    """Probability a trajectory sees at least one action divergence, if turns were independent."""
    return 1 - (1 - p_turn) ** turns


# ---------------------------------------------------------------- figures
def fig_kv_memory(layout, arms, contexts, path):
    import matplotlib.pyplot as plt

    from kvloop.kvmath import kv_bytes

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for arm in arms:
        y = [kv_bytes(layout, arm, c).total_bytes / 2**30 for c in contexts]
        ax.plot(np.array(contexts) / 1024, y, marker="o", ms=3, label=arm.name)
    ax.set_xlabel("context (K tokens)")
    ax.set_ylabel("KV cache per agent (GiB)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_open_vs_closed(open_s: pd.DataFrame, closed_s: pd.DataFrame, mean_turns: float, path):
    import matplotlib.pyplot as plt

    a = open_s[open_s.segment == "action"].set_index("arm")
    c = closed_s.set_index("arm")
    common = [x for x in a.index if x in c.index and "delta_vs_base" in c.columns and pd.notna(c.loc[x, "delta_vs_base"])]
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for arm in common:
        ax.scatter(a.loc[arm, "turn_divergence"], c.loc[arm, "delta_vs_base"])
        ax.annotate(arm, (a.loc[arm, "turn_divergence"], c.loc[arm, "delta_vs_base"]), fontsize=7)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("open-loop: P(action token flips) per turn")
    ax.set_ylabel("closed-loop: Δ resolve rate vs BF16")
    ax.set_title(f"mean trajectory length {mean_turns:.0f} turns", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
