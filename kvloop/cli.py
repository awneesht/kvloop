"""kvloop command line.

  kvloop arms        [--config-json C]               list arms and their vLLM flags
  kvloop serve-args  ARM [--config-json C]           print vLLM flags for one arm (used by scripts)
  kvloop kvmath      [--pool-gib G] [--config-json C] KV memory per agent and max concurrency
  kvloop behaviour   LOG [--out CSV]                 closed-loop behaviour metrics from a proxy log
  kvloop replay      --model M --log LOG --out CSV    open-loop teacher-forced replay
  kvloop analyze     --runs DIR [--replay CSV] --out DIR
  kvloop proxy       ...                             logging proxy (see kvloop.proxy)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _layout(path):
    from kvloop.arch import AttentionLayout

    return AttentionLayout.from_config_json(path) if path else AttentionLayout.gemma4_31b()


def cmd_arms(a):
    from kvloop.arms import load_arms

    lay = _layout(a.config_json)
    for arm in load_arms(a.arms):
        flags = " ".join(arm.vllm_args(lay)) if arm.closed_loop_capable else "(replay only)"
        print(f"{arm.name:14s} scheme={arm.scheme:10s} layers={arm.layers:8s} vllm: {flags or '(defaults)'}")


def cmd_serve_args(a):
    from kvloop.arms import get_arm, load_arms

    print(" ".join(get_arm(a.arm, load_arms(a.arms)).vllm_args(_layout(a.config_json))))


def cmd_kvmath(a):
    from kvloop.arms import load_arms
    from kvloop.kvmath import estimate_kv_pool, table

    lay = _layout(a.config_json)
    pool = a.pool_gib * 2**30 if a.pool_gib else estimate_kv_pool()
    rows = table(lay, load_arms(a.arms), a.contexts, pool, a.store_kv_once)
    print(f"layout={lay.name} sliding={len(lay.indices('sliding'))} global={len(lay.indices('global'))} "
          f"window={lay.sliding_window} pool={pool / 2**30:.1f} GiB")
    print(f"{'arm':14s} {'ctx':>6s} {'GiB/agent':>10s} {'global%':>8s} {'max agents':>10s}")
    for r in rows:
        print(f"{r['arm']:14s} {r['context']:6d} {r['kv_gib_per_seq']:10.3f} {100 * r['global_share']:7.1f}% "
              f"{r['max_concurrent']:10.1f}")


def cmd_behaviour(a):
    import pandas as pd

    from kvloop.trajectories import behaviour, group, match_tasks, read_jsonl

    trajs = group(read_jsonl(a.log))
    if a.tasks:
        match_tasks(trajs, [json.loads(x) for x in Path(a.tasks).read_text().splitlines() if x.strip()])
    df = pd.DataFrame([behaviour(t) for t in trajs])
    if a.out:
        df.to_csv(a.out, index=False)
    print(df.describe().T.to_string())


def cmd_replay(a):
    import pandas as pd
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from kvloop.arms import load_arms
    from kvloop.replay import replay_trajectory
    from kvloop.trajectories import group, read_jsonl

    arms = [x for x in load_arms(a.arms) if not a.only or x.name in a.only.split(",")]
    tok = AutoTokenizer.from_pretrained(a.model)
    kw = dict(dtype=torch.bfloat16, device_map=a.device_map, attn_implementation=a.attn)
    try:
        model = AutoModelForCausalLM.from_pretrained(a.model, **kw)
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(a.model, **kw)
    model.eval()
    trajs = group(read_jsonl(a.log))[: a.max_traj or None]
    out = Path(a.out)
    rows = []
    for i, tr in enumerate(trajs):
        rows += replay_trajectory(model, tok, tr, arms, chunk=a.chunk, max_len=a.max_len)
        pd.DataFrame(rows).to_csv(out, index=False)  # checkpoint after every trajectory
        print(f"[{i + 1}/{len(trajs)}] {tr.key} turns={len(tr.turns)} rows={len(rows)}", flush=True)


def cmd_analyze(a):
    import pandas as pd

    from kvloop import analysis as an
    from kvloop.arms import load_arms

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    arms = load_arms(a.arms)
    an.fig_kv_memory(_layout(a.config_json), arms, [1024, 4096, 8192, 16384, 24576, 32768, 65536],
                     out / "fig_kv_memory.png")
    closed_s = None
    if a.runs:
        tasks = [json.loads(x) for x in Path(a.tasks).read_text().splitlines() if x.strip()] if a.tasks else None
        df = an.load_closed(a.runs, tasks)
        df.to_csv(out / "closed_trajectories.csv", index=False)
        if len(df):
            closed_s = an.closed_summary(df)
            closed_s.to_csv(out / "closed_summary.csv", index=False)
            print(closed_s.to_string(index=False))
    if a.replay:
        rp = pd.read_csv(a.replay)
        open_s = an.open_summary(rp)
        open_s.to_csv(out / "open_summary.csv", index=False)
        an.drift_by_context(rp).to_csv(out / "drift_by_context.csv", index=False)
        print(open_s.to_string(index=False))
        if closed_s is not None:
            mean_turns = float(df.turns.mean()) if len(df) else 0.0
            an.fig_open_vs_closed(open_s, closed_s, mean_turns, out / "fig_open_vs_closed.png")
    print(f"wrote {out}")


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "proxy":
        from kvloop.proxy import main as pmain

        return pmain(argv[1:])
    p = argparse.ArgumentParser(prog="kvloop", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arms", help="arms YAML (default: built-in arms)")
    p.add_argument("--config-json", help="model config.json for the attention layout")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("arms").set_defaults(fn=cmd_arms)
    s = sub.add_parser("serve-args")
    s.add_argument("arm")
    s.set_defaults(fn=cmd_serve_args)
    s = sub.add_parser("kvmath")
    s.add_argument("--pool-gib", type=float)
    s.add_argument("--contexts", type=int, nargs="+", default=[4096, 8192, 16384, 32768])
    s.add_argument("--store-kv-once", action="store_true")
    s.set_defaults(fn=cmd_kvmath)
    s = sub.add_parser("behaviour")
    s.add_argument("log")
    s.add_argument("--tasks")
    s.add_argument("--out")
    s.set_defaults(fn=cmd_behaviour)
    s = sub.add_parser("replay")
    s.add_argument("--model", required=True)
    s.add_argument("--log", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--only", help="comma-separated arm names")
    s.add_argument("--max-traj", type=int, default=0)
    s.add_argument("--max-len", type=int, default=None)
    s.add_argument("--chunk", type=int, default=4096)
    s.add_argument("--device-map", default="auto")
    s.add_argument("--attn", default="sdpa")
    s.set_defaults(fn=cmd_replay)
    s = sub.add_parser("analyze")
    s.add_argument("--runs")
    s.add_argument("--replay")
    s.add_argument("--tasks")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_analyze)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
