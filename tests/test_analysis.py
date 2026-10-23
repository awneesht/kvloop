import json

import pandas as pd
import pytest
from conftest import make_record, tc

pytest.importorskip("matplotlib")


def _write_run(root, arm, run, resolved):
    d = root / arm / run
    (d / "results").mkdir(parents=True)
    recs = []
    for i, (tid, ok) in enumerate(resolved.items()):
        msgs = [{"role": "system", "content": "fix"}, {"role": "user", "content": f"Task {tid}: something broke"}]
        recs.append(make_record(msgs, tool_calls=[tc("read_file", {"p": i})], t=i))
        recs.append(make_record(msgs + [{"role": "tool", "content": "."}], tool_calls=[tc("submit_patch", {})], t=i + 0.5))
    (d / "requests.jsonl").write_text("\n".join(json.dumps(r) for r in recs))
    (d / "results" / "res.json").write_text(json.dumps([{"instance_id": k, "resolved": v} for k, v in resolved.items()]))
    (d / "meta.json").write_text(json.dumps({"wall_s": 3600}))
    (d / "vllm.log").write_text("INFO GPU KV cache size: 123,456 tokens\nMaximum concurrency for 32,768 tokens per request: 3.77x\n")


def test_analyze_end_to_end(tmp_path):
    from kvloop.cli import main

    tasks = [f"t{i}" for i in range(10)]
    runs = tmp_path / "runs"
    _write_run(runs, "bf16", "r1", {t: i < 6 for i, t in enumerate(tasks)})
    _write_run(runs, "fp8_all", "r1", {t: i < 4 for i, t in enumerate(tasks)})
    (tmp_path / "tasks.jsonl").write_text("\n".join(json.dumps({"instance_id": t, "problem_statement": f"Task {t}: something broke"}) for t in tasks))
    replay = pd.DataFrame([
        {"traj": "a", "task_id": "t0", "arm": arm, "turn": t, "prompt_len": 1000 * (t + 1), "completion_len": 50,
         "exact": True, "segment": seg, "n": 10, "d_nll_mean": 0.1 * k, "d_nll_max": 1, "kl_mean": 0.01 * k,
         "kl_max": 0.1, "flip_rate": 0.02 * k, "any_flip": t % 2 == 0, "base_teacher_agree": 0.8}
        for k, arm in enumerate(["fp8_all", "kivi2_all"], start=1) for t in range(4) for seg in ("all", "action")
    ])
    replay.to_csv(tmp_path / "replay.csv", index=False)
    out = tmp_path / "assets"
    main(["analyze", "--runs", str(runs), "--tasks", str(tmp_path / "tasks.jsonl"),
          "--replay", str(tmp_path / "replay.csv"), "--out", str(out)])
    cs = pd.read_csv(out / "closed_summary.csv").set_index("arm")
    assert cs.loc["bf16", "resolve_rate"] == pytest.approx(0.6)
    assert cs.loc["fp8_all", "resolve_rate"] == pytest.approx(0.4)
    assert cs.loc["fp8_all", "delta_vs_base"] == pytest.approx(-0.2)
    assert cs.loc["bf16", "kv_cache_tokens"] == 123456
    assert cs.loc["bf16", "resolved_per_gpu_hour"] == pytest.approx(6 / 4)
    os_ = pd.read_csv(out / "open_summary.csv")
    assert set(os_.arm) == {"fp8_all", "kivi2_all"}
    for f in ("fig_kv_memory.png", "fig_open_vs_closed.png", "drift_by_context.csv"):
        assert (out / f).exists()
