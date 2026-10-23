import json

from conftest import make_record, tc

from kvloop.results import load_results
from kvloop.stats import mcnemar_exact, wilson
from kvloop.trajectories import behaviour, group, match_tasks


def _msgs(issue, extra=()):
    return [{"role": "system", "content": "You fix bugs."}, {"role": "user", "content": issue}, *extra]


def test_grouping_and_behaviour():
    a = "Issue A: fastapi crashes on empty body"
    b = "Issue B: rich table wraps wrong"
    recs = [
        make_record(_msgs(a), tool_calls=[tc("read_file", {"path": "x.py"})], t=0),
        make_record(_msgs(b), tool_calls=[tc("run_command", {"cmd": "pytest"})], t=0.5),
        make_record(_msgs(a, [{"role": "tool", "content": "..."}]), tool_calls=[tc("read_file", {"path": "x.py"})], t=1),
        make_record(_msgs(a, [{"role": "tool", "content": "..."}] * 2), tool_calls=[tc("read_file", {"path": "x.py"})], t=2),
        make_record(_msgs(a, [{"role": "tool", "content": "..."}] * 3),
                    content='<tool_call>{"name": "edit_file"', t=3),  # unparsed call
        make_record(_msgs(a, [{"role": "tool", "content": "..."}] * 4),
                    tool_calls=[{"id": "z", "type": "function", "function": {"name": "edit_file", "arguments": "{bad"}}], t=4),
        make_record(_msgs(a, [{"role": "tool", "content": "..."}] * 5), tool_calls=[tc("submit_patch", {})], t=5,
                    finish="length"),
        {**make_record(_msgs(a), t=6, status=400), "response": {"error": {"message": "maximum context length is 32768"}}},
    ]
    trajs = group(recs)
    assert len(trajs) == 2
    ta = next(t for t in trajs if "Issue A" in t.first_user)
    m = behaviour(ta)
    assert m["turns"] == 7
    assert m["tool_calls"] == 5
    assert m["repeated_calls"] == 2 and m["max_spin"] == 3
    assert m["malformed"] == 2
    assert m["submitted"] and m["length_stops"] == 1 and m["context_errors"] == 1


def test_match_tasks():
    recs = [make_record(_msgs("Please fix: fastapi crashes on empty body when json is posted"), t=0),
            make_record(_msgs("rich: table wraps wrong with emoji in cells"), t=1)]
    trajs = group(recs)
    tasks = [{"instance_id": "fastapi-1", "problem_statement": "fastapi crashes on empty body when json is posted"},
             {"instance_id": "rich-7", "problem_statement": "table wraps wrong with emoji in cells"}]
    match_tasks(trajs, tasks)
    ids = {t.task_id for t in trajs}
    assert ids == {"fastapi-1", "rich-7"}


def test_load_results_formats(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps({"results": [{"instance_id": "t1", "resolved": True},
                                                             {"instance_id": "t2", "status": "FAILED"}]}))
    (tmp_path / "b.csv").write_text("task_id,passed\nt3,1\nt4,0\n")
    (tmp_path / "c.jsonl").write_text(json.dumps({"id": "t5", "success": "true"}) + "\n")
    r = load_results(tmp_path)
    assert r == {"t1": True, "t2": False, "t3": True, "t4": False, "t5": True}


def test_stats():
    lo, hi = wilson(50, 100)
    assert lo < 0.5 < hi
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(10, 0) < 0.01
    assert abs(mcnemar_exact(5, 5) - 1.0) < 1e-9


def test_divergence():
    from kvloop.trajectories import divergence

    m = lambda extra=(): _msgs("Issue A: crash", list(extra))  # noqa: E731
    run_a = [make_record(m(), tool_calls=[tc("read_file", {"p": 1})], completion_ids=[1, 2], t=0),
             make_record(m([{"role": "tool", "content": "x"}]), tool_calls=[tc("edit_file", {})], completion_ids=[3], t=1)]
    run_b_same = [dict(r) for r in run_a]
    run_b_diff = [run_a[0], make_record(m([{"role": "tool", "content": "x"}]), tool_calls=[tc("run_command", {})],
                                        completion_ids=[4], t=1)]
    assert divergence(run_a, run_b_same)[0]["identical"]
    d = divergence(run_a, run_b_diff)[0]
    assert not d["identical"] and d["first_diff_turn"] == 1
