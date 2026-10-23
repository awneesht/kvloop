import yaml

from kvloop.closed_loop import DEFAULT_AGENT, prepare_agent, read_cohort, read_eval_config


def test_cohort():
    ids = read_cohort()
    assert len(ids) == 48 and len(set(ids)) == 48
    assert all("_" in i for i in ids)


def test_prepare_agent_pins_sampling(tmp_path):
    dst = prepare_agent(tmp_path / "agent", temperature=0.0, seed=7)
    assert (dst / "agent.yaml").exists() and not (dst / "NOTICE.md").exists()
    for y in dst.glob("configs/*.y*ml"):
        cfg = yaml.safe_load(y.read_text())
        assert cfg["temperature"] == 0.0 and cfg["seed"] == 7
    # source is untouched
    src = yaml.safe_load((DEFAULT_AGENT / "configs" / "sampling.yaml").read_text())
    assert src["temperature"] == 0.2 and "seed" not in src


def test_eval_config():
    ev = read_eval_config(DEFAULT_AGENT)
    assert ev["max_time_minutes"] == 4.0 and ev["max_tool_calls"] == 45
