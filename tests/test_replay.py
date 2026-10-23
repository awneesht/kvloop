import numpy as np
import pytest
from conftest import make_record, tc

from kvloop.arms import get_arm
from kvloop.replay import TurnSeq, build_chains, segment_tokens, turn_sequences
from kvloop.trajectories import group


def test_build_chains_merges_extending_prompts():
    s1 = TurnSeq(0, [1, 2, 3], [4, 5])
    s2 = TurnSeq(1, [1, 2, 3, 4, 5, 6, 7], [8])  # extends s1 exactly
    s3 = TurnSeq(2, [1, 2, 9], [10])  # history rewritten -> new chain
    chains = build_chains([s1, s2, s3])
    assert len(chains) == 2
    assert chains[0].ids == [1, 2, 3, 4, 5, 6, 7, 8]
    assert chains[0].spans == [(0, 3, 5), (1, 7, 8)]
    assert chains[1].spans == [(2, 3, 4)]


def test_build_chains_respects_max_len():
    chains = build_chains([TurnSeq(0, [1] * 10, [2] * 5)], max_len=12)
    assert chains == []


def test_segment_tokens(char_tok):
    # completion text: reasoning "abc" + "dd" + tool name "edit" (e=4,d=3,i=8,t=19)
    ids = [0, 1, 2, 3, 3, 4, 3, 8, 19, 0]
    text = char_tok.decode(ids)
    assert text == "abcddedita"
    lab = segment_tokens(char_tok, ids, ["edit"], "abc")
    assert lab.tolist() == [1, 1, 1, 0, 0, 2, 2, 2, 2, 2]


def test_turn_sequences_from_token_ids():
    recs = [make_record([{"role": "user", "content": "x"}], tool_calls=[tc("read_file", {})],
                        prompt_ids=[1, 2], completion_ids=[3], t=0)]
    seqs = turn_sequences(group(recs)[0])
    assert seqs[0].prompt == [1, 2] and seqs[0].completion == [3] and seqs[0].exact
    assert seqs[0].tool_names == ["read_file"]


def test_replay_end_to_end(tiny_gemma4, char_tok):
    torch = pytest.importorskip("torch")
    from kvloop.replay import replay_trajectory

    rng = np.random.default_rng(0)
    p1 = rng.integers(0, 97, 60).tolist()
    c1 = rng.integers(0, 97, 12).tolist()
    p2 = p1 + c1 + rng.integers(0, 97, 20).tolist()
    c2 = rng.integers(0, 97, 10).tolist()
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    recs = [make_record(msgs, tool_calls=[tc("read_file", {})], prompt_ids=p1, completion_ids=c1, t=0),
            make_record(msgs + [{"role": "tool", "content": "o"}], tool_calls=[tc("edit_file", {})],
                        prompt_ids=p2, completion_ids=c2, t=1)]
    traj = group(recs)[0]
    arms = [get_arm("bf16"), get_arm("int8_all"), get_arm("kivi2_all"), get_arm("int4_global")]
    torch.manual_seed(0)
    rows = replay_trajectory(tiny_gemma4, char_tok, traj, arms, chunk=32)
    arm_names = {r["arm"] for r in rows}
    assert arm_names == {"int8_all", "kivi2_all", "int4_global"}  # baseline produces no rows
    alls = {(r["arm"], r["turn"]): r for r in rows if r["segment"] == "all"}
    assert alls[("int8_all", 0)]["n"] == 12 and alls[("int8_all", 1)]["n"] == 10
    k8 = np.mean([alls[("int8_all", t)]["kl_mean"] for t in (0, 1)])
    k2 = np.mean([alls[("kivi2_all", t)]["kl_mean"] for t in (0, 1)])
    assert 0 <= k8 < k2
