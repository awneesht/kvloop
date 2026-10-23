"""Open-loop (teacher-forced) replay of recorded agent turns under compressed KV.

For each recorded model call we have the exact prompt tokens and the exact
tokens the model produced (from the proxy with `return_token_ids`). We feed
prompt+completion through the model twice -- once with a BF16 cache, once with
a fake-quantised cache -- and compare the next-token distributions at every
completion position:

  d_nll   change in negative log-likelihood of the token actually produced
  kl      KL(base || arm) of the next-token distribution
  flip    the arm's greedy token differs from the baseline's greedy token

Positions are split into segments (reasoning / action / other) so we can ask
whether damage lands on the tokens that become tool calls.

Because each turn's history is fixed to what really happened, errors cannot
compound: this measures per-turn drift. Closed-loop runs measure the compounded
effect; the gap between the two is the paper's central quantity.

Consecutive turns whose prompts extend each other are merged into one forward
pass ("chains"), so a trajectory usually costs a handful of passes, not one per
turn.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from kvloop.trajectories import Trajectory


# ---------------------------------------------------------------- sequences
@dataclass
class TurnSeq:
    turn: int
    prompt: list[int]
    completion: list[int]
    exact: bool = True
    tool_names: list[str] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class Chain:
    ids: list[int]
    spans: list[tuple[int, int, int]]  # (turn, start, end) of completion tokens in ids


def turn_sequences(traj: Trajectory, tokenizer=None) -> list[TurnSeq]:
    """Exact token sequences from the proxy log; falls back to the chat template."""
    out = []
    for t in traj.turns:
        if t.record.get("status", 200) >= 400:
            continue
        names = [(tc.get("function") or {}).get("name", "") for tc in t.tool_calls]
        reasoning = t.message.get("reasoning_content") or t.message.get("reasoning") or ""
        p, c = t.prompt_token_ids, t.completion_token_ids
        if p is not None and c:
            out.append(TurnSeq(t.idx, list(p), list(c), True, names, reasoning))
            continue
        if tokenizer is None:
            continue
        req = t.request
        msgs = req.get("messages") or []
        tools = req.get("tools")
        p_ids = tokenizer.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=True)
        full = tokenizer.apply_chat_template(msgs + [t.message], tools=tools, tokenize=True)
        if hasattr(p_ids, "input_ids"):
            p_ids, full = p_ids["input_ids"], full["input_ids"]
        if full[: len(p_ids)] != p_ids or len(full) <= len(p_ids):
            continue
        out.append(TurnSeq(t.idx, list(p_ids), list(full[len(p_ids):]), False, names, reasoning))
    return out


def build_chains(seqs: list[TurnSeq], max_len: int | None = None) -> list[Chain]:
    chains: list[Chain] = []
    cur: Chain | None = None
    for s in seqs:
        if cur is not None and len(s.prompt) >= len(cur.ids) and s.prompt[: len(cur.ids)] == cur.ids:
            ids = s.prompt + s.completion
            if max_len is None or len(ids) <= max_len:
                start = len(s.prompt)
                cur.ids = ids
                cur.spans.append((s.turn, start, start + len(s.completion)))
                continue
        ids = s.prompt + s.completion
        if max_len is not None and len(ids) > max_len:
            continue
        cur = Chain(ids, [(s.turn, len(s.prompt), len(ids))])
        chains.append(cur)
    return chains


# ---------------------------------------------------------------- segments
def segment_tokens(tokenizer, completion: list[int], tool_names: list[str], reasoning: str) -> np.ndarray:
    """Label each completion token 0=other, 1=reasoning, 2=action (tool call / answer)."""
    pieces = [tokenizer.decode([t]) for t in completion]
    starts = np.cumsum([0] + [len(p) for p in pieces[:-1]])
    text = "".join(pieces)
    labels = np.zeros(len(completion), dtype=np.int8)
    r_end = 0
    if reasoning:
        head = reasoning.strip()[:60]
        tail = reasoning.strip()[-60:]
        i = text.find(head) if head else -1
        j = text.find(tail, max(i, 0)) if tail else -1
        if i >= 0 and j >= 0:
            r_end = j + len(tail)
            labels[(starts >= i) & (starts < r_end)] = 1
    a_start = None
    for name in tool_names:
        if name:
            k = text.find(name, r_end)
            if k >= 0:
                a_start = k if a_start is None else min(a_start, k)
    if a_start is None and not tool_names:
        a_start = r_end if r_end else None
    if a_start is not None:
        # back up to the start of the token containing a_start, then label to the end
        labels[starts + np.array([len(p) for p in pieces]) > a_start] = 2
        labels[(labels == 2) & (starts < r_end)] = 1
    return labels


# ---------------------------------------------------------------- model passes
def _final_norm(model):
    dec = model.get_decoder() if hasattr(model, "get_decoder") else None
    for cand in (dec, getattr(model, "model", None)):
        if cand is not None and hasattr(cand, "norm"):
            return cand.norm
        if cand is not None and hasattr(cand, "language_model") and hasattr(cand.language_model, "norm"):
            return cand.language_model.norm
    norms = [m for n, m in model.named_modules() if n.endswith("norm") and ".layers." not in n]
    if not norms:
        raise RuntimeError("could not locate the final norm module")
    return norms[-1]


def run_pass(model, ids: list[int], cache, positions: np.ndarray, chunk: int = 4096):
    """Chunked forward over `ids` with `cache`; return final hidden states at `positions`."""
    import torch

    norm = _final_norm(model)
    captured = []
    handle = norm.register_forward_hook(lambda m, i, o: captured.append(o[0] if isinstance(o, tuple) else o))
    dev = next(model.parameters()).device
    pos_set = np.asarray(positions)
    out = []
    try:
        with torch.inference_mode():
            for s in range(0, len(ids), chunk):
                e = min(len(ids), s + chunk)
                captured.clear()
                x = torch.tensor([ids[s:e]], device=dev)
                model(input_ids=x, past_key_values=cache, use_cache=True, logits_to_keep=1)
                h = captured[-1][0]  # [chunk, hidden]
                sel = pos_set[(pos_set >= s) & (pos_set < e)] - s
                if len(sel):
                    out.append(h[torch.as_tensor(sel, device=h.device)].to("cpu", torch.float32))
    finally:
        handle.remove()
    import torch as _t
    return _t.cat(out) if out else _t.empty(0)


def compare(model, h_base, h_arm, targets: np.ndarray, chunk: int = 256) -> dict[str, np.ndarray]:
    """Per-position comparison of the next-token distributions."""
    import torch

    head = model.get_output_embeddings()
    cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    cap = getattr(cfg, "final_logit_softcapping", None)
    dev = head.weight.device
    res = {k: [] for k in ("nll_base", "nll_arm", "kl", "top_base", "top_arm")}
    tg = torch.as_tensor(targets)
    with torch.inference_mode():
        for s in range(0, len(tg), chunk):
            sl = slice(s, s + chunk)
            lb = head(h_base[sl].to(dev, head.weight.dtype)).float()
            la = head(h_arm[sl].to(dev, head.weight.dtype)).float()
            if cap:
                lb = torch.tanh(lb / cap) * cap
                la = torch.tanh(la / cap) * cap
            pb, pa = lb.log_softmax(-1), la.log_softmax(-1)
            t = tg[sl].to(dev)
            res["nll_base"].append((-pb.gather(-1, t[:, None])[:, 0]).cpu())
            res["nll_arm"].append((-pa.gather(-1, t[:, None])[:, 0]).cpu())
            res["kl"].append((pb.exp() * (pb - pa)).sum(-1).clamp(min=0).cpu())
            res["top_base"].append(pb.argmax(-1).cpu())
            res["top_arm"].append(pa.argmax(-1).cpu())
    return {k: torch.cat(v).numpy() for k, v in res.items()}


# ---------------------------------------------------------------- driver
SEG_NAMES = {0: "other", 1: "reasoning", 2: "action"}


def _agg(mask, cmp, targets) -> dict:
    if not mask.any():
        return {}
    d = cmp["nll_arm"][mask] - cmp["nll_base"][mask]
    flips = cmp["top_arm"][mask] != cmp["top_base"][mask]
    return {
        "n": int(mask.sum()),
        "d_nll_mean": float(d.mean()),
        "d_nll_max": float(d.max()),
        "kl_mean": float(cmp["kl"][mask].mean()),
        "kl_max": float(cmp["kl"][mask].max()),
        "flip_rate": float(flips.mean()),
        "any_flip": bool(flips.any()),
        "base_teacher_agree": float((cmp["top_base"][mask] == targets[mask]).mean()),
    }


def replay_trajectory(model, tokenizer, traj: Trajectory, arms, model_config=None, chunk: int = 4096,
                      max_len: int | None = None) -> list[dict]:
    """Return one row per (arm, turn, segment) with drift metrics."""
    from kvloop.fakequant_cache import make_fakequant_cache

    cfg = model_config or model.config
    seqs = turn_sequences(traj, tokenizer)
    by_turn = {s.turn: s for s in seqs}
    rows = []
    for chain in build_chains(seqs, max_len):
        # completion token at index t is predicted from hidden state at t-1
        tgt_idx = np.concatenate([np.arange(a, b) for _, a, b in chain.spans])
        positions = tgt_idx - 1
        targets = np.asarray(chain.ids)[tgt_idx]
        labels = np.concatenate([
            segment_tokens(tokenizer, by_turn[t].completion, by_turn[t].tool_names, by_turn[t].reasoning)
            for t, _, _ in chain.spans
        ])
        turn_of = np.concatenate([np.full(b - a, t) for t, a, b in chain.spans])
        h_base = run_pass(model, chain.ids, make_fakequant_cache(cfg, "bf16"), positions, chunk)
        for arm in arms:
            if arm.scheme == "bf16" or arm.layers == "none":
                continue
            h_arm = run_pass(model, chain.ids, make_fakequant_cache(cfg, arm.scheme, arm.layers), positions, chunk)
            cmp = compare(model, h_base, h_arm, targets)
            for t, a, b in chain.spans:
                tm = turn_of == t
                seq = by_turn[t]
                base = {"traj": traj.key, "task_id": traj.task_id, "arm": arm.name, "turn": int(t),
                        "prompt_len": len(seq.prompt), "completion_len": len(seq.completion), "exact": seq.exact}
                for seg in (None, 0, 1, 2):
                    m = tm if seg is None else tm & (labels == seg)
                    agg = _agg(m, cmp, targets)
                    if agg:
                        rows.append({**base, "segment": "all" if seg is None else SEG_NAMES[seg], **agg})
            del h_arm
    return rows


def greedy_divergence(model, prompt: list[int], cfg, arm, max_new_tokens: int = 256) -> dict:
    """Greedy-decode the same prompt with BF16 and with `arm`; report first divergence."""
    import torch

    from kvloop.fakequant_cache import make_fakequant_cache

    dev = next(model.parameters()).device
    x = torch.tensor([prompt], device=dev)
    outs = {}
    for name, scheme, layers in (("base", "bf16", "none"), ("arm", arm.scheme, arm.layers)):
        cache = make_fakequant_cache(cfg, scheme, layers)
        with torch.inference_mode():
            g = model.generate(x, past_key_values=cache, max_new_tokens=max_new_tokens, do_sample=False)
        outs[name] = g[0, len(prompt):].tolist()
    a, b = outs["base"], outs["arm"]
    first = next((i for i, (u, v) in enumerate(zip(a, b)) if u != v), None)
    if first is None and len(a) != len(b):
        first = min(len(a), len(b))
    return {"arm": arm.name, "first_divergence": first if first is not None else math.inf,
            "identical": first is None, "len_base": len(a), "len_arm": len(b)}
