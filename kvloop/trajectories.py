"""Turn proxy logs into agent trajectories and behavioural metrics.

A trajectory is the sequence of model calls that share the same opening
(system prompt + first user message). This is harness-agnostic: it only needs
the proxy log, so the same code scores every arm.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# text that suggests the model tried to emit a tool call the parser did not accept
TOOL_MARKERS = (r"<\|?tool_call", r"tool_call>", r"\bcall:\w+\{", r"```tool", r'"name"\s*:\s*"\w+"\s*,\s*"arguments"')
_MARKER_RE = re.compile("|".join(TOOL_MARKERS))
_CTX_RE = re.compile(r"maximum context length|context length|too long|max_model_len", re.I)


def read_jsonl(path: str | Path) -> list[dict]:
    out = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # OpenAI content parts
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return str(content)


def opening_key(messages: list[dict]) -> str:
    sys = next((_text(m.get("content")) for m in messages if m.get("role") == "system"), "")
    usr = next((_text(m.get("content")) for m in messages if m.get("role") == "user"), "")
    return hashlib.sha1((sys + "\x00" + usr).encode()).hexdigest()[:16]


@dataclass
class Turn:
    idx: int
    record: dict

    @property
    def request(self) -> dict:
        return self.record.get("request") or {}

    @property
    def response(self) -> dict:
        return self.record.get("response") or {}

    @property
    def message(self) -> dict:
        ch = (self.response.get("choices") or [{}])[0]
        return ch.get("message") or {}

    @property
    def finish_reason(self):
        return (self.response.get("choices") or [{}])[0].get("finish_reason")

    @property
    def tool_calls(self) -> list[dict]:
        return self.message.get("tool_calls") or []

    @property
    def prompt_token_ids(self):
        return self.response.get("prompt_token_ids")

    @property
    def completion_token_ids(self):
        return (self.response.get("choices") or [{}])[0].get("token_ids")

    @property
    def usage(self) -> dict:
        return self.response.get("usage") or {}


@dataclass
class Trajectory:
    key: str
    turns: list[Turn] = field(default_factory=list)
    task_id: str | None = None

    @property
    def first_user(self) -> str:
        msgs = self.turns[0].request.get("messages") or []
        return next((_text(m.get("content")) for m in msgs if m.get("role") == "user"), "")


def group(records: list[dict]) -> list[Trajectory]:
    trajs: dict[str, Trajectory] = {}
    for rec in sorted(records, key=lambda r: r.get("t_start", 0)):
        msgs = (rec.get("request") or {}).get("messages")
        if not msgs:
            continue
        k = opening_key(msgs)
        t = trajs.setdefault(k, Trajectory(k))
        t.turns.append(Turn(len(t.turns), rec))
    return list(trajs.values())


def canonical_call(tc: dict) -> str:
    fn = tc.get("function") or {}
    args = fn.get("arguments", "")
    try:
        args = json.dumps(json.loads(args), sort_keys=True)
    except (json.JSONDecodeError, TypeError):
        pass
    return f"{fn.get('name')}|{args}"


def behaviour(traj: Trajectory) -> dict:
    """Closed-loop behavioural metrics for one trajectory."""
    calls: list[str] = []
    malformed = 0
    empty_turns = 0
    length_stops = 0
    ctx_errors = 0
    completion_tokens = 0
    cached = 0
    max_prompt = 0
    submitted = False
    for t in traj.turns:
        if t.record.get("status", 200) >= 400:
            raw = json.dumps(t.response)[:2000]
            ctx_errors += bool(_CTX_RE.search(raw))
            continue
        tcs = t.tool_calls
        content = _text(t.message.get("content"))
        if not tcs:
            empty_turns += 1
            if _MARKER_RE.search(content):
                malformed += 1
        for tc in tcs:
            args = (tc.get("function") or {}).get("arguments", "")
            try:
                json.loads(args) if args else {}
            except json.JSONDecodeError:
                malformed += 1
            name = (tc.get("function") or {}).get("name", "")
            submitted |= name == "submit_patch"
            calls.append(canonical_call(tc))
        if t.finish_reason == "length":
            length_stops += 1
        u = t.usage
        completion_tokens += u.get("completion_tokens") or 0
        max_prompt = max(max_prompt, u.get("prompt_tokens") or 0)
        cached += ((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)

    seen, repeated, spin, run = set(), 0, 0, 0
    for i, c in enumerate(calls):
        repeated += c in seen
        seen.add(c)
        run = run + 1 if i and calls[i - 1] == c else 1
        spin = max(spin, run)
    t0 = traj.turns[0].record.get("t_start", 0)
    t1 = traj.turns[-1].record.get("t_end", t0)
    return {
        "traj": traj.key,
        "task_id": traj.task_id,
        "turns": len(traj.turns),
        "tool_calls": len(calls),
        "malformed": malformed,
        "no_call_turns": empty_turns,
        "repeated_calls": repeated,
        "max_spin": spin,
        "length_stops": length_stops,
        "context_errors": ctx_errors,
        "completion_tokens": completion_tokens,
        "max_prompt_tokens": max_prompt,
        "cached_prompt_tokens": cached,
        "submitted": submitted,
        "wall_s": t1 - t0,
    }


def _words(s: str) -> set[str]:
    return set(re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", s.lower()))


def match_tasks(trajs: list[Trajectory], tasks: list[dict], id_key="instance_id",
                text_key="problem_statement", min_jaccard=0.3) -> None:
    """Attach task ids to trajectories by matching the first user message to task text."""
    task_words = [(t[id_key], _words(t.get(text_key, "")), t.get(text_key, "")[:200]) for t in tasks]
    for tr in trajs:
        u = tr.first_user
        best, best_s = None, 0.0
        for tid, w, head in task_words:
            if tid and tid in u:
                best, best_s = tid, 1.0
                break
            if head and head.strip() and head.strip()[:120] in u:
                best, best_s = tid, 0.99
                break
            uw = _words(u)
            s = len(uw & w) / max(1, len(uw | w))
            if s > best_s:
                best, best_s = tid, s
        tr.task_id = best if best_s >= min_jaccard else None


def _turn_signature(t: Turn) -> str:
    ids = t.completion_token_ids
    if ids:
        return "ids:" + ",".join(map(str, ids))
    calls = "|".join(canonical_call(tc) for tc in t.tool_calls)
    return "txt:" + _text(t.message.get("content")) + "#" + calls


def divergence(records_a: list[dict], records_b: list[dict]) -> list[dict]:
    """Pair trajectories of two runs by task opening and find the first turn whose output differs.

    first_diff_turn is None when both trajectories are identical turn for turn.
    """
    a = {t.key: t for t in group(records_a)}
    b = {t.key: t for t in group(records_b)}
    rows = []
    for key in sorted(set(a) & set(b)):
        ta, tb = a[key].turns, b[key].turns
        first = None
        for i in range(min(len(ta), len(tb))):
            if _turn_signature(ta[i]) != _turn_signature(tb[i]):
                first = i
                break
        if first is None and len(ta) != len(tb):
            first = min(len(ta), len(tb))
        ctx = None
        if first is not None and first < len(ta):
            ctx = (ta[first].usage or {}).get("prompt_tokens")
        rows.append({"traj": key, "turns_a": len(ta), "turns_b": len(tb), "identical": first is None,
                     "first_diff_turn": first, "prompt_tokens_at_diff": ctx})
    return rows
