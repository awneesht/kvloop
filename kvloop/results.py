"""Read per-task pass/fail from harness output without knowing its exact format.

Accepts a CSV (task_id,resolved), a JSON/JSONL file, or a directory of them.
Looks for an id field and a pass/fail field under common names. Print a sample
of your harness's result file and extend ID_KEYS / PASS_KEYS if needed.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ID_KEYS = ("instance_id", "task_id", "task", "id", "name")
PASS_KEYS = ("resolved", "passed", "pass", "success", "solved", "status", "result", "outcome")
TRUE_STRINGS = {"true", "pass", "passed", "resolved", "success", "1", "yes", "ok"}


def _truthy(v) -> bool | None:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v > 0
    if isinstance(v, str):
        s = v.strip().lower()
        if s in TRUE_STRINGS:
            return True
        if s in {"false", "fail", "failed", "unresolved", "error", "0", "no", "timeout"}:
            return False
    return None


def _from_record(rec: dict) -> tuple[str, bool] | None:
    tid = next((rec[k] for k in ID_KEYS if k in rec and isinstance(rec[k], (str, int))), None)
    ok = None
    for k in PASS_KEYS:
        if k in rec:
            ok = _truthy(rec[k])
            if ok is not None:
                break
    if tid is None or ok is None:
        return None
    return str(tid), ok


def _walk(obj, out: dict):
    if isinstance(obj, dict):
        r = _from_record(obj)
        if r:
            out[r[0]] = r[1]
        # {"task_a": true, ...} style
        elif obj and all(isinstance(v, (bool,)) for v in obj.values()):
            out.update({str(k): v for k, v in obj.items()})
        for v in obj.values():
            if isinstance(v, (dict, list)):
                _walk(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _walk(v, out)


def load_results(path: str | Path) -> dict[str, bool]:
    path = Path(path)
    out: dict[str, bool] = {}
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.suffix in {".json", ".jsonl", ".csv"})
    for f in files:
        if f.suffix == ".csv":
            with f.open() as fh:
                for row in csv.DictReader(fh):
                    r = _from_record(row)
                    if r:
                        out[r[0]] = r[1]
        elif f.suffix == ".jsonl":
            for line in f.read_text().splitlines():
                if line.strip():
                    try:
                        _walk(json.loads(line), out)
                    except json.JSONDecodeError:
                        pass
        else:
            try:
                _walk(json.loads(f.read_text()), out)
            except json.JSONDecodeError:
                pass
    return out
