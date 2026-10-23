"""Closed-loop runs on the competition harness, one KV arm at a time.

Mirrors the organisers' scorer: the harness's own `VllmServer` (TP 4, gpu_memory_utilization 0.90,
32K context, gemma4 parsers) and `swegemma.evaluate.Evaluator` with the scorer's context compaction
(14,336-token threshold) and code-graph settings. Two things are added:

* the arm's KV flags, passed to vLLM through `VllmConfig.extra_args`;
* the kvloop logging proxy between the agent and vLLM (agent -> :8000 proxy -> :8100 vLLM), which
  records every model call with exact token ids for behaviour metrics and open-loop replay.

Sampling is made deterministic (temperature 0, fixed seed) in a private copy of the agent, so that
two BF16 runs should produce the same trajectories and any divergence in an FP8 arm can be attributed
to the KV cache rather than to sampling noise. Public-LB measurements by other teams show per-task
outcomes churn by more than half between identical runs at temperature 0.2, which would swamp any
KV effect at this sample size.

Only importable inside the Kaggle environment (needs swegemma / adk_submission / vllm), except
`prepare_agent` and `read_cohort`, which are pure Python.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_AGENT = REPO / "agents" / "m4_e0"
DEFAULT_COHORT = REPO / "cohorts" / "dev-48.txt"
MODEL_ALIAS = "gemma-4-31b-it-qat-w4a16-ct"
PROXY_PORT, VLLM_PORT = 8000, 8100


# ------------------------------------------------------------------ pure-python helpers
def read_cohort(path: str | Path = DEFAULT_COHORT) -> list[str]:
    return [ln.strip() for ln in Path(path).read_text().splitlines() if ln.strip() and not ln.startswith("#")]


def prepare_agent(dst: str | Path, src: str | Path = DEFAULT_AGENT, temperature: float = 0.0,
                  seed: int | None = 20261023) -> Path:
    """Copy the agent and pin sampling in every sampling YAML it ships."""
    dst = Path(dst)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("NOTICE.md", "LICENSE", "README.md"))
    for y in list(dst.glob("configs/*.y*ml")):
        text = y.read_text()
        text = re.sub(r"^(temperature:\s*)[0-9.]+", rf"\g<1>{temperature}", text, flags=re.M)
        if seed is not None:
            if re.search(r"^seed:", text, flags=re.M):
                text = re.sub(r"^(seed:\s*).*$", rf"\g<1>{seed}", text, flags=re.M)
            else:
                text = text.rstrip() + f"\nseed: {seed}\n"
        y.write_text(text)
    return dst


def read_eval_config(agent_dir: Path) -> dict:
    import yaml

    for name in ("eval_config.yaml", "eval_config.yml"):
        p = agent_dir / name
        if p.is_file():
            cfg = yaml.safe_load(p.read_text()) or {}
            return cfg.get("evaluation", cfg)
    return {}


# ------------------------------------------------------------------ harness-dependent parts
def _run_sync(fn):
    import asyncio
    import concurrent.futures

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is not None and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(lambda: asyncio.run(fn())).result()
    return asyncio.run(fn())


def _wait_http(url: str, timeout_s: float = 60) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            urllib.request.urlopen(url, timeout=5)
            return True
        except Exception:  # noqa: BLE001
            time.sleep(1)
    return False


def _start_proxy(log_path: Path, inject_token_ids: bool = True) -> subprocess.Popen:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [sys.executable, "-m", "kvloop.proxy", "--upstream", f"http://127.0.0.1:{VLLM_PORT}",
           "--port", str(PROXY_PORT), "--log", str(log_path)]
    if inject_token_ids:
        cmd.append("--inject-token-ids")
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=open(log_path.with_suffix(".proxy.err"), "w"))
    if not _wait_http(f"http://127.0.0.1:{PROXY_PORT}/v1/models", 60):
        proc.kill()
        raise RuntimeError("kvloop proxy did not come up; see " + str(log_path.with_suffix(".proxy.err")))
    return proc


def _record(r, task, t0: float, t1: float) -> dict:
    patch = r.agent_patch or ""
    return {
        "instance_id": task.instance_id, "repo": task.repo, "resolved": bool(r.resolved),
        "status": str(getattr(r, "status", None)), "test_exit_code": getattr(r, "test_exit_code", None),
        "duration_s": round(float(getattr(r, "duration_seconds", 0) or 0), 1), "task_wall_s": round(t1 - t0, 1),
        "tool_calls": getattr(r, "tool_calls", None), "llm_calls": getattr(r, "total_llm_calls", None),
        "total_tokens": getattr(r, "total_tokens", None),
        "error": str(getattr(r, "error", None) or getattr(r, "error_message", None) or ""),
        "patch_chars": len(patch), "patch": patch,
    }


def run_arm(arm_name: str, run_id: str, out_root: str | Path, data_dir: str | Path, model_dir: str | Path,
            task_ids: list[str] | None = None, agent_src: str | Path = DEFAULT_AGENT,
            temperature: float = 0.0, seed: int | None = 20261023, config_json: str | Path | None = None,
            inject_token_ids: bool = True) -> Path:
    """Run one arm on the cohort and write runs/<arm>/<run_id>/{requests.jsonl,results/results.jsonl,meta.json}."""
    import litellm
    import torch
    from adk_submission import ModelRegistry, VllmConfig, VllmServer, discover_adapters
    from google.adk.agents.context_cache_config import ContextCacheConfig
    from google.adk.apps._configs import EventsCompactionConfig
    from google.adk.models.lite_llm import LiteLlm
    from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS, EvalConfig, build_submission_limits
    from swegemma.evaluate import Evaluator
    from swegemma.models import load_tasks

    from kvloop.arch import AttentionLayout
    from kvloop.arms import get_arm

    try:
        from adk_submission.resolvers.generation import install_litellm_reasoning_patch

        install_litellm_reasoning_patch()
    except Exception as exc:  # noqa: BLE001
        print("reasoning patch not installed:", exc)
    litellm.drop_params = True

    data_dir, model_dir = Path(data_dir), Path(model_dir)
    out = Path(out_root) / arm_name / run_id
    (out / "results").mkdir(parents=True, exist_ok=True)
    agent_dir = prepare_agent(out / "agent", agent_src, temperature, seed)
    ev = read_eval_config(agent_dir)
    layout = AttentionLayout.from_config_json(config_json or model_dir / "config.json")
    kv_args = tuple(get_arm(arm_name).vllm_args(layout))
    adapters = discover_adapters(str(agent_dir), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
    adapter_map = getattr(adapters, "adapters", None) or {}

    meta = {"arm": arm_name, "run": run_id, "kv_args": list(kv_args), "temperature": temperature, "seed": seed,
            "eval": {k: ev.get(k) for k in ("max_time_minutes", "max_tool_calls", "max_turns", "timeout_seconds")}}
    gpus = torch.cuda.device_count()
    tp = 4 if gpus >= 4 else (2 if gpus >= 2 else 1)
    t_start = time.time()
    server = VllmServer(VllmConfig(
        model=str(model_dir), port=VLLM_PORT, host="127.0.0.1", tool_call_parser="gemma4",
        reasoning_parser="gemma4", max_model_len=32768,
        dtype="bfloat16" if torch.cuda.is_bf16_supported() else "auto", gpu_memory_utilization=0.90,
        enable_auto_tool_choice=True, enable_lora=bool(adapter_map), max_loras=max(1, len(adapter_map)),
        max_lora_rank=128, scheduler_reserve_full_isl=False, tensor_parallel_size=tp,
        startup_timeout=60 * 20, extra_args=kv_args,
    ), adapter_manifest=adapters)
    server.start()
    meta["server_start_s"] = round(time.time() - t_start, 1)
    target_id = getattr(server, "effective_model_path", str(model_dir))
    proxy = _start_proxy(out / "requests.jsonl", inject_token_ids)
    try:
        models = ModelRegistry()
        for alias in {MODEL_ALIAS, "gemma4"}:
            models.register(alias, LiteLlm(model=f"openai/{target_id}", api_base=f"http://127.0.0.1:{PROXY_PORT}/v1",
                                           api_key="EMPTY"))
        limits, gen = build_submission_limits()
        cfg = EvalConfig(
            tasks_path=data_dir / "tasks.jsonl", snapshots_dir=data_dir / "snapshots", results_dir=out / "results",
            submission_dir=agent_dir, models=models, sandbox="subprocess",
            timeout_seconds=int(ev.get("timeout_seconds", 300)),
            verification_timeout_seconds=int(ev.get("verification_timeout_seconds", 300)),
            max_time_minutes=float(ev.get("max_time_minutes", 60.0)),
            max_tool_calls=int(ev.get("max_tool_calls", 100)),
            max_turns=int(ev["max_turns"]) if ev.get("max_turns") is not None else None,
            limits=limits, generation_constraints=gen, adapter_manifest=adapters,
            context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
            events_compaction_config=EventsCompactionConfig(compaction_interval=15, overlap_size=2,
                                                            token_threshold=14336, event_retention_size=5),
            graph_dir=data_dir / "graphs", embeddings_dir=data_dir / "embeddings",
            wheels_dir=data_dir / "wheels", verbose=False,
        )
        evaluator = Evaluator(cfg)
        tasks = load_tasks(data_dir / "tasks.jsonl")
        if task_ids:
            want = set(task_ids)
            tasks = [t for t in tasks if t.instance_id in want]
        results_path = out / "results" / "results.jsonl"
        done = set()
        if results_path.exists():  # resume a partially finished run
            done = {json.loads(x)["instance_id"] for x in results_path.read_text().splitlines() if x.strip()}
        for i, task in enumerate(tasks, 1):
            if task.instance_id in done:
                continue
            t0 = time.time()
            try:
                r = _run_sync(lambda: evaluator.evaluate_task(task=task, task_index=i, total_tasks=len(tasks)))
                row = _record(r, task, t0, time.time())
            except Exception as exc:  # noqa: BLE001 - one bad task must not end the run
                row = {"instance_id": task.instance_id, "repo": task.repo, "resolved": False,
                       "error": f"{type(exc).__name__}: {exc}", "task_wall_s": round(time.time() - t0, 1)}
            with results_path.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(f"[{arm_name}/{run_id} {i}/{len(tasks)}] {task.instance_id} resolved={row['resolved']} "
                  f"calls={row.get('tool_calls')} {row['task_wall_s']}s", flush=True)
    finally:
        proxy.terminate()
        for name in ("log_path", "log_file", "stdout_path"):
            p = getattr(server, name, None)
            if p and Path(str(p)).exists():
                shutil.copy(str(p), out / "vllm.log")
                break
        server.stop()
    meta["wall_s"] = round(time.time() - t_start, 1)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return out
