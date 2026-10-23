# %% [markdown]
# # KVLoop on Kaggle (4x L4)
#
# Does KV-cache compression survive the agent loop? This notebook runs the three
# stages of the study on the competition hardware:
#
# 1. **Probe** - which KV arms start on 4x L4 with Gemma 4 and the scorer's vLLM,
#    and how big a KV pool each gets.
# 2. **Closed loop** - run the agent on a fixed task subset per arm, logging every
#    model call through the kvloop proxy.
# 3. **Open loop** - replay the BF16 trajectories teacher-forced under every arm
#    (including INT8/INT4/KIVI, which vLLM 0.19.1 cannot serve).
#
# Lines marked `EDIT` depend on the competition data layout. Check them once
# against HARNESS_README.md.

# %%
import json
import os
import subprocess
from pathlib import Path

HARNESS_DIR = "/kaggle/input/gemma-4-developer-agent"          # EDIT: competition data root
MODEL = "/kaggle/input/gemma-4-31b-it-qat-w4a16-ct"            # EDIT: model dir (see `ls` output in stage 0)
SUBMISSION = f"{HARNESS_DIR}/sample_submission"                # EDIT: agent config to evaluate (keep fixed)
KVLOOP_SRC = "/kaggle/input/kvloop"                            # kvloop uploaded as a Kaggle dataset
WORK = Path("/kaggle/working")
RUNS = WORK / "runs"
N_TASKS = 40                                                   # fixed subset, same for every arm
os.environ.update(MODEL=MODEL, RUNS_DIR=str(RUNS), CONFIG_JSON=f"{MODEL}/config.json",
                  OPENAI_API_KEY="EMPTY")  # LiteLLM needs a key; vLLM ignores it


def sh(cmd, check=True):
    print("$", cmd)
    return subprocess.run(cmd, shell=True, check=check, text=True)


# %% [markdown]
# ## 0. Install kvloop and read the harness docs

# %%
sh(f"pip install -q --no-deps {KVLOOP_SRC} || pip install -q --no-deps git+https://github.com/awneesht/kvloop")
sh(f"pip install -q --no-index --find-links {HARNESS_DIR}/wheelhouse swegemma adk-submission", check=False)
sh(f"ls {HARNESS_DIR}; find {HARNESS_DIR} -maxdepth 2 -name 'HARNESS_README*' -exec sed -n '1,200p' {{}} \\;", check=False)
sh("swegemma --help; swegemma eval --help", check=False)
sh("python -c 'import vllm, transformers; print(vllm.__version__, transformers.__version__)'", check=False)

# %% [markdown]
# ## 1. Architecture and KV memory model (from the real config.json)

# %%
sh("kvloop --config-json $CONFIG_JSON arms")
sh("kvloop --config-json $CONFIG_JSON kvmath --contexts 4096 8192 16384 32768")

# %% [markdown]
# ## 2. Probe arms (about 10 min per arm and setting)
# Retires the main risks: FP8 KV and layer skipping on L4 + Gemma 4 head_dim 512, the real
# KV pool, decode speed, whether a ~28K prompt fits, and whether `return_token_ids` works.
# Two settings: the scorer's 4x L4 (TP=4), and one 24 GB GPU (TP=1), the consumer case
# where KV memory actually binds.

# %%
sh(f"bash {KVLOOP_SRC}/scripts/probe_arms.sh bf16 fp8_all fp8_global fp8_sliding", check=False)
sh(f"TP=1 GPU_UTIL=0.92 CUDA_VISIBLE_DEVICES=0 bash {KVLOOP_SRC}/scripts/probe_arms.sh "
   "bf16 fp8_all fp8_global fp8_sliding", check=False)
for f in sorted((RUNS / "probe").glob("*/*.json")):
    print(f.parent.name, json.loads(f.read_text()))

# %% [markdown]
# ## 3. Closed-loop runs
# Same task subset, same agent config, only the KV arm changes. Two runs per arm
# to measure run-to-run noise. `HARNESS_CMD` must point the harness at the proxy
# (http://127.0.0.1:8000/v1) and write results to `$OUT/results`.

# %%
import random

tasks = [json.loads(x) for x in open(f"{HARNESS_DIR}/tasks.jsonl") if x.strip()]
by_repo = {}
for t in tasks:
    by_repo.setdefault(t["instance_id"].rsplit("_", 1)[0], []).append(t["instance_id"])
rng = random.Random(20261023)  # fixed seed: the same subset for every arm and run
subset = []
for repo, ids in sorted(by_repo.items()):  # stratified by repo, proportional
    k = max(1, round(N_TASKS * len(ids) / len(tasks)))
    subset += rng.sample(sorted(ids), min(k, len(ids)))
subset = sorted(subset)[:N_TASKS]
(WORK / "tasks_subset.txt").write_text("\n".join(subset))
print(len(subset), "tasks:", {r: sum(s.startswith(r) for s in subset) for r in by_repo})

# swegemma eval talks to 127.0.0.1:8000 (the kvloop proxy); one task at a time, like the scorer
os.environ["HARNESS_CMD"] = (
    f"swegemma eval --tasks {HARNESS_DIR}/tasks.jsonl --snapshots-dir {HARNESS_DIR}/snapshots "
    f'--submission-dir {SUBMISSION} --results-dir "$OUT/results" --sandbox subprocess '
    f"--task-ids {' '.join(subset)} --concurrency 1 --display quiet"
)
for arm in ["bf16", "fp8_all", "fp8_global"]:
    for run in ["r1", "r2"]:
        sh(f"bash {KVLOOP_SRC}/scripts/run_arm.sh {arm} {run}", check=False)

# %%
sh("for d in runs/*/r1; do echo $d; kvloop behaviour $d/requests.jsonl --out $d/behaviour.csv | head -20; done",
   check=False)

# %% [markdown]
# ## 4. Open-loop replay (vLLM must be stopped: replay loads the model in transformers)
# Replays the BF16 trajectories under every arm. For a cheaper pass use the 12B
# model on its own BF16 trajectories (run stage 3 with MODEL set to the 12B).

# %%
sh("pkill -f '[v]llm serve' || true", check=False)
sh(f"kvloop replay --model {MODEL} --log runs/bf16/r1/requests.jsonl --out {WORK}/replay.csv "
   "--only fp8_all,fp8_global,fp8_sliding,int8_all,int4_all,int4_global,int4_sliding,kivi2_all "
   "--max-traj 20 --chunk 4096")

# %% [markdown]
# ## 5. Tables and figures

# %%
sh(f"kvloop --config-json $CONFIG_JSON analyze --runs {RUNS} --replay {WORK}/replay.csv --out {WORK}/assets")
sh(f"ls -la {WORK}/assets")
