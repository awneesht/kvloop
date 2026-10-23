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

MODEL = "/kaggle/input/gemma-4-31b-it-qat-w4a16-ct"            # EDIT: model dir in the competition data
HARNESS_DIR = "/kaggle/input/gemma-4-developer-agent"          # EDIT: competition data root
SUBMISSION = "/kaggle/input/gemma-4-developer-agent/sample_submission"  # EDIT: agent config to evaluate
KVLOOP_SRC = "/kaggle/input/kvloop"                            # kvloop uploaded as a Kaggle dataset
WORK = Path("/kaggle/working")
RUNS = WORK / "runs"
N_TASKS = 40                                                   # fixed subset, same for every arm
os.environ.update(MODEL=MODEL, RUNS_DIR=str(RUNS), CONFIG_JSON=f"{MODEL}/config.json")


def sh(cmd, check=True):
    print("$", cmd)
    return subprocess.run(cmd, shell=True, check=check, text=True)


# %% [markdown]
# ## 0. Install kvloop and read the harness docs

# %%
sh(f"pip install -q --no-deps -e {KVLOOP_SRC} || pip install -q --no-deps git+https://github.com/awneesht/kvloop")
sh(f"ls {HARNESS_DIR}; find {HARNESS_DIR} -maxdepth 2 -name 'HARNESS_README*' -exec sed -n '1,200p' {{}} \\;", check=False)
sh("swegemma --help; swegemma eval --help", check=False)
sh("python -c 'import vllm, transformers; print(vllm.__version__, transformers.__version__)'", check=False)

# %% [markdown]
# ## 1. Architecture and KV memory model (from the real config.json)

# %%
sh("kvloop --config-json $CONFIG_JSON arms")
sh("kvloop --config-json $CONFIG_JSON kvmath --contexts 4096 8192 16384 32768")

# %% [markdown]
# ## 2. Probe arms (about 10 min per arm)
# Retires the main risks: FP8 KV and layer skipping on L4 + Gemma 4 head_dim 512,
# the real KV pool size, and whether `return_token_ids` works.

# %%
sh(f"bash {KVLOOP_SRC}/scripts/probe_arms.sh bf16 fp8_all fp8_global fp8_sliding", check=False)
for f in sorted((RUNS / "probe").glob("*.json")):
    if not f.name.endswith(".resp.json"):
        print(json.loads(f.read_text()))

# %% [markdown]
# ## 3. Closed-loop runs
# Same task subset, same agent config, only the KV arm changes. Two runs per arm
# to measure run-to-run noise. `HARNESS_CMD` must point the harness at the proxy
# (http://127.0.0.1:8000/v1) and write results to `$OUT/results`.

# %%
TASKS_FILE = WORK / "tasks_subset.txt"
# EDIT: build the subset from the public task list, e.g. the first N_TASKS ids sorted by repo
# TASKS_FILE.write_text("\n".join(task_ids[:N_TASKS]))
os.environ["HARNESS_CMD"] = (
    f'swegemma eval --submission {SUBMISSION} --tasks-file {TASKS_FILE} '  # EDIT: real flags
    f'--base-url http://127.0.0.1:8000/v1 --out "$OUT/results"'
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
