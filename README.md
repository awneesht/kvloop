# KVLoop

**Does KV-cache compression survive the agent loop?**
Open-loop and closed-loop evaluation of KV-cache quantisation for Gemma 4 coding agents on consumer GPUs (4x L4).

Research code for the Google Gemma 4 Developer Agent Paper Track (Kaggle, 2026). It follows
[KVShuttle](https://doi.org/10.5281/zenodo.18764713), which found that per-element metrics like cosine
similarity badly mispredict generation quality under KV compression. KVLoop asks the agentic version of
that question: a coding agent reads its whole KV cache on every one of dozens of turns, and one wrong
token in a tool call can derail the rest of the trajectory.

## Why Gemma 4 makes this interesting

Gemma 4 31B interleaves 50 sliding-window layers (window 1024, 16 KV heads, head_dim 256) with 10 global
layers (4 KV heads, head_dim 512, K = V). The sliding layers' cache is capped at the window, so at agent
context lengths the **global layers hold most of the KV memory**:

| context | KV per agent (BF16) | share in global layers |
|---|---|---|
| 4K | 1.09 GiB | 29% |
| 32K | 3.28 GiB | 76% |

(`kvloop kvmath`; verify against the KV pool vLLM logs.) So compressing only the global layers buys most
of the memory, and vLLM can do exactly that with `--kv-cache-dtype fp8 --kv-cache-dtype-skip-layers
sliding_window`. Whether it is *safe* for an agent is the open question.

## Method

```
          closed loop (real serving, compounding errors)
harness ──► kvloop proxy :8000 ──► vLLM :8100 (arm flags)
              │ logs every call, exact prompt/completion token ids
              ▼
       requests.jsonl ──► behaviour metrics + pass/fail per task
              │
              ▼  open loop (teacher-forced, no compounding)
       replay each BF16 turn under each arm's fake-quantised cache
       ──► ΔNLL, KL, greedy-token flips, split by reasoning / tool-call tokens
```

* **Closed loop** measures what the user feels: resolve rate, malformed or repeated tool calls, spin,
  context-limit hits, and resolved bugs per GPU-hour. Only arms the scorer's vLLM 0.19.1 can serve (FP8,
  with per-layer skipping).
* **Open loop** replays the exact recorded turns with a fake-quantised Hugging Face cache, so every arm,
  including INT8, INT4 and KIVI-2bit, sees the same history. It measures per-turn drift with no
  compounding. The gap between open- and closed-loop damage is the central quantity.

## Arms

| arm | KV format | layers | closed loop |
|---|---|---|---|
| bf16 | model dtype | – | yes |
| fp8_all | FP8 E4M3, scale 1 (vLLM default) | all | yes |
| fp8_global | FP8 E4M3 | global only | yes |
| fp8_sliding | FP8 E4M3 | sliding only | yes |
| int8_all | INT8 per token-head | all | replay |
| int4_all / _global / _sliding | INT4 per token-head (as vLLM ≥ 0.25) | all / global / sliding | replay |
| kivi2_all | KIVI 2-bit, 32-token residual | all | replay |

Edit `configs/arms.yaml` to add arms (`kvloop --arms configs/arms.yaml ...`).

## Layout

```
kvloop/arch.py            attention layout (sliding vs global) from config.json
kvloop/arms.py            arms -> vLLM flags / replay schemes
kvloop/quant.py           FP8 / INT / KIVI fake-quantisers
kvloop/fakequant_cache.py transformers DynamicCache with per-layer-type compression (KIVI residual exact)
kvloop/kvmath.py          KV bytes per agent, max concurrency
kvloop/proxy.py           logging OpenAI-compatible proxy (streaming, return_token_ids)
kvloop/trajectories.py    proxy log -> trajectories -> behaviour metrics
kvloop/replay.py          teacher-forced replay engine, segment labelling, greedy divergence
kvloop/results.py         tolerant reader for harness pass/fail output
kvloop/analysis.py        tables, paired stats, figures
scripts/probe_arms.sh     does each arm start on this GPU? KV pool size? token ids?
scripts/run_arm.sh        one closed-loop run of one arm
notebooks/kaggle_kvloop.* end-to-end Kaggle notebook
paper/                    outline, pre-registered hypotheses
```

## Quickstart

```bash
pip install -e ".[all]"
pytest -q                                   # CPU only; uses a tiny random Gemma 4
kvloop arms                                 # arms and their vLLM flags
kvloop kvmath --pool-gib 48                 # KV per agent and max concurrency
```

On the GPU box (or the Kaggle notebook):

```bash
export MODEL=/path/to/gemma-4-31b-it-qat-w4a16-ct
bash scripts/probe_arms.sh                                    # 1. feasibility
HARNESS_CMD='swegemma eval ... --out "$OUT/results"' \
  bash scripts/run_arm.sh fp8_global r1                        # 2. closed loop
kvloop replay --model $MODEL --log runs/bf16/r1/requests.jsonl --out replay.csv   # 3. open loop
kvloop analyze --runs runs --replay replay.csv --out assets    # 4. tables + figures
```

## Caveats

* The scorer's vLLM flags in `scripts/scorer_flags.sh` come from public summaries of the harness README;
  diff them against the real one.
* Replay fake-quantises K/V but keeps attention arithmetic in BF16. vLLM's FP8 path may also quantise Q
  on some backends, so replay FP8 is a lower bound on FP8 damage there.
* KIVI in a single teacher-forced pass computes group statistics over tokens the query has not reached
  yet. Token-by-token decode through the same cache is exact (see `tests/test_quant_cache.py`).
* Closed-loop decoding uses the harness's sampling settings, so runs are noisy; use ≥ 2 runs per arm and
  the paired statistics in `analysis.py`.

## Citation

If you use this, please also cite KVShuttle:
Tiwari, A. (2026). *KVShuttle: A Benchmark and Decision Framework for KV Cache Transfer Compression in
Disaggregated LLM Serving.* Zenodo. https://doi.org/10.5281/zenodo.18764713

MIT licence.
