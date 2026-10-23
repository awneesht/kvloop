# Results log

Dated record of what each run showed. Hypotheses are frozen in `hypotheses.md`; this file only records
outcomes and exploratory findings.

## 2026-10-23 - Probe (Kaggle version 1, 4x L4 and 1x L4, vLLM 0.19.1, Gemma 4 31B QAT W4A16)

Raw data: `data/probe_2026-10-23/` (per-arm JSON + full vLLM logs).

**4x L4, scorer settings (TP 4, gpu_memory_utilization 0.90, max_model_len 32768):**

| arm | started | KV pool | max concurrency @32K | x bf16 | decode tok/s | 28K-token prompt |
|---|---|---|---|---|---|---|
| bf16 | yes | 10.61 GiB/GPU | 8.74 | 1.00 | 39.7 | ok, 25.2 s |
| fp8_all | yes | 10.61 GiB/GPU | 17.48 | 2.00 | 39.7 | ok, 23.1 s |
| fp8_global | yes | 10.61 GiB/GPU | 11.69 | 1.34 | 39.6 | ok, 23.8 s |
| fp8_sliding | yes | 10.61 GiB/GPU | 11.54 | 1.32 | 39.8 | ok, 24.6 s |

- All FP8 arms start on Ada (L4) with Gemma 4's head_dim 512 global layers. vLLM forces the TRITON_ATTN
  backend for Gemma 4 (heterogeneous head dims). Per-layer skipping applied exactly as intended (log shows
  `kv_cache_dtype=fp8` only on the intended layers).
- `return_token_ids` works, so open-loop replay can use exact token ids.
- The bf16 concurrency (8.74) matches an independent entrant's measurement exactly.
- FP8 KV does not change short-context decode speed at concurrency 1 (39.7 tok/s in every arm); long-prompt
  latency drops 3-8%.

**Finding (exploratory): serving-engine accounting erases the layer asymmetry.** The architecture-only
memory model predicts that at 32K, FP8 on global layers saves far more than FP8 on sliding layers
(1.62x vs 1.13x concurrency). Measured: 1.34x vs 1.32x, nearly equal. Cause: vLLM reserves
`window - 1 + max_num_batched_tokens` = 1023 + 2048 = 3071 tokens per sliding layer for chunked prefill, so
sliding layers hold 48% of per-request KV at 32K instead of 24%. With that reservation the model predicts
all four measured concurrencies within 1.1% (`kvloop kvmath --prefill-chunk 2048`;
`tests/test_arch_arms_kvmath.py::test_vllm_accounting_matches_probe`). Implication: on vLLM, the layer
choice is a quality decision, not a memory one, at agent context lengths.

**1x L4 (TP 1):** no arm started. The 31B QAT checkpoint needs about 22 GiB for weights alone
(19.5 GiB allocated, then an OOM on a further 2.6 GiB) on a 22 GiB L4, before any KV cache. KV precision
cannot change this.

**H4 status:** its single-GPU premise does not hold for the 31B: neither BF16 nor FP8 can serve on one L4
because the weights do not fit. Report H4 as *not testable as specified (precondition failed)*. The 4x L4
half of H4 (no material change in decode speed at concurrency 1) is *supported*. A single-GPU comparison
with the 12B model is exploratory, not pre-registered.

## 2026-10-23 - Closed-loop design decisions (before any closed-loop run)

- **Agent:** a fixed public agent, `agents/m4_e0` (Apache-2.0, from M4-on-Github/Gemma4_Dev_Agent_Kaggle, its
  "E0/EXP-005" config: 4.0 min / 45 tool calls / 70 turns, thinking budget 256). Reported locally at
  22.9% (11/48) on the task cohort below; public LB 4-7/58. Using a public agent keeps the study about the
  KV cache, not about agent engineering, and lets anyone reproduce it.
- **Tasks:** `cohorts/dev-48.txt`, 48 public tasks whose gold patch passes in the subprocess sandbox.
- **Deterministic decoding:** temperature 0 with a fixed seed in a private copy of the agent. Another
  team measured that at temperature 0.2 more than half of per-task outcomes change between identical
  runs, which would hide any KV effect at n = 48. With greedy decoding, a BF16 replicate run measures the
  remaining (numerical) noise floor, and the turn at which an FP8 trajectory first diverges from BF16
  (`kvloop diverge`) attributes the change to the KV cache. Resolve rate stays the primary metric
  (paired by task, as pre-registered); divergence is a mechanism measure.
- **Harness fidelity:** the run uses the harness's own `VllmServer` and `Evaluator` with the scorer's
  compaction (interval 15, overlap 2, 14,336-token threshold, retention 5) and graph settings, plus the
  arm's KV flags and the logging proxy.
