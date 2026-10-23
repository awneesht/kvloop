# Pre-registered hypotheses

Frozen 2026-10-23, before any GPU results. Commit this file before the first closed-loop run, and report
every hypothesis in the paper whatever the outcome.

**Setting.** Gemma 4 31B (QAT W4A16), scorer harness, 4x L4, vLLM 0.19.1, fixed subset of N public tasks
(same subset and agent config for every arm), ≥ 2 runs per arm.

**Primary metric.** Per-task resolve rate (pass/fail from the harness), paired by task across arms.
**Secondary.** Malformed tool calls, repeated calls and spin, context-limit hits, completion tokens,
resolved bugs per GPU-hour; open-loop ΔNLL, KL and greedy-flip rate on action (tool-call) tokens.

**H1 - proxy gap.** Open-loop per-token metrics rank arms in the same order as closed-loop resolve rate,
but understate the size of the closed-loop loss. Test: Spearman rank correlation across arms; compare the
closed-loop delta with the per-token flip rate.

**H2 - layer asymmetry.** At agent context lengths, global layers hold most of the KV memory (76% at 32K by
the KV model), and compressing only global layers keeps resolve rate within the non-inferiority margin of
BF16 while compressing sliding layers costs more quality per byte saved. Test: fp8_global vs fp8_sliding
vs fp8_all on resolve-rate delta and bytes saved; open-loop int4_global vs int4_sliding.

**H3 - context growth.** Open-loop drift grows with prompt length (late turns), and is larger on action
tokens than on reasoning tokens. Test: drift by context bin; action vs reasoning segment.

**H4 - budget conversion (amended 2026-10-23, before any results; see note).** On one 24 GB L4 (TP=1),
BF16 KV cannot serve a ~28K-token agent context, while FP8 on global layers can, and in closed loop at
that setting fp8_global's resolve rate is within the non-inferiority margin of fp8_all. On the scorer's
4x L4 at concurrency 1, KV compression does not change resolve rate or decode speed materially, because
the KV pool is mostly idle. Non-inferiority margin: 5 percentage points (paired bootstrap 95% CI of the
delta above −5 pts).

**Exploratory (not pre-registered).** INT4 and KIVI in replay; failure-mode taxonomy from behaviour
metrics; prefix-cache hit rates.

## Amendment log

- 2026-10-23, before any GPU result: H4 changed. Public measurements by another entrant show the scorer
  runs one request at a time, the 31B's KV pool on 4x L4 holds about 8.7 full 32K contexts (about 89%
  idle), and harness compaction keeps prompts near or below 14K tokens. The original H4 (more concurrent
  agents on 4x L4) would not bear on the scorer's setting, so H4 now targets the single-GPU setting where
  KV memory binds. H1-H3 unchanged.
