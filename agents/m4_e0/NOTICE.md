# Third-party agent configuration

This directory is an unmodified copy of `submission/` from
https://github.com/M4-on-Github/Gemma4_Dev_Agent_Kaggle (commit 09602b65a080db1dc698b79b3e007b1539c7cc88,
2026-10-22), licensed under the Apache License 2.0 (see `LICENSE`). It is that project's "E0/EXP-005"
agent: one prompt-only LlmAgent, 4.0 min / 45 tool calls / 70 turns per task, thinking budget 256.

KVLoop uses it unchanged as a fixed, public, reproducible agent so that only the KV-cache arm varies.
`kvloop.closed_loop.prepare_agent` writes a copy with deterministic sampling (temperature 0, fixed seed)
into the run directory; the files here are never edited.

`cohorts/dev-48.txt` (task ids only) comes from the same repository and license: 48 public tasks whose gold
patch passes under the subprocess sandbox, sampled with seed 20261013.
