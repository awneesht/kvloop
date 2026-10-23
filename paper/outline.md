# Paper outline (Kaggle Writeup, ≤ 3,000 words)

**Working title.** Does KV-Cache Compression Survive the Agent Loop? Layer-Aware KV Quantisation for
Gemma 4 Coding Agents on Consumer GPUs

**Subtitle.** Per-token metrics say FP8 KV is free. We measure what it does to a coding agent across
dozens of turns, and find which of Gemma 4's layers can safely be compressed.

**One-sentence claim (fill after results).** On 4x L4, compressing only Gemma 4's global-layer KV cache
to FP8 [frees X% of KV memory / Y more concurrent agents] while [keeping / losing Z pts of] resolve rate;
open-loop token metrics [under/over]-predict the closed-loop effect by [factor].

## 1. Introduction (300 words)
- Local coding agents are memory-bound: every turn re-reads a growing KV cache (32K cap in the scorer).
- KV quantisation is the standard lever, validated on single-turn reasoning and long-context retrieval
  (vLLM FP8 study), not on multi-turn tool use where errors compound.
- Prior finding (KVShuttle): high per-element fidelity can coexist with broken generation. Agents are the
  worst case: one flipped token in a path or argument derails the trajectory.
- Contributions: (1) open- vs closed-loop protocol that separates per-turn drift from compounding;
  (2) layer-type analysis for Gemma 4's hybrid attention; (3) budget view: bugs fixed per GPU-hour on
  consumer hardware; (4) released tool (proxy, replay engine, fake-quant cache).

## 2. Background and related work (300 words)
- Gemma 4 attention: 5:1 sliding/global, window 1024, global K=V with 4 KV heads, head_dim 512.
- KV quantisation: KIVI (per-channel K, per-token V), INT4 per token-head and FP8 in vLLM, layer skipping.
- Agent-level evaluation: UltraQuant (4-bit KV on SWE-bench Lite agents, datacenter models); quantised
  multi-turn tool calling ("Same Quantization, Different Failures"); shared-cache compression in
  multi-agent pipelines ("Breaking the Role"). Gap: hybrid-attention layer asymmetry, open/closed-loop
  separation, consumer-GPU budget.
- KVShuttle (own prior work).

## 3. Method (600 words)
- 3.1 KV memory model; Figure 1 (KV per agent vs context, by arm; global share).
- 3.2 Arms table.
- 3.3 Closed loop: proxy, task subset, runs per arm, metrics, paired statistics.
- 3.4 Open loop: exact token replay, fake-quant cache (KIVI residual), metrics, segment labelling,
  chain merging. Validity checks: BF16 replay is bit-exact.

## 4. Results (900 words)
- 4.1 Feasibility on L4: which arms start, measured KV pool, concurrency (Table 1).
- 4.2 Closed loop: resolve rate with CIs, deltas vs BF16, McNemar; behaviour metrics (Table 2). H2, H4.
- 4.3 Open loop: drift by arm and segment (Figure 2); drift vs context length (Figure 3). H3.
- 4.4 Open vs closed: Figure 4, open-loop flip rate vs closed-loop delta. H1.
- 4.5 Budget: resolved per GPU-hour, max concurrent agents.

## 5. Discussion and recommendations (400 words)
- Practitioner rule for hybrid-attention agents (which layers, which format, when to calibrate).
- Why token metrics mislead for agents; what to measure instead.
- Generality: any local agent on a hybrid-attention model (Gemma 3/4, gpt-oss, others).

## 6. Limitations (200 words)
- One model family, one harness, N tasks, sampling noise. Fake-quant vs kernel numerics. Teacher-forced
  KIVI caveat. Scorer flags reconstructed from public sources.

## 7. Reproducibility (100 words)
- Repo, Kaggle notebook, pre-registered hypotheses, raw per-task CSVs.

## Figures and tables
| id | content | source |
|---|---|---|
| Fig 1 | KV GiB per agent vs context, all arms | `fig_kv_memory.png` |
| Tab 1 | Probe: started, KV tokens, concurrency per arm | `runs/probe/*.json` |
| Tab 2 | Closed-loop resolve rate, CI, delta, behaviour metrics | `closed_summary.csv` |
| Fig 2 | Open-loop ΔNLL / flip rate by arm x segment | `open_summary.csv` |
| Fig 3 | Drift vs context length | `drift_by_context.csv` |
| Fig 4 | Open-loop flip rate vs closed-loop delta | `fig_open_vs_closed.png` |

## References
- Tiwari, A. (2026). KVShuttle. Zenodo. doi:10.5281/zenodo.18764713
- Kübler, Kurtić et al. (2026). The State of FP8 KV-Cache and Attention Quantization in vLLM. vLLM blog.
- UltraQuant: 4-bit KV Caching for Context-Heavy Agents. arXiv:2606.20474.
- Liu et al. (2024). KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache.
- Same Quantization, Different Failures: Multi-Turn Tool-Calling in Small Language Models. NeurIPS 2026.
- Sathu & Khare. Breaking the Role: compressing a shared KV cache in multi-agent pipelines. NeurIPS 2026.
- Gemma Team (2026). Gemma 4 model card.
