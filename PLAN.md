# Run plan to the Nov 12 deadline

Deadline: **Thu 2026-11-12 23:59 UTC (5:59 PM Chicago)**. Target submit: **Wed Nov 11**.

## Compute budget (estimate)
The scorer allows ~6 min per task on 4x L4. A 40-task closed-loop run is therefore roughly 2-4 GPU-hours
of wall time (4 GPUs count as one Kaggle session). The L4x4 quota is reported as 30 hours/week; confirm in the notebook editor.

| stage | runs | est. session hours |
|---|---|---|
| probe 4 arms x {4x L4, 1x L4} | 2 | 2-3 |
| closed loop bf16, fp8_all, fp8_global x 2 runs x 40 tasks | 6 | 15-24 |
| replay 20 trajectories x 8 arms | 1-2 | 4-6 |
| optional fp8_sliding x 1 run | 1 | 3-4 |

That is about two weeks of quota. If it runs short, cut to 30 tasks or one run for fp8_all, never fewer
than two runs for bf16 and fp8_global (they carry H2 and H4).

## Schedule and decision gates

**Fri Oct 23 - Sun Oct 26: set-up and feasibility**
- Join the Paper Track and the main competition; finish Kaggle identity verification.
- Push this repo to GitHub; upload it as a Kaggle dataset (`kvloop`).
- Commit `paper/hypotheses.md` (pre-registration) before any GPU result.
- Read HARNESS_README.md; fix `scripts/scorer_flags.sh` and `HARNESS_CMD` in the notebook.
- Run `probe_arms.sh`.
- **Gate G1 (Oct 26):** do fp8_all and fp8_global start on L4 with Gemma 4? At TP=1, which arms fit a ~28K prompt?
  - The TP=1 result decides where closed-loop runs happen: if BF16 cannot hold a long context on one L4 but
    FP8 can, the headline closed-loop comparison moves to TP=1 (fp8_all vs fp8_global), with 4x L4 BF16 as the reference.
  - Yes: continue.
  - No (e.g. no FP8 kernel for head_dim 512 on Ada): closed loop becomes bf16 only, plus FP8 on another
    GPU if available; the paper leans on open loop plus the memory model. Still a complete paper.
  - `return_token_ids` fails: set `INJECT_TOKEN_IDS=0`; replay falls back to the chat template.

**Mon Oct 27 - Sun Nov 2: closed loop**
- bf16 r1 first (it feeds replay), then fp8_global r1, fp8_all r1, then the r2 runs.
- Check `kvloop behaviour` after each run for harness problems (zero tool calls, all context errors).
- **Gate G2 (Nov 2):** at least bf16 x2 and fp8_global x2 done.

**Sat Nov 1 - Wed Nov 5: open loop**
- `kvloop replay` on 20 bf16 trajectories, all arms. Start with `--max-traj 2` to time it.

**Thu Nov 5 - Sun Nov 8: analysis**
- `kvloop analyze`; write up each hypothesis as supported, not supported or inconclusive.
- Optional fp8_sliding run if quota allows.

**Mon Nov 9 - Wed Nov 11: write and submit**
- Fill `paper/outline.md` to ≤ 3,000 words, figures, public notebook, repo link.
- Submit the Writeup Nov 11 (drafts are not judged).
