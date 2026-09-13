# gnn_draw_study_v1 — CLOSED

> **Status:** `CLOSED` &nbsp;·&nbsp; **Index:** [LINEAGES.md](../../LINEAGES.md) &nbsp;·&nbsp; **Record spans:** 2026-08-25 → 2026-08-25

**Outcome.** **INDETERMINATE**, and **"the GNN never collapses" is FALSIFIED** — 2 of 8 seeded draws collapse. Direction is clear, the test is under-powered.

**Related:** [p5b_draw_study](p5b_draw_study.md) · [trainer_determinism_v1](trainer_determinism_v1.md)

> **Split note.** This node was carved out of a single 4,995-line `LINEAGES.md` on
> 2026-08-27; the section bodies below are byte-for-byte as written. An *above* / *below*
> pointer inside one may refer to a section that now lives in another node — the
> **Related** links say which.

## Record

Newest first; the sections themselves are in chronological order below.

- [gnn_draw_study_v1 — OUTCOME: **INDETERMINATE**, and "the GNN never collapses" is FALSIFIED (2026-08-25)](#gnn-draw-study-v1-outcome-indeterminate-and-the-gnn-never-collapses-is-falsified-2026-08-25)
- [gnn_draw_study_v1 — PRE-REGISTRATION (written 2026-08-25, before any run)](#gnn-draw-study-v1-pre-registration-written-2026-08-25-before-any-run)

---

### gnn_draw_study_v1 — PRE-REGISTRATION (written 2026-08-25, before any run)

**The only open empirical question in the program.** `p5b_draw_study` retired the MLP's
reliability record by showing it is a draw lottery. The GNN's record — both arms 0/30, mean
margins −18.9% and −27.1% — survived only because nobody had varied its seed. It is 2–3
draws against a measured distribution in which **7 of 16** MLP draws are clean, so a clean
GNN draw is roughly a coin flip and p ≈ 0.125. Writing "the GNN never collapses and the MLP
does" on that evidence repeats exactly the error `p5b_draw_study` was run to find.

**Design.** 8 seeded draws of the **deployed** config
(`graphs_cache_full_corpus_siv1_dim14`, seeds 1–8, `NEAR_RTT_TRAIN_SEED`), gated on the same
30 cells (5 topology cells × 6 blocks) with the same Knative arms already in
`gate_stats_summary.json`. 8 × 30 = 240 gate tasks. One config, not four: the claim under
test is about the checkpoint that is deployed, and splitting the budget would halve the
power on the only question asked. Nothing may vary across arms but the seed — the split
stays fixed at `random_state=42` inside the trainer, and `NEAR_RTT_MP_RESIDUAL` /
`GNN_MP_NODE_EDGES` are asserted unset.

**Entry points.** `datalab/gnn_draw_study_{train,gate}.sbatch` →
`important/score_gnn_draw_study.py` (a sibling of `score_p5b_draw_study.py`, not an edit).

**The rule, registered now.** Collapse is `total_rtt ≥ +50%` vs the same-cell Knative arm,
sensitivity at +30/+50/+100, verdict INDETERMINATE if it does not hold at all three —
identical to `p5b_draw_study`, deliberately, because the two distributions are being
compared and a different threshold would compare different questions. A draw is **clean**
when it collapses on zero of its 30 cells.

- **Q1** — within-condition range across the 8 seeds: `LOTTERY` if ≥ 5, `STABLE` if ≤ 2.
- **Q2** — one-sided Fisher exact, GNN clean draws vs the frozen MLP 7/16, α = 0.05. The
  MLP side is **read from the same summary at scoring time**, not typed in.

**Power, computed before the run, and the reason there is an escalation clause.** Against
7/16:

| GNN clean | Fisher p | |
|---|---:|---|
| 8/8 | **0.0087** | significant |
| 7/8 | **0.0507** | misses, by 0.0007 |
| 6/8 | 0.1557 | |

At n=8 **one unclean draw takes the study from decisive to not-established.** That is not a
licence to read 7/8 as a positive afterwards; it is why the response is fixed in advance,
in the same shape as `gate_statistics.py`'s tier ladder and for the same stated reason ("a
run below the power table is VOID, not FAIL"):

> **7/8 clean → `ESCALATE`, verdict VOID**, train seeds 9–12 and re-gate (existing draws
> stay valid, nothing is re-run); at n=12 the rule tolerates 2 unclean draws
> (10/12 → p = 0.0398). **≤ 6/8 clean → `NOT-ESTABLISHED`**, and that is a real negative.

**What each outcome means.** `STABLE` + `GNN-MORE-RELIABLE` is the one result that lets the
reliability claim be written. `LOTTERY` closes the last open empirical question in favour of
the program's terminal negative: the GNN's 0/30 arms were lucky draws and the claim retires
the way the MLP's did. Anything else is reported as a table and summarised as neither.

**Status: PRE-REGISTERED.** Outcome row to follow.

---

### gnn_draw_study_v1 — OUTCOME: **INDETERMINATE**, and "the GNN never collapses" is FALSIFIED (2026-08-25)

240/240 gate tasks, 8 arms × exactly 30 cells, 720 steps COMPLETED, zero failures.
Job 712381 (train, 8×GPU) → 712389 (gate, 240 tasks). Artifacts:
`simulation_data/gnn_draw_study_verdict.json`, `gate_stats_summary.json` (930 results).

**Collapse counts at the primary threshold (`total_rtt ≥ +50%` vs same-cell Knative):**

| s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | range |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | **1** | 0 | 0 | **3** | 0 | 0 | 0 | 3 |

MLP comparison group, same cells, same rule: `0 0 0 0 0 0 0 3 5 7 8 10 11 16 21 26` → 7/16 clean.

**Verdict: Q1 INDETERMINATE, Q2 INDETERMINATE.** Neither survives the registered sensitivity
requirement — the verdict must hold at +30/+50/+100 and it does not:

| threshold | GNN range | Q1 | clean | Fisher p | Q2 |
|---|---:|---|---|---:|---|
| +30% | 10 | LOTTERY | 5/8 vs 6/16 | 0.235 | NOT-ESTABLISHED |
| +50% | 3 | PARTIAL | 6/8 vs 7/16 | 0.156 | NOT-ESTABLISHED |
| +100% | 0 | STABLE | 8/8 vs 7/16 | **0.0087** | GNN-MORE-RELIABLE |

The whole verdict is a function of where the collapse line is drawn, which is exactly what
the sensitivity clause exists to catch. Reporting the +100% row alone would be threshold
shopping; it was pre-registered as one of three, not as the answer.

**What IS established, and it is the point of the study:**

1. **"The GNN never collapses" is dead.** It rested on 2–3 draws reading 0/30. With the seed
   varied, **2 of 8 draws collapse cells** at the registered threshold (s5 worst cell
   +63.2%, s2 +51.5%). The claim the program was going to publish does not survive its own
   first multi-seed test — the same fate as the MLP claim, for the same reason.
2. **The GNN is nonetheless far better behaved than the MLP.** Worst GNN draw is 3/30; the
   MLP's worst is 26/30, and its range across seeds is 26 against the GNN's 3. The
   *direction* is not in doubt at any threshold; only whether it clears α=0.05 is.
3. **The gap is real but under-powered at n=8.** p=0.156 at the primary threshold. The
   registered escalation (7/8 → 12 draws) did **not** fire: 6/8 is below its floor, so this
   is a genuine NOT-ESTABLISHED at +50%, not a VOID awaiting more draws.

**The honest one-liner:** GNN reliability is not a coin flip the way MLP reliability is, but
it is not the invariant the 0/30 record implied either — it is a *tight* distribution with a
tail, and the program may not describe it as "never collapses".

**Do not** re-run this at a different threshold hoping for a cleaner verdict; the three
thresholds and α were fixed in `score_gnn_draw_study.py` before any draw was gated, and the
INDETERMINATE is the registered answer. A future study wanting to resolve Q2 needs more
draws (≥12 at the +50% line), not a different rule.

**Status: CLOSED.** The last open empirical question in `PROGRAM_VERDICT` is answered: the
GNN reliability claim cannot be written as stated. The terminal negative
(`program_verdict_v1`) is untouched — it never rested on any checkpoint.

**Addendum (2026-08-25): a POST-HOC EXPLORATORY rank statistic, checked against a
pre-written decision rule, does not reopen this.** The registered dichotomy (clean/not-clean)
discards the collapse-count magnitude — a 26-cell draw scores identically to a 3-cell one —
so an exact permutation rank-sum test was run on the full collapse-count vectors (same
`collapse_counts()` rule, same cells, same three thresholds) as a check on whether that loss
of power was hiding a real effect. It was pre-decided that this would only be actionable if
the rank statistic cleared α=0.05 at **all three** thresholds, matching the registered
sensitivity requirement:

| threshold | GNN counts (s1–s8) | rank-sum | exact p | binary Fisher p |
|---|---|---:|---:|---:|
| +30% | 0, 8, 0, 0, 10, 2, 0, 0 | 80.0 | 0.1041 | 0.235 |
| +50% | 0, 1, 0, 0, 3, 0, 0, 0 | 71.5 | 0.0274 | 0.156 |
| +100% | all zero | — | 0.0087 | 0.0087 |

The rank statistic clears α at +50% and +100% but **not at +30%** (p=0.1041), so it fails the
same sensitivity requirement the registered dichotomy failed, for the same reason: whichever
statistic is used, the verdict is not stable across the pre-registered threshold ladder. No
new registration, no escalation to n=12. This was the one outstanding number from
`gnn_draw_study_verdict.json`'s Q2 (the local summary predated job 712389 and only carried
clean/not-clean counts, not full vectors); it is now filled from the synced 930-result
`gate_stats_summary.json` and closes the question the same way. CLOSED stands.

---
