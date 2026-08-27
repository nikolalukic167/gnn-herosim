# p5b_candidate_relative — CLOSED

> **Status:** `CLOSED` &nbsp;·&nbsp; **Index:** [LINEAGES.md](../../LINEAGES.md) &nbsp;·&nbsp; **Record spans:** 2026-08-24 → 2026-08-24

**Outcome.** **INDETERMINATE — and the indeterminacy is the result.** Kills the mechanism sentence "a pointwise scorer collapses because it cannot condition on its peers": one arm has exactly that conditioning, uses it, and stops collapsing. Resolved by `p5b_draw_study` — it was the training draw.

**Entry points:** `src/policy/tabular/reduced_features.py` (`candidate_relative_queue_columns`), `train_mlp_dim22_from_batch.py --candidate-relative-queue`, `datalab/{fc_siv1_mlp_candrel,mlp_candrel_arm_all_gates}.sbatch`, `important/score_p5b_collapse_pairs.py`, `scripts_cosim/test_{candidate_relative_features,mlp_serving_layout}.py`

**Datasets:** no new datasets — derived in-process from `graphs_cache_full_corpus_siv1_dim14{,_tempfix}`

**Related:** [p5b_draw_study](p5b_draw_study.md) · [program_verdict_v1](program_verdict_v1.md)

## Standing (from the index table)

**Closed 2026-08-24 as INDETERMINATE — and the indeterminacy is the result.** Step 1 of `program_verdict_v1`'s sequence, pre-registered before submission (commit `2c5e676`), run clean (jobs `711675`/`711679`, 60/60 COMPLETED). Handing the pointwise MLP the candidate-relative view (`dim25cr`) moved the two cache arms in **opposite** directions: `mlpcandrel` 7/30 → **17/30** collapses, `mlpcandreltf` 7/30 → **2/30** (and negative mean margin vs Knative in 4 of 6 conditions — the first MLP arm to approach the GNN's record). Robust to dropping the registered detector for an RTT criterion. **Kills the mechanism sentence "a pointwise scorer collapses because it cannot condition on its peers"** — one arm has exactly that conditioning, uses it (28.8% ablation), and stops collapsing. Cache and seed are perfectly confounded (both `--random-state 42`). **RESOLVED 2026-08-24 by `p5b_draw_study` (below): it was neither the feature nor the cache — it was the training draw.** Outcome below.

> **Split note.** This node was carved out of a single 4,995-line `LINEAGES.md` on
> 2026-08-27; the section bodies below are byte-for-byte as written. An *above* / *below*
> pointer inside one may refer to a section that now lives in another node — the
> **Related** links say which.

## Record

Newest first; the sections themselves are in chronological order below.

- [p5b_candidate_relative — OUTCOME: **INDETERMINATE**, and the reason is the finding (2026-08-24)](#p5b-candidate-relative-outcome-indeterminate-and-the-reason-is-the-finding-2026-08-24)
- [p5b_candidate_relative — PRE-REGISTRATION (written 2026-08-24, before any gate run)](#p5b-candidate-relative-pre-registration-written-2026-08-24-before-any-gate-run)

---

### p5b_candidate_relative — PRE-REGISTRATION (written 2026-08-24, before any gate run)

**Everything in this subsection was committed before the training or gate jobs were
submitted.** `program_verdict_v1` closed by finding that its own headline positive — GNN
beats Knative 30/30, MLP collapses 14/120, GNN 0/120 — was scored under a win rule written
at scoring time. Repeating that mistake in the control that tests it would make the whole
exercise worthless, so the rule is fixed here first and
`important/score_p5b_collapse_pairs.py` takes no threshold arguments.

**The question.** The 14/120 collapse was read as architectural because retraining on a
corrected cache moved the victim set (7/30 → a different 7/30) without shrinking it. The
untested cheaper explanation: the MLP's 22 features describe one (task, platform) edge in
absolute terms and never say that this platform's queue is the deepest in the task's own
candidate set. Message passing gives the GNN that comparison for free. So hand it to the
MLP directly and re-run the same cells.

**The intervention.** `dim25cr` = dim22 + 3 columns per (task, candidate) edge, over the
same normalized queue column the model already reads (platform index 7):
`x_22 = q - min(q_cand)`, `x_23 = avg_rank(q)/max(1, n-1)`, `x_24 = (q - mean)/std`
(std == 0 → 0). Shift-invariant; rank and z are also scale-invariant; a single-candidate
task gets zeros. One shared definition
(`reduced_features.candidate_relative_queue_columns`) is imported by both the training
extractor and `MLPBatchScheduler.build_feature_matrix` — a second copy of the formula is
how train/serve skew recurs. **The strongest reasonable version of the feature was chosen
deliberately**: this control exists to break our own result, and a weak version that fails
would be uninformative.

**Design.** Paired, 30 `(cell, condition)` pairs per arm; two arms, each against the
baseline trained on the *same* cache, so the pairing is an A/B on the feature set alone:

| arm | cache | baseline |
|---|---|---|
| `mlpcandrel` | `graphs_cache_full_corpus_siv1_dim14` | `mlp` |
| `mlpcandreltf` | `graphs_cache_full_corpus_siv1_dim14_tempfix` | `mlptempfix` |

Cells, traces, hyperparameters (hidden 64, lr 1e-3, seed 42, test-size 0.2, epochs 100,
patience 10), physics and parity waivers are copied verbatim from
`mlp_tempfix_arm_all_gates.sbatch`.

**Collapse detector.** `chosen_queue_vs_min.p95` from the `.decode_stats.json` sidecar —
the detector measured to separate all 120 prior runs with no overlap (collapse
13,485–23,866, healthy 449–1,387; the *median* is normal in both, which is the direct
evidence that collapse is a compounding minority-of-decisions tail). **Collapse iff
p95 ≥ 5,000.** A cell landing in the never-observed band (1,387, 13,485) is reported as
such: the separation is an empirical fact about 120 runs, not a law.

**Primary test.** Exact one-sided McNemar on the paired collapse indicator. `b` = baseline
collapsed / candrel healthy, `c` = baseline healthy / candrel collapsed; under H0,
`b | b+c ~ Binom(b+c, ½)`.
- **REFUTE** — one-sided `p ≤ 0.05` with `b > c`, in **both** arms. Against a 7/30
  baseline that means `b=6, c=0` (p = 0.0156) or `b=7, c≤1` (p = 0.0352).
- **HARDEN** — ≥ 5 of the baseline's collapses still collapse, arm total ≥ 5/30, and
  `p > 0.05`, in **both** arms.
- **INDETERMINATE** — anything else, *including* the two arms disagreeing. Reported as
  indeterminate; not read as either result.

**Threshold sensitivity is part of the rule, not a robustness afterthought.** Each arm's
verdict is recomputed at 2,000 / 5,000 / 10,000; if it is not identical at all three, that
arm is INDETERMINATE regardless of its p-value at 5,000.

**Secondary, reported always, never the basis of the verdict.** Mean margin vs the
same-condition Knative arm and per-cell wins at the measured `< −0.4%` noise floor.

**Validity gates — both must pass before the 60 gate runs are submitted.** A null from a
model that ignored the new columns is evidence about nothing.
1. Held-out test edge accuracy ≥ its own baseline − 0.01 (`mlp` baseline: 0.8842).
2. Zeroing `x_22..x_24` moves ≥ 5% of held-out argmaxes.

Both are computed by the trainer, recorded in the checkpoint `.meta.json`, and asserted by
`fc_siv1_mlp_candrel.sbatch`, which refuses to finish if either fails. **If (2) fails the
feature is inert: fix the representation and retrain — do not report the null.** Measured
locally at 2 epochs on the `dim14` cache before submission: **29.1%**, so the feature is
demonstrably load-bearing.

**What each outcome does to the paper.**
- REFUTE → the reliability separation is feature engineering, not architecture. "The GNN
  is the only learned scheduler that never collapses" must be restated as a statement
  about a feature the baseline lacked, and P5a's pre-registered gate is not worth running
  in its current form.
- HARDEN → the architectural reading stands, now against the strongest hand-engineered
  version of the missing feature rather than against its absence. Proceed to P5a.

**Provenance.** Layout served as `dim25cr` and declared in the sbatch; a conflicting
declaration is a hard load error, verified before submission (undeclared → pins from the
sidecar; `dim22` declared → refuses). Checkpoints carry `inference_feature_layout`,
`candidate_relative`, the column spec, `code_provenance` and `python_env`. Scoring must
pass `--expect-layouts` to `score_live_gate_matrix.py`: the arms differ in layout **by
design** here, and the guard was changed from an equality check to a declaration check
precisely so that an *unintended* mixture still fails loud.

### p5b_candidate_relative — OUTCOME: **INDETERMINATE**, and the reason is the finding (2026-08-24)

**Ran as registered.** Retrains job `711675` (both validity gates passed: accuracy vs own
baseline +0.0000 / −0.0038; CR-ablation argmax change 21.3% / 28.8%). Gate array `711679`,
60/60 COMPLETED, 30 results + 30 decode-stats sidecars per arm, no failures, all runs
`commit=886f5593 dirty=False torch=2.5.1+cu121`. Verdict artifact:
`simulation_data/p5b_verdict.json`.

**Registered verdict: INDETERMINATE — the two paired arms moved in opposite directions.**

| arm | cache | collapses (registered detector) | fixed / broken / both | p (1-sided) | pair verdict |
|---|---|---|---|---|---|
| `mlpcandrel` | `dim14` | 7/30 → **17/30** | 2 / 12 / 5 | 0.9991 | HARDEN |
| `mlpcandreltf` | `dim14_tempfix` | 7/30 → **2/30** | 5 / 0 / 2 | 0.0312 | INDETERMINATE (threshold-unstable: 2,000 → p=0.062) |

**The split is not noise, and it survives dropping the registered detector entirely.**
Re-scored post-hoc on `total_rtt` vs the same-cell Knative arm, at +30/+50/+100%:
`mlpcandrel` goes 7→13, 6→12, 5→11 (p = 0.989 / 0.996 / 1.000); `mlpcandreltf` goes
7→2 at **all three** thresholds (p = 0.0312 each, i.e. *more* stable than under the
registered detector). Registered secondary — mean margin vs same-condition Knative:
`mlpcandrel` blows out to **+436.1%** and **+509.9%** on the two nobackbone blocks, while
`mlpcandreltf` turns **negative in 4 of 6 conditions** (−23.2%, −20.0%, −26.8%, −33.8%)
and wins 26/30 cells — the first MLP arm in this repo to approach the GNN's record.

**What this does and does not license.**
- **"A candidate-relative feature fixes the MLP collapse" is NOT supported** — it fixed
  one arm and made the other roughly twice as bad.
- **"A pointwise scorer collapses *because* it cannot condition on the candidate set" is
  also no longer supported**, and this is the load-bearing correction: `mlpcandreltf` has
  exactly that conditioning, uses it heavily (28.8% of argmaxes move when it is ablated),
  and largely stops collapsing. The mechanism sentence in the 2026-08-23 subsection above
  ("a pointwise scorer cannot condition on where its peers are going") must not be written
  as the explanation of the reliability gap.
- **The GNN's 0/120 record is untouched** by this lineage — no GNN arm was re-run.
- What the split *does* support is the existing architectural reading in its weaker,
  weights-not-data form (`memory/herosim-mlp-collapse-is-occupation-collapse.md`): all
  four MLP checkpoints sit within ±0.004 test edge accuracy of each other while their live
  collapse counts range 2/30 → 17/30. **Supervised accuracy does not constrain live
  reliability at all**, and the same feature added to two caches moves reliability in
  opposite directions. Pointwise reliability here is a property of the draw.

**Blocking confound, unresolved: cache or seed?** Both candrel arms used `--random-state
42`, so "cache" and "training draw" are perfectly confounded — exactly the confound
`memory/herosim-live-quality-is-a-training-draw-lottery.md` was written about. The 7→17
and 7→2 results cannot be attributed to the corpus difference until the same checkpoints
are retrained at ≥3 seeds per cache and re-gated. Cheap: ~4 min per train, 30 gate runs
per checkpoint. **No claim about which cache is "better" may be written before that.**

**Methodological finding — the pre-registered detector is confounded by this specific
intervention.** `chosen_queue_vs_min` p95 agrees with catastrophic RTT on 29/30 (`mlp`),
30/30 (`mlptempfix`) and 30/30 (`mlpcandreltf`) — but only **25/30 for `mlpcandrel`**,
where it fires on five cells whose RTT is fine, including one at **−2.5% (a win)**. The
cause is structural, not statistical: the CR columns make choosing a non-minimum-queue
candidate a *deliberate learned behaviour*, so the detector partly measures the
intervention itself. Its errors are one-directional in all 180 runs (fires-but-healthy,
never quiet-but-collapsed), so it remains sound as a *negative* test. **Any future gate on
a candidate-relative arm must score collapse on RTT, not on this detector** — and the
registered verdict above would be INDETERMINATE either way, so nothing is being rescued
by saying so after the fact.

**Status: CLOSED as INDETERMINATE.** It did its job: it was registered first, it ran
clean, and it falsified the mechanism sentence the paper was going to lean on. The
sequence does **not** proceed to P5a as written — P5a's win condition assumed the MLP arm
was the reliability foil, and one MLP arm now beats Knative in 4 of 6 conditions. Resolve
the seed/cache confound first — see the draw study below.
