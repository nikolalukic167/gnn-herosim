# peak_controls_v1 — is peak_load_v1's win over CD the objective, or the model class?

**Status:** `REGISTERED` (2026-09-30). Reads fixed below before any run.

**Question.** [`peak_load_v1`](peak_load_v1.md) found the self-refined GNN (`xs1load_selfref`) beats CD at ×2 / ×3 / ×5
on the grounded windows. Two confounds stand between that and "a learned graph scorer beats hand search":

- **C1, objective.** The GNN imitates labels that include a queueing externality,
  Σ_p (λ_p/2)[(B_p + A_p)² − B_p²] (`scripts_cosim/drift_label.py`), and no rule's score has it. **Arms:** `cdext` =
  CD with that term added per candidate at the label's rate (0.46 /s), and `cdextr` = the same at the rung's offered
  rate (0.46 × k). `HEROSIM_PG_EXT_RATE`, `src/policy/peer_greedy_network/scheduler.py`. Added work `a` is the rule's
  own service estimate (exec + I/O + exchange); λ_p is `drift_label`'s split: the rate times the batch's type share,
  spread evenly over the replicas the pass would consider for that type.
- **C2, model class.** No pointwise scorer was served on these cells. **Arm:** `xs1mpoff_selfref`, the MP-OFF twin
  of `xs1load` (`experiments/peak_controls_v1_xs1mpoff.yaml`): the same cache, split, labels, `partial_state_v4`
  block, full-context CE and exchange-in-seconds columns, every message-passing layer skipped, 4 seeds, served with
  the same 3 self-refine passes. Corpus: `backlog_corpus_v1` (5,032 graphs; 2,886 / 1,186 / 960).

**Design.**
- Cells: `peak_load_v1` Amendment 1's, unchanged (12 topologies × g0–g3 × ×2 / ×3 / ×5). Every other arm is read from
  its `groundedladder` summaries.
- Phase `peakctl`: `cdext` + `cdextr`, 288 runs. It also includes a witness: CD rerun on 9119 / 9420 `g0x30` must
  equal `groundedladder` to the digit.
- Phase `peakmlp`: `xs1mpoff_selfref` seeds 1–4, 576 runs.
- Training: `scripts_cosim/datalab/peak_controls_v1_train.sbatch`.
- Reader: `scripts_cosim/peak_controls_v1_read.py`, which uses `peak_load_v1`'s statistic. The paired % is taken per
  topology, as the median over (window, seed); the test is an exact Wilcoxon over topologies, and learned-vs-learned
  arms are paired on the seed.

**Reads, per rung** (labels as `peak_load_v1`, plus Holm over the three rungs within each family):
1. **R1** — `xs1load_selfref` vs `cdext`. A `CONFIRMED` result means the win over search survives search optimising
   the label's objective.
2. **R2** — `xs1load_selfref` vs `cdextr`: the same comparison at the offered rate.
3. **R3** — `xs1load_selfref` vs `xs1mpoff_selfref`. `CONFIRMED` makes it a message-passing effect. `DIRECTION-ONLY`
   is recorded as a direction. `NOT-SEPARATED` makes it a learned-scorer effect, not a graph one.
4. **Secondary** — `xs1mpoff_selfref` vs CD, `cdext` vs CD, and `cdextr` vs CD.

**Parents:** [`peak_load_v1`](peak_load_v1.md), [`exchange_seconds_v1`](exchange_seconds_v1.md),
[`backlog_corpus_v1`](backlog_corpus_v1.md), [`drainable_objective_v1`](drainable_objective_v1.md) (the externality
term).

## Record

### 2026-09-30 — registered

Code, config, sbatch and reader were committed before any run. The reader was checked on synthetic summaries (the
witness path, Holm, and seed pairing for the twin). The trainer determinism test passes.
