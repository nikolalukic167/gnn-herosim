# peak_controls_v1 — is peak_load_v1's win over CD the objective, or the model class?

**Status:** `CLOSED` (2026-09-30) — **`LEARNED-SCORER-WIN / NOT-MP`**. Registered 2026-09-30, and every read below
was fixed before its data.

**Outcome.** The peak-load win is a **learned-scorer** win, not a message-passing one, and only part of it is the
objective.
- **C2, MP-OFF twin.** `xs1mpoff_selfref` beats CD at every rung: −11.0 / −18.5 / −19.8 % (Holm ≤ .005). It also
  beats CD with the externality at the offered rate, `cdextr`: −8.2 / −14.0 / −13.1 % (Holm ≤ .007). Against the
  GNN it gives −1.5 % (direction only), then ties at ×3, then is **faster at ×5, +13.1 % for the GNN (1/12)**.
- **C1, externality in CD.** Adding the label's externality to CD recovers about half of the GNN's margin. The GNN
  still beats `cdextr` at ×2 / ×3 (−9.8 / −13.9 %) and ties it at ×5.
- **Quote this as:** "a learned pointwise scorer over plan-context features beats hand search, including search
  over the same objective". Never quote it as a GNN or MP win. The MLP twin is the stronger arm at saturation.

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

### 2026-09-30 — C2 read (job 820542, phase `peakmlp`); CLOSED

576/576 runs, 0 failed. Training: job 819912, 4 seeds, W&B `peak-controls-v1-xs1mpoff`, sidecars checked. Best val
regret was 6.83–7.08 s, against xs1load's 6.44–6.72 s. Read file: `peak_controls_v1/peak_controls_read.json`.

| Rung | GNN vs MLP | MLP vs CD | MLP vs `cdextr` | MLP vs `cdext` | Mean latency GNN / MLP (s) |
|---|---|---|---|---|---|
| ×2 | −1.5 % (Holm .024, 11/12) direction | −11.0 % ✓ | −8.2 % ✓ | −6.2 % ✓ | 16.8 / 17.4 |
| ×3 | −2.2 % (p .91) tie | −18.5 % ✓ | −14.0 % ✓ | −9.6 % (p .11) | 54.1 / 50.0 |
| ×5 | **+13.1 % (Holm .003, 1/12)** MLP faster | −19.8 % ✓ | −13.1 % ✓ | −14.8 % ✓ | 146.0 / 131.4 |

- MLP vs the other baselines at ×2 / ×3 / ×5: one-pass −17.4 / −27.2 / −21.8 %, Decima −32.8 / −49.9 / −31.6 %,
  Knative −34.4 / −71.9 / −37.4 %, random −72.9 / −91.6 / −71.5 %. Every one survives Holm.
- MLP vs self-predict: −5.5 / −13.0 / −12.6 %, none significant.
- Per seed vs CD: 10 of 12 (seed, rung) cells are `CONFIRMED`; seeds 2 and 4 at ×3 and seed 1 at ×5 are negative but
  not separated.
- Reading: offline, message passing buys ~5 % val regret. Live, it buys nothing at ×2 / ×3 and costs 13 % at ×5.
  The margin over search comes from the learned score on the `partial_state_v4` plan context, not from graph
  propagation.

### 2026-09-30 — C1 read (job 819913, phase `peakctl`)

290 runs: 289 summaries, 1 timeout. The timeout is `cdextr` on 9423 `g2x20`, the same cell where CD timed out in
`groundedladder`. Witness passed: CD matches to the digit on both cells. Every externality run charged the term on
every batch. Read file: `peak_controls_v1/peakctl_read.json`.

**R1 / R2 — GNN vs CD + externality** (median paired %, Wilcoxon p, Holm over rungs, faster/12):

| Rung | vs `cdext` (0.46 /s) | vs `cdextr` (offered rate) | `cdext` vs CD | `cdextr` vs CD |
|---|---|---|---|---|
| ×2 | −7.4 % (p .027, Holm .081), 11/12 | **−9.8 % (Holm .010) ✓**, 11/12 | −3.4 % (12/12) | −2.6 % (11/12) |
| ×3 | −9.5 % (p .11), 10/12 | **−13.9 % (Holm .004) ✓**, 11/12 | −16.0 % (p .043) | −6.3 % (p .52) |
| ×5 | −3.9 % (p .34), 8/12 | −1.3 % (p .38), 8/12 | −7.1 % (p .027) | −5.6 % (p .027) |

- The externality term makes CD faster, and it accounts for roughly half of the GNN's margin over CD. Mean latency
  (s) for GNN / CD / `cdext` / `cdextr`: ×2 16.8 / 24.2 / 18.9 / 19.7; ×3 54.1 / 68.7 / 61.5 / 65.1; ×5 146.1 /
  158.7 / 151.8 / 149.4.
- The GNN's median is ahead of both externality rules at every rung. That lead survives Holm against `cdextr` at ×2
  and ×3 only. Against `cdext` it is uncorrected at ×2 and not separated at ×3 or ×5. At ×5 neither contrast
  separates.
- Reading: the learned score is not only the objective. At ×2 and ×3 the GNN beats search that optimises the
  label's objective at the offered rate. At saturation the objective accounts for all of the margin.

### 2026-09-30 — registered

Code, config, sbatch and reader were committed before any run. The reader was checked on synthetic summaries (the
witness path, Holm, and seed pairing for the twin). The trainer determinism test passes.
