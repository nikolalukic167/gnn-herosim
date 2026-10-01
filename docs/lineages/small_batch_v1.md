# small_batch_v1 — does training on live-sized batches change how the learned scorers compare with CD?

**Status:** `REGISTERED` (2026-10-01). The offline check and its bar were fixed before any small-batch sweep was read.

**Question.** Every learned checkpoint in the program was trained on aligned 10-task peer groups and is served on the
grounded workload, whose groups average 3.5–4.1 tasks (`grounded_workload_v1`; `batch_size_strata_read.json` from the
raw GNN's own counters: 3.52 / 3.81 / 3.91 at ×2 / ×3 / ×5). The shift was disclosed as a risk and never measured.

**What is not yet known.** Inside the live range the learned-vs-CD gap does not track batch size (Spearman ρ +0.16 /
−0.21 / −0.11, signs flip), but every live cell has nearly the same batch size, so that read cannot test 10 → 4.

**A. Offline check (existing checkpoints, new small-batch VAL).** Capture the batches the live seat dispatches (whole
peer groups of the grounded workload on the 24 training topologies, windows g0–g3 at ×1), sweep each with the
`backlog_corpus_v1` recipe unchanged (synthetic backlog rungs 0/3/8/20, busy 0.5, cap-free draw), and decode the existing
checkpoints on the VAL cells (9207, 9208, 9209, 9213: the cells `backlog_corpus_v1` held out). Reference rows on the same
graphs: the CD cost model's coordinate-descent plan and the independent argmin (`small_batch_rules_eval.py`), all scored by
the full sweep.
- Statistic: mean val regret per arm (4 seeds averaged first), and the **ratio to the CD-replica regret**, on the 10-task
  VAL and on the small-batch VAL. A ratio that grows when batches shrink is degradation the rule does not share.
- Bar, signed now: **DEGRADES** if the `xs1load` ratio on small batches exceeds its 10-task ratio by more than 25 %;
  **ROBUST** if within ±10 %; otherwise **MIXED**. Same call for `rawgnn`, `xs1mpoff` and `rawmlp`. Descriptive: one
  VAL of 4 cells; it orders the work and does not close it (AGENTS.md rule 6).

**B. Retrain on small batches** (`experiments/small_batch_v1_sb1load.yaml`, `_sb1mpoff.yaml`: the `xs1load` and
`xs1mpoff` recipes verbatim, `epochs: 100`, 4 seeds, new cache and split). Offline: val regret on the small VAL against A.
Live gate (Phase D, the 12 development topologies, grounded ×2/×3/×5, 4 windows, 4 seeds): `sb1load` and `sb1mpoff` vs CD,
`cdextr`, `xs1load` and each other. Development topologies, descriptive; any claim needs fresh topologies.

**Expectations (judgement, not measurement):**
- A `DEGRADES` for `xs1load`: 35 %; `MIXED`: 40 %; `ROBUST`: 25 %.
- `sb1load` beats `xs1load` live at ×2 by more than 5 %: 30 %; beats CD at ×2: 10 %.

**Risks.** The captured states come from the batched rule on the ×1 grounded workload, plus synthetic backlog, so they
differ from live standing backlog as `backlog_corpus_v1` documented; small-batch labels still ignore the future; a
retrain changes the corpus size and seeds as well as the batches, so B is a bundle, not an isolated cause.

**Entry points.** `scripts_cosim/make_warm_corpus.py --variable-group-min`, `datalab/small_batch_v1_{capture,corpus}.sbatch`,
`queue_gap_diag_eval.py --cache --split`, `small_batch_rules_eval.py`, `batch_size_strata_read.py`.

## Record

- 2026-10-01 — **Check A read (offline, descriptive; `small_batch_v1/check_read.json`).** The 10-task-trained checkpoints on
  the small-batch VAL (981 graphs, mean 4.1 tasks) against the 10-task VAL restricted to the same cells (1,146 graphs, 10.0
  tasks); 4 seeds averaged first. Ratio = arm regret ÷ the CD-replica regret on the same graphs, self-refine 3:

  | arm | regret, 10 → small (s) | ratio to CD-replica | change | call |
  |---|---|---|---|---|
  | `xs1load` | 4.00 → 1.28 | 0.35 → 0.41 | +19 % | MIXED |
  | `xs1mpoff` | 3.87 → 1.28 | 0.34 → 0.41 | +23 % | MIXED |
  | `rawgnn` | 18.36 → 5.88 | 1.59 → 1.90 | +19 % | MIXED |
  | `rawmlp` | 30.49 → 7.83 | 2.65 → 2.53 | −4 % | ROBUST |

  Refine 0 gives the same calls (+16 / +17 / +20 / −3 %). CD-replica regret falls 11.5 → 3.1 s and the independent argmin
  41.4 → 7.9 s: small batches are much easier for every rule, and the learned arms track that.
  - **Reading:** the 10-task-trained engineered-context arms beat the CD-replica offline at both sizes (ratio 0.35–0.66), so the
    batch-size shift does not break them; they lose a fifth of their margin over CD, not the margin. The signed bar's
    DEGRADES (> +25 %) was not met for any arm.
  - **What it rules out:** batch size as the explanation of the live loss to CD. The same models beat the CD cost model offline
    at both sizes and still lose to CD live (queue-dominated), which is the offline-to-live gap `cd_gap_v1` already named.
  - **Limits:** 3 validation cells (9213 lost to a hung capture), one VAL, no live read. The CD-replica is CD's cost model,
    not the live CD arm. The `sb1load` / `sb1mpoff` retraining and its live gate (B) are still to read.
- 2026-10-01 — **Data built.** Capture (job 823068): 56 of 96 cell×window runs finished; the rest hung in the
  starved-client spin (logs frozen, no summary) and were cancelled and dropped, as `joint_burst_v1` did. That removes
  validation cell 9213 entirely, so check A reads cells 9207, 9208, 9209. Captured batch sizes (2–10 tasks): 2: 4,877,
  3: 6,002, 4: 2,171, 5: 1,412, 6: 627, 7: 345, 8: 165, 9: 158, 10: 1,043 (16,800 snapshots). Corpus (job 823139, limit 90
  per unit): 4,320 train + 720 held-out datasets, every sweep complete. Cache (job 823218): 35 datasets with `peer_norm 0`
  quarantined (backlog_corpus_v1 Amendment 2), 5,005 graphs, train 3,305 / val 981 / test 719, split sha256 `a580315e…`,
  backlog on 60.8 % of candidates (bc1: 49 %). Check A (job 823282) and the `sb1load` / `sb1mpoff` training (job 823283)
  submitted.
- 2026-10-01 — registered; capture submitted (job 823068, 24 cells × 4 windows).
