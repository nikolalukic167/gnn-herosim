# small_batch_confirm_v1 — does the small-batch GNN beat CD, CD+ext and its MP-OFF twin on unseen topologies?

**Status:** `REGISTERED` (2026-10-04). Arms, topologies, statistic and verdict were fixed before any run of this gate,
and before raw_plan_v2's Phase C read (which shares 12 of the 19 topologies).

**Question.** `small_batch_v1` Phase D (12 development topologies) put the retrained GNN `sb1load` ahead of CD, `cdextr`
and its retrained MP-OFF twin `sb1mpoff` at ×2 / ×3 / ×5 — the twin by only −0.8 / −2.7 / −3.6 %. Those topologies informed
the proposal. Does it hold on topologies no model or proposal has seen, with enough power to resolve a ~3 % twin margin?

**Topologies.** All 19 that raw_plan_v2's Phase C screen admitted from ids 9473–9568 (reactive queue share ≤ 0.80 and the
batch path finishes in all 4 base-load windows; unknown is not a pass): `small_batch_confirm_v1/selected.json`. Disjoint
from the training corpora and the 12 development topologies. raw_plan_v2 uses the lowest 12 of them for its own arms; no
result on any of the 19 had been read when this was signed.

**Arms** (grounded ×2 / ×3 / ×5, windows g0–g3, served exactly as in `small_batch_v1` Phase D):
- `sb1load` (GNN) and `sb1mpoff` (its MP-OFF twin, the per-candidate MLP on the same engineered features), **8 seeds each**:
  seeds 1–4 as in Phase D, seeds 5–8 trained now with the identical recipe (`small_batch_v1_train.sbatch`, `SEED_OFFSET=4`);
- CD, `cdextr`, reactive (1 run each); `xs1load` seeds 1–4 (descriptive).
- 5,244 runs: `datalab/small_batch_confirm_v1_gate.sbatch` (`BUILD=1`, then parts `sb1load` / `sb1mpoff` / `rules`).

**Statistic.** Per topology, the median paired % over (window, seed); learned arms pair on the seed, rules on their single
run. Median over the 19 topologies, exact two-sided Wilcoxon over topologies, n = 19.

**Primary family** (Holm across all 9): `sb1load` vs {CD, `cdextr`, `sb1mpoff`} × {×2, ×3, ×5}.

**Verdict, signed now** (`scripts_cosim/small_batch_confirm_v1_read.py`):
- **GNN-BEATS-ALL** — the same ≥ 2 rungs at which vs CD and vs `cdextr` are CONFIRMED (median ≤ −5 %, Holm p < 0.05) and
  vs `sb1mpoff` has Holm p < 0.05 with median < 0, and no reference Holm-confirmed faster at any rung.
- **SEARCH-WIN-NOT-MP** — the CD / `cdextr` clause holds at ≥ 2 common rungs, the twin clause does not, nothing faster.
- **NO-WIN** — anything else.
- **Deviation, disclosed:** the twin clause drops the program's −5 % magnitude bar, because Phase D measured a −0.8 to −3.6 %
  margin. A GNN-BEATS-ALL is therefore quoted with its twin magnitude, as a small MP edge, never as a large one.

**Expectations (judgement, not measurement).** CD / `cdextr` clause at ≥ 2 rungs: 80 %. Twin clause at ≥ 2 of those rungs:
35 %. GNN-BEATS-ALL: 30 %.

**Risks.** Seeds 5–8 are trained after Phase D; same code, recipe and split, but a draw lottery is possible (8 seeds
damp it). Training draws are not independent replicates, so n stays 19 topologies. Shared topologies with raw_plan_v2 mean
the CD / reactive runs are rerun here rather than borrowed; they are deterministic and must match to the digit where both
exist (checked at read time).

**Entry points.** `scripts_cosim/fresh_topo_burst_v1_gate.py sbconf`, `datalab/small_batch_confirm_v1_gate.sbatch`,
`small_batch_confirm_v1_read.py`.

## Record

- 2026-10-04 — Registered. Seeds 5–8 training and the gate are chained on datalab.
