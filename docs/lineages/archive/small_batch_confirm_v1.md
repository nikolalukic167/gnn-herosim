# small_batch_confirm_v1 — does the small-batch GNN beat CD, CD+ext and its MP-OFF twin on unseen topologies?

**Status:** `CLOSED` (2026-10-04) — **SEARCH-WIN-NOT-MP**. Registered 2026-10-04; arms, topologies, statistic and
verdict were fixed before any run of this gate and before raw_plan_v2's Phase C read (which shares 12 of the 19 topologies).

**Outcome.** On 19 unseen topologies with 8 seeds, the small-batch GNN `sb1load` **beats CD −9.7 / −35.2 / −17.0 % and
`cdextr` −6.8 / −25.4 / −16.3 %** at ×2 / ×3 / ×5 (18–19 of 19 each, all Holm-confirmed) — the program's first confirmed win
over the search rules on unseen topologies. It **ties its MP-OFF twin** (−0.18 / −0.68 / +0.94 %; Holm p 1.0 / 0.054 /
1.0), and the twin beats CD by the same margins. It is a learned-scorer win from the engineered context and live-sized
training batches, **not a message-passing win**; Phase D's −0.8 to −3.6 % twin margin did not replicate.
**Superseded as a claim about the trace (2026-10-07):** it was measured under scattered request origins; with one
origin per peer group CD is faster or tied, zero-shot and retrained ([`small_batch_so_v1`](small_batch_so_v1.md)).

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

- 2026-10-04 — **Gate read: SEARCH-WIN-NOT-MP** (`small_batch_confirm_v1/sbconf_read.json`; jobs 825927 training, 825929–31
  gate, 825932 read). Summaries: CD 225, `cdextr` 225, reactive 227, `xs1load` 907, `sb1load` 1,817, `sb1mpoff` 1,819 of
  228 / 228 / 228 / 912 / 1,824 / 1,824 (failures dropped, no topology short). Rule runs equal raw_plan_v2's on the 12
  shared topologies to the digit (427/427).

  | `sb1load` vs (Holm over 9) | ×2 | ×3 | ×5 |
  |---|---|---|---|
  | CD | −9.71 % (18/19) CONFIRMED | −35.24 % (19/19) CONFIRMED | −16.96 % (18/19) CONFIRMED |
  | `cdextr` | −6.82 % (19/19) CONFIRMED | −25.40 % (18/19) CONFIRMED | −16.26 % (19/19) CONFIRMED |
  | twin `sb1mpoff` | −0.18 % (10/19) NOT-SEPARATED | −0.68 % (15/19, Holm .054) NOT-SEPARATED | +0.94 % (8/19) NOT-SEPARATED |

  Descriptive: `sb1mpoff` vs CD −9.1 / −35.8 / −15.6 %; `sb1load` vs reactive −43 / −84 / −41 %; `sb1load` vs `xs1load`
  (seeds 1–4) −0.2 / −4.8 / −11.6 % (the small-batch retrain helps at ×3 and ×5).
  - **Reading:** a learned scorer beats both search rules at every rung on unseen topologies, at 8 seeds and n = 19. The graph
    contributes nothing measurable over the same features scored per candidate. Within one batch the pointwise twin already
    equals the GNN's decisions (`live_headroom_v1`), which bounds any MP margin in this seat to about 2 % of batch RTT.
- 2026-10-04 — Registered. Seeds 5–8 training and the gate are chained on datalab.
