# scale_sweep_v1 — does the GNN's margin over its MP-OFF twin grow with tasks per batch or cluster size?

**Status:** `CLOSED` (2026-10-04) — **SCALE-NOT-THE-LEVER**. Registered 2026-10-04 (exploratory, development topologies,
descriptive); fixed before any run.

**Outcome.** Neither more tasks per batch nor a bigger cluster opens a GNN-over-twin margin. For the live-batch-trained pair
(`sb1load` vs `sb1mpoff`) the margin stays −2 to −4 % in every condition (base −2.7 / −3.6 %, m2 −2.3 / −2.0, m4 −2.5 / −4.0,
s12 −4.1 / −2.5 at ×3 / ×5), never at the −5 % bar; the 10-task-trained GNN (`xs1load`) *loses* to its twin by 5–19 % in
every new condition. The search-rule wins hold or grow: `sb1load` beats CD −14 to −47 % and `cdextr` −17 to −34 % everywhere.
No condition qualifies for a confirmation.

**Question.** Within one ~4-task batch the engineered pointwise twin is within ~2 % of the batch optimum
(`live_headroom_v1`), which bounds any MP margin. Offline, the joint-placement headroom grows with batch size (independent
argmin 7.9 s regret at ~4 tasks vs 41.4 s at 10). Does a larger batch or a larger cluster open a live GNN-over-twin margin?

**Conditions** (x3 and x5 of the grounded ladder, the 12 development topologies, windows g0–g3):
- **m2 / m4** — the grounded windows re-minted with `grounded_workload_v1_mint.py --merge-k 2 / 4`: K consecutive Alibaba
  fan-out groups dispatched as one (raw ms lags kept), partners over the merged group, components bridged; task rate
  unchanged (0.4604 /s, g0). g0: mean group 4.09 → 8.18 → 16.37 tasks (k = 1 is byte-identical to the frozen mint).
  Cells take `scheduler.batch_size` 16, the masked_topo maximum, so m4's larger groups split into ≤ 16-task batches.
- **s12** — the unchanged windows on cells with 12 servers instead of 6 (`cluster_scale_v1_cells.mint`, servers the only
  field that moves). Candidate sets exceed the training corpus's support (≤ 5 per task): disclosed confound.
- **base** — the existing ×3 / ×5 runs (`groundedladder`, `peakctl`, `sb1dev` gate dirs).

**Arms.** `xs1load` / `xs1mpoff` (trained on 10-task groups — in distribution for m2/m4) and `sb1load` / `sb1mpoff` (trained
on live-sized ~4-task groups), seeds 1–4; CD and `cdextr`. 5,184 runs (`datalab/scale_sweep_v1_gate.sbatch`).

**Read** (`scripts_cosim/scale_sweep_v1_read.py`): per condition and rung, GNN vs twin for both pairs, each arm vs CD and
`cdextr`; per-topology median over (window, seed), median over 12, exact Wilcoxon, no Holm. Descriptive: a condition
where the GNN-vs-twin margin reaches ≤ −5 % in 10+/12 topologies is a candidate for a registered confirmation on fresh
topologies; nothing here is a claim.

## Record

- 2026-10-04 — **Read** (`scale_sweep_v1/scale_read.json`; gates 826316–18 plus fills 826486–88; 5,178 of 5,184 runs). 74
  runs died with the home quota at its limit (17:59–18:07, `rc=120` or empty markers) and were rerun. Per-topology median over
  (window, seed), median over 12, exact Wilcoxon:

  | condition / rung | `sb1load` vs `sb1mpoff` | `xs1load` vs `xs1mpoff` | `sb1load` vs CD | `sb1load` vs `cdextr` |
  |---|---|---|---|---|
  | base ×3 | −2.71 % (8/12) | −2.18 % (8/12) | −22.15 % | −17.49 % |
  | base ×5 | −3.56 % (10/12) | +13.14 % (1/12) | −23.41 % | −22.81 % |
  | m2 ×3 | −2.33 % (7/12) | +9.62 % (3/12) | −42.81 % | −33.71 % |
  | m2 ×5 | −1.95 % (9/12) | +14.88 % (1/12) | −19.84 % | −18.72 % |
  | m4 ×3 | −2.51 % (8/12) | +5.21 % (1/12) | −15.35 % | −22.61 % |
  | m4 ×5 | −4.02 % (9/12) | +10.10 % (1/12) | −13.99 % | −17.23 % |
  | s12 ×3 | −4.09 % (11/12, p .0015) | +18.67 % (1/12) | −46.96 % | −33.17 % |
  | s12 ×5 | −2.48 % (8/12) | +15.25 % (0/12) | −21.87 % | −19.82 % |

  - **Reading:** the within-batch MP margin does not scale with batch size, even for the arm trained on 10-task groups,
    which is in distribution at m2/m4 and loses to its twin there. A pointwise scorer with the engineered context keeps up
    with the GNN at 2–4x the tasks per decision. s12 is out of training support (more candidates per task) and still shows
    no margin above 5 %.
- 2026-10-04 — Registered.
