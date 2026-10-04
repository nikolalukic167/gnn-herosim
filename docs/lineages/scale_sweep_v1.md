# scale_sweep_v1 — does the GNN's margin over its MP-OFF twin grow with tasks per batch or cluster size?

**Status:** `REGISTERED` (2026-10-04) — exploratory, development topologies, descriptive only. Fixed before any run.

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

- 2026-10-04 — Registered.
