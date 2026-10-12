# Steady-state latency as the primary live statistic

**Status:** `PROPOSED` (drafted 2026-10-12 for the user's decision; not decided). If accepted, this is
breaking change #4 to the gate statistic and gets a row in `docs/gates/gate-tools.md`.

## The finding

Every live gate's primary statistic is `averageElapsedTime`, the mean over every task of the run. That
mean includes the cluster's own cold start. The live path runs the whole workload "no warmup tasks,
autoscaling from zero" (`src/executesimulation.py:5`, `:116`), with no pre-initialised replica plan
(`:202`), and `averageElapsedTime` is total RTT over all tasks with nothing discarded
(`src/placement/orchestrator.py:431`). The first minutes of each run are therefore scale-out, image
pulls and sandbox colds onto an empty cluster, not placement quality.

- **Size.** On S6's smoke cell (CD, topology 16490, g0; the spent topology, not a result;
  `slate_fix_v1.md:244`): heavy 2.33 s over the whole run, 1.40 s after 360 s, 10.69 s in the first
  decile; moderate 2.98 / 2.69 / 5.57 s. About 40 % of heavy mean latency is the deploy transient
  (SCIENTIST's estimate, `slate_fix_v2.md:5`).
- **Where the oracle's room sits.** 84 % (heavy) / 68 % (moderate) of the label oracle's gain over
  `cd_exactS_ib0` falls in the first 5 % of the run (SCIENTIST, oracle_diag). **Not yet recorded in a
  node**; it needs a home before this ADR can be accepted.
- **Training does not see the opening.** The accel corpus drops states before t = 360 s (MIN_TIME), so
  no accel-seat GNN was trained on the regime that carries ~40 % of the heavy statistic.
- **Surges started cold as well.** The ×k rungs of `peak_load_v1` and `peak_controls_v1` rescale time,
  so every surge runs from t = 0 onto an empty cluster (caveated on both nodes).

The summarizer at c81f0e88 (in reference-physics from 6a5c922c) adds `steady_state.after_360s`,
`steady_state.after_p5_dispatch`, `steady_state.decile_mean` and a per-task `<cell>.tasks.npy`
(`slate_fix_v1.md:242`). Existing keys are bit-identical. Since 2026-10-12 the steady-state read is a
registered secondary line on every gate. `slate_fix_v2` is the first gate registered with it as primary.

## The two candidate cuts

| cut | key | definition | for | against |
|---|---|---|---|---|
| **after 360 s** | `steady_state.after_360s` | mean of (done − dispatched) over tasks dispatched at or after 360 s | equals the accel corpus MIN_TIME, so the gate scores the regime the model was trained on; fixed in seconds, independent of the arm | at heavy (×27.6) 360 s is a smaller share of the run than at moderate, so the two rungs cut different fractions |
| **after the 5th-percentile dispatch** | `steady_state.after_p5_dispatch` | the same, from the 5th-percentile dispatch time of the cell | cuts the same fraction of tasks at every rung | not tied to the training window; the cut moves with the workload, so it is a different second on each rung |

Both are already computed on every cell from 6a5c922c on. `slate_fix_v2` registers `after_360s` as
primary and `after_p5_dispatch` as its robustness family.

## What can and cannot be re-read

**Cannot be re-read.** Every live gate summarised before c81f0e88. The per-task results were in
node-local `/tmp` and were deleted after summarising; the summaries hold whole-run means, latency
percentiles and per-quarter mean queue only. The one proxy is `queue_drift.quarter_mean_queue` over
Q2–Q4, which is queue time, not latency. Partial exception: `long_horizon_s0_v1` reports 4 h block
means, so its later blocks are already a steady-state-like read (one topology).

**Can be re-run.** The accel-seat gates in the top 25 rows of `LINEAGES.md`, with the seat
unchanged (160 × 24, two rungs, windows g0–g3 unless noted). Cells = arms × topologies × windows ×
rungs. A re-run is a reproduction only if each cell reproduces its old `averageElapsedTime` bit-equal
(the `slate_fix_v2` identity rule); for GNN arms that means the old checkpoint, its `.contract.json`,
and the old slate code (`declared_pruning_v1`) pinned in a worktree.

| gate | contrast that the record cites | design | cells per arm (×seed) |
|---|---|---|---|
| `ib0_confirm_v1` | `cd_exactS_ib0` vs `cd_exactS` (the hand bar) | 12 topologies × 4 windows × 2 rungs | 96 |
| `selfsearch_confirm_v1` | GNN self-search vs CD (paper headline) | 24 × 4 × 2 | 192 |
| `accel_nosplit_v1` | GNN no-split vs CD (paper headline) | 24 × 4 × 2 | 192 |
| `slate_bar_v1` | GNN self-search vs restricted `cd_exactS` (matched slate) | 24 × 4 × 2 | 192 |
| `slate_fix_v1` | uncapped retrain vs `cd_exactS_ib0`; MP vs twin | 12 × 2 × 2 | 48 — superseded by `slate_fix_v2` (already steady-state) |
| `cost_to_go_v1` | V re-rank vs `cd_exactS` | 24 × 4 × 2 | 192 |
| `accel_moderate_confirm_v1`, `accel_replica_v1`, `scale_160_v1` | earlier GNN vs CD reads | 24 or 12 topologies × windows × rungs | 96–192 |
| `big_groups_s0_v1`, `link_bound_s0_v1`, `gnn_overcap_v1`, `prune_g8_s0_v1` | side questions (G8, links, over-cap, pruning) | 12 × windows × 2 | 96 or fewer |
| `sparse_obs_s0_v1` | hand bar with degraded observation | 12 × 2 × g0 | 24 — stage 2 onwards runs on a base with c81f0e88 |
| `high_rho_s0_v1`, `plan_rerank_v1` | ρ ladder; live label oracle | 6 topologies | ≤ 24 |

The minimum set for the paper's claims is the first four rows: CD, `cd_exactS`, `cd_exactS_ib0`, the
restricted `cd_exactS`, GNN self-search and GNN no-split (with their MP-OFF twins if the MP contrast is
carried), about 1,150–1,750 cells. `cd_exactS_ib0`-class cells run about 3 ks wall (median) each
(`sparse_obs_s0_v1.md:64`); the GNN arms on the declared slate are faster.

## Options and their cost

- **A. Secondary line only, permanently.** Whole-run stays primary; every gate quotes the steady-state
  number beside it. Cost: none in compute, no history disclosure needed. Every headline keeps weighting
  the deploy transient at ~40 % (heavy), and a reader cannot tell whether an arm difference is placement
  or ramp-up.
- **B. Primary from 2026-10-12 onward, history disclosed.** New gates register `after_360s` (or
  `after_p5_dispatch`) as primary; every number before 2026-10-12 keeps its whole-run statistic with a
  one-line disclosure at its quote sites (the stale-numbers table in `gate-tools.md` gains a row).
  Cost: record edits only. Old and new numbers are not comparable, and the paper's headline contrasts
  (GNN vs CD, slate_bar) stay on the whole-run statistic.
- **C. Re-run the headline contrasts on steady state.** B, plus a re-run of the minimum set above at
  pinned commits, with the bit-equal identity per cell so that whole-run and steady-state come from the
  same cells. Cost: about 1,150–1,750 cells (one 24-shard array per gate, as for `slate_bar_v1`) and
  checkpoints and inputs that must still exist on datalab (not verified here). Gives the paper a
  steady-state line for every claim it makes.

## Recommendation

_(left for the user)_

## Consequences (whichever option)

The registered statistic of a gate is never swapped after its data; a gate signed with whole-run primary
is read on whole-run. Any claim about "the opening" (the GNN ahead early, oracle room early) is a
statement about the deploy transient, not about steady-state placement, and is quoted as such.
