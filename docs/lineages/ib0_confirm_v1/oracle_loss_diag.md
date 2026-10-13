# Oracle loss diagnostic (post-close, `ib0_confirm_v1`)

Measured by S7 on branch `rp/surrogate-gap` (`scripts_cosim/oracle_loss_diag.py`, head cfdec05d). Reads `oracle_diag/` of `/share/nikola.lukic/ib0_confirm/oracle_8cc890fa` (12 cells,
read-only) and, for the slate question, the brute-force tables of the uncapped accel corpus. No simulation. Outputs: `/share/nikola.lukic/surrogate_gap/oracle_loss_v1/`.
The node's verdict (FAMILY-CLOSES, moderate -2.82 %, heavy -1.96 % against `cd_exactS_ib0`) is not reopened.

**Question.** Where does the hand bar lose to the live oracle, and is the remaining room reachable by a scorer over the declared slate or only outside it?

## Pool 1: inside the slate (the live -2.8 % / -2.0 %)

Both oracle modes (`_pg_oracle_apply`, `_pg_oracle_search`) search only the declared top-5 slate product plus the couples of the hand plan, so every oracle plan is inside the slate by construction;
slate membership cannot be read from the diag. Gain = hand-plan label minus oracle-plan label per batch, summed.

| rung | oracle batches | improved | gain s | % of hand label s | within-cap (batches / improved / s) | over-cap (batches / improved / s) |
|---|---|---|---|---|---|---|
| moderate | 73,279 | 9,718 | 45,148 | 7.9 | 59,908 / 6,359 / 26,527 | 13,371 / 3,359 / 18,621 |
| heavy | 73,281 | 14,512 | 82,168 | 15.8 | 56,700 / 9,487 / 48,632 | 16,581 / 5,025 / 33,536 |

(Counts match `read_864107.log`; over-cap gains are replay-scored local-search lower bounds.) The batch-level gain is larger than the live latency gap, which is a paired mean over whole traces.

- **By batch size** the room is spread, not concentrated: gain as a share of hand label seconds, moderate n = 1..10: 1.4, 8.7, 8.2, 12.3, 5.4, 8.0, 10.6, 8.6, 2.6, 6.7 %; heavy: 6.5, 22.7, 20.2, 15.6, 18.1, 13.7, 11.6, 25.9, 5.4, 9.6 %. Full counts per size in `diag.txt`.
- **Scorer-reachable, but deep in the S rank.** Among improved within-cap batches the oracle plan is never the S-rank-1 plan of the slate table; it is in the S top 4 in 4,930 of 6,359 (78 %) moderate and
  7,195 of 9,487 (76 %) heavy, top 8 in 87 % / 86 %, top 32 in 95 % / 95 %. Weighted by gain seconds, S top 4 reaches only 28 % (moderate) / 21 % (heavy) of the within-cap gain, top 32 about 66 %.
- **Cold (proxy only).** Batches whose oracle plan uses a platform no applied plan of the cell had used before: 403 moderate / 500 heavy (0.5-0.7 % of batches) carry 25 % / 17 % of the gain seconds.
  The sim's real cold charge (previous task type differs) is not in the diag.

## Pool 2: outside the slate

On the 519 states of `rp/surrogate-gap` frontier_v1 (held-out 373), the full-table optimum is NOT in the declared top-5 product in 17 of 182 held-out moderate states (9 %) and 24 of 191 heavy (13 %)
(all sources 32 / 274 and 30 / 245). There the best slate plan is a median 92 % (moderate) / 19 % (heavy) worse than the optimum (p25 29 % / 10 %); almost all exceed 1 %. Multi-node optimum in 7 of the 17 and
3 of the 24; the optimum charges a real cold start in 16 / 182 and 2 / 191, of which 3 and 1 lie outside the slate. Only a wider (uncapped) slate reaches this room. Its **live size is unmeasured**:
corpus states are not the 12 live cells and a per-state regret is not a latency.

Not measured here: the labels' cold-start mix on the live cells, and an oracle over a wider slate.

## Amendment 2026-10-13: S term decomposition (interim) and the registered S_drain_weight diagnostic

A diagnostic on the bar, not a new lineage. `ORACLE_DIAG_TERMS=1` (default off; `rp/oracle-terms`, logging only, identity cell 19405 heavy equal to `oracle_8cc890fa` in every key but timings) writes per-plan S terms into the oracle diag. Interim read, 6 heavy cells and moderate 19405 (moderate 19406-19410 pending), in `oracle_terms_8b6f7352`:
- 87 % (heavy) / 66 % (moderate) of the oracle's gain seconds fall before t < 360 s, the cold-start deploy transient; heavy p5 arrival is at 184-186 s, so the middle bin is empty there.
- On 8 uniformly random logged plans per within-cap batch (unselected), S explains the label change in steady state with R2 0.98 (heavy, moderate), but not in the opening (heavy t < 360: 0.30, 0.68 with the terms re-weighted). Per-second label response: cold 0.9, exchange 1.0, **queue (drain + inflight) -0.06 heavy / -0.14 moderate, i.e. about 0**, while S charges it at weight 1. Collinearity of drain with the other terms is not excluded.
- In steady-state batches where the oracle gains >10 %, S prefers the hand plan in 95 % of them; the term S charges against the oracle plan is queue (+0.7 s heavy, +2.4 s moderate), most visibly on cnn xavierGpu->xavierGpu node moves (drain +1.4 to +1.7 s).

**Registered before data (signed by the coordinator): S_drain_weight.** Live test of whether S overprices queue drain. Arms, all `cd_exactS_ib0` with the platform's own drain weight w (in-batch stacking stays 0): w = 1 the registered ib0 (reference), w = 0.5 `cd_exactS_ib0_d05` (`HEROSIM_PG_DRAIN_ALPHA=0.5`, `HEROSIM_PG_BATCH_ALPHA=0`), w = 0 the existing `cd_exactS_a0` (`HEROSIM_PG_DRAIN_ALPHA=0`, which also zeroes the batch alpha). g0, both rungs, the 12 `ib0_confirm_selected` topologies, seed 0, at a pin carrying the steady summarizer (c81f0e88) and the name-shadow fix (e3d5e069); the reference arm is rerun at the pin (the d192081f summaries lack the after-360 s keys), and each rerun is checked against its d192081f summary. No default changes; the only code is the one arm entry (`rp/s7-drain-weight`).
- Predictions (coordinator): w = 0 moves ib0 by at most 1 % either way, whole-run and after_360s (sparse_obs stage 1: the drain channel is worth < 1 %).
- Labels, per rung on after_360s with Holm over w in {0, 0.5}: faster by >= 2 % -> `S-OVERPRICES-DRAIN` (a hand improvement, reported as one, the bar moves); within +-1 % -> `DRAIN-WEIGHT-NEUTRAL`; slower by >= 2 % -> `S-PRICES-DRAIN-RIGHT` (the regression coefficient is a collinearity artefact).
- Also reported per arm: the collapse count (end - last arrival > 300 s, or mean > 5x the cd median) beside the medians, since `warm_surge` found ib0 brittle under surges.

## 2026-10-13: type-aware residual of S — `RESIDUAL-IS-QUEUE / NO-PLAN-GAIN-STEADY`

`scripts_cosim/oracle_terms_resid.py` (`rp/oracle-terms` aaa18505), on the logged plans of 6 cells (heavy 19406-19410, moderate 19405; moderate 19406-19410 pending, so the moderate half is one topology). Row = a logged plan p of a within-cap batch against the hand plan h; target rho = (label_p - label_h) - (S_p - S_h); features = per-task-type deltas of the S terms (queue = drain + inflight, raw stacking, cold, exec, lat, exch), totals, moved-task counts, node-count change, batch size. Held out by topology (leave-one-heavy-topology-out, and train all heavy -> moderate 19405). Pools for the regret of argmin(S + rho_hat): **A** = the hand plan + the 4 lowest-S logged plans (no label information in the pool); **B** = A + the logged random and label-top plans (optimistic: the label-best plans are in it). Gains are the myopic batch label as % of hand label seconds in the bin, not live latency.

| R2 of rho, held-out heavy folds | all | t < 360 s | steady |
|---|---|---|---|
| type-aware | 0.49-0.63 | 0.48-0.64 | 0.26-0.47 |
| type-agnostic | 0.45-0.57 | 0.42-0.56 | 0.33-0.45 |
| moderate 19405 (heavy-trained) | 0.21 | 0.06 | 0.35 |

| permutation importance (R2 drop, term group permuted) | queue | exec | cold | exch | raw stacking |
|---|---|---|---|---|---|
| steady, heavy folds | 0.68-0.87 | -0.02..0.03 | 0.05-0.16 | ~0 | ~0 |
| steady, moderate 19405 | 0.67 | 0.01 | 0.09 | 0.00 | 0.00 |
| t < 360 s, heavy folds | 0.35-0.52 | 0.39-0.69 | ~0.07 | 0.03-0.08 | ~0 |

| gain of argmin(S + rho_hat), % of hand label s | pool A | pool B | oracle in pool A (upper bound) |
|---|---|---|---|
| t < 360 s, heavy (5 folds) | +6.0..+7.7 | +14..+19 | 7.8-12.9 |
| t < 360 s, moderate 19405 | -3.6 | +3.5 | 10.5 |
| steady, heavy (5 folds) | -0.63..+1.31, mean ~ +0.24 | -2.2..+0.5 | 1.7-7.1 |
| steady, moderate 19405 | +0.66 | +1.09 | 2.4 |

Reading. The residual is predictable (R2 far above 0.05) and, in steady state, carried by one term: queue. Correcting S with it buys within +-1 % with a fold-dependent sign, so there is no steady-state plan gain; type awareness adds nothing in steady state. The +6..+19 % in the opening is the cold-start deploy transient and is recorded as such, not as a lever. The single-term branch is tested live by the S_drain_weight probe above (drain weight scaled), not by a learned residual arm. Ruling (coordinator): no learned-residual arm, no new lineage; moderate cells fold in as a one-line update unless moderate steady pool-A gain exceeds +2 % on >= 3 topologies.
