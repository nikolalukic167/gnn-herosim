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

## Time split (SCIENTIST, 2026-10-12)

**Bottom line.** Most of the label room the oracle finds over the hand bar is in the opening of the run: 84 % (heavy) / 68 % (moderate) of the gain seconds sit in the first 5 % of each trace. That opening is the whole system's cold start, not a load surge. Once the cluster has warmed up, the batch-label room is 5.35 % / 2.75 % of hand seconds, and S still ranks the oracle plan near the top (median S rank 2 / 3).

**Scope.** Within-cap batches only, i.e. those whose diag row carries the full S table: 56,700 heavy and 59,908 moderate, on 6 cells per rung (19405–19410). Gain = hand-plan label − oracle-plan label, both from `oracle_diag/` of `/share/nikola.lukic/ib0_confirm/oracle_8cc890fa`. These are batch-label seconds, not live latency. The live oracle gap stays −0.92 % / −4.17 % (node). The run phase is the batch time t ÷ the cell's last batch time, which is 3,933 s heavy and 9,354 s moderate. The whole-run within-cap gain is 15.50 % (heavy) / 6.99 % (moderate) of hand label seconds. The table above (15.8 / 7.9 %) also counts over-cap batches.

**Gain by run phase.**

| rung | phase | batches | share of gain | batches with >10 % gain | mean hand label per task |
|---|---|---|---|---|---|
| heavy | first 1 % | 654 | 23.8 % | 31.0 % | 49.9 s |
| heavy | 1–5 % | 2,178 | 60.0 % | 41.8 % | 9.4 s |
| heavy | 5–25 % | 11,214 | 7.0 % | 8.0 % | 0.9 s |
| heavy | 25–100 % | 42,654 | 9.2 % | 7.8 % | 0.9 s |
| moderate | first 1 % | 622 | 59.0 % | 52.1 % | 30.3 s |
| moderate | 1–5 % | 2,369 | 9.3 % | 10.7 % | 2.1 s |
| moderate | 5–25 % | 11,883 | 9.6 % | 4.0 % | 1.7 s |
| moderate | 25–100 % | 45,034 | 22.0 % | 3.8 % | 1.8 s |

- **Concentration.** Run quarters 1–4 carry 90.9 / 2.8 / 3.2 / 3.2 % of the gain (heavy) and 78.0 / 6.5 / 7.5 / 8.0 % (moderate). Half of all heavy gain sits in 315 of 56,700 batches, 80 % in 739; for moderate it is 195 and 543 of 59,908.
- **After the first 5 % of the run,** the label gain is 5.35 % (heavy) / 2.75 % (moderate) of hand seconds. That is the steady-state batch-level headroom, and the live headroom is smaller still.

**Where the oracle plan sits in S, and how it differs from the hand plan.** These rows cover batches whose gain exceeds 10 % of the hand label. "Unstacks" means the oracle puts fewer task pairs on one platform. "Fewer co-located" means fewer task pairs on one node. "New node only" means the oracle keeps the hand plan's stacking and co-location but uses a node the hand plan did not. Gain columns are shares of the rung's total gain.

| rung | phase | batches | oracle S rank (median) | unstacks (gain) | stacks more | fewer co-located | more co-located | new node only (gain) |
|---|---|---|---|---|---|---|---|---|
| heavy | opening (<5 %) | 1,113 | 13 | 23.6 % (17.6 %) | 0.4 % | 37.3 % | 4.0 % | 42.3 % (34.9 %) |
| heavy | steady | 4,229 | 2 | 34.2 % (3.3 %) | 0.4 % | 24.1 % | 3.3 % | 45.8 % (7.3 %) |
| moderate | opening (<5 %) | 578 | 6 | 24.6 % (13.8 %) | 1.7 % | 26.1 % | 5.2 % | 50.0 % (31.9 %) |
| moderate | steady | 2,181 | 3 | 31.8 % (3.3 %) | 1.7 % | 22.3 % | 10.0 % | 41.9 % (14.1 %) |

- **The oracle's moves.** It almost never stacks or co-locates more than the hand plan. It spreads out: in 80 % (heavy) / 76 % (moderate) of these batches it uses a node the hand plan did not. Over the whole run the oracle plan's mean S rank is 12.1 / 13.8 (rank 1 = S's argmin). In the opening S misranks the oracle plan deeply (median rank 13 heavy); in steady state the oracle plan is near S's top (rank 2–3).

**Does anything beyond S predict the label error?** (`q2.py`, 25 % plan sample, cells 19409–19410 held out.) Per plan, y = (label − hand label) ÷ hand label.

| rung | rows | R² of y from ΔS alone (held out) | residual R² from plan-structure features | regret of argmin S | regret of a GBM argmin |
|---|---|---|---|---|---|
| heavy | all | 0.993 | 0.013 | 15.49 % | 11.37 % |
| heavy | phase ≥ 5 % | 0.996 | −0.049 | 6.45 % | 5.84 % |
| moderate | all | 0.989 | −0.214 | 8.86 % | 7.07 % |
| moderate | phase ≥ 5 % | 0.990 | −0.191 | 3.56 % | 4.19 % |

- ΔS explains ≥ 98.9 % of the label's variance across plans. The structure features (node, platform, stack and co-location counts and their changes vs the hand plan, batch size, plan count, run phase) add nothing to ΔS on held-out cells: residual R² is ≤ 0.013 and negative elsewhere.
- The permutation importance of run phase (tf) is the largest after ΔS. A GBM argmin beats argmin S on the sampled plans only when the opening is included (heavy 11.37 vs 15.49 %). Over the steady phase the two tie or the GBM loses (5.84 vs 6.45 %; 4.19 vs 3.56 %).
- Regret here is over the sampled plans, as a share of hand label seconds; it is not a live number.

**Why the opening is a cold start, not a surge.**
- The live path runs the full workload with no warm-up tasks and autoscales from zero replicas (`src/executesimulation.py:5`, `:116`), with no preinit, replica plan or deterministic placements (`:202`).
- `averageElapsedTime` is total_rtt ÷ n_tasks over every task, with no warm-up discard (`src/placement/orchestrator.py:431`).
- The arrivals have no burst: the grounded mint stretches every inter-arrival by one constant (`scripts_cosim/grounded_workload_v1_mint.py:101–123`).
- So the first ~5 % of every live run is the cluster scaling out, pulling images and cold-starting sandboxes. The mean hand label per task is 49.9 s in the first 1 % vs 0.9 s in steady state (heavy).

**Where S's terms are missing during the cold start.** This list comes from reading the code; it is not a measured decomposition. Each item makes S's error larger at the opening.
- An image pull is unpriced unless the cd_pull control is on (`src/policy/peer_greedy_network/scheduler.py:85–87`, `HEROSIM_PG_PULL_HOLD`).
- The queue drain walk charges each queued task its execution time, storage I/O, network latency and peer transfer, but no cold start or pull (`src/placement/live_audit.py:92–112`).
- The incoming cold start is free whenever a task is computing on the platform, and is priced against `previous_task` otherwise (`src/placement/scheduling_cost.py:71–83`), not against the queue tail.
- `cd_exactS_ib0` puts zero weight on in-batch stacking.

Splitting each opening batch's gap into these terms is the job of S7's term logging (step a). That logging, not this table, settles which term carries the opening.

**Reproduce.**
1. Run `docs/lineages/ib0_confirm_v1/scripts/oracle_feats.py OUT` (stdlib only; run it on the datalab login node). It reads the 12 cell directories of `oracle_diag/` in sorted order and writes `OUT/batches.tsv` (116,608 rows) and `OUT/plans.tsv` (a 25 % plan sample, `random.Random(0)`).
2. Run `q1.py OUT` (whole-run gain classes and quarters), `q1b.py OUT` (phase table), `q1c.py OUT` (S-rank and move-class table) and `q2.py OUT` (residual R²; needs pandas and scikit-learn) from the pipenv environment.
3. A re-extraction with this sorted order reproduced every number above exactly.
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

## 2026-10-13: S_drain_weight read — `DRAIN-WEIGHT-NEUTRAL` (moderate), heavy inconclusive and leaning slower

72 cells, jobs 864522-864524, pin 9bd1b145 (reference-physics 18161421 + the one arm), g0, 12 topologies per rung, seed 0, output `/share/nikola.lukic/ib0_confirm/drain_weight_9bd1b145`, reader `dw_read.py` there. **Identity:** the 24 weight-1 reruns (`cd_exactS_ib0`) equal their d192081f summaries in `total_rtt` exactly (max relative difference 0.0, 24/24). Paired per topology against weight 1; mean task latency, whole run (`total_rtt / num_tasks`) and `steady_state.after_360s.mean`; Holm over the two arms.

| rung, statistic | w = 0.5 mean (median), faster | w = 0 mean (median), faster | Holm p (w0.5 / w0) |
|---|---|---|---|
| moderate whole run | -0.08 % (+0.26), 6/12 | -0.45 % (-0.41), 8/12 | 0.91 / 1.0 |
| moderate after_360s | +0.07 % (+0.01), 6/12 | -0.88 % (-1.08), 8/12 | 0.94 / 0.59 |
| heavy whole run | +2.19 % (+2.77), 4/12 | +1.99 % (+1.54), 4/12 | 0.13 / 0.07 |
| heavy after_360s | +4.21 % (+5.78), 4/12 | +1.30 % (+0.12), 6/12 | 0.20 / 0.31 |

Per-topology ranges are wide (moderate after_360s -4.6..+5.9 %, heavy -8.1..+16.6 %). **Collapse count** (end - last arrival > 300 s, or mean > 5x the weight-1 median): 0/12 in every arm and rung; the largest end - last arrival is 81 s; 0 request failures.

Labels. Moderate: no arm is faster by >= 2 %, the means sit within +-1 % -> `DRAIN-WEIGHT-NEUTRAL`, as predicted. Heavy: no arm is faster; the point estimates are slower (w = 0.5 +4.2 % after_360s, +2.2 % whole run; w = 0 +1.3 % / +2.0 %) but no contrast survives Holm (smallest 0.07), so neither `S-PRICES-DRAIN-RIGHT` nor neutrality is established at n = 12. What it does rule out: removing or halving the drain term is not a hand improvement at either rung, so the regression coefficient of about 0 on the queue term (random logged plans) does not translate into a better S by down-weighting it; the bar `cd_exactS_ib0` does not move.
