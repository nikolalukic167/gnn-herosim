# gnn_seeded_cd_mp_ablation_v1 — learned MP-OFF seed in the fixed-replica search seat

**Status:** `CLOSED` (2026-09-22) — `NO-MESSAGE-PASSING-WIN` for the seed-1 `joint_burst_v2` checkpoint and fixed-replica coordinate-refinement seat. This ablation was designed after the GNN-versus-hand gate and uses its already opened topologies.

**Outcome.** The separately trained seed-1 MP-OFF twin, followed by the same six-pass hand coordinate refinement, beats the message-passing GNN seed in **8/8 live pairs** by **1.68% median** mean task elapsed time (exact one-sided sign p = **0.00390625**). It also beats the hand start in **8/8**, by **3.58% median**. Every contrast keeps its direction after measured inference time is added. The fixed-replica [parent gate](gnn_seeded_cd_fixed_v1.md) correctly establishes a learned-seed win over hand search; this live ablation shows that its positive result cannot be attributed to message passing. The 2026-09-23 32-task graph pilots show occasional GNN edges over MP-OFF or hand, but no reliable three-arm win; their initial offline move screens omitted peer-exchange physics and are invalid for headroom or training qualification. The live result does not rule out other graph models, checkpoints, workloads, or replica rules.

**Parents:** [gnn_seeded_cd_fixed_v1](gnn_seeded_cd_fixed_v1.md), [joint_burst_v2](joint_burst_v2.md).

## Method

The MP-OFF checkpoint is `joint-burst-v2-mpoff-lr2e3-seed1.pt`, matched to the parent's `joint-burst-v2-gnnedge0-lr2e3-seed1.pt` from the same training program. The sidecars differ only in `disable_message_passing`, `mp_bipartite_edge_attr_zero`, and `mp_bipartite_edge_conv`; all other feature, physics, architecture, and serving fields match. The MP-OFF checkpoint and its sidecar were read from datalab and hash-verified locally. The new arm uses the parent's tracked fixed-replica runner and eight topology configs, exchange weight 2, six refinement passes, the same 50,000-arrival workload, and the same replica manifest within each triplet. [Compact audit](gnn_seeded_cd_mp_ablation_v1/read.json) stores hashes, contracts, per-topology costs, counters, and parity checks.

## Record

- [2026-09-23 — connected degree-4 32-task exploration](#2026-09-23--connected-degree-4-32-task-exploration)
- [2026-09-23 — structured 32-task peer pilot](#2026-09-23--structured-32-task-peer-pilot)
- [2026-09-22 — fixed-seat debugging and move screens](#2026-09-22--fixed-seat-debugging-and-move-screens)
- [2026-09-22 — eight-topology live MP-OFF attribution](#2026-09-22--eight-topology-live-mp-off-attribution)

### 2026-09-23 — connected degree-4 32-task exploration

The [audit](gnn_seeded_cd_mp_ablation_v1/regular_g32_debug_read.json) and [reproduction bundle](gnn_seeded_cd_mp_ablation_v1/regular_g32_reproduction_2026-09-23.tar.gz) cover CPU-only fixed-replica live comparisons with 32-task connected degree-4 peer graphs and peer exchange enabled. With full payload and coordinate-descent scale 4 for all arms, the old GNN beats MP-OFF on **6/8** topologies (median **+2.85%** gain) but hand on only **3/8** (median **−1.29%**). Shuffling the same full-payload graph gives **4/8**, median **+0.36%**, versus MP-OFF and **7/8**, median **+1.55%**, versus hand. A disconnected regular-graph precursor split batches into eight-task components, so it was not a valid 32-task pilot.

Calibration selected scale 4 for GNN, 6 for MP-OFF, and 4 for hand. On four reserved topologies across the full-payload bridge and shuffled graph, GNN beats MP-OFF in only **2/8** paired cells. Halving payload on the shuffled graph gives GNN **5/8** wins over MP-OFF (median **+0.88%**) and **6/8** over hand (median **+1.50%**); six additional postselection topologies give **4/6** and **5/6**, respectively. An unequal-edge weighted swap that preserves the exact edge set and each task's weighted degree splits the GNN–MP-OFF result across two topologies. These exploratory cells do not establish a reliable message-passing win against both controls.

The physics check on one bridged snapshot shows why the earlier mini co-sim screens were invalid: with peer exchange off, a moved pair costs **15.363** versus a **17.219** baseline and has `averagePeerExchangeTime=0`; with it on, that moved pair costs **149.378** and has `averagePeerExchangeTime=0.613`, while the baseline remains **17.219**. With peer physics enabled, the selected 65-plan screen on 16 full-payload bridge snapshots has **0.0% median headroom** over the tuned hand scale-4 baseline. A separate weighted half-payload 65-plan screen has **17.60%** median finite-reference headroom, but the remote-payload hand selector gains **10.90%**, median residual beyond hand is **0**, and hand matches the best proposal in **9/16** snapshots. Neither selected slate establishes residual graph information.

All 32-task model pilots use old ten-task checkpoints out of distribution. No new model was trained, no GPU was used, and no fresh powered live gate was run.

### 2026-09-23 — structured 32-task peer pilot

The [structured pilot audit](gnn_seeded_cd_mp_ablation_v1/structured_debug_read.json) changes each 32-task group's peer edges to two overlapping eight-task partitions while preserving its total peer payload. On the same fixed-replica topology 9303, all three arms complete 160 groups (5,120 tasks): the old ten-task GNN scores **6.027** seconds mean elapsed per task, MP-OFF **6.176**, and hand **5.759**. Message passing edges MP-OFF in this short exploratory cell but still loses to hand; these checkpoints are out of distribution at 32 tasks.

The initial 65-plan mini co-sim screen for 16 MP-OFF snapshots **omitted peer-exchange physics**: the evaluator was invoked without `HEROSIM_PEER_EXCHANGE=1`, although its workload included `peer_exchange`. Its reported **19.95% headroom**, **15.87% hand gain**, **1.24% residual**, and proposal-type counts are invalid as evidence about the intended environment. The [structured pilot audit](gnn_seeded_cd_mp_ablation_v1/structured_debug_read.json) retains those superseded calculations for traceability; do not use them for qualification. The original overlapping-partitions slate has not been rerun with peer physics enabled. Its transformation preserves group payload but changes edge count and individual task degrees; it is not a degree-preserving topology counterfactual. No new learner or fresh live gate was run.

The separate [connected degree-4 exploration](#2026-09-23--connected-degree-4-32-task-exploration) contains the peer-on diagnostic and selected-slate screen. It is not a corrected result for this overlapping-partitions slate.

### 2026-09-22 — fixed-seat debugging and move screens

The [diagnostic audit](gnn_seeded_cd_mp_ablation_v1/debug_read.json) separates the learned seed from coordinate refinement on topology 9303. MP-OFF already leads the GNN at pass 0 (3.778 vs 3.951 seconds per task); its lead remains after pass 1 (3.029 vs 3.108) and pass 6 (2.809 vs 2.853). The six-pass search does not create the graph deficit.

The [reproduction bundle](gnn_seeded_cd_mp_ablation_v1/debug_source_2026-09-23.tar.gz) contains the scratch scripts, workloads, configuration, snapshots, and plan traces for both diagnostic screens; its hash is recorded in their audit JSONs.

For the ten-task burst, 12 of 16 captured fixed-state snapshots admit the bounded sweep. Its finite candidate reference has **7.66% median headroom** over the baseline plan. A hand exchange rule nearly matches the best two-move reference where two-move headroom exists, so the apparent search gap is not yet evidence of residual graph information. Four snapshots fail the sweep's eligibility condition.

An exploratory 32-task pilot on the same topology completed 160 full batches per arm with one fixed replica manifest. Mean elapsed time was **6.664** seconds for hand, **6.975** for the old GNN, and **6.975** for its MP-OFF twin; the ten-task checkpoints are out of distribution here. The default exact-corpus screen rejects all 16 attempted 32-task snapshots (even a binary slate has 65,536 combinations), so this pilot supplies no exact training labels.

The first 16 captured 32-task snapshots also received a 65-plan pair/triple mini co-sim screen. That screen used the same peer-off evaluator error as the structured screen above. Its reported **16.79% headroom**, **10.30% hand gain**, and **3.88% residual** are invalid for the intended peer-exchange environment; the [diagnostic audit](gnn_seeded_cd_mp_ablation_v1/debug_read.json) preserves the superseded calculations only for traceability. The ten-task evaluator ordering check could not detect this physics mismatch. The live pilot remains valid, but these offline numbers support no oracle, residual-information, training, or live-gate claim.

Any further coordinated-move screen must assert peer-exchange physics is enabled and exercise a nonzero peer-exchange metric on a moved-plan probe before measuring headroom. Before training, measure finite-reference headroom and residual improvement after strong 1/2/4-hop hand controls and equal-budget hand search; advance only if the residual is at least **8%**. Any learner that passes those screens still needs matched MP-OFF training and a fresh three-arm live gate.

### 2026-09-22 — eight-topology live MP-OFF attribution

| Topology | Message-passing GNN | MP-OFF twin | Hand start | MP-OFF reduction vs GNN |
|---|---:|---:|---:|---:|
| 9303 | 2.8528 | **2.8093** | 2.8457 | 1.53% |
| 9304 | 2.0078 | **1.9648** | 2.1122 | 2.14% |
| 9307 | 3.3871 | **3.3113** | 3.5671 | 2.24% |
| 9308 | 3.4340 | **3.3592** | 3.4626 | 2.18% |
| 9310 | 3.0467 | **3.0023** | 3.1132 | 1.46% |
| 9312 | 5.2468 | **5.1502** | 5.5308 | 1.84% |
| 9313 | 4.6363 | **4.6227** | 4.7955 | 0.29% |
| 9314 | 3.6540 | **3.6345** | 3.6969 | 0.53% |

Values are mean elapsed seconds per task. All eight MP-OFF runs completed 50,000 tasks, had zero scale events, used the same code fingerprint as the parent's 16-arm gate, and reproduced the parent's pairwise manifest. The comparison is exploratory because it was added after the parent gate's GNN-versus-hand outcomes were visible. Its uniformly negative graph contrast is nonetheless enough to reject *promoting this specific checkpoint and fixed search seat* as a message-passing advantage. A claim for a different graph-conditioned learner needs a new checkpoint and a fresh three-arm live gate against both MP-OFF and hand search.
