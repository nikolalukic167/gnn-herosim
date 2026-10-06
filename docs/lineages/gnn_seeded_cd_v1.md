# gnn_seeded_cd_v1 — GNN initialization for the existing peer-greedy refinement

**Status:** `CLOSED` (2026-09-22) — `NO-GNN-SEED-WIN-ESTABLISHED` on the four-topology exploratory live pilot and matched-score amendment. This was designed after inspecting the existing gate, not preregistered or powered. Closure is specific to this checkpoint, seed rule, and w0 burst seat.

**Outcome.** The `joint_burst_v2` seed-1 bipartite GNN supplies an initial complete batch plan for the existing peer-greedy coordinate refinement. The first four-topology w0 pilot beat the default hand CD rule in all four, but compared exchange weight 1 in the hybrid against weight 2 in the stronger hand control. The matched amendment sets **weight 2 and a six-pass cap on both**. At 50,000 tasks per arm, the GNN-seeded hybrid wins 2/4 topologies and loses 2/4; its median paired elapsed is **1.69% slower** than the hand start. The hybrid consumes about 53–56 seconds of measured inference per run versus 3–4 seconds for the hand rule. There is no established GNN seed advantage under dynamic autoscaling. The earlier 8.46% loss and 3/4 hand wins were a **confounded contrast**, retained below for provenance but not the standing answer. A follow-up on topology 9106 traced the weight-sensitive collapse to a diverged replica set that left overloaded node4 as the only candidate for most sampled affected tasks. The [fixed-replica successor](gnn_seeded_cd_fixed_v1.md) removes that feedback and passes a fresh hand comparison; it does not reverse this dynamic-autoscaler result.

**Parents:** [joint_burst_v2](joint_burst_v2.md) (checkpoint, burst seat, CD bar), [peer_greedy_live_v1](peer_greedy_live_v1.md) (hand score), [rollout_imitation_v1](rollout_imitation_v1.md) (offline-positive/live-negative warning).

## Question and method

Can the GNN contribute a useful starting plan to the hand rule's multi-pass search? The new `gnn_seeded_cd` policy calls the production GNN prefix decoder, validates every seed placement, builds the same committed-service and peer-node ledgers as the hand rule, then invokes `PeerGreedyNetworkCDScheduler._pg_refine` without changing its candidate set or score. The decisive amendment sets `HEROSIM_PG_EXCHANGE_SCALE=2.0` and `HEROSIM_PG_CD_PASSES=6` for **both** the hybrid and hand-start CD. Weight 2 was already probed in `joint_burst_v2`. Workload, topology, hand score and pass cap now match within each pair; total CPU time still favors the hybrid because it adds GNN inference.

All live arms use the saved `joint_burst_v2` w0 burst workload and four selected topology configs, node-disk warmth, peer exchange, peer-group batching, the 16-second batch window, and 50,000 arrivals. The hybrid uses the original seed-1 `gnnedge0` checkpoint with its mandatory contract. All 16 runs share one tracked-code diff fingerprint; every arm completes all tasks. [Compact run record](gnn_seeded_cd_v1/pilot_read.json) retains input/checkpoint hashes, the untracked policy file's separate SHA-256, per-arm decomposition, counters and provenance fingerprints. Full raw outputs were produced locally under `/tmp` and are ephemeral. This pilot reuses previously opened gate cells and has one checkpoint, so it supports a scoped decision rather than a population-level significance claim.

## Record

- [2026-09-22 — matched-score live amendment](#2026-09-22--matched-score-live-amendment)
- [2026-09-22 — topology 9106 basin diagnosis](#2026-09-22--topology-9106-basin-diagnosis)
- [2026-09-22 — exploratory live pilot](#2026-09-22--exploratory-live-pilot)

### 2026-09-22 — matched-score live amendment

The first pilot varied two things at once: the GNN seed and exchange weight 1 versus 2 in the refinement score. It therefore could not attribute its stronger-control loss to the seed. The correction reran the same seed-1 GNN policy with `HEROSIM_PG_EXCHANGE_SCALE=2.0` and `HEROSIM_PG_CD_PASSES=6`, matching the hand control on both knobs. Four new 50,000-task live runs completed. Average elapsed seconds per task:

| w0 topology | GNN seed + CD6, scale 2 | hand seed + CD6, scale 2 | GNN relative to hand |
|---|---:|---:|---:|
| 9101 | **4.450** | 4.551 | −2.22% |
| 9106 | 9.249 | **8.759** | +5.60% |
| 9114 | **4.942** | 5.301 | −6.76% |
| 9116 | 7.679 | **7.066** | +8.68% |

**Verdict: no GNN seed win established.** Median paired difference is **+1.69%** (GNN slower), with two wins and two losses; no significance claim is possible from four previously opened topologies and one checkpoint. On 9106, a subsequent six-pass, weight-1 rerun measured 3.543 s/task versus 9.249 at weight 2, isolating the refinement weight from the pass cap. The model seed costs 53–56 seconds of inference per run; the hand start costs 3–4 seconds. The matched comparison still grants the hybrid much more compute.

The existing `git diff HEAD` provenance hash excludes the new untracked policy file. The compact run record therefore stores its separate SHA-256 (`9df8b8712cce452d7d3c84ce4e9c66b9272f4bec89d1384210d1016cb8e1ce5b`) along with the shared tracked-code fingerprint. The new policy was not edited between these runs.

### 2026-09-22 — topology 9106 basin diagnosis

A six-pass, weight-1 rerun on the same 50,000-task trace measured **3.543 s/task** (queue 1.498, peer exchange 1.926), versus **9.249 s/task** (queue 7.252, exchange 1.881) at weight 2. The extra refinement passes do not explain the contrast. An `availability_v2` weight-2 probe still measured **9.231 s/task** (queue 7.232), so adding observed in-service work to the hand drain estimate did not repair it. These probes are on one opened topology; they diagnose this cell rather than establishing a new population result.

The added wait is localized. In task IDs 15,000–24,999, weight 2 sent **825** tasks to node4, Xavier CPU platform 218; their mean queue time was **335.4 s**. Weight 1 sent **56** tasks there, with **1.2 s** mean queue time. Those 825 weight-2 tasks were all `nofs-dnn2`; their mean execution time was **0.024 s**, but mean peer exchange was **7.97 s**. The weight-2 run's node4 tasks explain about **93%** of that interval's extra task elapsed time. Outside the interval, weight 2 was slightly faster. This is a concentrated overload, not a broad deterioration in placement quality.

Read-only diagnostic wrappers around the unchanged GNN seed, refinement, and candidate capture reproduced both uninstrumented totals exactly (weight 1: `177161.46078701335` s; weight 2: `462463.8752224153` s, within floating-point summation). In sampled task IDs 18,600–18,749, the weight-2 run placed 26 tasks on node4; **25 had only node4 in the candidate set**. At IDs 20,000–20,039, all 10 node4 placements likewise had only node4 available. The GNN seed and final plan agreed on these node4 choices. At the first weight-dependent node4 split, task 18662, weight 1 had node0 and node4 available and refinement moved the GNN's node4 seed to node0 (scores 5.01 versus 8.65 seconds); weight 2 had only node4. At task 20020, weight 2 estimated **222.28 s** of drain on node4 and still had no alternative. Thus the runaway is mediated by the closed-loop replica set: once available replicas diverge, neither the GNN seed nor coordinate refinement can select a missing placement. Earlier decisions and autoscaling jointly determine that set; this trace does not identify one initiating decision or prove which autoscaler rule caused the divergence.

The practical control for the next test is to hold eligible replicas fixed across policies or explicitly gate an autoscaler-aware action space before attributing an outcome to graph-conditioned placement. Any new GNN claim must compare against the hand rule under the same candidate sets and score weight, and report candidate-set divergence as well as queue and exchange cost.

### 2026-09-22 — exploratory live pilot

**Superseded as a seed-only comparison:** this first pilot used exchange weight 1 in the hybrid and weight 2 in the stronger hand control. Its numbers are measured, but the cross-weight contrast cannot isolate GNN initialization. Average elapsed seconds per task:

| w0 topology | GNN-seeded CD | hand CD, scale 1 | hand CD, scale 2 | hybrid vs scale 2 |
|---|---:|---:|---:|---:|
| 9101 | 4.738 | 4.883 | **4.551** | +4.11% |
| 9106 | **3.554** | 9.602 | 8.759 | −59.42% |
| 9114 | 5.979 | 6.223 | **5.301** | +12.80% |
| 9116 | 8.003 | 8.590 | **7.066** | +13.26% |

The hybrid's median paired contrast is −5.37% against default hand CD and +8.46% against the scale-2 hand control, **the latter confounded by score weight**. The 9106/w0 hand result is consistent with the older `joint_burst_v2` gate's 9.606 s/task on the same cell. All 12 arms completed 50,000 tasks, and the hybrid's `pg_batches` and `prefix_batches` both increase, showing the GNN and refinement paths actually ran.

The prior CPU-only diagnostic on 12 separated *training* graphs established two facts that motivated the hybrid: re-running the unchanged graph encoder after each placement changes logits by exactly 0, because prefix features enter only the final scorer; and the frozen bipartite stage changes 7/12 plans but has only 0.20% median exact-cost benefit over switching that stage off. A one-pass **oracle-priced** coordinate repair reduced median sweep regret from 5.94% to 0.28%; its exact sweep lookups are unavailable to live serving and were never counted as a deployable win. The live hybrid tests whether the actual hand score can capture that opportunity. Against the stronger hand control it does not.

No further training or seed sweep is justified by this pilot. A new attempt needs a demonstrable GNN advantage over a hand control with at least the same information and compute allowance, measured across independent topologies and windows; tuning only the current GNN seed or using the scorer-only prefix re-forward does not satisfy that requirement.
