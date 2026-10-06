# g32_peer_fabric_s0_v1 — contended peer transfers on the core fabric

**Status:** `CLOSED` (2026-09-23) — **NO-GO for platform-start peer-fabric transfers**. Registered before the opt-in physics pilot and fresh live outcomes.

**Outcome.** Actual core-link waiting is only 0.60–1.06% of exact group RTT at the four training-topology medians, far below the signed 10% bar on at least three topologies. All eight fresh live manipulation runs audit and reproduce the small core-wait share. The 32-task routes overlap heavily, but platform service staggers the transfers; the simultaneous-transfer paper estimate did not survive the simulator. No GNN was trained. This closes this platform-start, store-and-forward peer-fabric recipe, not transfers prefetched at group commit or every shared-fabric environment.

**Record** (newest first):

- [2026-09-23 — exact physics screen and fresh live gate: NO-GO](#2026-09-23--exact-physics-screen-and-fresh-live-gate-no-go)
- [2026-09-23 — registration before changing physics](#2026-09-23--registration-before-changing-physics)

**Question.** In a 32-task group with 64 peer edges, does serving peer state transfers through the existing capacity-one backbone links create unavoidable, graph-dependent contention at good placements? The [four-type single-move screen](g32_fourtype_move_s0_v1.md) has exact local headroom, but topology-held-out hand features capture almost all of it under additive peer-transfer physics. The earlier [DAG fabric screen](dag_fabric_contention_v1.md) failed because four- and eight-task optima avoided shared core links; it explicitly left larger concurrently transferring groups as a reopening condition.

## 2026-09-23 — registration before changing physics

The exploratory, read-only [overlap probe](g32_peer_fabric_s0_v1/overlap_probe_2026-09-23.tar.gz) (archive SHA-256 `cfa76d2059f7788c7f8b1b07573ee9de28af051e7edae40fd6e0b2d1410bc351`) maps both captured baselines and the already frozen 128-move pool onto the existing fabric routes for 64 training groups. The probe script has SHA-256 `efa00db6a2b80b27bbb0d6e381c51b192332b2623bd08cd06b39d073fe3db94f`; output SHA-256 `03aaab4de678c119b67cee764c6492d867e4878263e3357259b9b6f8a0fe8404`. Every group's candidate plans share at least one core link among peer transfers. Even the minimum core-link multiplicity in the pool has topology medians 12.5, 10.5, 12.0 and 21.0; a synthetic simultaneous-transfer queue term remains nonzero on 62/64 bounded augmented-cost winners. Its median share of synthetic augmented cost is 11.8%, 17.3%, 18.1% and 27.5% by topology. This is a **paper upper-bound-style screen**, not actual simulator waiting or a global optimum. It uses the simulator's current `bandwidth_mbps × 1024²` time convention; simultaneous starts can overstate real overlap.

Implement an opt-in `HEROSIM_PEER_FABRIC=1` path in the live and exact mini-simulator input stage. Each directed remote peer transfer traverses the existing `NetworkFabric.hops` path store-and-forward, requesting and releasing each capacity-one `pipe` in order; preserve the current additive peer-exchange path byte-for-byte when the flag is off. Fail loudly if the flag is on without a fabric or an unresolved peer. Record separate peer-fabric transfer and link-wait telemetry, including core-only wait. Deterministic task/peer/hop order must reproduce at a fixed seed. The experiment changes the environment and target, not the old checkpoint or frozen historical sources.

First, run focused parity and determinism tests, then a four-group opened-development smoke on topology 9303. On the training-only topologies **9801, 9803, 9804 and 9806**, replay the same first 16 captured groups and bounded move pools under actual contended physics. Score the two complete baselines and up to 128 retained moves per group. Count topology, not group, as the paired independent unit. The baseline is the better exact-scored hand or old MP-OFF plan. The bounded oracle is the best scored plan in that finite pool, not a certified optimum. Give the hand control the same two baseline evaluations plus 16 move evaluations; rank with frozen local, platform-load, peer-hop and **route-link** load features. Also fit any learned hand-feature control leave-one-topology-out, never on the held-out topology. Report exact RTT, throughput, telemetry, cap coverage and timing.

**GO to matched GNN training only if all four pretraining bars pass:** (1) actual core-link peer wait exceeds 10% of group RTT at the topology median on at least three of four training topologies; (2) median topology bounded-oracle gain over the better baseline is at least 8%; (3) an equal-budget route-aware hand search captures less than half that gain; and (4) a residual decision signal survives platform counts, route-link load and peer 1/2/4-hop hand controls, with exact-stakes matched ranking reversals by topology. A nonpositive oracle denominator cannot count toward GO. No GPU before these bars pass.

Regardless of an offline NO-GO, complete a fresh live manipulation gate on at least two structurally eligible topology seeds outside 9303, the four S0 seeds and prior live seeds 9807/9808. Freeze seeds, configs, workload, code hashes, arms and flags before reading RTT. Run hand and old MP-OFF under the new physics on the full 5,120-task workload, paired with their default-off counterparts, and audit 160 complete groups, fixed replicas, nonzero peer transfer, measured core wait, zero scaling and no failures. The live gate checks that the physics fires and preserves execution; it cannot establish an architecture win.

If every pretraining bar passes, register a separate topology-disjoint, matched full-bipartite/peer-only/MP-OFF move learner with exact-validation checkpoint selection and a fresh eight-topology live architecture gate. Its full model must beat each learned control by at least 2% median and on at least 7/8 topologies at equal serving budget. Keep checkpoint sidecars, Weights & Biases runs and the sealed split. A hand win alone is not a bipartite win.

## 2026-09-23 — exact physics screen and fresh live gate: NO-GO

The opt-in implementation was isolated under `/tmp/herosim_peer_fabric`, leaving the fingerprinted repository sources untouched. It serves each directed remote peer transfer hop by hop through the existing capacity-one pipes when `HEROSIM_PEER_FABRIC=1`, and preserves the original additive path when off. The off-path exact RTT on training group 9801/0 is bit-identical to the frozen baseline (26.30527108732717 s); a repeated on-path development replay is bit-identical too. The four opened-development groups on 9303 completed with core wait below 1.8% of RTT. Seventeen existing peer-exchange tests passed against the isolated source, and setting the flag without a fabric failed loudly as registered.

The exact screen then replayed both frozen hand and MP-OFF plans on all 64 training groups under the new physics. For each group the lower-RTT arm defines the baseline whose core-link wait share is measured. The result JSON has SHA-256 `1bcc694d8cb8a7d15a3b6e86e203719d3eb172010f66031ac505842808ed05e5`.

| Training topology | Median core peer-link wait / baseline RTT | Highest group share | Groups above 10% |
|---:|---:|---:|---:|
| 9801 | 0.604% | 1.867% | 0/16 |
| 9803 | 1.060% | 2.792% | 0/16 |
| 9804 | 0.892% | 1.694% | 0/16 |
| 9806 | 0.904% | 4.120% | 0/16 |

The first signed bar needs **at least 10% at the topology median on 3/4**; it fails on all four. The move-pool oracle and residual controls were therefore not rescored under this physics. The paper overlap result measured available paths, not simultaneous use: each platform's input stage starts after its queue service begins, and those starts spread the transfers over time.

The required fresh live gate selected the first two coverage-eligible seeds after 9809, **9810 and 9813**, before RTT inspection; 9811 and 9812 failed coverage. Its freeze manifest SHA-256 is `0cc5ca4b83f8d1daf0172fc65e2a96173bf7ce7b5b5a537470fb30bfe1a5f3bb`. Hand and frozen MP-OFF each ran with peer fabric off and on on both seeds. All **8/8** runs completed 5,120 tasks in 160 groups with identical fixed-replica manifests within topology, zero scaling, and nonzero peer exchange. The flag-on runs measured core wait; flag-off runs measured zero peer-fabric wait.

| Seed | Arm | RTT off (s) | RTT on (s) | On core wait / RTT |
|---:|---|---:|---:|---:|
| 9810 | Hand | 4,444.445 | 4,864.346 | 1.035% |
| 9810 | MP-OFF | 4,459.918 | 4,877.767 | 1.048% |
| 9813 | Hand | 5,521.420 | 6,140.587 | 1.460% |
| 9813 | MP-OFF | 5,541.585 | 6,109.923 | 1.344% |

The RTT increase includes changed transfer scheduling and hop service, not just core waiting; it cannot be quoted as nonlocal graph headroom. The [raw archive](g32_peer_fabric_s0_v1/peer_fabric_s0_2026-09-23.tar.gz) includes the isolated physics sources, exact screen, frozen live runner/configs, all eight raw results and traces, freeze and audit. Its 52 members were verified against the per-file audit hashes. Archive SHA-256: `23da4ab4fa4e315458ad85ec74edcd6bbafd1a6a0aba8f4adc85d8d45907ea80`; audit SHA-256: `cbdb407167e13adfa608f2f0f7faecbacefd69f17018ef9a1a8be9b7eeca0595`. This live result confirms the first-bar failure and closes the registered recipe. A distinct group-commit prefetch mechanism would need its own protocol and controls.
