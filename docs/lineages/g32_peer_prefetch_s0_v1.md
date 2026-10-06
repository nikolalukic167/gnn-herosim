# g32_peer_prefetch_s0_v1 — group-commit peer-transfer prefetch

**Status:** `CLOSED` (2026-09-23) — **NO-GO for unfiltered group-commit peer prefetch**. Registered before prefetch code and outcomes.

**Outcome.** Finite-pipe prefetch causes a large RTT penalty versus otherwise identical infinite-pipe prefetch, but most contention is on access links. Median core peer-link wait is 5.43–16.54% of exact group RTT across four training topologies and clears the signed 10% bar on only one of four. The required twelve-run fresh live gate passes its execution audit and confirms that core wait is material on some workloads, not consistently enough for this GNN training gate. No move labels or GNN weights were trained under this physics. This closes unfiltered access-plus-core prefetch, not a core-only capacity experiment.

**Record** (newest first):

- [2026-09-23 — causal exact screen and fresh live gate: NO-GO](#2026-09-23--causal-exact-screen-and-fresh-live-gate-no-go)
- [2026-09-23 — registration before prefetch code or outcomes](#2026-09-23--registration-before-prefetch-code-or-outcomes)

**Question.** Can peer transfers start when all 32 group placements are committed, independent of each platform's compute queue, so shared core-link capacity materially affects task RTT? The [platform-start fabric pilot](g32_peer_fabric_s0_v1.md) fails despite 10–21 overlapping transfers on a typical core link: serialized platform input stages stagger them and actual core waiting is under 1.1% of RTT at every training-topology median. This successor changes *when* transfers start, rather than merely raising payload or tuning the failed start semantics.

## 2026-09-23 — registration before prefetch code or outcomes

In an isolated experimental source copy, add opt-in `HEROSIM_PEER_PREFETCH=1` requiring `HEROSIM_PEER_FABRIC=1`. Once a batch has assigned every peer's destination, launch each task's directed peer transfers as SimPy processes, even if its compute platform is busy. Each remote transfer walks the existing `NetworkFabric.hops` route store-and-forward; a task may execute only after both its ordinary input stage and its own peer transfers complete. Avoid shared platform service-state mutation from the prefetch processes. Keep all original flag-off paths bit-identical and fail loudly if a peer is unresolved or the fabric is absent. Record peer transfer, total link wait and core-only link wait separately.

Build an **infinite-pipe counterfactual** with identical prefetch timing, routes, transmission holds and propagation latency, but no capacity-one pipe wait. This isolates contention from changed input scheduling. A direct platform-start versus prefetch comparison is descriptive only; it cannot establish core contention's causal RTT effect. Verify two repeated seeded prefetch runs bit-identical, the flag-off exact baseline unchanged, positive peer-transfer telemetry and valid 32-task completion. Focused peer-physics tests must pass.

Use the opened development topology **9303** for an initial four-group smoke. Then, on training-only topologies **9801, 9803, 9804 and 9806**, evaluate the frozen hand and MP-OFF complete plans for the first 16 captured groups under both finite and infinite pipes. The lower exact RTT of the two plans defines each group's finite-pipe baseline. Count topology as the independent unit. **Advance to any move-pool scoring only if** the median topology core-link peer wait is at least 10% of finite-pipe group RTT on 3/4 topologies *and* the median topology paired RTT penalty from finite versus infinite pipes is at least 8% on 3/4. Both bars were fixed before outcomes. If either fails, stop offline expansion and still run the registered fresh live gate.

If manipulation passes, score the same deterministic ≤128-move pool under finite-pipe physics on all 64 training groups. A GNN training GO additionally needs ≥8% median-topology bounded-oracle gain over the better baseline, a 16-evaluation route-aware hand search capturing <50% of that gain, and actionable residual exact-RTT ranking reversals after platform counts, route-link loads and peer 1/2/4-hop controls under leave-one-topology-out evaluation. The oracle is bounded by this pool and is not a global optimum. No GPU allocation precedes every bar.

Regardless of offline NO-GO, freeze and run a fresh paired full-workload live manipulation gate on at least two structurally eligible topology seeds outside 9303, 9801/9803/9804/9806 and earlier live seeds. For each seed run hand and old MP-OFF with finite and infinite prefetch pipes, plus the original additive peer path as an anchor. Audit all 5,120 tasks and 160 groups, fixed-replica equality, zero scaling/failures, nonzero peer transfer and measured core wait, input/config/source hashes and inference time. Report paired RTT and core wait; this is an environment gate, not a GNN result.

Only after all pretraining bars pass, register topology-disjoint matched full-bipartite, peer-only and MP-OFF move learning with exact-validation checkpoint selection, W&B logs, sidecars and an eight-topology fresh live gate. Require the full model to beat each learned control by ≥2% median and on ≥7/8 topologies at equal serving budget. A prefetch physics gain cannot substitute for message-passing attribution.

## 2026-09-23 — causal exact screen and fresh live gate: NO-GO

The opt-in implementation lived only in `/tmp/herosim_peer_fabric`, leaving fingerprinted repository sources untouched. The task's scheduling event starts transfer processes after placements are planned; the platform waits for that process before execution. Finite and infinite modes have the same hop holds and latency; only finite mode requests capacity-one pipes. The default-off exact score for frozen group 9801/0 remains bit-identical (26.30527108732717 s), two seeded finite development replays match bit-for-bit, and 17 existing peer-exchange tests pass. All four opened-development groups complete, with finite-pipe core wait from 4.1% to 23.2% of RTT. Development cases were not used to select a training topology.

The [exact baseline screen](g32_peer_prefetch_s0_v1/peer_prefetch_s0_2026-09-23.tar.gz) evaluates both frozen complete plans on 16 groups per training topology under finite and infinite pipes: 256 exact mini-simulations, all 32 tasks complete. For each group the lower finite-pipe RTT selects the baseline; its core-wait share and the paired finite-versus-infinite penalty for the same plan are measured. The result JSON has SHA-256 `61deb182bb499b3d5df510758a4025c47768c15ddc801d2d1f1bd5b4276fd330`.

| Topology | Median core wait / finite RTT | Groups ≥10% | Median paired finite-pipe RTT penalty |
|---:|---:|---:|---:|
| 9801 | 6.910% | 4/16 | 54.899% |
| 9803 | 5.434% | 6/16 | 32.480% |
| 9804 | 7.305% | 6/16 | 45.556% |
| 9806 | 16.535% | 10/16 | 35.338% |

The causal RTT-penalty bar passes on all four topologies, but core wait clears its **≥10% median on 3/4** bar on only **one**. Access-link queues supply much of the pipe penalty; those are indexed by endpoint traffic counts. The pool oracle and learned hand residual are not scored after this first-bar failure. This is a measured NO-GO for the registered training path, not proof that no core route effect exists.

The fresh live manipulation gate froze the first two coverage-eligible new seeds **9814 and 9815**, its two arms, three modes, workload, model and source hashes before reading outcomes (freeze SHA-256 `27b01349eefa854a8e13b1ec33c76ec710acb328a2ce256e12452a3218995ce0`). All **12/12** runs complete 5,120 tasks and 160 groups with matching fixed replicas within topology, zero scaling, nonzero peer transfer and no failures. Infinite and original modes have zero peer-fabric core wait; finite mode has positive core wait.

| Seed | Arm | Original RTT (s) | Infinite-pipe prefetch (s) | Finite-pipe prefetch (s) | Finite core wait / RTT |
|---:|---|---:|---:|---:|---:|
| 9814 | Hand | 9,721.764 | 6,246.717 | 8,381.285 | 10.277% |
| 9814 | MP-OFF | 9,826.358 | 6,364.371 | 8,449.941 | 9.736% |
| 9815 | Hand | 9,857.978 | 7,169.931 | 9,769.577 | 8.768% |
| 9815 | MP-OFF | 9,857.088 | 7,190.525 | 9,762.212 | 8.680% |

The original mode is an anchor, not the finite-pipe counterfactual: prefetch itself changes input overlap. Live finite-versus-infinite RTT differences therefore corroborate pipe contention, while the core share remains mixed. The [raw archive](g32_peer_prefetch_s0_v1/peer_prefetch_s0_2026-09-23.tar.gz) preserves the isolated sources, exact screen, live freeze/audit, all twelve raw results and traces; its 73 members were re-read against their hashes. Archive SHA-256 `5b0620ec95cce3d1bdb567b1f51e1ab193a8882c33d8cb00e4fc2453b5bacaff`; audit SHA-256 `3411dcc1c1fc913db14d0c0877cffc8510c5f606eacc5b56d5232b1f5e62eecb`. No GNN attribution follows from this physics gate.
