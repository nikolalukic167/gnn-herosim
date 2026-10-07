# gnn_seeded_cd_fixed_v1 — GNN-seeded refinement with fixed server replicas

> Produced on local-record-2026-10-06 at afea04bd; code not merged; old physics.

**Status:** `CLOSED` (2026-09-22) — `GNN-SEED-BEATS-HAND-ON-FIXED-REPLICAS`. Registered before the fresh gate; every threshold below was signed before its outcomes.

**Outcome.** On eight fresh eligible w0 topologies, the seed-1 bipartite GNN followed by six-pass coordinate refinement beat the matched hand start on fixed replicas in **7/8 pairs**, with **2.73% median lower mean task elapsed time** (one-sided sign p = 0.0352). Seven wins remain after measured inference time is included. All 16 runs completed 50,000 tasks, shared each pair's replica manifest, and had zero scaling events. The preregistered hand-comparison gate **passes**. The subsequent [matched MP-OFF live ablation](gnn_seeded_cd_mp_ablation_v1.md) beats this GNN seed in **8/8 pairs**, so the useful learned initialization is **not a message-passing win**. Neither result establishes transfer to dynamic autoscaling or another workload window. The [dynamic-autoscaler predecessor](gnn_seeded_cd_v1.md) remains closed negative.

**Parents:** [gnn_seeded_cd_v1](gnn_seeded_cd_v1.md), [joint_burst_v2](joint_burst_v2.md), [peer_greedy_live_v1](peer_greedy_live_v1.md).

## Method and registration

The [runner](../../scripts_cosim/important/gnn_seeded_cd_fixed_v1.py) seeds one compatible platform per server and task type by a fixed SHA-256 ranking, initializes those physical slots, checks that every client can reach a replica for every task type, and returns zero autoscaler scaling decisions. Both arms use the same manifest, peer-group batching, seed-1 `gnnedge0` checkpoint when applicable, exchange weight 2, and six refinement passes. The hand arm starts with the peer-greedy batch plan; the GNN arm starts with the bipartite decoder plan. The first four runs used a `/tmp` prototype of the runner with the same selection and scaling logic; their runner SHA-256 was `8a5ff28b3ea1f156c4902c77f288f7944e580add090e3dd9316509fd61bc58a7`. The tracked runner adds a fail-loud reachability check and a preflight exit after pool construction.

The fresh gate was registered in [the protocol](../../experiments/gnn_seeded_cd_fixed_v1_protocol.json) and [base config](../../experiments/gnn_seeded_cd_fixed_v1_base.json). It selects the first eight topology seeds from 9301 onward that pass *only* the replica-reachability preflight, without reading policy outcomes; it searches at most through 9340 and records every rejection. Each selected seed receives two 50,000-task w0 runs. The primary statistic is the median of eight paired percentage reductions in mean task elapsed time. Success requires at least seven GNN wins, at least 2% median reduction, and at least seven wins after adding measured inference time to each arm. Every pair must complete all tasks, use identical manifests, and have zero scaling events. The scope is this deterministic fixed-replica rule and one burst workload; the eight fresh topologies share one hardware/replica manifest and differ in their generated network topology.

## Record

- [2026-09-22 — MP-OFF supersession](#2026-09-22--mp-off-supersession)
- [2026-09-22 — fresh fixed-replica live gate](#2026-09-22--fresh-fixed-replica-live-gate)
- [2026-09-22 — fresh gate registration](#2026-09-22--fresh-gate-registration)
- [2026-09-22 — opened-topology fixed-pool pilot](#2026-09-22--opened-topology-fixed-pool-pilot)

### 2026-09-22 — MP-OFF supersession

The separately trained seed-1 MP-OFF twin was served through the same fixed pool and six-pass refinement on the eight now-opened gate topologies. It beats this GNN seed in all eight. The graph-attribution result and its audit live in [gnn_seeded_cd_mp_ablation_v1](gnn_seeded_cd_mp_ablation_v1.md); the GNN-versus-hand gate result below remains valid but is no longer a graph-specific positive answer.

### 2026-09-22 — fresh fixed-replica live gate

The preflight selected topology seeds **9303, 9304, 9307, 9308, 9310, 9312, 9313, 9314**. It rejected 9301, 9302, 9305, 9306, 9309, and 9311 for client-to-replica reachability, before either policy ran. Both arms then used the same manifest on each selected seed. The finalized tracked runner reproduced both opened 9106 weight-2 totals bit-exactly before the fresh gate: GNN `132346.20793916783` s; hand `143705.1737818597` s. The runner's separate SHA-256 and every input/checkpoint hash are in the [compact audit](gnn_seeded_cd_fixed_v1/gate_read.json), because the new untracked script is omitted by the historical `git diff HEAD` fingerprint.

| Fresh topology | GNN seed + CD6 | Hand seed + CD6 | GNN elapsed reduction |
|---|---:|---:|---:|
| 9303 | 2.8528 | **2.8457** | −0.25% |
| 9304 | **2.0078** | 2.1122 | +4.94% |
| 9307 | **3.3871** | 3.5671 | +5.04% |
| 9308 | **3.4340** | 3.4626 | +0.83% |
| 9310 | **3.0467** | 3.1132 | +2.13% |
| 9312 | **5.2468** | 5.5308 | +5.14% |
| 9313 | **4.6363** | 4.7955 | +3.32% |
| 9314 | **3.6540** | 3.6969 | +1.16% |

All values are mean elapsed seconds per task over the same 50,000-arrival workload. The median paired reduction is **2.727%**; 7/8 signs favor the GNN (one-sided exact sign p = **0.03515625**), and the inference-inclusive contrast favors the GNN in the same 7/8. All 16 output audits pass: success, 50,000 tasks, node-disk warmth, peer exchange, the checkpoint feature contract, identical manifests within each pair, zero scale events, and one tracked-code fingerprint. The gate therefore meets every preregistered *hand-comparison* success condition. The later MP-OFF ablation above resolves the graph-attribution question negatively for this checkpoint.

### 2026-09-22 — fresh gate registration

The seed search and gate thresholds above were signed after reading the four opened-topology pilot results, but before generating or running any fresh topology. Reachability is the only selection filter. The runner checks it before either policy is measured. The checkpoint, workload, and policy settings are frozen by the protocol. No new training or GPU allocation is part of this gate.

### 2026-09-22 — opened-topology fixed-pool pilot

The original dynamic-autoscaler experiment split two wins and two losses because the policy-dependent replica set could leave an overloaded node as the only candidate. A CPU-only fixed-pool control used one deterministic compatible platform per server and task type, with no scale events. Each pair's manifest SHA-256 was `1c0dbb430a5ce6853d2f30e1e8658e3240b51ff36f307f47354a6e4dc6085663`.

| Topology | GNN seed + CD6 | Hand seed + CD6 | GNN elapsed reduction |
|---|---:|---:|---:|
| 9101 | 3.1525 | 3.2389 | 2.67% |
| 9106 | 2.6469 | 2.8741 | 7.90% |
| 9114 | 3.1862 | 3.3263 | 4.21% |
| 9116 | 3.8122 | 4.0219 | 5.21% |

All values are mean elapsed seconds per task over the same 50,000 arrivals; the four topology pairs are previously opened and not independent test evidence. At weight 1 on 9106 alone, the GNN start also beat the matched hand start (3.417 vs 3.567 s/task), but the stronger weight-2 hand control reached 2.874. The weight-2 GNN start beat that stronger hand control at 2.647. Measured inference was roughly 55 seconds per GNN run versus 7–8 seconds for the hand run, about 0.001 second per task of extra overhead.
