# peer_affinity_v1 — REGISTERED

> **Status:** `REGISTERED` &nbsp;·&nbsp; **Index:** [LINEAGES.md](../../LINEAGES.md) &nbsp;·&nbsp; **Record spans:** 2026-09-09 → (open)

**Outcome.** Not yet measured. This node was written and committed **before**
`scripts_cosim/peer_affinity_probe.py` produced a number, so that the bars below are the
registered ones. Phase 0 is a paper screen: no simulation is run, no physics is built. It ends
GO (some cell passes every bar → build the physics, §A1/A2 of the plan) or NO-GO (status
`FALSIFIED`, the cell table stays here, one entry in `docs/hard-stops.md`).

**Related:** [throughline](throughline.md) (2026-09-09 section — the argument this lineage is
the one untried exception to) · [dag_fabric_contention_v1](dag_fabric_contention_v1.md) (the
paper-screen pattern) · [route_b_env_pivot_v1](route_b_env_pivot_v1.md) (count columns, α caps,
the "void rung" lesson) · [route_b_v1](route_b_v1.md) · [literature_reeval_v1](literature_reeval_v1.md)
(reading vocabulary for the later training registration).

**Entry points:** `scripts_cosim/peer_affinity_probe.py` → `simulation_data/peer_affinity_probe.json`;
paper datasets in the on-disk format for the cross-check under
`simulation_data/peer_affinity_paper_k8/` (not a training corpus; never registered in
`REGISTRY.json`).

## Record

- [peer_affinity_v1 — PHASE 0 REGISTRATION (2026-09-09)](#peer-affinity-v1-phase-0-registration-2026-09-09)
- [Amendment A1 — base-cost rule (2026-09-09, before the screen ran)](#amendment-a1-base-cost-rule-2026-09-09-before-the-screen-ran)

---

### peer_affinity_v1 — PHASE 0 REGISTRATION (2026-09-09)

**Why this is the candidate.** The throughline (2026-09-09) closes every supervised route by
one argument: each cost the simulator charges is indexed by a machine or by the task's own
route, so a pointwise scorer given per-machine counts expresses the label exactly; the one
exception (shared links) is routed around at the optimum. The three things named there as
"what would change the answer" are all environment changes. The untried structure that the
count argument does **not** cover is a cost indexed by **pairs of task instances** with
**no commit order** and **binding capacity** — a quadratic assignment: at task i's input
stage, charge Σ_j x_ij · transfer(node(i), node(j)) for an instance-specific, continuous
exchange volume x_ij. It is not node-indexed (so the composition theorem does not apply),
it is not routable-around (co-location is the only way to zero it, and capacity forbids
co-locating everyone), and a prefix column carries it only for *committed* peers — the
uncommitted part is exactly what a task↔task graph would aggregate. Two traps are designed
out up front: (i) a few discrete "shared dataset" labels make the term a per-(node, label)
count — so x_ij is continuous and per pair; (ii) link contention pipes stay OFF, because
the avoid-or-count shape of `dag_fabric_contention_v1` would otherwise reappear.

**Registered paper model (built from stored `arm_b0` datasets, data-locality OFF).**
- Source: `simulation_data/gnn_datasets_dag4_route_b_pilot_v1_arm_b0`, every 6th dataset
  in sorted order (34 sources), one paper dataset each per cell.
- Base cost c_i(p): the stored per-task duration (`task_times` end − start) of the real task
  the paper task copies, on placement p, **at that placement's earliest dispatch time in the
  sweep** (Amendment A1 below). The sweep total equals the sum of those durations on `arm_b0`
  (checked on ds_00000: 2.4068 = 0.1615 + 0.2075 + 0.1873 + 1.8506). The probe **fails loud**
  if a root task's duration varies at all, or any task's duration varies by more than 5 % at
  a fixed dispatch time (the S0 substrate is then not pointwise and the source is wrong).
- Paper tasks i = 0..k−1 resample the dataset's 4 real tasks with replacement (type, source
  client, candidate placements). Duplicates are intentional: interchangeable co-residents
  are the count-shaped base. Candidates: n_cand placements of the copied task, distinct
  nodes preferred, drawn with the paper seed.
- Per-instance demand: `demand_scale_i ~ U(0.5, 2.0)`;
  `demand_i(p) = demand_scale_i × memoryRequirements[type][platform_type(p)]`
  (`data/nofs-ids/task-types.json`). Caps: the `alpha_max` rule of
  `score_route_b_contention.Dataset.node_caps` — cap_n = α × max demand any candidate places
  on n; feasible = every node under its cap.
- Peer matrix: each task draws `partners` distinct peers; `x_ij = x_scale × 10^U(−1, 1)`
  bytes, symmetrised (first draw wins). One `random.Random(paper_seed)` per paper dataset.
- Exchange cost for one pair on nodes (n, m): exactly what `_payload_transfer_time` +
  `_dependency_transfer_time` would charge — `hops × bytes / (bottleneck_mbps × 1024²)` over
  the stored `link_topology.routes` and `links`, plus `network_maps[n][m]`; 0 when n = m.
- Sharing term (mirrors `--allow-non-unique-replicas`): two batch tasks on one platform
  serialise, so add `q_p × cnt_p (cnt_p − 1) / 2`, q_p = mean base cost of the tasks that list
  p. Count-shaped **by design**; it is there so the count competitor has something real to
  repair and so the paper sweep has the same shape as the simulated one.
- Cost of a plan π: `Σ_i c_i(π_i) + Σ_p q_p cnt_p(cnt_p−1)/2 + Σ_{i<j} x_ij · X(node(π_i), node(π_j))`.
  Every plan of the n_cand^k product is enumerated; statistics are read on the α-feasible
  subset; fits use the full product as fit rows (the scorer's convention).

**Count competitor columns (the `B2` repair set, all pointwise-expressible):** node
occupancy excess `1int`; per-(node, type) counts `kint`; per-platform pair count
`cnt_p(cnt_p−1)/2` (the sharing term's own sufficient statistic); demand-weighted node
load, load²/cap, over-cap load and node-load L2 (the `hetdem` shapes). Widest fit
≈ k·n_cand + n_platforms + n_nodes·n_types + 5 parameters, against ≥ 6.5k rows at the smallest
cell — the `2 × n_params` guard (`docs/lessons.md` 2026-08-27) holds on every cell.

**Registered bars (per cell, over the 34 paper datasets; medians unless stated).**

| Bar | Statistic | Pass |
|---|---|---|
| S0a control | x ≡ 0, **collision-free plans only** (all platforms distinct): pure additive R²; additive-argmin regret | R² ≥ 0.999; regret > 1 % in ≤ 2 % of datasets |
| S0b control | x ≡ 0, full feasible set: R² and argmin regret of additive + count columns | R² ≥ 0.999; regret > 1 % in ≤ 2 % of datasets |
| B0 cap binds | unconstrained optimum infeasible under α | ≥ 50 % of datasets |
| B1 joint gap | regret of the pointwise-only fit's argmin vs the feasible optimum | median ≥ 5 % |
| B2 count repair | (B1 − regret of the additive + count-column fit's argmin) / B1 | median < 0.5 |
| **B3 prefix greedy** | sequential greedy on the **true** marginals with exact prefix (committed peers' exchange charged, uncommitted ignored), task order = id, capacity mask | **median ≥ 5 %** |
| B4 peer-mass fit | B2's fit plus the plan-level peer-mass column Σ_i Σ_j x_ij · mean_{c∈cand(j)} X(node(π_i), node(c)); closure of B1 | median < 0.5 |
| **B5 hand-lookahead greedy** | B3's greedy plus the peer-mass term for uncommitted partners | **median ≥ 2 %** |
| C1 count oracle | regret of a uniformly random plan inside the optimum's per-(platform, type) count stratum — a scorer that knows every count and nothing else | median ≥ 5 % |
| Spread | `--spread-plans-only` (all nodes distinct) only where k ≤ n_server_nodes | reported; **VOID** otherwise, never a pass |

Reported without a bar: QAP share of the optimum's cost; B3/B5 under best-of-8 random task
orders and under "largest exchange first"; greedy-stuck fraction (a stuck greedy is counted
and excluded from the median, never silently dropped); row count vs 2 × n_params.

**Cells.** x_scale ∈ {50, 200, 800} MB × α ∈ {1.5, 2.0} × (k, n_cand) ∈ {(8,3), (8,4), (10,3),
(10,4), (12,3)} × partners ∈ {2, 3} = 60 cells. Link bandwidth is the stored 1000 (the A2 grid's
100 Mbps backbone is, for the transfer part, the 800 MB row at 1000).

**Decision rule.** GO ⇔ at least one cell passes S0a, S0b, B0, B1, B2, B3, B4, B5 and C1. The GO
cell carried to A2 is the passing cell with the smallest x_scale, then the largest k.
NO-GO ⇔ no cell passes ⇒ `FALSIFIED`, the full cell table here, one `docs/hard-stops.md` entry,
nothing built. A cell that passes B3 but fails B5 is written as **"hand lookahead suffices"** —
it is a NO-GO for a graph-reasoning claim, whatever else it passes.

**Cross-check of the instrument (before the reading is trusted).** For the k = 8 cells the probe
also writes each paper dataset in the on-disk format (`infrastructure.json` and
`space_with_network.json` copied from the source, synthetic `workload.json` with one event per
task and a top-level `peer_exchange` list, `placements/placements.jsonl`,
`placement_metadata.json` with `sweep_complete: true`). `separability_diagnostic.analyze_dataset`
must reproduce the probe's full-sweep additive R² and one-integer repair to 1e-6, and
`score_route_b_contention.Dataset` must reproduce the caps and the feasible-row count exactly.
Disagreement is a probe bug, not a finding.

**What a GO buys.** Only the right to build `HEROSIM_PEER_EXCHANGE=1` (plan §A1) and re-read
the same bars on a simulated sweep (§A2). Training is a separate registration, and its
registered prediction is already fixed: `gnn` ≥ +1 pp over `mlp_t1` and over `mpoff`; if
`mlp_t1x` (peer-mass columns) closes the gap the result is "hand lookahead suffices".

---

### Amendment A1 — base-cost rule (2026-09-09, before the screen ran)

The registered guard ("a task's duration must not vary with the other tasks' placements,
1e-6") fired on the first source dataset: on `arm_b0/ds_00000` task 1's duration on one
placement varies by 15.7 % across the sweep, task 2's by 17.9 %, task 3's by 128 %. Measured
before any screen statistic was computed:

| task | relative spread over the sweep | distinct dispatch times on the modal placement | spread at a **fixed** dispatch time | corr(duration, dispatch time) |
|---|---|---|---|---|
| 0 (root) | 0.000 | 1 | 0.000 | — |
| 1 | 0.157 | 4 | 0.000 | +0.32 |
| 2 | 0.179 | 4 | 9.3e-4 | +0.32 |
| 3 (sink) | 1.276 | 20 | 1.65e-2 | −0.92 |

So a child's duration is a function of its **dispatch time** (the platform's warm queue
drains while the parent runs) and of nothing else — the DAG timing effect, present with data
locality off and unrelated to the pairwise structure under test. A parallel batch of
single-task events is dispatched at t = 0, so the paper cost of (task, placement) is the
duration observed at that placement's **earliest** dispatch time in the sweep (mean over
ties). Guards now: a root task's spread must be < 1e-6, any task's spread at the chosen
dispatch time < 5 %; the sum-of-durations = rtt identity is still checked to 1e-6. Both
spreads are written into the report for every source. Nothing else in the registration
changes; no bar moved.
