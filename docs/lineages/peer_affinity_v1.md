# peer_affinity_v1 — ACTIVE

> **Status:** `ACTIVE` &nbsp;·&nbsp; **Index:** [LINEAGES.md](../../LINEAGES.md) &nbsp;·&nbsp; **Record spans:** 2026-09-09 → (open)

**Outcome.** **T1b (2026-09-11) supersedes T1's reading: message passing DOES help on this
environment once the GNN is not data-starved — and corpus size, not architecture, was what T1 was
actually measuring.** At 482 training datasets and 16 seeds, the cleanest contrast in the lineage
(`gnn` vs `mpoff`: same architecture, features, decoder, selector and seeds; the only difference is
whether `PeerConv` runs) is **+5.14 pp, p = 0.001, 13/16 seeds** — up from +2.08 pp, p = 0.15 at T1's
136 datasets. 3.5× data buys `gnn` −5.09 pp but `mpoff` only −2.43 pp. Held-out regret at the honest
selector: `gnn` **18.78 %**, `mlp_t1x` 22.54 %, `mpoff` 23.41 %, `mlp_t1` 29.80 %; every learned arm
beats reactive Knative (116.8 % excess) on the live gate. The environment itself remains the first in
this program whose joint structure defeats a *trained* pointwise scorer: `mlp_t1`, which sees the
exact exchange cost to already-committed peers but not the uncommitted-peer lookahead, loses 11.8 pp
to `gnn` on 16/16 seeds.

**Three qualifications carried with every quote.** (1) The advantage is at the **selected**
checkpoint: MP-ON memorises the training split exactly (train regret 0.00 %) and the two GNN arms
**tie** when both are read at the last epoch (−0.80 pp, p = 0.252). (2) The **live gate agrees in
direction but not in significance** (`gnn` vs `mpoff` +3.78 pp, p = 0.130) — an unconfounded
GNN-over-pointwise claim is established offline and NOT on the live gate. (3) `gnn` vs the MLP arms
carries a **selector asymmetry** favouring the GNN (GNN selects on exact val decode regret, MLP on val
edge accuracy), so those deltas are upper bounds; `gnn` vs `mpoff` carries none.

Upstream: Phase 0 paper screen GO (14 of 60 cells); physics built (`HEROSIM_PEER_EXCHANGE=1`,
bit-identical off); simulated rung **R2 blind on fresh seeds 7018–7034** passed every bar (cap binds
34/34, control count-repairable, pointwise-only regret 28.5 %, count repair 0.27, residual 23.4 %,
hand-lookahead analogue 19.5 %, count oracle 18.9 %). The 50 MB cell failed on the simulator and is
dropped; α₄ = 1.25 fails S0b by two control datasets. T1's own entry is kept below unchanged as the
n = 8 reading it was.

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
- [Amendment A2 — equal-tightness α, shared candidates, readability (2026-09-09, before the screen ran)](#amendment-a2-equal-tightness-α-shared-candidates-readability-2026-09-09-before-the-screen-ran)
- [Amendment A3 — tool: fit-argmin regrets read the tie band (2026-09-09, before the screen ran)](#amendment-a3-tool-fit-argmin-regrets-read-the-tie-band-2026-09-09-before-the-screen-ran)
- [peer_affinity_v1 — PHASE 0 PAPER SCREEN, GO (2026-09-10)](#peer-affinity-v1-phase-0-paper-screen-go-2026-09-10)
- [Amendment A4 — the simulated screen: physics, grid and table analogues of B3/B5 (2026-09-10, before any rung was read)](#amendment-a4-the-simulated-screen-physics-grid-and-table-analogues-of-b3b5-2026-09-10-before-any-rung-was-read)
- [Amendment A5 — R1 did not instantiate the GO cell; reachability raised, C1 void on simulated sweeps (2026-09-10)](#amendment-a5-r1-did-not-instantiate-the-go-cell-reachability-raised-c1-void-on-simulated-sweeps-2026-09-10)
- [Rung R1b — the simulated screen, first readable read (2026-09-10)](#rung-r1b-the-simulated-screen-first-readable-read-2026-09-10)
- [Amendment A6 — count competitor v2, the cap ladder on the simulated substrate, R2 registered blind (2026-09-10)](#amendment-a6-count-competitor-v2-the-cap-ladder-on-the-simulated-substrate-r2-registered-blind-2026-09-10)
- [Rung R2 — the blind read: PIVOT-CANDIDATE (2026-09-10)](#rung-r2-the-blind-read-pivot-candidate-2026-09-10)
- [Training registration T1 (2026-09-10, signed off before any cache or checkpoint exists)](#training-registration-t1-2026-09-10-signed-off-before-any-cache-or-checkpoint-exists)

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

**A1 addendum (same day, still before the screen ran).** The 5 % guard fired too: task 1 on
platforms 125/126 varies by 17 % *at* its earliest dispatch time. Grouped by co-residency
on that node: with no sibling on the node the duration is always 0.1688 s; with sibling task 2
on the same node it is 0.1688 s or 0.1989 s. That is the node-level storage serialisation of
`node_disk_v2` — node-indexed, count-shaped, the one-integer channel of the record — and the
paper model's sharing term stands in for it. Final rule: **c_i(p) = the minimum duration over
the rows at that placement's earliest dispatch time** (the uncontended value; a co-resident
only ever adds time). Guard kept: a root task (dispatched at t = 0, alone) must not vary at
all; the residual spreads are reported per source. Still no bar moved.

---

### Amendment A2 — equal-tightness α, shared candidates, readability (2026-09-09, before the screen ran)

A 3-source development run of one k = 8 cell (not the registered 34 sources; no bar is read
from it) showed two defects in the registration, both of the "instrument cannot fire" kind
that `route_b_env_pivot_v1` warns about:

1. **The registered α ∈ {1.5, 2.0} is a cliff at k ≥ 8.** The `alpha_max` cap is α × the
   largest single demand on a node, the source topologies host candidates on only 2–5 server
   nodes, and k tasks must fit under Σ caps. Feasibility calibration over the 34 registered
   sources (x = 200 MB, partners 2; **feasibility counts only, no bar was computed**):

   | (k, n_cand) | α = 1.5 | 2.0 | 2.5 | 3.0 | 4.0 | 5.0 | 6.0 | 8.0 |
   |---|---|---|---|---|---|---|---|---|
   | (8, 3) — sources with any feasible plan / of those, cap binds | 15 / 12 | 26 / 19 | 30 / 20 | **34 / 23** | **34 / 22** | 34 / 19 | 34 / 3 | 34 / 0 |
   | (8, 4) | 18 / 13 | 31 / 21 | 33 / 23 | **34 / 24** | **34 / 23** | 34 / 14 | 34 / 2 | 34 / 0 |
   | (10, 3) | 13 / 11 | 21 / 19 | 30 / 24 | 30 / 22 | **34 / 22** | **34 / 22** | 34 / 16 | 34 / 1 |
   | (12, 3) | 12 / 9 | 16 / 13 | 21 / 16 | 27 / 21 | 32 / 23 | **34 / 22** | **34 / 20** | 34 / 8 |

   Rule adopted: the cap for k tasks is **α_k = α₄ × k / 4** with the registered
   α₄ ∈ {1.5, 2.0} — the equal-tightness scaling `route_b` s9d and
   `dag_fabric_contention_v1` already used (α₈ = 4.0 for two 4-task instances at α₄ = 2.0).
   Every rung is then readable (32–34 of 34) and the cap binds in 59–72 % of datasets, so B0
   is a live bar, not a foregone one. Cell names keep α₄; the report records α_k. This is
   *not* the hard-stopped "relax α to get a readable rung": α₄ is unchanged, the scaling is
   the fixed per-task-count rule, and it was fixed before the screen ran.
2. **Copies of a real task drew independent candidate subsets**, so co-residents of one type
   were *not* interchangeable and the count-oracle strata had a median of 2 plans — C1 would
   have measured candidate-set identity, not the pairwise term. Candidates are now drawn
   **once per real task** and shared by all its paper copies, which is what the registration's
   "duplicates are intentional" sentence meant.
3. **Readability.** A cell in which fewer than 17 of the 34 sources have a feasible plan is
   **UNREADABLE**: reported, never a pass. (`route_b_env_pivot_v1`: "could not measure" is not
   "nothing there", and not a pass either.)

No bar value moved. Cross-check of the 3-source development run: `separability_diagnostic`
and `score_route_b_contention` agreed with the probe on 3/3 datasets (additive R², argmin
regret, one-integer repair to 1e-6; caps and feasible-row counts exactly).

---

### Amendment A3 — tool: fit-argmin regrets read the tie band (2026-09-09, before the screen ran)

The 6-source development cross-check disagreed on 1 of 6 datasets: both tools fit the same
model (R² equal to all printed digits) but read different argmin regrets (probe 1.46 %,
diagnostic 0.00 %). Cause: an indicator fit predicts the *same* value for plans that swap two
interchangeable copies of a real task (A2 made copies share candidates on purpose), so the
argmin is a tie set and a single argmin is a tie-break artifact — the exact defect
`score_route_b_contention.py`'s `r_exact_band` was written for (2026-08-27: "read
`mean_tied`, never `r_exact_pct` alone"). Every fit-argmin regret in the probe (S0a, S0b, B1,
B2, B4 and the cross-check) now reads the **mean true cost over the tie set** (predictions
within 1e-9 of the minimum), with the optimistic / pessimistic ends recorded alongside. The
cross-check accepts the diagnostic's single-argmin regret if it lies inside the probe's tie
band. Greedy bars (B3, B5) are unaffected: they break exact score ties by the lowest
(node, platform), as `decode_masked_topo_placement` does. No bar value moved.

---

### peer_affinity_v1 — PHASE 0 PAPER SCREEN, GO (2026-09-10)

`scripts_cosim/peer_affinity_probe.py --stages screen emit crosscheck` at commit `aaa2d57`
(the ordering-split fields were added mid-run and are absent from this report's per-dataset
records; they were recomputed separately, table below). 60 cells × 34 sources, 8 925 s on one
process. Report: `simulation_data/peer_affinity_probe.json`. Paper datasets for the two
cross-check cells: `simulation_data/peer_affinity_paper_k8/` (355 MB, untracked).

**Instrument checks first.** Cross-check against `separability_diagnostic.analyze_dataset`
and `score_route_b_contention.Dataset`: **34/34 and 34/34** on `x200_a2.0_k8c3_p2` and
`x200_a2.0_k8c4_p2` (additive R² to 1e-6, single-argmin regrets inside the probe's tie band,
caps and feasible-row counts exact). Source durations: residual spread at the earliest
dispatch time median 0.001, max 0.993 (the same-node storage serialisation of A1's addendum;
the uncontended minimum is used). Widest fit: rows ≥ 49.7 × (2 × n_params) in every cell.
Greedy stuck ≤ 11.8 % of datasets in any cell (counted, excluded from medians). Every cell
readable (34/34 scored; A2's α_k scaling did its job).

**Controls.** S0b (x ≡ 0, full feasible set, additive + count columns): R² = 1.0000 and
argmin regret ≤ 1 % on every dataset of every cell. S0a (x ≡ 0, collision-free plans, pure
additive): R² ≥ 0.99998 wherever it can be read, but it is **structurally void once k
approaches the 9 candidate platforms** — read on 15/34 sources at k = 8 (3 cand), 22/34 at
k = 10 (4 cand), 4/34 at k = 10 (3 cand), 0–1/34 at k = 12. The per-cell table marks it;
`x50_a1.5_k12c3_p2` passes every readable bar with S0a VOID and is **not** counted as a
pass under the registered rule. Spread control: VOID in all cells (k > hosting nodes, which
are 2–5 on these sources, not 6).

**Cell table** (medians over 34 sources; bold = bar passed; B2 shows the repair fraction with
the count-repaired residual regret in brackets; C1 shows the median stratum size in brackets):

| cell | α_k | scored | S0a read (R²) | S0b R² | B0 | B1 % | B2 rep (resid %) | B3 % | B4 clos | B5 % | C1 % (stratum) | QAP@opt | pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| x50_a1.5_k8c3_p2 | 3.0 | 34/34 | 15/34 (1.0000) | 1.0000 | **0.65** | **27.8** | 0.86 (3.8) | **20.8** | 0.86 | **21.3** | **14.8** (8) | 0.35 | no |
| x50_a2.0_k8c3_p2 | 4.0 | 34/34 | 17/34 (1.0000) | 1.0000 | **0.53** | **30.3** | 0.79 (11.0) | **19.2** | 0.79 | **13.3** | **22.3** (8) | 0.24 | no |
| x200_a1.5_k8c3_p2 | 3.0 | 34/34 | 15/34 (1.0000) | 1.0000 | **0.68** | **52.7** | 0.70 (7.9) | **63.9** | 0.70 | **50.2** | **22.6** (6) | 0.57 | no |
| x200_a2.0_k8c3_p2 | 4.0 | 34/34 | 17/34 (1.0000) | 1.0000 | **0.65** | **57.6** | 0.78 (12.6) | **69.2** | 0.78 | **53.7** | **20.6** (6) | 0.41 | no |
| x800_a1.5_k8c3_p2 | 3.0 | 34/34 | 15/34 (1.0000) | 1.0000 | **0.68** | **68.9** | 0.64 (10.5) | **100.7** | 0.64 | **71.4** | **27.2** (5) | 0.70 | no |
| x800_a2.0_k8c3_p2 | 4.0 | 34/34 | 17/34 (1.0000) | 1.0000 | **0.68** | **99.4** | 0.76 (15.3) | **118.6** | 0.76 | **70.7** | **8.9** (4) | 0.59 | no |
| x50_a1.5_k8c3_p3 | 3.0 | 33/34 | 7/33 (1.0000) | 1.0000 | **0.76** | **20.7** | 0.89 (2.9) | **28.2** | 0.89 | **39.2** | **7.9** (4) | 0.39 | no |
| x50_a2.0_k8c3_p3 | 4.0 | 34/34 | 7/34 (1.0000) | 1.0000 | **0.59** | **25.6** | 0.75 (5.1) | **20.0** | 0.75 | **20.0** | **17.4** (3) | 0.30 | no |
| x200_a1.5_k8c3_p3 | 3.0 | 33/34 | 7/33 (1.0000) | 1.0000 | **0.79** | **27.1** | 0.61 (5.8) | **61.7** | 0.61 | **68.1** | **20.9** (4) | 0.66 | no |
| x200_a2.0_k8c3_p3 | 4.0 | 34/34 | 7/34 (1.0000) | 1.0000 | **0.68** | **36.9** | 0.90 (3.8) | **48.0** | 0.90 | **33.3** | **9.4** (3) | 0.55 | no |
| x800_a1.5_k8c3_p3 | 3.0 | 33/34 | 7/33 (1.0000) | 1.0000 | **0.76** | **50.3** | 0.80 (6.0) | **53.2** | 0.80 | **54.3** | **30.0** (4) | 0.79 | no |
| x800_a2.0_k8c3_p3 | 4.0 | 34/34 | 7/34 (1.0000) | 1.0000 | **0.68** | **64.5** | 0.97 (1.6) | **58.7** | 0.97 | **53.5** | **37.9** (3) | 0.69 | no |
| x50_a1.5_k8c4_p2 | 3.0 | 34/34 | 29/34 (1.0000) | 1.0000 | **0.68** | **51.2** | 0.80 (9.0) | **31.6** | 0.80 | **14.3** | **27.2** (13) | 0.23 | no |
| x50_a2.0_k8c4_p2 | 4.0 | 34/34 | 29/34 (1.0000) | 1.0000 | **0.56** | **50.6** | 0.82 (13.8) | **24.4** | 0.82 | **15.9** | **20.5** (12) | 0.18 | no |
| x200_a1.5_k8c4_p2 | 3.0 | 34/34 | 29/34 (1.0000) | 1.0000 | **0.71** | **94.2** | 0.56 (24.4) | **104.4** | 0.56 | **28.3** | **33.8** (8) | 0.36 | no |
| x200_a2.0_k8c4_p2 | 4.0 | 34/34 | 29/34 (1.0000) | 1.0000 | **0.68** | **117.7** | 0.62 (30.7) | **78.7** | 0.62 | **23.0** | **29.1** (8) | 0.26 | no |
| x800_a1.5_k8c4_p2 | 3.0 | 34/34 | 29/34 (1.0000) | 1.0000 | **0.71** | **107.3** | **0.45** (33.7) | **157.0** | **0.45** | **104.7** | **54.1** (7) | 0.64 | **GO** |
| x800_a2.0_k8c4_p2 | 4.0 | 34/34 | 29/34 (1.0000) | 1.0000 | **0.68** | **101.7** | **0.44** (50.3) | **127.0** | **0.44** | **91.7** | **50.2** (9) | 0.57 | **GO** |
| x50_a1.5_k8c4_p3 | 3.0 | 33/34 | 29/33 (1.0000) | 1.0000 | **0.61** | **35.5** | 0.74 (6.2) | **21.0** | 0.74 | **15.9** | **10.2** (10) | 0.25 | no |
| x50_a2.0_k8c4_p3 | 4.0 | 34/34 | 30/34 (1.0000) | 1.0000 | **0.50** | **45.5** | 0.90 (4.2) | **20.7** | 0.90 | **16.1** | **10.8** (10) | 0.16 | no |
| x200_a1.5_k8c4_p3 | 3.0 | 33/34 | 29/33 (1.0000) | 1.0000 | **0.61** | **61.2** | 0.66 (23.2) | **72.7** | 0.66 | **31.1** | 0.0 (9) | 0.53 | no |
| x200_a2.0_k8c4_p3 | 4.0 | 34/34 | 30/34 (1.0000) | 1.0000 | **0.56** | **64.8** | 0.84 (14.3) | **71.2** | 0.84 | **39.8** | **23.1** (8) | 0.36 | no |
| x800_a1.5_k8c4_p3 | 3.0 | 33/34 | 29/33 (1.0000) | 1.0000 | **0.67** | **87.1** | 0.61 (20.5) | **76.9** | 0.61 | **70.3** | 0.0 (6) | 0.75 | no |
| x800_a2.0_k8c4_p3 | 4.0 | 34/34 | 30/34 (1.0000) | 1.0000 | **0.65** | **99.3** | 0.90 (14.2) | **100.6** | 0.86 | **75.5** | **30.6** (6) | 0.61 | no |
| x50_a1.5_k10c3_p2 | 3.8 | 34/34 | 4/34 (1.0000) | 1.0000 | **0.62** | **36.7** | **0.31** (17.6) | **49.2** | **0.31** | **30.8** | **27.1** (17) | 0.29 | **GO** |
| x50_a2.0_k10c3_p2 | 5.0 | 34/34 | 4/34 (1.0000) | 1.0000 | **0.59** | **35.4** | 0.51 (15.5) | **19.4** | 0.51 | **16.4** | **27.8** (24) | 0.21 | no |
| x200_a1.5_k10c3_p2 | 3.8 | 34/34 | 4/34 (1.0000) | 1.0000 | **0.65** | **70.6** | **0.49** (11.5) | **98.5** | **0.49** | **53.9** | **53.6** (13) | 0.43 | **GO** |
| x200_a2.0_k10c3_p2 | 5.0 | 34/34 | 4/34 (1.0000) | 1.0000 | **0.65** | **89.3** | 0.68 (13.0) | **68.3** | 0.68 | **45.5** | **60.0** (10) | 0.36 | no |
| x800_a1.5_k10c3_p2 | 3.8 | 34/34 | 4/34 (1.0000) | 1.0000 | **0.71** | **141.4** | 0.61 (20.6) | **150.4** | 0.61 | **92.2** | **69.6** (8) | 0.55 | no |
| x800_a2.0_k10c3_p2 | 5.0 | 34/34 | 4/34 (1.0000) | 1.0000 | **0.71** | **152.9** | 0.67 (17.1) | **186.5** | 0.67 | **86.6** | **74.1** (8) | 0.54 | no |
| x50_a1.5_k10c3_p3 | 3.8 | 34/34 | 2/34 (1.0000) | 1.0000 | **0.79** | **20.7** | 0.77 (5.7) | **44.6** | 0.77 | **28.5** | **24.5** (18) | 0.38 | no |
| x50_a2.0_k10c3_p3 | 5.0 | 34/34 | 2/34 (1.0000) | 1.0000 | **0.53** | **27.8** | 0.87 (4.8) | **33.4** | 0.87 | **25.1** | **17.1** (11) | 0.26 | no |
| x200_a1.5_k10c3_p3 | 3.8 | 34/34 | 2/34 (1.0000) | 1.0000 | **0.76** | **38.9** | 0.82 (8.2) | **90.9** | 0.82 | **58.5** | **36.0** (14) | 0.65 | no |
| x200_a2.0_k10c3_p3 | 5.0 | 34/34 | 2/34 (1.0000) | 1.0000 | **0.56** | **36.1** | 0.89 (8.6) | **75.5** | 0.89 | **51.9** | **26.9** (7) | 0.48 | no |
| x800_a1.5_k10c3_p3 | 3.8 | 34/34 | 2/34 (1.0000) | 1.0000 | **0.79** | **50.1** | **0.47** (16.0) | **114.0** | **0.47** | **97.2** | **48.4** (12) | 0.78 | **GO** |
| x800_a2.0_k10c3_p3 | 5.0 | 34/34 | 2/34 (1.0000) | 1.0000 | **0.59** | **72.0** | 0.85 (12.8) | **113.9** | 0.85 | **94.6** | **44.4** (7) | 0.73 | no |
| x50_a1.5_k10c4_p2 | 3.8 | 34/34 | 25/34 (1.0000) | 1.0000 | **0.65** | **54.3** | **0.43** (23.3) | **42.3** | **0.43** | **19.6** | **27.6** (34) | 0.20 | **GO** |
| x50_a2.0_k10c4_p2 | 5.0 | 34/34 | 25/34 (1.0000) | 1.0000 | 0.41 | **47.4** | 0.75 (10.2) | **24.2** | 0.75 | **14.3** | **27.2** (36) | 0.14 | no |
| x200_a1.5_k10c4_p2 | 3.8 | 34/34 | 25/34 (1.0000) | 1.0000 | **0.71** | **99.3** | **0.35** (50.8) | **88.0** | **0.35** | **45.2** | **60.0** (26) | 0.34 | **GO** |
| x200_a2.0_k10c4_p2 | 5.0 | 34/34 | 25/34 (1.0000) | 1.0000 | **0.62** | **94.9** | 0.56 (31.4) | **73.2** | 0.56 | **44.8** | **49.2** (21) | 0.21 | no |
| x800_a1.5_k10c4_p2 | 3.8 | 34/34 | 25/34 (1.0000) | 1.0000 | **0.74** | **153.5** | **0.26** (68.9) | **172.1** | **0.26** | **97.2** | **77.7** (26) | 0.64 | **GO** |
| x800_a2.0_k10c4_p2 | 5.0 | 34/34 | 25/34 (1.0000) | 1.0000 | **0.71** | **177.3** | **0.42** (50.2) | **208.1** | **0.42** | **107.4** | **93.2** (12) | 0.44 | **GO** |
| x50_a1.5_k10c4_p3 | 3.8 | 34/34 | 22/34 (1.0000) | 1.0000 | **0.62** | **48.2** | 0.65 (10.6) | **33.9** | 0.65 | **18.0** | **13.4** (27) | 0.25 | no |
| x50_a2.0_k10c4_p3 | 5.0 | 34/34 | 22/34 (1.0000) | 1.0000 | **0.53** | **48.4** | 0.77 (9.8) | **30.1** | 0.77 | **19.3** | **12.6** (24) | 0.20 | no |
| x200_a1.5_k10c4_p3 | 3.8 | 34/34 | 22/34 (1.0000) | 1.0000 | **0.65** | **69.1** | **0.42** (23.7) | **104.8** | **0.42** | **45.9** | **26.5** (22) | 0.50 | **GO** |
| x200_a2.0_k10c4_p3 | 5.0 | 34/34 | 22/34 (1.0000) | 1.0000 | **0.62** | **81.1** | 0.74 (17.8) | **89.5** | 0.74 | **49.3** | 0.0 (11) | 0.31 | no |
| x800_a1.5_k10c4_p3 | 3.8 | 34/34 | 22/34 (1.0000) | 1.0000 | **0.68** | **92.8** | **0.45** (29.7) | **141.3** | **0.43** | **80.7** | **32.7** (22) | 0.78 | **GO** |
| x800_a2.0_k10c4_p3 | 5.0 | 34/34 | 22/34 (1.0000) | 1.0000 | **0.68** | **125.2** | 0.76 (15.6) | **162.2** | 0.76 | **90.2** | 0.0 (11) | 0.62 | no |
| x50_a1.5_k12c3_p2 | 4.5 | 34/34 | VOID | 1.0000 | **0.62** | **44.3** | **0.42** (24.4) | **32.2** | **0.42** | **25.5** | **41.8** (98) | 0.23 | GO except S0a VOID |
| x50_a2.0_k12c3_p2 | 6.0 | 34/34 | VOID | 1.0000 | 0.41 | **45.9** | 0.54 (23.5) | **35.1** | 0.54 | **20.9** | **28.1** (66) | 0.19 | no |
| x200_a1.5_k12c3_p2 | 4.5 | 34/34 | VOID | 1.0000 | **0.68** | **95.5** | 0.53 (26.2) | **89.2** | 0.53 | **87.3** | **72.7** (38) | 0.40 | no |
| x200_a2.0_k12c3_p2 | 6.0 | 34/34 | VOID | 1.0000 | **0.59** | **91.3** | 0.69 (30.0) | **78.7** | 0.69 | **48.3** | **66.1** (32) | 0.35 | no |
| x800_a1.5_k12c3_p2 | 4.5 | 34/34 | VOID | 1.0000 | **0.76** | **214.6** | 0.63 (32.2) | **130.2** | 0.63 | **158.0** | **102.7** (28) | 0.64 | no |
| x800_a2.0_k12c3_p2 | 6.0 | 34/34 | VOID | 1.0000 | **0.68** | **247.1** | 0.78 (22.1) | **140.1** | 0.78 | **139.7** | **44.0** (12) | 0.52 | no |
| x50_a1.5_k12c3_p3 | 4.5 | 34/34 | 1/34 (1.0000) | 1.0000 | **0.76** | **30.7** | 0.62 (10.5) | **30.5** | 0.62 | **24.1** | **33.2** (59) | 0.38 | no |
| x50_a2.0_k12c3_p3 | 6.0 | 34/34 | 1/34 (1.0000) | 1.0000 | **0.50** | **19.3** | 0.73 (9.3) | **13.7** | 0.73 | **8.1** | **24.5** (48) | 0.27 | no |
| x200_a1.5_k12c3_p3 | 4.5 | 34/34 | 1/34 (1.0000) | 1.0000 | **0.74** | **31.3** | **0.24** (21.7) | **65.3** | **0.24** | **66.2** | **32.8** (16) | 0.62 | **GO** |
| x200_a2.0_k12c3_p3 | 6.0 | 34/34 | 1/34 (1.0000) | 1.0000 | **0.59** | **32.8** | **0.40** (18.5) | **28.2** | **0.40** | **29.4** | **39.3** (26) | 0.45 | **GO** |
| x800_a1.5_k12c3_p3 | 4.5 | 34/34 | 1/34 (1.0000) | 1.0000 | **0.74** | **45.3** | **0.38** (35.5) | **82.3** | **0.38** | **57.3** | **48.2** (14) | 0.72 | **GO** |
| x800_a2.0_k12c3_p3 | 6.0 | 34/34 | 1/34 (1.0000) | 1.0000 | **0.65** | **61.7** | 0.53 (29.5) | **71.9** | 0.52 | **21.1** | **55.2** (13) | 0.62 | no |

**Reading.**
- *What passes always:* B1 (pointwise-only fit argmin regret 19–247 %), B3 (prefix greedy
  14–208 %), B5 (hand-lookahead greedy 8–158 %) pass in all 60 cells; B0 (cap binds) in 58/60
  (fails only at α₄ = 2.0 with x = 50 MB at k = 10 c = 4 and k = 12); C1 (count oracle) in
  56/60 — the four failures are 3-partner α = 1.5 cells where the optimum's (platform, type)
  stratum has 1–11 members, i.e. counts nearly pin the plan.
- *What decides:* **B2/B4**, the count-column repair of the pointwise gap. Median repair ranges
  0.24–0.97 across cells; 14 cells are below 0.5. The passes cluster where the pairwise term is
  sparse and the cap is tight: α₄ = 1.5 (11 of 14), 2 partners (8 of 14), 4 candidates
  (7 of 14). With 3 partners per task the peer graph is dense enough that "how many of my
  type sit on node n" predicts most of the exchange, and counts repair 0.6–0.9 again. The
  peer-mass lookahead column (B4) adds **nothing** beyond counts in any cell (B2 = B4 to
  ≤ 0.03), so a plan-level hand lookahead is not the missing feature.
- *Ordering split for the passing cells* (medians, id order is the registered B3/B5; %):

| cell | id order B3 / B5 | largest-exchange-first B3 / B5 | best of 8 random orders B3 / B5 | oracle over all 10 orders B5 (frac ≤ 2 %) |
|---|---|---|---|---|
| x800_a1.5_k8c4_p2 | 157.0 / 104.7 | 78.5 / 26.1 | 26.0 / 6.5 | 0.0 (0.53) |
| x800_a2.0_k8c4_p2 | 127.0 / 91.7 | 36.9 / 9.7 | 17.3 / 3.9 | 0.0 (0.56) |
| x200_a2.0_k8c4_p2 | 78.7 / 23.0 | 34.3 / 11.4 | 11.4 / 1.3 | 0.0 (0.56) |
| x50_a2.0_k8c3_p2 | 19.2 / 13.3 | 5.5 / 2.1 | 1.9 / 1.8 | 0.0 (0.71) |
| x50_a1.5_k10c3_p2 | 49.2 / 30.8 | 22.0 / 8.3 | 3.2 / 3.2 | 1.8 (0.53) |
| x200_a1.5_k10c3_p2 | 98.5 / 53.9 | 49.5 / 39.2 | 10.5 / 12.3 | 6.3 (0.38) |
| x50_a1.5_k10c4_p2 | 42.3 / 19.6 | 24.1 / 9.3 | 5.2 / 3.1 | 0.3 (0.56) |
| x200_a1.5_k10c4_p3 | 104.8 / 45.9 | 29.8 / 17.1 | 12.1 / 10.8 | 0.0 (0.53) |

  A deterministic hand ordering with the hand lookahead still leaves 8–39 % on the table in
  every GO cell, so the B5 pass is not an artifact of the id order. An **oracle over ten
  orderings** — a multi-start greedy that evaluates each complete plan by its true cost and
  keeps the best — reaches the optimum in about half the datasets (median 0.0–1.8 % at k = 8
  and k = 10 c = 4; 1.8–6.3 % at k = 10 c = 3). That is a *search* competitor: it needs a
  plan-level cost model, which neither the tabular MLP nor the GNN's sequential decode has.
  It is recorded because a reviewer will ask, and because any training registration must
  give **both** arms the same decoder and the same ordering.
- *Magnitude:* the exchange term is 20–29 % of the optimum's cost in the x = 50 MB GO cells
  and 0.5–0.8 at x = 800 MB; the x = 50 MB cells are the ones where base physics still matters,
  which is why the rule prefers them.

**Decision.** GO by the registered rule (≥ 1 cell passes S0a, S0b, B0, B1, B2, B3, B4, B5, C1).
Tie-break inside the rule (smallest x, then largest k) leaves `x50_a1.5_k10c3_p2` and
`x50_a1.5_k10c4_p2`; the 3-candidate cell is carried because its count-repair margin is the
larger (0.31 vs 0.43) and its sweep is 20× smaller (59 049 vs 1 048 576 plans per dataset,
the size A2 must simulate); the 4-candidate cell is the registered backup. What A2 inherits
verbatim: k = 10 single-task events, 3 candidates per task (per type, shared across copies),
2 partners per task, x_ij = 50 MB × 10^U(−1, 1), demand_scale ~ U(0.5, 2.0), α₁₀ = 3.75 under
the `alpha_max` rule, the **stored 1000-Mbps backbone** (the paper screen used the sources'
own fabric; the plan's 100-Mbps grid would multiply the transfer part by 10 and is *not* the
GO cell), contention pipes OFF.

**Caveats carried forward.** (i) The deciding bar is near its threshold and the pass set is a
cluster, not a plateau — A2 reads the same bars on simulated RTTs, and a simulated count repair
≥ 0.5 in the GO cell is a NO-GO for training, whatever the paper number says. (ii) S0a cannot
be read at k = 10 c = 3 (4/34); S0b is the operative control there. (iii) The paper base cost
is the *uncontended* duration; the simulated sweep will carry the node-level storage
serialisation as well as the platform sharing term — both count-shaped, both inside S0b's
repair set. (iv) Nothing here says a learned scorer *will* exploit the structure; it says a
correctly-specified pointwise-plus-counts scorer cannot express it, which is the precondition
every earlier lineage failed.

---

### Amendment A4 — the simulated screen: physics, grid and table analogues of B3/B5 (2026-09-10, before any rung was read)

**Physics built** (commit `6d52571`, plan §A1): `HEROSIM_PEER_EXCHANGE=1` charges, at task i's
input stage, `Σ_j hops × x_ij / (bottleneck × 1024²) + network_map latency` for every peer j on
another node (`Platform._peer_exchange_time`, next to `_dependency_transfer_time`); co-located
peers are free; a peer that is neither placed nor planned fails loud. The trace carries a
top-level `peer_exchange` list of `[i, j, bytes]` over global task ids; the orchestrator
symmetrises it. Because the determined scheduler enqueues each batch member right after
assigning it, a pre-pass sets `task.planned_node_name` for the whole forced batch under the
flag. Flag off or no table ⇒ bit-identical: the stored `arm_b0` optimum replays to 1e-9 with
the flag unset and set (`tests/test_peer_exchange_replay.py`), and the whole test directory
passes. The per-row `peer_exchange_total` is retained with `HEROSIM_RETAIN_PEER_STATS=1`.

**Grid** `peer_affinity_screen` (`generate_gnn_datasets_fast.py`): the paper sources' own
topology/queue grid (conn 0.25/0.35, 6 server nodes, the three queue distributions, 1000-Mbps
backbone, server mesh), k = 10 single-task events cycling `dnn1, dnn2, rf, cnn` (a `dag` is
keyed by type, so a batch is k events; events are emitted grouped by application because
co-sim regroups them before assigning ids — `flatten_workloads` now fails loud if the order
would change), per-event `demand_scale ~ U(0.5, 2)`, 2 partners per task at
`50 MB × 10^U(−1, 1)`, one replica per type per hosting node with types sharing hosts
(`replica_overlap`), seeds 7001–7017, contention pipes OFF. Run with `--num-tasks 10
--allow-non-unique-replicas`; the sweep is the full Cartesian product of the tasks' candidate
replicas. Treated arm `HEROSIM_PEER_EXCHANGE=1`, control arm unset, same seeds.
Hosting fraction **0.67 (4 nodes)**: at 0.5 the 3-dataset pilot gave 3 hosts of which one
offers only `pynqFpga`, so three of the four types had 2 candidates and all ten tasks queued
on the same two platforms (sweeps 2 304–3 456 rows). No bar was read on that pilot.

**Reading a rung** — `peer_affinity_probe.py --from-simulated TREATED --control CONTROL`,
datasets paired by name. Same bars, same thresholds, same tie-band and readability rules,
with these definitions fixed now:
- *Physics agreement (a gate, not a bar):* the exchange term recomputed offline from the
  stored workload and infrastructure must equal the retained `peer_exchange_total` on every
  row (pilot: max relative gap 4e-16). Also reported: the exchange's share of
  `rtt_treated − rtt_control` (pilot ≈ 0.4; the rest is the queue knock-on of longer input
  stages on shared platforms — count-shaped, inside S0b's repair set).
- *S0a / S0b* are read on the **control arm's** rtt (the x ≡ 0 physics is a separate corpus
  here, not a switch). S0a is VOID where no collision-free plan exists.
- *B0, B1, B2, B4, C1* exactly as registered, on the treated rtt; the peer-mass column uses
  the actual bytes.
- *B3′ and B5′* (the greedy bars have no analytic marginal on a simulated sweep): sequential
  greedy on the **true table** with an exact prefix and the capacity mask. B3′ scores
  candidate c for task i by the row in which the prefix and (i, c) are fixed and every
  uncommitted task sits on its lowest-id candidate (committed peers' exchange and sharing
  charged exactly; the future is a fixed default). B5′ scores by the **mean over all
  completions** of the uncommitted tasks (the table's own lookahead). Bars unchanged
  (B3′ ≥ 5 %, B5′ ≥ 2 %); the ordering split is reported as before. These are decoders on
  the true cost, i.e. upper bounds on what a sequential pointwise scorer with the same
  information could do; they are not learned models.
- *Rung R1*: 12 datasets per arm locally (seeds 7001–7017 × the first grid cells), read at
  α₄ ∈ {1.5, 2.0}; the full 102-dataset grid follows on datalab only if R1 reads. A skipped
  dataset (`MAX_PLACEMENT_COMBINATIONS_SKIP`) is reported with its attribution, never dropped
  silently — the skip threshold tests the pre-uniqueness product (memory), so it is set from
  the measured candidate counts, not from sweep sizes.

---

### Amendment A5 — R1 did not instantiate the GO cell; reachability raised, C1 void on simulated sweeps (2026-09-10)

**R1 (12 datasets per arm, `peer_affinity_screen`, local; below the 17-dataset readability
rule by design — an instantiation check, not a read).** Every sweep complete, physics
agreement max relative gap 4.8e-16, exchange share of `rtt_treated − rtt_control` 0.46.
But the instance is not the registered GO cell: with 4 hosting nodes and client
connectivity 0.25/0.35 the source client of a task reaches a **median of 2** hosts
(sweeps 1 024–3 456 rows), and the exchange is **12 %** of the optimum's cost (paper cell
29 %). On that smaller instance the pointwise-only fit lands within 6.4 % (α₄ = 1.5) /
3.8 % (α₄ = 2.0) of the optimum with 5 of 12 datasets already at 0 %, count repair is
bimodal (0 or 1 per dataset; medians 0.26 / 0.98), B3′ 5.0 / 3.5 %, B5′ 5.5 / 0.1 %, and
**C1 is void**: the optimum's (platform, type) stratum has size 1 in every dataset, because
copies of a type have *different* reachable candidate sets and are therefore not
interchangeable — the count oracle then names the plan and measures nothing. On a
simulated sweep C1 is reported but is not a bar; the count **fit** (B2) is the operative
count control. S0b on the control arm: R² 0.992–0.9999, argmin regret ≤ 1 % in 12/12.

**Correction, before the readable rung.** Two grid variants that keep every other key:
`peer_affinity_screen_c3` (connectivity 0.6, replicas on 0.9 of the server nodes) gives a
median of 3 candidates per task (range 2–5 on a 4-dataset probe; sweeps 7.8k–31k rows), i.e.
the GO cell's shape; `peer_affinity_screen_c3_x200` is the same with 200 MB, the registered
backup cell (`x200_a1.5_k10c3_p2` also passed on paper, count repair 0.49) — read because
the simulator's base costs (deep warm queues, 12–40 s per batch) are larger than the paper
sources' uncontended durations, so 50 MB carries a smaller share here. **Rung R1b: 34
datasets per arm** (control, 50 MB, 200 MB), α₄ ∈ {1.5, 2.0}, read with A4's definitions.
The decision rule for training is unchanged: the treated cell must pass B2 (< 0.5) with S0b
on its control, or the simulated screen is a NO-GO for training.

---

### Rung R1b — the simulated screen, first readable read (2026-09-10)

34 datasets per arm (`peer_affinity_screen_c3` control and 50 MB; `_c3_x200` 200 MB), seeds
7001–7017, every sweep complete (median 10 944 rows, 2.5 candidates per task median, 2–5
range), physics agreement max relative gap 6e-16, exchange share of `rtt_treated −
rtt_control` 0.53. Reports: `simulation_data/peer_affinity_r1b_{x50,x200}_read.json`
(registered v1 count columns) and `..._read_v2.json` (A6's v2 columns).

| cell (α₄) | B0 binds | B1 % | B2 rep (resid %) v1 → v2 | B3′ % | B5′ % (big-first) | C1 % | S0b R² / >1 % (v1 → v2) | pass |
|---|---|---|---|---|---|---|---|---|
| 50 MB (1.5) | 0.41 | 3.3 | 0.01 (2.8) → 0.40 (0.4) | 11.1 | 1.3 (0.4) | 1.0 | 0.9989 / 5 of 34 → 1.0000 / 1 of 34 | no: B0, B1, B5, C1 |
| 50 MB (2.0) | 0.15 | 2.5 | 0.01 (3.4) → 0.01 (3.8) | 9.6 | 0.4 (0.1) | 0.0 | same | no: B0, B1, B5, C1 |
| 200 MB (1.5) | 0.32 | **14.5** | **0.16 (17.3) → 0.11 (16.5)** | **38.8** | **15.8 (5.8)** | **15.9** | 0.9989 / 5 of 34 → 1.0000 / 1 of 34 | no: **B0, S0b (by one dataset)** |
| 200 MB (2.0) | 0.18 | **17.1** | **0.24 (17.4) → 0.20 (13.7)** | **33.0** | **6.4 (5.2)** | **14.9** | same | no: B0, S0b (by one) |

**Reading.** (i) The **primary cell (50 MB) does not reproduce**: on the simulated substrate
the exchange is 15–16 % of the optimum (paper 29 %) because the simulator's base costs
(deep warm queues, 12–40 s per batch) dwarf the paper sources' uncontended durations, and
the pointwise fit already lands within 3 %. (ii) The **registered backup cell (200 MB)
passes every treated bar**: B1 14–17 %, count repair 0.11–0.24 with a count-repaired
residual of 14–17 %, B4 = B2 (peer mass adds nothing), B3′ 33–39 %, B5′ 6–16 % (5–6 % under
the largest-exchange-first order), C1 15–16 % with strata of median size 2. (iii) It fails
two **controls**: the cap binds in 18–32 % of datasets (bar 50 %) — the equal-tightness
scaling reproduced the *paper* binding fraction, not this substrate's demand geometry —
and S0b misses by **one dataset** (2.35 % regret on the control arm; the bar allows 0 of 34
above 1 %). (iv) With the registered v1 count columns the control arm read R² 0.9875–0.9999,
which is a **column-set** shortfall, not non-count structure: the simulator's serialisation
on a shared platform is a symmetric function of that platform's co-resident multiset, which
v1's node × type + platform-pair columns do not span. Adding per-(platform, type) counts
and their squares (v2, A6) takes the control to R² 1.0000 median (min 0.9973) and moves the
treated count repair at 200 MB from 0.16 to 0.11 — the pairwise term survives the stronger
competitor. (v) Ordering: at 200 MB / α₄ 1.5 the deterministic hand rule with lookahead
keeps 5.8 %; the oracle over ten orders reaches 0.0 % (a plan-cost search, as on paper).

**Exploratory look (not a read; seeds 7001–7017 are therefore excluded from R2):** at
α₄ ∈ {1.0, 1.25} (α₁₀ = 2.5 / 3.125) the 200 MB rung passes **every** bar including both
controls — B0 0.88 / 0.65, S0b 0 of 34 above 1 %, B1 15.1 / 20.0 %, count repair 0.15 / 0.11
(residual 10.3 / 13.4 %), B3′ 40 / 39 %, B5′ 13.5 / 16.5 % (11.4 / 11.7 % under the best
deterministic order), C1 6.0 / 17.1 %. Reports in the session scratchpad only.

---

### Amendment A6 — count competitor v2, the cap ladder on the simulated substrate, R2 registered blind (2026-09-10)

1. **Count competitor v2** is the registered competitor for simulated screens: v1's block
   plus per-(platform, type) counts, their squares, and the platform occupancy square (the
   second-order sufficient statistics of any symmetric function of a platform's co-resident
   multiset — what the simulator's serialisation is). Adequacy is judged on the control arm
   (S0b), exactly as registered; v1 failed it by column set, v2 passes it. Every treated
   bar is read against v2. (`--count-competitor v2`, default for `--from-simulated`.)
2. **Cap ladder.** A2's equal-tightness rule preserved the paper substrate's binding
   fraction; on the simulated substrate α₄ ∈ {1.5, 2.0} binds in 18–41 % of datasets (bar
   50 %). The ladder for the simulated screen is **α₄ ∈ {1.0, 1.25}** (α₁₀ = 2.5 / 3.125),
   which binds in 65–88 % on R1b. Chosen after an exploratory look at R1b's treated bars,
   so **R1b's seeds are excluded from the registered read**.
3. **R2, registered before it exists:** grid `peer_affinity_screen_c3_x200_r2` — identical
   to `_c3_x200` with **fresh seeds 7018–7034** (34 datasets per arm: control and 200 MB),
   read with A4's definitions, v2 columns, α₄ ∈ {1.0, 1.25}. Decision rule unchanged: R2
   is a PIVOT-CANDIDATE for training iff a cell passes S0b, B0, B1, B2, B3′, B4, B5′ and C1
   (S0a and the spread control are structurally VOID here and are reported as such);
   otherwise the simulated screen is a NO-GO for training and the node closes with R1b's
   table as the outcome. The 50 MB cell is **dropped**: it fails B1 at every cap.

---

### Rung R2 — the blind read: PIVOT-CANDIDATE (2026-09-10)

Grid `peer_affinity_screen_c3_x200_r2`, **fresh seeds 7018–7034**, 34 datasets per arm
(control 53.5 min, treated 69.7 min locally, 28 workers), every sweep complete, median
27 648 rows, 3 candidates per task median, physics agreement max relative gap 6.8e-16,
exchange share of `rtt_treated − rtt_control` 0.53, exchange 30–31 % of the optimum's cost.
Read once, with A4/A6's definitions (`--count-competitor v2 --alphas 1.0,1.25`).
Report: `simulation_data/peer_affinity_r2_read.json`.

| bar | α₄ = 1.0 (α₁₀ = 2.5) | α₄ = 1.25 (α₁₀ = 3.125) |
|---|---|---|
| S0a (collision-free additive) | VOID (no collision-free plan) | VOID |
| S0b control: R² median / min; datasets > 1 % regret | **1.0000 / 0.9964; 0 of 34** | 1.0000 / 0.9964; **2 of 34 (fail)** |
| B0 cap binds | **1.00** | **0.79** |
| B1 pointwise-only regret (median; frac > 5 %) | **28.5 % (0.88)** | **31.2 % (0.97)** |
| B2 count repair (median; frac < 0.5); residual | **0.27 (0.65); 23.4 %** | **0.32 (0.71); 22.8 %** |
| B4 peer-mass closure | 0.27 (= B2) | 0.32 (= B2) |
| B3′ prefix greedy (id / largest-first / best of 8) | **32.9 / 27.5 / 10.3 %** (stuck 23.5 %) | **37.2 / 24.0 / 11.4 %** (stuck 2.9 %) |
| B5′ expected-completion greedy (id / largest-first / best of 8) | **19.5 / 17.3 / 4.2 %** (stuck 11.8 %) | **17.5 / 9.7 / 3.9 %** (stuck 2.9 %) |
| C1 count oracle (stratum size median) | **18.9 % (2)** | **28.5 % (3)** |
| verdict (A6 rule) | **PASS — PIVOT-CANDIDATE** | fails S0b |

**Reading.** The α₄ = 1.0 cell passes every registered bar on seeds never looked at, with
margins: the pointwise fit is 28.5 % off the optimum, the strongest count competitor the
record has (v2: node × type, platform × type and their squares, demand-weighted loads)
repairs only 27 % of that gap and leaves 23 % on the table, a plan-level hand lookahead
adds nothing beyond counts, the sequential decoders on the true cost table lose 17–33 %
under any deterministic order, and even a ten-order search keeps 4 %. On the control arm
the same competitor is exact. This is the shape every previous lineage failed to produce:
joint structure that is neither node-indexed nor routable-around. Caveats carried: the
greedy analogue is stuck (no cap-feasible candidate at some step) in 23.5 % of datasets
at the tight cap — a decoder with a capacity mask needs backtracking or a relaxation there,
which the training registration must specify; C1 strata are small (median 2) because
copies differ by reachable candidates, so C1 is a weak oracle here and the count *fit* is
the operative control; α₄ = 1.25 fails S0b by two datasets and is not carried.

**What this does and does not say.** It says a correctly specified pointwise-plus-counts
scorer *cannot express* this environment's label, which was the precondition every earlier
route lacked. It does not say a graph model will learn it: that is the training question,
registered separately with both arms on one cache, one decoder, one ordering, honest
checkpoint selection and a convergence check (plan §"On PIVOT-CANDIDATE"; readings
GNN-NEEDED / TIE / POINTWISE-BETTER / INDETERMINATE). The registered prediction stands:
`gnn` ≥ +1 pp over `mlp_t1` and `mpoff`; if `mlp_t1x` closes it, "hand lookahead
suffices". No live-serving claim is implied (`MAX_BATCH_SIZE_FOR_GNN = 4`; separate work).

---

### Training registration T1 (2026-09-10, signed off before any cache or checkpoint exists)

**Question.** On the R2 environment, does a graph model with message passing over
task↔task peer edges place better than a pointwise scorer given every count and prefix
column it can compute — with both trained on one cache, decoded by one decoder in one
order, selected by one honest rule?

**Corpus.** Training: grid `peer_affinity_screen_c3_x200_train`, **136 fresh seeds
7035–7170**, treated arm only (`HEROSIM_PEER_EXCHANGE=1`), same physics and knobs as R2.
Held-out: **R2's 34 datasets** (seeds 7018–7034; no model has seen them; their use in the
environment decision is not model selection). Validation: a fixed split artifact over the
136 training datasets (≈ 20 %), written once and shared by every arm and seed.
One cache (`prepare_graphs_cache.py`) for all arms; contracts recorded in `metadata.json`.

**Graph.** Bipartite task↔candidate-replica edges (existing) plus undirected
`peer_edge_index` over task nodes with `peer_edge_attr = log1p(x_ij / 1 MB)`. Partial-state
columns (`partial_state_columns`, single source for every arm) gain a `peer` block:
committed-peer exchange (Σ over committed partners j of x_ij · X(cand node, node(j)),
normalised by the dataset's max) and the peer-mass lookahead (the B4 column); contract
`partial_state_v2`. The label is the near-RTT plan set from the full sweep, replica reuse
allowed (the environment's sweep is the full Cartesian product).

**Arms** (one trainer per model class, configs under `experiments/peer_affinity_v1_*.yaml`,
never a new `train_*.py`):
- `gnn`: message passing over bipartite + peer edges (`PeerConv`, edge_dim 1, recorded in
  the sidecar as `mp_peer_edges`), EdgeScorer with the v2 partial-state block.
- `mpoff`: the same network with message passing disabled — a two-tower pointwise scorer
  (say so in every table).
- `mlp_t1`: tabular MLP on the pointwise columns + per-node counts + capacity + the
  committed-peer exchange column (everything a prefix-conditioned pointwise model can see).
- `mlp_t1x`: `mlp_t1` + the peer-mass lookahead column.

**Decoder** (shared): masked sequential decode in task-id order with the capacity mask,
replica reuse **allowed**, per-step re-scoring with the partial-state columns; on a stuck
state (no cap-feasible candidate — 23.5 % of R2 datasets under B3′) the decoder backtracks
one step (at most k steps), then falls back to the least-loaded cap-violating candidate and
**counts** the relaxation. The same code path serves every arm.

**Protocol.** Honest checkpoint selector (`NEAR_RTT_VAL_EXACT_REGRET=1` with the full sweep
as the lookup) **and** the final checkpoint, both read; per-arm learning-rate sweep
{5e-4, 1e-3, 2e-3} selected on validation only; convergence check (validation-tail slope
within noise of flat) before any read; **8 seeds per arm**; determinism gate
(`tests/test_trainer_determinism.py`) before the first gated run; every run logs to W&B.

**Statistic.** Per seed, median over the 34 held-out datasets of decode regret vs the
enumerated cap-feasible optimum (%). Primary contrast **`gnn` vs `mlp_t1`**; secondary
`gnn` vs `mpoff`, `gnn` vs `mlp_t1x`. Exact Wilcoxon paired by seed, α = 0.05, effect bar
1 pp. Readings: GNN-NEEDED (p < 0.05 and median Δ ≥ 1 pp in the GNN's favour);
POINTWISE-BETTER (p < 0.05, Δ ≤ −1 pp); TIE (p ≥ 0.05, |Δ| < 1 pp); INDETERMINATE otherwise.
Also reported: train-split regret per arm (fit ceiling), relaxation rate of the decoder per
arm, and the ten-order search competitor from A6 as a non-learned reference line.

**Registered prediction.** `gnn` ≥ +1 pp over `mlp_t1` and over `mpoff`; the effect size
expected ≈ 2 pp. If `mlp_t1x` closes the gap, the result is written as "hand lookahead
suffices". If `mpoff` ties `gnn`, the edge (if any) belongs to the two-tower parametrisation,
not to graph reasoning, and is written that way.

**Out of scope.** Live serving (`MAX_BATCH_SIZE_FOR_GNN = 4`, no batch-of-independents
live path exists for k = 10) — a separate registration.

### Training registration T1 — the read: GNN-NEEDED over peer-blind pointwise, INDETERMINATE over peer-aware pointwise (2026-09-10)

192 training runs, all completed, none failed: 4 arms × 3 learning rates × 8 seeds. The GNN
arms saved both the honest-selector checkpoint and the final-epoch checkpoint (96 GNN
checkpoints, 48 MLP), each scored on the **34 held-out R2 datasets** (seeds 7018–7034, seen
by no arm) with the shared masked decoder and the tie-band regret. Reports:
`simulation_data/peer_affinity_t1_reports/` (144 files), reading
`simulation_data/peer_affinity_t1_read.json`, script `scripts_cosim/peer_affinity_t1_read.py`.

**Venue (amendment on the day, no registered quantity touched).** Every non-draining GPU node
on datalab was fully allocated and the GPU array sat at priority 1 for two hours, so the GNN
arms ran on `CPU-amd` (`scripts_cosim/datalab/peer_affinity_v1_t1_gnn_train_cpu.sbatch`, commit
`761130e`; same configs, seeds, checkpoint names and sidecar asserts as the GPU script). This is
not a compromise on this corpus: the per-epoch cost is dominated by the exact-regret validation
pass, which is CPU-bound. Measured on the real `gnn` config, 109 training graphs:

| venue | s / epoch |
|---|---|
| local Tesla T4, 8 threads | 11.0 |
| local CPU, 16 threads | 7.5 |
| datalab `CPU-amd`, 16 threads | 5.1 |

All 48 tasks started within 20 s, ran the declared `gnn` env (`torch 2.5.1+cu121`), completed
300/300 epochs, and passed the sidecar assert (arm flag, `mp_peer_edges`, `partial_state_v2`,
`dag_alpha_key`, replica reuse, relaxation, `peer_mass`, split sha256). Checkpoints rsynced back
md5-identical. Arms trained in one venue and compared only against each other; no cross-venue
number is quoted.

**Learning rate, selected on validation only** (mean over seeds of the per-seed validation median):

| arm | lr 5e-4 | lr 1e-3 | lr 2e-3 | chosen |
|---|---|---|---|---|
| `gnn` | 30.87 | 30.45 | **26.05** | 2e-3 |
| `mpoff` | 33.20 | 28.93 | **25.81** | 2e-3 |
| `mlp_t1` | 39.20 | 40.49 | **37.57** | 2e-3 |
| `mlp_t1x` | 32.84 | **32.54** | 35.57 | 1e-3 |

**Held-out decode regret** (%, median over the 34 test datasets per seed, then mean over 8 seeds;
train-split median at the same checkpoint for the fit ceiling):

| arm | peer columns | test (honest selector) | test (final) | train (honest / final) |
|---|---|---|---|---|
| `gnn` (MP + PeerConv) | committed + lookahead | **23.87** | 24.96 | 5.12 / **0.00** |
| `mpoff` (two-tower, no MP) | committed + lookahead | 25.84 | 26.69 | 9.08 / 0.29 |
| `mlp_t1x` (pointwise) | committed + lookahead | 25.78 | — | 19.96 |
| `mlp_t1` (pointwise) | committed only | **37.23** | — | 28.02 |

**Registered contrasts** (exact Wilcoxon paired by seed, α = 0.05, bar 1 pp; Δ > 0 favours the
first arm; "wins" = seeds favouring it):

| contrast | Δ median | p | wins | reading |
|---|---|---|---|---|
| `gnn` vs `mlp_t1` @val | **+13.38 pp** | **0.0078** | 8/8 | **GNN-NEEDED** |
| `gnn` vs `mlp_t1` @final | **+11.89 pp** | **0.0078** | 8/8 | **GNN-NEEDED** |
| `gnn` vs `mpoff` @val | +2.08 pp | 0.148 | 5/8 | INDETERMINATE |
| `gnn` vs `mpoff` @final | +0.77 pp | 0.313 | 5/8 | TIE |
| `gnn` vs `mlp_t1x` @val | +2.68 pp | 0.313 | 5/8 | INDETERMINATE |
| `gnn` vs `mlp_t1x` @final | +2.93 pp | 0.461 | 6/8 | INDETERMINATE |
| `mpoff` vs `mlp_t1` @val | +11.22 pp | **0.0078** | 8/8 | significant |
| `mlp_t1x` vs `mlp_t1` @val | +10.62 pp | **0.0078** | 8/8 | significant |

**The reading.** The registered prediction was `gnn` ≥ +1 pp over `mlp_t1` **and** over `mpoff`.
Half of it holds, and the half that fails is the informative one.

1. **The environment is the first in this program to break a pointwise scorer at training time.**
   `mlp_t1` — a pointwise MLP with the full partial state including the *exact* exchange cost to
   already-committed peers — loses by 11–13 pp to every arm that also sees the lookahead column,
   8/8 seeds, at the floor of the exact test's p-value. R2's paper bars said the count competitor
   could not repair this structure; T1 confirms it survives contact with a trained model. This is
   not the count theorem's shape and no previously registered stop covers it.
2. **Message passing is not the lever.** `mpoff` disables the encoder entirely — verified in
   `gnn_model.py:_encode`, where the `PeerConv` call sits inside the branch `_disable_mp` skips,
   so that arm does no task↔task mixing at all — and it ties the full GNN (+0.77 pp, p = 0.313 at
   the final checkpoint). Registration's own contingency applies: the edge, if any, belongs to the
   two-tower parametrisation, not to graph reasoning.
3. **Hand lookahead suffices, within what 8 seeds can resolve.** The whole 10.6 pp gap between
   `mlp_t1x` and `mlp_t1` is **one hand-built scalar column**: the expected exchange cost to peers
   not yet committed, averaged over each peer's candidate nodes (`reduced_features.py:505`,
   `$PARTIAL_STATE_PEER_MASS`). Both arms see the committed-peer column; only `mlp_t1x` sees the
   lookahead. A pointwise model with that one feature lands within 2.7 pp of the GNN, p = 0.31.
4. **The fit-ceiling split reappears, sharper than in `route_b_v1`.** At the final checkpoint the
   GNN drives train regret to **0.00 %** — exact memorisation of 109 sweeps — and still reads
   24.96 % held-out, while `mlp_t1` cannot get train regret below 28 %. Capacity to fit the joint
   target is not the same as generalising it; this corpus separates the two cleanly.

**What this read does NOT establish.** `gnn` vs `mlp_t1x` and `gnn` vs `mpoff` are
**INDETERMINATE, not ties**: the point estimates (+2.7 pp, +2.1 pp) sit right at the registered
expected effect size (≈ 2 pp) and n = 8 cannot resolve them. Observed per-seed sd is 3.8 pp and
2.9 pp, so 80 % power needs ≈ 31 and ≈ 16 seeds at the observed effect, and ≈ 115 and ≈ 64 seeds
at the 1 pp bar. **A powered rerun of exactly these two contrasts is the one open question this
node leaves**, and it is cheap: ~25 min per seed on `CPU-amd`, 48 idle-node slots.

**Decoder health** (chosen lr, 8 seeds × 34 datasets = 272 decodes per arm-variant): **zero
infeasible plans in all 2 176 held-out decodes**. Counted relaxations 8–15 per arm-variant for
the GNN arms, 13 (`mlp_t1`) and 27 (`mlp_t1x`) for the pointwise arms — 3–10 %, and every relaxed
plan is scored against the whole sweep, never a censored tie group.

**Convergence check — the registered bar is below its own metric's noise floor, and is reported,
not claimed passed.** All 48 GNN runs completed 300/300 epochs. The bar (validation-tail slope
< 2 % of level over the last 20 epochs) passes for **16 of 48** seeds. But the epoch-to-epoch
absolute change of that same validation metric is **4.5–8.0 % of level** (measured on tasks 0, 2,
14), so the bar sits under the noise floor and cannot be met reliably by a converged run; the
failing tails are mixed in sign (−0.11 to +0.12 relative), i.e. noise, not drift. The registered
mitigation is what protects the read: **both** the honest-selector and the final checkpoint are
reported for every GNN arm, and the two agree on every verdict above. Tool correction filed in
`docs/gates/gate-tools.md`.

**Provenance.** Cache `simulation_data/graphs_cache_peer_affinity_v1_t1` (170 graphs,
`partial_state_v2`, `dag_primary_alpha_key` 2.5, `peer_exchange_block`, `platform_feature_dim` 14);
split `experiments/peer_affinity_v1_t1_split.json` (train 109 / val 27 / test 34, sha256
`c7d98b21…`, asserted in every sidecar); α key 2.5; W&B project `gnn-peer-affinity-v1`. The
`pa-t1-preflight` job (753453) FAILED at its last step only — it trained a smoke GNN on a cluster
GPU and synced to W&B, then tried to score it against raw sweeps that exist only on the local box;
all scoring is local by design.

### Training registration T1b — corpus + power: message passing DOES help once the GNN is not data-starved (2026-09-11)

T1's two secondary contrasts were INDETERMINATE at n = 8, with point estimates sitting exactly on
the registered ≈ 2 pp effect. T1b changes **only** corpus size and seed count and re-reads the same
bars on the **same held-out block**: test is byte-identical to T1's 34 R2 datasets (seeds
7018–7034), disjoint from T1b train and val, so T1 and T1b are directly comparable.

| | T1 | T1b |
|---|---|---|
| training datasets (train + val) | 136 | **482** |
| seeds per (arm, lr) | 8 | **16** |
| held-out block | 34 (R2) | 34 (R2), identical |
| cache | `graphs_cache_peer_affinity_v1_t1` (170) | `graphs_cache_peer_affinity_v1_t1b` (516) |
| split sha256 | `c7d98b21…` | `0d66d1c0…` |

Corpus: 500 fresh seeds (7200–7699) generated on `CPU-amd` (job 754230, 60/60 COMPLETED, 0 failures,
8.9 GB). **346 of them entered T1b** — 350 pulled back (local disk holds no more alongside the cache),
of which 4 (`train2` ds_00003/00009/00157/00333) failed the cache build's training-contract check with
no feasible sweep rows at the α = 2.0 rung and were set aside rather than dropping the rung, which
would have changed the cache contract away from T1's. The T1 corpus had zero such failures in 170.
The exclusion is 1.1 % of the new seeds and mildly favours looser instances; the remaining 150
datasets stay on datalab. MLP arms got real early stopping (`patience` 40, was 600 = disabled); this
is a compute fix, **not** a correction to T1 — both trainers already restore best-val weights
(`train_mlp_dim22_from_batch.py:497`), so no T1 number was read at an overfit end state.

**Held-out decode regret** (%, median over the 34 test datasets per seed, then mean over 16 seeds):

| arm | T1 @val | **T1b @val** | T1b @final | T1b train @val / @final |
|---|---|---|---|---|
| `gnn` | 23.87 | **18.78** | 25.05 | 17.15 / **0.00** |
| `mpoff` | 25.84 | **23.41** | 23.76 | 18.01 / 10.86 |
| `mlp_t1x` | 25.78 | **22.54** | — | 20.18 |
| `mlp_t1` | 37.23 | **29.80** | — | 29.73 |

**The registered contrasts, at the honest selector** (every arm read at the checkpoint its trainer
selected on validation — the only like-for-like comparison):

| contrast | T1 (n = 8) | **T1b (n = 16)** | wins |
|---|---|---|---|
| `gnn` vs `mlp_t1` | +13.38 pp, p = 0.0078 | **+11.80 pp, p = 0.00003** | 16/16 |
| **`gnn` vs `mpoff`** | +2.08 pp, p = 0.148 (INDETERMINATE) | **+5.14 pp, p = 0.00101 — GNN-NEEDED** | 13/16 |
| **`gnn` vs `mlp_t1x`** | +2.68 pp, p = 0.313 (INDETERMINATE) | **+3.30 pp, p = 0.00214 — GNN-NEEDED** | 13/16 |
| `mpoff` vs `mlp_t1` | +11.22 pp, p = 0.0078 | +5.26 pp, p = 0.00015 | — |
| `mlp_t1x` vs `mlp_t1` | +10.62 pp, p = 0.0078 | +7.79 pp, p = 0.00003 | 16/16 |
| `mpoff` vs `mlp_t1x` | — | −1.02 pp, p = 0.229 (tie) | 7/16 |

**Reading: T1's "message passing is not the lever" is SUPERSEDED. The lever was corpus size.**
The cleanest contrast in the whole lineage is `gnn` vs `mpoff` — same architecture, same features,
same decoder, same selector, same 16 seeds; **the only difference is whether `PeerConv` runs**. It
moves from +2.08 pp (p = 0.15) at 136 training datasets to **+5.14 pp (p = 0.001, 13/16 seeds)** at
482. The mechanism is visible in the per-arm deltas: 3.5× data buys `gnn` **−5.09 pp** but `mpoff`
only −2.43 pp and `mlp_t1x` −3.24 pp. The extra capacity of message passing was real but
unexpressible at T1's corpus size — the same "corpus is the largest measured lever" finding that
`link_mp_v1` and `reliability_matched_v1` reported, here running in the GNN's favour for the first
time in this program.

**Three qualifications, all load-bearing.**

1. **The advantage lives at the selected checkpoint, not at convergence.** MP-ON overfits harder:
   at the last epoch its train regret is **0.00 %** (exact memorisation of 482 sweeps) against
   `mpoff`'s 10.86 %, and its held-out regret degrades 18.78 → 25.05. Read at the final epoch the
   two arms **tie** (−0.80 pp, p = 0.252). So the claim is "with the registered honest selector",
   which is the protocol T1/T1b registered — not "at convergence".
2. **The live gate agrees in direction but does not confirm significance.** Same 16 seeds, real
   engine, peer physics charged: `gnn` 70.89 %, `mlp_t1x` 71.79 %, `mpoff` 74.43 %, `mlp_t1`
   88.08 %, reactive Knative 116.76 % excess over the unconstrained sweep argmin. `gnn` vs `mpoff`
   is +3.78 pp, p = 0.130, 11/16; `gnn` vs `mlp_t1x` +3.14 pp, p = 0.553, 10/16; `gnn` vs `mlp_t1`
   +12.22 pp, p = 0.00003, 16/16. The per-seed spread on that statistic is far wider (59–86 % vs
   14.5–24.5 % offline), which costs the power. **An unconfounded GNN-over-pointwise claim is
   established offline and NOT established on the live gate.** Report:
   `simulation_data/peer_affinity_t1b_live_replay.json`.
3. **Do not quote the `@final` row against an MLP arm.** `gnn_vs_mlp_t1x@final` reads −1.59 pp
   "POINTWISE-BETTER" (p = 0.044) — but the MLP trainer has no final variant (it restores best-val
   weights before saving), so that row compares an **overfit GNN against a selected MLP**. It is the
   `route_b_v1` Phase 2 artifact in mirror image (`docs/gates/gate-tools.md`, 2026-09-07: a
   registration comparing model classes must name one selection rule for every arm). The `gnn` vs
   `mpoff` `@final` row IS fair (both arms have both variants) and is the tie quoted in (1).

**A selector asymmetry that favours the GNN, stated rather than buried.** The GNN arms select on
exact val decode regret (`NEAR_RTT_VAL_EXACT_REGRET=1`) — the statistic the read reports — while the
MLP arms select on val **edge accuracy**, a proxy. So `gnn` vs `mlp_t1x` is not selector-symmetric
and its +3.30 pp should be read as an upper bound. **`gnn` vs `mpoff` carries no such asymmetry**,
which is why it is the contrast this entry leads with.

**Provenance.** 192 runs (4 arms × 3 lrs × 16 seeds), 0 failures, all sidecar asserts passed; GNN on
`CPU-amd` (job 754846, 96/96 COMPLETED), MLP (job 754640, 96/96, early stop at epochs 62–78). 288
checkpoints scored on the held-out block, 0 eval failures. Chosen lrs: `gnn`/`mpoff` 2e-3,
`mlp_t1`/`mlp_t1x` 5e-4. Reading `simulation_data/peer_affinity_t1b_read.json`, reports
`simulation_data/peer_affinity_t1b_reports/` (288). Full suite 588 passed.

### Stage 3 — the live serving path exists, is bit-identical to the offline read, and a production-trace gate is registered (2026-09-11)

**Why.** Every number above the T1b live gate came from decoded plans replayed as `forced_placements`;
no prefix-conditioned checkpoint had ever *served* — `load_gnn_model` refused them, the live graph
builder emitted no one-hot / peer edges / partial-state context, batches were capped at 4, and an
inference error fell back to shortest-queue silently. The "co-sim → model → eval" path was three
quarters of a path. This entry builds the last quarter and proves it before using it.

**Built** (`src/policy/gnn/prefix_serving.py` is the one home; the offline evaluator was refactored
onto it and re-scored 516/516 T1b datasets bit-identically):
* `load_prefix_conditioned_gnn` — sidecar-validated construction (dims off the weights; peer edges,
  one-hot width, partial-state contract, peer-mass flag, MP flag, cap rung, decode options off the
  sidecar; contracts adopted into the environment when silent, verified when set). `load_gnn_model`
  serves such a checkpoint only under `GNN_DECODE_MODE=masked_topo`.
* `attach_live_prefix_block` — the live twin of the cache's `attach_dag_partial_state_block`, from live
  objects (tasks, nodes, the fabric's routes, the orchestrator's peer table): demands from
  `memoryRequirements` × the event's `demand_scale`, caps by the `alpha_max` rule over the batch's
  candidates, `krank_node_order`, ingress links, `node_exchange`, peer edges and norm.
* `GNNScheduler._process_task_batch_prefix` — joint masked decode, `planned_node_name` pre-pass before
  any enqueue (the determined scheduler's rule), tasks without a reachable replica deferred to the
  autoscaler, **no fallback path**: every failure raises. Batch range [1, 16] in masked_topo mode.
  `GNN_BATCH_BY_PEER_GROUP=1` collects the first task's peer group (waits up to the batch timeout for
  late members; other groups stay queued). `GNN_PREFIX_TRACE_PATH` records every served graph + plan.
* Physics: a peer that is neither placed nor planned at a task's input stage is a **rendezvous** (wait
  for it to be scheduled, recorded as `peerRendezvousWait`), where it used to raise. Fully-planned
  batches never wait — every co-sim run and forced replay is bit-identical (`tests/test_peer_exchange_replay.py`).
* `HEROSIM_SERVER_ONLY_REPLICAS=1` (shared autoscaler): no replica on a client node. The corpora host on
  servers only; a live episode with no replica plan otherwise falls back to the source client within a
  second at 2,650 arrivals/s, and the exchange physics has no client↔server map for the peer.
* Result stats carry `schedulerCounters` (prefix batches, tasks decoded/deferred, in-batch pairs, peers
  outside the batch, incomplete groups) and `totalPeerRendezvousWait`.

**Proof** — `scripts_cosim/peer_affinity_live_serve_check.py` runs the live `gnn_gnn` policy on the 34
held-out datasets (no forced placements) and holds it against the cache and the T1b read. **34/34
bit-identical for `gnn_s1` and 34/34 for `mpoff_s1`**: the served graph equals the cache graph on every
attribute (id-keyed), the decoded plan equals the offline report's `decoded_combo`, and the engine's
`total_rtt` equals the sweep row and the replay gate's number. It did not pass first time — three
serving defects, all in `docs/gates/gate-tools.md` (2026-09-11): rf/cnn had zero live candidates; the
live temporal capture had drifted from the SSC helper (platform column 11 off by ~4×10⁴ on flashCard
nodes; 4/34 batches, 3 plans changed); and the peer norm was 0 when every candidate sat on one node.
Tests: `tests/test_prefix_serving.py` (loader, ranges, v2 context, rendezvous, a reactive peer episode,
and the parity pin on two held-out datasets).

**Registered production gate (before any run).** Cell `cell_s7901`: the corpus space config verbatim
(20 clients / 6 servers / sparse / conn 0.6 / server mesh / backbone 1 Gbps) at an unseen topology
seed, live-regeneration parity PASS (`scripts_cosim/make_peer_affinity_gate_cell.py`). Workload:
`workload-150-150.json` — the real 450,729-event production trace (rps 150, 20 clients, dnn1/dnn2 only)
with the corpus's peer structure put on it by `scripts_cosim/make_peer_affinity_production_workload.py`
(seed 7300): consecutive arrivals in groups of 10, 2 partners per task, payload 200 MB × 10^U(−1, 1),
per-event `demand_scale` U(0.5, 2) — 45,073 groups, 801,643 pairs, group span median 3.0 ms. Arms (34):
`knative_network`, `knative_network_batch` (10-task / 20 ms window), `gnn` × 16 seeds and `mpoff` × 16
seeds (T1b lr 2e-3 checkpoints, honest selector, masked_topo, peer-group batching, 20 ms window).
Physics for all: `HEROSIM_PEER_EXCHANGE=1`, server-only replicas, `node_disk_v2`, keep-alive default
(production, not the co-sim 10⁶). **Statistic:** `total_rtt` over all 450,729 tasks per arm. **Primary
contrast:** `gnn` vs `mpoff`, paired by training seed (16 pairs), exact Wilcoxon, α 0.05, effect bar 1 %
of `mpoff`'s total; readings GNN-NEEDED / TIE / POINTWISE-BETTER / INDETERMINATE as T1. **Secondary:**
each learned arm vs the two reactive arms (16 seeds vs one number). Also reported: `totalPeerExchangeTime`,
`totalPeerRendezvousWait`, the scheduler counters (incomplete groups and peers outside the batch are the
two the peer-group design can be judged by).

**Stated before the numbers.** (a) The trace is 2,650 arrivals/s onto 6 servers: a saturation regime
(the 3k-event smoke ends with mean task elapsed ≈ 800–1,300 s), so queueing dominates and the peer term
is a small fraction of every arm's total. That is the production condition the user asked for, not a
regime the corpus sampled. (b) The trace carries dnn1/dnn2 only; the corpus cycles four types. (c) The
cap rung α = 2.5 is decode-side only — the reactive arms are unbounded, as in the T1b live gate. (d)
Peer-group batching is a real queueing cost the GNN arms pay and the reactive arms do not; the
`knative_network_batch` arm pays a comparable window. (e) MLP arms are out of scope live: the MLP batch
scheduler has no dim63crk path.

**Denser peer graph — registered, generation launched.** The Phase 0 screen's p3 cells at k = 10 pass
only at 800 MB (`x800_a1.5_k10c3_p3`) or with 4 candidates; `x200_…_p3` does not, so the partner count
cannot be moved alone at 200 MB. Two corpora on fresh seeds 7700–8199 (train) / 8200–8233 (held-out):
`c3_x800_p3` (3 partners, 800 MB) and the matched control `c3_x800_p2` (2 partners, 800 MB), same cell,
same k, same α ladder, same generation env as T1b (`scripts_cosim/datalab/peer_affinity_v1_x800_generate.sbatch`).
The training registration for this rung is T1b's verbatim (4 arms × 3 lr × 16 seeds, honest selector,
gnn-vs-mpoff primary) and is not re-signed here; the prediction is that the `gnn`−`mpoff` gap grows with
partners and the `mlp_t1x`−`gnn` gap grows too (the hand lookahead averages over more candidates).
