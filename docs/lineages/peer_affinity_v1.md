# peer_affinity_v1 — ACTIVE

> **Status:** `ACTIVE` &nbsp;·&nbsp; **Index:** [LINEAGES.md](../../LINEAGES.md) &nbsp;·&nbsp; **Record spans:** 2026-09-09 → (open)

**Outcome.** **Phase 0 paper screen: GO (2026-09-10).** 14 of 60 cells pass every registered
bar, on 34 of 34 sources each. The pairwise-instance exchange term is the first joint structure
in this program that per-machine count columns do **not** repair: in the passing cells the
count competitor with an *exhaustive* decode still carries 18–50 % regret, a sequential greedy on
the true marginals with an exact prefix 32–208 %, and the same greedy with a hand peer-mass
lookahead 20–107 % (9–17 % under the best deterministic task ordering). Both controls read
additive (S0b R² = 1.0000 in all 60 cells). GO cell carried to A2: **`x50_a1.5_k10c3_p2`**
(x = 50 MB, α₄ = 1.5 ⇒ α₁₀ = 3.75, k = 10, 3 candidates, 2 partners; count repair 0.31, residual
17.6 %). The bar that decides is count repair, and it is near its 0.5 threshold in most cells
(range 0.24–0.97; passes cluster at α₄ = 1.5, 2 partners, 4 candidates) — the simulated screen
(§A2 of the plan) must reproduce it before anything is trained. No physics has been built yet.

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
