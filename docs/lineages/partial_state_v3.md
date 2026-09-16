# partial_state_v3 — a served representation that does not pin the cluster size

**Status:** `REGISTERED` (2026-09-16). Every bar below is a module constant in
`scripts_cosim/partial_state_v3_read.py`, committed **before** any cache is built. Amend by
dated amendment only.

**Parents:** `cluster_scale_v1` (CLOSED — peer-group assembly cost falls 6.10 → 0.73 s as
arrivals speed up 13.3×, worth ~3.9–5.4 s of net latency on the best cell, and the served
feature layout **cannot represent more than six candidate-hosting nodes**),
`scheduler_residence_v1`, `queue_range_v1`, `drainable_objective_v1` (the checkpoints and
recipe this lineage re-trains under a new contract).

---

## The claim under test

`cluster_scale_v1` priced the cluster-scale axis and found it blocked by the representation,
not the model. `TaskPlacementGNN` is size-agnostic end to end — task encoder, platform
encoder, GIN over the bipartite graph, `PeerConv`, `EdgeScorer` — **except for one block**:
the 38 partial-state edge columns concatenated onto the scorer input carry a **24-column
one-hot of (candidate node's rank r, task type k) padded to `KRANK_WIDTH = 6`**
(`src/policy/tabular/reduced_features.py:252`, written at `:561`), and `krank_node_order`
raises at the seventh node. Every `gnn` arm above 6 servers died on that raise.

> **The claim:** replace the fixed-width rank one-hot with a size-free encoding of the same
> information, retrain the same recipe on the same 516-dataset corpus, and the resulting
> checkpoints (a) serve as well as the old ones where the old ones work, and (b) can be
> served on 12 / 24 / 80-server clusters at all — which makes "train on small clusters,
> serve on big ones" a measurable question for the first time.

**What the one-hot encodes.** For a candidate node with canonical rank r (ascending capacity
at α, then mean hop to the other candidate-hosting nodes, then name) among N candidate-hosting
nodes, and a task of type k, exactly one of 24 columns is 1. Summing that block over a plan's
edges reproduces the plan-level `krank_cols` construction — the closure that put the pooled
statistic inside MLP(T1)'s hypothesis space in route_b stage 2. That closure is the reason the
block is one-hot; it is not a property any served checkpoint in this family uses.

**`partial_state_v3`** keeps the 10 base columns and the 4 linkrank columns of `partial_state_v2`
byte-for-byte (including v2's peer-mass meaning of columns 7–9) and replaces the 24 krank
columns with **`KRANK_TYPES × 2 = 8` scalars**: in the task's own type slot k,

| column | value | range at N = 6 (training) | at N = 80 (serving) |
|---|---|---|---|
| `rank_frac` | r / (N − 1), 0 when N = 1 | {0, 0.2, …, 1.0} | [0, 1] — **in range** |
| `inv_n` | 1 / N | {1, 0.5, …, 0.167} | 0.0125 — bounded, one-sided extrapolation |

and 0 in the other three type slots. Feature dim 38 → **22**. The rank is *recoverable*
from `rank_frac` and `inv_n` exactly (r = round(rank_frac × (1/inv_n − 1))), so at N ≤ 6 the
new block carries the same information as the old one — that is bar P0, and it is what lets
P1 and P2 be read as an encoding change and nothing else.

## What is and is not being claimed

* This is a **representation** change. The corpus (516 datasets at 6 servers), the label
  (`drainable_objective_v1` V = 1), the recipe (T1b, lr 2e-3, CE-only, teacher-forced
  `masked_topo`), the split artifact and the 16 training seeds are unchanged.
* **Candidate support is the treatment, not a confound.** `cluster_scale_v1` S0.d measured
  candidates per task at 3.55 / 14.18 / 47.92 for 6 / 24 / 80 servers against a corpus
  maximum of 5. P3 asks precisely whether a checkpoint trained inside that support degrades
  outside it. `OUT-OF-SUPPORT` is therefore stated, not used to void a rung.
* **Offline evaluation at scale is impossible by construction**, and this lineage does not
  pretend otherwise: co-simulation brute-forces every placement of a 10-task batch
  (`MAX_PLACEMENT_COMBINATIONS_SKIP_DEFAULT = 250,000`); 3¹⁰ ≈ 59 k at 6 servers fits,
  14¹⁰ ≈ 3 × 10¹¹ at 24 servers does not. **Generalisation is read on the live gate only**
  (rule 6 — and the live gate is the cheap instrument, ~3 min per arm).
* `mpoff` is carried throughout as the pointwise control and **no GNN-vs-pointwise claim may
  be founded here** — the encoding change is policy-agnostic and must move both.
* The scaled rungs are **not load-matched** (S0.a: reactive queue 25× and 49× the baseline's
  at 24 and 80 servers, because the cold-start drain is ~0.077 tasks/s/server). P3's statistic
  is the arm's elapsed *relative to reactive on the same cell*, which is defined at any load;
  the absolute latency at those rungs is not read.

## Bars

**P0 — instrument (offline, blocking, free).** Build the v3 cache over the same 516 parents
with the same split artifact. On every dataset, columns 0–9 and the 4 linkrank columns must
be **bit-identical** to the v2 cache's (`P0_MAX_COLUMN_DIFF = 0.0`), and the v2 one-hot rank
must be recovered from (`rank_frac`, `inv_n`) on **100 %** of edges
(`P0_MIN_RANK_RECOVERY = 1.0`, `P0_MIN_DATASETS = 516`). A cache that fails P0 is not a
representation change and nothing downstream is read.

**P1 — offline tie (ordering only, never closing).** `gnn_v3` vs `gnn_v2` and `mpoff_v3` vs
`mpoff_v2`, held-out regret at the selected checkpoint, paired by training seed, 16 seeds
each (`P1_MIN_SEEDS = 12`), exact Wilcoxon. **TIE** if |median Δ| ≤ `P1_TIE_PP = 1.0` pp or
p ≥ `P1_ALPHA = 0.05`; otherwise `ENCODING-COSTS-OFFLINE` / `ENCODING-HELPS-OFFLINE`, recorded
and carried into P2 as context. Registered expectation: **TIE** — the block is
information-equivalent at N ≤ 6 (P0).

**P2 — live instrument at 6 servers (blocking for P3's reading).** On the three cells this
family is always gated on (`cell_s7901/s9001/s9002_f4000_pg16`), `gnn_v3` and `mpoff_v3`
against their v2 twins, 16 seeds per arm, paired by training seed, mean elapsed, exact
Wilcoxon. **TIE** if |median relative difference| ≤ `P2_TIE_PCT = 5.0` % or p ≥ 0.05 on
≥ 2 of 3 cells. **P2-a (blocking):** one v2 control arm per cell must reproduce
`queue_range_v1`'s `plain` medians **54.819 / 22.350 / 65.401** to three decimals — the
proof that the v2 serving path is untouched by the code that adds v3. If P2 reads
`REPRESENTATION-COSTS-LIVE`, P3 is still run and read, but reported as **confounded by the
encoding**, because a checkpoint that lost something at 6 servers cannot be read at 80.
Registered expectation: **TIE**.

**P3 — the PRIMARY, live: does a checkpoint trained at 6 servers survive 12 / 24 / 80?**

| rung | servers | workload | arrivals/s | candidates/task (S0.d) | support |
|---|---|---|---|---|---|
| R0 | 6 | `drainable_f4000_n50000` | 0.460 | 3.55 | in |
| R1 | 12 | `drainable_f2000_n50000` | 0.921 | ~7 | **out** |
| R2 | 24 | `drainable_f1000_n50000` | 1.842 | 14.18 | **out** |
| R3 | 80 | `drainable_f300_n50000` | 6.139 | 47.92 | **out** |

Arrivals per server are held at R0's ratio. Cells are minted by
`scripts_cosim/cluster_scale_v1_cells.py` (only `/nodes/server_nodes/count` and
`/network/topology/seed` differ from the base, asserted), **4 topology seeds per rung**
(`P3_TOPOLOGY_SEEDS = (9001, 9002, 9003, 9005)` — 9004 hangs on every policy), 25 % attrition
budgeted, `P3_MIN_CELLS = 3` readable per rung. Per cell: 1 `knative_network` + 4 `gnn_v3` +
4 `mpoff_v3` (checkpoint seeds `P3_CHECKPOINT_SEEDS = (1, 2, 4, 5)`), 36 arms per rung,
**144 arms**.

* **P3-a (blocking, and the reason the lineage exists):** every `gnn_v3` / `mpoff_v3` arm on
  a readable cell at R1–R3 must **complete** — no `krank_node_order` raise, no contract
  refusal. A representation that still cannot be served at scale closes the lineage as
  `STILL-PINNED` before any latency is read.
* **P3-b, the statistic:** per cell, `d = median over checkpoint seeds of
  (arm mean elapsed / reactive mean elapsed) − 1`; per rung, the median of `d` over readable
  cells. Reported for both learned arms at every rung with its cells.
* **P3 verdict:** `GENERALISES` if every scaled rung's `d_r ≤ d_R0 + P3_TOL` with
  `P3_TOL = 0.10`; `DEGRADES` otherwise, naming the first rung that breaks. **Registered
  expectation: DEGRADES** — candidate support is 2–9.6× outside the corpus and `inv_n` is
  extrapolated, and this program has measured that failure shape three times. The bar is
  placed to be falsifiable, and a `GENERALISES` here would be the first "train small, serve
  big" result in the record.
* **P3-c (feasibility):** median wall-clock per arm at R3 ≤ `P3_MAX_WALLCLOCK_MIN = 45`.
  Above it the rung is `RUNG-UNAFFORDABLE`, an answer and not a failure.

**P4 — the prize, conditional on P3 = GENERALISES.** `SCALE-HELPS` if `d_r` falls
monotonically across R0 → R3 and `d_R3 ≤ d_R0 − P4_MIN_IMPROVEMENT` with
`P4_MIN_IMPROVEMENT = 0.05`. This is the number `cluster_scale_v1` priced at ~3.9–5.4 s.

**Phase 2 — registered now, run only if P3 = DEGRADES: does *data* fix what the encoding
alone did not?** Generate `PH2_N_DATASETS = 200` co-sim datasets at `PH2_SERVERS = 12`
with client reachability thinned so candidates per task stay ≤ `PH2_MAX_CANDIDATES = 5.0`
(in-support and brute-force-feasible by construction), add them to the 516, retrain
`gnn_v3mix` / `mpoff_v3mix` with the same recipe, and re-run P3. Pre-declared readings:
`v3mix` GENERALISES where `v3` did not ⇒ **`DATA-NOT-ENCODING`**; both DEGRADE ⇒
**`OUT-OF-SUPPORT-IS-NOT-LEARNABLE-FROM-BELOW`**, closing the axis for supervised training
at small clusters. Phase 2 has its own cost (~200 co-sim datasets, ~4 h CPU; 32 runs) and is
not started until P3 is read.

## Recipe and cost

* **Contract:** `partial_state_v3` in `reduced_features.py`, `PARTIAL_STATE_FEATURE_DIM`
  becomes contract-dependent (`partial_state_feature_dim(contract)`), `krank_node_order` pads
  only under v1/v2. The v2 path must be bit-identical (P2-a). The MLP `dim63crk` layout is
  **not** extended — an MLP trainer meeting v3 fails loud; the MLP is not an arm here.
* **Cache:** `graphs_cache_drainable_objective_v1_v1_psv3`, same parents and split as
  `graphs_cache_drainable_objective_v1_v1`, `PARTIAL_STATE_CONTRACT=partial_state_v3`.
  ~1–2 h datalab CPU.
* **Training:** `experiments/partial_state_v3_{gnn,mpoff}.yaml`, T1b recipe, 16 seeds each,
  **`--patience 60 --min-epochs 100`** (new 2026-09-16; cannot truncate a selection — a stop
  fires only after `patience` epochs without improvement, so the selected epoch is always
  ≤ `epochs_run − patience − 1`). Selected epochs are recorded beside v2's. ~3–4 h GPU wall.
* **Live:** P2 96 arms + P3 144 arms, CPU arrays, ~3–5 min each below 80 servers.
* **Total:** ~2 days of cluster time plus the code.

## Declared in advance

* Checkpoint seed 3 excluded by name (deterministic livelock in the v2 family; the v3 draws
  are new, the exclusion is kept so the seed sets match).
* Topology seed 9004 excluded by name (hangs on every policy at every rung, job 769390).
* `SIM_FORCE_FULL_STATS=1`, window 16 s, poll 1 ms, batch size 10 — every policy time
  constant held fixed across rungs, as `cluster_scale_v1` registered and for the same reason.
* Every arm name carries **cell, policy, checkpoint seed and rung** (the lesson of job 769390).
* Nothing in this lineage supports a GNN-vs-MLP or GNN-vs-`mpoff` claim.

## Not in scope

Widening the pad (`KRANK_WIDTH = 80`: the extra slots get zero gradient at 6 servers and
are out of distribution the moment they light up — the failure shape of `xavierGpu`, the
queue column and S0.d); dropping the block (`partial_state_edges = 0`, which also changes the
decode); attention/set pooling over candidate nodes (a different representation — a
follow-on if v3 degrades and Phase 2 does not repair it); load-matching the scaled rungs
(closed by `cluster_scale_v1` S0.a for these traces); the MLP arm.

## Record

### 2026-09-16 — P0 read: **INSTRUMENT-PASS**, and a venue trap on the way

**Where the cache was built.** Datalab job 769544 (`partial_state_v3_cache.sbatch`) FAILED
after 8:44 on `Missing system_state_captured_unique.json for ds_00000` in
`gnn_datasets_peer_affinity_v1_c3_x200_train2`: datalab's copy of that directory is the
1500-dataset in-place extension and carries **0** SSC files. The v2 twin was never built there
either — `sacct` shows no job at its build time (2026-09-15 08:07 UTC) — it was built on the
**local** checkout, whose `train2` is the original 346 parents with all 346 SSC files, and
rsynced up; every earlier `dobj-cache` job on datalab (766312/766313/766316) had failed the
same way. Filed in `docs/gates/gate-tools.md`. The v3 cache was built locally by the same
command and env (`PARTIAL_STATE_CONTRACT=partial_state_v3`), 33.9 min: 516 datasets,
19,663,188 RTT rows, label `rtt_drift:1@lambda=0.46,clock=measured`, near-RTT sidecar 99,651
entries — every figure identical to the v2 twin's except the contract. Rsynced to datalab
with md5 agreement on the 8 critical files; the failed job's 186 stale `rtt_chunk_*` files
were set aside under `…_psv3.stale_769544/`, not deleted.

**P0 on the two caches, all 516 datasets** (`scripts_cosim/partial_state_v3_p0.py`,
`simulation_data/partial_state_v3/p0.json`): every partial-state ingredient identical dataset
for dataset; columns computed under each contract from its own cache, with an empty prefix
**and** along the first tied-optimal plan's teacher-forced prefix on 516/516 datasets:

| statistic | bar | measured |
|---|---|---|
| max |v2 − v3| on the 10 base + 4 linkrank columns | `P0_MAX_COLUMN_DIFF = 0.0` | **0.0** |
| v2 rank recovered from (`rank_frac`, `inv_n`) | `P0_MIN_RANK_RECOVERY = 1.0` | **1.0** (every edge) |
| datasets | ≥ 516 | **516** |

**⇒ INSTRUMENT-PASS.** v3 is a pure representation change. Also measured on the way: the
corpus never has more than **5** candidate-hosting nodes per dataset — the pad of 6 was never
full, so `inv_n` spans {1, 1/2, …, 1/5} in training and nothing narrower.

**Training submitted:** job 769631, 32 arms (`partial_state_v3_train.sbatch`), which refused
to start without the P0 artefact and read INSTRUMENT-PASS from it.

### 2026-09-16 — training and P1 read: **ENCODING-HELPS-OFFLINE on both arms** (expectation was TIE)

Job 769631: **32/32 COMPLETED**, every arm stopped on patience at 100–146 epochs (last
improvement at epochs 34–85), so ~55 % of the registered epoch budget was never spent and no
selection was truncated (a stop needs 60 epochs without improvement). The untrained eval the
trainer now logs reads **38.3 %** task accuracy before any gradient step against a 38.5 %
chance floor — the number the "why does it start at 40 %" question was about.

**P1** (`scripts_cosim/partial_state_v3_p1.py`, `simulation_data/partial_state_v3/p1.json`):
held-out regret at the selected checkpoint (`final/test/regret_masked_topo`), as a percentage
of the 34 test datasets' mean optimal RTT (442.11 s — identical under both caches by P0),
paired by training seed, exact Wilcoxon:

| arm | v3 median | v2 median | paired median Δ | p | v3 ahead | bar (|Δ| ≤ 1.0 pp or p ≥ 0.05) | read |
|---|---|---|---|---|---|---|---|
| `gnn` | **38.40 %** | 39.92 % | **−1.62 pp** | 0.0386 | 12/16 | misses both | **ENCODING-HELPS** |
| `mpoff` | **40.77 %** | 43.18 % | **−2.57 pp** | 0.0008 | 14/16 | misses both | **ENCODING-HELPS** |

Selected epochs: `gnn` v3 24–140 (median 72) vs v2 19–67 (median 32); `mpoff` v3 22–56 vs
v2 20–60. **Registered expectation was TIE** — the block is information-equivalent at N ≤ 6 —
and it is not a tie: two scalars in the task's own type slot fit the held-out set better than a
24-way one-hot over the same information, on both arms, with the pointwise twin gaining more.
Plausible and unregistered: a sparse one-hot with 4 type slots × 6 ranks is 24 parameters per
scorer input the corpus (≤ 5 nodes) never fills, and a scalar rank is a smoother function of
the same fact. **Carried into P2 as context, not as a claim:** `offline_live_transfer_v1`
measured that the offline score ranks epochs within a run and never arms, so this reads as
"the encoding costs nothing offline and may help" until P2 says what it does when served.

### 2026-09-16 — P2 read: **TIE on 3/3 cells for both arms; the v2 path is bit-identical**

Job 769871, **186/186 arms COMPLETED** (192 minus the six gnn-seed-3 exclusions), 0 hangs.
`simulation_data/partial_state_v3/p2.json`.

**P2-a (blocking) passes exactly:** the re-run v2 `gnn` arms' per-cell medians are
**54.819 / 22.350 / 65.401** — `queue_range_v1`'s `plain` medians to three decimals — so the
commit that adds v3 left the v2 serving path untouched.

| cell | `gnn` v3 vs v2 (n = 15) | `mpoff` v3 vs v2 (n = 16) |
|---|---|---|
| s7901 | +5.71 %, p = 0.078, v3 ahead 4/15 | +2.50 %, p = 0.57, 7/16 |
| s9001 | −1.20 %, p = 0.23, 10/15 | **−3.35 %, p = 0.0004, 16/16** |
| s9002 | +6.59 %, p = 0.46, 5/15 | +5.05 %, p = 0.53, 7/16 |

Every cell reads **TIE** under the registered bar (|median| ≤ 5 % or p ≥ 0.05), so **P2 = TIE**
on both arms and P3 is read cleanly. Descriptive, inside the tie band and not a claim: the v3
pointwise arm beats its v2 twin on s9001 on 16/16 seeds; the offline ENCODING-HELPS did not
carry to the other two cells. The encoding change costs nothing live where the old one works.
