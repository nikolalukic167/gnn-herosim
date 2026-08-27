# topology_transfer_v1 — FAILED

> **Status:** `FAILED` &nbsp;·&nbsp; **Index:** [LINEAGES.md](../../LINEAGES.md) &nbsp;·&nbsp; **Record spans:** 2026-08-18 → 2026-08-21

**Outcome.** Changed the win condition from per-plan accuracy to inductive generalization across topology sizes — and **all four arms FAILED**, including `gnn_topo`, the only arm that ever had backbone topology in the graph. ⚠ Never live-gated; checkpoints were not persisted until the 2026-08-21 unblock.

**Entry points:** `src/placement/topology_features.py`, `src/placement/network_graph.py`, `scripts_cosim/test_topology_features.py`, `scripts_cosim/test_network_graph.py`, grid `topo_transfer_v1`

**Datasets:** `gnn_datasets_4tasks_topo_transfer_v1` (3,744 datasets), graph cache `graphs_cache_topo_transfer_v1`

**Related:** [graph_structure_physics](graph_structure_physics.md)

## Standing (from the index table)

**Changes the win condition from per-plan accuracy to inductive generalization across topology sizes.** Phases 0-4 all landed. **`FAILED` 2026-08-20 — Phase 4 gate: `gnn_base` loses to `pointwise` on paired `win_rate` in 5/5 seeds** (CI excludes 0.5 below every time, effects 0.022-0.088); `gnn_node` never PASSES either (2/5 FAIL, 3/5 inconclusive-but-trending-to-null, none positive). **Same-day follow-up (2026-08-20, second pass): added the `gnn_topo` arm (`use_network_entities=True` — the only arm with backbone/link topology in the graph at all; `gnn_base`/`gnn_node` never had it) and re-ran the full 5-seed gate. `gnn_topo` also `FAILED`** (pooled win_rate 0.449, CI [0.417, 0.481], resolved not underpowered) — the FAIL is not an artifact of testing topology-blind models. **⚠ SCOPE CAVEAT, unresolved: every arm in this lineage (`pointwise`/`gnn_base`/`gnn_node`/`gnn_topo`, all 20 seed-runs) was only ever evaluated on brute-force-labeled 4-task synthetic co-sim snapshots (`rps=2, duration=1`, fixed regardless of cluster size) — none has ever been live-gated against a real trace (e.g. `data/nofs-ids/traces/workload-200-200.json`, 800k events). No lineage in this repo has ever live-gated across mismatched train/eval topology sizes; this is unexplored, not just untried here. ⚠ Trained model weights were never persisted to disk** (`AblationModel` in `gnn_necessity_ablation.py` had no `torch.save`/checkpoint call anywhere) **— every number in this lineage comes from in-process eval that discarded the model after each training run.** **UNBLOCKED 2026-08-21:** `--save-checkpoints DIR` now persists each arm's weights plus a `.contract.json` (split, held-out sizes, feature contracts, verified `serving_port`), and the serving port itself was measured — it is a **three-module rename** into `TaskPlacementGNN`, not the multi-session build it was costed as, but it requires `mp_residual=True` and getting that wrong is **silent**. The remaining cost of the partial gate is the ~14 GPU-hours, nothing else. See the 2026-08-21 subsection below and the "Co-sim-only scope and live-gate traceability" one.

> **Split note.** This node was carved out of a single 4,995-line `LINEAGES.md` on
> 2026-08-27; the section bodies below are byte-for-byte as written. An *above* / *below*
> pointer inside one may refer to a section that now lives in another node — the
> **Related** links say which.

## Record

Newest first; the sections themselves are in chronological order below.

- [topology_transfer_v1 — unblocked, and two cost estimates corrected (2026-08-21)](#topology-transfer-v1-unblocked-and-two-cost-estimates-corrected-2026-08-21)
- [topology_transfer_v1 — the `gnn_topo` arm and the co-sim/live-gate scope gap (2026-08-20)](#topology-transfer-v1-the-gnn-topo-arm-and-the-co-sim-live-gate-scope-gap-2026-08-20)
- [topology_transfer_v1 — the change of win condition (2026-08-18)](#topology-transfer-v1-the-change-of-win-condition-2026-08-18)

---

### topology_transfer_v1 — the change of win condition (2026-08-18)

**Why this lineage exists.** Five mechanisms have now failed to produce coupling a
pointwise model cannot express, and `--spread-plans-only` showed base physics is additive
to R² = 1.00000 exactly once collisions are removed — there is no reservoir of non-count
coupling left to find. Per-plan accuracy on a fixed topology is a dead axis. This lineage
changes what "winning" means to **inductive generalization**: train on small topologies,
evaluate on held-out larger ones. It does not depend on finding coupling at all.

**⚠ The motivating architectural claim is FALSE for this repo, and the framing is corrected
accordingly.** `PointwiseEdgeMLP` (`mlp_model.py:39-48`) maps `[N_edges, 21] -> [N_edges]`
with grouped-argmax decode per task (`mlp_scheduler.py:162-227`). `N` is the *candidate-pair*
count, not a topology-sized vector, so the MLP **runs unmodified on any cluster size**. The
claim "MLPs structurally cannot transfer" would be falsified by anyone reading
`mlp_model.py`. The claim this lineage tests is therefore **empirical**: does GNN regret
degrade more slowly than pointwise regret as topology grows? That must be won by
measurement.

**⚠ The GNN currently sees no network topology.** `build_inference_graph`
(`feature_builder.py:567-631`) builds `n_tasks + n_platforms` nodes with bipartite
task↔candidate-platform edges plus optional same-node platform↔platform edges (default OFF,
FALSIFIED in `mp_parity`). There are **zero node↔node link edges**; network latency enters
only as a static scalar in the 5-dim `edge_attr`. RouteNet generalizes across topologies
because its graph *is* the network. Putting links and routes into the graph is therefore a
**prerequisite** (Phase 2), not an enhancement — without it the study compares two
topology-blind models.

#### Phase 0 — the scale-dependent feature, removed (DONE)

Task feature dim 2 was `src_norm = index_of(source_node) / len(nodes)`: a node's **arbitrary
enumeration index**. Measured on 40 real `shallow_v1` datasets it is literally the ramp
`i/40` — 0, 0.025, 0.05, … — i.e. zero topological content. Its granularity *and*
distribution both change with cluster size (multiples of 0.02 at 40 nodes, 0.0125 at 80),
so any degradation measured across sizes would partly be this artifact. It was also
redundant: the source→candidate `latency` it stood in for is already edge attribute 1,
per-candidate and exact.

The formula lived in **six** independent copies (`feature_builder.py`,
`prepare_graphs_{cache,ram,cache_seq}.py`, `gnn_hetero/scheduler.py`,
`reduced_features.py`) — the same duplication `queue_features.py` was created to end, and
the same shape as the `mp_parity` train/serve split. It now lives once in
`src/placement/topology_features.py` and all six call it.

Two contracts, mirroring `queue_features.py`, so a checkpoint is never served a feature it
was not trained on (`TOPOLOGY_FEATURE_CONTRACT`, default `src_index_v0`):

| contract | dim 2 | measured on real data |
|---|---|---|
| `src_index_v0` (default) | `index(src) / n_nodes` | ramp `i/40`; **bit-exact vs the old formula on 40/40 datasets** |
| `size_invariant_v1` | reachable servers / total servers | 9 distinct values in [0.05, 0.50], bounded, size-invariant |

Guarded by `scripts_cosim/test_topology_features.py` (15 tests), including an explicit
control asserting v0 *does* vary with cluster size — if that ever stops being true, the
reason v1 exists needs rechecking.

#### Phase 1 — the topology-size axis (DONE)

**Every corpus in this repo was generated at exactly one size** (20 clients + 20 servers,
from `space_with_network.json`), so nothing could be held out. Note `cluster_size` in
`sample_simple.json` *looks* like the size knob but is **inert**: its only consumer,
`calculate_device_counts` (`executecosimulation.py:513`), is defined and **never called**.
Node counts come from the config.

`server_node_counts` is now a grid axis, crossed into `grid_topology_variants` so it needs
no separate loop level and lands in the dataset label. Grids omitting the key are
**unchanged** (verified: `shallow_v1` still 900 datasets, no size label, no kwarg). Only the
*server* tier scales — clients stay at 20 so the task-source draw is identical across arms.

**Measured combination-count probe** (1 dataset/size; the plan required this before any
corpus, since generating past the enumeration cap *silently skips* datasets and would bias a
held-out size toward its easier half):

| servers | nodes | sweep plans | gen time |
|---:|---:|---:|---:|
| 10 | 30 | 16 | 0.7s |
| 14 | 34 | 16 | 0.8s |
| 20 | 40 | 32 | 0.8s |
| 28 | 48 | 48 | 0.8s |
| 40 | 60 | 432 | 2.0s |
| 60 | 80 | 2,730 | 9.3s |
| 80 | 100 | 9,828 | 39.0s |

**⚠ SUPERSEDED 2026-08-19 — this table does not reproduce.** Re-measured at `--workers 8` on
a 32-core box, *both* the plan counts and the times differ:

| servers | plans (orig) | plans (re-run) | time (orig) | time (re-run) |
|---:|---:|---:|---:|---:|
| 20 | 32 | **18** | 0.8s | 0.4s |
| 28 | 48 | **44** | 0.8s | 0.5s |
| 40 | 432 | **343** | 2.0s | 3.3s |
| 60 | 2,730 | **2,231** | 9.3s | 23.0s |
| 80 | 9,828 | **8,698** | 39.0s | **117.2s** |

**Suspected cause, not confirmed: the 2026-08-18 workload-seeding fix changed the draw**, so
the two tables enumerate different workloads. Both are kept rather than one overwritten, so a
future re-run can tell which it matches. **Budget from the re-run numbers** — the top of the
ladder is ~3× more expensive than recorded. This also weakens the *low-end* justification the
ladder rests on: 20 servers enumerates **18** plans, not 32, so the coarsest rung is coarser
than the cutoff argument assumed. The ladder is unchanged (18 plans still resolves regret far
better than the 16 at 10–14 servers, and the alternative is dropping to a 3-rung ladder), but
it is a thinner margin than the original table implied.

**The cap is 250,000**, not the "100k" this section and the preset docstring both cited —
`MAX_PLACEMENT_COMBINATIONS_SKIP_DEFAULT` in `generate_gnn_datasets_fast.py`, exported as
`$MAX_PLACEMENT_COMBINATIONS`. The re-run peak (8,698 plans) is **3.5% of it**. Docstring
corrected 2026-08-19.

**The enumeration ceiling is not the binding constraint** — the sweep grows ~quartically but
stays inside the cap to ~100 servers. The *low* end binds instead: a 16-plan sweep at 10-14
servers makes regret far too coarse. Ladder set to train {20, 28, 40} / hold out {60, 80},
1.5-4x the largest training size, every label still a true sweep minimum.

**The candidates/task floor worry was unfounded.** Geometric-mean candidates/task grows
2.38 → 9.96 (4.19×) strictly monotonically across the ladder, replica-host nodes 7 → 29,
because `replica_server_pct = max(server_pct, 0.6)` is a *percentage* and scales with server
count.

#### Phase 2 — the network, in the graph (DONE)

`src/placement/network_graph.py` adds two entity types and four edge families to the
placement graph, behind `NETWORK_GRAPH_CONTRACT` (default `off`, so every existing cache,
checkpoint and corpus is untouched):

```
  [tasks] --candidate--> [platforms] --hosted_on--> [nodes]
                                       [nodes] --routes_via--> [links]
                                       [tasks] --routes_via--> [links]
                                       [links] --adjacent-----> [links]
```

**Only core links become entities.** Per `network_fabric.py`, access links carry one node's
traffic and are perfectly additive; core segments are the only shared, contended objects.
This is not just a fidelity choice — `GIN` aggregates with **sum**, so any degree that grows
with the cluster shifts embedding magnitudes with the cluster, and the transfer measurement
would be reading its own graph construction. That is Phase 0's confound reappearing as
structure. Attaching core *routers* as node entities would do exactly this (every node
attaches to one, so router degree ∝ N). Under `core`, every added degree is bounded by a
config constant — `n_core`, `attach_degree`, platforms-per-node — and none by N. Asserted
directly at 8/16/32/64 servers, alongside a test that no feature *column's* range drifts
with size (a bound catches a runaway; it does not catch a creep).

Scoring did not move: logits stay on task→platform edges, so `edge_attr` alignment and the
`to_undirected` reverse-edge pairing are untouched. The entities append **after** platforms,
leaving every pre-existing checkpoint's index layout intact; `forward()`'s platform slice
became bounded (`x[n_tasks : n_tasks+n_platforms]`) rather than open-ended.

Built once and called by **both** the cache builder and live inference — the `mp_parity`
discipline, applied up front rather than after a 12.4× regression. Verified on
`netc_multihop_v1_core4/ds_00000`: 40 node entities, 4 core-link entities, 458 route edges,
and **`task↔link`, `node↔link` and `link↔link` are byte-identical cache↔live**. Link
features carry real routing structure — two core links carry 75% of tasks and 17%/56% of
candidate routes, two carry nothing.

Serve-side guard closed: the encoders are visible in the weights, but *which contract built
their features* is not, so trainers record `network_graph_contract` in the checkpoint
sidecar and `load_gnn_model` refuses a mismatch. A model built with network entities and
handed a graph without them fails loudly rather than degrading into a silently different
(bipartite) model.

`scripts_cosim/test_network_graph.py` (31 tests); 143 pass across all affected suites;
cache↔live parity passes on dim24/dim22/dim14 with the contract off.

The parity failures this phase surfaced on the netc corpora turned out **not** to be a
Phase 2 problem, and not the problem they first looked like either. They are recorded as
their own lineage — see `cache_live_divergence_audit` below. Phase 2's own additions are
parity-clean.

#### Pre-registered gate — Phases 3-4 (registered BEFORE any corpus is generated)

Registered now because `--gate-coupled-fraction` is the cautionary precedent: a gate nobody
exercised, and a statistic that would have rejected the one configuration that worked.

**⚠ AMENDED 2026-08-19, before any corpus was generated** — see "the gate statistic
decision" below for the measurements. The original v1 text is kept struck through so the
amendment is auditable rather than silent; nothing had been generated, trained or gated
under v1, so no result changes hands.

~~v1: Gate on the **slope of additive-argmin regret against topology size**, per model.
**PASS** — pointwise regret grows monotonically across held-out sizes while GNN regret stays
flat within seed variance, and the gap widens monotonically. **FAIL** — both degrade at
comparable rates, or neither degrades.~~ **Falsified as a gate:** a decision rule of constant
expressive power satisfies that PASS condition on its own, by landscape drift alone.

**v2 — gate on the slope of `win_rate` against topology size, in excess of the drift
anchor:**

- **PASS** — GNN-vs-pointwise `win_rate` stays flat or rises across held-out sizes while the
  no-learning drift anchor over the same datasets does not, the separation exceeds the
  bootstrap CI at every size, and `regret_ratio_mean` agrees in sign.
- **FAIL** — `win_rate` tracks the drift anchor, or the two co-primary statistics disagree in
  sign, or the CI **excludes 0.5 on the pointwise side**. ~~or the CIs overlap 0.5 at the
  held-out sizes~~ — **amended 2026-08-19 (see "the power ladder" below): a straddling CI is
  an under-powered result, not a null one, and calling it FAIL manufactures a false negative
  at exactly the effect size this lineage expects.** Only a CI excluding 0.5 on the
  *reference's* side licenses "does not transfer".
- **VOID / ESCALATE (not FAIL)** — the CI straddles 0.5. Auto-escalates to the next
  pre-registered power tier if one can resolve the observed effect; otherwise reported as
  `INCONCLUSIVE_LADDER_EXHAUSTED` with the n it would need. Neither decides the lineage
  against the GNN.

Controls, none optional (1-3 pre-registered, 4-5 added by the amendment):

1. **≥5 seeds (was ≥3; re-derived 2026-08-19 — see "Seed count" below), distributions not
   point estimates.** ~~GIN training on CUDA is not
   bit-reproducible under a fixed seed~~ — **corrected 2026-08-19: it is not CUDA.** The
   non-determinism is in the GIN autograd path and fires on **CPU** too; `pointwise` is
   bit-identical run to run while `gnn_base`/`gnn_node` diverge in the *training loss*.
   Not intra-op threading (`OMP_NUM_THREADS=1` still diverges), not `PYTHONHASHSEED`.
   `torch.use_deterministic_algorithms(True, warn_only=True)` makes all three bit-identical
   and is now **on by default** in the harness (`--nondeterministic` to opt out).
   **Seeds run before that fix do not measure what they claim**: run-to-run noise was the
   *larger* term — `win_rate` moved 0.517 → 0.550 between two identical seed-44 commands
   against a seed-to-seed spread of only 0.517–0.533.
2. **Knative as a size-invariant reference**, so "the GNN held up" is distinguishable from
   "the task got easier at that size".
3. **`--gate-one-integer-repair` reported at every size.** If one count column repairs the
   pointwise model's failures at large sizes, the gap is the same degeneracy this repo has
   hit five times, not a transfer result.
4. **The no-learning drift anchor at every size** (additive-fit argmin + additive+one-integer,
   same held-out datasets, no training). Distinguishes "the GNN held up" from "the statistic
   drifted". Knative cannot substitute — its own quality may move with size.
5. **Power stated before the verdict.** Report `min_detectable_gap` and the `win_rate` CI at
   every size; a run below the power table is VOID, not FAIL.

**🔄 IN PROGRESS 2026-08-19 — Phase 4 corpus generation running on datalab at `tier_launch`
(900/size, 4,500 datasets total).** `TOPO_TRANSFER_V1_GRID`'s seed range bumped 30→75
(`scripts_cosim/generate_gnn_datasets_fast.py`) to hit 900/server_node_count uniformly across
all five sizes (`grid_total_datasets` confirms 4,500). Smoke-tested locally first (6
datasets, `--output-subdir` redirected away from the real corpus dir): backbone present (52
links, `n_core=12`, 1000 MB/s non-binding) on every dataset, `placements.jsonl` populated. A
local full-corpus run was started, then killed and moved to datalab for wall-clock (local
32-core estimate ≈9-10h; datalab CPU-amd has 16+ idle 128-core nodes). `src/`,
`scripts_cosim/` and this file rsynced to `/home/nikola.lukic/gnn-herosim` (datalab's
`feat/gnn-mp-residual` was clean at the same base commit, `852736c`, as local's uncommitted
diff — a plain overlay, no merge). Re-verified with a second smoke test on datalab itself
before submitting. New tool: `scripts_cosim/datalab/topo_transfer_v1_cosim.sbatch`, reusing
the generic `run_contention_regen_shard.sh` sharder with `GRID=topo_transfer_v1`,
`TOTAL_DATASETS=4500`, `NUM_SHARDS=50` (450 datasets/topology-variant means 50 shards splits
every variant into exactly 5 homogeneous-cost shards — the expensive `srv=80` variant does
not end up as one long serial shard), `ALLOW_NON_UNIQUE=0` (explicitly overriding the
sharder's netc-oriented default of 1 — the local smoke test succeeded at 100% without it).
Submitted as SLURM job **704238** (`sbatch --array=0-49`), 35 array tasks started
immediately across `os-cpu-slurm-{5,7-24}`.

#### Phase 3 landed — `topology_size` split mode (2026-08-19)

`gnn_necessity_ablation.py`'s `--split-mode` only had `canonical_parent` (random) and
`copy_shuffle` — no way to hold out topology sizes, so no run through this harness had ever
actually tested transfer. Added `split_ids_by_topology_size()` + `topology_sizes_by_parent()`
to `src/notebooks/non_unique_lib/training_contract.py`: server count isn't stored as a graph
attribute anywhere (`generate_infrastructure.py` never wrote it), so it's read back from each
dataset's `infrastructure.json` `network_maps`, counting non-client nodes
(`CLIENT_NODE_PREFIX`, single source of truth in `topology_features.py`). Wired in as
`--split-mode topology_size --train-sizes 20 28 40 --held-out-sizes 60 80`; `val` is drawn
only from the train-size pool, never held-out sizes, so model selection can't peek at the
transfer question. 11 new tests (`test_topology_size_split.py`).

**Also fixed while running it for real: the missing-plan hard-fail was checking the wrong
thing.** `eval_regret` raised `RuntimeError` on ANY predicted plan absent from the retained
placement sweep, on the assumption that only a corpus/harness bug could cause it. First real
run at held-out sizes 60/80 crashed at 120 epochs (not undertraining) with 265/619 `pointwise`
predictions missing. Diagnosis: **all 265 were collisions** (two tasks independently picked
the same node+platform) — a brute-force sweep correctly never enumerates a jointly-infeasible
combination, and `pointwise` has no mechanism to avoid one by construction (that's the whole
point of the ablation). `eval_regret` now splits `n_missing_plan` into `n_missing_collided`
(reported, not raised on — a first-class coordination-failure statistic) and `n_missing_clean`
(no collision, still missing — genuinely inexplicable, still raises). Schema 6 → 7.

#### Phase 4 — first real result, 5 seeds, `tier_launch` (2026-08-19/20)

Cache built via `NETWORK_GRAPH_CONTRACT=core_v1 TOPOLOGY_FEATURE_CONTRACT=size_invariant_v1`
on a **partial** corpus (3,194/4,500 datasets — generation was still running; caching raced it
and lost twice before landing, see GATE TOOLS and the datalab agent note below). Verified:
network entities non-zero (`net_node_features` [40,6], `net_edge_index` [2,536] on a sampled
graph), task feature dim 2 in the size-invariant `[0,1]` range.

**Seed 42 alone: both `gnn_base` and `gnn_node` PASS** (win_rate 0.545 / 0.531, CIs excluding
0.5 above, resolved not escalated) — the first PASS this lineage has ever produced.

**Pooled across all 5 seeds (`pooled_phase4_verdict`), the effect is not there:**

| model | pooled win_rate | 95% CI | verdict |
|---|---:|---|---|
| gnn_base | 0.4976 | [0.467, 0.529] | `INCONCLUSIVE_LADDER_EXHAUSTED` — effect ≈0.002, needs ~59,200/size |
| gnn_node | 0.4962 | [0.473, 0.520] | `INCONCLUSIVE_LADDER_EXHAUSTED` — effect ≈0.004, needs ~13,500/size |

Per-seed `win_rate` for `gnn_base`: 0.545 (42) / 0.496 (43) / 0.454 (44) / 0.476 (45) / 0.518
(46) — a spread that alone would read PASS, INCONCLUSIVE, FAIL, ESCALATE, and ESCALATE
respectively if read one seed at a time. This is precisely the noise the seed-calibration and
pooling work earlier in this session exists to catch, now catching it on a live result instead
of a retrospective one. `regret_ratio_median` sits at ≈0.998–0.999 across every seed for both
models — the co-primaries **agree** with each other (both read "no detectable difference"),
so this is a clean null, not a sign-disagreement artifact.

**Honest reading: on this partial `topo_transfer_v1` corpus, `gnn_base`/`gnn_node` and
`pointwise` pick the better plan at statistically indistinguishable rates on held-out topology
sizes.** `INCONCLUSIVE_LADDER_EXHAUSTED`, not `FAIL` — this does **not** say the GNN fails to
transfer, it says the effect (if any) is too small to resolve at this corpus size, and closing
that gap (~13,500–59,000 datasets/held-out size) is 15–65× `tier_launch`'s cost, not a
next-registered-tier decision. Two caveats on this specific result, not yet resolved:
generation was only 71% complete when this cache was built (3,194/4,500 — the full corpus may
tell a different story), and this is the ablation harness's own small internal GIN, not the
production `train_near_rtt.py` model. Frozen reports:
`simulation_data/gnn_necessity_topo_transfer_v1_seed{42..46}_20260819.json`.

**⚠ SUPERSEDED for `gnn_base` — see "Corpus generation, cache build, and the Phase 4 gate" below.**
A concurrent session ran the same pooling against the *complete* corpus (3,744/4,500) and got an
unambiguous answer: `gnn_base` pooled win_rate 0.456, CI [0.440,0.471], **FAIL** — the missing
29% of the corpus was enough to resolve what this partial run could only report as
under-powered. `gnn_node` is still not resolved (`ESCALATE`, needs n≥690). Left standing as a
historical record of a real methodological point: seed 42 alone read as a clean PASS on this
same partial corpus, and both the partial-corpus pooled result and the full-corpus one agree
that single-seed result did not hold up — first on pooling (this section), then on completing
the corpus (the section below).

#### Corpus decisions — settled 2026-08-19, before generation

Three probe findings needed a call before any `topo_transfer_v1` corpus could be generated.
All three are now fixed in the preset and in `gate_statistics.py`, not just written here.

**1. Backbone ON, at a deliberately non-binding 1000 MB/s.** The probe's blocking finding:
`--grid topo_transfer_v1` with default flags produced **`link_topology: null`**, because the
backbone block was written only when `--link-bandwidth-mbps` was passed, and
`build_network_graph_block` treats a missing fabric as *"a legitimate, silent no-op"*.
Training that corpus under `NETWORK_GRAPH_CONTRACT=core_v1` would have produced two
topology-blind models without a word of warning — the exact failure Phase 2 exists to
prevent. Measured on one generated dataset:

| preset | network nodes | core-link entities | network edges |
|---|---:|---:|---:|
| as it stood (no fabric) | 0 | 0 | 0 |
| with the grid-declared backbone | 20 | **12** | 32 |

12 link entities at `n_core=12` — Phase 2's bounded-degree property, confirmed on the corpus
that will actually be trained. The default now lives in the **grid preset**
(`backbone_defaults`), not in an operator's flag, because a grid whose entire question is
topology must not depend on someone remembering an argument. `--link-bandwidth-mbps` still
overrides.

Non-binding rather than contended, deliberately: this lineage asks whether the GNN uses
topology *structure* to generalize. Link contention is `link_contention_v1`'s question and it
is already answered (real, but 0.08–0.35% regret). Stacking a known-small, known-noisy
mechanism onto a signal being resolved at MDG ≈ 0.02 is how `netc_hotspot_v1` lost
attribution. **Contention-under-transfer is a follow-on lineage, not a rider on this one.**

**2. `n_core` stays FIXED at 12; it does not scale with servers.** So the transfer axis is
**candidate-set growth over a fixed-complexity fabric** — candidates/task 2.38 → 9.96 (4.19×)
while core links/route go 3.13 → 3.02 and routes using ≥1 core link 92% → 91%. Scaling
servers hangs more nodes off the same ring without lengthening routes.

**The claim this corpus can support is therefore "generalizes across candidate-set growth",
not "generalizes to larger networks"** — narrower, and to be reported that way. Scaling
`n_core` is defensible in principle but is **untested against Phase 2's aggregation-invariance
property** (GIN sums, so any degree growing with N shifts embedding magnitudes with N);
testing it honestly means re-running the degree-bound asserts at every rung, which is a
separate phase with its own budget. Folding it in here would mean a negative result could not
say which half failed.

**3. The power ladder — enter at tier 0.02, with escalation pre-committed now.**
`PHASE4_TIERS` in `gate_statistics.py`, fixed before any corpus exists, for the same reason
v1's criterion was: choosing a threshold *after* seeing a borderline number is how a gate gets
falsified. Tier 0.02 is a cheap first pass (360/held-out size, ≈ 3.6 h wall-clock at 32 cores,
+~10% with the backbone), **not a standalone decision** — a straddling CI there auto-escalates
to tier 0.01 rather than being reported as "topology transfer failed".

**Seed count, re-derived after the determinism fix — and the answer is "spend on datasets,
not seeds".** The old ≥3-seed control was set against a spread that measured run-to-run
autograd noise, so it had to be re-measured, not reused. Five deterministic seeds on
`shallow_v1` (frozen above):

| quantity | value |
|---|---|
| across-seed sd of `win_rate` | **0.0508** |
| sd from test-split resampling alone at n=30 | 0.0913 |
| ratio | **0.56** |

**Across-seed sd is *below* what pure test-split resampling would produce**, so seed-to-seed
variation here is dominated by which 30 datasets land in the split — not by initialization.
Two consequences:

- **More seeds is the wrong purchase.** Resolving the mean effect at this split size would
  need ~**19** seeds; the same resolution comes far cheaper from a larger held-out set.
- **The prior intuition that ≥3 was overkill is falsified** — 5 seeds still leave the CI on
  the mean straddling 0.5. Seeds were never the binding term in either direction.

Phase 4 allocation: keep seeds at **5** (cheap, and they still buy the variance estimate and
the tail-behaviour spread that seed 44 exposes) and put the budget into datasets per held-out
size, which is what the power ladder below is denominated in.

**A units correction the implementation forced.** The MDG table above is in *regret-gap*
units; the primary statistic is `win_rate`. Doing the power arithmetic properly in win_rate
units (CI half-width of a proportion, ≈1.96·√(0.25/n)) against the effects actually observed:

| observed effect \|`win_rate` − 0.5\| | datasets/held-out size needed | covered by |
|---|---:|---|
| 0.033 (seed 42) | ~880 | tier 0.01 (1,600) |
| 0.017 (seeds 43/44) | **~3,400** | **no registered tier** |

So if the true effect sits at the *bottom* of the observed range, **even tier 0.01 will not
resolve it** (~16 h wall-clock buys an answer only for the top half of the range). The gate
returns `INCONCLUSIVE_LADDER_EXHAUSTED` with the required n in that case — never `FAIL`.

**✅ DECIDED 2026-08-19 — `tier_launch` registered at 900/held-out size, not the ~3,400/size
tier.** 900/size covers the *stronger* observed effect (0.033, needs ~880) at roughly
`tier_0.02`'s already-budgeted cost, and is the tier Phase 4 corpus generation actually
launches at — `tier_0.02` is a cheap first pass, and a straddling CI there is expected to
escalate to `tier_launch`, not `tier_0.01`. The ~3,400/size tier for the *weaker* effect
(0.017) is deliberately **not** pre-registered: at ~35–40h wall-clock per held-out size it is
a datalab allocation, not a speculative local run, and whether it is worth running depends on
whether `tier_launch`'s result is itself informative. It stays documented here as a sized,
known escalation path — the trigger to ask for datalab time is `tier_launch` itself coming
back `INCONCLUSIVE_LADDER_EXHAUSTED`, not before.

#### Corpus generation, cache build, and the Phase 4 gate — `FAILED` for `gnn_base` 5/5 seeds (2026-08-20)

**Corpus generation completed.** SLURM job 704238 (50-way array, `topo_transfer_v1_cosim.sbatch`)
finished: 4,500 datasets total (900/topology-size at 20/28/40/60/80 servers), 756 legitimately
`SKIPPED (infeasible)`, **3,744 successfully generated** with full `placements/placements.jsonl`
sweeps. A repair pass (`refresh_optimal_full_stats.py --repair`) fixed 550 datasets that had an
`optimal_result.json` but were missing `system_state_captured_unique.json` — a race condition
where the cache-build's first read hit a still-running generation shard (this is the **fourth**
time this exact bug has bitten; see the GATE TOOLS table). Graph cache built successfully at
`simulation_data/graphs_cache_topo_transfer_v1` (SLURM job 705771, 128GB mem, CPU-amd,
`NETWORK_GRAPH_CONTRACT=core_v1` + `TOPOLOGY_FEATURE_CONTRACT=size_invariant_v1` +
`QUEUE_FEATURE_CONTRACT=scale_invariant_v1`): 3,744 graphs, 100% valid labels, avg 88.4
edges/graph, RTT hash table 120.5M entries.

**Phase 3 implemented.** `--split-mode topology_size` in `scripts_cosim/gnn_necessity_ablation.py`:
train on `server_node_count ∈ {20,28,40}`, hold out `{60,80}`; the validation slice is drawn only
from train-size parents and never touches a held-out size. Guarded by
`scripts_cosim/test_topology_size_split.py` (11 tests, passing). The split uses a plain
`random.Random` shuffle rather than `sklearn.train_test_split`, because datalab's `gnn`
micromamba environment currently has a broken scipy/sklearn ABI (scipy 1.17.1 vs sklearn 1.6.1,
incompatible compiled extensions) — a pre-existing, shared-environment problem, deliberately
**not** fixed here since other sessions depend on that environment and other splits
(`split_ids_by_canonical_parent`, used by every non-`topology_transfer_v1` lineage) still use
sklearn on purpose, so existing frozen reports keep their exact shuffling.

**A merge regression, found and fixed before the reported run.** Two independent uncommitted
working trees (local repo, datalab checkout) had each separately extended
`gnn_necessity_ablation.py` — datalab's copy gained network-entity model support
(`use_network_entities` on `AblationModel`, **not wired into any of the three trained configs
below — dead capability**, not exercised by this run) and a label-provenance preflight audit
(`audit_label_provenance`); local's gained the `topology_size` split. Merging datalab's version
as the base with the split layered on top silently regressed a correctness fix in the eval loop:
"predicted plan absent from the placement sweep" stopped distinguishing a plain task-collision
(expected, not a bug) from a plan missing with no collision to explain it (a real corpus/harness
bug), and crashed on the former. Fixed within this session — `eval_regret` now reports separate
`n_missing_collided` / `n_missing_clean`, and only `n_missing_clean` triggers the fail-loud
`RuntimeError`. The first submission (SLURM job 705777) hit this bug and all 5 array tasks
failed after ~50 min; job 705834 below is the corrected rerun. Recorded as its own row in the
GATE TOOLS table — this is the same class of loss as `mp_parity`'s train/serve split: two
diverging copies of one file silently dropping each other's fixes.

**The gate run.** SLURM job 705834 (5-task array, GPU-a100, ~2h47m/seed), 5 pre-registered seeds
(42–46), `--split-mode topology_size --train-sizes 20 28 40 --held-out-sizes 60 80
--power-tier tier_launch --epochs 120 --models pointwise gnn_base gnn_node`. Label-provenance
audit **passed 3,744/3,744 on every seed** (`label_regret_mean = 0.0`, every cached label is its
dataset's true sweep minimum). `n_train=2280, n_val=403, n_test≈1061` graphs/seed. Frozen reports:
`simulation_data/topo_transfer_v1_phase4_seed{42,43,44,45,46}.json` (schema_version 5).

Per-seed paired `win_rate` vs. `pointwise` (`n_paired` ≈ 587–600; verified against the JSON
`paired_comparisons` / `phase4_verdicts` blocks, not just stdout):

| seed | `gnn_base` win_rate [95% CI] | verdict | `gnn_node` win_rate [95% CI] | verdict |
|---|---|---|---|---|
| 42 | 0.469 [0.447, 0.489] | **FAIL** | 0.480 [0.459, 0.501] | ESCALATE (effect 0.020, needs n≥617) |
| 43 | 0.436 [0.415, 0.457] | **FAIL** | 0.412 [0.390, 0.433] | **FAIL** |
| 44 | 0.442 [0.422, 0.462] | **FAIL** | 0.500 [0.482, 0.517] | INCONCLUSIVE_LADDER_EXHAUSTED (effect ≈0.000) |
| 45 | 0.454 [0.435, 0.473] | **FAIL** | 0.466 [0.447, 0.486] | **FAIL** |
| 46 | 0.478 [0.459, 0.497] | **FAIL** | 0.497 [0.477, 0.518] | INCONCLUSIVE_LADDER_EXHAUSTED (effect ≈0.003, needs n≥37889) |

`gnn_base` also loses on `regret_gap_mean` (negative in every seed, i.e. `pointwise` has the
lower — better — regret) and its co-primary sign agrees with `win_rate` in all 5 seeds, so
nothing here rests on a single-seed sign flip the way `mp_parity`'s residual did.

**Pooled across the 5 seeds (`pooled_phase4_verdict`, same tool the partial-corpus run above
uses), the per-seed picture holds — this is not seed noise:**

| model | pooled win_rate | 95% CI | verdict |
|---|---:|---|---|
| gnn_base | 0.456 | [0.440, 0.471] | **FAIL** — CI excludes 0.5 below, `co_primary_sign_agree=False` |
| gnn_node | 0.471 | [0.439, 0.502] | `ESCALATE` — effect ≈0.029, needs n≥690 (next tier above `tier_launch`) |

Unlike the partial-corpus run's pooled result above (both `INCONCLUSIVE_LADDER_EXHAUSTED` on a
71%-complete corpus), this is the **complete** 3,744/4,500 corpus and `gnn_base`'s pooled CI is
unambiguous. This supersedes the partial-corpus Phase 4 section above for `gnn_base`: FAIL, not
inconclusive. `gnn_node` is still not resolved either way.

**The likely cause: the held-out topology sizes carry essentially no coupling to exploit.**
`greedy_baseline` (additive-argmin, no training) shows **0/1,022 held-out test datasets with
regret > 1%, on every one of the 5 seeds** — the same signature this repo has hit five times
before (`graph_structure_physics`, `shallow_v1`/`shallow_longexec_v1`, `contention_v4_v5`,
`link_contention_v1`, `mp_parity`). At 60/80-server topologies the task-placement problem the
corpus poses is additive, so a model with strictly *more* expressive power than an additive
baseline (GIN) has nothing extra to win on, while it still pays a generalization tax the
pointwise model — which has fewer parameters coupled to graph structure — does not.

**Control 4 (the no-learning drift anchor) is NOT present in the frozen reports.** Schema 5's
fields are `cache, corpus_root, coupled_dataset_ids, coupled_results, coupled_threshold, epochs,
greedy_baseline, label_audit, models, n_graphs, n_test, n_train, n_val, paired_comparisons,
paired_reference, phase4_verdicts, power_tier, results, schema_version, seed, split_mode,
test_fraction` — there is no `drift_anchor` (or equivalent) key. The pre-registered control that
was meant to distinguish "the GNN held up" from "the statistic drifted" was never wired into this
harness run, so **this gate result stands on `win_rate` + `regret_ratio_mean` sign-agreement
alone, not on the full v2 control set.** That does not change the `gnn_base` verdict (a CI
excluding 0.5 below is a FAIL under the v2 rule with or without the anchor), but it does mean the
`gnn_node` ESCALATE/INCONCLUSIVE results have one fewer corroborating signal than pre-registered,
and the anchor should be added before spending the ~3,400/size `tier_0.01` budget this file
documents above.

**Verdict: this is a gate FAILURE for `gnn_base`, unambiguously — 5/5 seeds, CI excludes 0.5 on
the pointwise side every time, effect sizes 0.022–0.088, all in the direction of the reference
winning.** `gnn_node` never PASSES in any seed either: 2/5 FAIL outright (43, 45), 3/5
INCONCLUSIVE-but-pointed-at-null (42 ESCALATE at 0.020, 44 and 46 at effects ≈0.000–0.003 that
would need n≥37,889 to resolve — i.e. indistinguishable from parity, not evidence of a hidden
win). Do not read the `gnn_node` non-FAILs as "might still win" — the observed effects for the
three non-FAIL seeds are converging on zero, not on a positive gap obscured by noise. Under this
lineage's own pre-registered rule, `tier_launch` licenses the conclusion "does not transfer" for
`gnn_base`; `gnn_node` would need the (documented, not-yet-approved) ~3,400/size escalation to
settle even the null it is trending toward, and given the 0/1,022 coupling finding above there is
no positive prior that spending it would change the sign. Candidate-set-growth topology transfer,
as this corpus supports the claim, is **falsified for `gnn_base`** and shows no positive signal
for `gnn_node`.

### topology_transfer_v1 — the `gnn_topo` arm and the co-sim/live-gate scope gap (2026-08-20)

**Why this pass happened.** The FAIL above was for `gnn_base`/`gnn_node`, and neither config in
`all_configs` (`gnn_necessity_ablation.py`) ever set `use_network_entities=True`. That flag is
the *only* pathway that gives the model access to backbone/link graph entities
(`net_node_features`, `net_link_features`, `net_edge_index` via `src/placement/network_graph.py`,
contract `core_v1`) — `gnn_node`'s `use_node_edges=True` only adds same-node platform↔platform
edges, not network topology. So the original gate never actually tested whether topology-aware
message passing helps; it tested two topology-blind bipartite GINs against pointwise. The graph
cache (`graphs_cache_topo_transfer_v1`) was already built under
`NETWORK_GRAPH_CONTRACT=core_v1 TOPOLOGY_FEATURE_CONTRACT=size_invariant_v1`, so the data needed
was present and unused.

**Fix and re-run.** Added a fourth arm to `gnn_necessity_ablation.py`'s `all_configs` and
`--models` choices: `gnn_topo = dict(use_gin=True, use_node_edges=False,
use_network_entities=True)`. Re-ran the identical pre-registered gate (topology_size split, train
sizes 20/28/40, held out 60/80, `tier_launch`, 120 epochs, seeds 42–46) via SLURM job 706415 on
datalab (GPU-a100s, ~2h/seed, all 5 array tasks completed cleanly, no errors). Training loss
curves for all 5 seeds are smooth and monotonic (1.05–1.07 → 0.49–0.50, no NaN/stall); the one
lower-performing seed (43: win_rate 0.393 vs. the other four's 0.43–0.49, top1_acc 0.819 vs.
0.82–0.85) converged identically to the rest — its weaker result is ordinary seed variance, not a
training pathology. Frozen reports:
`simulation_data/topo_transfer_v1_phase4_topo_seed{42,43,44,45,46}.json`.

| seed | `gnn_topo` win_rate | 95% CI |
|---|---:|---|
| 42 | 0.486 | [0.465, 0.508] |
| 43 | 0.393 | [0.371, 0.415] |
| 44 | 0.471 | [0.451, 0.491] |
| 45 | 0.434 | [0.413, 0.455] |
| 46 | 0.460 | [0.439, 0.482] |
| **pooled (5 seeds)** | **0.449** | **[0.417, 0.481]** |

Pooled `regret_ratio_median = 1.0004` (per-seed range 0.9989–1.0022, i.e. indistinguishable from
1.0 — `gnn_topo` and `pointwise` have essentially identical regret magnitude, not just a
placement-choice disagreement). Co-primaries agree (`win_rate` and `regret_ratio` both say the
reference wins). Effect is **resolved, not underpowered**: `required_n` for this effect size is
230, actual pooled `n_paired` is 593. Pre-registered verdict at `tier_launch`: **FAIL** — CI
excludes 0.5 below. **Giving the model backbone/link topology access did not close the gap.**

**The missing controls, now run (post-hoc, local, no retraining):**

- **Drift-anchor control** (`scripts_cosim/drift_anchor_check.py`, new this session): the
  no-learning identical-capacity pair (additive-fit argmin vs. additive+one-integer-repair,
  `separability_diagnostic.variance_decomposition`) drifts by only ≈0.005–0.006 across held-out
  sizes 60→80 on this corpus — nowhere near `shallow_v1`'s pathological 2.58 drift. This corpus's
  size axis is not obviously landscape-broken the way `shallow_v1`'s was.
- **`size_invariant_v1` feature-degeneracy check** (ad hoc, `src/placement/topology_features.py`
  `SourceFeatureContext.feature()` sampled directly per held-out dataset): not degenerate at
  60/80 servers — variance ≈0.01, 23 distinct values, mean 0.06–0.08 at both sizes. Rules out
  feature collapse as an explanation for the 0/1,022 zero-coupling finding above.
- **Per-size breakout of the original `gnn_base`/`gnn_node` result**
  (`scripts_cosim/topo_transfer_v1_per_size_gate.py`, new this session — re-buckets the frozen
  reports' `results[model]['per_ds']` by `server_node_count` recovered from
  `infrastructure.json`, no retraining): the pooled FAIL is **not a widening-with-size effect**.
  `gnn_base` win_rate goes 0.454 (size 60) → 0.460 (size 80), `gnn_node` goes 0.464 → 0.474 — both
  move *toward* parity at the larger held-out size, the opposite of a transfer-degradation
  signature, and both deltas (~0.006–0.010) are the same order of magnitude as the drift-anchor's
  own noise floor.

Net read: this is now a properly-controlled FAIL for the lineage's actual hypothesis
(topology-aware GNN vs. pointwise), not an artifact of an untested config, a degenerate feature,
a broken size axis, or a genuinely widening size-transfer gap — the loss looks like a small,
roughly size-flat generalization gap that giving the model topology access did not fix.

**⚠ Co-sim-only scope and live-gate traceability — read before planning further work on this
lineage.**

1. **Every result above, across all 20 seed-runs (5 seeds × 4 arms), comes from the co-sim
   brute-force pipeline only.** Each dataset is a **4-task synthetic snapshot**
   (`workload.json`: `rps=2, duration=1`, 4 `events`) — fixed regardless of `server_node_count`;
   only the *cluster* scales (20→80 servers), never the *workload*. This is a deliberate property
   of the `topo_transfer_v1` grid (brute-force enumeration over more than ~4 simultaneous tasks
   is not tractable), not a bug — but it means "topology size" in this lineage has always meant
   "cluster size around a fixed 4-task decision," never "workload scale." This project's actual
   goal (CLAUDE.md) is co-sim generation → training → **live-gate evaluation on real workloads**
   (e.g. `data/nofs-ids/traces/workload-200-200.json`, 800,413 events, `rps=200, duration=200s`).
   `topology_transfer_v1` has never reached that step. **This is not a documented methodology
   choice (do not conflate it with `regime_b`, which is an unrelated, already-`FALSIFIED`
   cold-burst physics lineage) — it is simply a step this lineage has not yet taken.**
2. **No lineage in this repo has ever live-gated across mismatched train/eval topology sizes.**
   Confirmed by search: existing live-gate scripts
   (`important/run_contention_v2_live_gate_one.sh`, `important/run_wssm_expanded_live_gate_one.sh`,
   etc.) take a single infra config used for both training assumptions and the live run; there is
   no train-size vs. eval-size plumbing anywhere in the harness. `src/executesimulation.py` takes
   one `--config` that fixes the live infra's node count and **does not cross-check it against
   any checkpoint's trained topology size** — running a model at the wrong size would fail
   silently, not loudly. Architecturally nothing blocks this (`PointwiseEdgeMLP` and the GNN's
   bipartite `build_inference_graph` are both candidate-pair-based, not fixed-size vectors), it
   has simply never been attempted for any lineage.
3. **No trained checkpoint from this lineage exists on disk.** `AblationModel`
   (`scripts_cosim/gnn_necessity_ablation.py`) has no `torch.save`/state-dict-persisting call
   anywhere in the file — every one of the 20 seed-runs (`pointwise`/`gnn_base`/`gnn_node` ×
   seeds 42–46, then `pointwise`/`gnn_topo` × seeds 42–46) trained a fresh model in-process,
   evaluated it, and discarded the weights. Only the eval-summary JSONs
   (`simulation_data/topo_transfer_v1_phase4{,_topo}_seed{42..46}.json`) survive. **There is
   nothing to deploy for a live-gate run yet.**
4. **`AblationModel` is not wired into the production live-serving path.** It is a standalone
   class defined only in `gnn_necessity_ablation.py`; `use_network_entities` and the
   backbone/link entity pathway do not exist anywhere under `src/policy/gnn/` or
   `src/policy/gnn_hetero/` (grep confirms zero hits). `set_models()`-based live inference
   (CLAUDE.md's "Model loading uses `set_models()`") has no code path for this config today.

**Before running any of this on real workloads (planned for a future session), that session will
need, in order:**
   a. Add checkpoint saving to `gnn_necessity_ablation.py` (or a follow-on script) — persist
      `state_dict()` plus a training manifest per run: arm name/config (`use_gin`,
      `use_node_edges`, `use_network_entities`), seed, `split_mode`/`train_sizes`/
      `held_out_sizes`, cache dir, and the corpus grid name (`topo_transfer_v1`) — so a saved
      checkpoint can be matched, unambiguously, to the exact topology size(s) it was trained on
      before it is ever pointed at a live `--config`. Nothing today enforces that match; a
      silent size mismatch would not raise.
   b. Either port `use_network_entities` support into the production scheduler
      (`src/policy/gnn/scheduler.py` / `gnn_hetero/scheduler.py`) or build a small live-inference
      adapter around `AblationModel` directly.
   c. Build (or select existing) infra configs at each topology size of interest compatible with
      `src/executesimulation.py --config`, and decide which real trace(s) under
      `data/nofs-ids/traces/` to run live-gate against (existing live-gate scripts in this repo
      use `workload-125-225.json`, not `workload-200-200.json` — neither has been used with a
      topology-size-varying infra before).

### topology_transfer_v1 — unblocked, and two cost estimates corrected (2026-08-21)

No new gate result. This closes the lineage's §a blocker and re-costs its §b, both by
measurement rather than by plan.

**§a — weights are now persisted.** `gnn_necessity_ablation.py --save-checkpoints DIR` writes
`<arm>_seed<N>.pt` plus a `<arm>_seed<N>.contract.json`. Off by default, so an eval-only run
is byte-unchanged. The sidecar is the substance, not the `.pt`: a checkpoint without one is
read as `{}` by `executesimulation._read_checkpoint_sidecar`, and every downstream contract
check then adopts its default silently — `legacy_v0`, `src_index_v0`, no infra provenance, no
record of which message passing it was fitted with. The contract records the queue / topology /
network-graph contracts, feature layout, arm config, seed, and — for the `topology_size` split
— **which sizes were trained on and which were held out**. A checkpoint that cannot say which
sizes it never saw cannot be used to test transfer to those sizes, which is the whole
hypothesis. 14 tests in `tests/test_ablation_checkpoint.py`.

**§b — the serving port is a rename, not a build.** The plan costed a production
`use_network_entities` serving path as a multi-session job and assumed `gnn_base`/`pointwise`
"already load through `src/policy/gnn/scheduler.py`". Measured:

| | result |
|---|---|
| `AblationModel` vs `TaskPlacementGNN` state dicts | **31 keys each, 15 shared, same shapes** |
| difference | three top-level module names: `task_enc`→`task_encoder`, `plat_enc`→`platform_encoder`, `scorer`→`edge_scorer` |
| renamed load, `mp_residual=False` (production **default**) | `load_state_dict(strict=True)` **succeeds, no error** — and max \|Δlogit\| **0.196**, different argmaxes |
| renamed load, `mp_residual=True` (`mp_gate` inits to 1.0) | **max \|Δlogit\| 0.0**, identical argmaxes, `gnn_base` and `gnn_node` both |

`AblationModel` is unconditionally `x0 + gin(x0)`; production applies the residual only under
`mp_residual`, whose default is `False`. **So the cheap port also contains a silent
wrong-numbers path that nothing in the stack catches** — precisely the class this file's
checkpoint-contract rows keep recording. The verified port now ships in the contract as a
`serving_port` block (target class, key rename, constructor kwargs), with a test that fails if
the residual ever stops changing the output, so the guard cannot pass vacuously.
Recorded honestly: `mp_node_edges_candidates_only=False` comes from reading
`AblationModel.forward`, **not** from the equivalence check — that used a fully connected
bipartite graph, where the flag is a no-op.

**Remaining cost of the partial gate:** the ~14 GPU-hours (train `gnn_base` + `pointwise` with
`--save-checkpoints`, mint live cells at 60/80 servers, run the 15-task gate per size). Not
launched — it is a large speculative spend on a lineage already `FAILED`, and it would contend
with the siv1 retrain for GPUs. Nothing else blocks it.

**A6 (`soft_combo` live retest) is NOT viable as planned — checked, not assumed.** The plan
called it "the only item testable today with zero training" and flagged one caveat to verify.
Two hold, and either is disqualifying: (1) neither
`near-rtt-v2-regime-b-oracle-split-cosim-dim16-{ce-only,soft-combo-conc}.pt` has a
`.contract.json`, so both would serve blind under adopted defaults; (2) both take **16**
platform features (`platform_encoder.net.0.weight` is `(64, 16)`) against the siv1 gate cells'
**14** — before even reaching the `platform_reuse_v1` vs `node_disk_v2` physics mismatch.
Do not spend time on A6 without first retraining that pair under a recorded contract.
