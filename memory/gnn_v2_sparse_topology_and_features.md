# GNN v2 Win Zones — Co-sim Grid, Feature Redundancy, Disk Warmth

**Last Updated:** 2026-06-11 (v0.28.1 — B1 shipped)  
**Companion docs:** `memory/placements_jsonl_required.md` · `memory/cosim_grid_and_regen.md` · `memory/cosim_warmth_gap.md` · `memory/warmth_model.md` · `memory/storage_contention.md` · `memory/compare.md`

> **One-sentence summary:** v2 retrain moves optimal policy from “spread to dodge phantom re-pulls” to “disk-warm consolidation + hub routing under real contention.” GNN wins concentrate on **skew/topology** (bipartite k>b) and **batch coupling via GIN** — not physical multi-hop graphs. Merged **824-cache** adds low conn {0.25–0.35} but **still no `degree_skewed_core`**. atomic21 **drops `src_norm`**, seq recache **`is_warm` degenerate**, **`has_function` absent**. Skew3 live: MLP **2/3** until features + hub grid land.

---

## Strategic Picture

```
v1 label world:  spread > co-locate (fake re-pulls punished)
v2 label world:  co-locate on warm disk OK; spread only for first-wave cold / FilterStore

GNN wins where:
  (A) topology is asymmetric     → skew, hubs, bipartite
  (B) decisions are sequential   → batch loop + node state
  (C) shared_fate ↔ queueTime    → contention finally in labels

MLP wins where:
  (D) symmetric, many replicas, pull already cheap → flat ranking problem
```

---

## 1. Co-sim Today and Sparse / Skew Support

### How co-sim runs

Each dataset is built by `scripts_cosim/generate_gnn_datasets_fast.py`:

1. **Grid sweep** → per-cell `space_with_network.json` + workload template + deterministic `infrastructure.json` (topology, replicas, prewarm queues).
2. **Brute-force placement** via `execute_brute_force_optimized` — enumerates placement combos, scores RTT with the determined batch scheduler (`batch_size = num_tasks`, currently 4).
3. **Labels** = optimal placement RTT from `best.json` / `placements.jsonl`.
4. **v2 defaults**: `warmth_physics="node_disk_v2"`, `defer_cold_replica_init` from base config, fast-forward warmup optional.
5. **Fast regen path**: `GNN_CAPTURE_DATASET_STATE=0` (no inline SSC), `COSIM_SUPPRESS_SIM_PRINTS=1`, then mandatory `refresh_optimal_full_stats.py --repair --force` before recache — **but repair never replaces `placements/placements.jsonl`**. JSONL = full placement–RTT sweep for `rtt_chunk_*.pkl`; always persist before deleting `.bf_scratch`. `memory/placements_jsonl_required.md`.

### Current v2 grid (500 datasets)

| Axis | Values | Count |
|------|--------|------:|
| Connectivity | **0.50 only** (Erdős–Rényi `connection_probability`) | 1 |
| Replicas + preinit | (1,2,0,0), (2,2,0,0), (1,3,0.3,0.5), (2,3,0.3,0.5), (2,2,0.5,0.7) | 5 |
| Seeds | 101–120 | 20 |
| Queue regimes | pois16, norm22, pois28, norm35, uniform20_80 | 5 |
| Workload | 4 tasks, 10 templates cycling dnn mixes | — |

**Total: 500.** The `1060_warmth_v2` dir name is historical.

**Not in warmth-only grid:** `degree_skewed_core`, asymmetric 5/30ms latency, bipartite hub sweeps (125–225).

### Sparse expansion + merged cache (2026-06-11)

| Item | Value |
|------|-------|
| Grid preset | `--grid sparse_warmth_v2` in `generate_gnn_datasets_fast.py` |
| Conn | **{0.25, 0.30, 0.35}** × reps {(1,1),(1,2),(2,2)} × queues × seeds 121–133 |
| Count | **351/351** → `gnn_datasets_4tasks_sparse_warmth_v2` |
| Merged cache | `graphs_cache_warmth_v2_sparse_merged` — **824 graphs** (473 warmth + 351 sparse) |
| Finetune | `near-rtt-v2-warmth-sparse-finetune-dim14-ce-only.pt` — val acc **25.6%** (offline weak) |
| **Still missing** | **`degree_skewed_core`**, asymmetric latency — Erdős–Rényi only, not hub topology |

**Does not transfer:** bipartite k>4 live wins require hub geometry in **co-sim labels**, not sparse conn alone.

### Where live sparse/skew eval lives

Skew configs are separate JSONs under `simulation_data/normal_sim_sweeps/atomic21_skew_configs/`, e.g.:

- `default_20_20_degree_skew.json` — `degree_skewed_core`, k_core=4, 40% hub-seekers, p_core=0.95 / p_periphery=0.15
- `05_sparse_40_40_p25_degree_skew.json` — same skew topology + lower preinit (40%/60%) + `connection_probability: 0.25`

`generate_infrastructure.py` builds hub topology: first `k_core` servers forced to xavier (fast hubs); hub-seeker clients connect to core with high probability; periphery clients connect to non-core servers.

**Gap:** Merged 824 adds low conn but **not hub geometry**. GNN skew live wins need **`degree_skewed_core` in co-sim labels**.

### Highest-leverage additions (grid + features)

**Grid (training distribution)**

| Add | Why |
|-----|-----|
| `topology.type = degree_skewed_core` with k_core ∈ {2,4,6} and hub_seeker_fraction ∈ {0.3, 0.5, 0.8} | Matches live skew-4 / tiered-hub / bipartite sweeps |
| Connectivity 0.25 + 0.80 | Old 1060 strength; periphery isolation (partially in sparse 351) |
| (1,1,0,0) replica row | Scarce-replica co-location pressure (in sparse grid) |
| Asymmetric latency 5ms core / 30ms periphery | Matches bipartite eval |

**Features (observability)**

| Feature | Role |
|---------|------|
| **`src_norm` (restore in seq recache)** | Hub-seeker vs periphery when edge sets overlap |
| **`node_disk_hit` / `has_function`** | v2 consolidation policy — dim 13 today reserved |
| Fix seq **`is_warm`** | Sandbox match, not replica flag |
| **`edge_latency`** | Keep — primary topology signal |

### What the graph sees for topology (dim14 / atomic21)

From `prepare_graphs_cache_seq.py` + `feature_builder.py`:

- **Task features:** seq recache **2-d** (no `src_norm`); hom `prepare_graphs_cache.py` **3-d** with `src_norm` — regression is **seq/atomic21 path only**
- **Graph topology:** **Bipartite task→platform edges only** — no client→hub physical edges; `edge_index` connects tasks to feasible platforms (`prepare_graphs_cache_seq.py`)
- **Per-edge latency** encodes client→server reachability for that candidate only
- **No explicit hub bit, degree, seeker class, or candidate-set size.**
- Message passing can aggregate queue/cold state on hub nodes, but **client identity is blind** — the model cannot distinguish hub-seeker vs periphery clients except via which edges exist and their latencies.

- Message passing aggregates queue/cold on **shared hub platform nodes** via GIN — **not** multi-hop routing over network topology
- **Expert trap:** “Multi-hop awareness” = implicit latency + GIN mixing on overlapping candidates — **not** message passing over physical hub-and-spoke edges

**GNN hetero scheduler** (`gnn_hetero/scheduler.py`) still uses **dim22-style** platform layout and **3-d tasks with `src_norm`** — train/serve parity risk if live GNN path ≠ recached atomic21.

---

## 4. GNN vs MLP — what actually separates them (code-grounded)

| Mechanism | GNN-only? | Evidence |
|-----------|:---------:|----------|
| **GIN batch coupling** on shared hub candidates | **Yes** | Platform embedding = f(all batch tasks on that plat); MLP scores edges in isolation |
| **Same decode loop** | No | `MLPBatchScheduler` inherits `GNNScheduler`; `decode_sequential_placement` argmax with **frozen logits** |
| **`src_norm` / client identity** | No (live) | dim22 MLP already has `src_norm` in skew/bipartite sweeps; cache restore fixes **GNN train/serve** |
| **`node_disk_hit`** | No | Same bit on every edge row if wired to dim13/dim22 — both models use it |
| **FilterStore avoidance at schedule time** | Weak | queue dim 7 = 0; shared_fate saturates; no pending-pull feature |
| **Offline edge accuracy** | No | GNN task_acc ~69% vs MLP edge acc ~68% — comparable; GNN `val/acc` ~23% is **harder joint target** |

### Phase boundary (live numbers — bipartite v1, 125–225)

| Regime | GNN vs MLP | Sums (9 cfg) |
|--------|------------|--------------|
| k > b (k=6,8) | GNN **6/6** vs MLP | GNN **17.90M** · MLP **21.49M** (−17%) |
| k = b (k=4) | Mixed; Knative competitive | 3-way GNN 5 · Kn 3 · MLP 1 |
| Uniform / triangle | MLP ≈ GNN | MLP +0.4% vs dim14-ce anchor |

**Knative** beats both at k6 seek50 (**1.45M** vs GNN 1.66M, MLP 3.46M) — Regime B per-arrival can win coordination band.

**warmth_v2 skew3 (2026-06-11):** node_disk_v2 physics but **no disk feature in cache** — MLP old wins **2/3**; v2 GNN sparse **1.23M** vs Kn **0.93M**.

### Minimum separation stack (ordered)

1. **[NEXT]** `skew_warmth_v2` co-sim grid — `degree_skewed_core` + 5/30ms (~288 ds)  
2. **[DONE]** `node_disk_hit` dim 13 + `capture_disk_snapshot()` → `disk_snapshot_by_task_type` in SSC  
3. **[DONE]** `src_norm` in seq recache (3-d tasks) + live `feature_builder` (no zero-pad)  
4. **[DONE]** seq `is_warm` = sandbox `previous_task_type_name` (not replica flag)  
5. **[NEXT]** `--repair --force` disk backfill on 824 ds → recache → retrain → gate bipartite/skew  

Adding tabular features **alone** does not exclusify GNN — need **GIN batch coupling + hub topology in labels**.

---

## 2. Redundant or Weak Features (GNN dim14 vs MLP dim22)

### Layout reference

| Layout | Task | Platform (14) | Edge (5) | Total |
|--------|------|---------------|----------|-------|
| **atomic21 / dim14** | 2 | raw queue, `is_cold`, raw tc, dim13=0 | exec, latency, is_warm, energy, comm | **21** |
| **dim22 / legacy MLP** | 3 (+src_norm) | norm queue, `shared_fate`, norm tc, **usage_ratio** | same 5 | **22** |

Platform dim indices (atomic21): 0–4 type onehot, 5–6 has_dnn1/2, 7 queue, 8 is_cold, 9–11 temporal, 12 target_concurrency, 13 reserved.

### Likely non-contributors to RTT

| Feature | Issue |
|---------|--------|
| **Platform dim 13 (reserved)** | Always `0.0` in atomic21 recache and seq_decode refresh |
| **`energy` (edge)** | Static per (task_type, platform_type); not in co-sim RTT objective |
| **`comm_time` (edge)** | Static I/O estimate from task priors; tiny vs exec+queue |
| **`target_concurrency` (platform)** | Derived from static `executionTime` priors, not live load |
| **`has_dnn1` / `has_dnn2` (platform)** | Edges are replica-filtered — for dnn1 task, every candidate has `has_dnn1=1` |
| **`is_warm` (edge) in recache** | Set from replica membership → **always 1** on feasible edges → zero gradient |
| **Platform one-hot (partial)** | `xavierDla`: dnn2 incompatible; `pynqFpga`: dnn2 only on rpi/xavier — several dims sparse |

### Redundant pairs (correlated signals)

| Pair | Relationship |
|------|----------------|
| **`exec_time` ↔ platform type one-hot** | Exec is deterministic from platform type for fixed task type |
| **`is_cold` ↔ temporal `cold_start_remaining`** | Heuristic sets cold_start from queue>0 when temporal missing |
| **dim22 `queue_len` (normalized) ↔ `usage_ratio`** | usage_ratio = queue / target_concurrency / 5 |
| **dim22 `shared_fate` ↔ per-platform `is_cold`** | Highly correlated on small nodes |
| **`is_warm` (feature_builder) ↔ `has_dnn*` (platform)** | Sandbox warm often aligns with replica presence; not disk warmth under v2 |

### Features that do matter (don't drop blindly)

- **`latency`** — dominant on skew/sparse offloading
- **`exec_time`** — hub servers are xavier under skew infra-gen
- **Raw `queue` (dim 7)** — queueTime driver
- **`is_cold` / `shared_fate`** — FilterStore / deferred init contention (v2+defer)
- **Temporal dims** — when SSC provides real `current_task_remaining` (post-enrichment)
- **`src_norm` (dim22 only)** — useful for topology; removal in atomic21 is a regression risk for skew

### Train/serve inconsistency (worse than redundancy)

| Path | `is_warm` on edge | Task dims |
|------|-------------------|-----------|
| `prepare_graphs_cache_seq.py` | Replica flag (=1 always) | 2 |
| `feature_builder.py` (MLP atomic21) | `previous_task` sandbox match | 2 |
| `gnn_hetero/scheduler.py` | Replica flag | 3 + src_norm |

---

## 3. Disk Feature (`has_function`, `node_disk_hit`)

### Simulator physics (`src/placement/warmth.py`)

Two warmth layers:

1. **Sandbox warm** (`sandbox_is_warm`): `platform.previous_task.type == current task type` — last task on that **platform** matched.
2. **Disk warm** (`node_has_cached_image` / `storage.has_function`): image for `(platform_short_name, task_type)` exists on **any local non-remote storage** on the **node**.

**Pull gate** (`needs_image_pull`):

| Physics | Skip pull when… |
|---------|-----------------|
| **v1 `platform_reuse_v1`** | Sandbox warm (previous_task match) — **coupled** |
| **v2 `node_disk_v2`** | **Disk has image only** — previous_task does **not** skip pull |

`has_function` on storage (`src/placement/infrastructure.py`):

```python
def has_function(self, platform: str, task_type: TaskType) -> bool:
    return (platform, task_type) in self.functions_cache
```

`image_pull_disk_hit()` in `warmth.py` = pull skipped due to node disk cache hit (v2 only).

### Behavioral consequence under v2 + defer_cold

- **First** dnn1 task on node A: real pull (~31s in audits).
- **Second** dnn1 on **another platform on A**: disk hit → no pull; may still pay sandbox cold if `previous_task` differs.
- **N same-function on one node**: **1× pull + (N−1)× queue/sandbox** — stair-step RTT pattern v2 labels expose.

`shared_fate` (dim22) tracks **fraction of co-located platforms still uninitialized** at schedule time — related to serialized pulls in queue, **not** the same as disk cache.

### What the GNN does not see today

| Simulator signal | In dim14 graph? |
|------------------|-----------------|
| Disk: `has_function` on node | **No** |
| Sandbox: `previous_task` match | Only in `feature_builder` edge `is_warm`; **not** in recache |
| Replica exists (`has_dnn1/2`) | Yes — autoscaler placement, not cache state |
| `node_pulls_in_flight` | Tier 3 stub only (`warmth.py` comments) |

Optimal v2 policy is often: **pay pull once per node per function, then stack tasks on that node**. The model must infer disk warmth from **`is_cold` / `initialized_snapshot` decay** and **`shared_fate`** — indirect and weaker than an explicit bit.

### Proposed additions

| Name | Definition | Placement |
|------|------------|-----------|
| **`node_disk_hit`** / **`has_function`** | `node_has_cached_image(node, plat_type, task_type)` for the task being scheduled | Edge attr or node-level broadcast to platforms on that node |
| **`sandbox_warm`** | `previous_task` type match | Edge attr — **replace** current misleading `is_warm` |

---

## Win zones (retrained on v2)

| Zone | Expect wins on |
|------|----------------|
| Degree-skew / hub topology | skew-4, tiered-hub, **bipartite k>b** |
| Disk-warm consolidation | contended queue regimes, low replica count |
| Sparse + skew live | v1 skew wins (+6–13%) — **needs hub in co-sim grid to hold** |

| Regime | MLP/Knative stay strong |
|--------|------------------------|
| Uniform symmetric | Triangle +0.4% MLP; many near-tie edges |
| Without disk feature in cache | skew3: MLP **2/3** despite v2 physics |

---

## Highest-Leverage Actions (beyond regen)

1. **Retrain on v2 corpus** — mandatory; pre-v2 comparisons are meaningless.
2. **Add `node_disk_hit` or `has_function`** to dim14 platform/edge features — closes main v2 observability gap; biggest uplift for uniform/contended configs.
3. **Evaluate on skew + bipartite first** — ship gate for “GNN wins v2 world”; don't gate on default alone.
4. **Keep decode fixes** (temp/LQB) for 125–225 uniform — warmth doesn't fix logit collapse.
5. **Extend co-sim grid** with `degree_skewed_core` + low connectivity before expecting sparse/skew live wins from v2 retrain alone.
6. **Restore `src_norm`** or add hub/core features for atomic21 skew routing.
7. **Fix recache `is_warm`** — use sandbox + disk split, not replica flag.

---

## Key Files

| Path | Role |
|------|------|
| `scripts_cosim/generate_gnn_datasets_fast.py` | Current 500-ds grid + v2 warmth default |
| `src/notebooks/prepare_graphs_cache_seq.py` | dim14 recache; degenerate `is_warm` |
| `src/policy/tabular/feature_builder.py` | atomic21 / dim22 inference; sandbox `is_warm` |
| `src/policy/tabular/constants.py` | FEATURE_DIM=21, platform layout |
| `src/policy/gnn_hetero/scheduler.py` | Legacy live GNN path (dim22-ish, replica `is_warm`) |
| `src/placement/warmth.py` | v1/v2 pull vs sandbox predicates |
| `src/generate_infrastructure.py` | `degree_skewed_core` topology generation |
| `simulation_data/graphs_cache_warmth_v2_sparse_merged` | Merged 824-graph training cache |
| `run_warmth_sparse_recache_finetune.sh` | Sparse recache + finetune pipeline |

---

## Changelog

| Date | Change |
|------|--------|
| 2026-06-11 | v0.28 pass: merged 824 sparse section; §4 GNN vs MLP mechanism; bipartite/skew3 numbers; minimum separation stack; bipartite graph truth |
| 2026-06-11 | Initial doc: co-sim grid gap for sparse/skew, feature redundancy audit, disk warmth observability gap |
