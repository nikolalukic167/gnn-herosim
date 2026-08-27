# FilterStore Storage Contention — Reference Map

**Last Updated:** 2026-06-11 (v0.28.1)  
**Verified by:** `test_memory_contention_ab.py` + `test_cold_start_queue_last_task_ab.py` (both SWEEP PASSED)  
**Related:** `LINEAGES.md` · `docs/notes/compare.md` · `docs/notes/warmth_model.md` · `docs/notes/cosim_warmth_gap.md` · `paper/cosimulation.md` · `paper/model.md`

> Maps N× RTT growth from co-located cold image pulls, cold-start / queueTime / last-task mechanics, metric naming traps, GNN/MLP feature gaps, and counterfactual proofs. Use **last-task RTT** for contention penalty — not `total_rtt` sum.

---

## At a Glance

| Item | Value |
|------|-------|
| **Root cause** | SimPy `FilterStore` on `node.storage` serializes cold image pulls when N replicas share one node's local `flashCard` |
| **Unit pull cost** | **T_pull ≈ 31.30s** (3.057 GB ÷ 100 MB/s + 0.00012s latency) |
| **Not the cause** | Platform execution queue depth at schedule time (always **0** in A/B) |
| **Last-task penalty (N=4)** | contended − parallel ≈ **93.91s ≈ (N−1)×T_pull** |
| **Run commands** | `test_memory_contention_ab.py` (mirror + sweep) · `test_cold_start_queue_last_task_ab.py` (deep audit + counterfactuals) |
| **Results JSON** | `simulation_data/memory_contention_ab/summary.json` · `simulation_data/cold_start_queue_last_task_ab/summary.json` |

---

## Theory Constants (`data/nofs-ids`)

| Parameter | Source file | Value |
|-----------|-------------|-------|
| `imageSize` (dnn1, rpiCpu) | `data/nofs-ids/task-types.json` | **3.057 GB** |
| Storage write (flashCard) | `data/nofs-ids/storage-types.json` | **171 MB/s** |
| Network bandwidth | infrastructure config | **100 MB/s** (bottleneck) |
| Pull speed | `min(171, 100)` | **100 MB/s** |
| **T_pull** | `image_gb / (pull_speed/1024) + write_latency` | **31.3038s** |
| Cold start (dnn1, rpiCpu) | task-types | **0.33s** |
| Exec (dnn1, rpiCpu) | task-types | **0.0029s** (not 0.78s — stale in `test_storage_contention.py`) |
| **T_baseline** | T_pull + cold + exec | **31.64s** |

**Formula (last task, contended):**

```
RTT_last ≈ N × T_pull + cold_start + exec + comm
penalty  ≈ RTT_last_contended − RTT_last_parallel ≈ (N−1) × T_pull
```

---

## Verified Sweep (determined scheduler, forced placements)

Config: `defer_cold_replica_init=True`, `fast_forward_warmup=True`, batch_size=N, all tasks t=0.

### Layer 1 — SimPy mirror vs Layer 2 — full HeROsim

| N | Contended last RTT | Parallel last RTT | Penalty % | Predicted contended | Multiplier |
|---|-------------------:|------------------:|----------:|--------------------:|-----------:|
| 1 | 31.64s | 31.64s | 0% | 31.64s | 1.00× |
| 2 | 62.94s | 31.64s | 99% | 62.94s | 1.99× |
| 3 | 94.24s | 31.64s | 198% | 94.24s | 2.98× |
| 4 | 125.55s | 31.64s | 297% | 125.55s | 3.97× |
| 5 | 156.85s | 31.64s | 396% | 156.85s | 4.96× |

Full sim matches mirror within **~0.02s** (tolerance 0.5s). Source: `simulation_data/memory_contention_ab/summary.json`.

### N=4 contended — per-task timing (full sim)

| Task | elapsed | waitT | **queueT** | initT | pullT | coldT | execT | commT | platQ@sched | sched→arr→start→done |
|------|--------:|------:|-----------:|------:|------:|------:|------:|------:|------------:|------------------------|
| 0 | 125.22s | 0 | **31.30s** | 0.33s | 0.00s | 0.33s | 0.003s | 0.017s | **0** | 0 → 31.30 → 31.63 → 125.22 |
| 1 | 125.22s | 0 | **62.61s** | 0.33s | 0.00s | 0.33s | 0.003s | 0.017s | **0** | 0 → 62.61 → 62.94 → 125.22 |
| 2 | 125.22s | 0 | **93.91s** | 0.33s | 0.00s | 0.33s | 0.003s | 0.017s | **0** | 0 → 93.91 → 94.24 → 125.22 |
| 3 | 125.57s | 0 | **125.22s** | 0.33s | 0.00s | 0.33s | 0.003s | 0.017s | **0** | 0 → 125.22 → 125.55 → 125.57 |

**Parallel N=4:** every task queueT=**31.30s**, elapsed=**31.65s**, platQ=**0**.

### Misleading aggregates

| Metric | N=4 contended | Notes |
|--------|--------------:|-------|
| Last-task RTT | **125.57s** | Correct penalty metric |
| `total_rtt` (sum of elapsed) | **501.21s** | **3.99×** last-task — do not use as system RTT |
| Task 0 `named_sum` (wait+queue+init+exec+comm) | **31.65s** | **Missing 93.56s** `invisible_gap` |
| Task 0 `exec_phase_extra` (compute−exec−comm) | **93.56s** | Second FilterStore wait during execution I/O |
| Task 3 `named_sum` | **125.57s** | Closes exactly — only last task is honest aggregate |

### Last-task mechanics (N=4 contended cold)

| Property | Value |
|----------|-------|
| All tasks share `doneTime` (tasks 0–2) | **~125.22s** — early tasks block on exec I/O while later pulls hold flashCard |
| Last-task penalty | contended − parallel = **93.91s** = **(N−1)×T_pull** exactly |
| Cold start per task | **0.33s** flat — does not scale with N |
| Exec per task | **0.003s** flat |
| Comm per task | **~0.017s** flat |
| Dominant term | **queueTime** (~31s steps) — ~100× larger than cold+exec+comm combined |

---

## Three Different “Queues” (do not conflate)

| Name | Scope | When captured | N=4 contended cold batch | Used by |
|------|-------|---------------|--------------------------|---------|
| **GNN `queue` feature** | Per platform | Batch queue snapshot at schedule | **0** | `platform.queue_length()` → dim 7 |
| **`shared_fate_signal` / `is_cold`** | Node density / per plat | `initialized_snapshot` at schedule | **1.0** (6/6 cold on node0) — **saturates**; cannot distinguish N=2 vs N=4 | dim 8 (layout-dependent) |
| **Task stat `queueTime`** | Per task (outcome) | After sim: `arrived − scheduled` | **31, 63, 94, 125s** — linear in pull ordinal | JSON `taskResults`, not GNN input |
| **Temporal dims 9–11** | Per platform | At schedule | **~0** (cold/exec/comm normalized /10) | Sub-second physics only — not pull wait |

### Task metric definitions (`src/placement/infrastructure.py`)

```python
wait_time  = scheduled_time - dispatched_time   # scheduler/autoscaler wait
queue_time = arrived_time - scheduled_time      # replica-init wait (mostly pull serialization)
initialization_time = started_time - arrived_time  # cold start (~0.33s)
```

**`queueTime` ≠ platform execution queue depth.** It is wait until `task.arrived` (replica ready after pull).

---

## Mechanism — Code Path

```
Scheduler places task on cold platform (defer_cold_replica_init=True)
  → src/policy/determined/scheduler.py: initialize_replica process started
  → src/policy/determined/autoscaler.py:
       node_storage = yield node.storage.get(lambda: not remote)  # blocks on FilterStore
       yield env.timeout(retrieval_duration)                       # ~31.3s
       yield node.storage.put(node_storage)
       yield platform.initialized.succeed()
  → src/placement/infrastructure.py platform_process:
       yield self.initialized → task.arrived.succeed() → cold start → exec → I/O
```

**Serialization point:** one `flashCard` per node in `FilterStore` (`src/placement/simulation.py:create_nodes`).

**Comment in autoscaler (line ~238):** *"Hold local storage for the full image transfer (FilterStore serializes concurrent pulls on the same node)."*

---

## Counterfactual Proofs (N=4, verified script)

Source: `test_cold_start_queue_last_task_ab.py` Layer 3.

| Scenario | Config | Last RTT | Max RTT | total_rtt | N× gone? |
|----------|--------|--------:|--------:|----------:|:--------:|
| **A Contended cold** | defer_cold=True, 1 flashCard | **125.57s** | 125.57s | 501.21s | ✗ |
| **B Parallel cold** | 1 task/node | **31.65s** | 31.65s | 126.62s | baseline |
| **C Contended warm** | defer_cold=False (pre-init) | **0.35s** | 0.35s | 1.40s | ✓ |
| **D Multi-storage** | 4× flashCard on node0 | **31.65s** | 31.65s | 126.62s | ✓ |
| **Remote-only storage** | no local flashCard | **HANG** (deadlock) | — | — | — |

Warm removes pull entirely (cold+exec only). Multi-storage parallelizes pulls — same co-location, no serialization.

---

## Metric Traps

### 1. `pullTime` = 0 with `fast_forward_warmup=True`

`platform_process` consumes `yield self.initialized` **once at startup** (line ~885) before the per-task loop. Second `yield self.initialized` in loop returns instantly → `pullTime = after − before = 0`.

| fast_forward_warmup | pullTime task 0 (N=4 contended) | queueTime task 0 |
|--------------------:|--------------------------------:|-----------------:|
| True (A/B default) | **0.00s** | **31.30s** |
| False | **31.30s** (= queueTime) | **31.30s** |

**Use `queueTime` or timestamps — not `pullTime` — under fast-forward.**

### 2. Hidden gap — `invisible_gap` (tasks 0..N−2)

| Task | elapsed | named_sum | invisible_gap | exec_phase_extra |
|------|--------:|----------:|--------------:|-----------------:|
| 0 | 125.22s | 31.65s | **93.56s** | **93.56s** |
| 1 | 125.22s | 62.96s | **62.26s** | **62.26s** |
| 2 | 125.22s | 94.26s | **30.95s** | **30.95s** |
| 3 | 125.57s | 125.57s | **0.00s** | ~0 |

Formula: `invisible_gap = elapsed − (wait + queue + init + exec + comm)`; equals `computeTime − executionTime − communicationsTime`.

Early contended tasks **start** at 31s/62s/93s but **finish** at ~125s. **Cause:** execution I/O `yield node.storage.get()` for output while later pulls hold the sole `flashCard`. Wait is **not** in `queueTime`, `pullTime`, `commTime`, or temporal features.

### 3. `average_contention` in scheduler state ≠ storage contention

| Field | Location | Actually measures |
|-------|----------|-------------------|
| `average_contention` | `determined/orchestrator.py` | Rolling mean of **`len(platform.queue.items)`** |
| FilterStore pull queue | SimPy internals | **Not exposed** |

Captured in SSC via `state_capture._capture_scheduler_state()` → `scheduler_state.average_contention`.

---

## GNN / MLP Feature Map

### Platform feature layouts

| Layout | File | Dim 8 | Dims total (plat) | Cache |
|--------|------|-------|-------------------|-------|
| **dim14 GNN** | `prepare_graphs_cache.py` | **`shared_fate_signal`** (cold/total on node) | 14 | `graphs_cache_gnn_datasets_4tasks_1060` |
| **atomic21 tabular** | `prepare_graphs_cache_seq.py`, `constants.py` | **`is_cold`** (0/1 per platform) | 14 → 21 with edges | seq cache v7.1 |
| **dim22 tabular** | `feature_builder.py` (`INFERENCE_FEATURE_LAYOUT=dim22`) | **`shared_fate`** (live) | 22 | inference only |

**dim14 platform vector:** `[type_onehot(5), has_dnn1, has_dnn2, queue, shared_fate, task_rem/10, cold_rem/10, comm_rem/10, target_conc, usage_ratio]`

### What is captured today

| Signal | Capture API | File | In GNN graph? | In MLP bundle? | Seconds scale? |
|--------|-------------|------|:-------------:|:-------------:|:--------------:|
| Platform queue depth | `capture_full_queue_snapshot()` | `state_capture.py:54` | dim 7 | dim 7 | No (count) |
| Initialized bool | `capture_initialized_snapshot()` | `state_capture.py:68` | → shared_fate / is_cold | dim 8 | No (0/1 or ratio) |
| Temporal remainders | `capture_temporal_state_*()` | `state_capture.py:106` | dims 9–11 (/10) | same | ~0.03 max (sub-second) |
| **Pending pull count** | — | **NOT IMPLEMENTED** | — | — | — |
| **Estimated pull remaining** | — | **NOT IMPLEMENTED** | — | — | — |
| Task outcome `queueTime` | post-sim Task metrics | `infrastructure.py:289` | **Label only** (oracle) | label | **Yes (~31s steps)** |

### Inference wiring

| Step | File |
|------|------|
| Batch queue snapshot | `gnn/scheduler.py:_capture_batch_queue_snapshot()` |
| Full queue + temporal | `gnn/scheduler.py:_build_inference_graph()` |
| Graph build | `tabular/feature_builder.py:build_pyg_inference_graph()` |
| Live shared_fate | `feature_builder.py:_shared_fate_by_position()` — reads `platform.initialized.triggered` |
| SSC write (co-sim) | `state_capture.py:get_captured_state()` → top-level `initialized_snapshot` |
| Placement capture | `determined/scheduler.py:capture_task_placement()` / `gnn/scheduler.py` same pattern |

### Feature vs label scale mismatch

| Observable at schedule | Typical value (cold batch, node0) | Label magnitude |
|------------------------|-----------------------------------|-----------------|
| queue feature | **0** | — |
| shared_fate | **1.0** (6/6 cold — saturates at N≥1 all-cold) | — |
| is_cold | **1** per cold plat | — |
| cold_rem / comm_rem | **~0.03 max** (normalized /10) | — |
| oracle RTT / queueTime (last task) | — | **31…125s** |

**At schedule for simultaneous cold batch:** GNN inputs = `{queue=0, shared_fate≈1.0, temporal≈0}`; 30s lives in the **label**, not input magnitudes. Model must learn sparse 0–1 node signals + graph structure → 30s-scale penalty. **`shared_fate` encodes risk density, not pull ordinal or seconds.**

### GNN vs MLP for storage contention

| Model | Node-level signal | Weakness under co-located cold batch |
|-------|-------------------|--------------------------------------|
| **GNN dim14** | `shared_fate_signal` (dim 8) | Saturates at 1.0; no message about +93s |
| **MLP atomic21** | `is_cold` (dim 8, per plat) | No node aggregation — weaker intra-batch co-location |
| **Both** | queue dim 7 | Always 0 at schedule for this failure mode |

**Orthogonal to GNN vs MLP separation:** FilterStore blindness affects both equally. **GIN batch coupling** (hub routing under k>b) is a separate axis — see `docs/notes/gnn_v2_sparse_topology_and_features.md` §4. **`node_disk_hit`** on dim13 would help both unless paired with batch-level pull observables.

---

## What We Do NOT Have (fair feature candidates)

| Proposed feature | Observable at schedule? | Fair for GNN/MLP? | Implementation sketch |
|------------------|:-------------------------:|:-----------------:|----------------------|
| `node_cold_count` | ✓ (count `not initialized.triggered`) | ✓ | **SHIPPED CACHE 5.6** dim 14 |
| `storage_busy` | ✓ (`len(node.storage.items) < expected`) | ✓ | Node-level checkout flag |
| `node_pulls_in_flight` | Needs instrumentation | ✓ | Counters in `initialize_replica` start/end |
| `estimated_pull_remaining_sec` | ✓ cold_count × T_pull(priors) | ✓ | **SHIPPED CACHE 5.6** dim 15 (`/100`) |
| Realized `queueTime` | Post-outcome only | **✗ leakage** | Do not use as input |
| FilterStore internal depth | SimPy-only unless exposed | Borderline | Would need explicit API |

**Status (2026-08-12):** `node_cold_count` + `estimated_pull_remaining_sec` wired
(CACHE 5.6). Retrain **negative** for residual. Diagnosis: absolute
`cold_count×T_pull` as cost → 62s; **marginal** `(committed+1)×T_pull`
(`ect_pull`) → 31.65s = oracle. Co-sim labels remain warm-path.

---

## Key Files Index

| Path | Role |
|------|------|
| `scripts_cosim/test_memory_contention_ab.py` | A/B sweep (mirror + full sim) → `memory_contention_ab/summary.json` |
| `scripts_cosim/test_cold_start_queue_last_task_ab.py` | Deep audit: timing dissection, 3-queue signals, metric traps, counterfactuals A–D → `cold_start_queue_last_task_ab/summary.json` |
| `scripts_cosim/test_storage_contention.py` | Standalone SimPy proof (uses stale exec=0.78 in header) |
| `src/policy/determined/autoscaler.py` | Image pull + FilterStore hold/release |
| `src/policy/determined/scheduler.py` | Deferred init trigger, queue snapshot, SSC capture |
| `src/placement/infrastructure.py` | Task metrics, `platform_process`, storage I/O blocking |
| `src/placement/simulation.py` | `create_nodes()` — one `FilterStore` per node storage |
| `src/policy/state_capture.py` | `capture_initialized_snapshot`, queue/temporal capture |
| `src/policy/tabular/feature_builder.py` | Live inference features (GNN + MLP) |
| `src/notebooks/prepare_graphs_cache.py` | dim14 cache, `shared_fate_signal` dim 8 |
| `src/notebooks/prepare_graphs_cache_seq.py` | atomic21 cache, `is_cold` dim 8 |
| `src/policy/tabular/constants.py` | `PLATFORM_IS_COLD_DIM = 8`, `CACHE_VERSION = 7.1-atomic21` |
| `scripts_cosim/backfill_initialized_snapshot.py` | Phase-1 SSC backfill for co-sim corpus |
| `data/nofs-ids/` | Task/storage/network priors for T_pull |
| `simulation_data/memory_contention_ab/summary.json` | N-sweep mirror + full sim |
| `simulation_data/cold_start_queue_last_task_ab/summary.json` | Deep audit: dissection, counterfactuals, metric traps |

---

## Pull Timeline (SimPy mirror, N=4 contended)

From `test_cold_start_queue_last_task_ab.py` Layer 1 (exec=0.0029s):

| Task | Pull starts | Pull ends | queueTime proxy | RTT |
|------|------------:|----------:|----------------:|----:|
| 0 | 0.00s | 31.30s | 0.00s | 31.64s |
| 1 | 31.30s | 62.61s | 31.30s | 62.94s |
| 2 | 62.61s | 93.91s | 62.61s | 94.24s |
| 3 | 93.91s | 125.22s | 93.91s | 125.55s |

Task i pull starts at **i × T_pull**; full-sim `queueTime` = **(i+1) × T_pull** (scheduled→arrived).

---

## Scheduler Blindness (why Knative-style queue fails)

At placement time for simultaneous cold batch on same node:

```
node0:platform_6  queue=0  ← looks free
node0:platform_7  queue=0  ← looks free
node0:platform_8  queue=0  ← looks free
node0:platform_9  queue=0  ← looks free
```

Actual last-task RTT: **~125s** vs heuristic expectation **~32s** → **~290% underestimation** at N=4.

`shared_fate_signal` (dim14) is the current mitigation — aggregates cold density per node; does not encode pull ordinal or ~30s magnitude.

---

## Co-sim Corpus Context (live training data)

| Stat | 1060 (historical) | warmth_v2 + sparse merged | Notes |
|------|-------------------|---------------------------|-------|
| Platform cold fraction | **~71%** | Same mechanism Phase 1 | `initialized_snapshot` |
| Task `coldStartTime>0` | **~0.1%** | TBD post-v2 | Different from plat cold |
| Label queueTime max | **18.4s** | v2 exposes consolidation | 1060 lacked N×31s pulls |
| Graph disk feature | No | **B1 DONE** `node_disk_hit` dim 13 | Needs `--repair --force` backfill on 824 ds |
| Hub topology in grid | Old ER mix | **824 ER only** — no `degree_skewed_core` | Bipartite wins need hub co-sim |

See `docs/notes/compare.md` § Verified claims audit · `scripts_cosim/audit_doc_claims.py`.

---

## v2 warmth interaction (node_disk_v2)

Under **v2**, second cold platform on same node after first pull skips **pull branch** (`node_has_cached_image`) — Gate B: N=4 contended last-task **125.57s → 31.65s** when disk warm eliminates redundant pulls.

| Scenario | v1 | v2 |
|----------|----|----|
| 2nd dnn1, another plat, same node | ~**31s** pull again | **0s** pull (disk hit) |
| N=4 cold, 1 node, 1 flashCard | Last ~**125s** | Same FilterStore serialization on first-wave pulls |
| Optimal label bias | Spread (phantom re-pulls) | Stack on warm node after one pull |

**GNN/MLP implication:** v2 labels reward consolidation; models still **cannot see disk cache** at schedule time without `node_disk_hit` feature. Adding the feature helps **both** models; GNN advantage still requires **batch coupling + hub topology** in training (see `docs/notes/gnn_v2_sparse_topology_and_features.md` §4).

---

## Decision Guide

| Question | Answer |
|----------|--------|
| Is N× from platform queue? | **No** — platQ=0 at schedule; penalty is FilterStore pull serialization |
| Best metric for contention penalty? | **Last-task RTT** or max per-task RTT |
| Best outcome field for pull wait? | **`queueTime`** (not `pullTime` with fast-forward) |
| Best current GNN node signal? | **`shared_fate_signal`** (dim14) — saturates, no seconds |
| Can shared_fate distinguish N=2 vs N=4 all-cold? | **No** — both ≈1.0 |
| Fair to add pull-remaining feature? | **Yes** if from schedule-time observables + static priors |
| Retrain required for new feature? | **Yes** — new dim → recache + retrain |
| Warmth / pull gate model | **`docs/notes/warmth_model.md`** — last-task predicate, disk cache gap |
| Do not use for contended runs | **`total_rtt` sum**, per-task elapsed (except last), **`pullTime`** with FF on |

---

## Changelog

| Date | Change |
|------|--------|
| 2026-06-11 | v0.28: v2 warmth interaction table; expanded co-sim corpus (824 merged); GNN vs MLP contention note + cross-ref gnn_v2 §4 |
| 2026-06-11 | Cross-link `cosim_warmth_gap.md` — 1060 labels lack 31s pull (pullTime=0, queue max 18s) |
| 2026-06-11 | Added cold-start/queueTime/last-task audit from `test_cold_start_queue_last_task_ab.py`: invisible_gap table, counterfactuals A–D verified, GNN vs MLP contention table, feature/label mismatch, three-queue saturation note |
| 2026-06-11 | Initial doc from `test_memory_contention_ab.py` verification + timing audit + GNN feature gap analysis |
