# Platform Warmth Model — Expert Reference

**Last Updated:** 2026-06-11 (v0.28.1 — B1 disk feature shipped)  
**Audience:** External reviewers, paper co-authors, simulator maintainers  
**Companion docs:** `memory/storage_contention.md` · `memory/cosim_warmth_gap.md` (1060 historical) · `memory/cosim_grid_and_regen.md` · `memory/gnn_v2_sparse_topology_and_features.md` · `memory/memory.md` · `memory/compare.md`

> **One-sentence summary:** Default `platform_reuse_v1` couples pull+sandbox on `previous_task` match. Opt-in `node_disk_v2` (`infrastructure.warmth_physics`, `src/placement/warmth.py`) decouples: **pull** skips on node `has_function` only; **sandbox** still uses `previous_task` only. Co-sim generator defaults to v2 + `defer_cold_replica_init=True`.

---

## Table of Contents

1. [Executive summary](#1-executive-summary)
2. [Cost layers and where time goes](#2-cost-layers-and-where-time-goes)
3. [The warm predicate — full code map](#3-the-warm-predicate--full-code-map)
4. [State lifecycle: previous_task, initialized, idle](#4-state-lifecycle-previous_task-initialized-idle)
5. [Disk cache (Storage.functions_cache)](#5-disk-cache-storagefunctions_cache)
6. [Policy comparison: Knative/determined vs HeROcache](#6-policy-comparison-knativedetermined-vs-herocache)
7. [Co-simulation seeding and preinit](#7-co-simulation-seeding-and-preinit)
8. [Replica scale-down vs warmth](#8-replica-scale-down-vs-warmth)
9. [GNN / MLP feature alignment](#9-gnn--mlp-feature-alignment)
10. [Measured metric split (co-sim vs live)](#10-measured-metric-split-co-sim-vs-live)
10b. [1060 corpus gap (labels vs A/B physics)](#10b-1060-corpus-gap-labels-vs-ab-physics)  
10c. [warmth_v2 post-regen status](#10c-warmth_v2-post-regen-2026-06-11)
11. [Thought experiments / failure modes](#11-thought-experiments--failure-modes)
12. [Scientific realism assessment](#12-scientific-realism-assessment)
13. [Design alternatives not implemented](#13-design-alternatives-not-implemented)
14. [Recommended paper wording](#14-recommended-paper-wording)
15. [File index](#15-file-index)
16. [Changelog](#16-changelog)

---

## 1. Executive summary

### What is implemented

| Question | `platform_reuse_v1` (default) | `node_disk_v2` (co-sim default) |
|----------|------------------------------|--------------------------------|
| Image pull skip? | `previous_task.type == f` | `has_function` on node local storage |
| Sandbox cold-start skip? | `previous_task.type == f` | `previous_task.type == f` |
| Scale-down clears `previous_task`? | No | No — resets `initialized` only (§8.2) |
| Flag | `env.warmth_physics` / `HEROSIM_WARMTH_PHYSICS` | same |

Central module: [`src/placement/warmth.py`](../src/placement/warmth.py). Autoscalers re-check disk after `FilterStore.get()` using `active_storage=` (checked-out flash still visible).

### What is *not* implemented (Tier 3 stubs in warmth.py)

- `estimated_pull_remaining_sec` / `node_pulls_in_flight` features
- ~~`node_disk_hit` / `has_function` in SSC or graph cache~~ **DONE v0.28.1** — needs `--repair --force` backfill on 824 ds
- Time-based sandbox reuse window (TTL)
- LRU / k-recent task types on platform
- Separate RAM-warm vs disk-warm vs pull-warm tiers on main path
- Clearing `previous_task` on replica scale-down (scale-down still resets `initialized` only — see §8.2)

### Impact on contention experiments

For N cold replicas of **the same function** on **one node**, the warmth model forces **N independent cold pulls** (~31 s each, serialized on one `flashCard`). A real Kubernetes node would typically **reuse a local image** after the first pull. See `memory/storage_contention.md` for verified N× numbers.

---

## 2. Cost layers and where time goes

Workload: `dnn1` on `rpiCpu`, data from `data/nofs-ids/`.

| Layer | Parameter | Typical value | Gated by warm predicate? | Where in code |
|-------|-----------|--------------:|:------------------------:|---------------|
| **Image pull** | `imageSize / min(storage_write, network)` | **~31.30 s** | Yes — skip entire pull if warm | `*/autoscaler.py: initialize_replica` |
| **Sandbox cold start** | `coldStartDuration` | **~0.33 s** | Yes — skip timeout if warm | `infrastructure.py: platform_process` |
| **Execution** | `executionTime` | **~0.003 s** | No | `platform_process` |
| **Comm / I/O** | stateSize read+write | **~0.017 s** | No | `platform_process` |

**Ratio:** pull : cold : exec ≈ **100 : 1 : 0.01** — architectural warmth decisions dominate **pull cost**, not sandbox cost.

Verified contended N=4 last task (`scripts_cosim/test_memory_contention_ab.py`):

```
queueTime ≈ 125.22s  (pull wait, scheduled → arrived)
coldStart ≈ 0.33s
exec+comm ≈ 0.02s
elapsed   ≈ 125.57s  (last task — components close)
```

Warm counterfactual (pre-initialized replicas, same node): last-task elapsed **~0.35s** (no pull).

---

## 3. The warm predicate — full code map

**Primary path (shipped 2026-06-11):** All Knative-family autoscalers gate pulls via `needs_image_pull()` in [`src/placement/warmth.py`](../src/placement/warmth.py). v2 (`node_disk_v2`) skips pull when `node_has_cached_image()`; sandbox still uses `sandbox_is_warm()` / `previous_task` in `platform_process()`. Sections below show legacy inline `warm_function` blocks — behavior is now routed through `warmth.py`.

### 3.1 Core predicate (legacy inline blocks — same semantics as warmth.py v1)

**Determined autoscaler — image pull gate:**

```python
# src/policy/determined/autoscaler.py (lines 192-203)
        # Check node RAM cache
        warm_function: bool = (
            platform.previous_task is not None
            and platform.previous_task.type["name"] == task_type["name"]
        )

        # Initialize image retrieval duration
        retrieval_duration: DurationSecond = 0.0

        # TODO: Retrieve image if function not in RAM cache nor in disk cache
        # FIXME: Should be factored in superclass
        if not warm_function:
```

**Knative autoscaler — same gate:**

```python
# src/policy/knative/autoscaler.py (lines 192-203)
        # Check node RAM cache
        warm_function: bool = (
            platform.previous_task is not None
            and platform.previous_task.type["name"] == task_type["name"]
        )

        # Initialize image retrieval duration
        retrieval_duration: DurationSecond = 0.0

        # TODO: Retrieve image if function not in RAM cache nor in disk cache
        # FIXME: Should be factored in superclass
        if not warm_function:
```

**Platform executor — sandbox cold-start gate:**

```python
# src/placement/infrastructure.py (lines 1021-1047)
            # Check node RAM cache
            warm_function: bool = (
                self.previous_task is not None
                and self.previous_task.type["name"] == task.type["name"]
            )

            # Cold start penalty is not incurred if task sandbox was in cache
            initialization_duration = (
                task.type["coldStartDuration"][self.type["shortName"]]
                if not warm_function
                else 0.0
            )

            # Compute total cold start duration
            cold_start_duration: float = initialization_duration

            if cold_start_duration > 0:
                task.cold_started = True

                logging.info(
                    f"[ {self.env.now} ] ❄️ {task} cold start (duration:"
                    f" {cold_start_duration}) on {self}"
                )

            # Cold start timeout
            yield self.env.timeout(cold_start_duration)
            task.cold_start_time = cold_start_duration
```

**Comment in code says “RAM cache” but implementation is last-task type match only — no RAM size model, no separate memory pool.**

---

### 3.2 Full cold pull path — determined autoscaler

When `not warm_function`, determined holds the storage device for the full transfer (FilterStore serialization):

```python
# src/policy/determined/autoscaler.py (lines 203-256)
        if not warm_function:
            logging.info(
                f"[ {self.env.now} ] 💾 {node} needs to pull image for {task_type}"
            )

            retrieval_size: SizeGigabyte = task_type["imageSize"][
                platform.type["shortName"]
            ]
            node_storage = yield node.storage.get(
                lambda storage: not storage.type["remote"]
            )
            retrieval_speed: SpeedMBps = min(
                node_storage.type["throughput"]["write"], node.network["bandwidth"]
            )
            retrieval_duration += (
                retrieval_size / (retrieval_speed / 1024)
                + node_storage.type["latency"]["write"]
            )

            stored = node_storage.store_function(platform.type["shortName"], task_type)

            if not stored:
                logging.error(
                    f"[ {self.env.now} ] 💾 {node_storage} has no available capacity to"
                    f" cache image for {self}"
                )

            # Hold local storage for the full image transfer (FilterStore serializes
            # concurrent pulls on the same node).
            yield self.env.timeout(retrieval_duration)
            yield node.storage.put(node_storage)
        else:
            retrieval_duration = 0.0

        platform.storage_time += retrieval_duration
        ...
        yield platform.initialized.succeed()
```

---

### 3.3 Full cold pull path — Knative autoscaler (order differs)

Knative releases storage **before** charging `retrieval_duration` timeout (no FilterStore hold during transfer):

```python
# src/policy/knative/autoscaler.py (lines 203-260)
        if not warm_function:
            ...
            node_storage = yield node.storage.get(
                lambda storage: not storage.type["remote"]
            )
            retrieval_speed: SpeedMBps = min(
                node_storage.type["throughput"]["write"], node.network["bandwidth"]
            )
            retrieval_duration += (
                retrieval_size / (retrieval_speed / 1024)
                + node_storage.type["latency"]["write"]
            )
            stored = node_storage.store_function(platform.type["shortName"], task_type)
            ...
            # Release storage
            yield node.storage.put(node_storage)

        ...
        # FIXME: Retrieve function image
        yield self.env.timeout(retrieval_duration)

        platform.storage_time += retrieval_duration
        ...
        yield platform.initialized.succeed()
```

**Contention physics differ between policies** (determined serializes on storage token; Knative does not hold token during timeout). Co-sim / A/B tests use **determined**.

---

### 3.4 previous_task update after each task

Only the **most recent** task is stored:

```python
# src/placement/infrastructure.py (lines 1178-1181)
            # Update platform cache
            self.previous_task = self.current_task
            self.current_task = None
            self.idle_since = self.env.now
```

No ring buffer, no timestamp on `previous_task`, no decay.

---

### 3.5 Platform fields at construction

```python
# src/placement/infrastructure.py (lines 604-617)
        self.previous_task: Task | None = None
        self.current_task: Task | None = None
        self.idle_since: SimTime = math.inf

        self.last_allocated: SimTime = math.inf
        self.last_removed: SimTime = math.inf

        self.load_time: SimTime = 0
        self.storage_time: SimTime = 0

        self.initialized = env.event()
        self.tasks_count: int = 0
        self.local_dependencies: int = 0
        self.cache_hits: int = 0
```

`initialized` is a one-shot SimPy event — pull complete / replica ready for execution loop.

---

### 3.6 Task metrics after completion

```python
# src/placement/infrastructure.py (lines 283-292)
        self.elapsed_time = self.done_time - self.dispatched_time

        self.wait_time = self.scheduled_time - self.dispatched_time
        self.queue_time = self.arrived_time - self.scheduled_time
        self.initialization_time = self.started_time - self.arrived_time
        self.compute_time = self.done_time - self.started_time
```

```python
# src/placement/infrastructure.py (lines 1009-1013) — pullTime measurement site
            task.cache_hit = after_initialize == before_initialize
            task.pull_time = (
                after_initialize - before_initialize if not task.cache_hit else 0.0
            )
```

Note: `cache_hit` here means “platform already initialized when task dequeued,” **not** `warm_function` / disk cache hit. With `fast_forward_warmup=True`, `pullTime` is often **0** while pull wait appears in `queueTime`. See `memory/storage_contention.md`.

---

## 4. State lifecycle: previous_task, initialized, idle

### 4.1 Three related but distinct “cold” concepts

| Concept | Field / metric | Meaning |
|---------|------------------|---------|
| **Pull complete** | `platform.initialized.triggered` | Autoscaler finished `initialize_replica` |
| **Sandbox warm** | `previous_task.type == new_task.type` | Skip 0.33 s cold-start timeout |
| **Task cold-start stat** | `task.coldStartTime > 0` | Sandbox penalty was applied this invocation |

A platform can be **initialized** (pull done) but still pay **sandbox cold start** if `previous_task` was a different type or `None`.

### 4.2 initialized snapshot (GNN / SSC)

```python
# src/policy/state_capture.py (lines 68-85)
    def capture_initialized_snapshot(self) -> Dict[str, bool]:
        """
        platform.initialized.triggered is True when the image pull has completed
        and the platform is ready to accept tasks without a pull wait.  False means
        the platform is cold: any task placed on it will block inside
        platform_process() at `yield self.initialized` until the pull finishes.
        """
        snapshot: Dict[str, bool] = {}
        for node in self.nodes.items:
            for platform in node.platforms.items:
                key = f"{node.node_name}:{platform.id}"
                snapshot[key] = bool(platform.initialized.triggered)
        return snapshot
```

This drives **`shared_fate_signal`** (dim14) and **`is_cold`** platform features — **not** the same as sandbox `warm_function`.

### 4.3 Deferred cold init (batch pulls)

When `defer_cold_replica_init=True`, pull starts at **placement**, not at replica creation:

```python
# src/policy/determined/scheduler.py (lines 178-190)
            # Deferred cold replicas: start image pull when task is placed (concurrent across batch).
            if (
                self.defer_cold_replica_init
                and not sched_platform.initialized.triggered
            ):
                self.env.process(
                    self.autoscaler.initialize_replica(
                        (sched_node, sched_platform),
                        replicas[task.type["name"]],
                        task.type,
                        system_state,
                    )
                )
```

Warmth predicate still applies per platform at `initialize_replica` time — new platform → `previous_task is None` → cold pull.

### 4.4 GNN fallback prefers initialized replicas

```python
# src/policy/gnn/scheduler.py (lines 672-680)
            initialized_replicas = [
                replica for replica in available_replicas if replica[1].initialized.triggered
            ]
            candidates = initialized_replicas if initialized_replicas else available_replicas
            target_node, target_platform = min(
                candidates, key=lambda couple: len(couple[1].queue.items)
            )
```

This is **pull-complete** filtering, not sandbox warmth.

---

## 5. Disk cache (Storage.functions_cache)

### 5.1 Data structure and API

```python
# src/placement/infrastructure.py (lines 393-394, 447-495)
        self.functions_cache: List[Tuple[str, TaskType]] = []
        ...
    def has_function(self, platform: str, task_type: TaskType) -> bool:
        return (platform, task_type) in self.functions_cache

    def store_function(self, platform: str, task_type: TaskType) -> bool:
        if (platform, task_type) not in self.functions_cache:
            while (self.used * 1e-9) + task_type["imageSize"][platform] > self.type[
                "capacity"
            ]:
                try:
                    self.cache_eviction()
                except CacheEvictionError as e:
                    logging.error(f"[ {self.env.now} ] {e.message}")
                    return False

            self.functions_cache.append((platform, task_type))
            self.used += int(task_type["imageSize"][platform] * 1e9)
            ...
        return True
```

**Cache key:** `(platform_type_shortName, task_type)` e.g. `("rpiCpu", dnn1)` — **not** platform id. So image is logically shared across all `rpiCpu` slots on a node **once stored**.

### 5.2 FIFO eviction

```python
# src/placement/infrastructure.py (lines 428-445)
    def eviction_fifo(self) -> None:
        try:
            removed_platform, removed_type = self.functions_cache.pop(0)
            ...
            self.used -= int(removed_type["imageSize"][removed_platform] * 1e9)
        except IndexError:
            raise CacheEvictionError(f"{self} function cache is already empty")
```

Eviction removes disk entry but **does not** invalidate `platform.previous_task` — warmth state and disk state can diverge.

### 5.3 Pull skip: v1 vs v2 on determined/Knative path

| Physics | 2nd platform on same node after 1st pulled same `dnn1` | Mechanism |
|---------|----------------------------------------------------------|-----------|
| **v1 `platform_reuse_v1`** | **~31s pull again** (serialized FilterStore) | `needs_image_pull` = not `sandbox_is_warm` (per-platform `previous_task`) |
| **v2 `node_disk_v2`** (co-sim default) | **0s pull** (disk hit) | `needs_image_pull` = not `node_has_cached_image` — any local storage on node |
| **HeROcache** (legacy) | **0s pull** | Inline `has_function` check (predates `warmth.py`) |

Under **v1**, after first pull `store_function` may succeed without extra bytes but **`retrieval_duration` timeout still applied** because `warm_function` is false on platform B.

Under **v2**, platform B still pays **sandbox cold** (~0.33s) if `previous_task` differs — disk hit does not skip sandbox unless types match.

**Legacy 1060 corpus** was generated under v1-equivalent pull physics — see `memory/cosim_warmth_gap.md`. **warmth_v2 regen** uses v2 defaults; Gate B: N=4 contended last-task **125.57s → 31.65s** when disk warm eliminates redundant pulls.

### 5.4 node.cache_hits statistic

```python
# src/policy/determined/autoscaler.py (line 274)
        node.cache_hits += 0
```

Hard-coded zero increment — disk/RAM cache hits **not counted** in stats on this path.

---

## 6. Policy comparison: Knative/determined vs HeROcache

### 6.1 HeROcache — disk-aware pull skip

```python
# src/policy/herocache/autoscaler.py (lines 337-357)
        warm_function: bool = (
            platform.previous_task is not None
            and platform.previous_task.type["name"] == task_type["name"]
        )

        cache_storage: bool = False
        node_storage: Storage
        for node_storage in node.storage.items:
            if node_storage.has_function(platform.type["shortName"], task_type):
                cache_storage = True
                break

        retrieval_duration: DurationSecond = 0.0

        if not warm_function and not cache_storage:
            logging.info(
                f"[ {self.env.now} ] 💾 {node} needs to pull image for {task_type}"
            )
            # ... full pull ...
```

**Pull skipped if:** RAM warm (`previous_task` match) **OR** disk hit (`has_function`).

**Sandbox cold start:** still governed only by `previous_task` in `platform_process` — disk hit does **not** skip 0.33 s sandbox unless previous task matched.

### 6.2 Which path is used where

| Use case | Autoscaler | Disk cache check on pull? |
|----------|------------|:-------------------------:|
| Co-sim / brute-force labels | **determined** | **Yes** (v2 default) |
| GNN dataset generation | **determined** | **Yes** (v2 default) |
| Live normal sim Knative | **knative** | v2 when configured |
| All Knative-family autoscalers | shared `warmth.py` | v1 unless `node_disk_v2` |

Legacy 1060 corpus used v1 — regen required (`scripts_cosim/run_warmth_full_regen_recache.sh`).

---

## 7. Co-simulation seeding and preinit

### 7.1 Deterministic replica preinit — warm only if queue > 0

```python
# src/placement/simulation.py (lines 274-301)
                    if not (defer_cold_init and queue_length == 0):
                        platform.initialized.succeed()

                    if env and simulation_policy and deterministic_queues:
                        if queue_length > 0:
                            platform.previous_task = type('Task', (), {'type': {'name': task_type_name}})()
                            try:
                                warmup_tasks = create_warmup_tasks(...)
                                for warmup_task in warmup_tasks:
                                    platform.queue.put(warmup_task)
                            ...
                        else:
                            # Platform has NO queue tasks - leave COLD (previous_task = None)
                            # This enables realistic cold start simulation
                            pass  # platform.previous_task remains None
```

Co-sim Phase-1 intentionally leaves most platforms **cold** (`previous_task = None`, `initialized` may still succeed if not deferring) → high **`initialized_snapshot` cold fraction (~71%)**.

### 7.2 Fast-forward warmup uses same warm predicate internally

```python
# src/placement/infrastructure.py (lines 914-919, 944-946)
                        warm_function = (
                            previous_task_type is not None
                            and previous_task_type == warmup_task.type["name"]
                        )
                        ...
                    if self._warmup_tasks:
                        self.previous_task = self._warmup_tasks[-1]
```

Virtual warmup seeding:

```python
# src/placement/infrastructure.py (lines 954-959)
        if self.virtual_warmup_count > 0 and self.virtual_warmup_total_time > 0:
            yield self.env.timeout(self.virtual_warmup_total_time)
            if self.virtual_warmup_task_type:
                self.previous_task = type(
                    'Task', (), {'type': {'name': self.virtual_warmup_task_type}}
                )()
```

### 7.3 fast_forward_warmup startup wait on initialized

```python
# src/placement/infrastructure.py (lines 883-885)
        if self.fast_forward_warmup:
            # Wait for initialization first
            yield self.initialized
```

Consumes pull wait before per-task loop → breaks `pullTime` metric (see storage_contention doc).

---

## 8. Replica scale-down vs warmth

### 8.1 Time-based scale-down (keep_alive) — idle only

```python
# src/policy/determined/autoscaler.py (lines 294-301)
        removed_couple = next(
            (
                replica
                for replica in sorted_replicas
                if replica[1].queue_length() == 0
                and not replica[1].current_task
                and (self.env.now - replica[1].idle_since) > self.policy.keep_alive
            ),
            None,
        )
```

`keep_alive` (default 30 s in many configs) controls **when replica is removed**, not **when sandbox is warm**.

### 8.2 Scale-down resets initialized, NOT previous_task

```python
# src/placement/autoscaler.py (lines 331-349)
                # Reset platform to uninitialized state
                removed_replica[1].initialized = removed_replica[1].env.event()
                ...
                removed_replica[1].last_removed = self.env.now
```

**`previous_task` is not cleared.** If the same physical `Platform` object is re-assigned:

- Next init may skip pull if `previous_task` still matches (warm_function True)
- But `initialized` was reset → platform loop waits for new init
- **Ambiguous / inconsistent** reuse semantics

---

## 9. GNN / MLP feature alignment

### 9.1 Edge feature `is_warm` — path-dependent (train/serve landmine)

**Hom dim14 cache** (`prepare_graphs_cache.py`) — sandbox match via `previous_task_type_name`:

```python
# src/notebooks/prepare_graphs_cache.py (lines 1125-1136)
                # is_warm: matches the simulator's actual cold-start predicate.
                # platform_process() fires cold start when
                #   previous_task.type["name"] != task.type["name"].
                plat_key_for_warm = f"{plat_node_name}:{plat_id}"
                prev_type = (
                    (temporal_state or {})
                    .get(plat_key_for_warm, {})
                    .get("previous_task_type_name")
                )
                is_warm = 1.0 if (prev_type is not None and prev_type == task_type) else 0.0
```

**Seq/atomic21 recache** (`prepare_graphs_cache_seq.py`) — **degenerate:** sets `is_warm` from replica membership (`has_dnn1_arr` / `has_dnn2_arr`) → **always 1** on feasible edges → zero gradient. **Live inference** (`feature_builder.py`) uses sandbox `previous_task` match — train/serve mismatch on active warmth_v2 pipeline.

### 9.2 Temporal state captures previous_task_type_name

```python
# src/policy/state_capture.py (lines 165-178)
            # Matches the predicate platform_process() uses for cold-start avoidance:
            #   warm = (previous_task.type["name"] == task.type["name"])
            prev_task_type_name: Optional[str] = None
            if platform.previous_task is not None:
                prev_task_type_name = platform.previous_task.type.get("name")

            temporal_state[key] = {
                ...
                "previous_task_type_name": prev_task_type_name,
            }
```

### 9.3 Platform features — two layouts

**dim14 GNN cache** (`prepare_graphs_cache.py`):

- Dim 8 = **`shared_fate_signal`** = `cold_uninitialized_plats / total_plats_on_node` from `initialized_snapshot`
- Measures **pull readiness density**, not sandbox warmth

**atomic21 tabular** (`constants.py`, `prepare_graphs_cache_seq.py`):

```python
# src/policy/tabular/constants.py (lines 15-18)
# Dim 7: raw queue length; dim 8: per-platform is_cold; dim 13: reserved (0.0).
PLATFORM_QUEUE_RAW_DIM = 7
PLATFORM_IS_COLD_DIM = 8
```

```python
# src/notebooks/prepare_graphs_cache_seq.py (lines 1064-1070)
            is_cold_arr[pos] = 0.0 if initialized_snapshot.get(key, True) else 1.0
```

**Live inference** (`feature_builder.py`):

```python
# src/policy/tabular/feature_builder.py (lines 209-212)
        is_cold = 0.0 if info.platform.initialized.triggered else 1.0
        shared_fate = (
            float(shared_fate_by_pos[info.position]) if shared_fate_by_pos is not None else 0.0
        )
```

- **atomic21 layout:** dim 8 = `is_cold` from live `initialized.triggered`
- **dim22 layout:** dim 8 = `shared_fate` from live cold count on node

**ML models see simulator observables** — but **not node disk cache state** (`has_function`) unless we add it. Dim 8 layouts differ (`shared_fate` vs `is_cold`) — not equivalent for FilterStore contention.

### 9.4 Disk cache in graph features (B1 shipped 2026-06-11)

| Simulator signal | In SSC? | In dim14 seq cache? |
|------------------|:-------:|:-------------------:|
| `initialized_snapshot` (pull complete) | Yes | → dim 8 `is_cold` |
| Sandbox `previous_task` match | Yes (`temporal_state`) | edge `is_warm` (**fixed** — sandbox not replica flag) |
| **`has_function` / node disk hit** | **Yes** `disk_snapshot_by_task_type` | dim 13 `node_disk_hit` |
| `node_pulls_in_flight` | Tier 3 stub only | No |

**Backfill:** `--repair --force` re-exports `schedulingStateCapture` with disk snapshot. `--rewrite-ssc` alone cannot invent disk state from old optimal_result.

### 9.5 Task `src_norm` (B1 fixed)

| Cache path | Task dims | `src_norm` |
|------------|----------:|:----------:|
| `prepare_graphs_cache.py` (hom dim14) | **3** | Yes |
| `prepare_graphs_cache_seq.py` (warmth_v2 seq) | **3** | **Yes** (v0.28.1) |
| Live dim22 inference | **3** | Yes |
| Live atomic21 GNN | **3** | **Yes** — real `src_norm` in `feature_builder.py` |

`src_norm` fixes GNN train/serve parity; **does not** exclusify GNN over dim22 MLP (MLP already has client identity live).

---

## 10. Measured metric split (co-sim vs live)

From `memory/compare.md` / `scripts_cosim/audit_doc_claims.py`:

| Metric | Co-sim (1060 corpus) | Live normal sim (dim14-ce) | Interpretation |
|--------|---------------------:|---------------------------:|----------------|
| Task `coldStartTime > 0` | **~0.1%** | **9.6–16.4%** | Sandbox penalty rate |
| Platform cold (`initialized_snapshot`) | **~71%** | — | Pull-not-complete at schedule |
| `shared_fate` mean in dim14 cache | **~0.71–0.82** | — | Node cold density feature |

**Co-sim is pull-dominated and platform-cold-heavy by construction** (preinit leaves replicas cold). Live sim has different autoscale dynamics → different cold-start proportion.

These are **different layers** of the warmth model — do not conflate task cold-start % with platform initialized %.

---

## 10b. 1060 corpus gap (labels vs A/B physics)

**Full audit:** `memory/cosim_warmth_gap.md` (2026-06-11, 1230 datasets).

| Finding | 1060 optimal labels | A/B scripts (`defer_cold=True`) |
|---------|--------------------:|--------------------------------:|
| `pullTime > 0` | **0%** | 0% FF on; =queueTime FF off |
| `cacheHit=True` | **100%** | False when cold pull at placement |
| queueTime max | **18.37s** | **125.22s** (N=4 contended) |
| dnn1 queueTime max | **1.22s** | ~31s+ per pull |
| coldStartTime > 0 | **0.06%** | ~0.33s per cold sandbox |
| Co-sim config | defer **False**, spread placements | defer **True**, co-locate |

**Why:** Preinit calls `initialized.succeed()` without pull when defer=False; forced placements require initialized platforms; optimal plans spread tasks — **N×31s pull never enters training labels** even though simulator supports it.

**On warmth model change:** mandatory co-sim regen + recache + retrain — see cosim_warmth_gap.md (1060 audit) + `cosim_grid_and_regen.md` (warmth_v2/sparse status).

---

## 10c. warmth_v2 post-regen (2026-06-11)

| Item | Status |
|------|--------|
| Code | `warmth.py` + all Knative-family autoscalers — **shipped** |
| Co-sim default | v2 + `defer_cold_replica_init` in generator |
| warmth_v2 grid | **473–500** ds cached · sparse **351/351** · merged cache **824 graphs** |
| Label shift | Gate B verified — consolidation optima under v2 |
| Graph features | **Still missing** disk hit, seq `is_warm` degenerate, no `src_norm` in seq cache |
| Live gate (skew3) | MLP wins **2/3** (old + ce-reduced); v2 GNN sparse **1.23M** vs Kn **0.93M** — features + hub grid not in training yet |

Pre-v2 dim14-ce comparisons on old physics are **stale** for placement/warmth claims.

---

## 11. Thought experiments / failure modes

### 11.1 Same function, two platforms, one node (contention case)

| Step | Platform | v1 pull? | v2 pull? | FilterStore |
|------|----------|:--------:|:--------:|-------------|
| 1 | node0:plat6 (cold) | **Yes ~31s** | **Yes ~31s** | Holds flashCard |
| 2 | node0:plat7 (cold, A pulled) | **Yes ~31s** (serialized) | **No** (disk hit) | v1: serial; v2: may skip pull branch |

**Real K8s node:** second pod typically **no registry pull** — **v2 matches this**; v1 does not.

N× **queueTime** steps still possible when multiple cold pulls serialize on first wave (FilterStore) — see `memory/storage_contention.md`. v2 removes **phantom re-pulls**, not necessarily all contention.

### 11.2 A → B → A on same platform (ping-pong)

| Invocation | previous_task | Pull? | Sandbox cold? |
|------------|---------------|:-----:|:-------------:|
| A | None | Yes | Yes |
| B | A (type A) | Yes (B≠prev type) | Yes |
| A | B (type B) | **Yes** | **Yes** |

Even if A's image remains in `functions_cache` from invocation 1, invocation 3 pays **full pull + sandbox** on main path.

### 11.3 A → A → A on same platform

| Invocation | Pull? | Sandbox cold? |
|------------|:-----:|:-------------:|
| 1 | Yes | Yes |
| 2 | **No** | **No** |
| 3 | **No** | **No** |

Only consecutive same-type chain is warm — matches “execution context reuse” narrative.

### 11.4 Idle 10 minutes, same function returns

| Field | Behavior |
|-------|----------|
| `idle_since` | Updated at task done — used for **scale-down only** |
| `previous_task` | **Unchanged** by idle time |
| Warmth | Still warm if same type was last — **no TTL decay** |

Real systems: sandbox may cold after idle; image often stays local. Sim: **optimistic sandbox, pessimistic pull** (depending on path).

### 11.5 Scale down → scale up same platform object

| Field | After scale-down |
|-------|------------------|
| `initialized` | **Reset** (new event) |
| `previous_task` | **Retained** |
| Next scheduling | Pull may be skipped if warm_function True, but must re-trigger `initialized` |

---

## 12. Scientific realism assessment

### 12.1 Sandbox cold start (~0.33 s) — partially defensible

**Real-world analog:** AWS Lambda execution context reuse, Knative warm pod, Firecracker snapshot reuse.

**Simplifications accepted in literature:**

- Binary warm/cold (no partial init)
- Per-instance memory of last function

**Simplifications that may mislead:**

- No idle TTL (warmth never expires)
- Warmth lost permanently after **one** different function (strict ping-pong penalty)
- No memory pressure evicting sandbox

**Verdict:** Acceptable **if labeled** as “consecutive-invocation reuse model” for **sub-second** init. Training features (`is_warm` edge) match this.

### 12.2 Image pull (~31 s) — v1 pessimistic; v2 decoupled

**Real-world analog:** Container image layers cached on node (containerd/kubelet); second deployment of same image → load from local store.

| Real behavior | v1 `platform_reuse_v1` | v2 `node_disk_v2` (co-sim default) |
|---------------|------------------------|-------------------------------------|
| Node-level image dedup | Per-platform `previous_task` only | **`has_function` on node** skips pull |
| Pull once, many replicas | N cold replicas → N pull timeouts | **1× pull per node per function** then stack |
| Disk hit → seconds not 31s | Ignored; full T_pull | **Pull branch skipped** |
| Separate from sandbox reuse | Coupled (same predicate) | **Decoupled** — disk vs `previous_task` |

**Verdict:** v1 pull model was **pessimistic** and coupled to sandbox state. **v2 fixes pull physics** on determined/co-sim path; sandbox still uses last-task match. **GNN still lacks explicit disk feature** — must infer from `is_cold` / `shared_fate` proxies.

### 12.3 Interaction with FilterStore contention

Warmth model **creates** N cold pulls; FilterStore **serializes** them. Both are required for observed N× ~31s steps:

- Fix warmth only (node disk hit) → same node may drop to ~1× T_pull
- Fix FilterStore only (4× flashCard) → N parallel ~31s pulls but each platform still cold

Verified: 4× flashCard → all tasks **31.65s** (`memory/storage_contention.md`).

### 12.4 What the codebase authors knew

Explicit TODOs acknowledge missing disk cache check:

```
# TODO: Retrieve image if function not in RAM cache nor in disk cache
```

HeROcache implements half the TODO. Main path does not.

---

## 13. Design alternatives not implemented

| Model | Pull behavior | Sandbox behavior | In repo? |
|-------|---------------|------------------|:--------:|
| **Current (last-task match)** | Skip if prev same type | Skip if prev same type | Knative, determined, GNN co-sim |
| **Node disk cache hit** | Skip if `has_function` | Unchanged | **HeROcache only** |
| **TTL warmth** | Optional | Warm if same type within Δt idle | No |
| **LRU k-recent types** | Optional | Warm if type in last-k | No |
| **Separate pull vs sandbox flags** | Disk / node scoped | Platform previous_task | Partially in HRC |
| **Load-from-disk latency** | Small timeout vs T_pull | Unchanged | No |
| **Clear previous_task on scale-down** | — | — | No |

### Fair feature candidates (future work — see storage_contention.md)

- `node_cold_count`, `storage_busy`, `estimated_pull_remaining_sec = pending × T_pull`
- Require autoscaler instrumentation; retrain GNN/MLP after recache

---

## 14. Recommended paper wording

### Defensible claims

> “We model **execution-context reuse** at the platform granularity: if the immediately prior invocation on that platform executed the same task type, we skip the sandbox cold-start delay (~0.33 s for dnn1/rpiCpu). Otherwise a cold-start penalty is applied.”

> “Cold **image retrieval** is modeled as a discrete-event transfer (~31 s for a 3 GB image over a 100 Mb/s link) when the platform has not previously executed the same task type in its local reuse state.”

### Required caveats

> “Co-simulation and GNN label generation use **`node_disk_v2`**: image pull is skipped when the function image is already present on **any local storage on the node** (`has_function`), independent of which platform last executed the task. Sandbox cold-start (~0.33 s) still uses per-platform `previous_task` type match only.”

> “Legacy pre-v2 training corpus (1060) used coupled v1 pull physics. Comparisons across warmth versions require regen + recache + retrain.”

> “Warmth does not time out with idle duration; replica scale-down resets `initialized` but not `previous_task`. Learned policies do not observe explicit disk-cache bits in the current dim14 graph — only pull-readiness proxies (`shared_fate`, `is_cold`).”

> “Co-sim reports ~71% platforms not yet initialized at scheduling vs ~0.1% of tasks recording non-zero sandbox cold-start time — reflecting that **pull wait dominates** and is captured in queue/arrival metrics, not `coldStartTime`.”

### Claims to avoid

- ❌ “Faithful end-to-end model of Kubernetes cold start including image layer caching”
- ❌ “Task coldStartTime proportion equals platform cold fraction”
- ❌ “Warmth and pull costs are independently modeled” (they share one predicate on main path)

---

## 15. File index

| File | Role in warmth model |
|------|---------------------|
| `src/placement/infrastructure.py` | `Platform.previous_task`, sandbox warm check, `Storage.functions_cache`, task metrics |
| `src/policy/determined/autoscaler.py` | Pull gate for co-sim / A/B tests; FilterStore hold during pull |
| `src/policy/knative/autoscaler.py` | Pull gate for live Knative sim |
| `src/policy/herocache/autoscaler.py` | Pull gate + `has_function` disk cache |
| `src/policy/determined/scheduler.py` | `defer_cold_replica_init`, placement-triggered pull |
| `src/placement/simulation.py` | Preinit warm/cold seeding, `previous_task` for warmup |
| `src/placement/autoscaler.py` | Scale-down: reset `initialized`, retain `previous_task` |
| `src/policy/state_capture.py` | `initialized_snapshot`, `previous_task_type_name` |
| `src/notebooks/prepare_graphs_cache.py` | dim14 `shared_fate`, edge `is_warm` |
| `src/notebooks/prepare_graphs_cache_seq.py` | atomic21 `is_cold` from snapshot |
| `src/policy/tabular/feature_builder.py` | Live inference warmth features |
| `src/policy/tabular/constants.py` | Platform dim layout docs |
| `src/policy/gnn/scheduler.py` | Initialized-replica fallback |
| `src/placement/warmth.py` | v1/v2 predicates, `needs_image_pull`, `sandbox_is_warm` |
| `scripts_cosim/test_cold_start_queue_last_task_ab.py` | Gate B v1/v2 matrix (~125s vs ~31.65s) |
| `scripts_cosim/pilot_warmth_regen_audit.py` | Pilot label-shift audit |
| `scripts_cosim/run_warmth_full_regen_recache.sh` | Full 1060 regen + recache + retrain |
| `scripts_cosim/generate_gnn_datasets_fast.py` | BF co-sim; **must** persist `placements/placements.jsonl` |
| `memory/placements_jsonl_required.md` | Policy: JSONL mandatory; repair ≠ substitute |
| `scripts_cosim/test_memory_contention_ab.py` | Warm vs cold counterfactual (~0.35s vs ~125s) |
| `memory/storage_contention.md` | N× pull serialization (companion) |
| `data/nofs-ids/task-types.json` | `coldStartDuration`, `imageSize` priors |

---

## 16. Changelog

| Date | Change |
|------|--------|
| 2026-06-11 | v2 memory pass: §5.3 v1/v2 split; §9.4 disk feature gap; §9.5 src_norm; §10c warmth_v2 status; seq `is_warm` degeneracy; §11.1/§12.2/§14 aligned to v2 default |
| 2026-06-11 | **Shipped** `node_disk_v2` via `warmth.py` + all Knative-family autoscalers; Gate B verified (N=4 contended 125s→31.65s); co-sim defaults v2+defer_cold |
| 2026-06-11 | §10b + cross-link to `cosim_warmth_gap.md` (1060 pullTime=0 audit, regen plan) |
| 2026-06-11 | Initial expert reference — warmth predicate, disk cache gap, policy diff, GNN alignment, failure modes, paper wording |
