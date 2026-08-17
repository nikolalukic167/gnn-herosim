# Simulator Landscape: HeroSim (fork + upstream) vs PureEdgeSim vs EdgeCloudSim vs NoServer

> Consolidated notes from 2026-06-16 (updated with deep PureEdgeSim + EdgeCloudSim analysis). Covers static/dynamic labels, GNN data generation, workloads, co-sim portability, timing, batch scheduling, graph semantics, scheduling-model vocabulary, porting strategy, PureEdgeSim sticky placement / Garmendia GNN paper, EdgeCloudSim per-task VM fit, and NoServer LB-only GNN suitability.
>
> Cloned repos for inspection: `/root/projects/PureEdgeSim` (v5.3.0), `/root/projects/EdgeCloudSim` (Oct 2025 refresh, CloudSim 7.0.0-alpha), `/tmp/noserver` (LB benchmarks 2026-06-16, incl. high-load). **HeroSim fork** = this repo (`my-herosim`). **HeroSim upstream** = [b-com/herosim](https://github.com/b-com/herosim).

---

## 1. Executive summary

| | **HeroSim (upstream)** | **HeroSim (this fork)** | **PureEdgeSim** | **EdgeCloudSim** | **NoServer** |
|---|---|---|---|---|---|
| **Language** | Python (SimPy), pipenv | Same base + co-sim/ML stack | Java 8+ (custom DES) | Java 21+ (CloudSim DES) | Python, `requirements.txt` |
| **Primary goal** | Serverless policy comparison (HRC/HRO/Knative baselines) | Co-sim label factory + GNN placement research | General edge/cloud/mist evaluator | Edge offloading policy evaluator (original Boun tool) | Serverless full-system queueing (Knative/K8s-inspired) |
| **ML built-in** | No | Yes (co-sim, graph cache, GIN/MLP, live GNN) | No (Garmendia et al. external) | Weka MLP/MAB in `sample_app5` only | No |
| **Graph for ML** | Policy state only | Task–platform bipartite (PyG) | Infrastructure routing (JGraphT) | None (proposed: task→VM) | Workflow DAG (NetworkX) |
| **Label generation** | Aggregate metrics per policy run | Brute-force `placements.jsonl` + SSC | Aggregate CSV only | Aggregate CSV (+ optional per-task logs) | None |
| **Per-request routing** | Scheduler → replica queue | Same + batch GNN / `determined` replay | Orchestrator (sticky per device) | **EdgeOrchestrator per task → VM** | **LB** → instance |
| **Best fit for our GNN pipeline** | Partial (no co-sim) | **Native** | Major port; sticky blocks per-task | Major port; per-task OK, wrong physics | LB-only port possible; not drop-in |

**Bottom line:** Upstream HeroSim is a clean serverless policy sandbox. **This fork** is a placement-oracle lab. **EdgeCloudSim** (predecessor to PureEdgeSim, same Boun research line) has **better routing semantics for our RQ than PureEdgeSim** — per-task VM selection, no sticky placement — but still **not a drop-in label factory**: static VM pools (no replica warmth/cold-start), CPU-% utilization model, no SSC, no brute-force replay, ~**1000× slower** than HeroSim per placement without `FastReplayManager`. PureEdgeSim adds mist tier/energy but **sticky placement** collapses ~11k tasks into ~100 orchestrator decisions. NoServer is strong for serverless control-plane studies; **GNN-at-LB-only** has modest leverage vs `least_loaded` when Knative autoscaler is on (~0.2–0.5% mean latency). EdgeCloudSim VM-fit policy spread in tutorial1 is **small** (~0.14 pp failure-rate, ~0.4% mean service time) under edge-only high-completion defaults.

---

## 2. HeroSim upstream vs this fork

| Area | **b-com/herosim (upstream)** | **my-herosim (fork)** |
|---|---|---|
| Input | Workload trace + infra + task/app/platform JSON priors | Same + `ds_*` co-sim datasets, live-eval traces |
| Policies | Random, RoundRobin, Knative, HeROfake, HeROcache, BPFF | + GNN, XGB, tabular, `determined`, hetero variants |
| Scheduling unit | **One request → one scheduler call** | + **batch collect** (`batch_size` 2–4, timeout) + joint GNN placement |
| Co-sim | Not first-class | **`executecosimulation.py`**, `placements.jsonl`, SSC, RTT hash |
| Forced placement | No | `forced_placements` + `DeterminedScheduler` |
| Graph/ML | Charts comparing policy runs | `prepare_graphs_cache.py`, `train.py`, live gate scripts |

Upstream README models the classic split: **Autoscaler** (how many replicas, where created) + **Scheduler** (which replica gets each request). The fork adds a **research layer** on top without changing the core “measured priors + SimPy queues” physics.

---

## 3. Static vs dynamic state and labels

### 3.1 HeroSim (upstream + fork)

**Static (topology / priors / scenario):**
- Task: type one-hot (`dnn1`/`dnn2`), normalized source node id
- Platform: type one-hot (`rpiCpu`, `xavierGpu`, …), replica flags (`has_dnn1`, `has_dnn2`)
- Edge attrs from priors: exec time, energy, comm time, network latency from `network_map`
- Infrastructure JSON, workload, replica plan
- **Execution/cold-start:** lookup `task-types.json` → `yield timeout(fixed)` on platform

**Dynamic (fork: frozen at scheduling time via SSC):**
- `full_queue_snapshot` → normalized queue depth per platform
- `full_temporal_state_at_scheduling` → remaining exec / cold / comm, `previous_task_type_name` (for `is_warm`)
- `initialized_snapshot` → `shared_fate_signal` (cold-replica density on co-located platforms)

**What actually moves at runtime (both):** queue wait, replica create/delete (autoscaler), cold-start *when* it happens — not per-run remeasurement of exec time.

**Labels (fork only):**
- Brute-force co-simulation: freeze state → enumerate feasible `(placement_plan → batch RTT)` → `placements/placements.jsonl`
- Per-task CE label `y` = optimal platform index on feasible edges
- RTT hash (`rtt_chunk_*.pkl`) for regret / near-RTT training
- See `paper/cosimulation.md`, `memory/placements_jsonl_required.md`

**Platform features (14-dim in `prepare_graphs_cache.py`):** 5 type + 2 replica + 1 queue + 1 shared-fate + 3 temporal + 2 consolidation.

### 3.2 PureEdgeSim

**Static (XML / config):**
- Task: MI length (`task_length`), request/result/container sizes (KB→bits in parser), per-app latency budget (`latency` seconds), generation rate (`rate` tasks/min), `usage_percentage` per app
- Node: `mipsPerCore`, `numberOfCPUCores`, RAM, storage, geo position, battery (edge devices)
- Network: WAN/MAN/LAN/WiFi/LTE bandwidth (Mbps) and latency constants in `simulation_parameters.properties`
- Topology: `edge_datacenters.xml` mesh + device→nearest-DC links; JGraphT shortest paths precomputed in `InfrastructureGraph`

**Dynamic (runtime, used inline, not exported for ML):**
- Per-node: `tasksQueue`, `availableCores`, `availableRam`/`availableStorage`, in-flight MI (`tasks`), `getAvgCpuUtilization()`
- Per-device: mobility position (`update_interval`), battery drain (`EnergyAwareNode`), `isApplicationPlaced` + `applicationPlacementLocation`
- Network: shared WAN when `one_shared_wan_network=true` — link utilization via `TransferProgress`
- Orchestrator: `historyMap` assignment counts (feeds TRADE_OFF scoring)

**RTT / delay definition** (`TaskAbstract.getTotalDelay()`):
```text
totalDelay = actualNetworkTime + waitingTime + actualCpuTime
failure if totalDelay >= maxLatency  (per-app deadline from applications.xml)
```
Exec time is **fixed** `task_length / mipsPerCore` per core — same “lookup physics, queue dynamics” pattern as HeroSim.

**Sticky placement (modern upstream):** first task per edge device runs `orchestrate()`; later tasks reuse `getApplicationPlacementLocation()` until failure/mobility/death. With 100 devices and ~11,100 tasks, orchestrator runs **~100 times**, not per-task. Fights per-task / joint-batch labeling unless bypassed.

**Garmendia et al. (CMES 2024, DOI [10.32604/cmes.2024.045912](https://doi.org/10.32604/cmes.2024.045912)):** GNN+DQN on infrastructure graph (machines + links), “offloading rating” edge feature; baselines TRADE_OFF/ROUND_ROBIN; optimizes success rate under latency budgets. Code **not in PureEdgeSim tree** (checked: no GNN/DQN references in Java sources). Does **not** document bypassing stickiness; likely custom `Orchestrator`/`SimulationManager` subclass. Not HeroSim-style bipartite co-sim with oracle RTT tables.

**Labels:** aggregate experiment CSV (`SimLog.resultsList`) — avg delay, failure rates, energy, per-tier execution counts. No per-decision oracle RTT table, no `placements.jsonl`.

**SSC mapping (fork contract vs PureEdgeSim):**

| SSC field (fork) | PureEdgeSim equivalent | Gap |
|---|---|---|
| `full_queue_snapshot` (208 platforms) | Per-node `tasksQueue` in `DefaultComputingNode` | No snapshot API |
| `temporal_state` (cold/comm remaining) | None | No cold-start duration model (optional container pull via `enable_registry`) |
| `initialized_snapshot` / warmth | Container RAM reservation | No replica warmth / `shared_fate_signal` |
| `replica_plan` | Sticky `applicationPlacementLocation` per device | Different semantics |
| Heterogeneous CPU/GPU (`dnn1`/`dnn2`) | Tier types (cloud/edge DC/mist) + MIPS | No GPU platform types |

### 3.3 EdgeCloudSim

**Static (properties + XML):**
- Task: MI length, upload/download sizes (KB), `prob_cloud_selection`, Poisson inter-arrival, active/idle periods, CPU util % on edge/cloud/mobile (`applications.xml`)
- Infrastructure: `edge_devices.xml` — datacenters, hosts, VMs (cores, MIPS, RAM); geo + `wlan_id`
- Network: WAN/LAN propagation delays; empirical WLAN/WAN throughput tables in `SampleNetworkModel` (many configs set `wlan_bandwidth=0` to force table lookup)

**Dynamic (runtime, not exported for ML):**
- CloudSim per-VM cloudlet queues + `getTotalUtilizationOfCpu()` (% capacity model)
- Network: WLAN/WAN client counters, M/M/1 queuing (`MM1Queue`); upload/download as separate `SimEvent` chains
- Mobility: precomputed `TreeMap<time, Location>`; failures if `servingWlanId` changes mid-transfer
- Orchestrator state: round-robin indices for NEXT_FIT (`lastSelectedHostIndex`)

**RTT / delay definition** (`SimLogger` / `LogItem`):
```text
serviceTime = taskEndTime - taskStartTime   (upload start → result delivery)
networkDelay = sum(WLAN + MAN + WAN + GSM upload/download segments)
processingTime ≈ serviceTime - networkDelay
```
Exec: CloudSim cloudlet length / MIPS with `CpuUtilizationModel_Custom` (% from XML). **Fixed per task** — queue wait moves completion, not remeasurement.

**Per-task placement (no stickiness):** every `CREATE_TASK` → `submitTask()` → `getDeviceToOffload()` + `getVmToOffload()`. ~15k orchestrator decisions per scenario (200 devices, 10 min sim). **Unlike PureEdgeSim**, aligned with per-task / joint-batch labeling semantics (still wrong physics for warmth).

**Labels:** `SimLogger` → aggregate CSV (`SIMRESULT_*_GENERIC.log`); optional per-task `SUCCESS.log`/`FAIL.log` when `deep_file_log_enabled=true`. No `placements.jsonl`, no RTT hash.

**SSC mapping (fork contract vs EdgeCloudSim):**

| SSC field (fork) | EdgeCloudSim equivalent | Gap |
|---|---|---|
| `full_queue_snapshot` (208 platforms) | Per-VM CPU utilization % | Different granularity; no snapshot API |
| `temporal_state` (cold/comm remaining) | None | No cold-start / comm-in-flight model |
| `initialized_snapshot` / warmth | N/A (VMs always exist) | No replica warmth / `shared_fate_signal` |
| `replica_plan` | Static VM inventory at sim start | No scale-to-zero |
| Heterogeneous CPU/GPU (`dnn1`/`dnn2`) | VM types + CPU % util | No GPU platform types |

**Relationship to PureEdgeSim:** Same Boun research line; PureEdgeSim is a ground-up rewrite (custom DES, mist tier, energy, TRADE_OFF/ROUND_ROBIN, sticky placement). EdgeCloudSim is the **older CloudSim-based** tool (Sonmez et al. ETT 2018).

### 3.4 NoServer

**Static:** per-DAG-node `duration_milli`, `memory_mib`, `vcpu`; invocation CSV / Poisson traces; HarvestVM pickles (`cores_table`, survival model) when enabled.

**Dynamic (control plane):** Knative autoscaler (panic/stable windows, scale-to-zero), K8s scheduler binding new instances to nodes, throttler queues, cold/warm CRI delays (3s / 1s default), core `book_cores()` FCFS contention, optional HarvestVM core shrink/grow + preemption (`failed=True` if preempted before `duration` met).

**Labels:** none. Runtime trace CSV from `cluster.dump()`; validation against vHive.

### 3.5 Dynamic comparison (simple)

| Dynamic effect | HeroSim | PureEdgeSim | EdgeCloudSim | NoServer |
|---|---|---|---|---|
| Queue wait | Per-platform FIFO | Per-node execution queue (core/RAM gated) | Per-VM cloudlet queue (CPU % gated) | Throttler + instance + core runqueue |
| Exec time varies with load | **No** (fixed lookup) | **No** (fixed MI/MIPS) | **No** (fixed MI + util %) | **No** (fixed `duration`) |
| Autoscale | Yes | **N/A** | **N/A** (static VMs) | **Knative-style** (always on) |
| Cold start | Fixed duration when replica missing | Container pull optional | **None** | Instance creation delay |
| Preemption mid-run | No | No | No | Yes (HarvestVM) |
| Mobility / energy / WAN contention | No | **Yes** (core) | Mobility + WLAN/WAN (**no energy**) | No (fixed 10ms DAG stage delay) |
| Heterogeneous platforms (CPU/GPU) | **Yes** | Tier types only | VM types only | `vcpu` only |
| Per-task placement decisions | **Yes** | **No** (sticky: ~1 per device) | **Yes** (per task → VM) | Yes (LB per request) |

**NoServer** is more dynamic than HeroSim for cluster/control-plane behavior. **PureEdgeSim** is more dynamic on mobility/energy/mist. **EdgeCloudSim** has per-task routing like HeroSim but static warm VMs and no replica warmth — **closer routing abstraction than PureEdgeSim, farther physics match than HeroSim fork**.

---

## 4. Scheduling models (vocabulary — easy to confuse)

### 4.1 HeroSim

```text
Request → Autoscaler (how many replicas? on which node+platform?)
       → Scheduler (which replica queue?)   ← this is LB in serverless terms
       → Platform queue → fixed exec time
```

### 4.2 NoServer — three names, three jobs

```text
Request → Throttler + LB (which existing instance?)     ← ≈ HeroSim Scheduler
       → Autoscaler (how many instances?)                 ← ≈ HeroSim Autoscaler
       → Scheduler (K8s: new instance → which worker?)   ← part of HeroSim Autoscaler
       → Instance + core booking → incremental CPU time
```

| NoServer term | Plain English | HeroSim analogue |
|---|---|---|
| **LB** (`loadbalance.py`) | Route request to a warm instance | **Scheduler** (e.g. Knative least-queue) |
| **Autoscaler** | Replica count, cold start, scale-to-zero | **Autoscaler** |
| **Scheduler** (`scheduler.py`) | Bind **new** pods to worker nodes | Where new replica lands |

**For co-sim:** freeze autoscaler + K8s scheduler after warmup → **only LB decisions remain** — same philosophy as fork co-sim with fixed `replica_plan`.

### 4.3 PureEdgeSim — orchestrator, stickiness, no autoscaler

```text
Edge device generates task
  → NetworkModel (device → orchestrator, if enable_orchestrators=true)
  → Orchestrator.orchestrate()          ← only on FIRST task per device
  → sticky: getApplicationPlacementLocation()  ← all later tasks skip orchestrator
  → NetworkModel (orch → destination; optional container pull from registry)
  → DefaultComputingNode.submitTask()   ← per-core queue, fixed MI/MIPS exec
  → results back over network → device
```

| PureEdgeSim term | Plain English | HeroSim analogue |
|---|---|---|
| **Orchestrator** (`DefaultOrchestrator`) | Pick cloud/edge/mist node for offload | Partially **Scheduler** + where new replica lands — but **once per device** |
| **Sticky placement** (`isApplicationPlaced`) | Device bound to one destination until failure/mobility | **No equivalent** — we route every task independently |
| **Architecture filter** (`CLOUD_ONLY`, `ALL`, …) | Which tier candidates exist | Topology + `network_map` reachability |
| **TRADE_OFF / ROUND_ROBIN** | Built-in orchestration algorithms | Knative / RoundRobin baselines |
| **DefaultComputingNode queue** | Per-node FIFO when cores/RAM busy | Per-platform SimPy queue |
| **Autoscaler / scale-to-zero** | **Not present** | Core HeroSim/NoServer dynamic |

**Sticky gate** (`DefaultSimulationManager.sendFromOrchToDestination()`):
- If `!task.getEdgeDevice().isApplicationPlaced()` → `edgeOrchestrator.orchestrate(task)` + set `applicationPlacementLocation`
- Else → `task.setOffloadingDestination(task.getEdgeDevice().getApplicationPlacementLocation())`

**Invalidation:** task failure, device/orchestrator death, mobility (`sameLocation()` distance check vs `edge_devices_range`).

**Built-in orchestration algorithms** (`DefaultOrchestrator`):
- **TRADE_OFF:** `(historyMap[i] + 1) * weight * task.length / node.mipsPerCore` with tier weights (cloud=1.8, edge device=1.3, edge DC=1.2); filtered by `offloadingIsPossible()` (architecture + proximity + link topology)
- **ROUND_ROBIN:** min `historyMap` count among feasible nodes

Not the same as LB among warm replicas — it's “this phone always uses edge server X until failure.”

### 4.4 EdgeCloudSim — tier + VM fit, per-task, no autoscaler

```text
CREATE_TASK (pre-scheduled at task.startTime)
  → MobileDeviceManager.submitTask()
  → EdgeOrchestrator.getDeviceToOffload()     ← cloud / edge / mobile tier
  → NetworkModel upload delay (WLAN/WAN/MAN SimEvents)
  → EdgeOrchestrator.getVmToOffload()       ← FIRST/NEXT/BEST/WORST/RANDOM fit
  → CloudSim VM cloudlet queue → fixed MI + CPU util %
  → NetworkModel download delay
  → SimLogger.taskEnded()
```

| EdgeCloudSim term | Plain English | HeroSim analogue |
|---|---|---|
| **EdgeOrchestrator** | Tier + VM pick **per task** | **Scheduler** + partial placement |
| **WORST_FIT / BEST_FIT / …** | VM bin-packing on residual CPU % | Knative / RoundRobin among replicas |
| **selectVmOnLoadBalancer()** (`TWO_TIER_WITH_EO`) | Global edge VM LB | Scheduler across warm replicas |
| **selectVmOnHost()** (location-aware) | VM on device's serving WLAN host | Topology-constrained routing |
| **Static VM pool** | All VMs created at start | Fixed `replica_plan` post-warmup |
| **Sticky / autoscaler / cold start** | **Not present** | Core fork dynamics |

**tutorial1** (`SampleEdgeOrchestrator`): `DEFAULT_SCENARIO` = edge-only; WORST_FIT scans **all VMs globally** (112 VMs / 14 datacenters) — not location-constrained. **tutorial3** adds cloud vs edge tier policies (`NETWORK_BASED`, `UTILIZATION_BASED`, `RANDOM`).

---

## 5. GNN data generation

| Capability | HeroSim fork | PureEdgeSim | EdgeCloudSim | NoServer |
|---|---|---|---|---|
| State snapshot at decision time | SSC contract (strict) | Not standardized | Not standardized | Rich state, no snapshot API |
| Brute-force placement sweep | `executecosimulation.py` → `placements.jsonl` | None | None | None (buildable LB-only) |
| Counterfactual RTT lookup | `rtt_chunk_*.pkl` | None | None | None |
| Bipartite graph export | `prepare_graphs_cache.py` → PyG | Infra graph (JGraphT) ≠ ML graph | Would be task→VM (new export) | DAG only; LB graph = request→instance |
| Train/serve pipeline | `train.py`, `GNNScheduler` | Garmendia (external) | Weka trainer (`sample_app5`) | None |
| Deterministic workload | Yes (`workload.json`) | **No** (`SecureRandom`) | **No** (`SimUtils.RNG` time-seeded) | Yes (seeded modes) |
| Forced placement hook | `forced_placements` + `DeterminedScheduler` | Override `Orchestrator` / sticky bypass | Custom `EdgeOrchestrator` / `submitTask` | Build `DeterminedLB` |

**HeroSim co-sim protocol:** warmup + SSC → enumerate placements → `placements.jsonl` (mandatory) → graph cache → train.

**PureEdgeSim co-sim (not implemented; major engineering):**
1. Custom `TraceTaskGenerator` with fixed seed/trace (replace `SecureRandom`)
2. Warmup → `StateSnapshot` (node queues, cores, RAM, network link state, `historyMap`)
3. **Disable stickiness** in `sendFromOrchToDestination()` for per-task / joint-batch labeling
4. Enumerate feasible placements → `FastReplayManager` (state clone + replay) → `placements.jsonl`
5. `BipartiteGraphBuilder` Java export or Python sidecar → train pipeline
6. Blockers today: no state freeze API, no per-placement RTT export, full re-sim ~1.7 s/scenario

**EdgeCloudSim co-sim (not implemented; major engineering):**
1. `TraceLoadGenerator` with fixed seed/trace (replace `SimUtils.RNG`)
2. Warmup → `StateSnapshot` (VM CPU %, network client counts, mobility positions)
3. Custom `EdgeOrchestrator` with `forcedVmId` per task/batch (per-task routing already native)
4. Enumerate feasible placements → `FastReplayManager` (CloudSim state clone is hard) → `placements.jsonl`
5. Bipartite export (task→VM) → train pipeline
6. Blockers: no warmth/contention semantics; tier+VM two-hop ≠ single platform index; CloudSim replay overhead ~1.8 s/scenario

**NoServer LB-only co-sim (not implemented; feasible):**
1. `test`/`rps` mode, single function, no DAG
2. Warmup → `capture_state()` (instances, queues, autoscaler scales)
3. Freeze AS + K8s scheduler; `USE_HARVESTVM=False`
4. Inject batch of N requests; enumerate joint `{req_i → instance_j}`
5. `DeterminedLB` forced routing → record latency → JSONL
6. Optional: mirror fork batching via `BatchThrottler` (collect N within timeout, joint assign)

---

## 6. Workloads

### HeroSim
- **Co-sim:** `workload.json` — 1–4 events, `timestamp`, `application` DAG, `node_name`
- **Live eval:** e.g. `workload-125-225.json` — ~28k events
- Deterministic, file-driven

### PureEdgeSim
- Synthetic from `applications.xml` + device count + `simulation_time` (minutes)
- `DefaultTaskGenerator`: assign apps to devices by `usage_percentage`; per device per minute emit `rate` tasks with jitter (`random.nextInt(15)`)
- Default ~11,100 tasks/scenario (100 devices, 10 min, 3 apps); **not seeded** (`SecureRandom.getInstanceStrong()`)
- Custom `TaskGenerator` subclass required for trace replay; no built-in trace format

**Default `applications.xml` apps:**

| App | rate/min | usage% | latency (s) | task_length (MI) |
|---|---|---|---|---|
| Health | 20 | 20% | 0.02 | 500 |
| Augmented reality | 30 | 30% | 0.5 | 5000 |
| HEAVY_COMP_APP | 3 | 50% | 300 | 30000 |

### EdgeCloudSim
- **Default:** `IdleActiveLoadGenerator` — devices assigned one app type (weighted); active/idle cycles; Poisson inter-arrivals within active periods; exponential task length/sizes
- Full `List<TaskProperty>` built **before** DES (`initializeModel()`); `SimManager` schedules all `CREATE_TASK` events upfront
- **Benchmark (tutorial1):** 200 devices, 10 min sim → **~15,060 tasks** per scenario
- **Not seeded** — `SimUtils.RNG = new Random(System.currentTimeMillis())`; task list **regenerated each scenario** (policy comparisons not apples-to-apples unless patched)
- Custom `LoadGeneratorModel` required for trace replay

**Default `applications.xml` apps (tutorial1):** AR (30%, poisson 2s), Health (20%, 3s), Infotainment (50%, 7s) — structurally similar to PureEdgeSim three-app pattern, different numeric parameters.

### NoServer
- Modes: `test`, `rps`, `dag`, `benchmark`, `trace`
- DAG JSON + invocation CSV or pickled `trace_dags.pkl`
- Azure-inspired validation setup (1s exec, 170 MiB)

**Not interchangeable** without adapters.

---

## 7. Batch scheduling

| System | What “batch” means |
|---|---|
| **HeroSim upstream** | One task per scheduler iteration |
| **HeroSim fork** | Collect 2–4 tasks (`batch_timeout`); **joint** GNN / co-sim optimal batch RTT |
| **PureEdgeSim** | `batch_size=100` = DES event chunking only (`SEND_TO_ORCH` scheduling); **not** joint placement; sticky per-device |
| **EdgeCloudSim** | Statistical iteration loops in `MainApp` / `run_scenarios.sh` only; **one cloudlet per orchestrator call** |
| **NoServer (today)** | One request per LB call; DAG releases successors sequentially |
| **NoServer (proposed)** | `BatchThrottler`: collect N requests in window → joint LB or co-sim sweep — **same pattern as fork** |

PureEdgeSim `batch_size` code (`DefaultSimulationManager.onSimulationStart()`): schedules up to `batchSize` future `SEND_TO_ORCH` events into DES queue at once to reduce memory — tasks still fire at individual `task.time`. Does **not** group tasks for joint orchestration.

EdgeCloudSim joint batching would require custom `EdgeOrchestrator` semantics (same as fork `contention_v2` — not built-in).

Batching in NoServer is **not built-in** but **straightforward** for `test`/`rps` + frozen AS: analogous to `_collect_task_batch` in `GNNScheduler`.

---

## 8. Graph types

| | Nodes | Edges | ML use |
|---|---|---|---|
| **HeroSim fork** | Tasks + platforms | Feasible task→platform | GNN input |
| **PureEdgeSim (JGraphT)** | Computing nodes | Network links (latency-weighted) | Routing only; Garmendia infra GNN |
| **PureEdgeSim (proposed co-sim)** | Tasks + edge/cloud nodes | Reachability | Would need new export; schema ≠ fork |
| **EdgeCloudSim (proposed co-sim)** | Tasks + edge/cloud VMs | Feasible task→VM | Would need new export; schema ≠ fork |
| **EdgeCloudSim (built-in)** | None | — | Weka feature vectors in `sample_app5` trainer only |
| **NoServer** | Functions (DAG) | Dependencies | Workflow only |
| **NoServer LB graph (proposed)** | Requests + instances@nodes | Feasible request→instance | LB GNN |

`InfrastructureGraph` (PureEdgeSim): Dijkstra shortest paths, Floyd-Warshall delay queries — **not** exposed as ML observation tensors. EdgeCloudSim topology is XML datacenters + mobility timelines — no JGraphT export.

---

## 9. Clock-time and co-sim economics

### HeroSim fork
| Mode | Wall time | Notes |
|---|---|---|
| Co-sim one placement | ~10 ms | Frozen state, `determined`; ~100 sim/s |
| Co-sim full dataset | `combinations / 100` s | Up to 160k combos worst case (4 tasks × ~20 platforms) |
| Live full trace | GNN ~10 min | Wall-clock policy eval |

### PureEdgeSim (benchmarked 2026-06-16)

| Mode | Wall time | Notes |
|---|---|---|
| Full default sweep (42 scenarios) | **~41 s** | 7 arch × 2 alg × 1 device count; charts on |
| Single scenario (11k tasks, headless) | **~1.7 s** | `ALL` arch, charts off, `pause_length=0` |
| Per-task amortized | **~0.15 ms** | 1.7 s / 11,100 tasks (but not per-placement replay) |
| Hypothetical 4-task brute force (160k combos) | **~74 hours** | If each combo = full scenario re-sim |
| First-task-only sweep (100 devices × 20 candidates) | **~1 hour** | 2,000 full sims × 1.7 s |

**Not comparable** to HeroSim ~10 ms/placement replay without a new state-clone fast path.

### EdgeCloudSim (benchmarked 2026-06-16, `/root/projects/EdgeCloudSim`)

| Mode | Wall time | Notes |
|---|---|---|
| 2 policies, 200 devices, 10 min sim (~15k tasks each) | **3.54 s** total | WORST_FIT + RANDOM_FIT, tutorial1 `DEFAULT_SCENARIO` |
| Per scenario | **~1.77 s** | Comparable to PureEdgeSim single-scenario |
| Per-task amortized | **~0.12 ms** | 1.77 s / 15,060 tasks |
| Hypothetical 4-task brute force (160k combos) | **~78 hours** | If each combo = full scenario re-sim |

**Build / run gotchas:**
- **Java 21+ required** — CloudSim 7.0.0-alpha JAR is class file version 65; Java 11 fails to compile
- **`vm_load_check_interval=0` bug:** README says logging disabled at 0, but `SimManager` still `schedule(..., 0, GET_LOAD_LOG)` → **infinite loop** (benchmark runs hung 12+ min until killed). Use `0.1` (default) or patch scheduler
- Compile: `cd scripts/tutorial1 && ./compile.sh` → `bin/`; classpath per README

### NoServer (benchmark 2026-06-16)
- 1 ms tick loop; stressed LB run (16 RPS, 1 function, 2×16-core workers, 60s sim) → **~15–44 s wall** per policy depending on logging
- Co-sim sweep would need fast path or accept slower labeling than HeroSim

---

## 10. Co-sim portability matrix

| HeroSim artifact | PureEdgeSim | EdgeCloudSim | NoServer |
|---|---|---|---|
| `workload.json` | Custom `TaskGenerator` + deterministic seed | Custom `LoadGeneratorModel` + seeded RNG | Custom ingress (`test`/`rps`) |
| `infrastructure.json` | XML suite (`edge_devices`, `edge_datacenters`, `cloud`) | `edge_devices.xml` + `config.properties` | Worker/instance model |
| `replica_plan` / warmth | Sticky `applicationPlacementLocation` — different semantics | Static VMs — freeze = fixed inventory | Freeze instances post-warmup |
| SSC | Build `StateSnapshot` from node queues + network state | Build from VM CPU % + network counters | Build `capture_state()` |
| `placements.jsonl` | New brute-force loop + `FastReplayManager` + JSONL writer | Same (CloudSim clone harder) | LB combo sweep (different semantics) |
| `rtt_chunk_*.pkl` | Java export / Python sidecar | Java export / Python sidecar | Same |
| `prepare_graphs_cache.py` | New bipartite builder (full rewrite) | New bipartite builder (task→VM) | Request→instance graph schema |

**Mandatory blockers for PureEdgeSim port of our protocol:**
1. Disable `isApplicationPlaced()` stickiness for collision labeling (`contention_v2` requires shared replicas)
2. Deterministic workload (replace `SecureRandom.getInstanceStrong()` in `DefaultTaskGenerator`)
3. State freeze + replay API (no `forced_placements` hook today)
4. Per-placement RTT export (`SimLog` is scenario-aggregate only)

**Mandatory blockers for EdgeCloudSim port of our protocol:**
1. No warmth / cold-start / `shared_fate_signal` — physics mismatch for `contention_v2`
2. Deterministic workload (replace time-seeded `SimUtils.RNG`)
3. No state freeze + fast replay (CloudSim event clone is non-trivial)
4. Two-hop tier+VM decision ≠ single platform index label
5. No joint batch placement

**EdgeCloudSim vs PureEdgeSim for port:** per-task routing is **already native** (no stickiness bypass needed); still weeks of engineering for Phases 1–3 and wrong warmth physics.

---

## 11. EdgeCloudSim port sketch

### Phase 1 — Deterministic workload + snapshot
- `TraceLoadGenerator`: fixed `(time, device, app)` trace; seeded `SimUtils.RNG`
- Warmup run → `StateSnapshot`: per-VM CPU utilization, network `wlanClients`/`wanClients`, mobility positions, orchestrator round-robin state

### Phase 2 — Counterfactual replay
- `CosimEdgeOrchestrator`: inject `forcedVmId` per task/batch in `getVmToOffload()`
- Enumerate feasible placements → `FastReplayManager` (CloudSim state clone or full re-sim) → `placements/placements.jsonl`
- **Economics:** ~1.8 s/scenario today → **~78 hours** for naive 160k-combo sweep vs **~27 min** on HeroSim fork

### Phase 3 — Graph export + train
- `BipartiteGraphBuilder` (Java) or sidecar → PyG schema parallel to `prepare_graphs_cache.py` **or** separate train pipeline
- Feature mapping: VM MIPS/cores/RAM/queue util vs our 14-dim platform features (no warmth, no GPU)

### Build / ops notes
- Package layout: `src/edu/boun/edgecloudsim/...`; per-app `scripts/<app>/compile.sh` → `../../bin/`
- CloudSim **7.0.0-alpha** pinned in `lib/` — do not upgrade blindly
- Extension point: `ScenarioFactory` → swap `LoadGeneratorModel`, `EdgeOrchestrator`, `NetworkModel`, `MobileDeviceManager`
- `sample_app4` needs `jFuzzyLogic`; `sample_app5` needs `weka.jar`

---

## 12. PureEdgeSim port sketch

### Phase 1 — Deterministic workload + snapshot
- `TraceTaskGenerator`: fixed (time, device, app) trace; seeded RNG
- Warmup run → `StateSnapshot`: per-node `tasksQueue`, `availableCores`, `availableRam/Storage`, network link utilization, `historyMap`, device positions

### Phase 2 — Counterfactual replay
- `CosimSimulationManager`: bypass stickiness in `sendFromOrchToDestination()`; inject `forcedDestination` per task/batch
- Enumerate feasible placements → `FastReplayManager` (state clone from snapshot, replay batch, record RTT) → `placements/placements.jsonl`
- **Economics:** must hit ≪1.7 s/placement or accept orders-of-magnitude slower labeling than fork

### Phase 3 — Graph export + train
- `BipartiteGraphBuilder` (Java) or sidecar export → PyG schema compatible with `prepare_graphs_cache.py` **or** parallel train pipeline
- Feature mapping: MIPS/cores/RAM/queue depth per node vs our 14-dim platform features (no warmth, no GPU)

### Build / ops notes
- Sources under `PureEdgeSim/com/mechalikh/...`, not `src/main/java/` — `pom.xml` lacks `<sourceDirectory>`; IDE build at `build/classes/` works
- README `mvn exec:java` needs `exec-maven-plugin` (not declared in `pom.xml`)
- Runnable: `Simulation.setCustomSettingsFolder("/path/")` — **trailing slash required**
- System-scoped jFuzzyLogic JAR for Example8 only

---

## 13. PureEdgeSim: orchestration, stickiness, GNN, benchmarks

### 13.1 What “orchestration” means in PureEdgeSim

- **Orchestrator** (`DefaultOrchestrator`) selects an offloading destination from architecture-filtered candidates (`CLOUD_ONLY` … `ALL`).
- **TRADE_OFF** and **ROUND_ROBIN** are the only built-in algorithms; custom orchestrators via `setCustomEdgeOrchestrator()` (see Example8 fuzzy logic).
- **No autoscaler / scale-to-zero / replica pool** — containers optionally pulled from registry (`enable_registry`); RAM/storage reserved per task.
- **`enable_orchestrators=false`** (default): each device self-orchestrates locally.
- PureEdgeSim does **not** ship serverless policies; comparison is orchestration algorithm × architecture sweep.

### 13.2 GNN influence vs edge dynamics (what each controls)

```text
End-to-end success ≈ network path + WAN contention + queue wait + fixed CPU + latency budget
                              ↑                    ↑              ↑
                         topology/mobility    orchestrator    orchestrator
                         (first task only)    (indirect)      (indirect via load balance)
```

| Training mode | Edge dynamics in labels? | GNN/orchestrator role |
|---|---|---|
| **Default sticky + live sim** | Full (mobility, WAN, battery) | ~100 first-task decisions; rest are replay |
| **Sticky disabled, per-task orchestrate** | Full | Per-task offload — closer to Garmendia paper |
| **Frozen-state co-sim (HeroSim-style)** | Off | Counterfactual placement sweep — **not built** |
| **First-task-only co-sim** | Partial | Device→DC binding — different RQ than task→replica |

**Garmendia et al. (CMES 2024):** infrastructure GNN + DQN; “offloading rating” edge feature; labels from TRADE_OFF/RR runs; optimizes **success rate** under latency budgets — not batch joint RTT minimization under warmth contention (`contention_v2`). Code not in repo.

### 13.3 Orchestrator benchmark (2026-06-16, `/root/projects/PureEdgeSim`)

#### Setup A — full default sweep (42 scenarios, charts on)
- 7 architectures × 2 algorithms × 100 edge devices
- 10 min sim, ~11,100 tasks/scenario, `wait_for_all_tasks=true`
- Wall: **~41 s** total → **~0.98 s/scenario** (prior run with charts reported ~3.1 s/scenario)

#### Setup B — fresh headless run (2026-06-16 afternoon)
- `ALL` architecture, 100 devices, TRADE_OFF vs ROUND_ROBIN
- `display_real_time_charts=false`, `save_charts=false`, `pause_length=0`
- Wall: **3.46 s** for 2 scenarios → **~1.7 s/scenario**

| Policy | Tasks succeeded | Success rate | Avg execution delay | Avg wait |
|---|---|---|---|---|
| **TRADE_OFF** | 6,825 / 11,100 | **61.5%** | 0.187 s | 0.0001 s |
| **ROUND_ROBIN** | 5,756 / 11,100 | **51.9%** | 0.191 s | 0.000 s |
| **Δ** | +1,069 tasks | **+9.6 pp** | **+2.0%** | ~0 | |

#### Setup A — policy deltas across architectures (full 42-row CSV)

| Architecture | Success Δ (TRADE_OFF − RR) | Avg delay Δ (RR vs TO) |
|---|---|---|
| CLOUD_ONLY | 0 pp (both ~11%) | 0% — WAN-saturated, policy irrelevant |
| EDGE_ONLY | +0.5 pp | RR **−7.7%** (RR wins on delay) |
| MIST_ONLY | +4.4 pp | TRADE_OFF **−22%** |
| MIST_AND_CLOUD | **−8.0 pp** (RR wins) | +3.3% |
| EDGE_AND_CLOUD | +2.7 pp | 0% |
| MIST_AND_EDGE | +1.2 pp | TRADE_OFF **−9.2%** |
| ALL | +2.6 pp (prior) / +9.6 pp (fresh) | TRADE_OFF **−5.8%** (prior) |

**Interpretation:**
- Under WAN-heavy `CLOUD_ONLY`, orchestrator choice is **irrelevant** (~89% fail on latency regardless) — physics dominates, analogous to NoServer with autoscale on.
- Under `ALL`, policy shifts **success rate** substantially (+9.6 pp) but **mean delay** only ~2% — failures dominate the metric mix and sticky placement dilutes per-decision impact to ~100 decisions per 11k tasks.
- Unlike NoServer's ~0.2–0.5% mean latency spread with Knative AS on, PureEdgeSim shows **large success-rate headroom** — but success ≠ our batch RTT objective and stickiness blocks per-task GNN leverage.

### 13.4 When GNN-at-orchestrator matters more

| Scenario | Expected lift vs TRADE_OFF |
|---|---|
| Default sticky, 11k tasks, 100 devices | **Low per-task** — one decision amortized over ~111 tasks/device |
| Sticky **disabled**, per-task orchestrate | **Moderate–high** on success rate under tight latency budgets |
| `CLOUD_ONLY` / WAN-saturated | **~0%** — physics dominates |
| Mobility + energy multi-objective (Garmendia setting) | **Moderate** — infra graph over nodes + links is natural |
| **Joint batch 2–4 tasks** (`contention_v2` RQ) | **Blocked** by sticky semantics unless redesigned |
| Frozen-state oracle labels for near-RTT regret | Requires **new co-sim layer** — not drop-in |
| Heterogeneous CPU/GPU replica warmth | **N/A** — PureEdgeSim has tier types, not our platform model |

### 13.5 Recommendation

| Approach | Verdict |
|---|---|
| Stay on HeroSim fork for GNN placement + `placements.jsonl` oracle | **Yes — still best** for current RQs |
| Use PureEdgeSim as drop-in label factory | **No** — wrong abstraction, no replay, ~1000× slower |
| Port bipartite co-sim to PureEdgeSim | **Only if RQ shifts** to mobility/energy/multi-tier edge continuum |
| GNN replacing TRADE_OFF live (sticky on) | Low per-task leverage; moderate on **success rate** under `ALL` |
| GNN per-task with stickiness off | Feasible (Garmendia path); still need custom train/serve + label pipeline |
| Joint batch placement (`contention_v2`) | **Incompatible** without replacing sticky model entirely |
| Use PureEdgeSim for serverless/Knative studies | **No** — use NoServer or HeroSim upstream |

**When PureEdgeSim would be the right base:**
- RQs explicitly about **mist/edge/cloud tier selection** under mobility, battery, WAN sharing, energy
- Labels optimize **deadline satisfaction rate** or energy, not batch RTT under replica warmth
- Willing to invest weeks in Phase 1–3 port (§12) — estimate **~74 hours** for naive 160k-combo sweep vs **~27 min** on HeroSim fork

### 13.6 Key source files (PureEdgeSim v5.3.0)

| Role | Path |
|---|---|
| Entry / launcher | `PureEdgeSim/com/mechalikh/pureedgesim/MainApplication.java`, `simulationmanager/Simulation.java` |
| DES loop + sticky gate | `simulationmanager/DefaultSimulationManager.java` |
| Orchestrator | `taskorchestrator/DefaultOrchestrator.java`, `taskorchestrator/Orchestrator.java` |
| Task execution / queues | `datacentersmanager/DefaultComputingNode.java` |
| Sticky state | `datacentersmanager/LocationAwareNode.java` |
| Task generation | `taskgenerator/DefaultTaskGenerator.java` |
| Delay model | `taskgenerator/TaskAbstract.java` |
| Infrastructure graph | `network/InfrastructureGraph.java` |
| Metrics / CSV | `simulationmanager/SimLog.java` |
| Settings | `PureEdgeSim/settings/simulation_parameters.properties`, `applications.xml` |

---

## 14. EdgeCloudSim: VM fit, per-task routing, GNN, benchmarks

### 14.1 What “orchestration” means in EdgeCloudSim

- **EdgeOrchestrator** (`EdgeOrchestrator` / `BasicEdgeOrchestrator` / per-app subclasses) makes **two decisions per task**: tier (`getDeviceToOffload`) then VM (`getVmToOffload`).
- Built-in VM policies: **FIRST_FIT, NEXT_FIT, BEST_FIT, WORST_FIT, RANDOM_FIT** on residual CPU % (`CpuUtilizationModel_Custom`).
- **No autoscaler / scale-to-zero / sticky sessions** — static VM pool for entire sim.
- **No cold start** — VMs exist from `SimManager.startEntity()`; tasks bind immediately when capacity allows.
- **sample_app4:** fuzzy-logic orchestrator (`FuzzyEdgeOrchestrator`). **sample_app5:** Weka ML + MAB + game theory (`VehicularEdgeOrchestrator`) — **no GNN/DQN in repo**.

### 14.2 GNN influence vs edge dynamics

```text
serviceTime ≈ WLAN/WAN upload + VM queue/exec + download
                    ↑              ↑                ↑
              contention      VM fit policy    mobility can fail
              (empirical)     (per-task)       mid-download
```

| Training mode | Edge dynamics in labels? | GNN/orchestrator role |
|---|---|---|
| **Live sim, per-task VM fit** | Full (mobility, WLAN contention) | **~15k decisions/scenario** — unlike PureEdgeSim sticky |
| **Frozen-state co-sim (HeroSim-style)** | Off | Counterfactual placement sweep — **not built** |
| **Edge-only tutorial1 (DEFAULT_SCENARIO)** | Partial (no cloud tier) | VM fit among 112 edge VMs globally |

### 14.3 VM-fit benchmark (2026-06-16, `/root/projects/EdgeCloudSim`)

**Setup:** tutorial1, 200 devices, 10 min sim, 2 min warm-up, `DEFAULT_SCENARIO` (edge-only), `vm_load_check_interval=0.1`, Java 21, config at `/tmp/edgecloudsim_bench/config/bench2_config.properties`.

| Policy | Completed | Failed | Fail % | Avg service time |
|---|---|---|---|---|
| **WORST_FIT** | 15,016 | 43 | 0.29% | 1.327 s |
| **RANDOM_FIT** | 15,050 | 22 | 0.15% | 1.333 s |
| **Δ** | — | RANDOM **−21** failures | **−0.14 pp** | RANDOM **+0.4%** slower |

**Wall:** **3.54 s** for 2 scenarios → **~1.77 s/scenario**, **~15,060 tasks/scenario**.

**Caveats:**
- Task lists differ between runs (unseeded `SimUtils.RNG`) — not perfectly controlled A/B.
- `DEFAULT_SCENARIO` forces edge-only; `prob_cloud_selection` unused.
- Failures are mobility/bandwidth, not deadline violations (`delay_sensitivity=0`).
- tutorial1 WORST_FIT scans all 112 VMs globally — closer to global LB than location-aware placement.

**Interpretation:** Under edge-only high-completion defaults, VM fit policy has **modest** impact — analogous to NoServer sub-1% LB spread, but different failure modes. Per-task routing means GNN has **full decision leverage** (unlike PureEdgeSim sticky), but warmth/contention physics still block our `contention_v2` RQ.

### 14.4 EdgeCloudSim vs PureEdgeSim (for our RQ)

| Dimension | EdgeCloudSim | PureEdgeSim |
|---|---|---|
| DES engine | CloudSim (heavier) | Custom (lighter) |
| Per-task routing | **Yes** | No (sticky) |
| Tiers | Edge + cloud (+ mobile optional) | Mist + edge + cloud |
| Energy | No | Yes |
| Orchestrator algos | VM fit heuristics + fuzzy/ML samples | TRADE_OFF / ROUND_ROBIN |
| Co-sim speed | ~1.8 s/scenario | ~1.7 s/scenario |
| GNN/co-sim port | Per-task helps; wrong warmth physics | Stickiness blocks; mist/energy native |

### 14.5 Recommendation

| Approach | Verdict |
|---|---|
| Stay on HeroSim fork for current RQs | **Yes** |
| Use EdgeCloudSim as drop-in label factory | **No** — no replay, wrong physics, ~1000× slower per placement |
| Prefer EdgeCloudSim over PureEdgeSim for co-sim port | **Marginally** — per-task routing aligned; still weeks of engineering |
| GNN replacing WORST_FIT live (tutorial1) | Low ceiling (~0.14 pp failure, ~0.4% latency) |
| Joint batch `contention_v2` | **Incompatible** without new orchestrator + warmth model |
| VM fit / mobility studies (classic EdgeCloudSim papers) | **Yes** — native use case |

### 14.6 Key source files (EdgeCloudSim, Oct 2025)

| Role | Path |
|---|---|
| DES coordinator | `src/edu/boun/edgecloudsim/core/SimManager.java` |
| Config loader | `src/edu/boun/edgecloudsim/core/SimSettings.java` |
| Extension factory | `src/edu/boun/edgecloudsim/core/ScenarioFactory.java` |
| Orchestrator API | `src/edu/boun/edgecloudsim/edge_orchestrator/EdgeOrchestrator.java` |
| Default VM policies | `src/edu/boun/edgecloudsim/edge_orchestrator/BasicEdgeOrchestrator.java` |
| Tutorial1 orchestrator (global VM scan) | `src/edu/boun/edgecloudsim/applications/tutorial1/SampleEdgeOrchestrator.java` |
| Task pipeline | `src/edu/boun/edgecloudsim/edge_client/DefaultMobileDeviceManager.java` |
| Workload | `src/edu/boun/edgecloudsim/task_generator/IdleActiveLoadGenerator.java` |
| Network | `src/edu/boun/edgecloudsim/network/MM1Queue.java` |
| Metrics | `src/edu/boun/edgecloudsim/utils/SimLogger.java` |
| ML orchestrator | `src/edu/boun/edgecloudsim/applications/sample_app5/VehicularEdgeOrchestrator.java` |
| Entry | `src/edu/boun/edgecloudsim/applications/tutorial1/MainApp.java` |

---

## 15. NoServer: LB, Knative dynamics, GNN, benchmarks

### 15.1 What “Knative” means in NoServer

- **Autoscaler** implements Knative KPA-style panic/stable windows, scale-to-zero, cold start on zero replicas.
- **LB `least_loaded`** ≈ HeroSim Knative scheduler (shortest queue / least busy node).
- NoServer does **not** ship policies named `knative` / `round_robin` / `random` for LB; defaults are `first_available` and `least_loaded`.

### 15.2 GNN influence vs Knative (what each controls)

```text
Total latency ≈ DAG wait + cold start + autoscale lag + queue wait + fixed duration
                     ↑              ↑                ↑              ↑
                  not LB         not LB           not LB         LB can shift
```

| Training mode | Knative dynamics in labels? | GNN role |
|---|---|---|
| **LB-only co-sim (recommended)** | **Off** (freeze AS after warmup) | Which instance; like fork + fixed `replica_plan` |
| **Live LB-only GNN** | **On** | Small routing refinement; AS can dominate |
| **LB + autoscale GNN** | On | Harder; non-stationary action space |

### 15.3 LB benchmark (2026-06-16, `/tmp/noserver`)

Setup: `scripts/benchmark_lb.py`; venv + stub HarvestVM pickles; HVM OFF; seed 42. Script scenarios: `medium`, `saturated`, `saturated_frozen`, `extreme` (see `/tmp/noserver/scripts/benchmark_lb.py`). Log: `/tmp/noserver_lb_highload.log`.

#### 15.3.1 Medium load (autoscaler ON)

16 RPS, 1 function, 2 workers × 16 cores, 60s sim.

| Policy | Success | Mean lat | p50 | p99 | Δ mean vs `least_loaded` |
|---|---|---|---|---|---|
| **least_loaded** (Knative LB) | 100% | 1143 ms | 7 ms | 7025 ms | baseline |
| first_available | 100% | 1148 ms | 7 ms | 7027 ms | **+0.5%** |
| round_robin | 100% | 1146 ms | 6 ms | 7027 ms | **+0.2%** |
| random | 100% | 1146 ms | 7 ms | 7025 ms | **+0.2%** |

Mean queue wait ~1137 ms — LB choice is a small fraction of total latency when autoscaling keeps provisioning instances.

#### 15.3.2 High load — saturated + frozen autoscale

**Goal:** isolate LB by freezing replica count after warmup (HeroSim co-sim analogue).

| Scenario | RPS | Sim | Workers | AS | Instances | Mean lat | p99 | Queue | Δ mean (all policies) |
|---|---|---|---|---|---|---|---|---|---|
| **saturated_frozen** | 40 | 15s warmup @ 10 RPS + 30s @ 40 RPS | 1 × 8 cores | **frozen** after warmup | 50 | **12.3 s** | 29.0 s | 12.3 s | **0.0%** |
| **saturated** | 36 | 30s | 1 × 8 cores | ON | 103 | **17.4 s** | 29.5 s | 17.4 s | **0.0%** |

All four policies (`least_loaded`, `first_available`, `round_robin`, `random`): **100% success** in both scenarios; latencies identical to rounding (e.g. saturated_frozen 12282 vs 12281 ms; saturated 17384 vs 17385 ms).

**Why higher load did not spread policies:**
- Bottleneck is **ingress/throttler queue** + **8 cores on one worker**, not replica pick (mean queue ≈ mean latency).
- Instance queue depth = **1** (AWS model); when saturated every policy falls through to “first `reserve()` that succeeds.”
- `least_loaded` has no idle instance → degrades to same scan as `first_available` (`loadbalance.py` fallback).
- Single worker → no node-level differentiation for `least_loaded`.
- Frozen AS still leaves **50 replicas on 8 cores** — core scheduling dominates.

**Benchmark notes:** NoServer ticks at 1 ms; high-load runs are **~4–12 min wall time per policy**. Early script had an unbounded post-arrival loop (fixed); use bounded `hard_limit` drain.

**Interpretation:**
- GNN replacing LB only: expect **~0% mean latency** even under **saturated, frozen-replica** regimes in this topology — contradicts earlier “moderate lift” hypothesis (§15.4).
- **Batch joint LB** (like fork batch-4) remains the main path to larger headroom on NoServer — same lesson as HeroSim `contention_v2`.
- vs `first_available`: no meaningful baseline separation at any tested load.

### 15.4 When GNN-at-LB matters more

| Scenario | Expected lift vs `least_loaded` |
|---|---|
| Frozen instances, saturated RPS (1 worker, depth-1 queues) | **~0%** (tested §15.3.2) |
| Batch 2–4 requests, joint assignment | **Higher** (mirrors fork; not yet benchmarked on NoServer) |
| Autoscale on, Knative provisioning | **~0%** (§15.3.1 + §15.3.2) |
| Multi-worker + heterogeneous cores + fixed replicas | Untested; last plausible LB-only regime |
| HarvestVM on | Noisy; bad for deterministic co-sim |

### 15.5 Recommendation

| Approach | Verdict |
|---|---|
| Port full HeroSim bipartite co-sim to NoServer | **No** — wrong abstraction |
| **Batch LB co-sim** on NoServer (`test`/`rps`, frozen AS) | **Yes** — closest viable port |
| GNN only on LB, live Knative on | **~0%** ceiling (§15.3) |
| GNN on LB + frozen AS (train and deploy) | **~0%** on tested topology; only viable if **batch** or **multi-worker** co-sim added |
| Stay on HeroSim fork for edge placement GNN | **Still best** for current RQs |

---

## 16. Five-way decision guide

| Goal | Choose |
|---|---|
| GNN placement + oracle labels (edge task→platform) | **HeroSim fork** (stay) |
| Classic serverless policy baselines, minimal machinery | **b-com/herosim** upstream |
| Per-task edge routing (no sticky) + VM fit under mobility | **EdgeCloudSim** (native); better routing semantics than PureEdgeSim for our RQ |
| Mobility / energy / mist tier + deadline satisfaction | **PureEdgeSim** + co-sim port (§12–13) — only if RQ shifts |
| Serverless AS / DAG / HarvestVM / vHive validation | **NoServer** |
| Batched LB labels in serverless setting | **NoServer** (build batch throttler + co-sim) or **HeroSim fork** |
| Joint batch placement under contention | **HeroSim fork** (`contention_v2`, `--allow-non-unique-replicas`) |
| Per-task orchestration under mobility (no warmth RQ) | **EdgeCloudSim** (native) or PureEdgeSim (disable stickiness) |
| Drop-in `placements.jsonl` factory | **HeroSim fork only** — EdgeCloudSim/PureEdgeSim/NoServer all need major port |

---

## 17. Related HeroSim docs

- `paper/cosimulation.md` — co-sim protocol, SSC contract
- `paper/inference.md` — live GNN batching, decode modes
- `memory/placements_jsonl_required.md` — JSONL mandatory for RTT hash
- `memory/gnn_necessity_separability.md` — when GNN beats MLP

---

## 18. Open items

- [x] NoServer LB benchmark (`least_loaded` vs `first_available` / `round_robin` / `random`) — see §15.3
- [x] PureEdgeSim orchestrator benchmark (TRADE_OFF vs ROUND_ROBIN, `ALL` arch + full sweep) — see §13.3
- [x] EdgeCloudSim VM-fit benchmark (WORST_FIT vs RANDOM_FIT, tutorial1, 200 devices) — see §14.3
- [x] Re-run NoServer benchmark with **frozen autoscale** + saturated RPS — **0% spread** (§15.3.2); ingress queue + core cap dominate
- [ ] Prototype `BatchThrottler` + LB-only co-sim JSONL on NoServer
- [ ] PureEdgeSim: fix `pom.xml` `<sourceDirectory>`; prototype `TraceTaskGenerator` + stickiness bypass
- [ ] EdgeCloudSim: seed `SimUtils.RNG` for reproducible policy A/B; prototype `TraceLoadGenerator` + `CosimEdgeOrchestrator`
- [ ] EdgeCloudSim: two-tier benchmark (`tutorial3`, cloud+edge) for fairer orchestrator comparison
- [ ] PureEdgeSim: RTT parity experiment on one `ds_*` (if port pursued)
- [ ] Document Garmendia code path if their fork is located (not in alb1183/ML-RL-PureEdgeSim; not in CharafeddineMechalikh/PureEdgeSim v5.3.0)
