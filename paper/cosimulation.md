# Co-Simulation: Data Generation Engine

> How training data is produced. Update when dataset generation, warmth physics, or SSC contract changes.

---

## What Co-Simulation Is

Co-simulation is a **snapshot-based brute-force oracle**: for a frozen system state (queues, replicas, temporal state) and a small incoming task batch (1–4 tasks), the simulator runs once per candidate placement and records total batch RTT for each. This is not multi-tool co-simulation (not Simulink coupling). It means: freeze state → enumerate all placements → label.

It is fundamentally different from real simulation:

| Aspect | Co-simulation | Real simulation |
|---|---|---|
| Entry point | `src/executecosimulation.py` | `src/executesimulation.py` |
| Workload | 1–4 tasks, fixed batch | Full trace (e.g., 100 RPS × 100s) |
| Scheduler | `determined` (forced placement) | `gnn`, `knative_network`, etc. |
| Replicas | Pre-created via `replica_plan` + warmup | Autoscaling from zero |
| Purpose | Build supervised labels | Evaluate policy end-to-end |

---

## Three-Phase Protocol
    
### Phase 1 — State Capture
Run a single task through the `determined` scheduler with a `replica_plan` to build up warmup state. After warmup, capture:
- Full queue snapshot (208 platforms: `node_name:platform_id → queue_depth`)
- Temporal state (per-platform: `current_task_remaining`, `cold_start_remaining`, `comm_remaining`)
- `initialized_snapshot` (per-node cold replica density → `shared_fate_signal`)

Written to: `system_state_captured_unique.json` (SSC)

### Phase 2 — Brute-Force Enumeration
Generate all feasible placement combinations (Cartesian product of per-task feasible replicas). Run each combination in `ProcessPoolExecutor` in parallel. Record `{placement_plan, rtt}` to `placements/placements.jsonl`.

**This file is mandatory, not optional disk.** It is the sole source for `rtt_chunk_*.pkl` / near-RTT counterfactual lookup at train time. `refresh_optimal_full_stats.py --repair` replays only the optimal placement and **does not** create or replace JSONL. Never delete `.bf_scratch` until JSONL is copied to `placements/`. Never `--resume` on `best.json` alone. See `docs/notes/placements_jsonl_required.md`.

### Phase 3 — Artifacts
- `best.json` — `{file, rtt}` for minimum-RTT placement
- `optimal_result.json` — full sim result for best placement
- `system_state_captured_unique.json` — scheduling-time state (SSC)

---

## Dataset Structure

```
ds_XXXXX/
├── space_with_network.json          # scenario parameters
├── infrastructure.json              # generated cluster topology
├── workload.json                    # task batch
├── placements/placements.jsonl      # all (placement_plan, rtt) pairs
├── best.json
├── optimal_result.json
├── system_state_captured_unique.json
└── placement_metadata.json
```

Example `placements.jsonl` line:
```json
{"placement_plan": {"0": [33, 170]}, "rtt": 1.2689091609396324}
```

---

## Parameter Space

Grid sweeps via `scripts_cosim/generate_gnn_datasets_fast.py` (LHS also in `src/sample.py` for legacy corpus):

| Dimension | warmth_v2 (500 ds) | sparse_v2 (351 ds) | Legacy 1060 |
|---|---|---|---|
| Connection probability | **0.50 only** | **{0.25, 0.30, 0.35}** | 0.25–0.80 |
| Replicas + preinit | 5 rows | **(1,1), (1,2), (2,2)** | 6 rows |
| Hub topology | **None (ER only)** | **None** | None |
| Queue regimes | 5 (pois16+) | 3 | 10 |
| Warmth physics | **`node_disk_v2`** + defer_cold | same | v1-era labels |
| Batch size | 1–4 tasks | 1–4 | 1–4 |

**Not in any current train grid:** `degree_skewed_core`, asymmetric 5/30ms latency (present in live bipartite eval only).

**Historical label-generation issue:** Default generation omits
`--allow-non-unique-replicas` (`action='store_true'` → off), so the original warmth/sparse training
labels enumerated distinct-platform placements only and were separable by construction.

**Final non-unique audit (2026-08-04):** after strict integrity filtering, retained sweeps are warmth
**487**, sparse **351**, contention_v2 **899**, contention_v3 **900**. Optimum collision rates are
**39.8% / 33.0% / 40.3% / 36.3%** respectively; coupled fractions at >1% marginal-greedy regret are
**9.7% / 10.3% / 7.1% / 4.2%**. The stale 0%/0% claim applies only to the historical
unique-replica pass. Sources: `simulation_data/separability_audit_4corpus_20260804.json` and
`simulation_data/placement_integrity_manifest.json`.

**Training-label caveat (2026-08-04):** graph-cache labels are currently read from
`optimal_result.sample.placement_plan`, not recomputed from the final sweep minimum. After
non-unique backfill, the retained label plans are nonoptimal for **210/487 warmth** and
**166/351 sparse** datasets; contention_v2 has **18 nonoptimal + 26 absent** label plans.
The strict 899-graph contention cache therefore does not yet represent a clean final-label
training corpus. Placement integrity and label integrity are separate gates.

---

## Active Training Corpus (inventory audited 2026-08-04)

| Item | Value |
|------|-------|
| warmth_v2 co-sim | `simulation_data/gnn_datasets_4tasks_1060_warmth_v2` — **487 retained jsonl**; 13 explicit exclusions |
| sparse co-sim | `simulation_data/gnn_datasets_4tasks_sparse_warmth_v2` — **351/351** ds |
| contention_v2 | `simulation_data/gnn_datasets_4tasks_contention_v2` — **899 retained jsonl** · strict 899 cache built, but ablation blocked by **26 absent labels** · deployed models historical |
| contention_v3 | `simulation_data/gnn_datasets_4tasks_contention_v3` — **900/900** jsonl on mitrix · train/cache/models **datalab-only** · **REJECT** for deploy |
| **Strategic merge cache** | `graphs_cache_strategic_merge_wss_cont_v2` — **3729 graphs** · live gates closed — **reject** vs contention-only |
| **Weighted merge cache** | `graphs_cache_warmth_sparse_contention_v2_weighted` — **2778 graphs** · live gates closed — **reject** |
| **Deploy ★** | `near-rtt-v2-contention-v2-dim14-ce-only.pt` + `batch_edge_mlp_contention_v2_dim22_batchcache.pt` |
| **Merged graph cache (legacy)** | `graphs_cache_warmth_v2_sparse_merged` — **824 graphs**, **6.06M** RTT rows |

---

## Historical Dataset (1060 — eval anchor cache)

- **Path:** `artifacts/run_queue_big/gnn_datasets_4tasks_1060`
- **Count:** 1,230 datasets
- **Graph cache:** `graphs_cache_gnn_datasets_4tasks_1060`
- **Role:** Checkpoint `near-rtt-v2-dim14-ce-only.pt` — bipartite/skew/triangle sweeps to date
- **Label caveat:** pre-v2 pull physics; pullTime **0%** in stored labels — see `docs/notes/cosim_warmth_gap.md` (historical)

---

## SSC Contract (Critical)

The `system_state_captured_unique.json` is the source of truth for scheduling-time state. It must contain:

- `replicas` — active replica set
- `available_resources` — per-platform resource state
- `scheduler_state` — top-level scheduling metadata
- `task_placements[0]` — fallback SSC (GNN-scheduler SSCs)
- `initialized_snapshot` — top-level key: `{node_name:platform_id → {initialized: bool}}`

`prepare_graphs_cache.py` hard-fails on missing or incomplete SSC (strict mode since 2026-06-07).

### Historical SSC Bugs (fixed)

1. **`slim_completed_task` wipe** (fixed 2026-06-06): after-completion task slimming cleared `full_queue_snapshot` and `temporal_state_at_scheduling` before stats export — all SSCs were null. Fixed: conditional slimming preserves scheduling-time snapshots.

2. **`initialized_snapshot` not forwarded** (fixed 2026-06-08): `load_extended_state_data` read `initialized_snapshot` but did not forward it into the dataset dict — all cached graphs had `shared_fate_signal=0`. Fixed in `prepare_graphs_cache.py`.

3. **`is_warm` edge attribute** — hom `prepare_graphs_cache.py`: fixed 2026-06-08 via `previous_task_type_name`. **Seq/atomic21 recache** (`prepare_graphs_cache_seq.py`) still sets `is_warm` from replica flags (degenerate = 1 on all feasible edges) — train/serve mismatch on warmth_v2 pipeline; fix pending before next retrain.

---

## Train/Serve Gap (Known Issue)

**Metric:** `stats.averageQueueTime` (mean queue wait, seconds) from co-sim `optimal_result.json` vs live result JSON. Separate axes: per-task `queueTime` max; `full_queue_snapshot` depth (tasks). **Source:** `scripts_cosim/verify_queue_gap_1060.py` (2026-06-10); legacy 3705 via `compare_sim_vs_cosim.py`.

| Metric | Co-sim warmth_v2 (partial) | Co-sim 1060 (historical) | Live default (dim14-ce) |
|---|---|---|---|
| max avgQueueTime | TBD post-full-500 | **11.54s** | **20.6s** |
| Warmth physics in labels | **v2 disk consolidation** | v1-era spread optima | node_disk_v2 in sim |
| Hub topology in train grid | **No** | No | degree_skew in eval only |

The co-sim oracle sees controlled prewarm queue distributions; live simulation starts cold and autoscales under sustained load. **Do not cite ~200× on mean queue wait** — that ratio applies to decode-time **qvm p95** (ranking ~2857 vs CE ~408), not avgQueueTime. This gap is the primary reason offline metrics don't reliably predict live RTT (see `rules.md` section 2.4).

---

## Fast-Forward Warmup

Warmup is accelerated by skipping per-task execution in `Platform.platform_process()` via aggregate timeout. This is ~23× faster than real warmup for a 4-task benchmark (ds_03001).

**Parity guarantee:** fast-forward warmup produces identical RTT and avgQueueTime labels to no-fast-forward. Verified: 2-case A/B (`ff_ab_test_fast_truthful_2cases_v3`); **1060 re-check 2026-06-10** on high-queue ds (`ds_00479`, `ds_00319`, `ds_00016`) — RTT/avgQT exact match, snapshot depth ±2–3 tasks. Warmup parity fix (2026-01-25): `infrastructure.py` honors existing `previous_task` warm state; removed forced first-task cold-start override.
