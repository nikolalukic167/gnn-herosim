# Co-sim Warmth Gap — 1060 Corpus Audit & Regen Plan

**Last Updated:** 2026-06-11  
**Status:** **HISTORICAL** — documents **OLD** `gnn_datasets_4tasks_1060` corpus (1230 ds). Mechanisms below explain why pre-v2 labels lacked N× pull physics. **Active training:** warmth_v2 473–500 ds + sparse 351 ds → merged **824-graph** cache — see `docs/notes/cosim_grid_and_regen.md`. **Live gate (skew3, v2 physics):** MLP still wins **2/3** without disk feature in cache.

**Companion docs:** `docs/notes/cosim_grid_and_regen.md` · `docs/notes/warmth_model.md` · `docs/notes/storage_contention.md` · `docs/notes/compare.md` · `docs/notes/gnn_v2_sparse_topology_and_features.md`

> **One-sentence summary (1060 audit):** Zero pullTime, queueTime max 18.4s, 100% cacheHit on 1060. warmth_v2 regen + merged 824 cache partially fix labels; skew3 live MLP **2/3** until disk feature + hub co-sim grid.

---

## At a Glance

| Item | Value |
|------|-------|
| **Corpus** | `simulation_data/artifacts/run_queue_big/gnn_datasets_4tasks_1060` |
| **Datasets analyzed** | **1230** (12 missing optimal, 0 empty taskResults) |
| **pullTime > 0** | **0 / 4920 tasks (0%)** |
| **cacheHit=True** | **4920 / 4920 (100%)** |
| **queueTime max** | **18.37s** (never ≥30s; **0** datasets with max queue >60s) |
| **coldStartTime > 0** | **3 / 4920 (0.06%)** |
| **initialized_snapshot cold** | **~70.9%** platforms at Phase 1 schedule |
| **Co-sim vs A/B physics** | **Different** — see gap table below |
| **Rerun co-sim on warmth change?** | **Yes** (mandatory for label truth) |
| **Code shipped (2026-06-11)?** | **Yes** — `warmth.py`, v2 default in `prepare_simulation_config`, pilot audit PASS |
| **Full warmth_v2 regen?** | **Partial** — warmth **473–500** ds · sparse **351/351** · merged cache **824 graphs** · see `cosim_grid_and_regen.md` |
| **Retrain?** | warmth dim14-ce + sparse finetune **done**; **live skew3 gate failed** (MLP 2/3) |
| **Fast-path enrichment** | **Verified** — `refresh_optimal_full_stats.py --repair --force` ≡ inline SSC on graph fields |
| **`placements.jsonl` policy** | **Mandatory** — repair/recache ≠ substitute; ~346 warmth dirs lost JSONL via `--resume` on `best.json` alone. `docs/notes/placements_jsonl_required.md` |

---

## How Co-sim Applied Warmth (1060 corpus — v1-era)

| Mechanism | 1060 co-sim config | Effect on labels |
|-----------|-------------------|------------------|
| Autoscaler | **determined** | Per-platform `previous_task` pull gate; **no `has_function`** |
| `defer_cold_replica_init` | **Missing / False** | `initialized.succeed()` at replica create **without pull** |
| `preinitialize_platforms` | **True** | Replicas spread; forced placement requires initialized plat |
| `fast_forward_warmup` | **True** (1230/1230) | `pullTime=0` always; use queueTime for wait |
| Optimal placement | Brute-force spread | 32% datasets co-locate 2+ tasks on same node; max co-located queue **8.2s** |
| Phase 1 SSC | `initialized_snapshot` | ~71% cold at schedule — **feature**, not label pull cost |

**Code paths:** `simulation.py:274-301` (preinit warm/cold seeding) · `determined/scheduler.py:363-374` (forced plat must be initialized unless defer) · `warmth_model.md` §7.

---

## Warmth v2 generator today (post-2026-06-11)

| Mechanism | warmth_v2 / sparse_v2 default | vs 1060 audit |
|-----------|------------------------------|---------------|
| `warmth_physics` | **`node_disk_v2`** | Disk hit skips pull on determined path |
| `defer_cold_replica_init` | **`True`** | Pull at placement; queueTime can reflect pulls |
| Grid | 500 warmth + 351 sparse | Still no `degree_skewed_core` in co-sim |
| Labels | Consolidation optima (Gate B shift) | **Still no** `node_disk_hit` in graph features |
| Live skew3 | node_disk_v2 physics in sim | MLP **2/3** — observability gap persists |

---

## 1060 Corpus — pullTime / queueTime / coldStart

### pullTime (dead field in stored labels)

| Stat | Value |
|------|-------|
| Min / max / mean | **0 / 0 / 0** |
| Nonzero tasks | **0** |
| Cause | FF warmup + pre-init `initialized.succeed()` → `cacheHit=True` → `pullTime=after−before=0` |

### queueTime (only varying wait metric in labels)

| Scope | Mean | Median | p95 | Max |
|-------|-----:|-------:|----:|----:|
| All tasks | 0.84s | 0.39s | 2.54s | **18.37s** |
| dnn1 only | 0.28s | — | 0.61s | **1.22s** |
| dnn2 only | 1.29s | — | 3.51s | **18.37s** |
| rpiCpu only | 1.29s | — | 6.43s | **18.37s** |
| xavierCpu only | 0.81s | — | 2.34s | 4.11s |

**Buckets (all tasks):** `<1s`: 3631 · `[1,10)`: 1264 · `[10,30)`: 20 · `≥30s`: **0**

**Per-dataset max queueTime:** mean **1.32s** · p95 **4.64s** · max **18.37s**

### coldStartTime

| Stat | Value |
|------|-------|
| Tasks with coldStart > 0 | **3 / 4920 (0.06%)** |
| Interpretation | Sandbox warmth/preinit skips ~all 0.33s penalties in labels |

### Co-location vs contention signature

| Pattern | Count / stat |
|---------|--------------|
| Datasets with 2+ tasks on same execution node | **399 / 1230 (32.4%)** |
| Max queueTime when co-located (per node batch) | mean **1.07s** · max **8.19s** |
| Datasets with max queue > 60s (N× pull signal) | **0 / 1230** |
| Task type mix | dnn2 **2734** · dnn1 **2186** |

### 3705 corpus (reference)

| Stat | Value |
|------|-------|
| pullTime == 0 | **100%** (14820 tasks) |
| Max queueTime | **11.68s** |

---

## Three-Way Gap: A/B Scripts vs 1060 Labels vs GNN Features

| Layer | A/B (`test_cold_start_queue_last_task_ab.py`) | 1060 optimal labels | GNN inputs at schedule |
|-------|----------------------------------------------|---------------------|------------------------|
| Config | `defer_cold=True`, co-located | defer **False**, spread | Phase 1 SSC |
| queueTime scale | **31…125s** steps | **0.4–18s** max | — |
| pullTime | 0 (FF on) or =queueTime (FF off) | **always 0** | — |
| Last-task RTT (N=4 contended) | **~126s** | N/A (no such batch) | — |
| `shared_fate` / cold fraction | ~1.0 | — | **~71%** plat cold |
| Platform queue dim 7 | 0 | — | **0** typical |
| Label pull elephant (~30s) | **Yes** | **No** | Features **~0** temporal |

**Conclusion:** Simulator **can** produce N× T_pull (proven A/B); **1060 training labels mostly don't contain it.** Model learns sparse cold-density features → mostly sub-second RTT labels.

---

## Warmth Model Change — Implementation Tiers

### Tier 1 — Node disk cache on determined

**Status: SHIPPED 2026-06-11** — `needs_image_pull()` in `src/placement/warmth.py`; all Knative-family autoscalers.

Co-sim generator must also set:

| Config | Current 1060 | Required for truthful pulls |
|--------|--------------|----------------------------|
| `defer_cold_replica_init` | False/missing | **`True`** |
| Preinit `initialized.succeed()` | Immediate for cold | Only after pull or when warm |
| `fast_forward_warmup` | True | OK if labels use **queueTime** not pullTime |
| Placements | Spread optimal | Keep — co-located cases gain real pull cost |

**Expected label shift:** Same-node same-type cold → **~1× T_pull** after first plat (not N×) if disk hit works; still N× if per-platform warm only.

### Tier 2 — Separate pull vs sandbox warmth

- Pull: node disk / `has_function`
- Sandbox: keep `previous_task` match
- **Recache required** — `is_warm` edge may diverge from `is_cold` platform dim

### Tier 3 — Pull-remaining features

- `estimated_pull_remaining_sec = pending × T_pull(priors)` + autoscaler counters
- See `docs/notes/storage_contention.md` § fair feature candidates
- Recache + retrain; schedule-time fair

---

## Co-sim Rerun Decision Matrix

| Change | Rerun co-sim? | Scope |
|--------|:-------------:|-------|
| Disk cache on determined | **Yes** | Phase 2+3 brute-force; all RTT labels shift for co-located cold |
| `defer_cold_replica_init=True` in generator | **Yes** | queueTime/RTT scale changes materially |
| Fix pullTime metric only (FF off) | **Partial** | `refresh_optimal_full_stats.py` re-runs optimal sim only |
| SSC backfill (`initialized_snapshot`) | **No** | Phase 1 only — does not fix optimal RTT |
| Graph recache without new co-sim | **No value** | Same wrong labels |
| Inference-only feature tweak | **No** | Train/serve mismatch |

**Not sufficient alone:** `backfill_initialized_snapshot.py` — Phase 1 SSC only.

**Placement optima may shift** under new pull physics → full brute-force regen safer than optimal-only refresh.

---

## GNN Retrain Implications

| Component | Today (1060 + dim14-ce) | After warmth fix + regen |
|-----------|-------------------------|--------------------------|
| RTT labels | Mostly <2s queue + exec | Fat tail 31s+ on co-located cold |
| `shared_fate` (dim 8) | Saturates ~1.0 | Still saturates — **seconds scale gap remains** |
| `queue` (dim 7) | Often 0 at schedule | Unchanged blind spot |
| `is_warm` edge | Sandbox predicate | Diverges if Tier 2 |
| Train/serve | determined physics | Knative deploy needs same disk fix for parity |

**Do not finetune** old dim14-ce on new labels — label scale shift. Full recache `graphs_cache_gnn_datasets_4tasks_1060` + retrain.

**Recommended bundle:** warmth Tier 1 + `defer_cold=True` in generator + optional `estimated_pull_remaining_sec` + pilot 50–100 ds → full 1060 regen → recache → retrain.

---

## Pragmatic Rollout

1. ~~Patch autoscalers + `warmth.py`~~ **Done** (2026-06-11)
2. ~~Co-sim defaults v2 + defer_cold~~ **Done** in `prepare_simulation_config`
3. ~~Pilot audit~~ **PASS** (`pilot_warmth_regen_audit.py`)
4. ~~Parallel regen~~ **Partial** — warmth 473–500 · sparse **351/351** · merged cache **824**
5. ~~Recache + finetune~~ **Done** — sparse finetune val acc **25.6%**; warmth dim14-ce anchor
6. **[DONE]** B1: `node_disk_hit` + seq `src_norm` + seq `is_warm` — code shipped 2026-06-11  
7. **[NEXT]** `--repair --force` disk backfill → recache → **`skew_warmth_v2` co-sim grid**
7. **[NEXT]** Live gate: standard5 + skew3 with warmth dim14-full (not sparse finetune alone)

**Grid note:** Merged **824** adds conn {0.25,0.30,0.35} but **not** hub topology — see `docs/notes/gnn_v2_sparse_topology_and_features.md`.

---

## Reproduce Audit

```bash
# Corpus path
CORPUS=simulation_data/artifacts/run_queue_big/gnn_datasets_4tasks_1060

# Quick checks (example — full audit was inline Python 2026-06-11):
# - optimal_result.json → stats.taskResults → pullTime, queueTime, cacheHit, coldStartTime
# - system_state_captured_unique.json → initialized_snapshot cold fraction
```

**A/B reference (contention physics):**

```bash
pipenv run python3 scripts_cosim/test_cold_start_queue_last_task_ab.py
pipenv run python3 scripts_cosim/test_memory_contention_ab.py
```

---

## Key Files

| Path | Role |
|------|------|
| `docs/notes/warmth_model.md` | Warm predicate, disk cache gap, policy diff |
| `docs/notes/storage_contention.md` | FilterStore N×, metric traps, feature candidates |
| `scripts_cosim/test_cold_start_queue_last_task_ab.py` | A/B with defer_cold=True, counterfactuals |
| `scripts_cosim/generate_gnn_datasets_fast.py` | Co-sim generator (FF default True) |
| `scripts_cosim/backfill_initialized_snapshot.py` | Phase 1 SSC only — **not** label regen |
| `scripts_cosim/refresh_optimal_full_stats.py` | Re-run optimal placement sim |
| `src/policy/determined/autoscaler.py` | Pull gate (patch target) |
| `src/placement/simulation.py` | Preinit / defer / initialized.succeed() |
| `simulation_data/artifacts/run_queue_big/gnn_datasets_4tasks_1060/` | Current training corpus |

---

## Changelog

| Date | Change |
|------|--------|
| 2026-06-11 | **Historical banner** — warmth_v2 partial regen + merged 824 cache + skew3 live gate; Tier 1 marked shipped; rollout steps 6–7 pending features/hub grid |
| 2026-06-11 | Parallel fast-path regen started; enrichment A/B PASS; grid comparison doc added (`cosim_grid_and_regen.md`) |
| 2026-06-11 | Implementation landed — pilot audit PASS; P0 autoscaler indent + P1 path fixes; full regen pipeline scripted |
| 2026-06-11 | Initial audit — 1060 pullTime/queueTime/coldStart scan, co-sim vs A/B gap, warmth regen plan, GNN retrain implications |
