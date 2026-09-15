# Co-sim Grid & Warmth Regen — Session Takeaways (2026-06-11)

**Last Updated:** 2026-06-11 (v0.28.1)  
**Companion docs:** `docs/notes/placements_jsonl_required.md` · `docs/notes/cosim_warmth_gap.md` · `docs/notes/warmth_model.md` · `docs/notes/compare.md` · `LINEAGES.md`

> **One-sentence summary:** The warmth_v2 regen uses a **compressed 500-dataset grid** (not 1500), runs via **4 parallel shards** with **GNN_CAPTURE=0 + print suppression**, and **post-hoc SSC enrichment** (`refresh_optimal_full_stats.py --repair --force`) verified equivalent to inline capture on graph-relevant fields.

> **CRITICAL — `placements/placements.jsonl`:** Mandatory for every dataset. Full `(placement_plan, rtt)` sweep → `rtt_chunk_*.pkl` / near-RTT training. **`--repair` + recache does NOT replace JSONL.** Fast path wrongly treated sweep as optional; `--resume` on `best.json` alone left ~346 warmth dirs without JSONL. Policy: `docs/notes/placements_jsonl_required.md`.

---

## Session Outcomes (2026-06-11)

### P0/P1 fixes (pre-regen)

| Issue | Fix |
|-------|-----|
| **P0:** 12/14 Knative-family autoscalers — `retrieval_size` / pull I/O outside inner `needs_image_pull(..., active_storage=)` block | Indentation corrected to match `determined/autoscaler.py` |
| **P1:** `run_warmth_full_regen_recache.sh` recache path pointed at `artifacts/run_queue_big/…` | Fixed to `simulation_data/gnn_datasets_4tasks_1060_warmth_v2` |
| **P1:** `train_near_rtt_v2_dim14_ce_only.py` hardcoded old cache dir | `NEAR_RTT_CACHE_DIR` env + export in regen script |
| **CRLF** in shell script broke `nohup` | `sed -i 's/\r$//'` on launcher scripts |

### Gates verified (session)

- `tests/test_warmth.py` — 15/15 PASS  
- `pilot_warmth_regen_audit.py` — v1 125.57s → v2 31.65s (N=4 contended)  
- `test_cold_start_queue_last_task_ab.py` — N× penalty, warmth matrix  
- `test_ssc_phase_alignment.py` — Phase 1 / Phase 3 SSC alignment  

### Single-process regen aborted

- Original `run_warmth_full_regen_recache.sh` nohup run stopped at **~235/500** datasets  
- Log `logs/warmth_full_regen_20260611_004930.log` grew to **~14GB** (worker stdout not suppressed)  
- **~127s/dataset** average; labels still **cacheHit 100%**, **pullTime 0** (optimizer picks warm/spread placements; pull at init outside label window)

### Fast-path regen (replaced single-process)

| Setting | Value |
|---------|-------|
| Launcher | `scripts_cosim/run_warmth_parallel_regen.sh` |
| Shards | 4 non-overlapping ranges, **125 ds/shard**, **7 workers/shard** |
| Output | `simulation_data/gnn_datasets_4tasks_1060_warmth_v2` |
| `GNN_CAPTURE_DATASET_STATE` | **0** during brute-force (skip inline SSC) |
| `COSIM_SUPPRESS_SIM_PRINTS` | **1** — worker stdout/stderr → `/dev/null` in `_init_worker` |
| CLI | `--quiet` (default) |
| Resume | `--resume` skips only when **`best.json` + non-empty `placements/placements.jsonl`** (fixed 2026-06-11) |

**Enrichment A/B (ds_00059):** `refresh_optimal_full_stats.py --repair --force` vs inline SSC — **RTT identical**; graph-relevant SSC fields match (`task_placements`, `scheduler_state`, `initialized_snapshot`, replica sets); replica **list order** may differ (harmless).

**Repair does NOT write `placements/placements.jsonl`.** SSC + per-task queues on the optimal trajectory only. Counterfactual RTT requires the BF sweep file — see `docs/notes/placements_jsonl_required.md`.

**Post-regen (mandatory before recache):**

```bash
pipenv run python3 scripts_cosim/refresh_optimal_full_stats.py \
  --base-dir simulation_data/gnn_datasets_4tasks_1060_warmth_v2 \
  --repair --force
```

Then: `prepare_graphs_cache.py` → `graphs_cache_gnn_datasets_4tasks_1060_warmth_v2` → retrain dim14-ce **from scratch** (no finetune on old labels).

### Code touched (fast path)

| File | Change |
|------|--------|
| `scripts_cosim/generate_gnn_datasets_fast.py` | Skip SSC when `GNN_CAPTURE=0`; delete stale SSC at start; redirect sim I/O when suppress env set |
| `src/executecosimulation.py` | `_init_worker`: devnull stdout/stderr when `COSIM_SUPPRESS_SIM_PRINTS=1` |
| `scripts_cosim/run_warmth_parallel_regen.sh` | 4-shard launcher (`TOTAL_DATASETS=500`) |

---

## Grid Comparison: Old 1060 Corpus vs Current warmth_v2

### Old grid (empirical from `artifacts/run_queue_big/gnn_datasets_4tasks_1060`, 1242 datasets, index to `ds_01389`)

| Axis | Values | Count |
|------|--------|------:|
| **Connectivity** | 0.25, 0.35, 0.50, 0.65, 0.80 | 5 |
| **Replicas** (per_client, per_server) | (1,1), (1,2), (1,3), (2,2), (2,3) | 5 |
| **Preinit** (client%, server%) | (0,0), (0.3,0.5), (0.5,0.7) | 3 |
| **Seeds** | 101, 202, 303, 404, 505 | 5 |
| **Queue regimes** | const8, pois12, pois18, pois28, norm16, norm24, norm35, uniform5_40, uniform20_80, uniform40_100 | 10 |

**Collapsed replica table (6 rows = rep × preinit baked):**  
(1,1,0,0), (1,2,0,0), (2,2,0,0), (1,3,0.3,0.5), (2,2,0.5,0.7), (2,3,0.3,0.5)

**Theoretical Cartesian:** 5 × 15 × 5 × 10 = **3750** · **1242 realized** (infeasible combos skipped)  
**User-facing index range:** ds_00000–ds_01499 target; **1242** on disk

**Workload templates:** 10 templates cycling 3 task mixes (0/100, 50/50, 100/0 dnn1/dnn2) — coupled to index, not a separate grid axis.

### Current grid (`generate_gnn_datasets_fast.py` → warmth_v2)

| Axis | Values | Count |
|------|--------|------:|
| **Connectivity** | 0.50 only | 1 |
| **Replicas** (per_client, per_server, client%, server%) | (1,2,0,0), (2,2,0,0), (1,3,0.3,0.5), (2,3,0.3,0.5), (2,2,0.5,0.7) | 5 |
| **Seeds** | 101–120 (20 consecutive) | 20 |
| **Queue regimes** | pois16, norm22, pois28, norm35, uniform20_80 | 5 |

**Total:** 1 × 5 × 20 × 5 = **500 datasets**  
**Dir name `1060_warmth_v2` is historical** — not 1060 combos.

### Compression intent (old → new)

```
19 conn × 15 rep-states × 9 queue  →  1 conn × 5 rep × 5 queue
5 spread seeds                     →  20 adjacent seeds (more realizations, fewer axes)
```

Deliberate trade: **brute-force search-space control** (e.g. `ds_00375` hit **82k** placements) over **topology / low-queue coverage**.

---

## Sparse expansion + merged cache (2026-06-11)

| Item | Value |
|------|-------|
| Preset | `--grid sparse_warmth_v2` in `generate_gnn_datasets_fast.py` |
| Conn | **{0.25, 0.30, 0.35}** |
| Replicas | **{(1,1), (1,2), (2,2)}** × queue regimes × seeds 121–133 |
| Output | `simulation_data/gnn_datasets_4tasks_sparse_warmth_v2` — **351/351** `best.json` |
| Cost | ~3–15s/ds typical @ conn≤0.35 |
| Merged cache | `graphs_cache_warmth_v2_sparse_merged` — **824 graphs** (473 warmth + 351 sparse), **6.06M** RTT rows |
| Finetune | `near-rtt-v2-warmth-sparse-finetune-dim14-ce-only.pt` — val acc **25.6%**; **pending live standard5 gate** |
| Pipeline | `run_warmth_sparse_recache_finetune.sh` · `train_near_rtt_v2_warmth_sparse_finetune.py` |

**What sparse fixes:** low-connectivity coverage in training (conn 0.25–0.35, light replicas).

**What sparse does NOT fix:** `degree_skewed_core`, asymmetric 5/30ms latency, bipartite hub geometry — Erdős–Rényi only.

**Planned `skew_warmth_v2` preset (~288 ds):** k_core∈{4,6,8} × seek∈{0.35,0.50,0.65} × replicas{(1,2),(2,2)} × queues{pois16,pois28} × seeds 141–148; 5/30ms; output `gnn_datasets_4tasks_skew_warmth_v2`.

### B1 feature regen pipeline (2026-06-11)

| Step | Status | Notes |
|------|--------|-------|
| Code: `capture_disk_snapshot`, dim13, src_norm, is_warm | **DONE** | `state_capture.py`, `prepare_graphs_cache_seq.py`, `feature_builder.py` |
| `--rewrite-ssc` on 824 ds | **Insufficient alone** | Old `optimal_result` lacks `disk_snapshot_by_task_type` in `schedulingStateCapture` |
| `--repair --force` | **NEXT** | Re-exports scheduling capture with disk snapshot |
| Recache → `graphs_cache_warmth_v2_features_v1` | **NEXT** | After repair |
| `skew_warmth_v2` co-sim | **NEXT** | Hub labels for GIN batch coupling |

**warmth finisher status:** mitrix warmth **473/480** best.json; full **500** rsync from datalab pending.

---

## Real-Life Sim vs GNN Training Fit

### Queue regime gap (from `verify_queue_gap_1060.py`)

| Regime | Max avg queue @ decision |
|--------|---------------------------|
| Old co-sim 1060 labels | ~11.5s avgQT · **18.4s** max queueTime |
| Live 150-150 Knative | **~53s** |
| Live default dim14-ce | **~21s** |

Co-sim remains **4–8× below** live 150-150 on queue depth; **~200×** claim applies to **qvm p95**, not avgQueueTime.

### Old grid strengths

- **Topology diversity** (sparse 0.25 → dense 0.80) → candidate-set / offloading coverage  
- **Low-queue anchors** (const8, pois12, uniform5_40) → sparse `default`/`00` live behavior  
- **(1,1) minimal replica** edge case  
- **1242 realized configs** — broader than 500  

### New grid strengths

- **Higher queue calibration** (pois16+, norm22+) — closer to contended live sim on average  
- **warmth v2 + defer_cold** defaults in co-sim  
- **500 complete Cartesian** — predictable count, faster regen  
- **20 seeds** — more topo realizations per (rep, queue) cell  

### New grid risks (early warmth_v2 sample vs old sample)

| Metric | Old sample | New sample (partial) |
|--------|------------|----------------------|
| queueTime max | 16.5s | 11.4s |
| queueTime p95 | 3.7s | 1.9s |
| mean RTT | 1.07s | 0.86s |
| cacheHit | 100% | 100% (pre-warmth-label shift) |

**Narrower RTT spread** → easier fit, less tail-regret signal. **Fixed conn=0.50** → weaker generalization on sparse/dense live configs unless hybrid grid added.

### GNN feature axes affected

| Feature | Old grid | New grid |
|---------|----------|----------|
| Candidate count (topology × replicas) | High variance | Low variance (conn fixed) |
| Queue dim 7 @ schedule | Low + high cells | Shifted high; **no low anchor** |
| `shared_fate` / `initialized_snapshot` | ~71% cold Phase 1 | Same mechanism; warmth v2 changes pull labels post-regen |
| RTT label tails | Up to ~18s queue | TBD after full warmth v2 regen |

---

## Recommendations

### If goal = live-sim robustness (hybrid)

Keep warmth v2 + high-queue regimes; **restore subset of old axes:**

- Connectivity: **0.25, 0.50, 0.80** (3 values)  
- Add **(1,1,0,0)** + **one low-queue anchor** (const8 or pois12)  
- → ~750–900 datasets; still parallelizable  

### If goal = fast warmth-v2 retrain (current path)

**500 grid is rational**; accept narrower deploy envelope (conn≈0.5, no (1,1), no low-queue floor). Do **not finetune** dim14-ce on old cache.

---

## Key Files

| Path | Role |
|------|------|
| `scripts_cosim/generate_gnn_datasets_fast.py` | Grid definition + fast-path SSC skip |
| `scripts_cosim/run_warmth_parallel_regen.sh` | 4-shard parallel launcher |
| `scripts_cosim/run_warmth_full_regen_recache.sh` | Full pipeline (regen → recache → retrain) |
| `scripts_cosim/refresh_optimal_full_stats.py` | Post-hoc SSC + full stats enrichment |
| `simulation_data/gnn_datasets_4tasks_1060_warmth_v2/` | warmth_v2 co-sim output |
| `scripts_cosim/run_warmth_sparse_recache_finetune.sh` | Sparse recache + merged cache + finetune |
| `simulation_data/gnn_datasets_4tasks_sparse_warmth_v2/` | Sparse co-sim output (351 ds) |
| `simulation_data/graphs_cache_warmth_v2_sparse_merged` | Merged 824-graph cache |

---

## Non-unique placement backfill (2026-06-15, datalab)

Older warmth_v2 + sparse_warmth_v2 corpora were generated **without** `--allow-non-unique-replicas` (distinct-platform enum only). Backfill re-enumerates the **full cartesian** placement space per existing `ds_*` without re-running topology regen.

| Corpus | Result | Notes |
|--------|--------|-------|
| `gnn_datasets_4tasks_1060_warmth_v2` | **492 SUCCESS + 9 SKIPPED** · **498 jsonl** · **490 non_unique_meta** | job `482773` |
| `gnn_datasets_4tasks_sparse_warmth_v2` | **348 SUCCESS** · **351 jsonl** | 3 ds cartesian-full (delta=0): `ds_00081`, `ds_00224`, `ds_00251` · job `482774` |
| `gnn_datasets_4tasks_contention_v2` | **900/900 jsonl** | finisher `482884` (was 711) |
| `gnn_datasets_4tasks_contention_v3` | **900/900 jsonl** | conn∈{0.15,0.20} + heavier queues; coupling **decreased** vs v2 (4.2% vs 7.2% coupled >1%) — hypothesis rejected |

**Scripts:**
- `scripts_cosim/generate_non_unique_placements_fast.py` — `--datasets-dir`, `--temp-dir`; fail-loud on 0 combos or incomplete sims
- `scripts_cosim/datalab/warmth_non_unique_{warmth,sparse}.sbatch`, `submit_warmth_non_unique_datalab.sh`
- Monitor: `cosim_health_report.sh`, `cosim_health_remediate.sh`, `cosim_monitor_loop.sh`

**Monitor gotcha:** sparse "348/351 SUCCESS" is **complete** when `non_unique_meta≥348` and jsonl=351 — 3 ds already had full cartesian labels.

**Separability re-audit DONE 2026-08-04:** retained clean sweeps warmth **487**, sparse **351**,
contention_v2 **899**, contention_v3 **900**. Collision / coupled>1%: warmth **39.8%/9.7%**,
sparse **33.0%/10.3%**, v2 **40.3%/7.1%**, v3 **36.3%/4.2%**. Frozen report:
`simulation_data/separability_audit_4corpus_20260804.json`.

**Recache follow-up DONE 2026-08-04:** CACHE 5.5 cache
`graphs_cache_contention_v2_873_v5.5` (873 graphs, 3,245,943 RTT rows) validates under
`validate_training_cache_contract.py`. The prior 899 retained set dropped **26**
`best_json_rtt_mismatch` datasets whose JSONL sweeps are network-incoherent with
`optimal_result` (sweep-min labels absent from SSC+network candidate edges); prepare now
fails loud instead of silently omitting them. Multi-seed RQ3 ablation frozen in
`gnn_necessity_ablation_contention_v2_873_v5.5_multiseed.json`.

### Mitrix pull status (2026-06-27)

| Corpus | jsonl on mitrix | Notes |
|--------|----------------:|-------|
| warmth_v2 | **487 retained** | 13 exclusions: 4 corrupt + 9 never-generated |
| sparse_warmth_v2 | **351** | complete |
| contention_v2 | **873 retained** | −26 network-incoherent + `ds_00751` truncated |
| contention_v3 | **900** | corpus only — train/cache/models on datalab |

---

## Changelog

| Date | Change |
|------|--------|
| 2026-08-04 | v5.5 recache 873 + multi-seed ablation DONE; 26 incoherent JSONL excluded; RQ3 offline claim updated |
| 2026-08-04 | Strict 899 contention cache built; ablation blocked by 26 absent graph labels; label/feature/replica-state contract must be repaired before retraining |
| 2026-08-04 | Placement integrity clean; four-corpus separability audit frozen; stale unique-pass 0%/0% claims replaced |
| 2026-06-27 | Mitrix rsync done (warmth 503 jsonl); merge live gates closed; 5 corrupt jsonl flagged; separability re-audit pending |
| 2026-06-16 | contention_v3 900/900 DONE; coupling audit v3 < v2; recache contention_v2 pending (711→900 graphs) |
| 2026-06-15 | Non-unique backfill DONE (warmth 498 jsonl, sparse 351, contention 900/900); monitor scripts; rsync+recache pending |
| 2026-06-11 | Sparse 351/351 + merged 824 cache + finetune section; warmth finisher 473/500 status |
| 2026-06-11 | Session doc: grid comparison, P0/P1 fixes, parallel fast-path regen, enrichment A/B PASS, hybrid grid recommendation |
