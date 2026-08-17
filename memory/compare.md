# Normal Sim Policy Comparison Guide

**Last Updated:** 2026-08-13 (Regime B oracle_split_v1 CLOSED via Phase 3.1)

> Reference for `/compare`. Project state: `memory/memory.md`. **GNN-necessity / separability:** `memory/gnn_necessity_separability.md`. **GNN vs MLP mechanism:** `memory/gnn_v2_sparse_topology_and_features.md` §4.

---

> ### >>> HEADLINE (2026-08-13): Regime B `oracle_split_v1` — learned policy hits physics ceiling <<<
>
> Primary metric = **max burst elapsed** (`regime_b_primary_score_s`) — **not** `total_rtt`.
>
> | Policy | Primary | vs oracle | Notes |
> |--------|--------:|----------:|-------|
> | **Oracle parallel** | **31.65s** | — | Forced 1 cold pull / server |
> | **`ect_pull` (physics teacher)** | **31.65s** | **0.00s** | FilterStore-aware ECT |
> | **Multiseed distill + `seq_reforward_pull` ★** | **31.66s** | **+0.00s** | Cold 6k-frame soft KL; ckpt `…-ect-pull-distill-multiseed.pt` |
> | Phase 3 single-stub distill + pull decode | 62.63s | +31s | PARTIAL (superseded) |
> | CE dim16 argmax | 125.28s† | +94s | †this eval; post-parity runs also ~93.95s |
> | Distill argmax (no pull ledger) | 125.28s | +94s | Decode required |
> | Warm/busy v1 harvest distill | 375.87s | +344s | **REJECTED** — teacher poison |
>
> **Claim bounded** to intel cell `oracle_split_v1`. Deploy path remains 873/v5.5 unless transfer eval says otherwise.
> Eval: `regime_b_phase3_ect_pull_distill_eval_multiseed_cold/summary.json` · PR #3 merge `040fdcb`
>
> ---
>
> ### >>> HEADLINE (2026-06-14): MLP dim22 **batchcache** beats GNN wssm **8/9** on bipartite v2 hub grid <<<
>
> | Policy | 9× sum | vs Knative | vs GNN | Wins | Train cache | Notes |
> |--------|-------:|-----------:|-------:|-----:|-------------|-------|
> | **MLP dim22 batchcache ★** | **6.09M** | **−67%** | **−39%** | **8/9** | `graphs_cache_warmth_v2_sparse_skew_merged` (same as GNN) | **Ship for hub/node_disk_v2** |
> | GNN wssm | 10.02M | −45% | — | 8/9 vs Kn only | same batch cache | Only wins k6_seek65 (tie) |
> | MLP ce_reduced | 939.76M | +5017% | — | 0/9 | seq cache | **REJECT** — dropped is_warm/is_cold, queue mismatch |
> | MLP dim22 seq-fix | 49.21M | +168% | +391% | 0/9 | seq cache (raw queue + is_cold) | train/serve mismatch |
>
> **Root cause ladder:** ce_reduced bugs → seq dim22 partial fix → **batch dim22 = parity with GNN features → MLP wins**.
> Model: `models/tabular/batch_edge_mlp_warmth_sparse_skew_merged_dim22_batchcache.pt` · Sweep: `bipartite_v2_mlp_dim22_batchcache_20260614/`

---


## At a Glance

| Item | Value |
|------|-------|
| **Metric** | `total_rtt` (lower is better) — **except Regime B: use max-burst elapsed** |
| **Workloads** | `workload-100-100.json` (201k tasks, standard 7-config) · `workload-125-225.json` (562k tasks, skew-4 / stress) |
| **Seed** | `42` (sim); workload traces unseeded unless noted |
| **Simulation** | Normal sim — cold autoscale from zero (**not** co-sim) |
| **Decode** | **argmax** (default); Regime B intel → **`seq_reforward_pull`** |
| **Δ% vs dim14-ce** | `(policy − dim14-ce) / dim14-ce × 100` — **positive = worse** |
| **Δ% vs Knative** | `(policy − knative) / knative × 100` — **negative = better than Knative** |

### Active policies (top of doc)

| Role | Policy | Model / layout |
|------|--------|----------------|
| **Regime B intel (★★ ceiling)** | `--gnn` · `seq_reforward_pull` · dim24 | `models/near-rtt-v2-regime-b-oracle-split-v1-ect-pull-distill-multiseed.pt` — **31.66s** |
| **Regime B physics ceiling** | `knative_network_ect_pull` | no learned weights — **31.65s** |
| **GNN anchor (warmth)** | `--gnn` · argmax | `models/near-rtt-v2-warmth-dim14-ce-only.pt` (14-dim; warmth_v2 cache) |
| **GNN contention_v2 (collision labels)** | `--gnn` · argmax · dim22 | `models/near-rtt-v2-contention-v2-dim14-ce-only.pt` — live gate **PARTIAL** (sum wins, per-config 1/1/1) |
| **MLP contention_v2 batchcache** | `--mlp_batch` · dim22 | `models/tabular/batch_edge_mlp_contention_v2_dim22_batchcache.pt` |
| **GNN anchor (legacy 7/7)** | `--gnn` · argmax | `models/near-rtt-v2-dim14-ce-only.pt` (1060 cache; bipartite/skew sweeps) |
| **MLP wssm dim22 batchcache (★★ hub deploy)** | `--mlp_batch` · `INFERENCE_FEATURE_LAYOUT=dim22` | `models/tabular/batch_edge_mlp_warmth_sparse_skew_merged_dim22_batchcache.pt` — **8/9 vs GNN+Kn, sum 6.09M** |
| **GNN wssm (bipartite v2)** | `--gnn` · argmax · `INFERENCE_FEATURE_LAYOUT=dim22` | `models/near-rtt-v2-warmth-sparse-skew-merged-dim14-ce-only.pt` (same batch cache as MLP-bc) |
| **MLP wssm ce_reduced** ⚠ REJECT | `--mlp_batch` · `ce_reduced` | `batch_edge_mlp_warmth_sparse_skew_merged_ce_reduced.pt` — 939M (broken) |
| **MLP warmth_v2** | `--mlp_batch` | `models/tabular/batch_edge_mlp_warmth_v2.pt` |
| **MLP dim22** | `--mlp_batch` · Regime A batch loop | `models/tabular/batch_edge_mlp.pt` · `INFERENCE_FEATURE_LAYOUT=dim22` |
| **MLP atomic21** | `--mlp_batch` | `models/tabular/batch_edge_mlp_atomic21.pt` · layout `atomic21` |
| **XGB Regime A** | `--xgboost_batch` | `models/tabular/batch_edge_ranker.json` |

**Ship gate (new dim14 checkpoints):** beat **dim14-ce** on 5-config sum; `default` not >+2% vs ce-only. **Contention_v2 gate (2026-06-15):** sums GNN **19.88M** < Kn **20.73M** < MLP **25.35M**; per-config **GNN 1 / MLP 1 / Kn 1** — **PARTIAL** (offline GNN necessity does not uniformly transfer). **Hub deploy (node_disk_v2, bipartite):** **MLP dim22 batchcache** (8/9, 6.09M, −67% vs Kn) **preferred over GNN wssm** (10.02M) when train/serve feature parity matters. **Regime B intel:** multiseed ect_pull distill + `seq_reforward_pull` (not argmax).

---

## Regime B — `oracle_split_v1` (CLOSED 2026-08-13)

| Item | Value |
|------|-------|
| Stub | `simulation_data/regime_b_cold_burst_v1/live_stub_oracle_split_v1` |
| Physics | `platform_reuse_v1` |
| N | 12 cold tasks @ burst |
| Primary | `regime_b_primary_score_s` (max burst elapsed) |
| Teacher | live `knative_network_ect_pull` |
| Student | soft KL distill (α=0.5, τ=0.25) on **6000** cold dim24 frames |
| Decode | `GNN_DECODE_MODE=seq_reforward_pull` |
| Result | **31.66s** ≡ ect_pull **31.65s** |
| Falsified | Set Transformer / soft_combo_conc / N=4 CE / hard-CE distill / warm-busy harvest |

```bash
pipenv run python3 scripts_cosim/run_phase3_ect_pull_distill_eval.py \
  --gnn-model models/near-rtt-v2-regime-b-oracle-split-v1-ect-pull-distill-multiseed.pt \
  --output-dir simulation_data/normal_sim_sweeps/regime_b_phase3_ect_pull_distill_eval_multiseed_cold
```

---

## GNN vs MLP — mechanism (not features alone)

| Fact | Implication |
|------|-------------|
| Graph = **bipartite task→platform edges only** | No physical multi-hop message passing |
| **Same decode** (argmax, frozen logits) | Batch advantage = **GIN coupling**, not re-planning after collisions |
| dim22 MLP has **`src_norm` live** | Restoring in cache = GNN train/serve parity, not MLP blindness |
| **`node_disk_hit`** on dim13/dim22 | Helps **both** models unless batch-level pull features added |
| Offline metrics | GNN **`val/task_acc`** (~69%) ↔ MLP **`val_edge_accuracy`** (~68%) — not GNN `val/acc` (~23%) |
| **Hub v2 batchcache** | **MLP-bc 8/9, 6.09M** beats GNN 10.02M when train cache = GNN batch cache + dim22 inference |
| Phase boundary | v1: GNN 17.90M vs MLP 21.49M (8/9 GNN). **v2 batchcache: MLP 6.09M vs GNN 10.02M — MLP 8/9** (feature parity); ce_reduced MLP 939M was encoding bug; triangle MLP +0.4% |

Full table: `memory/gnn_v2_sparse_topology_and_features.md` §4.

**Sweep sizes:** **standard5** (`default`, `01`, `02`, `03`, `05`) for dev · **all7** (+ `00`, `04`) for paper · **skew-4** (+ degree-skew variants) · **skew3** (warmth gate) for topology stress.

---

## FilterStore storage contention (verified 2026-06-11)

Full map: **`memory/storage_contention.md`**.

| Item | Value |
|------|-------|
| T_pull (nofs-ids, dnn1/rpiCpu) | **31.30s** |
| N=4 last-task contended / parallel | **125.57s / 31.65s** (+297%) |
| Last-task penalty | **93.91s = (N−1)×T_pull** |
| N=1..5 RTT multiplier | **1.00×, 1.99×, 2.98×, 3.97×, 4.96×** |
| Platform queue at schedule | **0** — not the driver |
| GNN queue feature (dim 7) | **0** — blind to pull serialization |
| GNN dim8 (`shared_fate`) | **1.0** when all node plats cold — **saturates** (no N ordinal) |
| Task stat `queueTime` | **31…125s** steps — outcome/label, not GNN input |
| Temporal dims (cold/exec/comm /10) | **~0.03 max** — sub-second; not pull wait |
| `pullTime` under fast_forward | **0** (artifact); FF off → equals queueTime |
| `total_rtt` vs last-task (N=4 contended) | **501s vs 126s (3.99×)** — do not sum elapsed |
| Early-task `invisible_gap` (task 0) | **93.56s** — exec-phase FilterStore wait |
| Counterfactual warm (same node) | **0.35s** — N× gone |
| Counterfactual 4× flashCard node0 | **31.65s** — N× gone |
| Pending pull count feature | **dim24 pull-obs + `seq_reforward_pull` ledger (Regime B CLOSED)** |

**Run:**
```bash
pipenv run python3 scripts_cosim/test_memory_contention_ab.py
pipenv run python3 scripts_cosim/test_cold_start_queue_last_task_ab.py
```

**Results:** `simulation_data/memory_contention_ab/summary.json` · `simulation_data/cold_start_queue_last_task_ab/summary.json`

---

## Platform warmth model (expert reference 2026-06-11)

Full map: **`memory/warmth_model.md`**. Send to external reviewers with `storage_contention.md`.

| Item | Value / finding |
|------|-----------------|
| Warm predicate (sandbox) | `previous_task.type == task_type` on **same platform** |
| Pull skip (v2 co-sim default) | **`node_disk_v2`:** skip when **`has_function`** on node local storage |
| Pull skip (v1 legacy) | Coupled to sandbox — 2nd plat same node still ~31s pull |
| Gates | **~31s image pull** + **~0.33s sandbox cold-start** |
| Node disk cache on pull | **v2:** yes (determined/co-sim) · **v1/1060 corpus:** no |
| A→B→A same platform | Full pull + cold sandbox even if image still on disk |
| N same-function plats, 1 node | N × ~31s pull (serialized) — pessimistic vs K8s node cache |
| Co-sim task `coldStartTime>0` | **~0.1%** |
| Co-sim platform cold (`initialized_snapshot`) | **~71%** |
| GNN edge `is_warm` | Matches sandbox predicate via `previous_task_type_name` |
| `node.cache_hits` stat | Hard-coded `+= 0` on determined path |

**1060 label gap (historical):** pullTime **0%** · queueTime max **18.4s** — see **`memory/cosim_warmth_gap.md`**. **warmth_v2 regen** changes pull labels; graph features still lack disk hit.

---

## contention_v2 live gate (2026-06-15, node_disk_v2)

**Purpose:** Test GNN/MLP trained on **collision-aware** co-sim labels (`contention_v2`, 711-graph cache) vs Knative on sparse ER topologies.

**Sweep:** `contention_v2_live_gate_20260615/` · workload-125-225 · argmax · seed 42

| Config | Knative | GNN | MLP dim22 | Winner | GNN/Kn | MLP/Kn |
|--------|--------:|----:|----------:|--------|-------:|-------:|
| sparse_p25 | 6.90M | 7.66M | **5.81M** | MLP | 1.11× | **0.84×** |
| sparse_p35 | 12.77M | **11.08M** | 16.79M | GNN | **0.87×** | 1.31× |
| sparse_p25_skew | **1.06M** | 1.14M | 2.75M | Kn | 1.07× | 2.60× |
| **SUM** | 20.73M | **19.88M** | 25.35M | **GNN sum** | **0.96×** | 1.22× |

**Models:** GNN `near-rtt-v2-contention-v2-dim14-ce-only.pt` (val 66.2%) · MLP `batch_edge_mlp_contention_v2_dim22_batchcache.pt` (val edge 90.0%)

**Compare:** `pipenv run python3 scripts_cosim/important/compare_contention_v2_live_gate.py --sweep-dir simulation_data/normal_sim_sweeps/contention_v2_live_gate_20260615`

**Takeaway:** GNN beats Kn+MLP on **3-config sum** but per-config wins split 1/1/1. Offline ablation GNN advantage (collision robustness) does **not** uniformly transfer — MLP wins sparse_p25 live. Next: decode collision repair.

**Offline ablation (same corpus):** `gnn_necessity_ablation.py` — gnn_base beats pointwise on top1/p90/opt-recov; see `memory/gnn_necessity_separability.md`.

---

## contention_v3 live gate (2026-06-20, node_disk_v2)

**Purpose:** End-to-end train+deploy on **contention_v3** grid (conn 0.15/0.20, heavier queues) — test whether pushing co-sim sparsity improves live sparse ER performance.

**Sweep:** `contention_v3_live_gate_20260620/` · workload-125-225 · argmax · seed 42 · cache **900 graphs**

| Config | Knative | GNN v3 | MLP v3 | Winner | GNN/Kn | MLP/Kn |
|--------|--------:|-------:|-------:|--------|-------:|-------:|
| sparse_p25 | 7.05M | 8.26M | 9.52M | Kn | 1.17× | 1.35× |
| sparse_p35 | 12.00M | 26.45M | 33.16M | Kn | 2.20× | 2.76× |
| sparse_p25_skew | 1.22M | 1.29M | **1.08M** | MLP | 1.06× | **0.88×** |
| **SUM** | **20.27M** | 36.00M | 43.76M | **Kn sum** | 1.78× | 2.16× |

**Models:** GNN `near-rtt-v2-contention-v3-dim14-ce-only.pt` (val **65.6%**) · MLP `batch_edge_mlp_contention_v3_dim22_batchcache.pt` (val edge **88.2%**)

**Compare:** `python3 scripts_cosim/important/compare_contention_v2_live_gate.py --sweep-dir simulation_data/normal_sim_sweeps/contention_v3_live_gate_20260620`

**Takeaway:** **REJECT v3 for deploy.** Offline coupling drop (4.2% vs v2 7.2%) predicts live regression: GNN sum **+81%** vs contention_v2 **19.88M**; Kn wins 2/3. Ship **contention_v2** GNN only. **Mitrix:** corpus 900/900 jsonl local; train/cache/models/live-gate sweep on **datalab** only unless pulled.

---

## Strategic-merge deployment (2026-06-16, mitrix · closed 2026-06-27)

**Train:** warmth + sparse + contention_v2, strategic coupled oversample · cache `graphs_cache_strategic_merge_wss_cont_v2` (3729 graphs) · GNN val **48.8%** · MLP val **78.5%**

### WSSM hub gate (`strategic_merge_wss_live_gate_20260616/`)

| Config | Knative | GNN | MLP | Winner |
|--------|--------:|----:|----:|--------|
| hub_k4_seek50 | 686k | 1.95M | 1.23M | Kn |
| hub_k6_seek50 | 646k | 2.38M | 1.39M | Kn |
| hub_k8_seek50 | 582k | 1.60M | 718k | Kn |
| **SUM** | **1.91M** | **5.93M** | **3.34M** | Kn **3/3** |

### Contention sparse gate (`strategic_merge_contention_live_gate_20260616/`)

| Config | Knative | GNN | MLP | Winner |
|--------|--------:|----:|----:|--------|
| sparse_p25 | 8.03M | 8.77M | 12.70M | Kn |
| sparse_p35 | 10.57M | 11.24M | 68.38M | Kn |
| sparse_p25_skew | 1.72M | 3.14M | 4.61M | Kn |
| **SUM** | **20.32M** | **23.15M** | **85.69M** | Kn **3/3** |

**vs contention-only (711-cache):** GNN sum **+17%** worse; MLP **+238%** (sparse_p35 cliff). **Reject** merged training for sparse ER deploy.

**Compare:** `pipenv run python3 scripts_cosim/important/compare_wssm_expanded_live_gate.py --sweep-dir …/strategic_merge_wss_live_gate_20260616` (or read `compare.txt`)

---

## Weighted-merge deployment (2026-06-16, mitrix · closed 2026-06-27)

**Train:** 8× coupled oversample · cache `graphs_cache_warmth_sparse_contention_v2_weighted` (2778 graphs) · GNN val **26.3%** · MLP val **72.3%**

**Sweep:** `merged_contention_weighted_live_gate_20260616_105500/`

| Config | Knative | MLP | gnn_uniq | gnn_argmax | Best |
|--------|--------:|----:|---------:|-----------:|------|
| sparse_p25 | 7.53M | 14.91M | 9.14M | 9.30M | kn |
| sparse_p35 | 12.29M | 64.89M | **11.34M** | 12.17M | gnn_uniq |
| sparse_p25_skew | 1.33M | 1.84M | 1.82M | 3.12M | kn |
| **SUM** | **21.14M** | **81.64M** | **22.30M** | 24.59M | kn sum |

**Wins:** kn=2 · gnn_uniq=1 · mlp=0. **gnn_uniq** beats argmax on sum; still loses 3-config sum to Kn (+5%).

**Compare:** `compare_merged_contention_live_gate.py --sweep-dir …/merged_contention_weighted_live_gate_20260616_105500` (or read `compare.txt`)

---

## warmth_v2 skew3 live (2026-06-11, node_disk_v2 physics)

**Configs:** `default`, `05_sparse`, `default_degree_skew` (3-config skew3 gate).

### Old models (v1 MLP + pre-warmth GNN checkpoint)

**Sweep:** `warmth_v2_physics_skew3_20260611/` · sums Kn **5.85M** · MLP **4.25M** · GNN **5.02M** · **MLP wins 2/3**

### Ce-reduced v2 models

**Sweep:** `warmth_v2_ce_reduced_skew3_20260611/` · **v1 MLP still wins 2/3** · v2 GNN sparse **1.23M** vs Kn **0.93M** (regression)

**Takeaway:** v2 physics in sim ≠ GNN win without **`node_disk_hit` + hub topology in co-sim grid**. Gate pending: warmth dim14-full + standard5.

### wssm models skew3 gate (2026-06-14, node_disk_v2)

**Sweep:** `skew3_full_gate_20260614/` · GNN wssm (`near-rtt-v2-warmth-sparse-skew-merged-dim14-ce-only.pt`) · MLP wssm (broken)

| Config | GNN wssm | MLP wssm | Knative | Winner | GNN vs Kn |
|--------|--------:|---------:|--------:|--------|----------:|
| `default` | 5.29M | 31.84M | **3.10M** | Knative | +70.6% |
| `default_degree_skew` | **0.51M** | 16.71M | 0.61M | GNN | −15.6% |
| `05_sparse_degree_skew` | 1.00M | 12.37M | **0.79M** | Knative | +26.9% |

**Wins:** GNN wssm **1/3** · Knative **2/3** · MLP broken (16–32M). GNN wssm wins only on `degree_skew`; `default` still loses to Knative (+70%). **Gate still not passed.** MLP wssm is broken (ce_reduced layout mismatch).

**Compare:** `pipenv run python3 scripts_cosim/important/compare_mega_matrix.py --group 3`

---

## Co-sim warmth gap — 1060 corpus (historical)

Full audit: **`memory/cosim_warmth_gap.md`**.

| Item | Value |
|------|-------|
| Corpus | `gnn_datasets_4tasks_1060` · **1230** datasets · **4920** tasks |
| pullTime > 0 | **0%** — dead field in stored labels |
| cacheHit=True | **100%** — preinit `initialized.succeed()` without pull |
| queueTime max | **18.37s** — **never ≥30s** (no N× pull in labels) |
| dnn1 queueTime max | **1.22s** |
| coldStartTime > 0 | **0.06%** |
| initialized_snapshot cold | **~71%** at Phase 1 schedule (feature only) |
| defer_cold in stored config | **missing / False** |
| Co-located 2+ tasks same node | **32%** datasets · max queue **8.2s** |
| A/B vs 1060 | A/B: queue **31…125s** · 1060: max **18s** |
| Rerun co-sim on warmth fix? | **Yes** — regen + recache + retrain |

---

## Train/Serve queue gap (verified 2026-06-10)

**Do not cite ~200× on mean queue wait.** Source: `scripts_cosim/verify_queue_gap_1060.py` · `logs/queue_gap_1060_verification.json`.

| Metric | Co-sim 1060 | Co-sim 3705 | Live 150-150 Kn | Live default (dim14-ce) |
|--------|------------:|------------:|----------------:|------------------------:|
| max avgQueueTime | 11.54s | 6.50s | 52.7s | 20.6s |
| p99 avgQueueTime | 5.48s | 2.28s | — | — |
| max per-task queueTime | 18.4s | 11.7s | — | — |
| max snapshot depth (tasks) | 98 | 67 | — | — |

- **avgQT scale:** ~4.6–8× (1060/3705 max → 150-150); ~1.8–3.2× (1060 max → default)
- **qvm p95 (separate):** CE ~408 vs ranking ~2857 on default pre-fix — decode-time platform queue task-count, not avgQueueTime
- **Fast-forward warmup:** RTT/avgQT identical ff on/off on 1060 high-queue A/B (`ds_00479`, `ds_00319`, `ds_00016`)

---

## Verified claims audit (2026-06-10)

Source: `scripts_cosim/audit_doc_claims.py` → `logs/doc_claims_audit.json`. Re-run after new sweeps.

| Claim | Verified value | Status |
|-------|----------------|--------|
| CE-only 7-config sum | **17.893M** | ✓ JSON |
| CE wins vs Knative / HRC | **7/7** each | ✓ (Kn `default` from `baseline_default_100100/`) |
| Post-fix 3cfg CE sum | **8.025M** | ✓ `dim14_3model_3cfg_queuefix_20260609/` |
| Ranking post-fix `default` | **5.478M** (+27% vs CE 4.308M) | ✓ |
| qvm p95 CE / ranking | **408 / 2857** | ✓ decode_stats sidecars |
| Bipartite sums GNN / MLP / Kn | **17.899 / 21.488 / 18.370M** | ✓ 9 configs each |
| 150-150 Kn / seqblend | **23.761 / 20.360M** (−14.3%) | ✓ |
| Weak baseline Δ vs pre-fix CE | Kn **+25–36%**, Random **+250–455%**, RR **+78–124%** | ✓ |
| Co-sim task cold start | **~0.1%** (`coldStartTime>0`, n=1230) | ✓ |
| Live CE `coldStartProportion` | **9.6–16.4%** (mean 13.4%) | ✓ |
| Co-sim platform cold (`initialized_snapshot`) | **~71%** | ✓ |
| Prior “28% vs 88%” cold-start | — | **REJECTED** (no `sim_gap_analysis.py`; numbers don't match any metric) |
| Intra-batch collision rate (CE default) | **18.6%** batches | ✓ decode_stats |
| LQB probe vs 7/7 sweep `default` | **4.264M vs 4.610M** | ✓ (don't conflate) |

---

## dim14-ce anchor — 7/7, workload-100-100, seed 42

| Field | Value |
|-------|-------|
| Checkpoint | `models/near-rtt-v2-dim14-ce-only.pt` |
| Cache | `graphs_cache_gnn_datasets_4tasks_1060` (14-dim) |
| Sweep | `gnn_near_rtt_v2_dim14_ce_only_20260609/results/` |

| Config | **dim14-ce** | Knative | HeroCache |
|--------|-------------:|--------:|----------:|
| `default` | **4.14M** | 5.59M | 5.00M |
| `00` | **3.00M** | 3.94M | 3.45M |
| `01` | 2.25M | 2.82M | 2.82M |
| `02` | **1.84M** | 2.31M | 2.80M |
| `03` | **2.54M** | 3.33M | 2.92M |
| `04` | **1.67M** | 2.19M | 2.59M |
| `05` | **2.45M** | 3.35M | 2.85M |

**Wins vs Knative/HRC:** dim14-ce **7/7** each. Knative: `knative_network_20260606_192413/results/` (00–05) + `baseline_default_100100/results/knative_default_20_20_p50.json`. HRC: `herocache_network_20260606_205112/results/`.

---

## Regime A — Reviewer Triangle (all7, workload-100-100, datalab)

**Regime A = shared batch loop** (2–4 tasks, sequential argmax). dim14-ce anchor 7/7 **already is** Regime A; triangle reruns co-locate tabular JSONs.

**Sweep:** `reviewer_triangle_all7_20260609/results/` · seed 42 · anchor GNN from `gnn_near_rtt_v2_dim14_ce_only_20260609/`

| cfg | dim14-ce | MLP | XGB | MLP Δ vs GNN | XGB Δ vs GNN |
|-----|----------|-----|-----|--------------|--------------|
| default | 4.14M | **3.75M** | 10.89M | **−9.6%** | +163% |
| 00 | 3.00M | 3.05M | 5.34M | +1.7% | +78% |
| 01 | 2.25M | 2.30M | 2.73M | +2.5% | +22% |
| 02 | 1.84M | 1.88M | 2.06M | +1.9% | +12% |
| 03 | 2.54M | **2.49M** | 3.49M | **−2.1%** | +37% |
| 04 | 1.67M | 1.82M | 1.93M | +9.1% | +15% |
| 05 | 2.45M | 2.68M | 3.91M | +9.4% | +60% |

**7-config sums:** GNN **17.89M** · MLP **17.96M (+0.4%)** · XGB **30.34M (+70%)** · MLP wins **2/7** vs anchor · XGB **0/7**.

Models: MLP `batch_edge_mlp.pt` (dim22 layout) · XGB `batch_edge_ranker.json`.

---

## Skew-4 — dim14 GNN vs dim22 MLP vs Knative

**Configs:** `default_20_20_p50` · `05_sparse_40_40_p25` · `default_20_20_degree_skew` · `05_sparse_40_40_p25_degree_skew` (from `atomic21_skew_configs/`).

**GNN:** dim14-ce · **MLP:** `batch_edge_mlp.pt` + `INFERENCE_FEATURE_LAYOUT=dim22` · **Knative:** `--knative_network` (per-arrival, not batch).

### workload-100-100 (local)

**Sweep:** `dim14_old_models_skew4_20260610/results/`

| Config | GNN | MLP dim22 | Best | GNN win? |
|--------|----:|----------:|------|----------|
| `default` | 4.33M | **4.10M** | MLP | no |
| `05_sparse` | 2.66M | **2.59M** | MLP | no |
| `default_degree_skew` | **1.03M** | 1.09M | GNN | yes (+6%) |
| `05_sparse_degree_skew` | **997k** | 1.14M | GNN | yes (+13%) |

**GNN wins 2/4** (degree-skew only).

### workload-125-225 (datalab)

**Generate workload:** `pipenv run python3 -m src.generator -d data/nofs-ids --generate-traces -r 125 -s 225` → `data/nofs-ids/traces/workload-125-225.json` (562k events).

**Sweep:** `dim14_old_models_skew4_125225_20260610/results/` · SLURM `scripts_cosim/datalab/skew4_knative.sbatch` + `dim14_old_models_skew4.sbatch`

| Config | GNN | MLP dim22 | Knative | Best |
|--------|----:|----------:|--------:|------|
| `default` | 30.66M | **21.50M** | 27.00M | MLP |
| `05_sparse` | 10.81M | **7.82M** | 10.43M | MLP |
| `default_degree_skew` | **1.27M** | 1.85M | 2.19M | GNN |
| `05_sparse_degree_skew` | **1.28M** | 1.51M | 1.56M | GNN |

**Δ vs Knative (negative = beats Knative):**

| Config | GNN | MLP |
|--------|----:|----:|
| `default` | +13.6% | **−20.4%** |
| `05_sparse` | +3.7% | **−25.0%** |
| `default_degree_skew` | **−42.1%** | −15.6% |
| `05_sparse_degree_skew` | **−17.7%** | −3.2% |

**4-config sums:** GNN 44.0M · MLP **32.7M** · Knative 41.2M · wins **GNN 2/4 · MLP 2/4 · Knative 0/4**.

**Takeaway:** MLP dim22 wins uniform/high-load configs; GNN wins degree-skew (largest gap vs Knative on skew). GNN RTT explodes on uniform 125-225 while skew stays ~1.2–1.8M.

---

## Tiered-hub — dim22 GNN vs MLP (`degree_skewed_core`)

**Configs:** 9 hubs `hub_k{2,4,6}_seek{30,50,80}` + controls `default_20_20_p50` · `05_sparse_40_40_p25` (11 total for 125-225 sweep).

**Policies:** `gnn_dim22` (dim14-ce + `INFERENCE_FEATURE_LAYOUT=dim22`) · `mlp_dim22` (`batch_edge_mlp.pt`) · **Knative** `--knative_network` (per-arrival; 125-225 job pending).

**Topology:** `k_core` Xavier hub servers; `hub_seeker_fraction` = seek%; clients wired to core (`p_core=0.95`) vs periphery (`p_periphery=0.15`). Generator: `generate_tiered_hub_configs.py`.

### workload-100-100 (datalab job 477929)

**Sweep:** `tiered_hub_gnn_mlp_20260610/results/` · seed 42

| Config | GNN dim22 | MLP dim22 | Best | GNN Δ vs MLP |
|--------|----------:|----------:|------|-------------:|
| `hub_k2_seek30` | **2.25M** | 2.43M | GNN | −7.7% |
| `hub_k2_seek50` | 2.07M | **1.91M** | MLP | +8.6% |
| `hub_k2_seek80` | **3.04M** | 3.05M | GNN | −0.4% |
| `hub_k4_seek30` | **922k** | 931k | GNN | −1.0% |
| `hub_k4_seek50` | **1.04M** | 1.12M | GNN | −7.4% |
| `hub_k4_seek80` | 1.42M | **1.08M** | MLP | +31.1% |
| `hub_k6_seek30` | **846k** | 945k | GNN | −10.5% |
| `hub_k6_seek50` | 886k | **902k** | MLP | −1.8% |
| `hub_k6_seek80` | 951k | **928k** | MLP | +2.5% |

**9-hub sums:** GNN **13.41M** · MLP **13.30M (−0.8%)** · wins **GNN 5/9 · MLP 3/9** (atomic21 rejected everywhere).

### workload-125-225 (datalab job 478100)

**Sweep:** `tiered_hub_gnn_mlp_125225_20260610/results/` · SLURM `tiered_hub_dim22_gpu.sbatch` · 22 GPU tasks (11×2)

| Config | GNN dim22 | MLP dim22 | Knative | Best |
|--------|----------:|----------:|--------:|------|
| `default_20_20_p50` | 25.99M | **21.18M** | pending | MLP |
| `05_sparse_40_40_p25` | 12.44M | **7.76M** | pending | MLP |
| `hub_k2_seek30` | **TIMEOUT** | **9.35M** | pending | MLP* |
| `hub_k2_seek50` | 19.15M | **9.76M** | pending | MLP |
| `hub_k2_seek80` | 17.19M | **17.08M** | pending | MLP |
| `hub_k4_seek30` | **1.39M** | 2.03M | pending | GNN |
| `hub_k4_seek50` | **1.51M** | 1.64M | pending | GNN |
| `hub_k4_seek80` | 3.87M | **3.86M** | pending | MLP |
| `hub_k6_seek30` | **1.16M** | 1.42M | pending | GNN |
| `hub_k6_seek50` | **1.19M** | 1.41M | pending | GNN |
| `hub_k6_seek80` | **1.28M** | 1.92M | pending | GNN |

**Status (2026-06-10):** ML **21/22** done (GNN **10/11** — `hub_k2_seek30` GNN failed 3600s timeout; retry 478309 pending) · MLP **11/11** · Knative **0/11** (478314 pending, `--gres=gpu:l40s:1`).

**Paired hub wins (10 configs with both ML):** **GNN 5 · MLP 5** — GNN on **k≥4** moderate seek (k4 seek30/50, k6 all); MLP on **controls**, **k2**, k4 seek80 (~tie).

**Δ% GNN vs MLP (hubs only, where both exist):**

| k | seek30 | seek50 | seek80 |
|---|--------|--------|--------|
| 2 | — | +96% | +0.6% |
| 4 | **−31%** | **−8%** | +0.2% |
| 6 | **−19%** | **−15%** | **−33%** |

**Takeaway:** Under 125-225 load, **k_core≥4 + seek 30–50%** favors GNN (GIN batch coupling across shared hub platforms); **k=2** or **uniform controls** favor MLP; GNN volatile (k2 seek50 **19M** vs MLP **9.8M**). **GNN-favorable design:** larger `k_core` (6–8), seek ∈ [35%,55%], avoid k=2 at heavy workload. **Superseded for clean phase-boundary compare by bipartite v1** (k≥4 only, asymmetric 5/30ms, b=4 enforced).

**Scripts:** `transfer_tiered_hub_dim22_125225_{to,from}_datalab.sh` · `tiered_hub_dim22_gpu.sbatch` · `tiered_hub_knative_125225.sbatch` · `sketch_topologies.py` → `simulation_data/topology_sketches/`.

---

## Bipartite coordination v1 — dim22 GNN vs MLP vs Knative (125-225)

**Purpose:** Clean phase-boundary sweep — **k∈{4,6,8} × seek∈{35,50,65}%**, no k=2, no uniform controls.

| Field | Value |
|-------|-------|
| Sweep | `sweep_bipartite_coordination_v1/` |
| Workload | `workload-125-225.json` (562k tasks) |
| Latency | **5ms core / 30ms periphery** (asymmetric) |
| GNN batch | **b=4 fixed** (`GNN_BATCH_SIZE=4`; >4 fails loud) |
| Policies | `gnn_dim22` · `mlp_dim22` (Regime A) · `knative` (Regime B, per-arrival) |
| Models | `near-rtt-v2-dim14-ce-only.pt` + `INFERENCE_FEATURE_LAYOUT=dim22` · `batch_edge_mlp.pt` |
| Status | **9/9 complete** all three (datalab 478411+478454; local Knative) |

| Config | k | seek | GNN | MLP | Knative | GvsM | Best |
|--------|--:|-----:|----:|----:|--------:|-----:|------|
| `hub_k4_seek35` | 4 | 35% | 3.20M | **2.99M** | 3.30M | +6.7% | MLP |
| `hub_k4_seek50` | 4 | 50% | 2.92M | 3.34M | **2.52M** | −12.4% | Knative |
| `hub_k4_seek65` | 4 | 65% | **2.95M** | 3.32M | 3.09M | −11.2% | GNN |
| `hub_k6_seek35` | 6 | 35% | **1.53M** | 1.76M | 2.29M | −12.8% | GNN |
| `hub_k6_seek50` | 6 | 50% | 1.66M | 3.46M | **1.45M** | −52.1% | Knative |
| `hub_k6_seek65` | 6 | 65% | 1.76M | 2.13M | **1.68M** | −17.5% | Knative |
| `hub_k8_seek35` | 8 | 35% | **1.28M** | 1.44M | 1.30M | −11.4% | GNN |
| `hub_k8_seek50` | 8 | 50% | **1.29M** | 1.52M | 1.32M | −15.1% | GNN |
| `hub_k8_seek65` | 8 | 65% | **1.31M** | 1.52M | 1.41M | −13.7% | GNN |

**9-config sums:** GNN **17.90M** · MLP **21.49M** · Knative **18.37M**

**Win counts:** GNN vs MLP **8/9** (MLP only `k4_seek35`) · 3-way best **GNN 5 · Knative 3 · MLP 1**

**Regime labels:** k=4 → k=b marginal · k=6,8 → k>b coordination

**Takeaways:**
- **GNN dominates MLP** once k>b (6/6 configs) and at k=4 seek65; MLP only wins k=b seek35.
- **Knative competitive** at k=4 seek50 and k=6 seek50/65 — especially where MLP collapses (k6 seek50 **3.46M**).
- **Phase boundary confirmed:** GNN advantage when **k > b=4**; at k=b results are mixed (three-way toss-up).
- **Do not mix** with tiered-hub 125-225 table above (symmetric 5ms, includes k=2 + controls).

**Compare:** `pipenv run python3 scripts_cosim/important/compare_bipartite_coordination_sweep.py --sweep-dir simulation_data/normal_sim_sweeps/sweep_bipartite_coordination_v1`

**Scripts:** `prepare_bipartite_coordination_sweep.sh` · `transfer_bipartite_coordination_{to,from}_datalab.sh` · `submit_bipartite_coordination_partition.sh` · `run_bipartite_knative_local.sh` · datalab `bipartite_coordination_gpu.sbatch`

---

## Bipartite v2 — wssm + node_disk_v2 (125-225, 2026-06-14)

> **CANONICAL 3-way table (2026-06-14).** After MLP train/serve parity fix (batch cache dim22), **MLP beats GNN 8/9** — overturns prior "GIN necessary on hub" claim from broken/seq MLP runs.

**Purpose:** Phase-boundary grid with warmth-skew-merged models + `node_disk_v2`. Same 9 hub configs as bipartite v1.

| Field | Value |
|-------|-------|
| Workload | `workload-125-225.json` (562k tasks) |
| Physics | `HEROSIM_WARMTH_PHYSICS=node_disk_v2` |
| GNN | `near-rtt-v2-warmth-sparse-skew-merged-dim14-ce-only.pt` · dim22 · b=4 |
| **MLP (deploy)** | `batch_edge_mlp_warmth_sparse_skew_merged_dim22_batchcache.pt` · dim22 · batch cache |
| Knative ref | `sweep_bipartite_coordination_v1/` (legacy physics batch-loop) |

### ★ Canonical results — GNN vs MLP batchcache vs Knative

| Config | GNN wssm | **MLP-bc** | Kn v1 | bc/Kn | GNN/Kn | Best |
|--------|--------:|-----------:|------:|------:|-------:|------|
| `hub_k4_seek35` | 2.47M | **0.98M** | 3.30M | −70% | −25% | **MLP-bc** |
| `hub_k4_seek50` | 1.19M | **0.63M** | 2.52M | −75% | −53% | **MLP-bc** |
| `hub_k4_seek65` | 1.53M | **0.77M** | 3.09M | −75% | −51% | **MLP-bc** |
| `hub_k6_seek35` | 0.77M | **0.63M** | 2.29M | −72% | −67% | **MLP-bc** |
| `hub_k6_seek50` | 1.56M | **0.68M** | 1.45M | −53% | +8% | **MLP-bc** |
| `hub_k6_seek65` | **0.73M** | 0.74M | 1.68M | −56% | −56% | GNN |
| `hub_k8_seek35` | 0.62M | **0.54M** | 1.30M | −58% | −52% | **MLP-bc** |
| `hub_k8_seek50` | 0.56M | **0.55M** | 1.32M | −58% | −58% | **MLP-bc** |
| `hub_k8_seek65` | 0.59M | **0.55M** | 1.41M | −61% | −58% | **MLP-bc** |

**9-config sums:** **MLP-bc 6.09M** · GNN 10.02M (−39% vs GNN) · Kn 18.37M (−67% vs Kn)

**Win counts:** **MLP-bc 8/9** · GNN 1/9 (`k6_seek65` only, 0.73M vs 0.74M tie) · Kn 0/9

### MLP training ladder (same 9 configs — documents the bug hunt)

| MLP variant | Train cache | Platform encoding at train | 9× sum | Wins | Verdict |
|-------------|-------------|---------------------------|-------:|-----:|---------|
| ce_reduced | seq | raw queue, no is_warm/is_cold (11-d) | **939.76M** | 0/9 | **REJECT** |
| dim22 seq-fix | seq | raw queue + is_cold (22-d) | **49.21M** | 0/9 | train/serve mismatch |
| **dim22 batchcache ★** | **batch (GNN cache)** | **norm queue + shared_fate (22-d)** | **6.09M** | **8/9** | **SHIP hub** |

**Why batchcache wins:** Same co-sim corpus and **same platform feature encoding as GNN train + dim22 inference** (`prepare_graphs_cache.py`: norm queue dim 7, shared_fate dim 8, usage_ratio dim 13). Seq cache uses raw queue + is_cold — mismatches inference even with 22-d edge features.

**GNN vs MLP-bc implication:** GIN batch coupling is **not required** for hub coordination when MLP has feature parity. Prior GNN 8/9 vs MLP 0/9 was an **MLP encoding artifact**, not architecture.

**Regression warning:** Both wssm models regress on legacy all7 (+45% vs dim14-ce). Hub deploy only under node_disk_v2.

**Sweeps:** GNN `bipartite_v2_skew_merged_20260614/` · MLP-bc `bipartite_v2_mlp_dim22_batchcache_20260614/` (job 482182)

**Training:** `train_mlp_dim22_from_batch.py` · `train_mlp_v2_warmth_sparse_skew_merged_dim22_batchcache.py` · val_edge_acc **0.640**

**Compare:** `pipenv run python3 scripts_cosim/important/compare_mega_matrix.py --group 2`

---


### seq_reforward overlay (k6 only, `GNN_DECODE_MODE=seq_reforward`)

**Status:** 2/2 complete (datalab job 478527 + local T4 seek65 agree). Only k6 configs run so far (~90–111 min wall each vs ~15 min argmax).

| Config | GNN argmax | GNN seq_reforward | Knative | srf vs GNN | srf vs Kn | Best |
|--------|----------:|------------------:|--------:|-----------:|----------:|------|
| `hub_k6_seek50` | 1.66M | 1.55M | **1.45M** | **−6.5%** | +7.1% | Knative |
| `hub_k6_seek65` | 1.76M | **1.63M** | 1.68M | **−7.4%** | **−3.3%** | **GNN-srf** |

**Decode stats (562k tasks, b=4):**

| Config | Mode | RTT | decode ms/batch (mean) | decode wall total | collision batch rate | qvm p95 |
|--------|------|----:|-----------------------:|------------------:|---------------------:|--------:|
| k6 seek50 | argmax | 1.66M | 0.19 | ~30s | 30.3% | 123 |
| k6 seek50 | seq_reforward | 1.55M | 36.7 | ~97 min | 14.7% | 0 |
| k6 seek65 | argmax | 1.76M | 0.11 | ~17s | 32.1% | 118 |
| k6 seek65 | seq_reforward | 1.63M | 33.2 | ~88 min | 14.8% | 0 |

**Why seq_reforward is slow:** per batch (b=4), runs **4 full GNN forwards** + queue-feature refresh on all platforms (~208) after each task pick — argmax runs **1 forward** + cheap argmax loop. ~158k batches → ~632k forwards vs ~158k (**~4× forwards × ~8ms/forward ≈ 200× decode wall**). Inference wall is **not** in SimPy `total_rtt`.

**Takeaway:** seq_reforward fixes k6 seek65 vs Knative (−3.3%) and halves intra-batch collisions; seek50 still loses to Knative (+7.1%). Next Tier-1: uniqueness mask + collision repair (cheaper than full reforward).

**Result files:** `{config}_gnn_dim22_seq_reforward.json` in `sweep_bipartite_coordination_v1/results/` · runner `run_bipartite_seq_reforward_one.sh` · datalab `seq_reforward_probe.sbatch`

---

## atomic21 tabular (21-dim)

| Item | Value |
|------|-------|
| Model | `models/tabular/batch_edge_mlp_atomic21.pt` |
| Cache | `graphs_cache_gnn_datasets_4tasks_atomic21_seq` |
| Layout | `INFERENCE_FEATURE_LAYOUT=atomic21` |
| Skew sweep script | `run_mlp_atomic21_skew_sweep_nohup.sh` · pairs with `run_gnn_near_rtt_v2_atomic21_ce_only_skew_sweep_nohup.sh` |

Skew-4 co-located; tiered-hub 100-100: `tiered_hub_gnn_mlp_20260610/` · 125-225: `tiered_hub_gnn_mlp_125225_20260610/`.

---

## How to run

### Standard dim14 sweep (5-config)

```bash
bash scripts_cosim/important/run_gnn_near_rtt_v2_5cfg_sweep_common.sh \
  simulation_data/normal_sim_sweeps/my_run_$(date +%Y%m%d) \
  models/near-rtt-v2-dim14-ce-only.pt my_label argmax
```

### Skew-4 local (dim14 + dim22 MLP)

```bash
export INFERENCE_FEATURE_LAYOUT=dim22
bash scripts_cosim/important/run_dim14_old_models_skew4_sweep.sh \
  simulation_data/normal_sim_sweeps/dim14_old_models_skew4_$(date +%Y%m%d)
```

### Bipartite coordination v1 (125-225, 9 hub configs)

```bash
bash scripts_cosim/important/prepare_bipartite_coordination_sweep.sh
bash scripts_cosim/transfer_bipartite_coordination_to_datalab.sh
bash scripts_cosim/datalab/submit_bipartite_coordination_partition.sh GPU-a40 gpu:a40:1 a40
# Local Knative (Regime B): bash scripts_cosim/run_bipartite_knative_local.sh
# Pull + compare: bash scripts_cosim/transfer_bipartite_coordination_from_datalab.sh
```

### Datalab tiered-hub dim22 (125-225, 11 configs)

```bash
bash scripts_cosim/transfer_tiered_hub_dim22_125225_to_datalab.sh
# GNN+MLP: sbatch scripts_cosim/datalab/tiered_hub_dim22_gpu.sbatch  # 22 tasks
# Knative: sbatch scripts_cosim/datalab/tiered_hub_knative_125225.sbatch  # 11 tasks, --gres=gpu:l40s:1
# GNN retry: bash scripts_cosim/datalab/submit_tiered_hub_dim22_retry.sh  # TIMEOUT=7200
# Pull: bash scripts_cosim/transfer_tiered_hub_dim22_125225_from_datalab.sh
```

### Datalab skew-4 (125-225)

```bash
SWEEP_DIR=simulation_data/normal_sim_sweeps/dim14_old_models_skew4_125225_20260610 \
WORKLOAD=data/nofs-ids/traces/workload-125-225.json \
bash scripts_cosim/transfer_dim14_old_models_skew4_to_datalab.sh
# GNN+MLP: sbatch scripts_cosim/datalab/dim14_old_models_skew4.sbatch
# Knative: sbatch scripts_cosim/datalab/skew4_knative.sbatch
# Pull: transfer_dim14_old_models_skew4_from_datalab.sh
```

Fix CRLF on Windows-saved scripts: `sed -i 's/\r$//' <script.sh>`.

---

## Result paths (active)

Base: `simulation_data/normal_sim_sweeps/`

| Label | Path |
|-------|------|
| **★ dim14-ce 7/7** | `gnn_near_rtt_v2_dim14_ce_only_20260609/results/` |
| Reviewer Triangle all7 | `reviewer_triangle_all7_20260609/results/` |
| Skew-4 100-100 | `dim14_old_models_skew4_20260610/results/` |
| Skew-4 125-225 | `dim14_old_models_skew4_125225_20260610/results/` |
| Tiered-hub 100-100 | `tiered_hub_gnn_mlp_20260610/results/` |
| Tiered-hub 125-225 | `tiered_hub_gnn_mlp_125225_20260610/results/` |
| **warmth_v2 skew3** | `warmth_v2_physics_skew3_20260611/` · `warmth_v2_ce_reduced_skew3_20260611/` |
| **contention_v3 live gate** | `contention_v3_live_gate_20260620/results/` (datalab) |
| **Strategic merge WSSM** | `strategic_merge_wss_live_gate_20260616/results/` · `compare.txt` |
| **Strategic merge contention** | `strategic_merge_contention_live_gate_20260616/results/` · `compare.txt` |
| **Weighted merge contention** | `merged_contention_weighted_live_gate_20260616_105500/results/` · `compare.txt` |
| **★ Bipartite v1 9/9** | `sweep_bipartite_coordination_v1/results/` |
| **★★ Bipartite v2 wssm 8/9** | `bipartite_v2_skew_merged_20260614/results/` (node_disk_v2) |
| Mega compare all7 | `mega_compare_all7_20260614/results/` (6 new model variants) |
| Skew3 full gate wssm | `skew3_full_gate_20260614/results/` (node_disk_v2, 3 configs) |
| Skew4 new models wssm | `skew4_new_models_20260614/results/` (node_disk_v2, 125-225) |
| **MLP dim22 fix (seq cache)** | `bipartite_v2_mlp_dim22_fix_20260614/results/` (49M — superseded) |
| **★★ MLP dim22 batchcache 8/9** | `bipartite_v2_mlp_dim22_batchcache_20260614/results/` (6.09M, hub deploy) |
| Topology sketches | `topology_sketches/*.png` |
| Knative 7/7 | `knative_network_20260606_192413/results/` |
| HeroCache 7/7 | `herocache_network_20260606_205112/results/` |

**Runtime (workload-100-100):** GNN GPU ~6–12 min/config · Knative/HRC ~2 min · full 7 GNN ~80 min. **125-225:** GNN/MLP uniform ~20–40 min/config on datalab L40S; Knative ~15–30 min.

---

## Archive — classical & weak baselines

### Random + Round-robin (7/7, workload-100-100)

**Sweep:** `random_rr_3cfg_20260609/results/` · fair RR fix 2026-06-09 (per-arrival `RoundRobinNetworkScheduler` on Knative stack).

| Config | CE-only | Knative | Random | RoundRobin | Δ Rand vs CE | Δ RR vs CE |
|--------|--------:|--------:|-------:|-----------:|-------------:|-----------:|
| `default` | 4.14M | 5.59M | 20.54M | 7.45M | +396% | +80% |
| `00`–`05` | pre-fix CE | … | +250–455% | +78–124% | … | … |

**CE-only 7/7** vs Random/RR/Knative. All Δ vs **pre-fix CE** (`gnn_near_rtt_v2_dim14_ce_only_20260609/`). Random catastrophic; RR worse than Knative everywhere.

---

## Archive — superseded dim14 GNN variants

Do **not** use for active compare unless ablating. All workload-100-100 unless noted.

### dim14-full (`near-rtt-v2-dim14-1060.pt`, CE + ranking)

| Config | dim14-ce | dim14-full | Winner |
|--------|----------|------------|--------|
| `default` | **4.14M** | 11.62M | ce |
| `01`/`02`/`04` | ~tie | slight edge | full |
| `05` | **2.45M** | 2.61M | ce |

Sweep: `gnn_near_rtt_v2_dim14_1060_20260608/results/`. **Rejected** for deploy (sparse collapse pre-fix; post-fix `default` still +27% vs ce-only).

### Track B r030 (`near-rtt-v2-dim14-ce-init-r030.pt`)

Post queue-map 3cfg sum **+1.9%** vs ce-only — **rejected**. Sweep: `dim14_3model_3cfg_queuefix_20260609/results/`.

### dim14-full + LQB (λ=1.5)

7/7 sum **+3.7%** vs ce-only — **rejected**. Sweep: `gnn_near_rtt_v2_dim14_1060_lqb15_20260609_100843/results/`.

### Post queue-map fix — 3-model × 3-config (2026-06-09)

| Config | ce-only | dim14-1060 | r030 | Winner |
|--------|--------:|-----------:|-----:|--------|
| `default` | **4.31M** | 5.48M | 4.45M | ce-only |
| `02` | **1.91M** | 1.94M | 1.95M | ce-only |
| `04` | 1.80M | **1.78M** | 1.78M | 1060 |
| **sum** | **8.03M** | 9.20M | 8.18M | ce-only |

Pre-fix anchor RTTs ≠ post-fix Option 1 inference — do not mix without relabeling.

### Option 1 inference parity (code shipped, no retrain)

Queue-map + temporal dims at batch start on same ce-only checkpoint. 3cfg sum **+5.7%** vs anchor — no RTT lift. Sweep: `gnn_near_rtt_v2_dim14_ce_only_opt1_frozen_3cfg_20260609_135559/results/`.

### dim14 decode modes (reference)

| Mode | Placement | Queue roll-forward |
|------|-----------|-------------------|
| `argmax` | per-task argmax | yes |
| `frozen` | per-task argmax | no (telemetry only for CE-only) |

For dim14-ce, **argmax ≡ frozen** on platform picks; qvm p95 differs, not placement.

### dim14-ce decode telemetry (argmax)

| Config | qvm p95 | RTT |
|--------|--------:|----:|
| `default` | 408 | 4.14M |
| `02` | 118 | 1.84M |
| `05` | 144 | 2.45M |

Live RTT tracks **qvm p95**, not median.
