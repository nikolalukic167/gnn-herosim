# Results & Benchmarks

> Two regimes. **Part A** tables below use `total_rtt` on 100-100 / 125-225.
> **Part B (RQ9)** is a constructed N=12 FilterStore cell — max-burst elapsed, not `total_rtt`.
> See `paper/case_study_regime_b.md`. Do not mix columns.
>
> Publishable simulation results for the dim14-CE-only GNN anchor and Regime A dim22 tabular peers.
> Default tables use `workload-100-100` (~201k tasks), seed 42, unless noted.
> **Stress workload:** `workload-125-225` (~562k tasks) for skew-4 and **bipartite v1** (primary hub stress).
> **Inference stack:** sweeps labeled **post queue-map fix** use Option 1 shipped inference (full-infra queue map, `platform.queue_length()`, real temporal dims). Pre-fix sweeps (e.g. `gnn_near_rtt_v2_dim14_ce_only_20260609/`) used the old partial queue snapshot — do not mix RTT numbers across stacks.
> **Eval checkpoint:** `near-rtt-v2-dim14-ce-only.pt` (1060 cache) unless noted. **Warmth train:** `near-rtt-v2-warmth-dim14-ce-only.pt` on merged 824-cache — skew3 live gate below.
> **RQ3 sealed answer:** 873/v5.5 holdout below — **not** the pre-fix 7/7 table.
> Archive: `archived/legacy_results.md` — 150-150 benchmarks, seqblend, LQB, old models.

---

## RQ3 sealed live holdout (canonical Part A — rebuilt 2026-08-13 from artifact)

**Sweep:** `contention_v2_873_v5.5_sealed_holdout_20260806/compare.json`  
**Models:** GNN `near-rtt-v2-contention-v2-873-v5.5-dim14-ce-only.pt` · MLP dim22 batchcache · Knative  
**Workload:** 125-225 · `node_disk_v2` · unused topologies × seeds 42–46 · **20 paired cells**  
**Do not cite n=899** (retained corpus is **873**). Do not cite RQ1 7/7 as the RQ3 answer (7/7 is pre queue-map-fix; post-fix coverage is 3/7).

| config | Kn mean | GNN mean | MLP mean | wins (G/M/K) |
|---|---:|---:|---:|---|
| balanced_p50 | 5.98M | 6.71M | **4.58M** | 0/5/0 |
| balanced_p60 | **4.66M** | 5.28M | 4.94M | 0/1/4 |
| client_heavy_p50 | 8.30M | 8.15M | **5.33M** | 0/5/0 |
| server_heavy_p50 | 3.67M | 3.66M | 3.77M | 1/2/2 |
| **paired cells** | sum 113.0M | sum 119.1M (1.05×Kn) | sum 93.1M (0.82×Kn) | **1 / 13 / 6** |

**Claim:** offline GIN>MLP on coupled 873 labels is real; it does **not** transfer to this sealed live set.

---

## RQ9 case study (Part B — not a `/compare` scoreboard)

Primary = **max burst elapsed**. Cell = `oracle_split_v1`, N=12, `platform_reuse_v1`.

| Policy | Primary | vs oracle |
|---|---:|---:|
| Oracle parallel | **31.65s** | — |
| `ect_pull` teacher | **31.65s** | 0.00s |
| Distill + `seq_reforward_pull` | **31.66s** | +0.00s |
| Distill argmax (no ledger) | 125.28s | +94s |
| CE dim16 argmax | 125.28s† | +94s |
| Hard-CE distill (α=0) | 125.28s | +94s |
| Warm/busy harvest distill | 375.87s | +344s |

† This eval; some post-parity CE argmax runs ~93.95s — still pile band.  
Full ladder + limitations: `paper/case_study_regime_b.md`.  
Artifact: `regime_b_phase3_ect_pull_distill_eval_multiseed_cold/summary.json`.

**Multi-cell (same physics, 2026-08-13):** distill stays at 31.66s on zero-jitter / latency-shift / dual-burst. **Jitter 0.5s → 125.24s pile; jitter 2.0s → 62.63s.** Teacher stays ~31.65s. Artifact: `regime_b_multicell_20260813/summary.json`.

---

## Post Queue-Map Fix: 3-Model × 3-Config Gate (canonical model comparison)

**Inference:** Option 1 queue-map fix (full-infra snapshot at batch start · real temporal dims · p90 norm over all graph platforms) · **argmax** · seed 42 · configs `default` / `02` / `04`.

**Sweep dir:** `dim14_3model_3cfg_queuefix_20260609/results/` · wall ~111 min (9 runs)

| Model | Checkpoint | `default` | `02` | `04` | 3cfg sum | Wins |
|---|---|---:|---:|---:|---:|---|
| **CE-only (anchor)** | `near-rtt-v2-dim14-ce-only.pt` | **4.31** | **1.91** | 1.80 | **8.03M** | **2/3** |
| Pairwise ranking | `near-rtt-v2-dim14-1060.pt` | 5.48 | 1.94 | **1.78** | 9.20M | 1/3 (`04`) |
| Track B r030 | `near-rtt-v2-dim14-ce-init-r030.pt` | 4.45 | 1.95 | 1.78 | 8.18M (+1.9% vs CE) | 0/3 |

RTT in millions of seconds unless noted.

**Key findings (post-fix):**
- **CE-only remains the deployable anchor** — lowest 3cfg sum; wins `default` and `02`.
- **Ranking sparse collapse is largely a train/inference queue-input artifact, not purely objective failure:** with fixed queue map, ranking `default` drops from **11.62M (pre-fix)** to **5.48M (−53%)** — but still **+27% vs CE-only** on `default`; ranking wins `04` only (−1.3% vs CE).
- **Track B r030 rejected:** CE init → regret λ=0.30 does not beat CE-only on 3cfg sum (+1.9%); not shippable.
- **CE-only RTT rises +4–8% vs pre-fix 7/7 anchor** on the same checkpoints when inference inputs are corrected (`default` 4.14→4.31M, `02` 1.84→1.91M, `04` 1.67→1.80M) — no deploy lift from the fix alone.

---

## Primary Benchmark: 7-Config Normal Sim Sweep (pre queue-map fix)

**Model:** `near-rtt-v2-dim14-ce-only.pt` (GNN anchor, 14-dim platform features)
**Sweep dir:** `gnn_near_rtt_v2_dim14_ce_only_20260609/`
**Decode mode:** `argmax` (per-task marginal, decomposed from single frozen inference graph)
**Note:** Pre-Option1 inference (partial queue snapshot). For cross-model gates use the post-fix 3×3 table above.

### Infrastructure Configurations

| Config | Description | Nodes | Conn prob |
|---|---|---|---|
| `default` | 20×20 balanced | 20+20 | 0.50 |
| `00` | 30×30 balanced | 30+30 | 0.50 |
| `01` | 40×40 balanced | 40+40 | 0.50 |
| `02` | 40×40 dense | 40+40 | 0.60 |
| `03` | 30×30 balanced | 30+30 | 0.50 |
| `04` | 40×40 dense | 40+40 | 0.60 |
| `05` | 40×40 sparse | 40+40 | 0.25 |

### Total RTT (millions of seconds)

| Policy | default | 00 | 01 | 02 | 03 | 04 | 05 |
|---|---|---|---|---|---|---|---|
| **GNN CE-only (anchor)** | **4.14** | **3.00** | **2.25** | **1.84** | **2.54** | **1.67** | **2.45** |
| Knative | 5.59 | 3.94 | 2.82 | 2.31 | 3.33 | 2.19 | 3.35 |
| Fair HRC | 5.00 | 3.45 | 2.82 | 2.80 | 2.92 | 2.59 | 2.85 |
| Random network‡ | 20.54 | 15.47 | 9.54 | 6.43 | 11.63 | 7.21 | 13.62 |
| Round-robin‡ | 7.45 | 5.34 | 4.27 | 3.49 | 4.94 | 3.73 | 4.84 |
| XGBoost batch† | 886.9k | — | — | — | — | — | — |

† 30k task subset only; full results in Regime A section below.
‡ 7/7 sweep `random_rr_3cfg_20260609/`; fair per-arrival stack (see Weak Baselines section). Δ vs **pre-fix** CE-only (`gnn_near_rtt_v2_dim14_ce_only_20260609/`): Random +250–455%, RR +78–124%, Knative +25–36%.

**GNN CE-only: 7/7 wins vs Knative + HRC + Random + RoundRobin on the pre-fix stack.**
RQ1 status is **PARTIAL** — canonical post-fix coverage is 3/7. Do not use this table as the RQ3 live answer.

---

## Weak Baselines: Random + Round-Robin (7-config, seed 42)

**Sweep:** `random_rr_3cfg_20260609/results/` · per-arrival · network-aware · Knative autoscale stack

| Config | CE-only | Knative | Random | RoundRobin | Δ vs CE (Kn) | Δ vs CE (Rand) | Δ vs CE (RR) |
|--------|--------:|--------:|-------:|-----------:|-------------:|---------------:|-------------:|
| `default` | 4.14M | 5.59M | 20.54M | 7.45M | +35% | +396% | +80% |
| `00` | 3.00M | 3.94M | 15.47M | 5.34M | +32% | +416% | +78% |
| `01` | 2.25M | 2.82M | 9.54M | 4.27M | +25% | +325% | +90% |
| `02` | 1.84M | 2.31M | 6.43M | 3.49M | +25% | +250% | +90% |
| `03` | 2.54M | 3.33M | 11.63M | 4.94M | +31% | +357% | +94% |
| `04` | 1.67M | 2.19M | 7.21M | 3.73M | +31% | +332% | +124% |
| `05` | 2.45M | 3.35M | 13.62M | 4.84M | +36% | +455% | +98% |

All Δ use **pre-fix CE-only** denominators from `gnn_near_rtt_v2_dim14_ce_only_20260609/` (verified 2026-06-10). Knative `default` from `baseline_default_100100/`; other configs from `knative_network_20260606_192413/`.

**Δ% = `(policy − CE-only) / CE-only × 100`** — positive means worse than CE anchor.

**Fairness:** `random_network` always used Knative network stack. `roundrobin` was refactored (2026-06-09) from batched 5×100ms to per-arrival `KnativeScheduler` subclass with round-robin placement — matches Knative/HRC scheduling regime.

**Paper role:** Random and round-robin are sanity-check weak baselines; CE-only wins 7/7 vs both. Knative remains the primary classical competitor (+25–36% vs CE).

---

Same 14-dim cache and seed. **Pre-fix inference** (7-config sweep); post-fix 3×3 gate in section above.

| Objective | default | 00 | 01 | 02 | 03 | 04 | 05 | default qvm p95 |
|---|---|---|---|---|---|---|---|---|
| **CE-only (deployed)** | **4.14** | **3.00** | **2.25** | **1.84** | **2.54** | **1.67** | **2.45** | **408** |
| Pairwise ranking (pre-fix) | 11.62 | 4.90 | **2.25** | **1.80** | **2.47** | **1.74** | 2.56 | **2857** |
| Pairwise ranking (post-fix, 3cfg) | 5.48 | — | — | 1.94 | — | **1.78** | — | — |

**Key finding (pre-fix):** Pairwise combo-sum ranking achieves lower RTT on 4/7 dense-connectivity configs (0.2–2.7% margin) but causes catastrophic hot-spotting on `default` (+181% vs CE). qvm p95 2857 vs CE 408.

**Post-fix update:** Correct queue-map inputs remove most of ranking's sparse `default` collapse (11.62→5.48M) but CE-only still wins the 3cfg sum; ranking remains +27% on `default` vs CE. The joint-training/marginal-inference mismatch still dominates deploy quality — queue-input correction alone does not make ranking shippable.

---

## Decode Diagnostics: `chosen_queue_vs_min` (qvm)

The qvm metric measures, per task, the queue depth of the chosen platform minus the minimum queue among all feasible candidates at decode time. Computed from the `*.decode_stats.json` sidecar produced by every GNN simulation run.

| Model | default qvm p95 | Interpretation |
|---|---|---|
| CE-only | 408 | Acceptable — model occasionally picks non-minimum queues for exec/network quality |
| Ranking (argmax) | 2857 | Hot-spotting: model consistently picks overloaded platforms |

**qvm p95 > ~500 is the empirical threshold for hot-spotting.** The decode stats sidecar is therefore a more reliable pre-deployment signal than any offline regression metric.

---

## Regime A: Reviewer Triangle — GNN vs MLP vs XGB (all7, workload-100-100)

**Sweep:** `reviewer_triangle_all7_20260609/results/` · dim14-ce GNN anchor · MLP `batch_edge_mlp.pt` (dim22) · XGB `batch_edge_ranker.json` · seed 42

| Config | GNN CE | MLP dim22 | XGB | MLP Δ vs GNN |
|--------|-------:|----------:|----:|-------------:|
| `default` | 4.14M | **3.75M** | 10.89M | **−9.6%** |
| `00` | 3.00M | 3.05M | 5.34M | +1.7% |
| `01` | 2.25M | 2.30M | 2.73M | +2.5% |
| `02` | 1.84M | 1.88M | 2.06M | +1.9% |
| `03` | 2.54M | **2.49M** | 3.49M | **−2.1%** |
| `04` | 1.67M | 1.82M | 1.93M | +9.1% |
| `05` | 2.45M | 2.68M | 3.91M | +9.4% |

**7-config sums:** GNN **17.89M** · MLP **17.96M (+0.4%)** · XGB **30.34M (+70%)** · MLP wins **2/7** vs GNN.

**Takeaway:** At standard workload, dim22 MLP ≈ GNN; XGB batch fails badly. Neither tabular model beats GNN on sum at 100-100 under uniform/sparse 7-config sweep — but **does not imply GNN wins on hub stress** (see Bipartite v1).

---

## Primary Hub Stress: Bipartite Coordination v1 (125-225)

> **Canonical RQ3 phase-boundary experiment.** Supersedes tiered-hub for clean $k$ vs $b$ claims. Asymmetric **5ms core / 30ms periphery**; $k \in \{4,6,8\}$; seek $\in \{35,50,65\}\%$; batch **$b=4$** fixed.

**Sweep:** `sweep_bipartite_coordination_v1/results/` · policies: gnn_dim22 · mlp_dim22 · knative · checkpoint `near-rtt-v2-dim14-ce-only.pt`

| Config | k | seek | GNN | MLP | Knative | Best |
|--------|--:|-----:|----:|----:|--------:|------|
| `hub_k4_seek35` | 4 | 35% | 3.20M | **2.99M** | 3.30M | MLP |
| `hub_k4_seek50` | 4 | 50% | 2.92M | 3.34M | **2.52M** | Kn |
| `hub_k4_seek65` | 4 | 65% | **2.95M** | 3.32M | 3.09M | GNN |
| `hub_k6_seek35` | 6 | 35% | **1.53M** | 1.76M | 2.29M | GNN |
| `hub_k6_seek50` | 6 | 50% | 1.66M | 3.46M | **1.45M** | Kn |
| `hub_k6_seek65` | 6 | 65% | 1.76M | 2.13M | **1.68M** | Kn |
| `hub_k8_seek35` | 8 | 35% | **1.28M** | 1.44M | 1.30M | GNN |
| `hub_k8_seek50` | 8 | 50% | **1.29M** | 1.52M | 1.32M | GNN |
| `hub_k8_seek65` | 8 | 65% | **1.31M** | 1.52M | 1.41M | GNN |

**9-config sums:** GNN **17.90M** · MLP **21.49M** · Knative **18.37M**

**Win counts:** GNN vs MLP **8/9** (MLP only `k4_seek35`) · 3-way best **GNN 5 · Knative 3 · MLP 1**

**Regime:** $k > b$ → GNN wins 6/6 vs MLP; $k = b$ mixed; Knative competitive at k4 seek50 and k6 seek50/65 (MLP collapse k6 seek50 **3.46M**).

**seq_reforward overlay (k6 only):** seek65 **1.63M** (−3.3% vs Kn 1.68M); seek50 still loses to Kn (+7.1%); ~200× decode wall vs argmax.

**Mechanism note:** GNN and MLP share the same argmax decode loop; GNN advantage here is **GIN batch coupling** (platform embeddings encode joint batch demand), not decode re-planning after collisions.

---

## Merged sparse-merged ce-reduced Hub9 Live Gate (125-225, 2026-06-11)

> **Same topology grid as bipartite v1** — not comparable RTT to dim22 table above without noting **different models + `node_disk_v2` physics**. Tests merged from-scratch **ce-reduced** checkpoints on hub stress before ship.

**Sweep:** `warmth_sparse_merged_ce_reduced_hub9_20260611/results/` · datalab job **480403** (27/27) · seed 42

| Field | Value |
|-------|-------|
| Workload | `workload-125-225.json` (~562k tasks) |
| Topology | k∈{4,6,8} × seek∈{35,50,65}% · 5ms core / 30ms periphery |
| Physics | `HEROSIM_WARMTH_PHYSICS=node_disk_v2` |
| GNN | `near-rtt-v2-warmth-sparse-merged-ce-reduced.pt` · `INFERENCE_FEATURE_LAYOUT=ce_reduced` · b=4 |
| MLP | `batch_edge_mlp_warmth_sparse_merged_ce_reduced.pt` (11-d, verified) · ce-reduced slice |
| Knative | per-arrival (Regime B) |

| Config | Kn | MLP ce-red | GNN ce-red | Best |
|--------|---:|-----------:|-----------:|------|
| `hub_k4_seek35` | **0.87M** | 26.54M | 1.35M | Kn |
| `hub_k4_seek50` | **1.52M** | 27.27M | 4.63M | Kn |
| `hub_k4_seek65` | **1.39M** | 26.41M | 2.15M | Kn |
| `hub_k6_seek35` | **0.63M** | 21.89M | 0.73M | Kn |
| `hub_k6_seek50` | **0.68M** | 22.92M | 0.97M | Kn |
| `hub_k6_seek65` | **0.70M** | 22.37M | 0.76M | Kn |
| `hub_k8_seek35` | 0.53M | 18.75M | **0.56M** | Kn |
| `hub_k8_seek50` | **0.56M** | 18.19M | 0.57M | Kn |
| `hub_k8_seek65` | **0.55M** | 18.18M | 0.60M | Kn |

**9-config sums:** Knative **7.44M** · GNN **12.31M (+65%)** · MLP **202.53M (+2620%)**

**Win counts:** Knative **9/9** · GNN **0/9** · MLP **0/9**

**vs dim22 bipartite v1 (reference):** GNN **17.90M** · MLP **21.49M** · Kn **18.37M** — merged ce-reduced **does not reproduce** dim22 phase-boundary story; Knative much lower under `node_disk_v2` (**7.44M** vs **18.37M**).

**MLP failure mode:** avg queue **32–48s** vs Kn **1–3s** per task; decode `chosen_queue_vs_min` mean **~1122** (systematic hub pile-up); cold/pull rates similar to Kn — RTT gap is queue backlog, not warmth physics artifact.

**GNN:** not broken like MLP; k6/k8 **~0.5–1M** (near Kn); k4 **1.3–4.6M** (weak vs Kn, still **~6×** better than MLP on k4).

**Ship verdict:** **Reject** merged ce-reduced MLP for hub/125-225; **reject** merged ce-reduced GNN as Knative-beater on this grid; keep dim22 bipartite table for RQ3 until merged **dim14-full** or retrained reduced MLP with hub co-sim + queue-pressure features gated.

---

---

## Bipartite v2: wssm + node_disk_v2 (125-225, 2026-06-14) ★ Updated

> **Key result (2026-06-14): MLP dim22 batchcache achieves 8/9 wins, sum 6.09M (−67% vs Knative, −39% vs GNN wssm)** on the bipartite hub grid under node_disk_v2. **Tabular MLP beats GNN when trained from the same batch cache with matching dim22 platform features.** Prior GNN-only wins were driven by MLP train/serve encoding bugs (ce_reduced 939M; seq dim22 49M).

**Sweeps:** GNN `bipartite_v2_skew_merged_20260614/` · MLP-bc `bipartite_v2_mlp_dim22_batchcache_20260614/` · seed 42

| Field | Value |
|-------|-------|
| Workload | `workload-125-225.json` (~562k tasks) |
| Topology | k∈{4,6,8} × seek∈{35,50,65}% · 5ms core / 30ms periphery |
| Physics | `HEROSIM_WARMTH_PHYSICS=node_disk_v2` |
| GNN | `near-rtt-v2-warmth-sparse-skew-merged-dim14-ce-only.pt` · dim22 · b=4 |
| **MLP (canonical)** | `batch_edge_mlp_warmth_sparse_skew_merged_dim22_batchcache.pt` · dim22 · batch cache |

### ★ Canonical 3-way table (MLP batchcache vs GNN vs Knative)

| Config | GNN wssm | MLP-bc | Knative | bc/Kn | GNN/Kn | Best |
|--------|--------:|-------:|--------:|------:|-------:|------|
| `hub_k4_seek35` | 2.47M | **0.98M** | 3.30M | −70% | −25% | **MLP-bc** |
| `hub_k4_seek50` | 1.19M | **0.63M** | 2.52M | −75% | −53% | **MLP-bc** |
| `hub_k4_seek65` | 1.53M | **0.77M** | 3.09M | −75% | −51% | **MLP-bc** |
| `hub_k6_seek35` | 0.77M | **0.63M** | 2.29M | −72% | −67% | **MLP-bc** |
| `hub_k6_seek50` | 1.56M | **0.68M** | 1.45M | −53% | +8% | **MLP-bc** |
| `hub_k6_seek65` | **0.73M** | 0.74M | 1.68M | −56% | −56% | GNN |
| `hub_k8_seek35` | 0.62M | **0.54M** | 1.30M | −58% | −52% | **MLP-bc** |
| `hub_k8_seek50` | 0.56M | **0.55M** | 1.32M | −58% | −58% | **MLP-bc** |
| `hub_k8_seek65` | 0.59M | **0.55M** | 1.41M | −61% | −58% | **MLP-bc** |

**9-config sums:** **MLP-bc 6.09M** · GNN 10.02M · Knative 18.37M

**Win counts:** **MLP-bc 8/9** · GNN 1/9 · Knative 0/9

### MLP encoding ladder (ablation on same grid)

| Variant | Train source | 9× sum | Wins vs Kn | Finding |
|---------|-------------|-------:|-----------:|---------|
| ce_reduced | seq cache, 11-d | 939.76M | 0/9 | Dropped is_warm, is_cold; queue train/serve mismatch |
| dim22 seq-fix | seq cache, 22-d | 49.21M | 0/9 | Restored edge features; platform still raw queue + is_cold |
| **dim22 batchcache** | **batch cache (GNN)** | **6.09M** | **8/9** | **Norm queue + shared_fate — matches inference** |

**Key findings:**
- **MLP-bc is the best hub policy on this grid** — beats GNN on 8/9 configs including k6_seek50 where GNN loses to Knative (+8%).
- **GNN wssm remains strong** (10.02M, −45% vs Kn) but is **secondary** to parity-trained MLP on hub topology.
- **GIN batch coupling is not necessary** for hub coordination when MLP train/serve features match GNN cache encoding.
- **Regression warning:** wssm models regress on legacy 7-config benchmark (+45% vs dim14-ce). Deploy warmth checkpoints only on node_disk_v2 hub targets.

### MLP dim22 seq-fix (superseded by batchcache)

Seq-cache retrain (`batch_edge_mlp_warmth_sparse_skew_merged_dim22.pt`, val_acc 0.731) recovered from 939M→49M but still lost all 9 vs Knative (+168%). Superseded — do not cite as final MLP result.

**vs Hub9 ce-reduced gate (superseded):** ce-reduced GNN was Knative 9/9; wssm GNN is 8/9. The difference is the skew-merged co-sim training grid which includes `degree_skewed_core` topologies.


## contention_v2 Live Gate (2026-06-15, collision-aware labels)

**Purpose:** Validate GNN trained on **non-separable** co-sim labels (`contention_v2` grid, `--allow-non-unique-replicas`) vs feature-matched MLP and Knative.

**Sweep:** `contention_v2_live_gate_20260615/` · `node_disk_v2` · workload-125-225 · 3 sparse configs · argmax · seed 42

**Models (trained on 711-graph cache):**
- GNN: `near-rtt-v2-contention-v2-dim14-ce-only.pt` (val acc 66.2%)
- MLP: `batch_edge_mlp_contention_v2_dim22_batchcache.pt` (val edge acc 90.0%)

| Config | Knative | GNN | MLP | Winner |
|--------|--------:|----:|----:|--------|
| sparse_p25 | 6.90M | 7.66M | **5.81M** | MLP |
| sparse_p35 | 12.77M | **11.08M** | 16.79M | GNN |
| sparse_p25_skew | **1.06M** | 1.14M | 2.75M | Kn |
| **SUM** | 20.73M | **19.88M** | 25.35M | GNN sum |

**Status: PARTIAL** — GNN wins 3-config sum (−4% vs Kn, −22% vs MLP) but per-config wins are **1/1/1**. Offline ablation showed GNN collision robustness; live transfer limited by train/serve gap + argmax decode. See `memory/gnn_necessity_separability.md` §8.

**Offline ablation (same corpus, n=142 test):** gnn_base beats pointwise on top-1 (90.7% vs 89.1%), p90 regret (2.39% vs 2.81%), opt-recovery (69% vs 65%), and eliminates catastrophic tail (max 7% vs 3164%).


## contention_v3 Live Gate (2026-06-20, datalab)

**Purpose:** Full pipeline on **contention_v3** co-sim grid (conn 0.15/0.20, heavier queues) — offline audit showed *lower* coupling vs v2; this closes the end-to-end deploy question.

**Sweep:** `contention_v3_live_gate_20260620/` · `node_disk_v2` · workload-125-225 · 3 sparse configs · argmax · seed 42

**Models (900-graph cache `graphs_cache_contention_v3`):**
- GNN: `near-rtt-v2-contention-v3-dim14-ce-only.pt` (val acc 65.6%)
- MLP: `batch_edge_mlp_contention_v3_dim22_batchcache.pt` (val edge acc 88.2%)

| Config | Knative | GNN | MLP | Winner |
|--------|--------:|----:|----:|--------|
| sparse_p25 | 7.05M | 8.26M | 9.52M | Kn |
| sparse_p35 | 12.00M | 26.45M | 33.16M | Kn |
| sparse_p25_skew | 1.22M | 1.29M | **1.08M** | MLP |
| **SUM** | **20.27M** | 36.00M | 43.76M | Kn sum |

**Status: REJECT** — GNN sum **+81%** vs contention_v2 (19.88M); Kn wins 2/3; sparse_p35 cliff worse than v2 (GNN 26.4M vs 11.1M). Confirms offline separability audit: v3 is not a deploy path.

**vs contention_v2 (§ above):** v2 GNN beats Kn on sum; v3 GNN loses badly. **Ship contention_v2 GNN** for sparse ER. **Mitrix:** v3 corpus 900/900 jsonl local; pipeline artifacts (cache, models, live gate) on datalab unless pulled.


## Strategic-merge deployment (2026-06-16, mitrix · closed 2026-06-27)

**Train:** warmth + sparse + contention_v2, strategic coupled oversample · GNN val **48.8%** · MLP val edge **78.5%**

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

**vs contention-only baseline (§ above):** GNN sum regresses +17%; MLP sum +238% (sparse_p35 cliff). Knative ~flat.

**Closure:** compare table archived `strategic_merge_contention_live_gate_20260616/compare.txt`; `logs/strategic_merge_pipeline/phase_all.done`.

---

## Weighted-merge deployment (2026-06-16, mitrix · closed 2026-06-27)

**Train:** 8× coupled oversample on warmth+sparse+contention_v2 · GNN val **26.3%** · MLP val edge **72.3%**

**Sweep:** `merged_contention_weighted_live_gate_20260616_105500/` · workload-125-225 · `node_disk_v2`

| Config | Knative | MLP | gnn_uniq | gnn_argmax | Best |
|--------|--------:|----:|---------:|-----------:|------|
| sparse_p25 | 7.53M | 14.91M | 9.14M | 9.30M | kn |
| sparse_p35 | 12.29M | 64.89M | **11.34M** | 12.17M | gnn_uniq |
| sparse_p25_skew | 1.33M | 1.84M | 1.82M | 3.12M | kn |
| **SUM** | **21.14M** | **81.64M** | **22.30M** | 24.59M | kn |

**Wins:** kn=2 · gnn_uniq=1 · mlp=0. **gnn_uniq** preferred over argmax (lower sum). Neither learnable policy beats Knative on 3-config sum; MLP fails on sparse_p35 same as strategic merge.

**Closure:** compare table archived `merged_contention_weighted_live_gate_20260616_105500/compare.txt`; `logs/merged_contention_pipeline/phase_all.done`.


## warmth_v2 Skew3 Live Gate (2026-06-11, node_disk_v2 physics)

**Sim physics:** `node_disk_v2` + defer_cold (consolidation-optimal labels). **Models lack explicit `node_disk_hit` in graph cache.**

**Configs:** `default`, `05_sparse`, `default_degree_skew` (3-config gate).

| Sweep | Policies | Result |
|-------|----------|--------|
| `warmth_v2_physics_skew3_20260611/` | old MLP + old GNN | MLP **4.25M** sum vs GNN **5.02M** — **MLP wins 2/3** |
| `warmth_v2_ce_reduced_skew3_20260611/` | v2 ce-reduced GNN/MLP | v1 MLP still **2/3**; v2 GNN sparse **1.23M** vs Kn **0.93M** (regression) |

**Takeaway:** v2 sim physics alone does not flip GNN vs MLP without disk feature + hub topology in co-sim training grid. Do not ship sparse finetune on skew3 alone.

### wssm models skew3 gate (2026-06-14, node_disk_v2)

**Sweep:** `skew3_full_gate_20260614/results/` · GNN wssm (`near-rtt-v2-warmth-sparse-skew-merged-dim14-ce-only.pt`) · MLP wssm · Knative

| Config | GNN wssm | MLP wssm | Knative | Winner | GNN vs Kn |
|--------|--------:|---------:|--------:|--------|----------:|
| `default` | 5.29M | 31.84M | **3.10M** | Knative | +70.6% |
| `default_degree_skew` | **0.51M** | 16.71M | 0.61M | GNN | −15.6% |
| `05_sparse_degree_skew` | 1.00M | 12.37M | **0.79M** | Knative | +26.9% |

**GNN wssm 1/3; Knative 2/3; MLP wssm broken.** GNN wssm wins `degree_skew` (−15.6% vs Kn) but badly loses `default` (+70.6%). Gate not passed. MLP wssm failure is layout mismatch (ce_reduced on non-warmth-optimized feature slice).

---

## Topology Stress: Skew-4 (degree-skewed core, 4 configs)

**Policies:** dim14-ce GNN · MLP dim22 (`INFERENCE_FEATURE_LAYOUT=dim22`) · Knative per-arrival.

**Configs:** `default_20_20_p50` · `05_sparse_40_40_p25` · `default_20_20_degree_skew` · `05_sparse_40_40_p25_degree_skew` (`atomic21_skew_configs/`, `k_core=4`, seek=40%).

### workload-100-100

**Sweep:** `dim14_old_models_skew4_20260610/results/`

| Config | GNN | MLP | Best |
|--------|----:|----:|------|
| `default` | 4.33M | **4.10M** | MLP |
| `05_sparse` | 2.66M | **2.59M** | MLP |
| `default_degree_skew` | **1.03M** | 1.09M | GNN |
| `05_sparse_degree_skew` | **997k** | 1.14M | GNN |

**GNN wins 2/4** (degree-skew only).

### workload-125-225 (datalab)

**Sweep:** `dim14_old_models_skew4_125225_20260610/results/`

| Config | GNN | MLP | Knative | Best |
|--------|----:|----:|--------:|------|
| `default` | 30.66M | **21.50M** | 27.00M | MLP |
| `05_sparse` | 10.81M | **7.82M** | 10.43M | MLP |
| `default_degree_skew` | **1.27M** | 1.85M | 2.19M | GNN |
| `05_sparse_degree_skew` | **1.28M** | 1.51M | 1.56M | GNN |

**4-config sums:** GNN 44.0M · MLP **32.7M** · Knative 41.2M · wins **GNN 2/4 · MLP 2/4**.

**Takeaway:** Under heavy load, MLP wins uniform/sparse; GNN wins degree-skew (largest gap vs Knative on skew configs). GNN RTT explodes on uniform 125-225 (`default` 30.7M) while skew stays ~1.2–1.8M.

### Skew-4 wssm models (workload-125-225, node_disk_v2, 2026-06-14)

**Sweep:** `skew4_new_models_20260614/results/` · same 4 skew configs · `HEROSIM_WARMTH_PHYSICS=node_disk_v2`

| Config | GNN wssm | MLP wssm | Knative | Best | GNN vs Kn |
|--------|--------:|---------:|--------:|------|----------:|
| `default_20_20_p50` | 38.18M | 27.42M | **19.47M** | Knative | +96.1% |
| `05_sparse_40_40_p25` | 14.02M | 10.37M | **8.15M** | Knative | +72.0% |
| `default_20_20_degree_skew` | **0.55M** | 14.05M | 0.68M | GNN | −18.7% |
| `05_sparse_40_40_p25_degree_skew` | 1.36M | 10.91M | **1.19M** | Knative | +14.3% |

**Wins:** Knative **3/4** · GNN wssm **1/4** · MLP **0/4**. Same pattern as v1: GNN only wins `default_degree_skew`. **GNN wssm regresses** on uniform configs vs dim14-ce (38.18M vs 30.66M). MLP wssm broken (ce_reduced layout). This topology type (flat skew) is not a good deployment target for wssm models without node_disk_hit feature engineering.

---

## Topology Stress: Tiered-Hub (exploratory — superseded by Bipartite v1 for RQ3)

> **Use bipartite v1 for phase-boundary claims.** Tiered-hub retains k=2 toxicity examples and 100-100 ≈tie data; asymmetric latency not enforced in generator; Knative 125-225 was pending at sweep time.

**Topology knobs:** `k_core` Xavier hub servers; `hub_seeker_fraction`; generator: `generate_tiered_hub_configs.py`.

**Policies:** `gnn_dim22` (dim14-ce + dim22 layout) · `mlp_dim22` · Knative (125-225 pending).

### workload-100-100 (9 hubs, datalab job 477929)

**Sweep:** `tiered_hub_gnn_mlp_20260610/results/`

| Config | GNN | MLP | Best |
|--------|----:|----:|------|
| `hub_k2_seek30` | **2.25M** | 2.43M | GNN |
| `hub_k2_seek50` | 2.07M | **1.91M** | MLP |
| `hub_k2_seek80` | **3.04M** | 3.05M | GNN |
| `hub_k4_seek30` | **922k** | 931k | GNN |
| `hub_k4_seek50` | **1.04M** | 1.12M | GNN |
| `hub_k4_seek80` | 1.42M | **1.08M** | MLP |
| `hub_k6_seek30` | **846k** | 945k | GNN |
| `hub_k6_seek50` | 886k | **902k** | MLP |
| `hub_k6_seek80` | 951k | **928k** | MLP |

**9-hub sums:** GNN **13.41M** · MLP **13.30M (−0.8%)** · wins **GNN 5/9 · MLP 3/9**. atomic21 layouts rejected (+16%/+54%).

### workload-125-225 (11 configs, datalab job 478100)

**Sweep:** `tiered_hub_gnn_mlp_125225_20260610/results/`  
**Status:** ML **21/22** (GNN timeout on `hub_k2_seek30`). Knative column incomplete at sweep time — see **Bipartite v1** for complete 3-way hub compare.

| Config | GNN RTT | MLP RTT | Knative RTT | Best Policy |
| :--- | ---: | ---: | ---: | :--- |
| `default_20_20_p50` | 25.99M | **21.18M** | *pending* | MLP |
| `05_sparse_40_40_p25` | 12.44M | **7.76M** | *pending* | MLP |
| `hub_k2_seek30` | TIMEOUT @3600s | **9.35M** | *pending* | MLP* |
| `hub_k2_seek50` | 19.15M | **9.76M** | *pending* | MLP |
| `hub_k2_seek80` | 17.19M | **17.08M** | *pending* | MLP (tie) |
| `hub_k4_seek30` | **1.39M** | 2.03M | *pending* | GNN |
| `hub_k4_seek50` | **1.51M** | 1.64M | *pending* | GNN |
| `hub_k4_seek80` | 3.87M | **3.86M** | *pending* | MLP (tie) |
| `hub_k6_seek30` | **1.16M** | 1.42M | *pending* | GNN |
| `hub_k6_seek50` | **1.19M** | 1.41M | *pending* | GNN |
| `hub_k6_seek80` | **1.28M** | 1.92M | *pending* | GNN |

\* GNN result unavailable; MLP only.

**Summary of relational advantage (Δ% GNN vs MLP):**  
*Positive = GNN worse; negative = GNN advantage.*

| Regime | Observation |
|---|---|
| **$k=2$ (capacity-starved)** | Extreme volatility. Seeks 30–50% favor MLP; GNN hot-spots or sim timeout (+96% on seek50). |
| **$k=4$ (marginal capacity)** | GNN wins moderate seek (seek30: −31%, seek50: −8%); statistical tie at seek80 (+0.2% favoring MLP). |
| **$k=6$ (coordinated routing)** | GNN wins all seek fractions (seek30: −19%, seek50: −15%, seek80: −33%). |

**Paired wins (10 configs with both ML):** **GNN 5 · MLP 5**.

**Takeaway (125-225):** GNN advantage emerges at **`k_core > b`** (6 hubs vs batch size 4) and moderate seek; **`k=2`** or **uniform controls** favor MLP. At 100-100 the same grid was ≈tie (+0.8% sum); heavy workload sharpens the split.

**GNN-favorable design (hypothesis):** `k_core ∈ {6,8}`, seek ∈ [35%,55%], avoid k=2 under 125-225; optional future: differentiate `latency_core_ms` vs `latency_periphery_ms` in generator (currently single latency for all links).

**Figures:** `simulation_data/topology_sketches/` (`sketch_topologies.py`).

---

## Regime A: Batch Scheduler Comparison (30k tasks, default topology, seed 42)

All three schedulers in Regime A use the same batch collection strategy (20ms window, max 4 tasks). This controls for the batch vs. per-arrival architecture difference.

| Policy | RTT | vs GNN CE | Wall-clock (full workload-100-100 sim) |
|---|---|---|---|
| **GNN CE-only** | **669.8k** | — | ~10 min (GPU) |
| XGBoost batch | 886.9k | **+32%** | ~80 min (CPU) |
| Knative-batch | 1.62M | +142% | ~6 min |

**Source:** `simulation_data/normal_sim_sweeps/regime_a_compare_20260609/results/`

**On simulation wall-clock:** The ~80 min for XGBoost is the cost of running the full simulation with XGBoost as the live scheduler (per-event graph-build + XGB predict on CPU). The ~10 min for GNN is GPU inference amortized across batches. Neither is charged to `total_rtt` — this is engineering wall-clock only.

**On the RTT gap:** XGBoost scores each task–platform edge independently (22-dim row). MLP dim22 uses the same rows with a pointwise MLP. Both use the **same argmax decode** as GNN (`MLPBatchScheduler` inherits `GNNScheduler`). The GNN difference is **GIN message passing** before edge scoring — platform embeddings aggregate batch context when tasks share hub candidates — not a different deployment decode loop.

---

## Offline Metric Divergence: The Evaluation Fallacy

This table shows that offline validation metrics are unreliable predictors of live deployment quality for ranking-trained models.

| Model | Offline metric | Value | Live `default` RTT | Divergence |
|---|---|---|---|---|
| CE-only (deployed) | val accuracy @ep90 | 24.4% | **4.14M** | Consistent |
| Pairwise ranking | test top-5 regret @ep67 | **0.063s** | 11.62M | Catastrophic (+181%) |

The ranking model has a 5× better offline top-5 regret than CE's equivalent accuracy-based metric — yet produces 181% higher live RTT. This divergence is not a measurement anomaly; it is structural. Offline evaluation uses static co-sim snapshots where oracle mean queue wait (`averageQueueTime`) stays in single digits (1060 max **11.5s**, p99 **5.5s**; legacy 3705 max **6.5s**). Live deployment under sustained load reaches **~53s** mean queue wait on 150-150 Knative (**~21s** on default 20×20) — a **~5–8×** gap on avgQueueTime vs 1060 max, not ~200×. The **~200×** figure applies to decode-time platform queue depth (**qvm p95** ~2857 vs CE ~408). The ranking model's logits, sharpened for combo-sum discrimination on training-range queues, do not generalize to live-range congestion.

**Proposed alternative:** qvm p95 from a short (≥30k task) active simulation sweep is a significantly more reliable predictor of live scheduling quality than offline top-k regret or per-task accuracy.

---

## Train/Serve Distribution Gap

**Source:** `scripts_cosim/verify_queue_gap_1060.py` (2026-06-10). Metric: `stats.averageQueueTime` unless noted.

| Metric | Co-sim 1060 | Co-sim 3705 | Live 150-150 Kn | Live default (dim14-ce) |
|---|---|---|---|---|
| max avgQueueTime | 11.54s | 6.50s | 52.7s | 20.6s |
| p99 avgQueueTime | 5.48s | 2.28s | — | — |
| max per-task queueTime | 18.4s | 11.7s | — | — |
| max snapshot depth (tasks) | 98 | 67 | — | — |
| Cold-start (task `coldStartProportion`) | ~0.1%† | — | 9.6–16.4% | 4.3%‡ |
| Cold platform fraction (`initialized_snapshot`) | ~71% | — | — | — |

† Co-sim optimal 4-task runs: `coldStartTime > 0` on placement tasks (n=1230). ‡ Live 150-150 Knative aggregate; CE 7-config mean **13.4%**.
| avgQT scale vs 1060 max | 1× | — | **~4.6×** | **~1.8×** |

**CE-only handles this gap natively.** Because the CE objective trains per-task marginal logits aligned with per-task marginal argmax at serve time, it does not distort logit scales in response to queue distribution. A platform with slightly higher queue but better hardware will receive a lower logit — calibrated to the co-sim label — and the model's decision is stable even when live mean queue wait is several-fold higher than co-sim oracle labels.

**Ranking is fragile under this gap** because the combo-sum objective sharpens logit magnitudes until the fastest hardware platform dominates all task assignments, regardless of queue depth. At inference, live platform queue depths (qvm p95 ~2857) far exceed co-sim feature range — a separate axis from avgQueueTime — making live congestion invisible to magnitude-maximizing ranking logits.
