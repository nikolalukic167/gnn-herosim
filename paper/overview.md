# System Overview

> What we built, why, and what the central claim is.
> Update when system goals or top-level results change.

---

## One-Sentence Thesis

A bipartite GIN scheduler trained on co-simulation oracle labels with **per-task cross-entropy** (structurally aligned with marginal argmax at serve time) beats Knative and fair HRC on standard 7-config connectivity sweeps, and beats batch MLP/XGB on **hub coordination regimes where core capacity exceeds batch size** ($k_{\text{core}} > b=4$). Advantage over tabular MLP is **regime-dependent**, not universal (+0.4% MLP sum on triangle all7; MLP wins uniform skew-4 at 125–225). Pairwise combo-sum ranking remains a separate failure mode (joint train / marginal serve).

---

## The Problem

Serverless FaaS platforms on heterogeneous edge clusters must place arriving inference tasks (DNN1, DNN2) onto replicas distributed across RPi CPUs, Jetson Xavier GPU/CPU/DLA, and PYNQ FPGAs. Three compounding challenges:

1. **Cold-start penalty** — first access to a replica triggers initialization; live normal sim `coldStartProportion` is **9.6–16.4%** on CE 7-config (mean **13.4%**), not a uniform ~88%
2. **Joint batch contention** — two tasks arriving in the same 20ms window may both be routed to the same fast platform, serializing on its queue; Knative assigns each task independently, blind to intra-batch contention
3. **Network topology variation** — edge latency between client and server nodes varies; placement must trade off execution quality vs communication cost

Knative uses shortest-queue per task, independently. No batching, no joint contention awareness. HRC (HeROcache) scores placements but is CPU-heavy and, despite near-oracle performance in isolated co-sim batches (2.01s avg regret vs Knative 5.67s per batch), still yields +4.9% total RTT live due to joint load effects at scale.

---

## The Approach

```
Co-simulation oracle
  (brute-force RTT enumeration
   over task batches × placements)
          │
          ▼
    Bipartite GNN
  (task nodes, platform nodes,
   task→platform edges only,
   GIN message passing,
   edge-score logits per task)
          │
          ▼
  CE training objective
  (per-task marginal classification
   structurally aligned with
   per-task marginal argmax at serve)
          │
          ▼
  Live deployment
  (batch scheduler in full
   HeROsim simulation,
   decomposed per-task argmax
   from single frozen inference graph)
          │
          ▼
  7-config benchmark sweep
  vs Knative, HRC, XGBoost batch
```

---

## Three Core Contributions

### 1. Co-simulation as a capacity-aware supervised learning oracle

The HeROsim discrete-event simulator is used as the label generator. For each sampled (infrastructure state, queue state, incoming task batch) tuple, all feasible placements are enumerated and run to completion. The optimal placement label incorporates real queue contention, cold-start delays, and network latency — not a proxy cost function. This produces a supervised learning dataset grounded in actual system dynamics.

### 2. The joint-training / marginal-inference structural mismatch finding

Pairwise combo-sum margin ranking achieves better offline top-5 regret (0.063s) than CE's validation accuracy, yet causes severe live RTT degradation on the default topology under pre-fix inference (+181%, qvm p95 2857 vs 408). Post queue-map fix, ranking `default` improves to 5.48M but CE-only remains the deployable anchor (4.31M, 3cfg sum 8.03M). We identify the joint-training/marginal-inference structural mismatch as the primary failure mode; partial queue-input correction does not make ranking shippable.

### 3. qvm p95 as a deployment-quality diagnostic

The `chosen_queue_vs_min` (qvm) P95 metric — collected from a short active simulation sweep — predicts live deployment quality more reliably than any offline validation metric. CE-only: qvm p95=408; ranking: qvm p95=2857. We propose this metric as a standard pre-deployment gate for GNN-based schedulers.

---

## Central Claim (Submission Grade)

> A bipartite GIN trained with pointwise marginal cross-entropy supervision from a capacity-aware brute-force simulator oracle achieves lower total RTT than Knative's shortest-queue heuristic across all balanced connectivity regimes. However, under extreme workload stress (125 RPS), its advantage over tabular edge-rankers is strictly bounded by a topological phase transition: the GNN requires sufficient parallel core capacity ($k_{\text{core}} \geq 4$, ideally $k > b$ where $b=4$ is the max batch size) to exploit its relational inductive bias and avoid intra-batch collisions. When capacity is constrained ($k=2$), its structural bias becomes a toxic bottleneck, causing queue collapse, whereas a pointwise tabular MLP degrades gracefully and serves as a more robust default under flat or capacity-starved topologies. Pairwise combo-sum margin ranking — despite superior offline validation metrics — remains a separate failure mode: catastrophic queue hot-spotting ($\text{qvm } P_{95} = 2857$ vs $408$) from a joint-training/marginal-inference objective mismatch.

---

## Simulator: HeROsim

- Discrete-event simulation in Python SimPy
- Heterogeneous hardware: RPi CPU, Jetson Xavier (GPU/CPU/DLA), PYNQ FPGA
- Task types: DNN1 (rpiCpu, xavierGpu, xavierCpu, pynqFpga), DNN2 (rpiCpu, xavierGpu, xavierCpu)
- 10 client nodes + 10 server nodes (configurable)
- Autoscaling from zero (real sim); pre-created replicas with warmup (co-sim)
- Network topology: adjacency matrix with per-link latency; connection probability parameterized

## Policies Compared (Publishable Set)

| Policy | Strategy | Notes |
|---|---|---|
| `gnn` — CE-only | Batch GNN argmax | Primary contribution |
| `knative_network` | Per-task shortest-queue | Industry baseline |
| `herocache_network` (fair) | Per-task scored | kn-autoscale variant |
| `random_network` | Per-task random valid replica | Weak sanity baseline |
| `roundrobin` | Per-task RR on Knative stack | Weak sanity baseline |
| `xgboost_batch` | Batch XGBoost edge ranker | Tabular baseline (Regime A) |
| `knative_network_batch` | Batch Knative | Regime A peer |

---

## Checkpoints & Training Corpus (Two Tracks)

| Track | Checkpoint | Cache | Role |
|-------|------------|-------|------|
| **Eval anchor (published Kn/HRC 7/7)** | `near-rtt-v2-dim14-ce-only.pt` | `graphs_cache_gnn_datasets_4tasks_1060` (1230 ds) | Bipartite, skew, triangle sweeps to date |
| **Train anchor (warmth v2)** | `near-rtt-v2-warmth-dim14-ce-only.pt` | merged **824** (`graphs_cache_warmth_v2_sparse_merged`) | v2 pull physics + sparse conn; **live skew3 gate pending** |

Pre-v2 1060 label comparisons are stale for warmth/placement claims. See `cosimulation.md`.

---

## Key Numbers (Always Keep Current)

| Metric | Value | Config | Source sweep |
|---|---|---|---|
| GNN CE-only `default` RTT (post-fix) | **4.31M** | 20×20 p0.5, 201k tasks | `dim14_3model_3cfg_queuefix_20260609/` |
| GNN CE-only 3cfg sum (post-fix) | **8.03M** | default+02+04 | same |
| Ranking `default` RTT (post-fix) | 5.48M (−53% vs pre-fix) | same | same |
| Ranking 3cfg sum (post-fix) | 9.20M (+14.6% vs CE) | same | same |
| Track B r030 3cfg sum (post-fix) | 8.18M (+1.9% vs CE, rejected) | same | same |
| GNN CE-only `default` RTT (pre-fix anchor) | 4.14M | same | `gnn_near_rtt_v2_dim14_ce_only_20260609/` |
| Ranking ablation `default` RTT (pre-fix) | 11.62M | same | `gnn_near_rtt_v2_dim14_1060_20260608/` |
| GNN CE 7/7 wins vs Knative+HRC (pre-fix) | 7/7 | all configs | CE-only 7-config sweep |
| Random vs CE (7-config) | +250–455% RTT | all configs | `random_rr_3cfg_20260609/` (pre-fix CE denominators) |
| RoundRobin vs CE (7-config) | +78–124% RTT | all configs | same |
| Knative vs CE (7-config) | +25–36% RTT | all configs | `gnn_near_rtt_v2_dim14_ce_only_20260609/` + Knative sweeps |
| GNN CE qvm p95 | 408 | default | decode_stats sidecar (pre-fix sweep) |
| Ranking qvm p95 | 2857 | default | decode_stats sidecar (pre-fix sweep) |
| XGBoost vs GNN CE (Regime A) | +32% (886.9k vs 669.8k) | 30k tasks, default | `regime_a_compare_20260609` |
| MLP dim22 vs GNN CE (Triangle all7) | +0.4% sum (17.96M vs 17.89M) | 201k tasks | `reviewer_triangle_all7_20260609/` |
| Skew-4 125-225 | GNN 2/4 hubs; MLP wins uniform | 562k tasks | `dim14_old_models_skew4_125225_20260610/` |
| Tiered-hub 100-100 | MLP sum −0.8% vs GNN; GNN 5/9 | 201k tasks | `tiered_hub_gnn_mlp_20260610/` |
| Tiered-hub 125-225 | GNN 3/3 on $k=6$; paired split 5–5; MLP wins controls + $k=2$ | 562k tasks | `tiered_hub_gnn_mlp_125225_20260610/` |
| GNN $k=6$ sweep gains | RTT reduction 15%–33% vs MLP across all seeks | 125-225 workload | same |
| GNN $k=2$ peak catastrophe | 19.15M RTT vs MLP 9.76M (GNN hub hot-spotting) | k2 seek50, 125-225 | same |
| hub_k2_seek30 GNN timeout | Hard simulation wall-clock timeout @3600s; retry job 478309 | 125-225 workload | job 478100 |
| Temperature T=3 on ranking model | 10.74M (−7.6%, insufficient) | default | probe sweep |
| Train/serve queue gap (avgQueueTime) | 1060 max 11.5s → 150-150 ~53s (~4.6×); 3705 max 6.5s (~8×); default ~21s (~1.8×) | — | `verify_queue_gap_1060.py` |
| Train/serve qvm p95 gap (ranking) | CE ~408 vs ranking ~2857 (~7× task-count excess at decode) | default pre-fix | decode_stats sidecar |
| Cold-start gap | co-sim task cold **~0.1%**; platform cold **~71%** (`initialized_snapshot`) | live task cold **9.6–16.4%** (CE 7-config) | `audit_doc_claims.py` |
| **warmth_v2 skew3 (node_disk_v2 physics)** | MLP wins **2/3**; v2 GNN sparse **1.23M** vs Kn **0.93M** | 3 cfg | `warmth_v2_*_skew3_20260611/` |
| **Merged train cache** | **824 graphs** (473 warmth + 351 sparse) | — | `graphs_cache_warmth_v2_sparse_merged` |
| **Bipartite v1 9× (125-225)** | GNN sum **17.90M** · MLP **21.49M** · Kn **18.37M** | 562k tasks | `sweep_bipartite_coordination_v1/` |
| Bipartite GNN vs MLP | **8/9** wins | same | same |
| Bipartite 3-way best | GNN 5 · Kn 3 · MLP 1 | same | same |
