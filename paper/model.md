# Model Architecture & Training

> GNN design, platform features, loss functions, training objectives.
> Update when architecture changes, new checkpoints become anchors, or loss variants are added.

---

## Graph Representation

The input to the GNN is a **bipartite task–platform graph** constructed by `prepare_graphs_cache.py` (`build_graph`) and replicated at inference by `feature_builder.py` (`build_inference_feature_bundle`).

### Task Nodes

| Cache path | Dims | Notes |
|------------|-----:|-------|
| `prepare_graphs_cache.py` (hom dim14) | **3** | `[type_dnn1, type_dnn2, src_norm]` |
| `prepare_graphs_cache_seq.py` (warmth seq) | **2** | one-hot only — **`src_norm` stripped** (train/serve landmine) |

### Platform Nodes (14-dim) — "dim14"

| Dim | Feature | Notes |
|---|---|---|
| 0–4 | `type_onehot(5)` | rpiCpu, xavierCpu, xavierGpu, xavierDla, pynqFpga |
| 5 | `has_dnn1` | 1 if dnn1 replica active on this platform |
| 6 | `has_dnn2` | 1 if dnn2 replica active on this platform |
| 7 | `queue_depth` (normalized) | raw queue / p90 adaptive norm |
| 8 | `shared_fate_signal` | `cold_replicas_on_node / total_replicas_on_node` |
| 9 | `current_task_remaining_norm` | remaining exec time / 10s |
| 10 | `cold_start_remaining_norm` | remaining cold-start time / 10s |
| 11 | `comm_remaining_norm` | remaining comm time / 10s |
| 12 | `target_concurrency_norm` | estimated optimal concurrency / 20 |
| 13 | `usage_ratio_norm` or **reserved** | hom cache: usage_ratio; atomic21 seq recache: **0.0** (candidate for `node_disk_hit`) |

**Critical:** dim 8 (`shared_fate_signal`) was zeroed due to a bug until 2026-06-08. Cache must be rebuilt from scratch to use valid `shared_fate_signal`. Do not load dim-13 checkpoints on dim-14 caches.

### Edges (5-dim)
```
[exec_time, network_latency, is_warm, energy, comm_time]
```

`is_warm=1` iff sandbox `previous_task.type["name"] == task.type["name"]` at inference (`feature_builder.py`). **Hom cache** (`prepare_graphs_cache.py`): fixed 2026-06-08. **Seq recache**: still uses replica flag → always 1 on feasible edges — fix pending.

---

## GNN Architecture: `TaskPlacementGNN`

File: `src/policy/gnn/gnn_model.py`

```
Task features (3) ──► MLPEncoder ─┐
                                   ├─► concat ──► GIN (3 layers, hidden=64)
Platform features (14) ──► MLPEncoder ─┘               │
                                                per-edge:
                                         concat(task_emb[64], plat_emb[64], edge_attr[5])
                                                        │
                                                  EdgeScorer MLP → scalar logit
Output: List[Tensor]  (variable length per task — one logit per feasible replica)
```

- **Message passing:** GIN over **task→platform edges only** (no physical client→hub topology edges)
- **Edge scoring:** independent 2-layer MLP over concatenated embeddings + edge features
- **Output structure:** per-task logit vector; length varies with feasible replica count per task; NOT padded

---

## Active Checkpoints

| Name | File | Cache | Role |
|---|---|---|---|
| **dim14-ce-only (eval anchor)** | `near-rtt-v2-dim14-ce-only.pt` | 1060 (1230 ds) | Published Kn/HRC 7/7; bipartite/skew sweeps |
| **warmth dim14-ce (train)** | `near-rtt-v2-warmth-dim14-ce-only.pt` | merged **824** | v2 physics; skew3 gate pending |
| dim14-ranking (rejected) | `near-rtt-v2-dim14-1060.pt` | 1060 | CE + ranking — not shippable |
| Track B r030 (rejected) | `near-rtt-v2-dim14-ce-init-r030.pt` | 1060 | +1.9% vs CE post-fix |

**Offline metrics:** compare GNN **`val/task_acc`** (~69%) to MLP **`val_edge_accuracy`** (~68%) — not GNN `val/acc` (~23%) vs MLP edge acc (harder joint target).

---

## Loss Functions

### 1. Cross-Entropy Only (`ce_only`)

```python
loss = CrossEntropy(logits[t], y[t])   # for each task t independently
```

- `y[t]` = index of optimal replica in task t's feasible set (from brute-force oracle)
- Trains per-task marginal distribution
- **Deploy with:** `GNN_DECODE_MODE=argmax`
- **Structural alignment:** PERFECT — same operation at train and serve time

### 2. NearRttRankingLoss v2 (`ranking`)

```python
combo_score(logits, indices) = sum(logits[t][indices[t]] for t in tasks)
loss = softplus(neg_combo_score - opt_combo_score + margin(ΔRTT)) * band_weight
```

Bands (ΔRTT from optimal):

| Band | Range | Weight |
|---|---|---|
| near | 0 – 0.05s | 3.0 |
| close | 0.05 – 0.30s | 2.0 |
| mid | 0.30 – 1.0s | 1.0 |
| far | 1.0 – 5.0s | 0.75 |
| trash | > 5.0s | 1.0 (v2 only) |

Margin: `min(margin_cap, max(0.05, clamp(ΔRTT/rtt_scale, clip)))` — exponential in v2

- **Deploy with:** `GNN_DECODE_MODE=argmax` or `frozen_topk`
- **Structural alignment:** BROKEN — joint training, marginal serve
- **Consequence:** logit over-sharpening, hot-spotting qvm p95=2857 on default (pre-fix inference); post queue-map fix `default` RTT 5.48M vs CE 4.31M — still not shippable

### 3. Soft-Combo CE (`soft_combo`)

```python
S(combo) = sum(logits[t][combo[t]] for t in tasks)
P*(combo) ∝ exp(-regret(combo) / τ)    # τ=0.25
loss = KL(P* || softmax(S_all_combos))
```

Trains toward a soft Boltzmann distribution over all placement combos weighted by exact RTT regret. Temperature τ=0.25 sharpens the target toward near-optimal combos.

---

## Training Configuration

- Optimizer: Adam, lr=5×10⁻⁴, weight decay
- Batch size: 32 graphs
- Epochs: 100 (standard), early stopping via collapse guard
- Collapse guard: revert to best checkpoint if val acc < 5% or sidecar hash hit < 10%
- WandB project: `gnn-near-rtt-jun2026`
- Checkpoint path: `src/notebooks/models/{wandb_run_name}.pt` → copy to `models/` for sim

## Two-Phase Training (Track B)

- Phase A: CE-only (produces anchor `near-rtt-v2-dim14-ce-only.pt`)
- Phase B: Load Phase A checkpoint, fine-tune with sequential re-forward regret
- Phase B checkpoint metric: `val/regret_seq_reforward` (per-task argmax with queue rollforward between tasks in eval)
- **r030 live gate (post queue-map fix, 3×3):** 3cfg sum 8.18M vs CE 8.03M (+1.9%) — **rejected**; sweep `dim14_3model_3cfg_queuefix_20260609/`; next recipe r002

---

## Tabular Baselines: MLP dim22 + XGBoost

File: `src/policy/tabular/`

**MLP (`PointwiseEdgeMLP`):** same 22-d edge rows as XGB — `[task(3), plat(14), edge(5)]`. Scores edges **independently**. Uses **identical argmax decode** as GNN via `MLPBatchScheduler`. Live dim22 includes **`src_norm`**; seq warmth recache may not.

**XGBoost:** `rank:pairwise` on same rows.

**Results:**
- Triangle all7: MLP sum **+0.4%** vs GNN CE (≈ tie)
- Bipartite v1: GNN **8/9** vs MLP; MLP collapse k6 seek50 **3.46M**
- Regime A 30k: XGB **+32%** vs GNN CE

**Why GIN can win (regime-dependent):** When batch tasks share hub candidates, GIN platform embeddings encode **joint batch demand** before per-edge scoring. MLP and XGB cannot — but MLP matches GNN on flat 7-config sums and wins uniform skew-4 at 125-225.
