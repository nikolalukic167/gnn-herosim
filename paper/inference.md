# Inference in Real Simulation

> How the trained GNN is deployed. Update when scheduler wiring, decode modes, or feature parity changes.
> Seqblend and LQB are NOT production policies — they are archived post-hoc interventions for ranking models. See `archived/legacy_results.md`.

---

## Runtime Stack

```
executesimulation.py  →  GNNScheduler  →  seq_decode.py
      │                      │
      │               build_inference_graph
      │               (feature_builder.py — shared with training)
      │
      └── load_gnn_model (TaskPlacementGNN, dim14)
```

---

## Batch Collection

The scheduler collects arriving tasks into a batch with:
- `MAX_BATCH_SIZE_FOR_GNN = 4`
- `batch_timeout = 0.02` (20ms SimPy time)

Batch size distribution in observed runs: size-4 batches dominate (~82% of batches). Batches outside [2,4] fall back to Knative shortest-queue.

The 20ms batch timeout is the **only** SimPy time cost of GNN scheduling. ML inference wall-clock time (`gnn_decision_time`) is tracked separately and explicitly NOT charged to `total_rtt`.

---

## Graph Construction at Inference

`feature_builder.build_pyg_inference_graph()` constructs the same 14-dim platform features and 5-dim edge features as the training cache. Feature parity is the primary guard against train/serve mismatch. Key parity guarantees:

1. Same adaptive queue norm (`p90` over all platform queues)
2. Same `shared_fate_signal` calculation (`cold_replicas / total_replicas` per node)
3. Same `is_warm` at **live inference** (`previous_task` sandbox match) — **seq recache training** may use degenerate replica-flag `is_warm` (see `model.md`)
4. Same `task_logit_to_placement` mapping (replica-filtered edges only)

**Option 1 (shipped 2026-06-09):** `_capture_full_queue_snapshot()` at batch start captures all 208 platforms. `_capture_temporal_state_snapshot()` wired into `_build_inference_graph`. p90 norm over all graph platforms.

**3-model post-fix gate** (`default`/`02`/`04`, seed 42, argmax): `dim14_3model_3cfg_queuefix_20260609/`

| Model | 3cfg sum | vs CE-only |
|---|---:|---:|
| CE-only | **8.03M** | — |
| Pairwise ranking | 9.20M | +14.6% |
| Track B r030 | 8.18M | +1.9% (rejected) |

**Verdict:** Correct inputs improve ranking `default` from 11.62M (pre-fix) to 5.48M but **do not beat CE-only**; CE-only RTT also rises ~4% on corrected inputs vs pre-fix anchor — fix is for train/serve correctness, not deploy lift. Pre-fix single-model gate (ce-only only): +5.7% sum vs old anchor (`gnn_near_rtt_v2_dim14_ce_only_opt1_frozen_3cfg_20260609_135559/`).

---

## Classical baseline schedulers (fair per-arrival regime)

For apples-to-apples comparison with `knative_network` / `herocache_network` (not GNN batching):

| Policy | Stack | Placement rule |
|---|---|---|
| `knative_network` | Knative network orchestrator + autoscaler | Shortest queue among network-valid replicas |
| `random_network` | **Same Knative stack** | Uniform random among network-valid replicas; `random.seed(seed)` |
| `roundrobin` | **Same Knative stack** (2026-06-09) | Per-arrival RR via `scheduled_count` min among valid replicas; `RoundRobinNetworkScheduler` subclasses `KnativeScheduler` |

Old `roundrobin` used a batched 5×100ms scheduler — **not comparable**; replaced before the 7-config weak-baseline sweep.

---

## Decode Modes

All modes execute **after** a single GNN forward pass. The GNN runs exactly once per batch.

### `argmax` (default, CE-only deployment)

```python
placement[t] = argmax(logits[t])   # for each task t independently
live_queues[chosen_key] += 1       # telemetry only — does NOT affect placement
```

For CE-only, `argmax` and `frozen` are **identical placement decisions**. The queue rollforward only populates `chosen_queue_vs_min` telemetry for the decode stats sidecar.

### `seqblend` (ablation only — not deployed for CE-only)

A post-hoc override that replaces the GNN's chosen platform with the minimum-queue platform when the GNN's choice exceeds the minimum queue by more than `margin`. Applied only to ranking-trained models to partially compensate for logit hot-spotting. Not required for CE-only because CE-only has no hot-spotting problem. Full data in `archived/legacy_results.md`.

### `frozen_topk`

```python
# Enumerate top-k per task, pick best by additive sum
combos = cartesian_product([top_k(logits[t]) for t in tasks])
placement = argmax(combo_score(combo) for combo in combos)
```

Used for near-RTT ranking models. `k=10` default; decode CPU ~2.6ms/batch (vs argmax ~0.17ms).

### Sequential re-forward (Track B Phase B evaluation)

```python
for t in tasks:
    placement[t] = argmax(logits[t])
    # update queue feature for platform chosen by t
    platform_features[plat_pos][dim7] += 1 / adaptive_queue_norm
    # re-run GNN forward pass
    logits = gnn(updated_graph)
```

Only used in Phase B training evaluation (`val/regret_seq_reforward`). **Live results (2026-06-10):** k6 seek65 **1.63M** (−3.3% vs Kn 1.68M); k6 seek50 still loses to Kn (+7.1%); ~200× decode wall vs argmax. See `results.md` § Bipartite seq_reforward overlay.

---

## MLP Batch Inference (same decode stack)

`MLPBatchScheduler` subclasses `GNNScheduler` — same batch collection, same `decode_sequential_placement` argmax, same frozen logits. Difference is **pointwise edge MLP** vs GIN forward. Set `INFERENCE_FEATURE_LAYOUT=dim22` for hub/skew sweeps; `MLP_MODEL_PATH` → `batch_edge_mlp.pt`.

---

## Key Env Variables for Inference

| Variable | Default | Effect |
|---|---|---|
| `GNN_MODEL_PATH` | `models/near-rtt-v2-dim14-ce-only.pt` | checkpoint to load (sweeps); legacy default was dg-26 |
| `GNN_DECODE_MODE` | `argmax` | decode strategy (`seq_reforward`, `seqblend`, `frozen_topk`, …) |
| `GNN_DECODE_TOP_K` | `10` | k for frozen_topk |
| `GNN_LQB_LAMBDA` | (unset) | log1p queue penalty — ablation only, rejected for deploy |
| `GNN_LOGIT_TEMPERATURE` | (unset) | softmax temperature — ablation only, insufficient alone |
| `GNN_QUEUE_NORM_MODE` | `adaptive` | queue normalization mode |
| `GNN_CAPTURE_DATASET_STATE` | `0` | capture SSC for future training (OOM risk at scale) |

---

## Inference Time vs Simulation Time

**Critical distinction:**

- `gnn_decision_time` = wall-clock ML inference time (amortized per task across batch)
- `batch_timeout = 0.02s` = the only SimPy time deducted from simulation
- `total_rtt` does NOT include ML inference wall-clock time
- `rtt_overview` in result JSON exports: `total_inference_time`, `total_rtt_plus_inference` for transparency

This means GNN RTT results are directly comparable to Knative (which has no ML overhead). The 80 min wall time for XGBoost on full workload vs ~10 min for GNN is a wall-clock engineering difference, not an RTT difference.

---

## Decode Stats Sidecar

Every GNN simulation produces a `*.decode_stats.json` sidecar with:
- `decode_mode`, `top_k`, `gnn_batches`
- `chosen_queue_vs_min`: `{mean, median, p95}` — the primary hot-spotting diagnostic
- `intra_batch_platform_collisions`: rate of two tasks choosing the same platform
- `decode_time_ms`: `{mean, p95, total}`
- seqblend-specific: `p1_override_rate`, queue stats on overridden tasks

**Key diagnostic threshold:** qvm p95 > ~500 indicates hot-spotting. CE-only: 408. Ranking: 2857.
