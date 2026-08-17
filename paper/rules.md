# Paper Writing Rules & Session Knowledge

> These are not codebase documentation. They are paper-writing guardrails derived from rigorous fact-checking against the actual implementation. Update whenever experimental results change or new claims are validated.

---

## 1. Terminology Contracts

These are the **only** acceptable terms for the described concepts in any paper draft, review, or academic query.

| Concept | CORRECT term | BANNED terms |
|---|---|---|
| The loss used during training | "pairwise margin ranking softplus over additive combo-sum scores" | "ListNet", "pairwise ListNet", "combinatorial softmax ranking" |
| How we score a placement combination | `combo_score = Σ per-task edge logits` | "joint probability", "product score" |
| How inference works at serve time | "decomposed per-task marginal argmax from a single frozen inference graph" | "autoregressive decode", "autoregressive argmax", "rolling message-passing" |
| The train/serve structural mismatch | "joint training objective vs. decomposed marginal inference" | "distribution shift", "covariate shift" |
| Why CE doesn't hot-spot | "per-task marginal supervision is structurally aligned with per-task marginal argmax at serve time" | "the oracle avoided queue explosions", "CE is a system regularizer" |
| GNN topological generalization | "relational inductive bias on shared hub candidates" | "size invariance", "multi-hop routing" |
| GNN batch advantage | "GIN batch coupling via platform embeddings before edge scoring" | "GNN orchestrates batch while MLP piles tasks" (unless seq_reforward) |
| MLP vs GNN features | "dim22 MLP already has src_norm at live inference" | "GNN-only client identity" (false for dim22 sweeps) |
| Offline accuracy | "compare GNN task_acc to MLP edge acc" | "GNN val/acc 23% vs MLP 68% edge acc" (different targets) |
| XGBoost's structural blindness | "absence of relational inductive bias; each candidate edge is treated independently without inter-node context" | "structural context blindness", "O(V×E) unrolling" |
| Training data source | "capacity-aware brute-force simulator oracle" | "ground truth", "labeled dataset" |
| CE target | "Masked Cross-Entropy over feasible-placement candidates, labels from simulator oracle" | "target paths", "sequence targets" |
| RTT delta vs CE anchor | `(policy − dim14-ce) / dim14-ce × 100`; positive = worse | using policy as denominator (inverts sign) |
| Regime B primary | max-burst elapsed (`regime_b_primary_score_s`) | `total_rtt` / sum elapsed as Regime B score |
| RQ9 write-up | "case study: sequential cost alignment on `oracle_split_v1`" | "we solved serverless scheduling" / "architecture is a trap" / "falsified Set Transformers" |
| Distill GNN role | "instrument on the constructed FilterStore cell" | "new deploy model" / drop-in for 873 |
| RQ3 live answer | sealed 873/v5.5 holdout MLP 13/20 · Kn 6/20 · GNN 1/20 | 7/7 pre-fix table; n=899 collision stats |

---

## 1b. Approved Scientific Framings (Publication-Grade)

These are the peer-review-safe ways to express the three core findings. Do not deviate.

### On the XGBoost comparison (RQ3)

> "While gradient-boosted tree edge-rankers can be configured to score placement candidates via explicit candidate-edge enumeration—transforming the variable bipartite graph into $O(|\mathcal{T}| \times |\mathcal{P}|)$ independent 22-dimensional rows—this tabular unrolling discards all inter-node relational context. Each placement candidate is evaluated in isolation, with no mechanism to propagate how assigning one task to a platform alters the contention context of co-scheduled tasks. Conversely, our bipartite GIN with SUM aggregation natively propagates platform load state across neighboring task nodes before scoring any edge, capturing intra-batch contention implicitly. At the tested scale, this relational inductive bias yields a 32% reduction in total RTT over the tabular baseline."

**Correction from the proposal:** The 80 min vs 10 min figure is **simulation wall-clock** (running the full `workload-100-100` with each policy as the live scheduler), NOT training time. The proposal incorrectly called it "training wall-clock." Use: "simulation wall-clock."

### On the offline evaluation fallacy (RQ7)

> "We document a systematic divergence between offline validation metrics and live deployment quality for GNN-based schedulers: a model trained with pairwise combo-sum margin ranking achieves offline top-5 regret of 0.063s — superior to the CE-trained model's equivalent — yet produces 181% higher total RTT in live simulation. This divergence is structural: offline evaluation uses static co-simulation snapshots where oracle mean queue wait stays in single digits (1060 max ~11.5s, legacy 3705 max ~6.5s), while live simulation under sustained load reaches ~53s mean queue wait on 150-150 (~21s on default 20×20) — a ~5–8× gap on averageQueueTime, with decode-time platform queue depth (qvm $P_{95}$) reaching ~2857 vs CE ~408. Models trained with joint ranking losses, whose logit magnitudes are sharpened for combo-sum discrimination, cannot represent live-range queue congestion under a decomposed per-task argmax deployment. We propose $\text{qvm } P_{95}$ — the 95th-percentile chosen-queue-vs-minimum collected from a short active simulation sweep — as a significantly more reliable pre-deployment quality signal."

### On CE-only and the train/serve gap (RQ4)

> "A primary challenge in learned scheduler deployment is the train/serve distribution gap. Our co-simulation oracle operates over controlled queue distributions (1060 max mean queue wait ~11.5s; legacy 3705 max ~6.5s), while live simulation under sustained load reaches ~53s mean queue wait on 150-150 (~21s on default 20×20) — roughly 5–8× on averageQueueTime, with much larger decode-time platform queue depths (qvm $P_{95}$). We demonstrate empirically that pairwise margin ranking objectives are highly fragile under this scale expansion, as their logit over-sharpening strategy collapses when live queues fall outside the training distribution. In contrast, pointwise masked cross-entropy maintains structural parity between training and deployment — both optimize the same per-task marginal classification — allowing the model to generalize across severe, unmodeled runtime congestion boundaries. This structural alignment, rather than explicit queue-range generalization, provides robustness to the train/serve gap."

**One phrase to never use:** "we prove" — use "we empirically demonstrate" or "we show through controlled ablation."

### On the Tiered-Hub Phase Transitions and GNN Volatility (Tiered-Hub 125-225 Sweep)

> "Under extreme input rates (125 RPS), evaluating network topology configurations through a singular, global metric introduces an evaluation fallacy. Instead, we empirically demonstrate that the efficacy of relational inductive biases (GNN) versus pointwise edge factorizability (MLP) undergoes distinct phase transitions governed by the ratio of premium hub capacity ($k_{\text{core}}$) to the maximum scheduler batch size ($b=4$). Schedulers executing independent, edge-wise inferences act as stable distributed load-balancers when the topology is flat or when hub counts are heavily constrained ($k=2$), whereas the GNN exhibits high volatility and queue hot-spotting due to over-allocation onto narrow hubs. The GNN's relational advantage emerges consistently only in the coordinated routing regime where $k > b$ (e.g., $k=6$). Here, message passing over the bipartite graph allows the scheduler to distribute intra-batch traffic across parallel cores — a cooperative maneuver that independent tabular models are structurally blind to."

---

## 2. Empirically Verified Claims

These claims are directly backed by simulation sweep data. Safe to assert in a paper.

### 2.1 Hot-spotting under ranking loss

- **Metric:** `chosen_queue_vs_min` (qvm), measured as queue depth of chosen platform minus minimum queue at decode time
- **Pre-fix inference (7-config sweep):**
  - Ranking: qvm p95 = **2857**, `default` RTT = **11.62M**
  - CE-only: qvm p95 = **408**, `default` RTT = **4.14M**
- **Post queue-map fix (3×3 gate, Option 1 inference):**
  - CE-only: `default` **4.31M**, 3cfg sum **8.03M** (anchor)
  - Ranking: `default` **5.48M** (−53% vs pre-fix), 3cfg sum **9.20M** (+14.6% vs CE)
  - Track B r030: 3cfg sum **8.18M** (+1.9% vs CE) — **rejected**
- **Source:** pre-fix `gnn_near_rtt_v2_dim14_ce_only_20260609/` vs `gnn_near_rtt_v2_dim14_1060_20260608/`; post-fix `dim14_3model_3cfg_queuefix_20260609/`

### 2.2 Temperature scaling is insufficient to fix logit over-sharpening

- T=3 applied to ranking model at inference: `default` RTT **13.84M → 10.74M** (still catastrophic vs CE 4.14M)
- LQB λ=1.5 log1p queue penalty: probe `default` **4.26M** (recovered vs pre-fix ranking); **7/7 sweep `default` 4.61M** — sum still loses to CE (18.55M vs 17.89M)
- **Conclusion:** logit magnitude distortion from ranking training cannot be fixed post-hoc without modifying the deployment decision rule
- **Source:** `gnn_near_rtt_v2_dim14_1060_lqb15_20260609_100843/`, `memory.md` logit sharpness probe entry

### 2.3 CE-only beats ranking on default topology; ranking competitive on dense configs

- **Pre-fix 7-config sweep:** CE-only wins 3/7 vs ranking (`default`, `00`, `05`); ranking wins 4/7 dense (`01`–`04`, margins 0.2–2.7%)
- **Post queue-map fix 3×3 gate:** CE-only wins 2/3 (`default`, `02`); ranking wins `04` only; CE-only lowest 3cfg sum (8.03M vs 9.20M)
- **Ranking `default`:** 11.62M pre-fix → 5.48M post-fix (−53%) with corrected queue inputs; still +27% vs CE-only post-fix
- **Source:** pre-fix sweeps above; post-fix `dim14_3model_3cfg_queuefix_20260609/`

### 2.4 Train/serve queue gap (~5–8× on avgQueueTime; qvm p95 ~7×)

- Co-sim max `averageQueueTime`: **1060 active cache 11.54s** (p99 5.48s); **legacy 3705 6.50s**
- Live mean queue wait: **150-150 Knative 52.7s** (GNN 45–63s); **default 20×20 dim14-ce 20.6s**
- avgQueueTime scale: **~4.6–8×** (1060/3705 max → 150-150) — **do not cite ~200× on mean queue wait**
- Decode qvm p95: CE **~408** vs ranking **~2857** (platform queue task-count at inference)
- Cold-start: co-sim task `coldStartTime>0` **~0.1%** (n=1230); co-sim platform cold **`initialized_snapshot`** **~71%**; live CE `coldStartProportion` **9.6–16.4%** (7-config mean **13.4%**)
- Fast-forward warmup: RTT/avgQT identical ff on/off on 1060 high-queue A/B (2026-06-10)
- **Source:** `scripts_cosim/verify_queue_gap_1060.py`, `scripts_cosim/audit_doc_claims.py`; legacy 3705 `compare_sim_vs_cosim.py`

### 2.5 GNN generalizes across connectivity regimes

- Trained on 20–90% connection probability (LHS-sampled)
- Deployed on 50% (default), 25% (sparse), 60% (dense) — all 7 configs
- CE-only wins 7/7 vs Knative and HRC across all configs
- **Source:** 7-config normal sim sweep; `generate_gnn_datasets_fast.py` connection probability range

### 2.6 GNN (CE-only) vs XGBoost batch ranker vs Knative-batch (Regime A, 30k tasks)

- GNN dim14-CE: **669.8k RTT**
- XGBoost batch: **886.9k RTT** (+32% vs GNN)
- Knative-batch: **1.62M RTT** (+142% vs GNN)
- Same infra, same workload, same seed 42
- **Source:** `simulation_data/normal_sim_sweeps/regime_a_compare_20260609/results/`

### 2.7 Weak baselines: Random + Round-robin (7-config, seed 42)

- **Δ% convention:** `(policy − CE-only) / CE-only × 100` — positive = worse than CE anchor
- CE-only wins **7/7** vs Random, RoundRobin, and Knative
- Random: **+250–455%** vs CE (catastrophic hot-spotting without queue logic)
- RoundRobin (fair per-arrival Knative stack): **+78–124%** vs CE; worse than Knative every config
- Knative: **+25–36%** vs CE — primary classical baseline
- **Source:** `random_rr_3cfg_20260609/results/`; RR scheduler refactored from batched to `KnativeScheduler` subclass

### 2.8 Bipartite v1 hub stress (125-225, primary RQ3)

- GNN vs MLP **8/9**; sums **17.90M vs 21.49M**
- 3-way best **GNN 5 · Knative 3 · MLP 1**
- **Source:** `sweep_bipartite_coordination_v1/`

**150-150 / 450k benchmarks:** archived — dg-26, 13-dim, not comparable. See `archived/legacy_results.md` only.

---

## 3. Claims Requiring Caveats

These are true but need precise qualification.

### 3.1 "GNN generalizes across network topologies"

**What is true:** The model trained on varying connection probabilities (20–90%) achieves strong results across 7 connectivity-variation configs in deployment.

**What needs qualifying:** The task node feature includes `src_norm = src_node_idx / max(len(nodes), 1)`. This encoding IS topology-scale-sensitive — the semantic meaning of a normalized index changes when node count changes. Parameter sharing in GIN provides better generalization than XGBoost, but strict mathematical size invariance does not hold for our feature encoding. Say: **"better topological generalization"**, not "provably size-invariant."

### 3.2 "CE-only serves as a stable training objective for online schedulers"

**What is true:** CE-only produces calibrated per-task logits that align with the per-task marginal argmax at inference, avoiding the joint-vs-marginal structural mismatch that causes hot-spotting.

**What needs qualifying:** We have not performed a calibration analysis (ECE/reliability diagrams). We observe behavioral alignment (no hot-spotting), not formal calibration. Claim: **"structurally aligned training and inference objectives"**, not "provably calibrated logits."

### 3.3 "XGBoost fails to scale to larger topologies"

**What is NOT shown:** We have no 50×50 XGBoost experiment. The 30k comparison is at one topology (default 20×20 p0.5).

**What IS shown:** GNN outperforms XGBoost at a fixed topology (669.8k vs 886.9k). XGBoost lacks relational inductive bias theoretically.

**What to claim:** Use the connectivity-variation result (7 configs: sparse p0.25 to dense p0.60) to demonstrate robustness, NOT scale-up.

---

## 4. Mechanistic Explanations (Paper-Grade)

### 4.1 Why pairwise margin ranking causes hot-spotting

The ranking loss trains on `combo_score = Σ logits[t][placement[t]]` for all tasks in a batch. The margin `softplus(neg_combo_score - opt_combo_score + margin)` pushes the optimal combo's summed score as high as possible above all suboptimal combos. This is optimized by sharpening individual task logits toward the optimal placement, since all tasks' contributions are summed.

At training time, co-sim oracle mean queue wait (`averageQueueTime`) reaches **11.5s** max on the 1060 cache (**6.5s** on legacy 3705). The model never sees decode-time platform queue depth ~2857 (qvm p95 on ranking live runs). At serve time, when platform queue depths reach hundreds to thousands of tasks, the ranking model has no basis to penalize a congested platform — its logits were sharpened for execution-time and network-time discrimination, not queue avoidance. Result: all tasks pile onto the same fastest-hardware platform regardless of live queue state.

Temperature scaling cannot fix this because the relative ordering of logits is preserved after temperature — the same platform still wins, just with softer confidence. Only a penalty that is linear in queue depth can override sharpened logits (the log1p blend).

CE does not exhibit this because it trains per-task marginal classification: `CrossEntropy(logits[t], optimal_placement[t])`. This loss is minimized when `logits[t][optimal]` is higher than other logits for that task. It never couples logits across tasks into a sum, so no cross-task logit sharpening occurs.

### 4.2 Why CE training is structurally aligned with argmax inference

At training time: loss = `CrossEntropy(logits[t], y[t])` for each task `t` independently.
At inference: `placement[t] = argmax(logits[t])` for each task `t` independently.

The optimization target and the deployment function are the same operation. There is no structural mismatch. The joint placement quality is an emergent consequence of each task independently finding a good platform — not explicitly optimized, but also not corrupted by a joint objective that the deployment function cannot replicate.

### 4.3 Why the oracle label is capacity-aware

The brute-force co-simulation runs each candidate placement in the actual SimPy simulator with real queue contention, cold-start, and network latency. The label `y[t]` is not the fastest execution-time platform in isolation — it is the platform that minimizes the total batch RTT under the queue conditions captured in the SSC. A platform with faster hardware but a longer queue will be labeled suboptimal when its queue wait exceeds the hardware advantage. CE therefore learns to account for queue state implicitly through the oracle labels.

---

## 5. Architecture Quick Reference

**Model:** `TaskPlacementGNN` — bipartite GIN with edge scoring

```
Task nodes (3-dim): [type_dnn1, type_dnn2, src_node_norm]
Platform nodes (14-dim): [type_onehot(5), has_dnn1, has_dnn2, queue_norm,
                          shared_fate_signal, task_remaining_norm,
                          cold_start_remaining_norm, comm_remaining_norm,
                          target_concurrency_norm, usage_ratio_norm]
Edges (5-dim): [exec_time, latency, is_warm, energy, comm_time]

GIN: 3 layers, hidden_dim=64, embedding_dim=64
EdgeScorer: MLP(concat(task_emb, plat_emb, edge_attr[5])) → scalar logit
Output: List[Tensor] — variable-length logit vector per task
```

**Active eval checkpoint:** `models/near-rtt-v2-dim14-ce-only.pt` (1060 cache)

**Train checkpoint:** `models/near-rtt-v2-warmth-dim14-ce-only.pt` (merged 824-cache)

---

## 6. Loss Functions Reference

### CrossEntropy (CE-only)
```python
loss = CrossEntropy(logits[t], y[t])  # per-task, independent
```
Train objective: `NEAR_RTT_TRAIN_OBJECTIVE=ce_only`
Deploy: `GNN_DECODE_MODE=argmax`
Structural alignment: PERFECT

### NearRttRankingLoss (v2, trash-exp)
```python
combo_score = sum(logits[t][placement[t]] for t in tasks)
loss = softplus(neg_combo_score - opt_combo_score + margin(ΔRTT)) * band_weight
```
Bands: near ≤0.05s (w=3), close ≤0.30s (w=2), mid ≤1.0s (w=1), far 1–5s (w=0.75), trash >5s (w=1.0)
Deploy: `GNN_DECODE_MODE=argmax`
Structural alignment: BROKEN (joint train, marginal serve)

### Soft-combo CE
```python
S(combo) = Σ logits[t][combo[t]]
P*(combo) ∝ exp(-regret/τ)   # τ=0.25
loss = KL(P* || softmax(S))
```
Trains toward a soft distribution over all combos weighted by exact RTT regret.

---

## 7. Decode Modes Reference

| Mode | Code | Queue roll-forward affects placement? | Use case |
|---|---|---|---|
| `argmax` | `logits_t.argmax()` per task | No (telemetry only) | CE-only deployment |
| `frozen` | same as argmax | No | Identical to argmax for CE-only |
| `seqblend` | argmax with override if `gnn_q > min_q + margin` | Yes (seqblend choice changes) | Ranking model mitigation |
| `frozen_topk` | cartesian top-k per task, best additive score | No | Near-RTT ranking models |
| Sequential re-forward | argmax, then update queue features, re-run GNN | Yes (GNN input changes) | Track B Phase B |

**Critical note:** `argmax` ≡ `frozen` for CE-only. Roll-forward only increments `live_queues` dict for telemetry — it does NOT change the `logits_t.argmax()` result because logits come from a single frozen GNN pass that saw the pre-batch queue snapshot.

---

## 8. What Makes This Work Novel

1. **Co-simulation as a labeling oracle** — using the production simulator itself as the supervised learning oracle, not a proxy cost function. Labels are real queue-contented RTT measurements from SimPy.

2. **The joint-vs-marginal mismatch finding** — experimentally demonstrating that pairwise combo-sum ranking losses cause severe hot-spotting (qvm p95 = 2857 vs 408) specifically in serverless edge scheduling, where live decode-time platform queue depths far exceed co-sim oracle mean queue wait (~5–8× on avgQueueTime under heavy load).

3. **Topology generalization from LHS sampling** — training on Latin Hypercube Sampled connectivity regimes (20–90%) and showing the GNN generalizes across 7 deployment configs including sparse (p=0.25) and dense (p=0.60) variants, while trained with fixed 10-node topologies.

4. **Platform feature design for contention** — the 14-dim platform feature vector encodes `shared_fate_signal` (cold replica density per node) and three temporal state features, enabling the model to reason about shared cold-start risk before it appears in queue depth.

5. **Train/serve feature parity** — `feature_builder.py` is shared between cache baking and live inference; same normalization, same feature extraction logic, same edge attributes. This eliminates a common source of train/serve mismatch.

---

## 9b. What Is Archived (Not Published)

All of the following have been moved to `paper/archived/legacy_results.md`:

| Item | Reason |
|---|---|
| 150-150 / 450k task benchmarks | Based on 13-dim model with broken features |
| seqblend as a deployment policy | Post-hoc override for ranking model; not needed for CE-only |
| LQB full ablation table | Rejected for deployment; 7-config sum worse than CE |
| good-plasma-43 co-sim audit | Different training pipeline; not comparable to dim14-CE |
| silvery-sun-4 comparisons | Superseded; pre-dates all fixes |
| Track B r030 gate (post-fix 3×3) | **Rejected** — +1.9% vs CE; canonical note in `results.md` |
| Dim13 vs dim14 comparisons | Multiple confounds; not a clean ablation |
| warmth_v2 skew3 failed gate | MLP 2/3 on v2 physics without disk features — see `results.md` |

---

## 10. Open Questions (as of 2026-06-11)

| Question | Status |
|---|---|
| Can Track B beat CE-only? | **r030 rejected** post-fix (+1.9% sum) |
| Does queue-map fix make ranking shippable? | **No** — still +27% on `default` vs CE |
| warmth_v2 skew3: GNN beats MLP with v2 physics only? | **Failed** — MLP 2/3; need disk feature + hub co-sim grid |
| Does seq_reforward beat Knative on all k6 configs? | **Partial** — seek65 yes (−3.3%); seek50 no (+7.1%) |
| Hub co-sim grid (`degree_skewed_core`) in training? | **Not started** |
| Can GIN generalize to 50×50 without retrain? | Not tested |
| Is XGB competitive on non-default configs? | Triangle all7 tested (+70% vs GNN); hubs not tested |

---

*Last updated: 2026-06-11 | Paper v0.28 — bipartite primary, warmth skew3 gate, mechanism guardrails*
