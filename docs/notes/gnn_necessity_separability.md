# GNN Necessity & Co-Sim Label Separability

> **Created:** 2026-06-15 · **Owner question (RQ3):** Is a GNN architecturally necessary, or
> does a pointwise edge MLP match it? **Answer:** The original unique-replica training labels were
> **per-task separable**, so a pointwise scorer was Bayes-optimal. The non-unique backfill reveals
> coupled tails in every retained corpus (4.2–10.3% at >1% greedy regret), while the controlled
> contention-v2 ablation shows the GNN's advantage is tail robustness rather than average capacity.
> This doc records the diagnostic, fix, corpus, and ablation — all with numbers.

---

## 1. Root cause: the oracle produced separable labels

`scripts_cosim/separability_diagnostic.py` quantifies *irreducible joint coupling* in
`placements/placements.jsonl` per `ds_*`:

- **M1 — marginal/greedy regret:** build each task's best platform independently (min RTT over all
  plans containing that task→platform), assemble the marginal plan, look up its true RTT, compare to
  the optimum. `0%` ⇒ independent per-task choice is already optimal ⇒ **pointwise is enough**.
- **M2 — identical-task symmetry**, **M3 — collision in the optimum**.

**Historical finding on the training corpora (`warmth_v2`, `sparse_warmth_v2`; oracle run with unique replicas):**

| corpus | optimum collides | greedy/marginal regret |
|---|---|---|
| warmth_v2 / sparse (unique-replica oracle) | **0%** | **0.00%** |

**Final non-unique audit (2026-08-04):** strict inventory retained **487 warmth**, **351 sparse**,
**899 contention_v2**, and **900 contention_v3** full sweeps. Five corrupt/truncated sweeps and nine
never-generated warmth cells are explicitly excluded; exact duplicates in warmth `ds_00158/00188`
were removed with backups. Frozen report:
`simulation_data/separability_audit_4corpus_20260804.json`.

| retained corpus | optimum collides | greedy regret mean / p90 / p99 / max | coupled >1% / >5% / >10% |
|---|---:|---:|---:|
| warmth_v2 (487) | **39.8%** | 0.51% / 0.36% / 9.77% / 13.9% | **9.7% / 4.7% / 0.8%** |
| sparse_warmth_v2 (351) | **33.0%** | 0.71% / 1.13% / 12.37% / 17.0% | **10.3% / 5.7% / 2.6%** |
| contention_v2 (899) | **40.3%** | 0.33% / 0.00% / 7.93% / 22.4% | **7.1% / 2.4% / 0.3%** |
| contention_v3 (900) | **36.3%** | 0.16% / 0.00% / 4.80% / 9.5% | **4.2% / 0.9% / 0.0%** |

The stale 0% collision / 0% regret figures apply **only to the pre-backfill unique-replica pass**,
not to the current retained placement inventories.

**Label-integrity blocker (2026-08-04):** the sweep audit above is valid, but current graph-cache
labels are not uniformly derived from those sweep optima. `prepare_graphs_cache.py` reads
`optimal_result.json` `sample.placement_plan`; after backfill, the retained labels are:

- warmth: **277/487 sweep-optimal**, **210 nonoptimal**;
- sparse: **185/351 sweep-optimal**, **166 nonoptimal**;
- contention_v2: **855/899 sweep-optimal**, **18 nonoptimal**, **26 absent from the sweep**.

The strict cache `simulation_data/graphs_cache_contention_v2_899_20260804` was built and verified
(899 graphs, 3,396,943 RTT rows), but the controlled rerun failed loudly after the pointwise
120-epoch phase because five held-out predictions matched absent oracle-label plans. No frozen
899-result JSON exists; no retraining or live gate was accepted. The 26 absent plans coincide with
the manifest's `best_json_rtt_mismatch` set, so placement-file cleanliness must not be interpreted as
label/cache cleanliness. Logs:
`logs/{prepare_graphs_cache,gnn_necessity_ablation}_contention_v2_899_20260804.log`.

**Why:** `generate_gnn_datasets_fast.py` `--allow-non-unique-replicas` is `action='store_true'`,
i.e. defaults **False**. The brute-force enumerator therefore only emits placements with **distinct
platforms** → collisions never enter the label space → the optimum can always be reached by
independent per-task argmax → **MLP == GNN by construction**. This is the mechanistic explanation
behind the prior RQ3 result ("dim22 batch MLP matches CE-only GNN, +0.4% sum").

---

## 2. The fix is in the label space, not the model

Making collisions *frequent* is necessary but **not sufficient**:

- **`contention_v1`** (all-cold replicas, sparse, heavy queues, `--allow-non-unique-replicas`):
  optimum collides **~43%**, but **greedy regret stayed ~0%**. Reason: when colliding *is* optimal,
  both tasks already independently prefer that platform — no coupling to learn.
- **Greedy only breaks** when two tasks share their #1 platform **AND co-locating there is
  expensive**, forcing the optimum to *split* them (anti-correlated preferences). That needs a
  **scarce attractive resource**: warm replicas (a few platforms clearly best) + heavy/pre-loaded
  queues (stacking serializes and destroys the advantage) + sparse topology (few fallbacks).
  → **`contention_v2`**.

Both grids live in `scripts_cosim/generate_gnn_datasets_fast.py` (`--grid contention_v1|contention_v2`).
**Must** be generated with `--allow-non-unique-replicas`.

---

## 3. `contention_v2` corpus (datalab, 2026-06-15)

- Grid: conn∈{0.25,0.35} × rep∈{(1,1,.7,.9),(1,2,.7,.9),(2,2,.5,.7)} × queue∈{norm35,uniform20_80,pois28} × seeds 301–350 → 900 cells.
- Generated on datalab SLURM array (multi-partition GPU-a40/a100s/l40s/a100). **899 retained**
  `placements.jsonl` after integrity audit (`ds_00751` excluded as truncated; metadata recorded 7,838
  completions but only 1,922 valid local rows).
- `--allow-non-unique-replicas` ON; parallel-safe per-shard `workload-10_<job>_<task>.json` (race fix).
- Pulled to `simulation_data/gnn_datasets_4tasks_contention_v2/`; SSC via
  `refresh_optimal_full_stats.py --rewrite-ssc` (711 ok on first pull; **recache pending** for +189 ds); cache
  `simulation_data/graphs_cache_contention_v2` (CACHE_VERSION 5.4, **711 graphs** — stale vs corpus).

**Final corpus separability (899 retained sweeps, 2026-08-04):**

| metric | value |
|---|---|
| optimum collides | **40.3%** (362/899) |
| greedy regret mean / median / p90 / p99 / max | 0.33% / 0.00% / 0.00% / 7.93% / **22.4%** |
| coupled (>1% / >5% / >10%) | **7.1%** / 2.4% / 0.3% |

⇒ Even in the contention-tuned regime **~93% of datasets remain separable**. The GNN advantage is
real but lives in the **tail/coupled minority**, which bounds the *average* lift.

---

## 4. Ablation: pointwise vs GIN vs GIN+node-edges

`scripts_cosim/gnn_necessity_ablation.py` — identical task/platform/edge features, identical CE loss,
identical train/test split, only the scorer differs:
- **pointwise** — per-edge MLP, no message passing (MLP-equivalent).
- **gnn_base** — GIN over the bipartite task↔platform graph.
- **gnn_node** — gnn_base + same-node platform↔platform edges (`node_edge_index`), residual GIN,
  candidate-restricted edges.

**Result (`contention_v2`, 711 ds, test n=142, 120 epochs, seed 42):**

| model | top-1 | regret mean | regret p90 | regret max | opt-recovery |
|---|---|---|---|---|---|
| pointwise | 89.1% | 22.99%¹ | 2.81% | **3164%**¹ | 64.8% |
| **gnn_base** | **90.7%** | **0.61%** | **2.39%** | **7.03%** | **69.0%** |
| gnn_node | 89.6% | 0.99% | 3.58% | 16.91% | 67.6% |

¹ **Mean regret is not robust on this corpus**: optima are tiny (opt_rtt≈2 units), so a single
catastrophic collision plan blows the relative error to 3164% (p90 is only 2.81%). The greedy
*oracle* caps at 22.4%; the *trained* pointwise model does worse because, lacking batch context, it
emits a collision-cliff plan on one dataset. **Robust comparators = top-1, p90 regret,
opt-recovery — gnn_base wins all three and bounds the tail (max 7% vs 3164%).**

**Coupled subset (greedy regret >1%, n=12, noisy):** pointwise 3.08% · gnn_base 3.42% · gnn_node
4.88% · greedy 4.69%. Small-n; all learned models beat greedy in aggregate but per-subset ranking is
unstable across corpus sizes (572-ds run had gnn_base < pointwise here).

---

## 5. Conclusions (grounded)

1. **The GNN's advantage is collision/contention robustness, not average accuracy.** On separable
   labels MLP==GNN; on contention labels gnn_base wins top-1 (+1.6pt), p90 regret, opt-recovery, and
   eliminates the catastrophic tail the pointwise model exhibits.
2. **Node-aggregation edges (`gnn_node`) add nothing here.** Same-node platform↔platform edges are
   **redundant with `shared_fate_signal`** (platform feat dim 8 = per-node cold density), and they
   destabilize training unless restricted to candidate platforms + given a residual. Keep them off
   by default; revisit only if `shared_fate` is removed or node-level contention is richer.
3. **The environment is still mostly separable (~93%).** To make the GNN *decisively* win on
   average, push the corpus further into coupling: lower conn (sparser fallbacks), bump batch size
   b>4 so more tasks contend, raise queue load, and over-sample the (warm, heavy-queue) cells.

---

## 6. Files & repro

- Grids / generation: `scripts_cosim/generate_gnn_datasets_fast.py` (`CONTENTION_V1_GRID`,
  `CONTENTION_V2_GRID`, `--allow-non-unique-replicas`, per-shard workload paths).
- Diagnostic: `scripts_cosim/separability_diagnostic.py`.
- Ablation harness: `scripts_cosim/gnn_necessity_ablation.py`.
- Architecture: `src/policy/gnn/gnn_model.py` (`build_same_node_edge_index`, residual GIN, defensive
  edge-scoring mask); cache `src/notebooks/prepare_graphs_cache.py` (CACHE_VERSION 5.4,
  `data.node_edge_index`).
- Corpus: `simulation_data/gnn_datasets_4tasks_contention_v2/` (**899 retained** jsonl);
  strict cache `simulation_data/graphs_cache_contention_v2_899_20260804/` (**899 graphs**; ablation
  blocked by 26 absent labels). Historical deployed models and the 711-graph ablation remain stale.
- Frozen four-corpus audit: `simulation_data/separability_audit_4corpus_20260804.json`;
  integrity manifest: `simulation_data/placement_integrity_manifest.json`.
- Non-unique backfill (warmth/sparse): `scripts_cosim/generate_non_unique_placements_fast.py --datasets-dir`,
  datalab `scripts_cosim/datalab/warmth_non_unique_{warmth,sparse}.sbatch`, monitor `cosim_health_report.sh`.
- Repro ablation: `pipenv run python3 scripts_cosim/gnn_necessity_ablation.py --cache
  simulation_data/graphs_cache_contention_v2 --corpus-root simulation_data --epochs 120 --seed 42`.

---

## 7. Open next steps

- ~~**Live gate**~~ **DONE 2026-06-15** — see §8 below. Train/serve gap persists on per-config wins; sum-level GNN advantage holds.
- ~~**Merge deployment gates**~~ **CLOSED 2026-06-27 (mitrix)** — see §10; compare tables in sweep `compare.txt`; `phase_all.done`.
- ~~**Build strict contention_v2 cache**~~ **DONE; 899 retained graphs**. **BLOCKED:** repair graph
  labels to use sweep-min plans and rebuild before rerunning the ablation; the current strict cache
  reproduces 26 labels absent from its RTT sweeps.
- ~~**Four-corpus separability re-audit**~~ **DONE 2026-08-04** — final table in §1; frozen JSON report and integrity manifest recorded above.
- **contention_v3 artifacts pull (optional):** corpus local; cache/models/live gate on datalab — rsync or local `run_contention_v3_train_and_live_gate_nohup.sh`.
- **Push coupling:** v3 (conn 0.15/0.20 + heavier queues) **did not** increase coupling (4.2% vs v2's 7.2%) — need different levers (batch size b>4, anti-correlated task types, live-queue-calibrated co-sim).

---

## 8. Live gate (2026-06-15)

**Models trained on `graphs_cache_contention_v2` (711 graphs):**
- GNN: `models/near-rtt-v2-contention-v2-dim14-ce-only.pt` (val acc **66.2%**, CE-only 100ep)
- MLP: `models/tabular/batch_edge_mlp_contention_v2_dim22_batchcache.pt` (val edge acc **90.0%**, early stop ep30)

**Sweep:** `contention_v2_live_gate_20260615/` · `node_disk_v2` · workload-125-225 · 3 sparse configs:

| config | Knative | GNN | MLP | winner |
|---|---:|---:|---:|---|
| sparse_p25 | 6.90M | 7.66M (1.11×) | **5.81M (0.84×)** | MLP |
| sparse_p35 | 12.77M | **11.08M (0.87×)** | 16.79M (1.31×) | GNN |
| sparse_p25_skew | **1.06M** | 1.14M (1.07×) | 2.75M (2.60×) | Kn |
| **SUM** | 20.73M | **19.88M (0.96×)** | 25.35M (1.22×) | **GNN sum** |

**Wins: GNN 1/3 · MLP 1/3 · Knative 1/3.** GNN beats Knative and MLP on **3-config sum** (−4% vs Kn, −22% vs MLP), but per-config split is inconclusive. Offline ablation GNN advantage does **not** uniformly transfer to live (MLP wins sparse_p25; train/serve gap + argmax decode).

**Scripts:** `run_contention_v2_train_and_live_gate_nohup.sh`, `compare_contention_v2_live_gate.py`

---

## 9. contention_v3 corpus + live gate (datalab, 2026-06-15 / 2026-06-20)

Grid: conn∈{0.15,0.20} + heavier queues (norm40, uniform25_90, pois32) · **900/900 ds DONE**.

| metric | v2 (711 ds) | v3 (900 ds) |
|---|---|---|
| optimum collides | 41.2% | **36.3%** |
| coupled (>1%) | **7.2%** | 4.2% |
| greedy regret max | 22.4% | 9.5% |

**Offline hypothesis rejected:** sparser topology + heavier queues **decreased** coupling vs v2.

**End-to-end pipeline (2026-06-20, datalab):** `graphs_cache_contention_v3` (900 graphs) · GNN val **65.6%** · MLP val edge **88.2%** · live gate `contention_v3_live_gate_20260620/` (9/9, workload-125-225, node_disk_v2, argmax):

| config | Knative | GNN v3 | MLP v3 | winner |
|---|---:|---:|---:|---|
| sparse_p25 | 7.05M | 8.26M | 9.52M | Kn |
| sparse_p35 | 12.00M | 26.45M | 33.16M | Kn |
| sparse_p25_skew | 1.22M | 1.29M | **1.08M** | MLP |
| **SUM** | **20.27M** | **36.00M** | **43.76M** | **Kn sum** |

**Wins:** Kn **2/3** · MLP **1/3** · GNN **0/3**. vs contention_v2 GNN sum **19.88M**: v3 **+81% worse**. sparse_p35 cliff amplified (GNN 26.4M vs v2 11.1M). **Do not ship v3**; keep contention_v2 GNN for sparse ER deploy.

**Scripts:** `transfer_contention_v3_pipeline_to_datalab.sh`, datalab `contention_v3_{recache,gnn_train,mlp_train,live_gate_gpu,compare}.sbatch`, `compare_contention_v2_live_gate.py`

---

## 10. Merge deployment live gates (2026-06-16, mitrix)

Two merged training runs — **strategic merge** (coupled oversample manifest, no warmth v3/skew) and **weighted merge** (8× coupled threshold on warmth+sparse+contention_v2) — evaluated on `node_disk_v2`, workload-125-225, seed 42.

### 10a. Strategic merge (warmth + sparse + contention_v2, strategic oversample)

**Train:** GNN val acc **48.8%** · MLP val edge acc **78.5%** · cache `graphs_cache_strategic_merge_wss_cont_v2`

**WSSM hub gate** (`strategic_merge_wss_live_gate_20260616/`, hub k4/k6/k8 @ seek50):

| config | Knative | GNN | MLP | winner |
|---|---:|---:|---:|---|
| hub_k4_seek50 | 686k | 1.95M | 1.23M | Kn |
| hub_k6_seek50 | 646k | 2.38M | 1.39M | Kn |
| hub_k8_seek50 | 582k | 1.60M | 718k | Kn |
| **SUM** | **1.91M** | **5.93M** | **3.34M** | **Kn 3/3** |

**Contention sparse gate** (`strategic_merge_contention_live_gate_20260616/`):

| config | Knative | GNN | MLP | winner |
|---|---:|---:|---:|---|
| sparse_p25 | 8.03M | 8.77M | 12.70M | Kn |
| sparse_p35 | 10.57M | 11.24M | 68.38M | Kn |
| sparse_p25_skew | 1.72M | 3.14M | 4.61M | Kn |
| **SUM** | **20.32M** | **23.15M** | **85.69M** | **Kn 3/3** |

**vs contention-only baseline** (`contention_v2_live_gate_20260615/`, §8): same 3 sparse configs, contention-only cache — GNN sum **19.88M** beat Kn **20.73M** and MLP **25.35M**. Strategic-merge GNN sum **23.15M** (+17% vs contention-only GNN); MLP sum **85.69M** is catastrophic (+238% vs contention-only MLP). Knative unchanged (~20.3M). **Train/serve gap:** higher offline GNN acc (48.8% vs 66.2%) does not improve live sum; merged oversample hurts MLP on sparse_p35.

### 10b. Weighted merge (8× coupled oversample on warmth+sparse+contention_v2)

**Train:** GNN val acc **26.3%** · MLP val edge acc **72.3%** · cache `graphs_cache_warmth_sparse_contention_v2_weighted`

**Contention sparse gate** (`merged_contention_weighted_live_gate_20260616_105500/`):

| config | Knative | MLP | gnn_uniq | gnn_argmax | best |
|---|---:|---:|---:|---:|---|
| sparse_p25 | 7.53M | 14.91M | 9.14M | 9.30M | kn |
| sparse_p35 | 12.29M | 64.89M | **11.34M** | 12.17M | gnn_uniq |
| sparse_p25_skew | 1.33M | 1.84M | **1.82M** | 3.12M | kn |
| **SUM** | **21.14M** | **81.64M** | **22.30M** | 24.59M | kn sum |

**Wins:** kn=2 · gnn_uniq=1 · mlp=0 · gnn_argmax=0. **gnn_uniq** beats argmax on sum (22.30M vs 24.59M) and wins sparse_p35; still loses to Knative on 3-config sum (+5%). MLP again collapses on sparse_p35 (64.9M).

**Scripts:** `run_mitrix_remaining_learnable_nohup.sh`, `compare_wssm_expanded_live_gate.py`, `compare_contention_v2_live_gate.py`, `compare_merged_contention_live_gate.py`, `finish_live_gate_sweeps.sh`

**Closure (2026-06-27):** all 30 sim JSONs validated on mitrix; compare archived to each sweep's `compare.txt`; phase markers `logs/strategic_merge_pipeline/phase_all.done` + `logs/merged_contention_pipeline/phase_all.done`.
