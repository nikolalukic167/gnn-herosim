# Research Questions

> The paper's RQ structure. Update if scope changes or new experiments open/close an RQ.
> Each RQ has a status: ANSWERED / PARTIAL / OPEN.

## Two-regime freeze (do not mix)

The paper has **two evaluation regimes**. Do not promote Regime B numbers into the main
scoreboard, and do not use `total_rtt` as the Regime B primary.

| | **Part A — live gates** | **Part B — FilterStore case study** |
|---|---|---|
| RQs | RQ1–RQ3, RQ3b, RQ8 (RQ4–RQ7 methods) | **RQ9** |
| Workloads | 100-100 / 125-225 | N=12 cold burst stub |
| Metric | `total_rtt` (primary), p90/p99 (secondary, RQ3/RQ3b) | max-burst elapsed |
| Deploy GNN | 873/v5.5 dim14 **argmax** | distill dim24 + `seq_reforward_pull` (instrument only) |
| Write-up | main experiments | dedicated case-study subsection |

Case study text: `paper/case_study_regime_b.md`.

---

## Primary Research Questions

### RQ1: Can a GNN trained on co-simulation oracle data generalize to live scheduling in full serverless simulation?

**Status: PARTIAL** (7/7 is pre queue-map-fix; canonical post-fix coverage is 3/7)

The CE-only GNN (`near-rtt-v2-dim14-ce-only.pt`, 1060 cache) achieved 7/7 wins vs Knative,
HRC, Random, and RoundRobin on the pre-fix 7-config sweep. The canonical queue-map-corrected
stack has only been checked on `default`, `02`, and `04`; it confirms CE-only remains best vs
ranking and Track B r030 on that 3-config set, but does not yet replicate the 7/7 claim.

**Not claimed:** universal dominance vs batch MLP (triangle all7 +0.4% MLP sum) or on warmth_v2 skew3 live gate (MLP 2/3) — see RQ3 and RQ8.

**Evidence:** `gnn_near_rtt_v2_dim14_ce_only_20260609/` — 7/7 vs Knative (`default` Kn from `baseline_default_100100/`,其余 from `knative_network_20260606_192413/`) + HRC; `random_rr_3cfg_20260609/` — CE-only 7/7 vs Random (+250–455%) and RoundRobin (+78–124%); Knative +25–36% vs CE.

**Paper claim (bounded):** The model generalized across seven tested connectivity regimes under
the original inference stack and retained its relative objective advantage on three post-fix
configurations. Full post-fix seven-config and multi-seed replication remain pending.

---

### RQ2: Does the training objective choice determine live deployment quality for GNN task schedulers?

**Status: ANSWERED**

Strongly yes. Pairwise combo-sum margin ranking and CE-only classification use the same 14-dim cache, same architecture, and the same argmax deployment. Pre-fix inference (7-config sweep):

| Objective | default RTT | qvm p95 | vs Knative |
|---|---|---|---|
| CE-only | 4.14M | 408 | wins |
| Pairwise ranking | 11.62M | 2857 | loses (+181%) |

**Post queue-map fix** (3-config gate, same checkpoints, Option 1 inference):

| Objective | default | 02 | 04 | 3cfg sum |
|---|---:|---:|---:|---:|
| CE-only | **4.31M** | **1.91M** | 1.80M | **8.03M** |
| Pairwise ranking | 5.48M | 1.94M | **1.78M** | 9.20M |
| Track B r030 | 4.45M | 1.95M | 1.78M | 8.18M (+1.9% vs CE) |

Ranking sparse collapse drops from 11.62→5.48M when queue inputs are corrected (−53%), but CE-only still wins 3cfg sum; ranking remains +27% on `default`. Track B r030 **rejected**.

**Root cause:** Ranking optimizes `combo_score = Σlogits` (joint) → deployment applies `argmax(logits[t])` per task (marginal). CE optimizes `CrossEntropy(logits[t], y[t])` per task (marginal) → deployment applies `argmax(logits[t])` per task (marginal). CE training is structurally aligned with inference by construction; ranking is not.

**Post-hoc remedy evidence:** Temperature T=3 applied to ranking model at inference reduces `default` RTT only from 11.62M to 10.74M (−7.6%), confirming logit over-sharpening cannot be repaired post-hoc because temperature preserves the relative ordering that caused the hot-spotting. Full ablation data in `archived/legacy_results.md`.

**Paper claim:** The structural mismatch between joint training objectives and decomposed marginal inference is the primary failure mode of GNN-based schedulers, not model capacity, feature quality, or distribution shift alone.

---

### RQ3: Is a GNN architecturally necessary, or can tabular edge-rankers achieve comparable performance?

**Status: ANSWERED**

**Claim:**
1. **Offline:** On repaired contention labels (CACHE 5.5, n=873, 3 seeds), a GIN beats a
   feature-matched pointwise MLP on top-1 / p90 / opt-recovery and removes collision-cliff
   failures.
2. **Live:** That offline edge does **not** carry to sealed full-sim holdout (unused topologies
   × 5 seeds): MLP wins **13/20** cells; GNN **1/20**; Knative **6/20**. Sums: MLP **0.82×** Kn,
   GNN **1.05×**.

**Not claimed:** GNN is required for live deploy on these traces; merge/v3 recipes help; the old
711-era 1/1/1 development gate is the sealed answer.

**Evidence:** offline
`simulation_data/gnn_necessity_ablation_contention_v2_873_v5.5_multiseed.json`; live
`simulation_data/normal_sim_sweeps/contention_v2_873_v5.5_sealed_holdout_20260806/compare.json`.

#### Why offline mattered

Early warmth/sparse labels used unique-replica enumeration
(`--allow-non-unique-replicas` default False), so collisions never entered the label space and
MLP≈GNN was an oracle artifact. CACHE 5.5 labels from `placements.jsonl` sweep minima; contention_v2
retained **873** coupled graphs. Offline ablation (parent 70/15/15, seeds 42/43/44):

| model | top-1 | regret p90 | regret max (per-seed) | cliff seeds (max>100%) |
|---|---|---|---|---|
| pointwise (MLP) | 87.1±0.9% | 3.88±0.53% | **1578%** / 13.9% / 14.9% | **1/3** |
| **gnn_base** (GIN) | **88.9±0.8%** | **3.03±0.55%** | **27.8%** / 13.9% / 16.9% | **0/3** |

#### Sealed live holdout

Development trio (`sparse_p25` / `p35` / skew) stays development-only. Sealed cells: unused
topologies `balanced_p50`, `balanced_p60`, `client_heavy_p50`, `server_heavy_p50` × seeds 42–46 ×
CE-GNN / MLP / Kn (`node_disk_v2`, workload-125-225). Models: 873/v5.5 deploy ckpts.

| config | Kn mean | GNN mean | MLP mean | wins (G/M/K) |
|---|---:|---:|---:|---|
| balanced_p50 | 5.98M | 6.71M | **4.58M** | 0/5/0 |
| balanced_p60 | **4.66M** | 5.28M | 4.94M | 0/1/4 |
| client_heavy_p50 | 8.30M | 8.15M | **5.33M** | 0/5/0 |
| server_heavy_p50 | 3.67M | 3.66M | 3.77M | 1/2/2 |
| **paired cells** | — | — | — | **1 / 13 / 6** |

**Conclusion:** Offline GNN>MLP on coupled labels is real; it does not justify claiming GNN for
live deploy on this sealed set.

#### Tail latency on the sealed set (added 2026-08-13)

`total_rtt` is a sum and hides the collision cliff the offline ablation predicted, so the same 60
runs were re-scored for response-time quantiles (`scripts_cosim/sweep_metrics.py`). Means over
seeds 42–46:

| config | Kn p99 | GNN p99 | MLP p99 | p99 winner |
|---|---:|---:|---:|---|
| balanced_p50 | 58.3 | 48.7 | **25.0** | MLP |
| balanced_p60 | 36.1 | 25.3 | **24.4** | MLP |
| client_heavy_p50 | 90.6 | 57.2 | **31.6** | MLP |
| server_heavy_p50 | 33.1 | 25.5 | **24.4** | MLP |

MLP wins p99 **4/4**, consistent with its `total_rtt` result — the tail does not rescue the GNN on
these sealed (largely uncoupled) topologies. **GNN does beat Knative on p99 4/4**, so the GNN's
learned signal is real but weaker than the MLP's here.

**Comparability caveat:** this sweep ran 2026-08-06, before `25732cf` serialized co-located cold
pulls per node. Both tables above are pre-serialization. A HEAD re-baseline of the identical 60
cells is in `contention_v2_873_v5.5_sealed_holdout_rebaseline_20260813/`; cite the two records
separately and never in one table.

#### Earlier probes (footnotes)

- 711-era development gate `contention_v2_live_gate_20260615/`: per-config **1/1/1** (not sealed).
- Strategic / weighted merge live gates: **rejected** (worse vs contention-only).
- contention_v3 train+live: **rejected** (GNN sum +81% vs v2).
- Hub / bipartite / skew grids: regime-dependent; with feature-parity batchcache MLP often wins —
  see `docs/notes/compare.md`. Not the sealed RQ3 answer.

---

### RQ3b: On *coupled* topologies, does the GNN remove the MLP's tail catastrophe?

**Status: ANSWERED — FAIL (2026-08-13).** Pre-committed criteria not met. The pre-serialization n=1 MLP cliff does not replicate. RQ3's negative live answer stands on HEAD physics, stronger. **Addendum (same day):** the original run carried a queue-feature train/serve magnitude mismatch; fixing it (`scale_invariant_v1`) makes the **MLP beat Knative 13/15 cells and p99 3/3**, while the **GNN still loses 0/15** — see "Post-mortem" below. RQ3b's verdict is unaffected (it asks whether the GNN beats the MLP's tail), but the MLP result is new and needs a sealed coupled holdout.

**Motivation (the observation that opened this):** re-scoring the 711-era development gate
`contention_v2_live_gate_20260615/` for quantiles inverts the sealed-set ordering on the sparsest
cell:

| cell | metric | Kn | GNN | MLP |
|---|---|---:|---:|---:|
| sparse_p35 | total_rtt | 12.77M | **11.08M** | 16.79M |
| sparse_p35 | p99 | 164.7 | **110.2** | **783.5** |
| sparse_p25 | total_rtt | 6.90M | 7.66M | **5.81M** |
| sparse_p25 | p99 | 120.8 | 98.2 | **53.2** |
| sparse_p25_skew | p99 | 27.3 | 25.4 | **24.5** |

The MLP's 783.5s p99 on `sparse_p35` is **7.1× the GNN's** and is the live analogue of the
pointwise model's 1578% offline max-regret cliff — the exact failure the GIN removed offline. But
this is **n=1 seed**, on **pre-serialization physics**, from a **development** (non-sealed) gate,
and the direction is not consistent across the trio. It is a hypothesis, not a result.

**Hypothesis:** the GNN's advantage is *tail robustness under coupling*, not mean objective. Where
the optimum forces tasks apart (sparse fallback + scarce warm replica + heavy queue), the pointwise
model occasionally piles a batch onto one platform and pays a FilterStore-serialized cliff; the GIN
sees `shared_fate` and splits.

**Pre-registered design (fixed before any result was read):**

- Cells: `sparse_p25`, `sparse_p35`, `sparse_p25_skew` — the coupled development trio.
- Seeds: 42, 43, 44, 45, 46. Policies: Knative / batch-edge MLP / CE-GNN. **45 runs.**
- Models: 873/v5.5 deploy ckpts (`…-contention-v2-873-v5.5-dim14-ce-only.pt`,
  `batch_edge_mlp_contention_v2_873_v5.5_dim22_batchcache.pt`). Workload `workload-125-225`.
- Physics **pinned**: `HEROSIM_WARMTH_PHYSICS=node_disk_v2` with
  `HEROSIM_REQUIRE_EXPLICIT_PHYSICS=1` (fail loud, no implicit default).
- Reported metrics fixed in advance: `total_rtt` **and** p50/p90/p99, per cell, mean over seeds.
- Runner: `scripts_cosim/important/run_contention_v2_873_coupled_trio.sh` → sweep
  `contention_v2_873_v5.5_coupled_trio_20260813/`.

**Success criterion (pre-committed):** GNN p99 beats MLP p99 on **≥2 of 3 cells**, with the
`sparse_p35` MLP cliff (p99 > 3× GNN p99) replicating on **≥3 of 5 seeds**. Anything less means the
cliff was a single-seed artifact and RQ3's negative live answer stands unchanged.

**Interpretation limits agreed up front:** a PASS supports a *tail-robustness on coupled
topologies* claim only. It does **not** reopen the sealed `total_rtt` verdict, and these cells were
used during development, so they cannot become the headline sealed result — a PASS would require a
fresh sealed coupled holdout before any paper claim.

**Result (post-`25732cf`, `node_disk_v2`, 873/v5.5, seeds 42–46):**

| cell | Kn rtt | GNN rtt | MLP rtt | Kn p99 | GNN p99 | MLP p99 |
|---|---:|---:|---:|---:|---:|---:|
| sparse_p25 | **16.54M** | 115.0M | 41.5M | **183** | 1650 | 1389 |
| sparse_p35 | **24.21M** | 185.0M | 118.7M | **254** | 2929 | 2889 |
| sparse_p25_skew | **4.14M** | 36.7M | 8.9M | **130** | 723 | 440 |

PRIMARY 0/3 · TAIL 0/3 · `sparse_p35` cliff seeds 0/5. Kn wins every cell. GNN `sparse_p35` s42 `averageQueueTime` ~503s / pull ~0.017s — pile onto serialized FilterStore, not a wiring bug. Sealed re-baseline of the 60 unused-topology cells on the same physics: **G/M/K = 0/0/20** (was 1/13/6). Do not mix with `…_sealed_holdout_20260806`.

Full paired stats in `contention_v2_873_v5.5_coupled_trio_20260813/compare.json` (15 paired cells) and
`…_sealed_holdout_rebaseline_20260813/compare.json` (20 paired cells); each sweep's `manifest.json`
pins physics, code commit `040fdcb`, and checkpoint md5s, and the re-baseline manifest records
`incomparable_with` the pre-serialization record.

**Post-mortem changed the diagnosis, and re-running under the fix flips the MLP arm
(2026-08-13).** The rows above were produced with a train/serve *magnitude* mismatch, not a
policy failure: platform queue depth reaches both models only through dim 7
(`raw_q / clamp(p90(depths), 1, 100)`) and dim 13 (`(raw_q / target_concurrency) / 5`). The cap
binds **0/200** times in training (per-dataset p90 depth 26–70) and **always** live (chosen-platform
depth ≈ 14.5k), so dim 7 arrived ~100× and dim 13 ~360× outside the training manifold with no
clipping anywhere — a silent shift. Fix: `src/placement/queue_features.py` with versioned contracts,
`scale_invariant_v1` uncapping the dim-7 divisor and log1p-compressing dim 13; the same 873 labels
were recached (`CACHE_VERSION` 5.7) and both models retrained with identical splits and
hyperparameters. `legacy_v0` reproduces the deployed v5.5 cache numerically, so the two arms differ
only in queue-feature scaling.

**Re-run (`contention_v2_873_v5.7_siv1_coupled_trio_20260813/`, seeds 42–46, `node_disk_v2`,
Knative arm reused verbatim since it reads no checkpoint):**

| cell | Kn rtt | MLP rtt | mlp/kn | cells won | Kn p99 | MLP p99 | GNN rtt |
|---|---:|---:|---:|---:|---:|---:|---:|
| sparse_p25 | 16.54M | **13.07M** | 0.79× | 4/5 | 183 | **108** | 312.8M |
| sparse_p35 | 24.21M | **17.99M** | 0.74× | 5/5 | 254 | **145** | 307.6M |
| sparse_p25_skew | 4.14M | **3.73M** | 0.90× | 4/5 | 130 | **103** | 56.0M |

The MLP wins **13/15 paired cells** (SUM 0.775×) and **p99 in 3/3 cells**; p90 2/3 (loses
`sparse_p25_skew`, 22.4 vs 21.6). It is not a disguised shortest-queue rule: `chosen_queue_vs_min`
median *rises* (9→36 on `sparse_p35`) while p95 collapses (12668→980), i.e. it still deviates
deliberately but no longer saturates. Two cells still lose — `sparse_p25` s45 (1.07×) and
`sparse_p25_skew` s43 (**2.14×**, which is the entire ±2.17M std of that config) — so the supportable
claim is **"beats Knative on seed-averaged `total_rtt` and on p99, not uniformly per seed."**

**The GNN's failure is separate and is neither the features nor model quality: 0/15 cells, SUM
15.07× Knative even under the fix.** Init-seed variance with the canonical-parent split held fixed
(n=3 per contract) gives test accuracy `legacy_v0` 55.5% ±4.2 (52.7/53.4/60.3) vs
`scale_invariant_v1` 53.9% ±2.5 (51.1/55.0/55.7) — overlapping ranges, so the deployed v5.5 GNN's
60.3% was the top of its own seed distribution and the retrained GNN is within noise. What worsens
live is its *joint* decode: `intra_batch_platform_collisions` rises 0.179→0.236 and
`chosen_queue_vs_min` median 734→3676. Any further GNN attempt must target the 4-task argmax
coupling, not input scaling or more offline accuracy.

**Status of the RQ3b claim itself is unchanged (FAIL):** the pre-registered question was whether the
*GNN* removes the *MLP's* tail catastrophe, and the GNN loses every cell under both contracts. The
new result is an *MLP-vs-Knative* finding on coupled cells, and these cells were used during
development, so it needs a fresh sealed coupled holdout before it can carry a paper claim.

---


### RQ8: Does v2 warmth physics + disk/hub observability enable GNN over MLP on skew live configs?

**Status: PARTIAL — hub topology gate PASSED (bipartite v2, 2026-06-14); skew3 flat gate still FAILED**

**Setup:** Live sim uses `node_disk_v2` (disk hit skips pull; consolidation-optimal). Training merged cache **824 graphs** (warmth + sparse ER) — **no `degree_skewed_core`**, no `node_disk_hit` in graph features, seq recache strips `src_norm`, seq `is_warm` degenerate.

**Results (`warmth_v2_*_skew3_20260611/`):**
- Old models on v2 physics: MLP wins **2/3**
- Ce-reduced v2 GNN: sparse config **1.23M** vs Kn **0.93M** (regression)

**Results (2026-06-14 mega-compare):**
- **Bipartite v2 batchcache (hub topology, node_disk_v2):** MLP-bc **8/9**, sum **6.09M vs Kn 18.37M (−67%) vs GNN 10.02M (−39%)** — hub gate **PASSED** (MLP preferred over GNN)
- **Skew3 wssm gate (flat topology, node_disk_v2):** GNN wssm **1/3** · Kn 2/3 — gate **FAILED**; GNN wssm loses `default` (+70.6% vs Kn)
- **Skew4 wssm (125-225, node_disk_v2):** GNN wssm **1/4** (only `degree_skew`) — gate **FAILED**; regresses on uniform configs
- **MLP wssm:** Broken across all topologies (ce_reduced layout mismatch, catastrophic queue backlog)

**Paper claim (bounded, revised):** "Warmth-skew merged co-sim + node_disk_v2 enables **MLP dim22 batchcache 8/9 wins** on bipartite hub topologies (6.09M, −67% vs Knative). GNN wssm is secondary (10.02M). Hub superiority requires **batch-cache feature parity** (norm queue + shared_fate), not GIN. Flat/skew gates still fail (GNN 1/3 skew3). Warmth wssm **regresses on legacy all7** (+45%). Deploy MLP-bc on hub/node_disk_v2; keep dim14-ce anchor for legacy."

---

### RQ4: Can the train/serve distribution gap be fully characterized and mitigated?

**Status: CHARACTERIZED, MITIGATION SHOWN FOR CE-ONLY**

**Gap quantified** (`verify_queue_gap_1060.py`, 2026-06-10; metric = `averageQueueTime` unless noted):
- Co-sim oracle mean queue wait: 1060 max **11.5s**, p99 **5.5s**; legacy 3705 max **6.5s**
- Live mean queue wait: 150-150 Knative **52.7s** (GNN policies 45–63s); default 20×20 **20.6s**
- avgQueueTime scale: **~4.6–8×** (1060/3705 max → 150-150) — **not ~200×**
- Decode qvm p95: CE **~408** vs ranking **~2857** (platform queue task-count at inference — separate axis)
- Cold-start: co-sim task cold **~0.1%**; platform cold **`initialized_snapshot`** **~71%**; live CE `coldStartProportion` **9.6–16.4%** (prior “28% vs 88%” unverified)
- Fast-forward warmup A/B on 1060: RTT/avgQT identical ff on/off (labels not corrupted)

**CE-only mitigation:** CE training is inherently robust to this gap because it does not distort logit scales — a platform with higher queue received a lower label from the oracle (since queue inflates RTT in co-sim), so the model learned queue-aversion implicitly from oracle labels without requiring explicit queue-range generalization. When live mean queue wait is several-fold higher than co-sim labels, the relative ordering of logits — learned from co-sim queue signals — still guides the model away from overloaded platforms.

**Ranking fragility:** Ranking sharpens logit magnitudes until the fastest-hardware platform dominates all tasks regardless of queue. Pre-fix inference amplified this (11.62M `default`). Post queue-map fix: ranking `default` improves to 5.48M but still loses to CE-only (+27%). Temperature scaling and LQB partially repair pre-fix collapse but 7-config LQB sum still loses to CE (+3.7%; archived).

**Option 1 inference parity (shipped):** Full queue map at batch start improves train/serve correctness. CE-only RTT rises ~4% on corrected inputs vs pre-fix anchor; no deploy lift. Canonical cross-model comparisons must use post-fix sweeps (`dim14_3model_3cfg_queuefix_20260609/`).

**Paper claim:** "The train/serve queue-wait gap (~5–8× on mean queue time under heavy load; much larger on decode-time platform queue depth via qvm) is not mitigated by post-hoc inference corrections for ranking objectives. CE-only training implicitly acquires queue-aversion through oracle label structure, providing robustness to queue scale volatility that ranking objectives do not."

---

### RQ5: Does the 14-dim platform feature vector (with `shared_fate_signal`) improve placement under congestion?

**Status: PARTIAL — improvement observed, clean ablation pending**

**What is shown:** The 14-dim CE-only model outperforms the prior 13-dim CE-only model on `default` config (4.14M vs 4.34M, −4.5%). `shared_fate_signal` (dim 8) encodes cold replica density per node — a direct signal for shared cold-start risk within a co-located replica group.

**What is NOT cleanly ablated:** The 13-dim vs 14-dim comparison also coincides with the `is_warm` edge attribute fix and a full cache rebuild. Multiple variables changed simultaneously. A clean dim-8 ablation (14-dim model vs 14-dim model with dim-8 zeroed) has not been run.

**Paper claim (bounded):** "Expanding the platform feature space to include `shared_fate_signal` — the fraction of co-located replicas in cold-start state — correlates with improved performance on the default topology (−4.5% RTT). A clean feature ablation is left for future work."

---

### RQ6: Is brute-force co-simulation a scalable labeling strategy?

**Status: ADDRESSED**

- **Fast-forward warmup:** 23× speedup vs no-fast-forward; RTT label parity verified by A/B test (exact RTT ties)
- **4-task batch:** tractable — ~4 tasks × ~20 feasible replicas = up to ~160,000 combos max; runs in parallel via `ProcessPoolExecutor`
- **Scale limit:** 5+ task batches face combinatorial explosion — not attempted
- **Dataset scale:** warmth merged **824 graphs** + legacy 1060 **1230 ds** (eval anchor)

**Paper claim:** "Brute-force co-simulation with fast-forward warmup (23× speedup, label-parity verified) is tractable for task batches of up to 4 tasks and produces a high-fidelity labeled dataset grounded in actual system dynamics. Scalability beyond 4-task batches requires combinatorial pruning strategies not explored in this work."

---

### RQ7: Do offline validation metrics reliably predict live RTT?

**Status: ANSWERED — No, for ranking objectives; bounded yes for CE**

**Proposed alternative:** qvm p95 from a short active simulation sweep

| Predictor | CE reliable? | Ranking reliable? |
|---|---|---|
| Test top-k regret | Weak | NO (0.063s → 11.62M live) |
| Val accuracy | Moderate | N/A |
| qvm p95 (decode stats) | YES (408 → good) | YES (2857 → catastrophe) |
| Intra-batch collision rate | Moderate | YES (>15% → danger) |

**Paper claim:** "We identify a methodological divergence between offline validation and live deployment quality for GNN-based schedulers: a ranking model with 5× better offline top-5 regret produces 181% higher live RTT than the CE-only model. Compare GNN **`val/task_acc`** to MLP **`val_edge_accuracy`** (~69% vs ~68%) — not GNN **`val/acc`** (~23%) to MLP edge acc. We propose `qvm P_{95}` as a more reliable pre-deployment quality signal than static offline regression metrics."

---

### RQ9: Can static co-simulation labels resolve sequential storage contention (cold bursts), and what is required to close the gap?

**Status: ANSWERED** (bounded to constructed cell `oracle_split_v1` / `platform_reuse_v1`)

**Answer:** No — static N=4 co-sim CE labels do not close FilterStore serialization. Closing the gap requires **alignment** of dim24 sequential features, a live `ect_pull` teacher (soft KL + hard CE), and the `pulls_committed` ledger at serve time.

Write this as a **case study**, not a global victory. Full text: `paper/case_study_regime_b.md`.

**Alignment ladder (live primary = max burst elapsed):**

| Approach | Primary | Verdict |
|---|---:|---|
| Features alone (dim24 pull-obs) | 125.28s | Fail |
| Ledger alone (`seq_reforward_pull` on CE weights) | 125.28s | Fail / hurt |
| Distill without ledger (argmax) | 125.28s | Fail |
| Hard-CE distill (α=0) | 125.28s | Fail |
| Warm/busy v1 harvest | 375.87s | Catastrophic |
| **dim24 + cold soft-distill + ledger** | **31.66s** | **≡ oracle 31.65s** |

**Paper claim (verbatim):**

> On a FilterStore-serialized cold burst (N=12), a GIN trained with static N=4 co-simulation CE labels remains trapped in a pull pile (94–125s). A live greedy teacher scoring marginal FilterStore wait matches the parallel oracle (31.65s). Distilling that teacher with mixed hard CE and soft ECT-Boltzmann on cold-only sequential frames—while providing the student the identical `pulls_committed` ledger at serve time—recovers 31.66s. Removing the ledger, the soft target term, or poisoning the trajectories via busy-as-warm initialization reverts the model to the pile. This demonstrates that sequential storage costs cannot be resolved by static batch labels, but require strict alignment across train-time features, teacher cost topology, and decision-time state.

**Multi-cell (2026-08-13):** zero-jitter / latency / second burst stay at 31.66s. Arrival jitter **0.5s → 125s pile**, **2.0s → 62.63s**. Do not claim jitter robustness. Artifact: `regime_b_multicell_20260813/summary.json`.

**Not claimed:** 873/v5.5 replacement; Set Transformer falsification; jitter robustness; transfer to 100-100 / 125-225 (gate3 `/compare` running separately).

**Evidence:** `…/regime_b_phase3_ect_pull_distill_eval_multiseed_cold/summary.json`; ckpt `…-ect-pull-distill-multiseed.pt`; PR #3 `fb4e729` / merge `040fdcb`.

---

## Paper Section → RQ Mapping

| Paper section | Primary RQ |
|---|---|
| 3. Co-Simulation Oracle | RQ6, RQ8 |
| 4. GNN Architecture | — |
| 5. Training Objectives | RQ2, RQ5 |
| 6. Inference | RQ4 |
| 7. Experiments — Main results | RQ1 |
| 7. Experiments — Ablation | RQ2, RQ3 |
| 7. Experiments — Coupled-trio tail replication (pre-registered) | RQ3b |
| 7. Experiments — Bipartite v1 (hub stress) | RQ3 |
| 7. Experiments — **Bipartite v2 batchcache** (hub deploy, ★ canonical) | RQ3, RQ8 |
| 7. Experiments — Bipartite v2 GNN wssm | RQ3, RQ8 |
| 7. Experiments — warmth_v2 skew3 gate | RQ8 |
| 7. Experiments — wssm skew3/skew4 gate | RQ8 |
| 7. Experiments — **Case study: sequential cost alignment (Regime B)** | **RQ9** |
| 8. Analysis | RQ4, RQ7 |
| 9. Limitations / future work | RQ9 bounds; Set Transformer open; deploy-transfer |
| 10. Restructuring roadmap | Regime B case study **DONE**; transfer `/compare` + multi-cell in progress |
