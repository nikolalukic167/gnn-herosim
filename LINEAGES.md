# LINEAGES.md — what is current

**Read this before starting work.** It is the map of which experiment lineages are live
and which are retired. `simulation_data/REGISTRY.json` does the same for datasets; this
does it for code.

Statuses:

| Status | Meaning |
|---|---|
| `ACTIVE` | Current work. Change this code. |
| `SUPERSEDED` | Replaced by a later lineage. Result still stands; don't build on it. |
| `FALSIFIED` | Hypothesis was disproven. **Do not revive without new evidence.** |
| `PAPER` | Frozen because the paper cites it. Change only with a paper edit. |

Retired code lives in [`archive/`](archive/README.md) — moved with `git mv`, so
`git log --follow` still works. Nothing was deleted. Restore point: tag
`pre-cleanup-2026-08`.

---

## ACTIVE

| Lineage | Entry points | Datasets | Notes |
|---|---|---|---|
| **siv1_full_corpus** | `scripts_cosim/datalab/full_corpus_siv1_{recache,gnn_train,mlp_train}.sbatch` → `run_full_corpus_siv1_*.sh` | whole `legacy_v0_node_disk_v2_4task` group | Trains on the full corpus under `scale_invariant_v1`. GNN: `src/notebooks/train_near_rtt.py`. MLP: `src/policy/tabular/train_mlp_dim22_from_batch.py`. Recache: `src/notebooks/prepare_graphs_cache.py`. **Outcome 2026-08-17 — see `mp_parity` below.** **First live-gate on a real trace: FAILED (2026-08-21)** — `datalab/full_corpus_siv1_live_gate.sbatch` (array 0-14 = 3 policies × 5 cells) → `important/run_full_corpus_siv1_live_gate.sh`, cells minted by `important/make_full_corpus_siv1_gate_cells.py`, parity by `scripts_cosim/verify_live_infra_parity.py`, scored by `important/compare_sealed_live_holdout.py`. GNN loses to Knative on all 5 cells after fixing a `PYTHONHASHSEED`-dependent tie-break bug that had made the first attempt look like a sparse-topology win. See "the first real live-gate" below. **SUPERSEDED 2026-08-21 (same day) — do not cite the FAIL.** It was measured through a train/serve-divergent live feature path: the uncommitted dims 9-11 fix was absent on datalab. The **formal synced-code re-gate (job 709163, 15/15 COMPLETE)** reproduces the local re-grading on every cell to within +0.03%/+0.40%: **2W/1T/2L on `workload-125-225`**, and the same checkpoint **wins 5/5 on `workload-150-100` and 5/5 on `workload-175-100`**. `gnn/cell01` went 65.8M → 50.6M on a code change alone. Two follow-ups now open: the MLP's catastrophic tail is root-caused (occupation collapse, not collisions), and the deployed checkpoint is trained on a cache that disagrees with its serving features on **31.7% of platform rows** — corrected-cache retrain is job 709234, ungated. See the resolution subsection below. |
| **mp_parity** | `scripts_cosim/test_train_serve_mp_parity.py`, `experiments/full_corpus_siv1_gnn_mp_residual{,_node_edges}.yaml`, `datalab/mp_arm_gnn_train.sbatch` | full corpus siv1 | Train/serve message-passing parity, and what to do about it. Outcomes below. |
| **graph_structure_physics** | `scripts_cosim/separability_diagnostic.py` (M4 + `--gate-additive-r2`) | all co-sim collections | Does the simulator produce a target a GNN could ever beat a pointwise MLP on? **Outcome 2026-08-17 below: no, not today.** Phases 1-4 (node contention, congestible links, fan-out DAGs, batch size) planned against that measurement. |
| **network_contention_v1** | `src/placement/scheduling_cost.py` (`ingress_transfer_time`, `ingress_wait`), `scripts_cosim/test_network_contention.py`, `datalab/netc_v1_cosim.sbatch`, grids `netc_{scarce,funnel,hotspot}_v1` | `netc_pilot_*` (local, n=12-16) | Shared per-node ingress bandwidth, opt-in via `--ingress-bandwidth-mbps`. Physics works; the corpus lever is replica concentration, not bandwidth alone. **Outcomes below.** |
| **link_contention_v1** | `src/placement/network_fabric.py`, `src/generate_infrastructure.py` (`build_core_backbone`), `scripts_cosim/{link_overlap_precheck,test_link_contention,test_link_repair_control}.py`, grid `netc_multihop_v1` | `netc_multihop_v1_mh_{off,bw1p5}` (local, n=48 each) | Per-link capacity over a multi-hop core backbone, opt-in via `--link-bandwidth-mbps`. **`FALSIFIED` 2026-08-18 — gate FAILED on both criteria.** The link controls do *not* repair (median 0.000), which is a genuinely new signature, but node-collision coupling still dominates (node repair median 1.000). **Outcomes below.** |
| **topology_transfer_v1** | `src/placement/topology_features.py`, `src/placement/network_graph.py`, `scripts_cosim/test_topology_features.py`, `scripts_cosim/test_network_graph.py`, grid `topo_transfer_v1` | `gnn_datasets_4tasks_topo_transfer_v1` (3,744 datasets), graph cache `graphs_cache_topo_transfer_v1` | **Changes the win condition from per-plan accuracy to inductive generalization across topology sizes.** Phases 0-4 all landed. **`FAILED` 2026-08-20 — Phase 4 gate: `gnn_base` loses to `pointwise` on paired `win_rate` in 5/5 seeds** (CI excludes 0.5 below every time, effects 0.022-0.088); `gnn_node` never PASSES either (2/5 FAIL, 3/5 inconclusive-but-trending-to-null, none positive). **Same-day follow-up (2026-08-20, second pass): added the `gnn_topo` arm (`use_network_entities=True` — the only arm with backbone/link topology in the graph at all; `gnn_base`/`gnn_node` never had it) and re-ran the full 5-seed gate. `gnn_topo` also `FAILED`** (pooled win_rate 0.449, CI [0.417, 0.481], resolved not underpowered) — the FAIL is not an artifact of testing topology-blind models. **⚠ SCOPE CAVEAT, unresolved: every arm in this lineage (`pointwise`/`gnn_base`/`gnn_node`/`gnn_topo`, all 20 seed-runs) was only ever evaluated on brute-force-labeled 4-task synthetic co-sim snapshots (`rps=2, duration=1`, fixed regardless of cluster size) — none has ever been live-gated against a real trace (e.g. `data/nofs-ids/traces/workload-200-200.json`, 800k events). No lineage in this repo has ever live-gated across mismatched train/eval topology sizes; this is unexplored, not just untried here. ⚠ Trained model weights were never persisted to disk** (`AblationModel` in `gnn_necessity_ablation.py` had no `torch.save`/checkpoint call anywhere) **— every number in this lineage comes from in-process eval that discarded the model after each training run.** **UNBLOCKED 2026-08-21:** `--save-checkpoints DIR` now persists each arm's weights plus a `.contract.json` (split, held-out sizes, feature contracts, verified `serving_port`), and the serving port itself was measured — it is a **three-module rename** into `TaskPlacementGNN`, not the multi-session build it was costed as, but it requires `mp_residual=True` and getting that wrong is **silent**. The remaining cost of the partial gate is the ~14 GPU-hours, nothing else. See the 2026-08-21 subsection below and the "Co-sim-only scope and live-gate traceability" one. |
| **cache_live_divergence_audit** | `scripts_cosim/audit_cache_live_divergence.py`, `scripts_cosim/verify_cache_live_feature_parity.py` | all 18 collections with `optimal_result.json` | Where do the cache and live feature builders actually disagree? **Platform reordering: 18/18 collections, BENIGN** — the model has no per-position parameter; logits agree to 3e-8 under the identity permutation, so no recache and no asterisk on any result. **Dims 9-11 temporal estimate: 8/18 collections, REAL** (incl. `shallow_v1`; live-gate corpora clean). Parity verifier now compares by platform identity. **Outcomes below.** |
| **contention_v4_v5** | `scripts_cosim/datalab/contention_v4_deepq_cosim.sbatch`, `contention_v5_quick_test.sbatch` | `contention_v4_pilot`, `contention_v5_quick_test` | Deep queues + coupling optimisation — the attempt at giving the GNN real graph structure to exploit. **`FALSIFIED` 2026-08-17: moved the corpus the wrong way** (additive R² 0.988 → 0.9997). See `graph_structure_physics`. |
| **contention_v2_v3** | `important/run_contention_v{2,3}_train_and_live_gate_nohup.sh`, `important/compare_contention_v2_live_gate.py` | `contention_v2{,_verify}`, `contention_v3` | Baseline contention series the v4/v5 work is measured against. Trainers: `train_near_rtt_v2_contention_v{2,3}_dim14_ce_only.py`, `train_mlp_contention_v{2,3}_dim22_batchcache.py`. |
| **sealed_holdout** | `important/run_contention_v2_873_sealed_holdout{,_rebaseline}.sh`, `compare_sealed_live_holdout.py`, `datalab/sealed_holdout_gpu.sbatch` | `contention_v2` | The honest generalisation gate. |
| **coupled_trio** | `important/run_contention_v2_873_coupled_trio.sh`, `chain_coupled_trio_then_rebaseline.sh` | `contention_v2` | See memory note: ECT is not a ceiling. |
| **encoder_ablation** | `important/run_gnn_encoder_ablation.sh`, `compare_encoder_ablation.py` | contention series | Is the graph encoder doing work, or is it the features? |
| **seed_variance** | `scripts_cosim/run_gnn_seed_variance_siv1.sh` | contention_v2 | Uses `train_near_rtt_v2_contention_v2_dim14_ce_only.py`. |
| **queue_feature_contract** | `src/placement/queue_features.py`, `scripts_cosim/test_queue_features.py`, `verify_cache_live_feature_parity.py` | all | `legacy_v0` vs `scale_invariant_v1`. See CLAUDE.md. |
| **dataset_metadata** | `scripts_cosim/{extract_dataset_metadata,validate_dataset_collection,compute_compatibility_matrix}.py` | all | Produces `REGISTRY.json`, `METADATA.json`, `COMPATIBILITY_MATRIX.json`. |
| **cosim_deepdive_v1** | `scripts_cosim/{audit_sweep_truncation,audit_regen_reproducibility,snapshot_separability_sweep,analyze_snapshot_separability}.py`, `datalab/live_audit_capture_{all_gates,mlp_collapse}.sbatch`, `datalab/snapshot_separability_sweep{,_mlpcollapse}.sbatch` | `snapshot_sweeps{,_mlpcollapse}` (44 cells × 100 pseudo-datasets) | **Closed 2026-08-23.** Does the co-sim target's additivity come from the synthetic t=0 snapshot regime? **No — live-visited states are equally additive** (4,400 swept live states incl. all 14 MLP collapse trajectories: median additive R² 0.99999, median additive-choice regret 0.000; jobs 710774/710775/710818/710819). The GNN's dispersal edge is a closed-loop property no single-batch regret target can express under current physics. Plus a pipeline-integrity census (sweep truncation, label provenance, contract audit of the collapse events). **Outcomes below.** |

| **program_verdict_v1** | this file (2026-08-24 subsection); artifacts cited in place | P7 frozen reports (scratchpad) | **Closed 2026-08-24.** Terminal answer to the D3 fork: the supervised co-sim path to "GNN > MLP on latency" is closed by measurement (5 mechanisms + live-state additivity + the P7 warmth-stratum controls, which take the least-additive 31% of the cache to spread-plans R² = 1.00000 exactly). The reliability/regime win over both baselines exists on the 30-cell backbone record but is exploratory — it needs one pre-registered gate. P2 ruled out (labeller is the one-step oracle); P4 ruled out on the empirical rule (the built slot holds exec only; the residency-hold variant is unbuilt but node-indexed); P3 (in-horizon dynamics, tail-sensitive pre-registration) is the highest-upside open measurement; P1 (closed-loop objective) the only path to the latency claim. **Outcomes below.** |

| **p5b_candidate_relative** | `src/policy/tabular/reduced_features.py` (`candidate_relative_queue_columns`), `train_mlp_dim22_from_batch.py --candidate-relative-queue`, `datalab/{fc_siv1_mlp_candrel,mlp_candrel_arm_all_gates}.sbatch`, `important/score_p5b_collapse_pairs.py`, `scripts_cosim/test_{candidate_relative_features,mlp_serving_layout}.py` | no new datasets — derived in-process from `graphs_cache_full_corpus_siv1_dim14{,_tempfix}` | **Closed 2026-08-24 as INDETERMINATE — and the indeterminacy is the result.** Step 1 of `program_verdict_v1`'s sequence, pre-registered before submission (commit `2c5e676`), run clean (jobs `711675`/`711679`, 60/60 COMPLETED). Handing the pointwise MLP the candidate-relative view (`dim25cr`) moved the two cache arms in **opposite** directions: `mlpcandrel` 7/30 → **17/30** collapses, `mlpcandreltf` 7/30 → **2/30** (and negative mean margin vs Knative in 4 of 6 conditions — the first MLP arm to approach the GNN's record). Robust to dropping the registered detector for an RTT criterion. **Kills the mechanism sentence "a pointwise scorer collapses because it cannot condition on its peers"** — one arm has exactly that conditioning, uses it (28.8% ablation), and stops collapsing. Cache and seed are perfectly confounded (both `--random-state 42`). **RESOLVED 2026-08-24 by `p5b_draw_study` (below): it was neither the feature nor the cache — it was the training draw.** Outcome below. |

| **p5b_draw_study** | `scripts_cosim/datalab/p5b_draw_study_{train,gate}.sbatch`, `important/score_p5b_draw_study.py`; `torch.manual_seed` fix in `train_mlp_dim22_from_batch.py` | none — 16 checkpoints over the two existing caches | **Closed 2026-08-24. Q1 = LOTTERY, Q2 = DRAW-DOMINATED, stable at +30/+50/+100%.** Found first that the MLP trainer **never seeded torch** — `--random-state` pinned the split, not the weights — so every MLP checkpoint here before today is an unreproducible draw (the GNN trainer always seeded; the asymmetry went unnoticed). Then measured the full `{dim14,tempfix} × {dim22,dim25cr} × seeds{1..4}` grid, 480 gate runs: collapse counts swing **0→10, 0→11, 0→21, 0→26** on the seed alone, and the candrel effect flips sign *within* both caches. **Retires "the MLP collapses 7/30" (that config gives 0, 0, 21, 16), the "same count, different set ⇒ architectural" inference, and P5b's split — all noise.** The GNN's 0-collapse record survives on these cells (0/30 both arms, −18.9%/−27.1%) but at 2–3 draws vs a measured MLP draw distribution it is p ≈ 0.125 — **unfalsified, not established.** Any future reliability gate must compare draw *distributions*. Outcome below. |

| **m3_batch_makespan_v1** | `src/executecosimulation.py` (`HEROSIM_RETAIN_TASK_TIMES=1` retains `task_times` per placements.jsonl row), `scripts_cosim/score_makespan_vs_sum.py` | `gnn_datasets_4tasks_m3_makespan_pilot` (local, n=200, shallow_v1 grid) | **Closed 2026-08-24 — below both registered thresholds; the escape is real but thin.** The cheapest possible M3 test: the 4 co-sim tasks are already a fan-out of width 4, so re-score existing-physics sweeps under the batch makespan `max(done) − min(dispatched)` instead of the sum. Rule registered before the run (session record + `simulator round 3` doc): fires iff sum-vs-makespan argmin disagreement ≥ 10% of datasets full-sweep or ≥ 5% spread-plans-only (conservative ties: a dataset disagrees only if NO sum-optimal plan is makespan-optimal). Measured at n=200: **full-sweep 3.5% (7/200), spread-only 4.0% (8/200)** — neither fires. But of the 9 datasets involved, **6 disagree identically with collisions removed** (spread regret up to 17.25%, mean-when-firing ~5%) — the first mechanism whose escape is not collision-derived and not count-shaped, with the link-contention profile: genuine non-pointwise structure, thin base rate. Mechanistic read: at width 4 with queue-dominated branch times, sum and max share the bottleneck-avoiding argmin; the max−sum gap grows with fan-out width and branch-time variance (~σ√(2 ln w)), so wider synthesized fan-out is the lever this pilot did not test. Makespan optima tie 2–34 deep where the sum argmin is unique — any makespan label is multi-modal. Frozen reports: `simulation_data/m3_makespan_vs_sum_pilot_n{48,200}.json`. Retention flag is opt-in and default-off; rows without it are refused by the scorer (fail loud), and the scorer cross-checks `rtt == Σ task elapsed` per row before scoring. **CLOSED PERMANENTLY same day by the registered argmax-flip diagnostic** (`scripts_cosim/score_argmax_flip.py`, reopen iff mean flip ≥ 0.25 AND flip correlates with disagreement): mean flip rate **0.497** — threshold met, against the registered "flat" expectation — but corr(flip, disagree) = **0.027**, corr(flip, regret) = 0.050, disagreeing datasets flip at 0.517 vs 0.496 for agreeing; the conjunction fails. The mechanism, now understood: branches are *not* near-tied (mean max-vs-2nd-max gap 0.446), the argmax genuinely moves with the plan — and it doesn't matter, because **when per-branch costs are separable and component choices are free, min-max and min-sum share the componentwise-minimizing argmin for ANY monotone composition** (min max_b f_b = max_b min f_b). Max composition amplifies existing coupling; it cannot create structure from separable physics — so re-scoring under makespan was never an independent escape, at any width. The residual disagreement is exactly the coupled residue: 3.5% full-sweep ≈ the collision channel, and the 4.0% spread-only cases live on the distinct-node *matching constraint* (the one place component choices stop being free). Frozen: `simulation_data/m3_argmax_flip_n200.json`. |

Shared core (not a lineage — everything depends on it): `src/placement/`, `src/policy/{gnn,tabular,knative*,determined,evaluator}/`, `src/executecosimulation.py`, `src/executesimulation.py`, `scripts_cosim/generate_gnn_datasets_fast.py`, `src/notebooks/non_unique_lib/`.

### graph_structure_physics — outcomes (2026-08-17)

**The co-sim target is pointwise-separable, so a pointwise MLP is the correctly specified
model class and the GNN cannot beat it by training.** This reframes `mp_parity`: the GNN
did not lose to the MLP because of a bug, it lost because there is nothing non-pointwise
left to learn.

Measured by the new **M4** block in `separability_diagnostic.py`, which fits
`rtt(plan) ≈ μ + Σ_t f_t(plan[t])` — exactly what `PointwiseEdgeMLP` can express — by least
squares over every plan in `placements.jsonl`:

| collection | additive R² | additive-fit argmin regret | datasets where additive = optimum |
|---|---|---|---|
| contention_v2 (80 ds) | 0.98812 | 0.29% mean | 81% |
| contention_v4_pilot (27 ds) | 0.99973 | 0.00% | **100%** |
| contention_v5_quick_test (35 ds) | 0.99973 | 0.02% | 97% |
| highq_safe_20260606 (25 ds) | 1.00000 | 0.00% | **100%** |

**`FALSIFIED` — deep queues as a coupling lever, confirmed on the full 899-dataset
contention_v2 corpus.** Queue depth *predicts* separability, monotonically:

| quartile | mean queue depth | additive R² | collision R² gain |
|---|---|---|---|
| shallowest 25% | 27.6 | **0.97822** | **+1.986 pp** |
| middle 50% | 35.5 | 0.99211 | +0.672 pp |
| deepest 25% | 50.8 | 0.99803 | +0.181 pp |

`corr(depth, additive_r2) = +0.256`, `corr(depth, collision_gain) = −0.259` (n=899). The
collision term is **11× weaker** in the deepest quartile than the shallowest.

The mechanism is arithmetic: queue work is `depth × exec_time` and grows with depth, while
the interaction term `added_in_batch × exec_time` does **not** — so deeper queues dilute
the only coupling the corpus has. contention_v4/v5 deepened queues further and landed at
R² 0.9997. **The series' core lever is backwards.** Do not extend it.

**The lever that follows: shallow queues plus long-exec task types** (cnn is 3.09 s on
rpiCpu vs dnn2's 0.024 s), making `added_in_batch × exec_time` dominate rather than
vanish. A grid-preset change, not a physics change. Note the corpus only spans depth
26–56.5, so depths below ~26 are an extrapolation — but the trend is monotone across all
899 datasets and the mechanism explains it.

Two grids added: **`shallow_v1`** (shallow queues, stock dnn1/dnn2 — isolates the measured
lever) and **`shallow_longexec_v1`** (adds the `("cnn", "rf")` task pair via the new
`task_type_pair` preset key).

### `shallow_v1` — lowers the pointwise ceiling, but MISSES the coupling gate (2026-08-17)

> **Both tables in this subsection are superseded.** They were measured on 12- and
> 168-dataset snapshots taken while the corpus was still generating. The shipped 200-dataset
> numbers, and what changed, are in the retraction further down — read that before citing
> anything here. Kept for the record of how the estimate moved with n.

12-dataset pilot (`pilot_shallow_20260817`), queue depths ~0-8 instead of ~26-56, same
topology/replica axes as `contention_v2`, stock dnn1/dnn2:

| metric | contention_v2 baseline | **shallow_v1** |
|---|---|---|
| additive R² | 0.98584 | **0.93838** (gate: PASS at <0.95) |
| collision R² gain — platform | +1.261 pp | **+3.143 pp** |
| collision R² gain — node | +1.256 pp | **+4.247 pp** |
| additive-argmin regret mean / max | 0.30% / 3.6% | **2.26% / 16.6%** |
| additive argmin = optimum | 92% | **67%** |
| M1 greedy regret mean / p99 | 0.11% / 4.31% | **1.92% / 16.64%** |
| coupled datasets (>1%) | 3.8% | **16.7%** |

Every metric moves the right way, most by 2.5-7x. Shallowing the queues did in one grid
change what deepening them was supposed to do.

**Corrected at n=168 (full pilot corpus).** The n=12 R² was optimistic:

| metric | contention_v2 | shallow_v1 @ n=12 | **shallow_v1 @ n=168** |
|---|---|---|---|
| additive R² mean | 0.98584 | 0.93838 | **0.96074** |
| additive R² median | — | 0.99839 | **0.99990** |
| coupled >1% | 7.1% | 16.7% | **31.0%** |
| coupled >5% | 2.4% | — | **19.6%** |
| additive argmin optimal | 92% | 67% | **62%** |
| additive regret max | 3.6% | 16.6% | **48.4%** |

At full size mean R² regresses to 0.961 and would **fail** the 0.95 gate. (Both the 0.96074
and the 31.0% here are **retracted** — see below; on the finished 200-dataset corpus they
are 0.95556 and 4.5%.)

### shallow_v1 ablation — reproduced with the label-provenance audit (2026-08-17, superseded same day)

200-ds corpus, cache `graphs_cache_shallow_v1` (v5.7, `scale_invariant_v1`), 120 epochs,
seed 42, test n=30. Report: `simulation_data/gnn_necessity_ablation_shallow_v1_20260817.json`
(`schema_version: 3`, `label_audit`: 200/200 labels are the sweep minimum — the retraction
below is unfounded, see there for why).

The table originally posted here is **superseded by a second run** using the identical
command (same cache, seed, split) but with the new `audit_label_provenance()` preflight
compiled in. Same test split, same greedy baseline (0.00%/0.00%/0.00%, n=30, 0 coupled at
>1%) — but the trained-model numbers moved, because GIN training on CUDA is not
bit-reproducible under a fixed seed (scatter/gather ops are non-deterministic by default):

| model | top-1 | regret mean | regret p90 | regret max | opt-recovery |
|---|---|---|---|---|---|
| pointwise | 90.0% | 2.68% | 6.24% | 53.43% | 76.7% |
| gnn_base | 90.8% | 1.09% | 5.79% | 9.96% | 76.7% |
| **gnn_node** | **90.8%** | **0.88%** | **1.44%** | **9.96%** | **80.0%** |

`pointwise` is reproduced exactly (CPU-only path in this model, or coincidentally stable);
`gnn_base` and `gnn_node` are not. **`gnn_node` beat `gnn_base` on this run** — the reverse
of the original table below, and the reverse of the FALSIFIED verdict this section
previously recorded.

**Verdict downgraded to inconclusive, not confirmed either way.** Both `gnn_base` and
`gnn_node` beat `pointwise` by 2-3x on regret_mean across both runs — that part replicates.
Which of the two GNN variants is better did not. Before trusting an ordering between
`gnn_base` and `gnn_node`, either (a) set `torch.use_deterministic_algorithms(True)` and
confirm a third run matches one of these two, or (b) run several seeds and compare
distributions, not point estimates. Do not cite a `gnn_base` vs `gnn_node` winner from a
single run again.

<details>
<summary>Original table (2026-08-17, first run, no label audit) — kept for the
non-reproducibility record above, not as a result</summary>

| model | top-1 | regret mean | regret p90 | regret max | opt-recovery |
|---|---|---|---|---|---|
| pointwise | 90.0% | 2.68% | 6.24% | 53.43% | 76.7% |
| gnn_base | 91.7% | 0.90% | 2.35% | 9.96% | 80.0% |
| gnn_node | 89.2% | 2.83% | 6.76% | 53.43% | 73.3% |

This run reported `gnn_node` losing even to `pointwise` and carried a "stays FALSIFIED, do
not retry" verdict. That verdict does not survive the rerun above — retracted along with the
label-contamination claim it was bundled with.
</details>

**Retraction withdrawn 2026-08-17 — the table above stands.** This section previously
carried a "DO NOT CITE — labels are contaminated" warning claiming 24/200 cache labels were
not the sweep optimum. That was wrong on both of its legs, and the correction matters more
than the original claim:

1. **The cache labels are clean.** `validate_training_cache_contract.py --cache-dir
   simulation_data/graphs_cache_shallow_v1` audits **200/200 `label_matches_sweep_min`, 0
   failures.** `prepare_graphs_cache.py` has labelled from `load_sweep_minimum(jsonl)` since
   commit `2a591ed` (2026-08-13); the cache records `label_source =
   placements.jsonl_sweep_minimum`. The §1 blocker was already fixed — the retraction
   reasoned from the doc's open item instead of from the code.
2. **The 29/200 discrepancy is real but harmless.** `optimal_result.json`'s
   `sample.placement_plan` differs from the sweep minimum in 29/200 datasets (24 with
   nonzero regret; mean 13.06%, max 92.55%) — that is where the "24/200" came from. But
   `build_graph` reads only three columns off that file's task table: `task_type`,
   `source_node`, and `optimal_platform_id` (the caller-supplied *sweep-min* label). The
   realized-outcome fields (`elapsed_time`, `execution_node`, `execution_platform`, …) go
   to `task_metrics_analysis.csv` only and never reach a feature tensor. Verified by
   enumerating every `df_tasks[...]` access in `build_graph`.
3. **The 0/30 coupled test split needed no special explanation.** The true M1 coupled(>1%)
   of this corpus is **4.5%** (~9/200), not the 31.0% asserted above — see the retraction
   directly below. Drawing 0 coupled datasets in a 30-dataset holdout is then a p≈0.24
   event, i.e. unremarkable. The contradiction that motivated the whole retraction was an
   artifact of the wrong baseline number.

The lesson survives even though the finding didn't: the ablation *had* no check that the
labels it scores against are the sweep optima it claims. That gate now exists (below), and
it passes on this corpus.

### ⚠ RETRACTED — `shallow_v1` coupled(>1%) = 31.0% does not reproduce (2026-08-17)

Re-running `separability_diagnostic.py` on the shipped 200-dataset corpus
(`simulation_data/gnn_datasets_4tasks_shallow_v1`, queue depth mean 2.04) gives numbers
well below the n=168 row in the table above:

| metric | contention_v2 (899) | shallow_v1 claimed @ n=168 | **shallow_v1 measured @ n=200** |
|---|---|---|---|
| additive R² mean | 0.98812 | 0.96074 | **0.95556** |
| additive R² median | — | 0.99990 | **0.99527** |
| **coupled >1%** | **7.1%** | **31.0%** | **4.5%** |
| coupled >5% | 2.4% | 19.6% | **4.5%** |
| M1 greedy == optimum | — | — | **92.5%** |
| additive argmin optimal | 92% | 62% | **62%** |
| additive regret max | 3.6% | 48.4% | **133.0%** |

**The M4 axis held; the M1 axis did not.** Additive R² and additive-argmin regret both
still beat contention_v2 — the pointwise ceiling really is lower on shallow queues. But
M1 coupling, *the statistic this lineage picked as its gate*, came in at **4.5%, below
contention_v2's 7.1%**. shallow_v1 does not clear the headroom bar it was declared to
clear.

The 31.0% almost certainly measured an in-flight corpus: dataset generation
(`logs/progress_4tasks.txt`) was still writing until 13:52, and a truncated
`placements.jsonl` yields a wrong "optimum" and inflates apparent coupling. Nothing
recorded which corpus snapshot that run read, which is the reason it can only be
"almost certainly".

**Consequences.** The ablation table above is measured on a corpus that is *barely*
coupled, so `gnn_base`'s win is real but rests on a handful of datasets — treat it as a
promising signal, not a result to build on. Two things to do before extending this
lineage: (a) re-run the diagnostic only on corpora that are finished, and (b) get the
coupled fraction genuinely up, since that is still the thing the GNN needs.

### Label-provenance gate — the third gate (2026-08-17)

`gnn_necessity_ablation.py` now runs `audit_label_provenance()` as a preflight before a
single epoch: for every graph it decodes `y` through `task_logit_to_placement` into a joint
plan, streams that dataset's `placements.jsonl`, and fails loud unless the label's RTT is
the sweep minimum. It reports suboptimal labels, labels absent from the sweep, and
undecodable labels separately, and writes a `label_audit` block into the frozen report
(`schema_version: 3`). `--skip-label-audit` exists and marks the run unreportable.

This closes the structural gap the retraction exposed: the separability gate measures
coupling in `placements.jsonl`, the ablation measures models against cache labels, and
until now nothing checked the two described the same optimum. Tests:
`scripts_cosim/test_label_provenance.py` (5 cases, including drift, absent-label, and
missing-sweep).

### `shallow_longexec_v1` — unblocked (2026-08-17)

The grid failed every dataset instantly with `No workload parameter in the sample mapping
for: ['nofs-cnn', 'nofs-rf']`. Two independent causes, both now fixed:

1. **No sampled workload factor for substituted task types.** `prepare_workloads` needs a
   `workload_<app>` entry for every app in `wsc`, and the shared sampled space
   (`sample_simple.json`, `lhs_samples_simple*.{npy,pkl}`) ships only
   `workload_nofs-dnn{1,2}`. New helper `ensure_workload_params()` in `src/sample_loader.py`
   grows *this run's copy* of the sample instead — substituted types inherit the factor of
   the type whose position they take, so ("cnn", "rf") gets dnn1's and dnn2's. The shared
   files are untouched, so every other grid reads them at unchanged indices.
2. **`create_config_for_iteration` hardcoded `dnn1`/`dnn2`.** It rewrites `config['replicas']`
   and `config['prewarm']` wholesale, silently discarding the entries `main()` synthesizes
   for a new pair. The infrastructure then carried dnn1/dnn2 replicas while the workload
   asked for cnn/rf, and system-state capture failed. Both dicts are now keyed off
   `task_type_pair`. `generate_infrastructure.py` skips replica configs whose task type is
   absent from `task-types.json` **silently** (`continue`), which is why this surfaced as a
   capture failure rather than a loud error — worth fixing separately.

Smoke test: `--grid shallow_longexec_v1 --max-datasets 1 --allow-non-unique-replicas` now
generates cnn: 22 / rf: 22 replicas and a complete dataset with `placements/placements.jsonl`
(90 rows). Also fixed in passing: the workload-template logger referenced `num_dnn1`/
`num_dnn2`, which no longer existed — a `NameError` on any non-quiet run.

**Pipeline trap — the generator does not emit `system_state_captured_unique.json`.**
`prepare_graphs_cache.py` requires it and fails with
`FileNotFoundError: Missing system_state_captured_unique.json for ds_00000`. It needs a
separate pass first:

```bash
pipenv run python3 scripts_cosim/refresh_optimal_full_stats.py \
  --base-dir simulation_data/<collection> --rewrite-ssc
```

This is the same trap recorded for contention_v2 in June (`gnn_necessity_separability.md`
§3, "SSC via refresh_optimal_full_stats.py --rewrite-ssc; recache pending for +189 ds").
Add the SSC pass to any new-corpus runbook — it is not optional and it is not automatic.

**Methodological consequence: mean additive R² is the wrong gate statistic here.** Median
0.99990 against mean 0.96074 means the target is **bimodal** — most datasets stay perfectly
separable and all the structure sits in a large minority. Averaging hides exactly the thing
worth measuring. **Use the coupled fraction (M1 regret >1%) as the gate.**
`--gate-coupled-fraction` now exists alongside `--gate-additive-r2`.

(The "4.4x contention_v2" claim that stood here is **retracted** — it rested on the 31.0%
figure. On the finished corpus the coupled fraction is 4.5%, *below* contention_v2's 7.1%.
The bimodality argument for preferring the coupled fraction over mean R² stands on its own
and is unaffected.)

**The node-collision column now explains MORE than the platform-collision column**
(+4.247 vs +3.143 pp). In every prior corpus the two were indistinguishable
(1.256 vs 1.261 on the baseline) because nothing distinguished co-location on a node from
co-location on a platform. This is the first corpus with a *node-level* signal beyond the
platform-level one — i.e. the first structure a graph encoder could hold that a pointwise
scorer cannot. (Measured on the finished 200-dataset corpus the gap narrows but holds:
platform +2.185 pp, node +2.384 pp.) The ablation this called for has since run twice; the
two runs disagree on whether `gnn_node` or `gnn_base` wins (see "reproduced with the
label-provenance audit" above) — not reproducible enough yet to call either direction.

**`shallow_longexec_v1` was BLOCKED — now unblocked, see the section above.** New task types
need a `workload_nofs-<type>` parameter in the sampled space, not just
`wsc`/`prewarm`/`replicas` entries; `ensure_workload_params()` now supplies it in memory.
Two loud-failure fixes went in while finding this:
`prepare_workloads` silently dropped apps missing from the sample mapping (now raises with
the known keys listed), and `calculate_workload_stats` returned `average_rps` while
`flatten_workloads` reads `stats['rps']`, so an empty workload died as a bare
`KeyError: 'rps'`.

**The whole interaction is one integer.** Adding a single collision-count column takes v2
from 0.98812 → 0.99912 (+1.10 pp); on v4/v5 it is worth +0.03 pp. That is a feature you
hand an MLP, not graph structure.

**Why the physics does this** (`src/placement/scheduling_cost.py:108-132`): every term of
`current_work + queue_work + cold_start + exec_time + comm_time + network` is a function of
`(task, platform)` alone except `added_in_batch`. Network latency is a static table lookup
paid once per task (`infrastructure.py:985-993`), never congestible. Co-located platforms
share **nothing** — the only `capacity` in `infrastructure.py` is disk cache;
`memoryRequirements` is never enforced as a contended resource. Measured RTT split on a
real optimum: queue 1.330s (~95%), network 0.0719s (5.1%), exec 0.0239s, comm 0.0170s,
cold-start and pull **exactly 0**.

**New evidence on the `FALSIFIED` same-node edges (mp_parity, Arm B).** That arm was
measured against physics with no node-level coupling at all, so the edges carried a signal
worth ≤1.1 pp of variance. The edges were fine; the physics was missing. Re-run Arm B once
Phase 1 lands — this is the new evidence its "do not revive" note asks for.

**DAG data locality — chains are genuinely blocked, fan-out is not.** `workflow_process`
(`orchestrator.py:717-748`) does `yield task.done` before submitting the next task, and
`scheduler.py:78-82` filters on dependencies finished, so `A→B→C` can never be co-decided.
But `orchestrator.py:739-745` takes `ordered[current_index + 1]` — the flat topological
*linearization* — so `A→{B,C,D}` is silently run as a chain. Dispatching all ready
successors would make siblings co-decidable; the scheduler already admits them. Blocked on
magnitude, not mechanism: local-vs-remote input read is `S·6.29e-9 + 0.01455` s, i.e. 15.5 ms
at today's 153,600 B `stateSize` (1.2% of the queue term) and ~1.0 s only at S ≈ 160 MB.

**Phase 1 in progress — `node_contention_v3`.** A node-level pool of shared execution
slots (`Node.compute_slots`, a `simpy.Resource`) that co-located platforms contend for.
Opt-in via `--compute-slots-per-node` / `config.nodes.compute_slots_per_node`; left unset
the node has no pool at all and physics is bit-identical to `node_disk_v2`, so existing
corpora regenerate unchanged. Guarded by `scripts_cosim/test_node_contention.py` (9 tests).

One non-obvious trap found while building it: **queue depth is seeded as a compressed
warmup backlog** (`Platform.virtual_warmup_total_time`), drained in a single
`env.timeout`, not as `queue.items`. Wrapping only the per-task execution path left
`nodeContentionTime` at exactly 0 — the backlog is ~95% of RTT and was bypassing
contention entirely. Both the drain and the ECT mirror
(`scheduling_cost.node_contention_wait`) now account for it.

**Result — no usable effect. Matched 12/12 pilots on the `contention_v2` grid**, same
seeds, differing only in `--compute-slots-per-node 1` (`pilot_baseline_20260817` vs
`pilot_nodecontention_20260817`):

| arm | additive R² | collision R² gain | additive argmin = optimum | max regret |
|---|---|---|---|---|
| baseline (no slot pool) | 0.98584 | +1.261 pp | 92% | 3.6% |
| shared slots, capacity 1 | 0.98847 | +1.000 pp | **75%** | 2.0% |

**The metrics disagree and n=12 is underpowered**, so this is not a clean result in either
direction: additive R² moved the wrong way by 0.003 (noise-level), while the
additive-argmin optimality rate moved the *right* way (92% → 75%). Do not cite either as
an effect without a larger pilot.

**The mechanistic finding is not statistical and does stand:** `nodeContentionTime` is
**exactly 0.0 on every placed task** in every dataset. The placed tasks never contend with
each other — their exec times are ~0.024 s and the seeded backlogs have fully drained by
the time they run. Slot contention only serialized the backlogs, which is a
per-`(task, node)` quantity and therefore an *additive* term. So whatever moved in the
table above, it was not the intended mechanism.

**The design lesson:** adding a shared resource is not sufficient. **The contended resource
must be one the placed tasks hold long enough to overlap each other.** Slot-held-during-
execution is far too brief against a queue-dominated RTT. Candidates that would satisfy
it: memory held across the whole residency (cold start + exec, where cold starts reach
38 s), or a slot held for a replica's warm lifetime rather than just its execution. A
cheaper untested alternative is a *scenario* change rather than a physics one — shallow
queues plus long-exec task types (cnn at 3.09 s on rpiCpu), which would make the existing
`added_in_batch × exec_time` collision term dominate instead of vanish.

**Data-integrity findings.** `contention_v5_quick_test` has 3/38 datasets with no
`placements/` directory at all (`ds_00015`, and two others), violating the mandatory-JSONL
rule. `highq_safe_20260606` cannot pass the existing M1 check — its sweep is sampled, so the
marginal-greedy combo is often not enumerated. M4 does not need it; **M1's strictness will
need relaxing for sampled sweeps before Phase 4 raises the batch size.**

### topology_transfer_v1 — the change of win condition (2026-08-18)

**Why this lineage exists.** Five mechanisms have now failed to produce coupling a
pointwise model cannot express, and `--spread-plans-only` showed base physics is additive
to R² = 1.00000 exactly once collisions are removed — there is no reservoir of non-count
coupling left to find. Per-plan accuracy on a fixed topology is a dead axis. This lineage
changes what "winning" means to **inductive generalization**: train on small topologies,
evaluate on held-out larger ones. It does not depend on finding coupling at all.

**⚠ The motivating architectural claim is FALSE for this repo, and the framing is corrected
accordingly.** `PointwiseEdgeMLP` (`mlp_model.py:39-48`) maps `[N_edges, 21] -> [N_edges]`
with grouped-argmax decode per task (`mlp_scheduler.py:162-227`). `N` is the *candidate-pair*
count, not a topology-sized vector, so the MLP **runs unmodified on any cluster size**. The
claim "MLPs structurally cannot transfer" would be falsified by anyone reading
`mlp_model.py`. The claim this lineage tests is therefore **empirical**: does GNN regret
degrade more slowly than pointwise regret as topology grows? That must be won by
measurement.

**⚠ The GNN currently sees no network topology.** `build_inference_graph`
(`feature_builder.py:567-631`) builds `n_tasks + n_platforms` nodes with bipartite
task↔candidate-platform edges plus optional same-node platform↔platform edges (default OFF,
FALSIFIED in `mp_parity`). There are **zero node↔node link edges**; network latency enters
only as a static scalar in the 5-dim `edge_attr`. RouteNet generalizes across topologies
because its graph *is* the network. Putting links and routes into the graph is therefore a
**prerequisite** (Phase 2), not an enhancement — without it the study compares two
topology-blind models.

#### Phase 0 — the scale-dependent feature, removed (DONE)

Task feature dim 2 was `src_norm = index_of(source_node) / len(nodes)`: a node's **arbitrary
enumeration index**. Measured on 40 real `shallow_v1` datasets it is literally the ramp
`i/40` — 0, 0.025, 0.05, … — i.e. zero topological content. Its granularity *and*
distribution both change with cluster size (multiples of 0.02 at 40 nodes, 0.0125 at 80),
so any degradation measured across sizes would partly be this artifact. It was also
redundant: the source→candidate `latency` it stood in for is already edge attribute 1,
per-candidate and exact.

The formula lived in **six** independent copies (`feature_builder.py`,
`prepare_graphs_{cache,ram,cache_seq}.py`, `gnn_hetero/scheduler.py`,
`reduced_features.py`) — the same duplication `queue_features.py` was created to end, and
the same shape as the `mp_parity` train/serve split. It now lives once in
`src/placement/topology_features.py` and all six call it.

Two contracts, mirroring `queue_features.py`, so a checkpoint is never served a feature it
was not trained on (`TOPOLOGY_FEATURE_CONTRACT`, default `src_index_v0`):

| contract | dim 2 | measured on real data |
|---|---|---|
| `src_index_v0` (default) | `index(src) / n_nodes` | ramp `i/40`; **bit-exact vs the old formula on 40/40 datasets** |
| `size_invariant_v1` | reachable servers / total servers | 9 distinct values in [0.05, 0.50], bounded, size-invariant |

Guarded by `scripts_cosim/test_topology_features.py` (15 tests), including an explicit
control asserting v0 *does* vary with cluster size — if that ever stops being true, the
reason v1 exists needs rechecking.

#### Phase 1 — the topology-size axis (DONE)

**Every corpus in this repo was generated at exactly one size** (20 clients + 20 servers,
from `space_with_network.json`), so nothing could be held out. Note `cluster_size` in
`sample_simple.json` *looks* like the size knob but is **inert**: its only consumer,
`calculate_device_counts` (`executecosimulation.py:513`), is defined and **never called**.
Node counts come from the config.

`server_node_counts` is now a grid axis, crossed into `grid_topology_variants` so it needs
no separate loop level and lands in the dataset label. Grids omitting the key are
**unchanged** (verified: `shallow_v1` still 900 datasets, no size label, no kwarg). Only the
*server* tier scales — clients stay at 20 so the task-source draw is identical across arms.

**Measured combination-count probe** (1 dataset/size; the plan required this before any
corpus, since generating past the enumeration cap *silently skips* datasets and would bias a
held-out size toward its easier half):

| servers | nodes | sweep plans | gen time |
|---:|---:|---:|---:|
| 10 | 30 | 16 | 0.7s |
| 14 | 34 | 16 | 0.8s |
| 20 | 40 | 32 | 0.8s |
| 28 | 48 | 48 | 0.8s |
| 40 | 60 | 432 | 2.0s |
| 60 | 80 | 2,730 | 9.3s |
| 80 | 100 | 9,828 | 39.0s |

**⚠ SUPERSEDED 2026-08-19 — this table does not reproduce.** Re-measured at `--workers 8` on
a 32-core box, *both* the plan counts and the times differ:

| servers | plans (orig) | plans (re-run) | time (orig) | time (re-run) |
|---:|---:|---:|---:|---:|
| 20 | 32 | **18** | 0.8s | 0.4s |
| 28 | 48 | **44** | 0.8s | 0.5s |
| 40 | 432 | **343** | 2.0s | 3.3s |
| 60 | 2,730 | **2,231** | 9.3s | 23.0s |
| 80 | 9,828 | **8,698** | 39.0s | **117.2s** |

**Suspected cause, not confirmed: the 2026-08-18 workload-seeding fix changed the draw**, so
the two tables enumerate different workloads. Both are kept rather than one overwritten, so a
future re-run can tell which it matches. **Budget from the re-run numbers** — the top of the
ladder is ~3× more expensive than recorded. This also weakens the *low-end* justification the
ladder rests on: 20 servers enumerates **18** plans, not 32, so the coarsest rung is coarser
than the cutoff argument assumed. The ladder is unchanged (18 plans still resolves regret far
better than the 16 at 10–14 servers, and the alternative is dropping to a 3-rung ladder), but
it is a thinner margin than the original table implied.

**The cap is 250,000**, not the "100k" this section and the preset docstring both cited —
`MAX_PLACEMENT_COMBINATIONS_SKIP_DEFAULT` in `generate_gnn_datasets_fast.py`, exported as
`$MAX_PLACEMENT_COMBINATIONS`. The re-run peak (8,698 plans) is **3.5% of it**. Docstring
corrected 2026-08-19.

**The enumeration ceiling is not the binding constraint** — the sweep grows ~quartically but
stays inside the cap to ~100 servers. The *low* end binds instead: a 16-plan sweep at 10-14
servers makes regret far too coarse. Ladder set to train {20, 28, 40} / hold out {60, 80},
1.5-4x the largest training size, every label still a true sweep minimum.

**The candidates/task floor worry was unfounded.** Geometric-mean candidates/task grows
2.38 → 9.96 (4.19×) strictly monotonically across the ladder, replica-host nodes 7 → 29,
because `replica_server_pct = max(server_pct, 0.6)` is a *percentage* and scales with server
count.

#### Phase 2 — the network, in the graph (DONE)

`src/placement/network_graph.py` adds two entity types and four edge families to the
placement graph, behind `NETWORK_GRAPH_CONTRACT` (default `off`, so every existing cache,
checkpoint and corpus is untouched):

```
  [tasks] --candidate--> [platforms] --hosted_on--> [nodes]
                                       [nodes] --routes_via--> [links]
                                       [tasks] --routes_via--> [links]
                                       [links] --adjacent-----> [links]
```

**Only core links become entities.** Per `network_fabric.py`, access links carry one node's
traffic and are perfectly additive; core segments are the only shared, contended objects.
This is not just a fidelity choice — `GIN` aggregates with **sum**, so any degree that grows
with the cluster shifts embedding magnitudes with the cluster, and the transfer measurement
would be reading its own graph construction. That is Phase 0's confound reappearing as
structure. Attaching core *routers* as node entities would do exactly this (every node
attaches to one, so router degree ∝ N). Under `core`, every added degree is bounded by a
config constant — `n_core`, `attach_degree`, platforms-per-node — and none by N. Asserted
directly at 8/16/32/64 servers, alongside a test that no feature *column's* range drifts
with size (a bound catches a runaway; it does not catch a creep).

Scoring did not move: logits stay on task→platform edges, so `edge_attr` alignment and the
`to_undirected` reverse-edge pairing are untouched. The entities append **after** platforms,
leaving every pre-existing checkpoint's index layout intact; `forward()`'s platform slice
became bounded (`x[n_tasks : n_tasks+n_platforms]`) rather than open-ended.

Built once and called by **both** the cache builder and live inference — the `mp_parity`
discipline, applied up front rather than after a 12.4× regression. Verified on
`netc_multihop_v1_core4/ds_00000`: 40 node entities, 4 core-link entities, 458 route edges,
and **`task↔link`, `node↔link` and `link↔link` are byte-identical cache↔live**. Link
features carry real routing structure — two core links carry 75% of tasks and 17%/56% of
candidate routes, two carry nothing.

Serve-side guard closed: the encoders are visible in the weights, but *which contract built
their features* is not, so trainers record `network_graph_contract` in the checkpoint
sidecar and `load_gnn_model` refuses a mismatch. A model built with network entities and
handed a graph without them fails loudly rather than degrading into a silently different
(bipartite) model.

`scripts_cosim/test_network_graph.py` (31 tests); 143 pass across all affected suites;
cache↔live parity passes on dim24/dim22/dim14 with the contract off.

The parity failures this phase surfaced on the netc corpora turned out **not** to be a
Phase 2 problem, and not the problem they first looked like either. They are recorded as
their own lineage — see `cache_live_divergence_audit` below. Phase 2's own additions are
parity-clean.

#### Pre-registered gate — Phases 3-4 (registered BEFORE any corpus is generated)

Registered now because `--gate-coupled-fraction` is the cautionary precedent: a gate nobody
exercised, and a statistic that would have rejected the one configuration that worked.

**⚠ AMENDED 2026-08-19, before any corpus was generated** — see "the gate statistic
decision" below for the measurements. The original v1 text is kept struck through so the
amendment is auditable rather than silent; nothing had been generated, trained or gated
under v1, so no result changes hands.

~~v1: Gate on the **slope of additive-argmin regret against topology size**, per model.
**PASS** — pointwise regret grows monotonically across held-out sizes while GNN regret stays
flat within seed variance, and the gap widens monotonically. **FAIL** — both degrade at
comparable rates, or neither degrades.~~ **Falsified as a gate:** a decision rule of constant
expressive power satisfies that PASS condition on its own, by landscape drift alone.

**v2 — gate on the slope of `win_rate` against topology size, in excess of the drift
anchor:**

- **PASS** — GNN-vs-pointwise `win_rate` stays flat or rises across held-out sizes while the
  no-learning drift anchor over the same datasets does not, the separation exceeds the
  bootstrap CI at every size, and `regret_ratio_mean` agrees in sign.
- **FAIL** — `win_rate` tracks the drift anchor, or the two co-primary statistics disagree in
  sign, or the CI **excludes 0.5 on the pointwise side**. ~~or the CIs overlap 0.5 at the
  held-out sizes~~ — **amended 2026-08-19 (see "the power ladder" below): a straddling CI is
  an under-powered result, not a null one, and calling it FAIL manufactures a false negative
  at exactly the effect size this lineage expects.** Only a CI excluding 0.5 on the
  *reference's* side licenses "does not transfer".
- **VOID / ESCALATE (not FAIL)** — the CI straddles 0.5. Auto-escalates to the next
  pre-registered power tier if one can resolve the observed effect; otherwise reported as
  `INCONCLUSIVE_LADDER_EXHAUSTED` with the n it would need. Neither decides the lineage
  against the GNN.

Controls, none optional (1-3 pre-registered, 4-5 added by the amendment):

1. **≥5 seeds (was ≥3; re-derived 2026-08-19 — see "Seed count" below), distributions not
   point estimates.** ~~GIN training on CUDA is not
   bit-reproducible under a fixed seed~~ — **corrected 2026-08-19: it is not CUDA.** The
   non-determinism is in the GIN autograd path and fires on **CPU** too; `pointwise` is
   bit-identical run to run while `gnn_base`/`gnn_node` diverge in the *training loss*.
   Not intra-op threading (`OMP_NUM_THREADS=1` still diverges), not `PYTHONHASHSEED`.
   `torch.use_deterministic_algorithms(True, warn_only=True)` makes all three bit-identical
   and is now **on by default** in the harness (`--nondeterministic` to opt out).
   **Seeds run before that fix do not measure what they claim**: run-to-run noise was the
   *larger* term — `win_rate` moved 0.517 → 0.550 between two identical seed-44 commands
   against a seed-to-seed spread of only 0.517–0.533.
2. **Knative as a size-invariant reference**, so "the GNN held up" is distinguishable from
   "the task got easier at that size".
3. **`--gate-one-integer-repair` reported at every size.** If one count column repairs the
   pointwise model's failures at large sizes, the gap is the same degeneracy this repo has
   hit five times, not a transfer result.
4. **The no-learning drift anchor at every size** (additive-fit argmin + additive+one-integer,
   same held-out datasets, no training). Distinguishes "the GNN held up" from "the statistic
   drifted". Knative cannot substitute — its own quality may move with size.
5. **Power stated before the verdict.** Report `min_detectable_gap` and the `win_rate` CI at
   every size; a run below the power table is VOID, not FAIL.

**🔄 IN PROGRESS 2026-08-19 — Phase 4 corpus generation running on datalab at `tier_launch`
(900/size, 4,500 datasets total).** `TOPO_TRANSFER_V1_GRID`'s seed range bumped 30→75
(`scripts_cosim/generate_gnn_datasets_fast.py`) to hit 900/server_node_count uniformly across
all five sizes (`grid_total_datasets` confirms 4,500). Smoke-tested locally first (6
datasets, `--output-subdir` redirected away from the real corpus dir): backbone present (52
links, `n_core=12`, 1000 MB/s non-binding) on every dataset, `placements.jsonl` populated. A
local full-corpus run was started, then killed and moved to datalab for wall-clock (local
32-core estimate ≈9-10h; datalab CPU-amd has 16+ idle 128-core nodes). `src/`,
`scripts_cosim/` and this file rsynced to `/home/nikola.lukic/gnn-herosim` (datalab's
`feat/gnn-mp-residual` was clean at the same base commit, `852736c`, as local's uncommitted
diff — a plain overlay, no merge). Re-verified with a second smoke test on datalab itself
before submitting. New tool: `scripts_cosim/datalab/topo_transfer_v1_cosim.sbatch`, reusing
the generic `run_contention_regen_shard.sh` sharder with `GRID=topo_transfer_v1`,
`TOTAL_DATASETS=4500`, `NUM_SHARDS=50` (450 datasets/topology-variant means 50 shards splits
every variant into exactly 5 homogeneous-cost shards — the expensive `srv=80` variant does
not end up as one long serial shard), `ALLOW_NON_UNIQUE=0` (explicitly overriding the
sharder's netc-oriented default of 1 — the local smoke test succeeded at 100% without it).
Submitted as SLURM job **704238** (`sbatch --array=0-49`), 35 array tasks started
immediately across `os-cpu-slurm-{5,7-24}`.

#### Phase 3 landed — `topology_size` split mode (2026-08-19)

`gnn_necessity_ablation.py`'s `--split-mode` only had `canonical_parent` (random) and
`copy_shuffle` — no way to hold out topology sizes, so no run through this harness had ever
actually tested transfer. Added `split_ids_by_topology_size()` + `topology_sizes_by_parent()`
to `src/notebooks/non_unique_lib/training_contract.py`: server count isn't stored as a graph
attribute anywhere (`generate_infrastructure.py` never wrote it), so it's read back from each
dataset's `infrastructure.json` `network_maps`, counting non-client nodes
(`CLIENT_NODE_PREFIX`, single source of truth in `topology_features.py`). Wired in as
`--split-mode topology_size --train-sizes 20 28 40 --held-out-sizes 60 80`; `val` is drawn
only from the train-size pool, never held-out sizes, so model selection can't peek at the
transfer question. 11 new tests (`test_topology_size_split.py`).

**Also fixed while running it for real: the missing-plan hard-fail was checking the wrong
thing.** `eval_regret` raised `RuntimeError` on ANY predicted plan absent from the retained
placement sweep, on the assumption that only a corpus/harness bug could cause it. First real
run at held-out sizes 60/80 crashed at 120 epochs (not undertraining) with 265/619 `pointwise`
predictions missing. Diagnosis: **all 265 were collisions** (two tasks independently picked
the same node+platform) — a brute-force sweep correctly never enumerates a jointly-infeasible
combination, and `pointwise` has no mechanism to avoid one by construction (that's the whole
point of the ablation). `eval_regret` now splits `n_missing_plan` into `n_missing_collided`
(reported, not raised on — a first-class coordination-failure statistic) and `n_missing_clean`
(no collision, still missing — genuinely inexplicable, still raises). Schema 6 → 7.

#### Phase 4 — first real result, 5 seeds, `tier_launch` (2026-08-19/20)

Cache built via `NETWORK_GRAPH_CONTRACT=core_v1 TOPOLOGY_FEATURE_CONTRACT=size_invariant_v1`
on a **partial** corpus (3,194/4,500 datasets — generation was still running; caching raced it
and lost twice before landing, see GATE TOOLS and the datalab agent note below). Verified:
network entities non-zero (`net_node_features` [40,6], `net_edge_index` [2,536] on a sampled
graph), task feature dim 2 in the size-invariant `[0,1]` range.

**Seed 42 alone: both `gnn_base` and `gnn_node` PASS** (win_rate 0.545 / 0.531, CIs excluding
0.5 above, resolved not escalated) — the first PASS this lineage has ever produced.

**Pooled across all 5 seeds (`pooled_phase4_verdict`), the effect is not there:**

| model | pooled win_rate | 95% CI | verdict |
|---|---:|---|---|
| gnn_base | 0.4976 | [0.467, 0.529] | `INCONCLUSIVE_LADDER_EXHAUSTED` — effect ≈0.002, needs ~59,200/size |
| gnn_node | 0.4962 | [0.473, 0.520] | `INCONCLUSIVE_LADDER_EXHAUSTED` — effect ≈0.004, needs ~13,500/size |

Per-seed `win_rate` for `gnn_base`: 0.545 (42) / 0.496 (43) / 0.454 (44) / 0.476 (45) / 0.518
(46) — a spread that alone would read PASS, INCONCLUSIVE, FAIL, ESCALATE, and ESCALATE
respectively if read one seed at a time. This is precisely the noise the seed-calibration and
pooling work earlier in this session exists to catch, now catching it on a live result instead
of a retrospective one. `regret_ratio_median` sits at ≈0.998–0.999 across every seed for both
models — the co-primaries **agree** with each other (both read "no detectable difference"),
so this is a clean null, not a sign-disagreement artifact.

**Honest reading: on this partial `topo_transfer_v1` corpus, `gnn_base`/`gnn_node` and
`pointwise` pick the better plan at statistically indistinguishable rates on held-out topology
sizes.** `INCONCLUSIVE_LADDER_EXHAUSTED`, not `FAIL` — this does **not** say the GNN fails to
transfer, it says the effect (if any) is too small to resolve at this corpus size, and closing
that gap (~13,500–59,000 datasets/held-out size) is 15–65× `tier_launch`'s cost, not a
next-registered-tier decision. Two caveats on this specific result, not yet resolved:
generation was only 71% complete when this cache was built (3,194/4,500 — the full corpus may
tell a different story), and this is the ablation harness's own small internal GIN, not the
production `train_near_rtt.py` model. Frozen reports:
`simulation_data/gnn_necessity_topo_transfer_v1_seed{42..46}_20260819.json`.

**⚠ SUPERSEDED for `gnn_base` — see "Corpus generation, cache build, and the Phase 4 gate" below.**
A concurrent session ran the same pooling against the *complete* corpus (3,744/4,500) and got an
unambiguous answer: `gnn_base` pooled win_rate 0.456, CI [0.440,0.471], **FAIL** — the missing
29% of the corpus was enough to resolve what this partial run could only report as
under-powered. `gnn_node` is still not resolved (`ESCALATE`, needs n≥690). Left standing as a
historical record of a real methodological point: seed 42 alone read as a clean PASS on this
same partial corpus, and both the partial-corpus pooled result and the full-corpus one agree
that single-seed result did not hold up — first on pooling (this section), then on completing
the corpus (the section below).

#### Corpus decisions — settled 2026-08-19, before generation

Three probe findings needed a call before any `topo_transfer_v1` corpus could be generated.
All three are now fixed in the preset and in `gate_statistics.py`, not just written here.

**1. Backbone ON, at a deliberately non-binding 1000 MB/s.** The probe's blocking finding:
`--grid topo_transfer_v1` with default flags produced **`link_topology: null`**, because the
backbone block was written only when `--link-bandwidth-mbps` was passed, and
`build_network_graph_block` treats a missing fabric as *"a legitimate, silent no-op"*.
Training that corpus under `NETWORK_GRAPH_CONTRACT=core_v1` would have produced two
topology-blind models without a word of warning — the exact failure Phase 2 exists to
prevent. Measured on one generated dataset:

| preset | network nodes | core-link entities | network edges |
|---|---:|---:|---:|
| as it stood (no fabric) | 0 | 0 | 0 |
| with the grid-declared backbone | 20 | **12** | 32 |

12 link entities at `n_core=12` — Phase 2's bounded-degree property, confirmed on the corpus
that will actually be trained. The default now lives in the **grid preset**
(`backbone_defaults`), not in an operator's flag, because a grid whose entire question is
topology must not depend on someone remembering an argument. `--link-bandwidth-mbps` still
overrides.

Non-binding rather than contended, deliberately: this lineage asks whether the GNN uses
topology *structure* to generalize. Link contention is `link_contention_v1`'s question and it
is already answered (real, but 0.08–0.35% regret). Stacking a known-small, known-noisy
mechanism onto a signal being resolved at MDG ≈ 0.02 is how `netc_hotspot_v1` lost
attribution. **Contention-under-transfer is a follow-on lineage, not a rider on this one.**

**2. `n_core` stays FIXED at 12; it does not scale with servers.** So the transfer axis is
**candidate-set growth over a fixed-complexity fabric** — candidates/task 2.38 → 9.96 (4.19×)
while core links/route go 3.13 → 3.02 and routes using ≥1 core link 92% → 91%. Scaling
servers hangs more nodes off the same ring without lengthening routes.

**The claim this corpus can support is therefore "generalizes across candidate-set growth",
not "generalizes to larger networks"** — narrower, and to be reported that way. Scaling
`n_core` is defensible in principle but is **untested against Phase 2's aggregation-invariance
property** (GIN sums, so any degree growing with N shifts embedding magnitudes with N);
testing it honestly means re-running the degree-bound asserts at every rung, which is a
separate phase with its own budget. Folding it in here would mean a negative result could not
say which half failed.

**3. The power ladder — enter at tier 0.02, with escalation pre-committed now.**
`PHASE4_TIERS` in `gate_statistics.py`, fixed before any corpus exists, for the same reason
v1's criterion was: choosing a threshold *after* seeing a borderline number is how a gate gets
falsified. Tier 0.02 is a cheap first pass (360/held-out size, ≈ 3.6 h wall-clock at 32 cores,
+~10% with the backbone), **not a standalone decision** — a straddling CI there auto-escalates
to tier 0.01 rather than being reported as "topology transfer failed".

**Seed count, re-derived after the determinism fix — and the answer is "spend on datasets,
not seeds".** The old ≥3-seed control was set against a spread that measured run-to-run
autograd noise, so it had to be re-measured, not reused. Five deterministic seeds on
`shallow_v1` (frozen above):

| quantity | value |
|---|---|
| across-seed sd of `win_rate` | **0.0508** |
| sd from test-split resampling alone at n=30 | 0.0913 |
| ratio | **0.56** |

**Across-seed sd is *below* what pure test-split resampling would produce**, so seed-to-seed
variation here is dominated by which 30 datasets land in the split — not by initialization.
Two consequences:

- **More seeds is the wrong purchase.** Resolving the mean effect at this split size would
  need ~**19** seeds; the same resolution comes far cheaper from a larger held-out set.
- **The prior intuition that ≥3 was overkill is falsified** — 5 seeds still leave the CI on
  the mean straddling 0.5. Seeds were never the binding term in either direction.

Phase 4 allocation: keep seeds at **5** (cheap, and they still buy the variance estimate and
the tail-behaviour spread that seed 44 exposes) and put the budget into datasets per held-out
size, which is what the power ladder below is denominated in.

**A units correction the implementation forced.** The MDG table above is in *regret-gap*
units; the primary statistic is `win_rate`. Doing the power arithmetic properly in win_rate
units (CI half-width of a proportion, ≈1.96·√(0.25/n)) against the effects actually observed:

| observed effect \|`win_rate` − 0.5\| | datasets/held-out size needed | covered by |
|---|---:|---|
| 0.033 (seed 42) | ~880 | tier 0.01 (1,600) |
| 0.017 (seeds 43/44) | **~3,400** | **no registered tier** |

So if the true effect sits at the *bottom* of the observed range, **even tier 0.01 will not
resolve it** (~16 h wall-clock buys an answer only for the top half of the range). The gate
returns `INCONCLUSIVE_LADDER_EXHAUSTED` with the required n in that case — never `FAIL`.

**✅ DECIDED 2026-08-19 — `tier_launch` registered at 900/held-out size, not the ~3,400/size
tier.** 900/size covers the *stronger* observed effect (0.033, needs ~880) at roughly
`tier_0.02`'s already-budgeted cost, and is the tier Phase 4 corpus generation actually
launches at — `tier_0.02` is a cheap first pass, and a straddling CI there is expected to
escalate to `tier_launch`, not `tier_0.01`. The ~3,400/size tier for the *weaker* effect
(0.017) is deliberately **not** pre-registered: at ~35–40h wall-clock per held-out size it is
a datalab allocation, not a speculative local run, and whether it is worth running depends on
whether `tier_launch`'s result is itself informative. It stays documented here as a sized,
known escalation path — the trigger to ask for datalab time is `tier_launch` itself coming
back `INCONCLUSIVE_LADDER_EXHAUSTED`, not before.

#### Corpus generation, cache build, and the Phase 4 gate — `FAILED` for `gnn_base` 5/5 seeds (2026-08-20)

**Corpus generation completed.** SLURM job 704238 (50-way array, `topo_transfer_v1_cosim.sbatch`)
finished: 4,500 datasets total (900/topology-size at 20/28/40/60/80 servers), 756 legitimately
`SKIPPED (infeasible)`, **3,744 successfully generated** with full `placements/placements.jsonl`
sweeps. A repair pass (`refresh_optimal_full_stats.py --repair`) fixed 550 datasets that had an
`optimal_result.json` but were missing `system_state_captured_unique.json` — a race condition
where the cache-build's first read hit a still-running generation shard (this is the **fourth**
time this exact bug has bitten; see the GATE TOOLS table). Graph cache built successfully at
`simulation_data/graphs_cache_topo_transfer_v1` (SLURM job 705771, 128GB mem, CPU-amd,
`NETWORK_GRAPH_CONTRACT=core_v1` + `TOPOLOGY_FEATURE_CONTRACT=size_invariant_v1` +
`QUEUE_FEATURE_CONTRACT=scale_invariant_v1`): 3,744 graphs, 100% valid labels, avg 88.4
edges/graph, RTT hash table 120.5M entries.

**Phase 3 implemented.** `--split-mode topology_size` in `scripts_cosim/gnn_necessity_ablation.py`:
train on `server_node_count ∈ {20,28,40}`, hold out `{60,80}`; the validation slice is drawn only
from train-size parents and never touches a held-out size. Guarded by
`scripts_cosim/test_topology_size_split.py` (11 tests, passing). The split uses a plain
`random.Random` shuffle rather than `sklearn.train_test_split`, because datalab's `gnn`
micromamba environment currently has a broken scipy/sklearn ABI (scipy 1.17.1 vs sklearn 1.6.1,
incompatible compiled extensions) — a pre-existing, shared-environment problem, deliberately
**not** fixed here since other sessions depend on that environment and other splits
(`split_ids_by_canonical_parent`, used by every non-`topology_transfer_v1` lineage) still use
sklearn on purpose, so existing frozen reports keep their exact shuffling.

**A merge regression, found and fixed before the reported run.** Two independent uncommitted
working trees (local repo, datalab checkout) had each separately extended
`gnn_necessity_ablation.py` — datalab's copy gained network-entity model support
(`use_network_entities` on `AblationModel`, **not wired into any of the three trained configs
below — dead capability**, not exercised by this run) and a label-provenance preflight audit
(`audit_label_provenance`); local's gained the `topology_size` split. Merging datalab's version
as the base with the split layered on top silently regressed a correctness fix in the eval loop:
"predicted plan absent from the placement sweep" stopped distinguishing a plain task-collision
(expected, not a bug) from a plan missing with no collision to explain it (a real corpus/harness
bug), and crashed on the former. Fixed within this session — `eval_regret` now reports separate
`n_missing_collided` / `n_missing_clean`, and only `n_missing_clean` triggers the fail-loud
`RuntimeError`. The first submission (SLURM job 705777) hit this bug and all 5 array tasks
failed after ~50 min; job 705834 below is the corrected rerun. Recorded as its own row in the
GATE TOOLS table — this is the same class of loss as `mp_parity`'s train/serve split: two
diverging copies of one file silently dropping each other's fixes.

**The gate run.** SLURM job 705834 (5-task array, GPU-a100, ~2h47m/seed), 5 pre-registered seeds
(42–46), `--split-mode topology_size --train-sizes 20 28 40 --held-out-sizes 60 80
--power-tier tier_launch --epochs 120 --models pointwise gnn_base gnn_node`. Label-provenance
audit **passed 3,744/3,744 on every seed** (`label_regret_mean = 0.0`, every cached label is its
dataset's true sweep minimum). `n_train=2280, n_val=403, n_test≈1061` graphs/seed. Frozen reports:
`simulation_data/topo_transfer_v1_phase4_seed{42,43,44,45,46}.json` (schema_version 5).

Per-seed paired `win_rate` vs. `pointwise` (`n_paired` ≈ 587–600; verified against the JSON
`paired_comparisons` / `phase4_verdicts` blocks, not just stdout):

| seed | `gnn_base` win_rate [95% CI] | verdict | `gnn_node` win_rate [95% CI] | verdict |
|---|---|---|---|---|
| 42 | 0.469 [0.447, 0.489] | **FAIL** | 0.480 [0.459, 0.501] | ESCALATE (effect 0.020, needs n≥617) |
| 43 | 0.436 [0.415, 0.457] | **FAIL** | 0.412 [0.390, 0.433] | **FAIL** |
| 44 | 0.442 [0.422, 0.462] | **FAIL** | 0.500 [0.482, 0.517] | INCONCLUSIVE_LADDER_EXHAUSTED (effect ≈0.000) |
| 45 | 0.454 [0.435, 0.473] | **FAIL** | 0.466 [0.447, 0.486] | **FAIL** |
| 46 | 0.478 [0.459, 0.497] | **FAIL** | 0.497 [0.477, 0.518] | INCONCLUSIVE_LADDER_EXHAUSTED (effect ≈0.003, needs n≥37889) |

`gnn_base` also loses on `regret_gap_mean` (negative in every seed, i.e. `pointwise` has the
lower — better — regret) and its co-primary sign agrees with `win_rate` in all 5 seeds, so
nothing here rests on a single-seed sign flip the way `mp_parity`'s residual did.

**Pooled across the 5 seeds (`pooled_phase4_verdict`, same tool the partial-corpus run above
uses), the per-seed picture holds — this is not seed noise:**

| model | pooled win_rate | 95% CI | verdict |
|---|---:|---|---|
| gnn_base | 0.456 | [0.440, 0.471] | **FAIL** — CI excludes 0.5 below, `co_primary_sign_agree=False` |
| gnn_node | 0.471 | [0.439, 0.502] | `ESCALATE` — effect ≈0.029, needs n≥690 (next tier above `tier_launch`) |

Unlike the partial-corpus run's pooled result above (both `INCONCLUSIVE_LADDER_EXHAUSTED` on a
71%-complete corpus), this is the **complete** 3,744/4,500 corpus and `gnn_base`'s pooled CI is
unambiguous. This supersedes the partial-corpus Phase 4 section above for `gnn_base`: FAIL, not
inconclusive. `gnn_node` is still not resolved either way.

**The likely cause: the held-out topology sizes carry essentially no coupling to exploit.**
`greedy_baseline` (additive-argmin, no training) shows **0/1,022 held-out test datasets with
regret > 1%, on every one of the 5 seeds** — the same signature this repo has hit five times
before (`graph_structure_physics`, `shallow_v1`/`shallow_longexec_v1`, `contention_v4_v5`,
`link_contention_v1`, `mp_parity`). At 60/80-server topologies the task-placement problem the
corpus poses is additive, so a model with strictly *more* expressive power than an additive
baseline (GIN) has nothing extra to win on, while it still pays a generalization tax the
pointwise model — which has fewer parameters coupled to graph structure — does not.

**Control 4 (the no-learning drift anchor) is NOT present in the frozen reports.** Schema 5's
fields are `cache, corpus_root, coupled_dataset_ids, coupled_results, coupled_threshold, epochs,
greedy_baseline, label_audit, models, n_graphs, n_test, n_train, n_val, paired_comparisons,
paired_reference, phase4_verdicts, power_tier, results, schema_version, seed, split_mode,
test_fraction` — there is no `drift_anchor` (or equivalent) key. The pre-registered control that
was meant to distinguish "the GNN held up" from "the statistic drifted" was never wired into this
harness run, so **this gate result stands on `win_rate` + `regret_ratio_mean` sign-agreement
alone, not on the full v2 control set.** That does not change the `gnn_base` verdict (a CI
excluding 0.5 below is a FAIL under the v2 rule with or without the anchor), but it does mean the
`gnn_node` ESCALATE/INCONCLUSIVE results have one fewer corroborating signal than pre-registered,
and the anchor should be added before spending the ~3,400/size `tier_0.01` budget this file
documents above.

**Verdict: this is a gate FAILURE for `gnn_base`, unambiguously — 5/5 seeds, CI excludes 0.5 on
the pointwise side every time, effect sizes 0.022–0.088, all in the direction of the reference
winning.** `gnn_node` never PASSES in any seed either: 2/5 FAIL outright (43, 45), 3/5
INCONCLUSIVE-but-pointed-at-null (42 ESCALATE at 0.020, 44 and 46 at effects ≈0.000–0.003 that
would need n≥37,889 to resolve — i.e. indistinguishable from parity, not evidence of a hidden
win). Do not read the `gnn_node` non-FAILs as "might still win" — the observed effects for the
three non-FAIL seeds are converging on zero, not on a positive gap obscured by noise. Under this
lineage's own pre-registered rule, `tier_launch` licenses the conclusion "does not transfer" for
`gnn_base`; `gnn_node` would need the (documented, not-yet-approved) ~3,400/size escalation to
settle even the null it is trending toward, and given the 0/1,022 coupling finding above there is
no positive prior that spending it would change the sign. Candidate-set-growth topology transfer,
as this corpus supports the claim, is **falsified for `gnn_base`** and shows no positive signal
for `gnn_node`.

### topology_transfer_v1 — the `gnn_topo` arm and the co-sim/live-gate scope gap (2026-08-20)

**Why this pass happened.** The FAIL above was for `gnn_base`/`gnn_node`, and neither config in
`all_configs` (`gnn_necessity_ablation.py`) ever set `use_network_entities=True`. That flag is
the *only* pathway that gives the model access to backbone/link graph entities
(`net_node_features`, `net_link_features`, `net_edge_index` via `src/placement/network_graph.py`,
contract `core_v1`) — `gnn_node`'s `use_node_edges=True` only adds same-node platform↔platform
edges, not network topology. So the original gate never actually tested whether topology-aware
message passing helps; it tested two topology-blind bipartite GINs against pointwise. The graph
cache (`graphs_cache_topo_transfer_v1`) was already built under
`NETWORK_GRAPH_CONTRACT=core_v1 TOPOLOGY_FEATURE_CONTRACT=size_invariant_v1`, so the data needed
was present and unused.

**Fix and re-run.** Added a fourth arm to `gnn_necessity_ablation.py`'s `all_configs` and
`--models` choices: `gnn_topo = dict(use_gin=True, use_node_edges=False,
use_network_entities=True)`. Re-ran the identical pre-registered gate (topology_size split, train
sizes 20/28/40, held out 60/80, `tier_launch`, 120 epochs, seeds 42–46) via SLURM job 706415 on
datalab (GPU-a100s, ~2h/seed, all 5 array tasks completed cleanly, no errors). Training loss
curves for all 5 seeds are smooth and monotonic (1.05–1.07 → 0.49–0.50, no NaN/stall); the one
lower-performing seed (43: win_rate 0.393 vs. the other four's 0.43–0.49, top1_acc 0.819 vs.
0.82–0.85) converged identically to the rest — its weaker result is ordinary seed variance, not a
training pathology. Frozen reports:
`simulation_data/topo_transfer_v1_phase4_topo_seed{42,43,44,45,46}.json`.

| seed | `gnn_topo` win_rate | 95% CI |
|---|---:|---|
| 42 | 0.486 | [0.465, 0.508] |
| 43 | 0.393 | [0.371, 0.415] |
| 44 | 0.471 | [0.451, 0.491] |
| 45 | 0.434 | [0.413, 0.455] |
| 46 | 0.460 | [0.439, 0.482] |
| **pooled (5 seeds)** | **0.449** | **[0.417, 0.481]** |

Pooled `regret_ratio_median = 1.0004` (per-seed range 0.9989–1.0022, i.e. indistinguishable from
1.0 — `gnn_topo` and `pointwise` have essentially identical regret magnitude, not just a
placement-choice disagreement). Co-primaries agree (`win_rate` and `regret_ratio` both say the
reference wins). Effect is **resolved, not underpowered**: `required_n` for this effect size is
230, actual pooled `n_paired` is 593. Pre-registered verdict at `tier_launch`: **FAIL** — CI
excludes 0.5 below. **Giving the model backbone/link topology access did not close the gap.**

**The missing controls, now run (post-hoc, local, no retraining):**

- **Drift-anchor control** (`scripts_cosim/drift_anchor_check.py`, new this session): the
  no-learning identical-capacity pair (additive-fit argmin vs. additive+one-integer-repair,
  `separability_diagnostic.variance_decomposition`) drifts by only ≈0.005–0.006 across held-out
  sizes 60→80 on this corpus — nowhere near `shallow_v1`'s pathological 2.58 drift. This corpus's
  size axis is not obviously landscape-broken the way `shallow_v1`'s was.
- **`size_invariant_v1` feature-degeneracy check** (ad hoc, `src/placement/topology_features.py`
  `SourceFeatureContext.feature()` sampled directly per held-out dataset): not degenerate at
  60/80 servers — variance ≈0.01, 23 distinct values, mean 0.06–0.08 at both sizes. Rules out
  feature collapse as an explanation for the 0/1,022 zero-coupling finding above.
- **Per-size breakout of the original `gnn_base`/`gnn_node` result**
  (`scripts_cosim/topo_transfer_v1_per_size_gate.py`, new this session — re-buckets the frozen
  reports' `results[model]['per_ds']` by `server_node_count` recovered from
  `infrastructure.json`, no retraining): the pooled FAIL is **not a widening-with-size effect**.
  `gnn_base` win_rate goes 0.454 (size 60) → 0.460 (size 80), `gnn_node` goes 0.464 → 0.474 — both
  move *toward* parity at the larger held-out size, the opposite of a transfer-degradation
  signature, and both deltas (~0.006–0.010) are the same order of magnitude as the drift-anchor's
  own noise floor.

Net read: this is now a properly-controlled FAIL for the lineage's actual hypothesis
(topology-aware GNN vs. pointwise), not an artifact of an untested config, a degenerate feature,
a broken size axis, or a genuinely widening size-transfer gap — the loss looks like a small,
roughly size-flat generalization gap that giving the model topology access did not fix.

**⚠ Co-sim-only scope and live-gate traceability — read before planning further work on this
lineage.**

1. **Every result above, across all 20 seed-runs (5 seeds × 4 arms), comes from the co-sim
   brute-force pipeline only.** Each dataset is a **4-task synthetic snapshot**
   (`workload.json`: `rps=2, duration=1`, 4 `events`) — fixed regardless of `server_node_count`;
   only the *cluster* scales (20→80 servers), never the *workload*. This is a deliberate property
   of the `topo_transfer_v1` grid (brute-force enumeration over more than ~4 simultaneous tasks
   is not tractable), not a bug — but it means "topology size" in this lineage has always meant
   "cluster size around a fixed 4-task decision," never "workload scale." This project's actual
   goal (CLAUDE.md) is co-sim generation → training → **live-gate evaluation on real workloads**
   (e.g. `data/nofs-ids/traces/workload-200-200.json`, 800,413 events, `rps=200, duration=200s`).
   `topology_transfer_v1` has never reached that step. **This is not a documented methodology
   choice (do not conflate it with `regime_b`, which is an unrelated, already-`FALSIFIED`
   cold-burst physics lineage) — it is simply a step this lineage has not yet taken.**
2. **No lineage in this repo has ever live-gated across mismatched train/eval topology sizes.**
   Confirmed by search: existing live-gate scripts
   (`important/run_contention_v2_live_gate_one.sh`, `important/run_wssm_expanded_live_gate_one.sh`,
   etc.) take a single infra config used for both training assumptions and the live run; there is
   no train-size vs. eval-size plumbing anywhere in the harness. `src/executesimulation.py` takes
   one `--config` that fixes the live infra's node count and **does not cross-check it against
   any checkpoint's trained topology size** — running a model at the wrong size would fail
   silently, not loudly. Architecturally nothing blocks this (`PointwiseEdgeMLP` and the GNN's
   bipartite `build_inference_graph` are both candidate-pair-based, not fixed-size vectors), it
   has simply never been attempted for any lineage.
3. **No trained checkpoint from this lineage exists on disk.** `AblationModel`
   (`scripts_cosim/gnn_necessity_ablation.py`) has no `torch.save`/state-dict-persisting call
   anywhere in the file — every one of the 20 seed-runs (`pointwise`/`gnn_base`/`gnn_node` ×
   seeds 42–46, then `pointwise`/`gnn_topo` × seeds 42–46) trained a fresh model in-process,
   evaluated it, and discarded the weights. Only the eval-summary JSONs
   (`simulation_data/topo_transfer_v1_phase4{,_topo}_seed{42..46}.json`) survive. **There is
   nothing to deploy for a live-gate run yet.**
4. **`AblationModel` is not wired into the production live-serving path.** It is a standalone
   class defined only in `gnn_necessity_ablation.py`; `use_network_entities` and the
   backbone/link entity pathway do not exist anywhere under `src/policy/gnn/` or
   `src/policy/gnn_hetero/` (grep confirms zero hits). `set_models()`-based live inference
   (CLAUDE.md's "Model loading uses `set_models()`") has no code path for this config today.

**Before running any of this on real workloads (planned for a future session), that session will
need, in order:**
   a. Add checkpoint saving to `gnn_necessity_ablation.py` (or a follow-on script) — persist
      `state_dict()` plus a training manifest per run: arm name/config (`use_gin`,
      `use_node_edges`, `use_network_entities`), seed, `split_mode`/`train_sizes`/
      `held_out_sizes`, cache dir, and the corpus grid name (`topo_transfer_v1`) — so a saved
      checkpoint can be matched, unambiguously, to the exact topology size(s) it was trained on
      before it is ever pointed at a live `--config`. Nothing today enforces that match; a
      silent size mismatch would not raise.
   b. Either port `use_network_entities` support into the production scheduler
      (`src/policy/gnn/scheduler.py` / `gnn_hetero/scheduler.py`) or build a small live-inference
      adapter around `AblationModel` directly.
   c. Build (or select existing) infra configs at each topology size of interest compatible with
      `src/executesimulation.py --config`, and decide which real trace(s) under
      `data/nofs-ids/traces/` to run live-gate against (existing live-gate scripts in this repo
      use `workload-125-225.json`, not `workload-200-200.json` — neither has been used with a
      topology-size-varying infra before).

### siv1_full_corpus — the first real live-gate, and the infra-parity gap it closed (2026-08-20)

**Why this lineage went first.** Of the ACTIVE lineages, only this one is *deployable
today*: `models/near-rtt-v2-full-corpus-siv1-dim14-ce-only.pt` exists with a contract
sidecar, is plain bipartite (no MP, no network entities), and loads through the production
`src/policy/gnn/scheduler.py` path with no porting work. `topology_transfer_v1` — the
lineage that most loudly asks for a live gate — cannot take the step: it has **zero saved
checkpoints**, no production code path for `use_network_entities`, and an already-`FAILED`
co-sim gate (see its scope caveat above). `link_contention_v1` and `contention_v4_v5` are
`FALSIFIED`.

**The gap this exposed: nothing tied a checkpoint to the infrastructure it trained on.**
`src/executesimulation.py` never loads a co-sim `infrastructure.json`; it *regenerates*
topology from a space config + seed (`prepare_infrastructure_for_real_simulation`). Three
concrete consequences, all measured, none previously recorded:

1. **The existing sealed-holdout gate was out-of-distribution and said nothing.**
   `important/run_contention_v2_873_sealed_holdout.sh` runs 40/40 p50 and 50/50 p60 configs;
   the corpus is **20/20** on every one of its 2,651 datasets. Now warns loudly with the
   numbers.
2. **`--seed` overrides the *topology* seed** (`executesimulation.py:842-854`), so a seed
   sweep is a topology sweep. Measured against `contention_v2/ds_00000`: at `--seed 42`,
   **144/210 directed edges differ and 64/64 shared edges disagree on latency.** The new
   gate passes no `--seed`; replication comes from distinct verified cells.
3. **`TOPOLOGY_FEATURE_CONTRACT` was enforced nowhere in `src/`.**
   `require_matching_topology_feature_contract` existed with only test callers, so a
   `size_invariant_v1` checkpoint served `src_index_v0` dim-2 features would have failed
   silently. Now raises, as does a warmth-physics mismatch and a feature-layout mismatch.

**The one legitimate divergence, characterized.** Live topology is always a strict
*subgraph* of the corpus topology, differing only by the replica-reachability repair
`generate_infrastructure.py` applies after replica placement (step 2b, ~lines 712-778) —
edges a live run cannot reproduce because it autoscales from zero and has no replicas.
Every such edge is client↔server at exactly `base_latency`, and zero live-only edges exist.
**Its size scales with sparsity**, which is worth knowing before choosing gate cells:

| connection_probability | repair edges (directed) |
|---|---|
| 0.15 | 34/174 (**19.5%**) |
| 0.20 | 12/172 (7.0%) |
| 0.25 | 4-14/182-226 (1.8-7.7%) |
| 0.35 | **0/282** |
| 0.50 | **0/380** |

At p=0.15 nearly a fifth of the corpus's edges do not exist live. That is not a bug, but a
model gated only at p=0.15 would be evaluated on a materially sparser graph than it trained
on. The gate spans p=0.15-0.50 deliberately so this is visible rather than averaged away.

**Cells: minted, not selected.** All 52 datasets on disk but absent from the training cache
share their exact topology cell with a trained dataset (**0/52 unseen**, across 268 distinct
trained cells) — they were dropped for data quality (`exclude_bad31`), not held out. So the
gate mints 5 fresh cells instead: 20/20 `sparse`, connection probabilities drawn from the
corpus's own six values, topology seeds 9001-9005 (the corpus used 142 seeds, max 609).
In-distribution on every axis the sidecar declares, memorized on none.

**Checkpoint sidecars now carry corpus provenance.** New
`src/placement/corpus_provenance.py` derives it from the cache's own `dataset_ids` rather
than a hand-written constant, so the record cannot drift from the data. Single-valued axes
are emitted as scalars, genuinely multi-valued ones as `<field>_values` sets tested for
membership (the full corpus spans six connection probabilities — equality would be wrong).
`train_near_rtt.py` writes it going forward; the three `full-corpus-siv1` sidecars were
backfilled through the same function. Note `models/` is gitignored and moves by rsync, so
these sidecars are **not** version-controlled and can drift between machines — the mitigation
is that `derive_corpus_provenance(cache_metadata)` regenerates them exactly from the cache,
so a suspect sidecar can always be re-derived rather than argued about.

Tools: `scripts_cosim/verify_live_infra_parity.py` (+ `test_live_infra_parity.py`, 13 tests
including a control for each fatal class and for the `--seed` divergence).

**Outcome (2026-08-21): FAILED — GNN loses to Knative on all 5 cells.** First gate attempt
(job 707307, 2026-08-20) reported GNN beating Knative by 17-26% at the two sparsest cells
(p=0.15, p=0.20) and losing narrowly elsewhere — a pattern that looked like real
sparse-topology structure worth investigating. It was not: it was measurement noise.

**Root cause: unpinned `PYTHONHASHSEED` made every placement tie-break non-reproducible.**
`system_state.replicas` (and related node/hardware pools) are built as Python `set()`s in
`src/policy/{knative,knative_network,gnn}/orchestrator.py`; the schedulers then resolve ties
— least-connected `min()`, most-available `max()`, first-match hardware iteration, and the
candidate order fed into the GNN's argmax decode — by iterating those sets directly. Python
randomizes string/object hashes per process by default, so **the exact same
config+workload+policy produces a different `total_rtt` every time it's run in a fresh
process**, with nothing to do with pipenv vs. micromamba. Measured directly: four separate
runs of `knative/cell01_p25_s9001` (identical inputs, only `PYTHONHASHSEED` varying) gave
`total_rtt` of 41.5M / 42.7M / 47.9M / 64.8M — a 56% spread from tie-break luck alone, the
same order of magnitude as the "GNN wins" margin the first gate reported.

**Fix**: sort every such tie-break deterministically on `(node.id, platform.id)` (or plain
`sorted()` for hardware-type strings) at the point a set feeds a decision — 6 files:
`{knative,knative_network,gnn}/scheduler.py` and `{knative,knative_network,gnn}/autoscaler.py`.
Also pinned `PYTHONHASHSEED=0` in the live-gate scripts as defense-in-depth. Verified: two
different `PYTHONHASHSEED` values now produce bit-identical `total_rtt` on the fixed code,
both locally (pipenv) and on datalab (micromamba) — the two independently landed on the exact
same value (`46,556,946.73649`) for `knative/cell01`, closing the pipenv/micromamba
cross-check that first surfaced the discrepancy.

**Re-gated (job 708549, 2026-08-21, all 15/15 COMPLETED, 0 failures) — clean result, no more
noise:**

| cell (density) | knative | mlp | gnn | gnn vs knative |
|---|---|---|---|---|
| cell01 (p=0.25) | 46.6M | 219.6M | 65.8M | +41.4% worse |
| cell02 (p=0.35) | 39.5M | 27.1M | 50.5M | +27.9% worse |
| cell03 (p=0.15) | 43.9M | 259.6M | 61.5M | +40.0% worse |
| cell04 (p=0.50) | 34.9M | 27.0M | 42.7M | +22.1% worse |
| cell05 (p=0.20) | 46.9M | 374.1M | 52.6M | +12.0% worse |

`compare_sealed_live_holdout.py` (paired, single seed per cell): **GNN wins 0/5 cells**
against Knative (Knative 3/5, MLP 2/5); GNN also loses on p90/p99 elapsed-time at every cell
where it isn't already losing on `total_rtt`. No sparse-topology advantage survives the fix —
the density-dependent pattern in the first attempt was entirely tie-break noise. MLP's
earlier collapse at low density (4.7-8.0x Knative at p=0.15/0.20/0.25, ~0.7x at p=0.35/0.50)
is unaffected by this bug and stands as a separate, real finding.

> **QUALIFIED 2026-08-21 — the reasoning above was unsound, but the conclusion survives
> re-measurement.** The dims 9-11 estimator bug lives in `build_inference_feature_bundle`,
> which serves the **MLP path as well as the GNN path**; only Knative never touches it. So
> "unaffected by this bug" was an assertion with nothing behind it — the MLP column was
> measured through exactly the same divergent features as the GNN column.
>
> Job 709163 re-ran these cells with the fix in place, same trace, same cells, same
> (leaked) environment — a clean code-only A/B. The MLP barely moved:
>
> | cell | 708549 (pre-fix) | 709163 (post-fix) | Δ |
> |---|---:|---:|---:|
> | cell01 (p=0.25) | 219.6M | 223,741,718 | +1.9% |
> | cell02 (p=0.35) | 27.1M | 27,037,296 | −0.2% |
> | cell03 (p=0.15) | 259.6M | 262,234,499 | +1.0% |
>
> (Knative reproduced bit-identically at cell01: 46,556,947 both runs.) So the collapse is
> **real and not an artifact of the bug** — but note the asymmetry that needs explaining: the
> same fix moved the GNN by −23.3% and the MLP by ~1%. The likely reading is that the MLP's
> fitted weights put little mass on dims 9-11 while the GNN's decode is highly sensitive to
> them; that is a hypothesis, not a measurement. cell04/cell05 MLP arms were still running
> when this was written. See also the partial walk-back later in this file ("A related MLP
> finding that corrects prior speculation"), which reached a compatible conclusion from the
> trace side.

**This is `siv1_full_corpus`'s first live gate on a real workload, and it FAILED.** The
deployable checkpoint (`near-rtt-v2-full-corpus-siv1-dim14-ce-only.pt`) does not beat Knative
live. Per `topology_transfer_v1`'s entry above, an ablation-only or offline-greedy result is
not a substitute for this step — this *is* the step, and the checkpoint does not clear it.

### siv1_full_corpus — re-testing the FAIL against untested real traces (2026-08-21, 🔄 IN PROGRESS)

**Why.** The FAIL above rests on exactly one trace, `workload-125-225.json`. Five of this
repo's six lineage-level `FALSIFIED`/`FAILED` verdicts were decided entirely on 4-task
synthetic co-sim snapshots; this is the only one ever gated on a real trace, and n=1 trace is
the entire evidence base for its headline verdict. `data/nofs-ids/traces/` has 5 named
full-scale traces that have never been used in any gate
(`workload-{100-50,150-100,175-100,200-200}.json`, `workload-150-100-30k.json`) — this reuses
`run_full_corpus_siv1_live_gate.sh`/`.sbatch` unmodified (both already take `WORKLOAD` as an
env var) against the same 5 parity-verified cells and the same deployed checkpoint.

**⚠ Preliminary reversal found — GNN beats Knative 5/5 on `workload-150-100.json`, not 0/5.**
301,352 events (rps=150, dur=100), same cells, same checkpoint, run locally (pipenv, not
datalab), `PYTHONHASHSEED=0`, `OMP_NUM_THREADS=4` pinned:

| cell (density) | knative | mlp | gnn | gnn vs knative | recorded on 125-225 |
|---|---:|---:|---:|---:|---:|
| cell03 (p=0.15) | 27,287,566 | 20,596,204 | 23,260,591 | **−14.8%** | +40.0% |
| cell05 (p=0.20) | 26,455,031 | 161,323,506 | 23,047,726 | **−12.9%** | +12.0% |
| cell01 (p=0.25) | 24,471,537 | 22,206,010 | 24,252,626 | **−0.9%** | +41.4% |
| cell02 (p=0.35) | 24,300,898 | 16,637,375 | 21,176,714 | **−12.9%** | +27.9% |
| cell04 (p=0.50) | 20,426,553 | 16,520,619 | 19,334,793 | **−5.3%** | +22.1% |

Margins 0.9–14.8%, against a measured local noise floor of 0.05% (see GATE TOOLS below) — 18×
to 296× the noise. MLP beats Knative 4/5, also a reversal from the recorded 2/5.

**Environment ruled out as the explanation — replication control PASSED exactly.** Re-ran the
*recorded* trace (`workload-125-225.json`) locally on the same cells as a control, since the
recorded gate ran on datalab/micromamba and this session runs on pipenv/local. Knative
reproduced the recorded numbers to 3 significant figures on all 5 cells, and matched the one
full-precision value `LINEAGES.md` already quotes (the pipenv/micromamba cross-check for
`knative/cell01`) **exactly**: `46,556,946.73649` both times. So the local harness is
validated; the reversal on `workload-150-100` is not a pipenv-vs-datalab artifact. The
control's GNN arm — the model-dependent half of the check — was still running when this entry
was written; the reversal above should be read as preliminary until that lands.

**What this would mean if it holds.** `workload-150-100` differs from `workload-125-225` in
both rps (150 vs 125) and duration (100 vs 225), so this shows trace-dependence without yet
isolating which property drives it. `workload-175-100.json` (351,767 events, rps=175,
duration=100 — holds duration fixed against `workload-150-100`) is queued to separate rps
from duration. `workload-200-200.json` (800,413 events — the trace this file has named three
times as an aspirational target and never run) is queued last, at reduced parallelism (GNN
measured ~2.9 GB RSS/worker on the 301k-event trace, so PAR was cut from 5 to 2 for the 800k
one to avoid OOM on a 32 GB box).

**Do not treat the FAIL above as retracted yet.** One trace reversing does not overturn a
result — it demonstrates the result is trace-dependent, which is itself the finding worth
having either way. If the GNN control arm also passes and `workload-175-100` /
`workload-200-200` corroborate, the honest updated claim is "the deployed siv1 checkpoint
beats Knative on some real traces and loses on others" — a materially different, more
interesting result than an unconditional FAIL, and one that would need its own root-cause
work (what about `workload-125-225` specifically makes Knative win there) before either verdict
is final. If they contradict `workload-150-100` instead, the original FAIL stands and
`workload-150-100`'s result becomes the thing needing an explanation.

**A related MLP finding that corrects prior speculation.** The recorded gate's open thread
(this file, "MLP's earlier collapse at low density") does not reproduce as a density effect on
`workload-150-100`: the *sparsest* cell (p=0.15) is fine at 0.75× Knative; only p=0.20 collapses
(6.10×); MLP beats Knative in 4/5 cells. The `decode_stats.json` sidecars (previously unread —
see GATE TOOLS) show the collapsing cell's `chosen_queue_vs_min` **median** is unremarkable
(119, in line with the other cells' 84–147) while its **p95 is 15,401**, ~30× the other cells'
~500 — a rare catastrophic-choice tail, not a systematically worse policy. Collision rates are
flat across all 5 cells (0.146–0.179), ruling out intra-batch collisions. Which cell exhibits
the tail moved between the two traces (p=0.15/0.20/0.25 on `workload-125-225`, only p=0.20 on
`workload-150-100`), so this reads as a trace-dependent instability, not a property of sparse
topology — revise the "sparse topologies force remote placements" hypothesis accordingly.

**Group A retest scope, decided this session.** `LINEAGES.md` records six lineage-level
`FALSIFIED`/`FAILED` rows; this siv1 retest and a `link_contention_v1` real-trace A/B (below)
are the two live-gate-shaped ones. The other four were evaluated for the same treatment and
found not to fit:
- **`contention_v4_v5`** — its failure is a ratio between two terms of the ECT cost formula
  (`depth × exec_time` grows with the lever, `added_in_batch × exec_time` does not), provable
  by reading `scheduling_cost.py` and confirmed on all 899 `contention_v2` datasets. A live
  trace reports `total_rtt`, which has no additive-R² to report — not retestable this way.
- **`topology_transfer_v1`** — is live-gate-shaped in principle (see its own §a/b/c list
  above) but needs checkpoint saving, a production `use_network_entities` serving path, and a
  multi-size GPU retrain (~14 GPU-hours at job 705834's pace) before a live cell can even be
  minted. Blocked this session on GPU availability (none locally; needs datalab).
- **`regime_b`** (RETIRED) — its 5 on-disk checkpoints have no `.contract.json` sidecar, so
  `load_gnn_model`'s warmth-physics guard (`executesimulation.py:486`) silently no-ops instead
  of raising, and its only real traces are `rps=0` cold-burst files (28–64 events). Also
  superseded per CLAUDE.md.
- **`soft_combo`** (RETIRED) — its matched checkpoint pair
  (`near-rtt-v2-regime-b-oracle-split-cosim-dim16-{ce-only,soft-combo-conc}.pt`) inherits every
  `regime_b` problem above, and its training collection has no `space_with_network.json` to
  mint parity-verified cells from at all.

#### Resolution (2026-08-21, later the same day): the replication control FAILED on its GNN arm — and the cause is an uncommitted code fix, not the trace and not the environment

**The control's knative arm passed bit-exactly; its GNN arm did not reproduce on any cell.**
Local re-run of the recorded trace (`workload-125-225.json`), same cells, same checkpoint
(sha256 `4df64b6a…` verified identical local↔datalab), same pinned env:

| cell (density) | recorded gnn (datalab) | local gnn | recorded margin vs kn | local margin vs kn |
|---|---:|---:|---:|---:|
| cell01 (p=0.25) | 65,822,323.78 | 50,407,465.33 | +41.4% | +8.3% |
| cell02 (p=0.35) | 50,469,878.45 | 37,358,224.81 | +27.9% | **−5.3% (win)** |
| cell03 (p=0.15) | 61,505,759.92 | 51,550,526.55 | +40.0% | +17.4% |
| cell04 (p=0.50) | 42,661,515.61 | 32,269,908.29 | +22.1% | **−7.6% (win)** |
| cell05 (p=0.20) | 52,580,830.03 | 46,905,755.20 | +12.0% | −0.06% (tie) |

**Root cause isolated by a 7-run probe matrix on `gnn/cell01`** (every value below is the
same cell, trace, checkpoint, and thread pinning unless noted):

| run | code | machine / device | total_rtt |
|---|---|---|---:|
| recorded gate (job 708549_10) | datalab `4db48d9` (clean) | datalab GPU (A40) | 65,822,323.78 |
| GPU repro ×2 (jobs 709154_0/1, same node) | datalab `4db48d9` | datalab GPU (A40) | 65,849,376.41 / 65,865,441.89 |
| CPU-forced (job 709155, `CUDA_VISIBLE_DEVICES=`) | datalab `4db48d9` | datalab CPU-amd | 65,806,356.23 |
| **clean worktree at `4db48d9`** | **committed tree** | **local CPU** | **65,795,161.49** |
| local working tree (repl + exact re-run) | uncommitted diff | local CPU | 50,407,465.33 / 50,358,532.75 |
| local working tree, `OMP_NUM_THREADS=1` | uncommitted diff | local CPU | 50,518,223.26 |

Two tight clusters, 23.3% apart, split **exactly on the code version**: committed tree
65.80–65.87M (spread 0.11% across two machines, two devices, two BLAS thread counts, two
numpy/pyg versions — numpy 1.26.4/pyg 2.7.0 on datalab vs 2.3.0/2.6.1 locally); working
tree 50.36–50.52M (spread 0.32%). Environment, GPU-vs-CPU numerics, CUDA scatter-add
atomics, and library versions are all **exonerated** — GPU run-to-run wobble is ±0.04%,
thread-count wobble ±0.2%, both the same order as the residual F1 tie-break noise
(0.05–0.1%, see GATE TOOLS), and none of it cascades.

**The responsible diff is the dims 9-11 live-feature fix that was never committed.**
`src/placement/temporal_features.py` (in the working tree since 2026-08-19, untracked) and
the `feature_builder.py` hunk that calls it fix the audit's Bug 1 (estimate gated per
snapshot vs per platform) and Bug 2 (live estimate averaged over ALL task types incl. the
9.5×-outlier `cnn`, serving 0.0815 where the vocab-restricted training-side formula gives
0.0086). Job 708549 ran on datalab's clean tree — i.e. **the recorded FAIL was measured
serving the known-divergent live features the audit had already flagged, two days after
those features were fixed locally.** Everything else in the working-tree diff is
eliminated by the knative arm's bit-exactness (`46,556,946.73649` reproduced exactly
across both trees and both machines — the shared physics/simulation path is provably
unchanged) plus code review of the remaining GNN-path hunks (`gnn_model.py` changes are
opt-in-gated or behavior-identical refactors; `topology_features` v0 is bit-exact by its
own 40/40 verification).

**Consequences.**
1. **The recorded 0/5 FAIL is re-graded, not merely trace-dependent**: it measured the
   checkpoint through a train/serve-divergent live feature path. With train-consistent
   features (the fixed path — strictly closer to what the checkpoint saw in training,
   where the affected cache rows were 0.0, not 0.0815), the same checkpoint on the same
   trace goes **2 wins / 1 tie / 2 losses**, and wins **5/5 on `workload-150-100`** and
   **5/5 on `workload-175-100`** (kn 34.19/33.21/35.26/31.11/36.63M vs gnn
   30.99/30.67/32.63/27.77/32.67M — margins −7.5% to −10.8%; MLP wins 4/5 by −19 to −30%
   but collapses at cell05, +366%). The trace-dependence that remains is real but mild:
   `workload-125-225` is the hardest of the three traces for the GNN, not a 0/5 outlier.
2. **The audit's "decision impact is unmeasured" thread is closed**: measured, it is
   **23.3% of live `total_rtt`** on `gnn/cell01`, enough to flip gate verdicts. The
   audit's "live-gate corpora unaffected" statement was about the *cache* side of those
   corpora and remains true; the *live* side of every gate run on the committed tree
   served Bug 2.
3. **Process fix, mandatory before the next datalab gate**: the temporal fix (and the rest
   of the reviewed working-tree diff) must be committed and pushed so datalab and local
   run the same code — the gate/live scripts sync `models/` by rsync but `src/` by git,
   and an uncommitted src fix silently splits the two sides. Additionally,
   `run_provenance` should record `git describe --dirty` + a working-tree diff hash in
   every live result JSON (small `executesimulation.py` addition; deferred only until the
   currently-running sweeps finish, same rule as the F1 fix).
4. Measured noise floors for future margins: GNN local run-to-run 0.1–0.3% (residual F1
   tie-break + thread wobble), datalab GPU run-to-run ±0.04%. The live margins above are
   25–300× these floors.

`workload-200-200` (800k events) was still running when this was written — its result
extends the trace ladder but cannot change the re-grading above, which rests on the
matched-code probe matrix.

#### The environment axis, measured directly and closed (2026-08-21)

The resolution above exonerated the environment *by inference* — two clusters splitting on
code version, with library differences absorbed inside a 0.11% within-cluster spread. That
left one axis genuinely untested, and one assertion in this file simply false.

**The false assertion.** `CLAUDE.md`, `full_corpus_siv1_live_gate.sbatch` and this file all
stated that datalab gates run in the micromamba `gnn` env. They never have.
`run_full_corpus_siv1_live_gate.sh` calls `pipenv run python3` (50 call sites across 18 shell
scripts), and `pipenv run` resolves its own venv and shells straight past
`micromamba activate gnn`. Every cluster gate has actually run in an undeclared third
environment, `~/.local/share/virtualenvs/gnn-herosim-2TQKssTQ` — **torch 2.12.0+cu130**, where
the `gnn` env and local both have torch 2.5.1+cu121. The probe matrix above therefore compared
across a torch major *and* a CUDA major without anyone knowing.

**The measurement.** New tool `scripts_cosim/verify_venue_parity.py --mode logits` forwards a
committed 64-graph fixture (`tests/fixtures/venue_parity/`, 256 decisions / 1,738 scored edges)
through `near-rtt-v2-full-corpus-siv1-dim14-ce-only.pt` and diffs logits and per-decision
argmax against a committed reference. Run at `22e8f27` with md5-verified identical weights and
import closure on both sides:

| stack | python | torch | numpy | PyG | max\|Δlogit\| | argmax flips |
|---|---|---|---|---|---:|---:|
| local pipenv (reference) | 3.12.3 | 2.5.1+cu121 | 2.3.0 | 2.6.1 | — | — |
| cluster `gnn` env | 3.12.12 | 2.5.1+cu121 | 1.26.4 | 2.7.0 | **0.0** | **0 / 256** |
| **cluster pipenv venv (what gates actually used)** | 3.12.12 | **2.12.0+cu130** | 1.26.4 | 2.8.0 | **0.0** | **0 / 256** |
| cluster pipenv venv, 4 threads | " | " | " | " | **0.0** | **0 / 256** |
| local, `--device cuda` (negative control) | 3.12.3 | 2.5.1+cu121 | 2.3.0 | 2.6.1 | 1.9e-5 | **0 / 256** |

The last row is a deliberate negative control: it produces a nonzero delta, proving the probe
can detect a difference, so the zeros are a result rather than a broken comparison.

**Outcome: the library-version axis contributes exactly zero.** A torch major across a CUDA
major, a numpy major, and a PyG minor together move the logits by zero bits. Only the
accelerator moves them, at 1.9e-5, flipping no decisions — consistent with the ±0.09% GPU↔CPU
`total_rtt` figure already recorded above. **Keeping the repo in sync is sufficient for GNN
inference numerics; the environment leak is a provenance and governance defect, not a
numerical one.** It stays worth fixing (`${HEROSIM_PY:-pipenv run python3}` +
`export HEROSIM_PY=python3`) because a declared environment that is not the running
environment made this take three sessions to rule out — but it is hygiene, and it does not
block the lineage.

**The generalisable diagnostic, recorded so the next incident is cheaper.** Environment and
float drift perturb decisions *symmetrically* — a flip helps as often as it hurts. The
observed gap was one-directional on all five cells and never flipped sign, which is a
**biased-estimator signature**, not an environment signature. That pattern had already ruled
out the environment on day one. Diagnose sign pattern before venue. Full protocol and the
comparability checklist now live in **`PARITY.md`**; the hard rules are in `CLAUDE.md` and
datalab-pitfalls #8.

**Read-off for job 709163** (fixed code, still through the unfixed leak, GPU): with the env
axis measured at zero, 709163's GNN arm was predicted to land at the local ~50.4M cluster, not
the recorded 65.8M. If it landed at 65.8M instead, this measurement would be contradicted and
the `--mode logits` probe would not be capturing whatever the difference is.

#### ✅ The prediction is CONFIRMED — job 709163 complete, 15/15 (2026-08-21, 21:2x)

All 15 array tasks COMPLETED at datalab commit `6f24e36`. The GNN arm lands in the local
working-tree cluster on **every cell**, and the whole gate reproduces the local re-grading
cell for cell:

| cell (density) | knative | GNN (datalab 709163) | GNN vs kn | GNN (local worktree) | local vs kn | **datalab vs local** |
|---|---:|---:|---:|---:|---:|---:|
| cell01 (p=0.25) | 46,556,947 | 50,609,830 | +8.7% | 50,407,465 | +8.3% | **+0.40%** |
| cell02 (p=0.35) | 39,463,080 | 37,371,050 | −5.3% | 37,358,225 | −5.3% | **+0.03%** |
| cell03 (p=0.15) | 43,922,518 | 51,640,719 | +17.6% | 51,550,527 | +17.4% | **+0.17%** |
| cell04 (p=0.50) | 34,942,834 | 32,281,245 | −7.6% | 32,269,908 | −7.6% | **+0.04%** |
| cell05 (p=0.20) | 46,941,809 | 46,793,121 | −0.3% | 46,905,755 | −0.1% | **−0.24%** |

The recorded gate (708549) had `gnn/cell01` at **65.8M / +41.4%**. Same checkpoint, same cells,
same trace, same cluster, same GPU partition — **the only thing that changed is the committed
code**, and 15.2M of `total_rtt` went with it. Cross-venue agreement is now +0.03% to +0.40%,
i.e. inside the GNN's own 0.1–0.4% run-to-run floor on four of five cells and 0.40% on the
fifth. **The environment measurement is confirmed end-to-end at the `total_rtt` level, not
just at the logit level**, and nothing is left of the "datalab and local disagree" hypothesis.

Scored by `compare_sealed_live_holdout.py`: three-way paired cell wins **GNN 1/5 · MLP 2/5 ·
Knative 2/5**; head-to-head against Knative the GNN is **2W/1T/2L** (better on cells 02/04,
tie on 05 at −0.3%, worse on 01/03) — identical to the local verdict on this trace. The tool
prints *"GNN does not dominate MLP on sealed holdout — do not claim uniform live transfer"*,
which remains the correct reading **for this trace**; `workload-150-100` and `workload-175-100`
are where the checkpoint wins 5/5.

**The recorded "+12% to +41% worse, GNN 0/5" FAIL is now formally superseded**, not merely
re-graded: it was a measurement of the uncommitted dims 9-11 diff. Do not cite it.

**MLP arm, all 5 cells (final).** The catastrophic tail is real, survived the fix, and is
*worse* on this trace than on any other: 4.81× / 5.97× / 8.17× Knative on cells 01/03/05
(223.7M, 262.2M, 383.5M), against 0.69× and 0.77× on cells 02/04. Pre-fix vs post-fix the MLP
moved −0.2% to +1.9% — so the fix that moved the GNN 23.3% barely touches the MLP, confirming
the asymmetry hypothesised in the qualification box above. Root cause now measured; see the
next subsection.

#### The MLP catastrophic tail, root-caused (2026-08-21)

Standing open thread since the recorded gate ("why does the pointwise baseline blow up at
sparse topologies?"), guessed at twice as `intra_batch_platform_collisions`. It is not that.
Measured across 4 local sweeps + job 709163 — two traces, two venues, 8 collapse instances:

| stat (`stats` block of the result JSON) | collapsed MLP | healthy MLP | knative | gnn |
|---|---:|---:|---:|---:|
| `averageOccupation` | **1.0–1.5** | 13.6–13.8 | 8.7–10.6 | 5.1–5.6 |
| `endTime` | 5.1–7.8× knative | 0.4–0.5× | 1.0× | 1.4–1.7× |
| `averageQueueTime` | 467–535 s | 48–54 s | 78–88 s | 76–92 s |
| `chosen_queue_vs_min` p95 | **8,964–23,731** | 504–1,129 | — | 539–1,083 |
| `chosen_queue_vs_min` **median** | 4–129 | 69–192 | — | 8–163 |
| `intra_batch_platform_collisions` | 0.14–0.18 | 0.13–0.17 | — | 0.22–0.45 |

Every physics component (cold start, execution, communication, `offloadingRate`, cache hits)
is unchanged between collapsed and healthy runs, and the collision rate is *normal* — the
collision hypothesis is dead. What separates them is **occupation collapsing to ~1**: the
cluster idles while a few replicas hold runaway queues.

**The MLP wins by packing and loses by packing.** Its healthy occupation (13.6) is higher than
Knative's (10.6) and far above the GNN's (5.6) — aggressive packing is exactly why it beats
Knative on raw margin on most cells. Its pointwise score has no term that sees a platform's
queue *relative to the alternatives*, so when a packed platform cannot drain, the queue runs
away and `total_rtt` goes 4–8×. The median `chosen_queue_vs_min` is normal in every collapsed
run, so this is a compounding minority of decisions, not a systematic mis-ranking. Collapse
risk rises as topology gets sparser and durations get longer: only cell05 (p=0.20) collapses
on `workload-150-100`/`175-100`, while the longer-duration `workload-125-225` collapses the
three sparsest cells and spares p=0.35/0.50. **The GNN has never collapsed in 20 cell-runs**
(worst p95 1,083). `averageOccupation ≈ 1` is the cheap detector — no need to parse the 200 MB
result JSON. A fix belongs in the MLP decode (a queue-relative term, or a cap on
`chosen_queue_vs_min`), not in the simulator.

#### The corrected-cache retrain — the residual train/serve gap, measured (2026-08-21)

The dims 9-11 fix landed 2026-08-19; `graphs_cache_full_corpus_siv1_dim14` was built
2026-08-15. **The deployed checkpoint was therefore trained on pre-fix features and is now
served post-fix ones**, which is the same defect class as [[gnn-train-serve-mp-mismatch]].
Rebuilt the cache post-fix into a **new** directory (`..._tempfix`, job 709232, 2 min;
`CACHE_DIR` had to be made overridable first — the script `rm -rf`s it, and the default is the
deployed checkpoint's only training data) and diffed the two, 2,651 shared datasets:

| | siv1 full corpus | `shallow_v1` (for scale) |
|---|---:|---:|
| platform rows changed | **175,034 / 551,408 = 31.7%** | 18.7% |
| graphs with any change | **2,651 / 2,651 (100%)** | — |
| max abs delta | **2.775** | 0.0086 |
| changes escaping dims 9-11 | 2 graphs | 0 |
| labels `y` changed | 2 graphs (0.08%) | 0 |

**The perturbation is ~320× larger than the one measured on `shallow_v1`, and it touches every
graph in the corpus.** So the live wins on `workload-150-100`/`175-100` were achieved by a
checkpoint whose training inputs disagree with its serving inputs on a third of all platform
rows — the wins are real but they are not the ceiling. Retrain submitted as job **709234**
(`near-rtt-v2-full-corpus-siv1-dim14-ce-only-tempfix`, GPU-a40, corrected cache, 2,658 graphs;
`OUT_CKPT` is now derived from `WANDB_RUN_NAME` so it lands beside the deployed checkpoint
rather than on top of it). **Not yet gated — the new checkpoint means nothing until it runs
the same 5-cell live gate on all three traces.** The MLP has the same mismatch and moved only
~1% under the fix, so a matched MLP retrain is the fair comparison and has not been run.

#### `workload-200-200` — the fourth trace lands: MLP sweeps, GNN 3W/2L, and no MLP collapse (2026-08-22)

The queued 800k-event trace (`workload-200-200.json`, rps=200, dur=200 s — the trace this file
had named three times as an aspirational target) completed locally as sweep `a4_wl200200`
(PAR=2, 15/15 results, pre-merge working tree — same fixed dims 9-11 live path as every other
row in this retest; `fix/deferred-gate-fixes` was merged only after the sweep finished).
Deployed checkpoint, same 5 parity-verified cells:

| cell (density) | knative | mlp (vs kn) | gnn (vs kn) | occ mlp / gnn |
|---|---:|---:|---:|---:|
| cell01 (p=0.25) | 167,472,990 | 109,546,546 (**−34.6%**) | 130,089,834 (−22.3%) | 7.8 / 6.7 |
| cell02 (p=0.35) | 113,833,627 | 95,576,423 (**−16.0%**) | 117,217,941 (+3.0%) | 12.5 / 9.5 |
| cell03 (p=0.15) | 136,967,398 | 112,812,746 (**−17.6%**) | 126,429,628 (−7.7%) | 7.5 / 7.5 |
| cell04 (p=0.50) | 107,589,923 | 94,604,668 (**−12.1%**) | 114,534,379 (+6.5%) | 12.2 / 8.8 |
| cell05 (p=0.20) | 139,429,374 | 130,150,818 (**−6.7%**) | 135,213,788 (−3.0%) | 3.6 / 6.1 |

`compare_sealed_live_holdout.py`: paired cell wins **GNN 0/5 · MLP 5/5 · Knative 0/5**; MLP
also takes 4/5 cells on both p90 and p99 per-task tails. Three findings:

1. **The trace-dependence claim is now the settled reading.** Deployed checkpoint vs Knative
   across the four full-scale traces: 2W/1T/2L (125-225) · 5/5 W (150-100) · 5/5 W (175-100) ·
   **3W/2L (200-200)**. "Beats Knative on some real traces and loses on others" is confirmed;
   the losses concentrate on the two *densest* cells here (p=0.35/0.50, +3.0%/+6.5%), the
   opposite cells from the 125-225 losses (p=0.15/0.25).
2. **The MLP's catastrophic tail does not appear at the heaviest load.** No collapse on any
   cell (occupations 3.6–12.5, nothing near 1; worst p99 is cell05's 2,416 s, still ~2× not
   4–8×). This bounds the collapse subsection above: 225 s durations collapse three cells,
   200 s durations at *higher* rps collapse none — so duration alone is not the driver either;
   the instability needs both long durations and something this trace lacks. First trace of
   the four where the MLP dominates both baselines outright.
3. **The GNN has still never collapsed** (20 → 25 cell-runs), but at rps=200 its conservative
   low-occupation placement (5.6–9.5 vs MLP's 7.5–12.5) costs it every head-to-head vs the
   MLP — the packing that destroys the MLP on 125-225 is exactly what wins at sustained high
   throughput when queues do drain.

#### ❌ The corrected-cache retrain FAILS its live gate — 0/15, uniformly, on all three traces (2026-08-22) — **FALSIFIED 2026-08-23**

> **⚠ This FAIL was a serving artifact, not a model or cache result.** The `tempfix`
> checkpoint's sidecar declares `inference_feature_layout: null`, so it served **`atomic21`**
> while the deployed checkpoint it was compared against served **`dim22`** from its sidecar.
> Re-serving `tempfix` under `dim22` on the same cells improves `total_rtt` by **13.3–40.8%
> (mean −29.8%)**, and it then wins **5/5 vs Knative at −14.0% mean** without a backbone and
> **5/5 at −34.1%** with one — beating the deployed checkpoint in both conditions. The
> corrected cache is **better**, not worse. See *"The backbone win is REGIME-LEVEL"*
> (`link_contention_v1`, 2026-08-23). The table below is retained as the record of what was
> measured, not as a finding about the cache.

The retrain (job 709235, `near-rtt-v2-full-corpus-siv1-dim14-ce-only-tempfix.pt`, corrected
post-dims-9-11 cache, contract sidecar present) ran the full 15-task gate on all three traces
(jobs 709296 / 709435 / 709436, datalab commit `37e5004` — pre-merge, same code lineage as
709163; `workload-{150,175}-100.json` had to be rsynced first, md5-verified both sides — they
had **never existed on datalab**, which also retro-explains why every prior datalab gate used
125-225 only). Deployed-checkpoint margins vs Knative alongside, for contrast:

| trace | tempfix GNN vs kn (cells 01..05) | deployed GNN vs kn | verdict |
|---|---|---|---|
| workload-125-225 | +47.3 / +53.4 / +63.3 / +42.2 / +27.0 % | +8.7 / −5.3 / +17.6 / −7.6 / −0.3 % | **0/5** (deployed: 2W/1T/2L) |
| workload-150-100 | +9.3 / +34.7 / +13.5 / +43.8 / +19.8 % | −0.9 / −12.9 / −14.8 / −5.3 / −12.9 % | **0/5** (deployed: 5/5 W) |
| workload-175-100 | +14.6 / +36.7 / +31.7 / +39.0 / +5.6 % | −9.4 / −7.6 / −7.5 / −10.8 / −10.8 % | **0/5** (deployed: 5/5 W) |

The comparison is clean: the Knative and deployed-MLP arms in these sweeps reproduce the
recorded values **exactly** (e.g. `knative/cell01` 24,471,537 and `mlp/cell01` 22,206,010 on
150-100, digit for digit) — same cells, same traces, same code; only the GNN checkpoint
differs.

**The inversion is the finding.** In-distribution the retrain is the *better* model: best val
acc 70.7% vs the deployed 66.3% (different caches, same corpus/split/hyperparameters/seed —
the runner was identical except `CACHE_DIR`/`WANDB_RUN_NAME`). Live it is uniformly 1.06–1.63×
Knative and loses every cell to the deployed checkpoint, on the traces where the deployed one
wins 5/5. Making training features consistent with serving features — removing the 31.7%-of-rows
mismatch — made the live policy drastically worse while making the co-sim metric better.

**What this does and does not establish.**
- It *supports* the §"retrain's own risk" reading (HANDOVER 2026-08-21): some of the deployed
  checkpoint's live edge is attributable to the train/serve mismatch itself — training on
  pre-fix dims 9-11 acted as an accidental regularizer / decision-bias that happens to help
  live, and "fixing" it removed that.
- It does *not* yet separate that from **retrain variance**: this is one training run against
  one training run. The discriminating control is a retrain on the **pre-fix** cache with the
  identical pipeline (only `CACHE_DIR` differing) — if that lands near the deployed checkpoint,
  the cache is causal; if it also collapses live, the deployed checkpoint is a lucky draw and
  the corpus/live gap is wider than either cache. **Not run.**
- The `logit_tied_rate ≈ 0.54` thread gets its first answer: the dims 9-11 feature bug was
  not what was holding the model back — correcting it helps no trace.

**Decision: the deployed checkpoint stays deployed.** The tempfix checkpoint is evidence, not
a candidate. — **REVERSED 2026-08-23:** at a fixed `dim22` serving layout `tempfix` beats the
deployed checkpoint on mean margin in both conditions (5/5 each), so it is now the leading
*candidate*. Promotion still wants a re-gate on `workload-175-100` and `workload-200-200`,
since its advantage is so far measured on one trace.

**The matched MLP control landed the same day (jobs 709495-97, `..._tempfix.pt` MLP retrained
on the same corrected cache, MLP arm only, same cells/traces) and confirms the prediction with
one refinement.** On the 9 healthy cells across the three traces the tempfix MLP moves
**−6.6% to +2.6%** vs the deployed MLP (most within ±3%) — the feature fix is inert for the
pointwise model, consistent with its ~1% logit movement. On the collapse-implicated cells the
fix does not fix or worsen the tail so much as **re-roll it**: 150-100 cell03 (p=0.15,
previously healthy at 0.75× Knative) *newly* collapses (+281%, occupation 1.10) while 125-225
cell01 improves 23% yet stays collapsed (occ 0.98); the always-collapsing cell05 gets worse on
both 125-225 (+45%) and 175-100 (+48%). So which cell collapses is chaotic under small feature
perturbations — reinforcing that the MLP tail is a knife-edge packing instability
([[herosim-mlp-collapse-is-occupation-collapse]]), not a feature-quality effect. **The
contrast stands: the GNN degrades systematically and one-directionally under the corrected
cache (every cell, every trace); the MLP is indifferent except where it was already unstable.
The mismatch-sensitivity is GNN-specific.**

#### ⚠ The variance control answers: it is a lottery — the deployed checkpoint is a lucky draw (2026-08-22) — **PARTLY SUPERSEDED 2026-08-23**

> **⚠ Read the correction first.** This comparison is **confounded by the serving layout**:
> the deployed checkpoint served `INFERENCE_FEATURE_LAYOUT=dim22` (declared in its sidecar)
> while `prefixctl` and `tempfix` served `atomic21` (sidecar silent, and `load_gnn_model`
> defaulted). Re-serving `prefixctl` under `dim22` on these same cells improves it by
> 1.2–14.8%, and `tempfix` by 13.3–40.8%. At a fixed layout the draw spread is **much**
> smaller than the 5–36% below, `tempfix` becomes the best checkpoint on disk, and the
> backbone win is regime-level. See *"The backbone win is REGIME-LEVEL"*
> (`link_contention_v1`, 2026-08-23). The qualitative point — that a single-checkpoint gate
> is a claim about one draw — survives; the magnitudes do not.

`...-prefixctl.pt` (train job 709516, gate job 709534): **same pre-fix cache, same pipeline,
same seed** as the deployed checkpoint — the only difference is GPU/dataloader nondeterminism
in one training run. (`MIN_GRAPHS` overridden to 2651: the guard postdates the deployed run,
which itself trained on 2,651 graphs; the tempfix cache has 2,658 — the rebuild picked up 7
extra datasets, a second small train-set delta.) In-distribution it is the deployed model's
twin: val acc **66.8% vs 66.3%**, greedy regret identical to 4 decimals (0.6467s). Live, on
`workload-150-100` (same cells, GNN arm only):

| cell | deployed | **prefixctl (control)** | tempfix |
|---|---:|---:|---:|
| cell01 (p=0.25) | −0.9% | **+4.8%** | +9.3% |
| cell02 (p=0.35) | −12.9% | **+14.8%** | +34.7% |
| cell03 (p=0.15) | −14.8% | **−0.6%** | +13.5% |
| cell04 (p=0.50) | −5.3% | **+28.5%** | +43.8% |
| cell05 (p=0.20) | −12.9% | **+4.2%** | +19.8% |

**Control: 1/5 vs Knative (and that one at −0.6%, inside noise) where the deployed draw won
5/5.** Per cell the control is +5.8% to +35.7% worse than the deployed checkpoint — from
training nondeterminism alone, against a 0.1–0.4% run-to-run simulation floor.

Three conclusions, in order of weight:

1. **Live quality of this pipeline is a training-draw lottery.** Two checkpoints
   indistinguishable in-distribution (0.5pp val acc, identical greedy regret) differ by
   5–36% of live `total_rtt` per cell. In-distribution metrics have **zero** discriminating
   power over which draw wins live — the sharpest form yet of the co-sim/live disconnect
   (`graph_structure_physics` / `logit_tied_rate` thread).
2. **The tempfix FAIL above is therefore confounded.** Cache ordering is consistent
   (deployed < control < tempfix on every cell), so the corrected cache *may* still be worse —
   but with one draw per cache and within-cache variance this size, the cache effect is not
   separable from draw luck. The "mismatch was an accidental regularizer" reading is
   downgraded from supported to *possible*; the demonstrated effect is the variance.
3. **Any single-checkpoint live-gate verdict in this lineage is a claim about a draw, not
   about the recipe.** Every siv1 gate row above (including the 709163 CONFIRMED one, which
   compared the *same* checkpoint file across venues, and is unaffected as a venue
   measurement) generalizes to the training pipeline only up to this lottery. A future gate
   that wants a recipe-level claim needs multiple training draws per arm — which the ablation
   harness already does (5 seeds) and the production pipeline does not.

**The deployed checkpoint stays deployed** — it is demonstrably the best live artifact on
disk. But it should be described as "the checkpoint that won", not "what the pipeline
produces".

### topology_transfer_v1 — unblocked, and two cost estimates corrected (2026-08-21)

No new gate result. This closes the lineage's §a blocker and re-costs its §b, both by
measurement rather than by plan.

**§a — weights are now persisted.** `gnn_necessity_ablation.py --save-checkpoints DIR` writes
`<arm>_seed<N>.pt` plus a `<arm>_seed<N>.contract.json`. Off by default, so an eval-only run
is byte-unchanged. The sidecar is the substance, not the `.pt`: a checkpoint without one is
read as `{}` by `executesimulation._read_checkpoint_sidecar`, and every downstream contract
check then adopts its default silently — `legacy_v0`, `src_index_v0`, no infra provenance, no
record of which message passing it was fitted with. The contract records the queue / topology /
network-graph contracts, feature layout, arm config, seed, and — for the `topology_size` split
— **which sizes were trained on and which were held out**. A checkpoint that cannot say which
sizes it never saw cannot be used to test transfer to those sizes, which is the whole
hypothesis. 14 tests in `tests/test_ablation_checkpoint.py`.

**§b — the serving port is a rename, not a build.** The plan costed a production
`use_network_entities` serving path as a multi-session job and assumed `gnn_base`/`pointwise`
"already load through `src/policy/gnn/scheduler.py`". Measured:

| | result |
|---|---|
| `AblationModel` vs `TaskPlacementGNN` state dicts | **31 keys each, 15 shared, same shapes** |
| difference | three top-level module names: `task_enc`→`task_encoder`, `plat_enc`→`platform_encoder`, `scorer`→`edge_scorer` |
| renamed load, `mp_residual=False` (production **default**) | `load_state_dict(strict=True)` **succeeds, no error** — and max \|Δlogit\| **0.196**, different argmaxes |
| renamed load, `mp_residual=True` (`mp_gate` inits to 1.0) | **max \|Δlogit\| 0.0**, identical argmaxes, `gnn_base` and `gnn_node` both |

`AblationModel` is unconditionally `x0 + gin(x0)`; production applies the residual only under
`mp_residual`, whose default is `False`. **So the cheap port also contains a silent
wrong-numbers path that nothing in the stack catches** — precisely the class this file's
checkpoint-contract rows keep recording. The verified port now ships in the contract as a
`serving_port` block (target class, key rename, constructor kwargs), with a test that fails if
the residual ever stops changing the output, so the guard cannot pass vacuously.
Recorded honestly: `mp_node_edges_candidates_only=False` comes from reading
`AblationModel.forward`, **not** from the equivalence check — that used a fully connected
bipartite graph, where the flag is a no-op.

**Remaining cost of the partial gate:** the ~14 GPU-hours (train `gnn_base` + `pointwise` with
`--save-checkpoints`, mint live cells at 60/80 servers, run the 15-task gate per size). Not
launched — it is a large speculative spend on a lineage already `FAILED`, and it would contend
with the siv1 retrain for GPUs. Nothing else blocks it.

**A6 (`soft_combo` live retest) is NOT viable as planned — checked, not assumed.** The plan
called it "the only item testable today with zero training" and flagged one caveat to verify.
Two hold, and either is disqualifying: (1) neither
`near-rtt-v2-regime-b-oracle-split-cosim-dim16-{ce-only,soft-combo-conc}.pt` has a
`.contract.json`, so both would serve blind under adopted defaults; (2) both take **16**
platform features (`platform_encoder.net.0.weight` is `(64, 16)`) against the siv1 gate cells'
**14** — before even reaching the `platform_reuse_v1` vs `node_disk_v2` physics mismatch.
Do not spend time on A6 without first retraining that pair under a recorded contract.


### cache_live_divergence_audit — outcomes (2026-08-19)

Its own lineage, not part of `topology_transfer_v1`: this is shared-infrastructure
correctness, found *by* the Phase 2 work but not caused by or specific to it. Tool:
`scripts_cosim/audit_cache_live_divergence.py`; full report in
`simulation_data/audit_cache_live_divergence_20260819.json`.

Phase 2's parity run against a netc corpus failed on ~20 checks. The first reading — a
platform-ordering train/serve mismatch of the `mp_node_edges` class — was **wrong**, and the
audit that was supposed to size its blast radius falsified it instead. Two separate things
were tangled together.

#### 1. Platform reordering — universal (18/18 collections) and BENIGN

The cache enumerates platforms from `stats.nodeResults`, live from
`config.infrastructure.nodes`. Those orders differ in **every collection in the repo**, by up
to 229 of ~230 rows. Two independent causes: `nodeResults` node order is not always ascending
(netc), and platform ids *within* a node get reordered by FilterStore churn (regime_b:
`[2,3,4,1]`).

It reads like a fatal bug because graph position is how a platform is addressed —
`platform_emb[edge_index[1] - n_tasks]`, with nothing carrying the platform id into the
lookup. **But `TaskPlacementGNN` has no per-position parameter.** The platform encoder is
row-wise and edges are relabelled consistently with the rows, so a different order is a
relabelling. Verified on `netc_multihop_v1_core4/ds_00000` (208 platforms, 74 rows moved) by
matching platforms on `(node_name, platform_id)`:

| compared by identity | result |
|---|---|
| platform identity sets | identical |
| bipartite edges | identical |
| candidate sets per task | identical |
| `node_edge_index` | identical |
| `task_features`, `edge_attr` | identical (0.0) |
| per-candidate logits, dims 9-11 equalized | **max diff 3e-8** (float32 noise) |

**No recache, no reverification, no asterisk on any result — including the GNN win.** The
`(node_id, platform_id)` sort fix was considered and is **not needed**; it would churn every
cached platform position for zero correctness gain.

What *was* broken is the gate tool. `verify_cache_live_feature_parity.py` compared platform
rows by position, so it reported ~20 failures on any reordered corpus — burying the one real
one and, worse, making it impossible to run the parity gate on the netc family at all. It now
compares **by platform identity** (features via a permutation, edges/candidates/same-node
edges as identity-keyed sets, candidate lists as sets since decoding is by identity) and
prints the reordering as a `note:` rather than a failure. Same dataset now reports 3 findings,
all one root cause; regime_b still passes on dim24/dim22/dim14.

#### 2. The dims 9-11 temporal estimate — REAL, on 8/18 collections

Dims 9-11 (`current_task_remaining`, `cold_start_remaining`, `comm_remaining`) are estimated
from queue depth when no remainder was recorded — but the two paths decide *at different
granularity*:

```
cache   prepare_graphs_cache.build_graph:  if temporal_state: <use recorded>  else: <estimate>
live    feature_builder:                   per platform: if queue > 0 and remaining == 0: <estimate>
```

So on a snapshot that has *some* recorded temporal data but a queued platform with no
remainder, **the cache writes 0 where live estimates**. Measured (20 datasets/collection where
SSC exists):

| collection | datasets affected | worst-case platforms |
|---|---|---|
| `gnn_datasets_1task` | 20/20 | 75 |
| `gnn_datasets_4tasks_skew_warmth_v2` | 12/12 | 70 |
| `gnn_datasets_4tasks_contention_v2` / `v3` / `v4_pilot` | 20/20 each | 54 |
| `gnn_datasets_4tasks_1060_warmth_v2` | 20/20 | 53 |
| **`gnn_datasets_4tasks_shallow_v1`** | 20/20 | 49 |
| `gnn_datasets_4tasks_sparse_warmth_v2` | 20/20 | 37 |

Clean: `highq_safe_20260606`, `regime_b_*_oracle_split_cosim`, and all three
`hetero_*_knative_eval` collections — i.e. **the live-gate and Knative-comparison corpora are
unaffected**. `shallow_v1` — the corpus behind the current necessity ablation — is affected.

Magnitude is small (0.0815 in a `/10`-normalized dim, i.e. ~0.8s of estimated remaining
execution counted as 0 during training) on 37-75 of ~200 platforms. **Decision impact is
unmeasured** — the 2e-4 logit shift observed on a random-init model says nothing about a
trained one's sensitivity. *(Measured 2026-08-21: on live inference it is 23.3% of
`total_rtt` on `gnn/cell01`, enough to flip live-gate verdicts — see the siv1 resolution
subsection above. The "unaffected" list below is the cache side only; the live side served
Bug 2 in every gate run on the committed tree until the fix is committed.)*

#### Fix + re-verification (2026-08-19)

Both bugs are fixed, and the formula now lives once in `src/placement/temporal_features.py`
(15 contract tests) with all four call sites calling it — the `queue_features` /
`topology_features` pattern. `netc_multihop_v1_core4/ds_00000` now passes cache↔live parity
outright, as does regime_b on dim24/dim22/dim14 and on the synthetic deep-queue variants that
specifically exercise the estimator.

**Cache diff** (`shallow_v1`, 200 graphs, old vs corrected): labels `y` unchanged in 200/200;
`task_features` / `edge_attr` / `edge_index` / `node_edge_index` byte-identical; dims 9-11
changed on **7793/41600 platform rows (18.7%)**, max 0.0086. So this is purely an
input-feature correction — the targets never moved.

**Re-verification: the win holds, and the aggregate numbers are outlier-driven.** Ablation
re-run 3 seeds x 2 caches, CPU (deterministic, so seed variance is real):

| | seed 42 | seed 43 | seed 44 |
|---|---|---|---|
| **old** cache | GNN | GNN | **POINTWISE** (gnn 1.044 vs pw 0.038) |
| **corrected** | GNN | GNN | GNN |

`top1_acc` barely moves anywhere (0.867-0.917; GNN ≥ pointwise in 5 of 6 runs). What moves
`regret_mean` is **one dataset**: `ds_00157`, regret exactly 30.422 wherever it appears
(old/s44 `gnn_base` + `gnn_node`, corrected/s44 `gnn_node`). Excluding the single worst
dataset per run, GNN beats pointwise in **6/6** runs on both caches:

| | s42 | s43 | s44 |
|---|---|---|---|
| old: gnn_base / pointwise | 0.0092 / 0.0093 | 0.0175 / 0.0279 | 0.0306 / 0.0295 |
| fix: gnn_base / pointwise | 0.0099 / 0.0127 | 0.0160 / 0.0206 | 0.0085 / 0.0309 |

**⚠ SUPERSEDED 2026-08-19 — this table does not reproduce, and the win does not survive the
corrected gate statistic.** Re-run at 120 epochs on the same corrected cache with
determinism enabled, seed 44 gives **POINTWISE** on `regret_mean` (gnn 103.41% vs pw 4.13%,
and again 104.36% on an identical repeat), not the GNN recorded above. Two causes, both now
fixed: the runs behind this table were made **before** `use_deterministic_algorithms` (so
they are one draw of a non-reproducible process — see GATE TOOLS), and `regret_mean` is the
statistic since demoted for drifting with sweep size. Under the primary statistic the result
is a coin flip in every seed:

| seed | `regret_mean` verdict | `win_rate` (gnn_base) | 95% CI | sign p |
|---|---|---|---|---|
| 42 | GNN (1.03% / 3.01%) | 0.533 | [0.450, 0.617] | 0.688 |
| 43 | GNN (1.76% / 2.80%) | 0.517 | [0.433, 0.600] | 1.000 |
| 44 | **POINTWISE** (103.41% / 4.13%) | 0.517 | [0.433, 0.600] | 1.000 |

`win_rate` is stable at 0.52–0.53 while `regret_mean` swings ~100× and flips the verdict.
Every CI straddles 0.5 and no sign test approaches significance. **The honest reading: on
`shallow_v1` the GNN and the pointwise model pick the better plan at statistically
indistinguishable rates.**

**⚠ The three-seed table above is itself unfrozen and pre-determinism on its GNN columns.**
It was computed in-session and never written to a report JSON, so it cannot be reproduced by
inspection. The provenance is now identifiable: re-running seed 42 deterministically
reproduces the recorded `pointwise` number *exactly* (3.01%) while the `gnn_base` number
differs (1.03% recorded → **1.10%** measured) — the signature of a pre-fix GIN draw against a
model that was always reproducible. **Superseded by the frozen 5-seed calibration below**
(`simulation_data/gnn_necessity_seed_calibration_shallow_v1_20260819.json`).

#### The frozen 5-seed re-run — deterministic, and it corrects the tail-risk claim

Same cache, 120 epochs, determinism on (the default), n=30 held-out per seed. Verified
bit-identical on a repeat of seed 42, across a harness edit, to the last digit:

| seed | gnn `regret_mean` | pw `regret_mean` | verdict | `win_rate` | 95% CI | ratio | gnn `regret_max` | pw `regret_max` |
|---|---:|---:|---|---:|---|---:|---:|---:|
| 42 | 1.10% | 3.01% | GNN | 0.500 | [0.417, 0.583] | 0.987 | 9.96% | 53.43% |
| 43 | 3.24% | 2.80% | **POINTWISE** | 0.450 | [0.367, 0.533] | 1.006 | 18.97% | 24.28% |
| 44 | **103.61%** | 4.13% | **POINTWISE** | 0.533 | [0.467, 0.617] | 1.999 | **3042.19%** | 34.35% |
| 45 | 4.30% | 5.44% | GNN | 0.550 | [0.450, 0.650] | 0.991 | 35.95% | 34.11% |
| 46 | 1.11% | 4.30% | GNN | 0.583 | [0.483, 0.683] | 0.975 | 13.32% | 34.11% |

**The coin-flip finding survives and strengthens.** `win_rate` mean **0.5233**, sd 0.0508,
95% CI on the mean **[0.479, 0.568]** — every per-seed CI straddles 0.5, and so does the CI
on the mean across five seeds. `regret_mean` still flips the verdict (GNN 3/5, POINTWISE 2/5)
while swinging 1.10% → 103.61%.

**But the tail-risk claim does NOT survive.** It rested on seed 42 alone. Across five seeds
the GNN's `regret_max` is better in 3 (42, 43, 46), a wash in 1 (45: 35.95% vs 34.11%), and
**catastrophically worse in 1 — seed 44 at 3042% vs 34%**. So "what the GNN buys is tail-risk
reduction" is *also* a one-seed artifact, and the honest surviving claim is narrower still:

> On `shallow_v1`, the GNN and the pointwise model pick the better plan at statistically
> indistinguishable rates, and the GNN's tail is better in most seeds but occasionally
> **far** worse. Neither a win nor a tail-risk advantage is established.

The co-primary is bimodal across seeds for the same reason (`regret_ratio_mean` ≈ 0.99 in
three seeds, ≈ 2.0 in the blow-up seeds), which was a **live gate-design risk**: the v2 FAIL
condition "the two co-primary statistics disagree in sign" fires whenever one seed lands in
the blow-up mode — seed 44 has `win_rate` 0.533 (GNN ahead) *and* `ratio` 1.999 (GNN far
behind). That disagreement is informative, not a defect, so a single tail seed should not
fail the lineage on its own.

**✅ RESOLVED 2026-08-19 — see GATE TOOLS.** `pooled_phase4_verdict()` evaluates the
sign-disagreement check on the pooled, multi-seed statistic (mean-of-seeds `win_rate`,
**median**-of-seeds `regret_ratio_mean`) instead of per seed, so one blow-up seed no longer
moves the pooled ratio enough to flip a verdict the other seeds agree on.

**The overlap check is negative, which is the reassuring answer.** The changed rows do *not*
concentrate where the models disagreed: mean changed-row fraction 0.171 on disagreement
datasets vs 0.198 on agreement datasets (old), 0.204 vs 0.193 (corrected), against a
corpus-wide mean of 0.187. Per-dataset flips are small and symmetric (pointwise 2 newly
optimal / 2 newly suboptimal; `gnn_base` 0/1; `gnn_node` 3/2). **The original win was not
riding on this bug.**

#### ⚠ Gate risk this surfaced — regret is heavy-tailed on a cliff-shaped corpus

`ds_00157`'s sweep: min 0.989, **median 33.06**, max 186.16 — a 188x spread, so almost every
plan is catastrophic and the optimum is a needle. Regret 30.4 means the model picked a plan
*better than the median* and still scored 30x. That is not a training collapse; it is the
landscape. And it is not rare:

| shallow_v1 (200 datasets) | share |
|---|---|
| median plan > 2x optimum | 74% |
| median plan > 5x optimum | 32% |
| median plan > 30x optimum | **22%** |

**This puts the pre-registered Phase 4 gate at risk.** It gates on the *slope of regret
against topology size* using mean / p90 / max — every one of which is dominated by whether a
model happens to find the needle on a handful of cliff datasets. Larger held-out topologies
have more plans and wider spreads, so the slope could measure outlier-catching rather than
transfer. **RESOLVED 2026-08-19 — see the next section. The tentative fix suggested here
(trimmed / log-regret, or promoting `top1_acc` / `opt_recovered_frac`) was measured and does
not work; the actual fix is structural.**

#### ✅ RESOLVED — the gate statistic decision (2026-08-19)

Settled by measurement, not argument, and **without training anything**: substitute a
decision rule whose expressive class is *identical at every size* — the additive-fit argmin
(M4), literally what `PointwiseEdgeMLP` can express — then bin `shallow_v1`'s 200 datasets by
sweep size (16 → 17,248 plans) and see which aggregate statistics still move. Since the rule
is constant, **every bit of movement is landscape, and would be misread as transfer
degradation.** Tools: `scripts_cosim/gate_statistics.py` (+15 contract tests).

All rows are the **gap between the two fixed rules**, normalised the same way
(range ÷ mean |value|) so they are comparable to each other. An honest gate would read 0.00.

| statistic (gap between two constant-quality rules) | drift | verdict |
|---|---|---|
| `regret_mean` | **2.58** | confounded |
| 10% trimmed mean | **3.68**, and it **flips sign** | trimming makes it *worse* |
| `regret_p90` | **2.83** | confounded |
| `regret_max` | **1.41** | confounded |
| log-mean (`log1p`) | **2.64** | log does *not* fix it |
| headroom-normalised (÷ median-plan regret) | **2.92** | *worse* than raw |
| `opt_recovered_frac` | **2.12**, and it **flips sign** in the top bin | confounded |
| median per-dataset ratio | 0.00 | **degenerate** — 61.5% of datasets are solved exactly by the additive rule, so the median is 1.0 in every bin and has no resolution |
| mean per-dataset regret ratio | **0.27** | usable |
| **win rate (per-dataset paired)** | **0.36** | **usable — and chance is exactly 0.5 at every size** |

**The binding problem was never outlier-robustness; it was aggregation order.** Averaging raw
regrets *across* datasets lets between-dataset scale heterogeneity — which tracks sweep size —
leak into the aggregate. Comparing the two models *within* each dataset first and only then
aggregating a bounded comparison removes most of it. That is a ~10× improvement in measured
drift (2.9 → 0.27), whereas the trimmed-vs-log choice makes things **worse** (2.58 → 2.64 →
3.68). The tentative fix recorded above was aimed at the wrong failure mode.

Decision:

- **Primary — `win_rate`** (fraction of held-out datasets where the model's plan beats the
  reference's, ties 0.5), with a bootstrap CI and an exact two-sided sign test. Its null is
  0.5 *by construction at every topology size*, which is the size-invariant reference the raw
  regret slope simply does not have.
- **Co-primary — `regret_ratio_mean`**, mean of per-dataset `(1+r_model)/(1+r_ref)`. `win_rate`
  discards magnitude; this restores it without crossing dataset scales.
- **Demoted to reported diagnostics — `regret_mean` / `p90` / `max`, `top1_acc`,
  `opt_recovered_frac`.** Kept for continuity with every earlier row in this file, but they
  no longer decide anything. `top1_acc` additionally has a **size-dependent chance level**:
  per the Phase 1 probe table, plans grow 32 → 9,828 across the ladder, so candidates/task
  grow ≈ 32^¼ = 2.4 → 9,828^¼ = 10.0 and random-guess top-1 falls 0.42 → 0.10. Its slope
  would be ~4× candidate-set growth before any model effect.

**Two structural additions, both mandatory.**

1. **A no-learning drift anchor at every held-out size.** Run the additive-fit argmin *and*
   the additive+one-integer rule on the same held-out datasets at each size and subtract their
   trend. This is the aggregation-level analogue of the Knative control already in the gate,
   it is free (no training), and it is not optional: on `shallow_v1` that constant-quality
   pair drifts 2.58 in `regret_mean` and reverses the sign of the `opt_recovered` gap between
   bins — **large enough to satisfy the original "the gap widens monotonically" PASS condition
   on its own.** Knative cannot serve this role; it is a real policy whose own quality may
   move with size, whereas the additive rule's expressive class provably does not.

2. **The gate is under-powered by 7–50× at the test-split sizes used so far**, and no choice
   of statistic fixes that. Paired bootstrap on the same corpus: SE of the mean-regret gap is
   **0.037 at n=30**, i.e. a minimum detectable gap of **0.149** — against GNN-vs-pointwise
   gaps actually observed on `shallow_v1` of **0.003–0.02**. *That is the arithmetic behind
   seed 44 reversing its own verdict between two identical commands*: it was never a seed
   pathology, the gate could not resolve its own effect. Pairing recovers little (per-dataset
   regret correlation between two rules is only **0.349**; paired SE 0.0737 vs unpaired
   0.0838 at n=30). Required held-out datasets **per size**, from `2·SE ≤ target`:

   | target MDG | datasets/size |
   |---|---|
   | 0.05 | ~65 |
   | 0.02 | ~400 |
   | 0.01 | ~1,600 |

   **This is a corpus-sizing decision, so it binds Phase 1's probe**: the probe must time
   generation at the ladder's top end (80 servers enumerated 9,828 plans in 39 s), because
   400/size × 2 held-out sizes is a materially different generation budget than the 1/size
   probe implies.

Wired into the tool, not just written down: `gnn_necessity_ablation.py` now prints the paired
block and an `!! UNDERPOWERED` line whenever the regret gap sits below the noise floor, and
records `paired_comparisons` in its frozen report (`schema_version` 3 → 4).

#### Recache status

| cache | graphs | status |
|---|---|---|
| `graphs_cache_shallow_v1_temporalfix` | 200 | ✅ **canonical going forward**; `graphs_cache_shallow_v1` kept only to reproduce the pre-fix result |
| `graphs_cache_contention_v2_873_v5.7_siv1_dim14_temporalfix` | 873 | ✅ rebuilt, **exactly matching the original 873**. Labels unchanged in 873/873; dims 9-11 changed on 60005/181584 rows (33.0%), max 0.0086. Needed an allowlist: the first attempt **failed loudly** with 26 datasets whose sweep-min labels are absent from scheduling-time candidate edges — precisely the 26 `contention_v2` entries in the `bad31` list, recovered from `run_full_corpus_siv1_recache.sh`. Manifest written to `logs/full_corpus_siv1_pipeline/oversample_manifest_contention_v2_only.json`. |
| `graphs_cache_contention_v2_873_v5.5` | 873 | ⛔ **not rebuilt.** `queue_feature_contract` / `platform_feature_dim` recorded as `None` (predates the contract system). Guessing them would swap a known, bounded bug for an unknown one. Any checkpoint on this cache carries the dims 9-11 bug **and** unverified contract settings. |
| `graphs_cache_contention_v3` | 900 | ⛔ same as above |
| `graphs_cache_full_corpus_siv1_dim14` | 2651 | ⚠️ **rebuildable after all.** The "missing" `oversample_manifest_exclude_bad31.json` is regenerable from `scripts_cosim/datalab/run_full_corpus_siv1_recache.sh`, which carries the 31 excluded ids inline plus the generator; regenerated locally (2663 weights vs the recorded 2651 — resolve at build time). Full rebuild is a datalab job (6 corpora, needs the `--rewrite-ssc` pass first). |

The `regime_b_*` caches are unaffected (their corpora are clean).

### network_contention_v1 — outcomes (2026-08-18)

**The physics works and is opt-in.** Each server node gets one shared ingress pipe;
propagation latency stays an un-serialized timeout (additive), while input transmission
(`stateSize / bandwidth`) is served through the pipe, so concurrent inbound transfers
queue. Unset ⇒ no pipe, no transmission term, `node_disk_v2` physics. The transfer formula
lives once in `scheduling_cost.py` and is imported by both the simulation and the ECT
mirror. 12 tests in `test_network_contention.py`.

Verified on a matched A/B (identical workload and topology, only the flag differs): mean
RTT delta by extra co-located tasks was **3.20 → 4.63 → 6.50 transfers** — monotone, i.e.
plans that co-locate pay real *waiting*. Contrast `node_contention_v3`, whose
`nodeContentionTime` was 0.0 everywhere.

**⛔ BLOCKER FIXED — the workload draw was unseeded.** `generate_workload_templates` drew
task source nodes from the *global* `random`, with no `random.seed()` anywhere in the
generator. Two runs of the same grid with the same seeds produced different workloads and
RTTs. **No existing corpus is reproducible from its recorded seed, and matched A/B arms
were impossible** — `node_contention_v3`'s "metrics disagreed and were underpowered" pilot
was built this way. Now a local `random.Random(--workload-seed)`, default 42. Infrastructure
was always deterministic; `placements.jsonl` row *order* still varies (parallel completion
order) — compare sweeps as sets, never with `diff`.

**Bandwidth alone does not create a joint decision.** M4 moves monotonically with dose but
M1 stays at exactly 0%:

| arm | additive R² | collision gain (plat/node) | additive-argmin regret mean/max | argmin optimal | M1 coupled |
|---|---|---|---|---|---|
| baseline | 0.9667 | 2.62 / 2.76 pp | 3.01% / 15.5% | 67% | **0%** |
| 1.5 MB/s | 0.9596 | 3.32 / 3.46 pp | 3.94% / 19.6% | 58% | **0%** |
| 0.5 MB/s | 0.9478 | 4.38 / 4.57 pp | 2.42% / 14.7% | 33% | **0%** |

**Why: spreading was free.** A no-simulation pre-check (min premium `θ*` admitting a
task→distinct-node matching, by Hall's condition) found `θ* = 0` in **92%** of datasets —
each task's single favourite node was already distinct (3.83 of 4 tasks).

**`FALSIFIED` — scarcity-by-count and topology funnelling.** Both *reduced* the overlap
they were meant to create, because shrinking candidate sets makes them more **disjoint**
when tasks are anchored to different clients. Mean pairwise candidate-node overlap
(of 4 tasks) and union:

| grid | cand/task | pairwise overlap | union nodes | θ*=0 |
|---|---|---|---|---|
| shallow_v1 | 4.56 | 0.93 | 13.58 | 92% |
| `netc_scarce_v1` (sparser links) | 3.23 | **0.36** | 11.00 | 92% |
| `netc_funnel_v1` (degree_skewed_core) | 2.18 | **0.14** | 8.00 | **100%** |

**`ACTIVE` — replica concentration is the lever.** The blocker was
`generate_infrastructure.py`'s `replica_server_pct = max(server_pct, 0.6)`, which spread
replicas over ≥60% of servers whatever the grid asked. Now overridable via
`preinit.replica_server_percentage` / `--replica-server-percentage`. Dense links + few
replica hosts gives overlap 2.29, union **2.56 nodes for 4 tasks** (pigeonhole), θ*=0 in
**0%** of datasets:

| replica_server_pct | additive R² | collision gain | additive-argmin regret mean/max | argmin optimal | sweep size | best RTT |
|---|---|---|---|---|---|---|
| 0.6 (floor) | 0.9478 | 4.4 pp | 2.42% / 14.7% | 33% | 456 | 0.76 s |
| 0.45 | 0.9210 | 5.6 pp | 1.40% / 9.6% | 62% | 130 | 1.44 s |
| 0.30 | 0.8280 | 12.2 pp | 2.13% / 13.9% | 69% | 92 | 3.44 s |
| **0.15 + 0.5 MB/s** | **0.5645** | **31.7 pp** | **21.51% / 93.0%** | **25%** | 38 | 10.8 s |

Ingress bandwidth roughly **doubles** the effect at 0.15 (collision gain 16.5 → 31.7 pp,
argmin optimal 44% → 25%), so the two levers compose.

**⚠ METHODOLOGICAL — M1 is the wrong gate, and the pre-registered gate would have rejected
the one configuration that works.** At `hotspot + 0.5 MB/s` the M1 coupled fraction is
**0%** while a pointwise fit picks a suboptimal plan **75% of the time at 21.5% mean
regret**. M1's "marginal greedy" scores each per-task option as `min RTT over all joint
plans with task t there` — it has **oracle access to the joint sweep**, so it is not a
pointwise model and trivially recovers the optimum once the candidate set is small.
**The statistic matched to `PointwiseEdgeMLP`'s expressive power is M4's additive-fit
argmin** (`additive_choice_regret_rel` / `additive_choice_is_optimal`), since the additive
fit is literally what that model class can express. Gate on that, not on
`--gate-coupled-fraction`.

**`FALSIFIED` — the hotspot (0.15) configuration is degenerate. Do not build on it.**
Two checks settled it, both on existing local data.

*Cliff, not a regime.* Filling in the gap between 0.30 and 0.15 (host-node counts in
brackets) shows the intermediate points sit in one flat band and 0.15 jumps
discontinuously:

| replica_server_pct | hosts | sweep | additive R² | coll gain | argmin regret mean/max | argmin optimal |
|---|---|---|---|---|---|---|
| 0.6 (floor) | 14 | 456 | 0.9478 | 4.4 pp | 2.42% / 14.7% | 33% |
| 0.45 | 9 | 130 | 0.9210 | 5.6 pp | 1.40% / 9.6% | 62% |
| 0.30 | 7 | 92 | 0.8280 | 12.2 pp | 2.13% / 13.9% | 69% |
| 0.25 | 7 | 76 | 0.9049 | 6.1 pp | 3.36% / 19.6% | 69% |
| 0.20 | 6 | 80 | 0.8883 | 7.8 pp | 4.26% / 33.5% | 62% |
| **0.15** | **2** | **38** | **0.5645** | **31.7 pp** | **21.51% / 93.0%** | **25%** |

*The interaction is one integer, again.* Adding a **single** per-plan node-occupancy-excess
column to the additive fit repairs **9/12 (75%)** of its wrong picks at 0.15 and cuts mean
regret **21.51% → 4.15%**. The failures concentrate on **2 distinct node identities — which
are the only two hosts that exist**. With 4 tasks over 2 nodes the "joint decision" is just
*how many tasks land on node 21 vs node 22*: a scalar a pointwise MLP learns from one extra
feature, not graph structure. This is the same trap already recorded above ("the whole
interaction is one integer"), reproduced at larger magnitude. By contrast that column
repairs only 33-40% of failures at 0.20/0.25, where failures spread over 4 node identities.

*My own sweep was confounded.* `netc_hotspot_v1` changed **two** things at once —
`replica_server_percentage` 0.6→0.15 **and `per_client` 1→0**, which deletes client-local
replicas entirely. The intermediate arms varied only the percentage, so they never fell
below 6 hosts. The cliff coincides with removing client-local replicas, not with crossing a
percentage. "0.15" is therefore not even a clean descriptor of the regime.

*No realism anchor.* A literature check found no reported "fraction of nodes hosting
replicas" metric in serverless-edge placement work, so 0.15 has no external justification —
it is where the arithmetic moved, which is exactly the reasoning `netc_scarce_v1` was
careful to avoid.

**Retroactive scope of the M1 finding: nil for automated gating, real for reasoning.**
`--gate-coupled-fraction` appears in no committed script, sbatch or log — **no corpus was
ever accepted or rejected by it**. But the recommendation to make it primary is recorded in
both `LINEAGES.md` and `memory/memory.md`, and shallow_v1's since-retracted "31.0% coupled"
was the stated justification for the whole shallow lever. Correct the recommendation:
**gate on M4's `additive_choice_regret_rel` / `additive_choice_is_optimal`**, and always
report how much of the regret a single collision-count column repairs — without that
control, a degenerate one-integer corpus looks like a GNN opportunity.

**Nothing has been trained; no ablation has been run; nothing submitted to datalab.**

### The throughline — four mechanisms, one collapse (2026-08-18)

**`shallow_longexec_v1` is UNBLOCKED and is the fourth confirmation.** The config gap its
note described is closed — `sample_loader.ensure_workload_params` synthesizes the missing
`workload_nofs-{cnn,rf}`, and the grid now generates cleanly (cnn 3.086 s on rpiCpu vs rf
0.004 s, a 730× exec-time contrast). Gated locally at n=16 per arm:

| arm | additive R² | argmin regret mean/max | argmin optimal | **one-integer repair** | +col optimal |
|---|---|---|---|---|---|
| `shallow_longexec_v1` | 0.9330 | 3.05% / 37.0% | 62% | **100%** (n=6) | 81% |
| ...+ 0.5 MB/s ingress | 0.9290 | 0.99% / 4.8% | 56% | **86%** (n=7) | 81% |

Every dataset where the pointwise fit picked a suboptimal plan was repaired by one scalar,
and optimal-recovery rises 62%→81% / 56%→81% from that single column. *(Honest wrinkle: the
augmented fit's mean regret is higher (5.24%/4.57%) because the extra column shifts the
argmin on some datasets that were already optimal — the degeneracy claim rests on the
repair of actual failures and the optimal-recovery jump, not on mean regret.)*

**So five structurally different attempts to inject coupling all end the same way:**

| # | mechanism | how it failed |
|---|---|---|
| 0 | `added_in_batch` (base physics) | +1.10 pp — "a column you hand an MLP" |
| 1 | execution-slot pool (`node_contention_v3`) | no interaction at all — `nodeContentionTime` ≡ 0.0 |
| 2 | deep queues (`contention_v4_v5`) | interaction diluted by depth; R² moved the wrong way |
| 3 | node-ingress bandwidth (`network_contention_v1`) | R² scales continuously, but at its only high-regret setting one integer repairs 75% |
| 4 | long-exec task types (`shallow_longexec_v1`) | one integer repairs 86-100% |
| 5 | per-link capacity (`link_contention_v1`) | **first mechanism to escape the one-integer control** — isolated on spread plans the node column repairs 0% and link scalars only 11-22% — but the effect is 0.08-0.10% regret against a 5% gate, and bandwidth is a null lever (wait/transfer 0.0100 → 0.0088 across a 3× cost change) |

**Whenever coupling in this simulator shows up with teeth, it is capturable by one
count-like feature.** That now looks like a property of this class of scheduling problem
rather than of any single experiment — and it is the reusable finding, together with the
diagnostic that detects it before any GPU-hours are spent.

**`link_contention_v1` sharpens that statement rather than breaking it.** It is the first
mechanism to produce coupling a scalar count cannot capture — the escape was real, and the
reason it worked is now understood: its contended object (a link) has more identities than
the destination-node count, so two tasks that share *no* node can still contend. But the
resulting effect has no teeth (0.08-0.10% regret), so the observed rule survives in its
stronger form:

> **In this simulator, coupling is either count-shaped or negligible.** Five mechanisms, two
> failure modes, and the second one is now demonstrated rather than assumed — the isolation
> control shows base physics is additive to R² = 1.00000 exactly once collisions are removed,
> so there is no reservoir of non-count coupling waiting to be uncovered by a better lever.

The practical corollary for anyone extending this: **check the additive/interaction scaling of
a proposed lever before building it.** Deep queues failed because the additive term grew with
the lever and the interaction term did not; link bandwidth failed because both grow together
and the ratio is invariant. A lever only helps if it moves interaction *faster* than additive
— that is a two-line calculation, and it would have predicted both outcomes.

⚠ **The scaling test rules levers out; it does not rank the survivors.** Tested against the
hub↔mesh sweep above, `wait / transfer` got the ordering backwards — the configuration with
the lowest ratio had the highest regret, because the ratio is measured on optimal plans, which
select *against* contention, while regret is a property of the spread across the whole sweep.
A bandwidth-invariant ratio is decisive evidence to stop; a favourable ratio is not evidence
to proceed.

**The corrected gate is wired into the tool, not just written down here.**
`separability_diagnostic.py` gained `--gate-additive-argmin-regret` (primary) which
**argparse-errors unless `--gate-one-integer-repair` is supplied**, plus a first-class
`one_integer_repair_frac` in the M4 block and a `!! DEGENERATE` banner at ≥0.5. Verified:
both failure modes fire and the tool exits 1. `--gate-coupled-fraction`'s help text now
carries the deprecation and the reason.

**2026-08-18 additions from `link_contention_v1`,** both covered by
`scripts_cosim/test_link_repair_control.py` (11 tests) so the gate is not itself untested —
the `--gate-coupled-fraction` episode was a gate nobody had exercised:

- **`--gate-link-repair`** — repair columns for the busiest link load (`k1`), the top-2 loads
  (`k2`), and total link-sharing excess (`excess`). Needed because the existing node-collision
  control is *structurally blind* to link contention and would have returned a false PASS on
  any per-link mechanism.
- **`--spread-plans-only`** — the isolation control described above. Reusable for any future
  mechanism: it answers "is there coupling here that is not the collision term?", which the
  headline gate cannot, because the collision term dominates every corpus in this repo.

### link_contention_v1 — outcomes (2026-08-18)

**The hypothesis.** Every mechanism above was repaired by a *node-occupancy excess* column
("how many tasks landed on host X"), because in every case the contended object was indexed
by the destination node. A network **link** is crossed by paths to many destinations, so two
tasks on different nodes can queue behind each other — a coupling no node-occupancy count can
express at any value. That is the one structural escape none of the previous four had.

**Two facts had to be fixed before the idea could even be tested.**

1. **There were no multi-hop paths.** `generate_network_topology_deterministic` only ever
   wrote client↔server pairs — 0 server↔server and 0 client↔client edges, density ~0.25,
   hop count always exactly 1. Latency was a single table lookup, so "the link" a task
   crossed was its own private access edge.
2. **The 4 tasks always come from 4 distinct clients** (`random.Random(42)`; all 10 workload
   templates), so capacity on a client↔server edge is paid by at most one task and is
   additive by construction. The contended object had to be a *shared core segment*.

**Physics.** `src/placement/network_fabric.py` holds one `simpy.Resource(capacity=1)` per
link plus the frozen route table, built once in `simulation.py` because a link belongs to no
node. Propagation stays the un-serialized `env.timeout` (additive, unchanged); the input
transmission walks the route store-and-forward, holding each hop. `build_core_backbone` is a
post-processing overlay applied *after* every connectivity repair, so `network_maps` keeps
its meaning (candidate filtering still works, all six copies of the lookup untouched) and
only its latency becomes a path sum. Absent a `network.backbone` block nothing is built and
replay is bit-identical. 23 tests in `test_link_contention.py`, 9 in
`test_link_repair_control.py`.

**P0 — the overlap pre-check `PASSED` decisively.** `scripts_cosim/link_overlap_precheck.py`
measures route overlap with **no simulation**, extending the Hall's-condition instinct that
killed `netc_scarce_v1`/`netc_funnel_v1` before they cost a corpus. Swept on `shallow_v1`:

| backbone | core links | pairs sharing a core link | ...of those, DIFFERENT destinations |
|---|---:|---:|---:|
| n_core 6, attach 2, chords 3 | 9 | 5.2% | 77.8% |
| n_core 6, attach 1, chords 0 | 6 | 25.2% | 90.2% |
| **n_core 12, attach 1, chords 0** | **12** | **30.3%** | **91.3%** |

Chords and a second attachment both let paths diverge and collapse the overlap. A pure ring
with single attachment was chosen on this measurement, **not** on RTT.

**The gate `FAILED` on both pre-registered criteria.** Matched arms, same tree, same
`--workload-seed 42`, grid `netc_multihop_v1` (shallow queues + `per_client=0` so every task
must cross the network), n=48 each, thresholds registered before generating:
`--gate-additive-argmin-regret 0.05 --gate-one-integer-repair 0.5 --gate-link-repair 0.5`.

| arm | additive R² | argmin regret mean / max | node repair | link repair k1 / k2 / excess |
|---|---:|---:|---:|---:|
| `mh_off` (no backbone) | 0.91658 | 3.57% / 22.9% | **0.817** | — |
| `mh_bw1p5` (1.5 MB/s per link) | 0.91910 | **5.00%** / 36.6% | **0.633** | **0.149 / 0.195 / 0.178** |

**What is genuinely new, and worth keeping.** The *link* controls do not repair: medians are
**0.000** for all three (k1, k2, excess), with the means pulled up by a handful of datasets at
1.0. This is the first mechanism whose coupling is not a scalar summary of the contended
resource — contrast node-ingress, where one integer repaired 75%.

**Why it still fails.** The *node-collision* column repairs 63% (median **1.000**). The
coupling that dominates this corpus is still the pre-existing `added_in_batch` collision
effect, not the link effect; the backbone adds only a small, mixed increment on top of it.
Paired across the 48 matched datasets the backbone **raised** regret in 17, **lowered** it in
13, and left 18 unchanged — mean +1.43 pp, and the resulting 0.049961 lands a hair under the
0.05 threshold. A weak, two-directional effect riding on a dominant confound is not a corpus
worth training on.

**Do not** read this as "links don't couple" — the pre-check and the sweep both show real
cross-destination link contention (877 core-link-sharing task pairs on a single dataset, 92%
of them with different destinations). Read it as: **the link term is small next to the node
term already present**, so the corpus's coupling stays node-shaped and one integer still
repairs most of it.

#### The isolation control — and the actual size of the link effect

The mixed n=48 result could not separate the link term from the collision term that
dominates it, so `separability_diagnostic.py` gained **`--spread-plans-only`**: restrict every
metric to plans placing each task on a **distinct node**. Node-occupancy excess is then
identically zero across the retained plans — the column that collapsed the four previous
mechanisms becomes a constant and can explain nothing — and `added_in_batch` is zero too,
since distinct nodes imply distinct platforms. Link contention survives untouched, because it
acts *between* tasks on different destinations. (Guarded by two tests asserting exactly those
two halves of the contract; it is an isolation control, **not** a corpus gate, since it
discards most of the sweep.)

Same 48 matched datasets, restricted to spread plans (mean 187 of ~600 plans retained):

| arm | additive R² | argmin regret mean / max | additive argmin optimal | node repair | link repair k1 / k2 / excess |
|---|---:|---:|---:|---:|---:|
| `mh_off` | **1.00000** | **0.00% / 0.0%** | **100%** | — | — |
| `mh_bw1p5` | 0.99686 | 0.08% / 1.7% | 90% (n=5) | **0%** | 20% / 20% / 20% |
| `mh_bw0p5` | 0.99351 | 0.10% / 1.7% | 81% (n=9) | **0%** | 11% / 22% / 22% |

**Two findings, and they point opposite ways.**

1. **The structural claim is VERIFIED.** Without a backbone the target is additive to
   R² = **1.00000** with **0.00%** regret in **100%** of datasets — once collisions are
   removed, the base physics has *literally no* remaining coupling. With the backbone, regret
   becomes non-zero while the node column repairs **0%** by construction and the link scalars
   repair only 11-22%. This is the **first mechanism in the series to produce coupling that is
   neither collision-shaped nor node-count-shaped**, which is exactly what it was built to do.
2. **The magnitude is two orders of magnitude too small.** 0.08-0.10% mean regret against a
   5% gate, max pinned at **1.7%**, touching 10-19% of datasets.

**Bandwidth is a NULL LEVER — this is the deep-queue arithmetic one level down.** Tripling the
link cost (1.5 → 0.5 MB/s) widened the effect (5 → 9 datasets) but barely deepened it
(0.08% → 0.10%, max unchanged). Measured over the optimal plans of all 48 datasets per arm:

| arm | additive transfer | contention wait | **wait / transfer** |
|---|---:|---:|---:|
| `mh_bw1p5` | 73.73 s | 0.740 s | **0.0100** |
| `mh_bw0p5` | 210.35 s | 1.861 s | **0.0088** |

Store-and-forward charges each hop a full transmission, so the **additive** term
(hops × transfer) and the **interaction** term (crossings × transfer) both scale as
1/bandwidth. The ratio is therefore invariant — capacity changes the absolute size of the
network term but never its additive/interaction split, and contention stays ~1% of the link
cost at any bandwidth. Compare the deep-queue failure, where additive `depth × exec_time` grew
with the lever while interaction `added_in_batch × exec_time` did not.

⇒ **Do not tune bandwidth to rescue this lineage.** The only lever that could move the ratio is
the number of *crossings per segment* — topology and attachment, not capacity.

#### The closing sweep: the whole hub↔mesh spectrum, and a prediction that was wrong

Since the ratio is `crossings / hops`, the topology lever predicts that **fewer** cores should
raise coupling (shorter routes, more traffic per segment). That was expected to run into a
tension: concentrating traffic is exactly what makes "load on the busiest link" a sufficient
scalar, so any gain in magnitude should be paid for in degeneracy. Tested at n_core ∈ {2, 4,
12}, all `attach_degree=1`, pure ring, 1.5 MB/s, 48 datasets each, spread-plans-only:

| n_core | hops/route | wait / transfer | additive R² | regret mean / max | link repair k1 |
|---:|---:|---:|---:|---:|---:|
| 2 (hub) | 2.26 | **0.0189** | 0.99126 | 0.04% / 1.4% | **33%** |
| 4 | 2.60 | 0.0072 | 0.98753 | **0.35% / 7.2%** | 25% |
| 12 (mesh) | 3.93 | 0.0100 | 0.99686 | 0.08% / 1.7% | 20% |

**The tension is real in direction but never binds.** `link_repair_k1` rises monotonically as
the core shrinks (20% → 25% → 33%), confirming that concentrating traffic makes the coupling
more scalar-summarisable. But even at the extreme hub it only reaches 33%, comfortably below
the 0.5 threshold. Degeneracy was never what stopped this mechanism.

**The magnitude prediction was wrong, and the correction matters.** Coupling is *not* monotone
in hub-ness — it peaks in the interior at n_core=4 (0.35% mean, max **7.2%**, the only
configuration whose max clears the 5% gate) and falls off at *both* ends. n_core=2 collapses
because with a single shared segment and one attachment each, roughly half of all
(client, server) pairs hang off the same core and their routes contain **no core link at all**
— the hub wins crossings per link but loses coverage (P0: core-link pair sharing 0.214 at
n_core=2 vs 0.306 at n_core=12).

Nor does `wait / transfer` predict regret: n_core=4 has the *lowest* ratio (0.0072) and the
*highest* regret. The ratio is measured on optimal plans, which select against contention, so
it describes the optimum's composition rather than the spread across the sweep that regret
actually measures. **Use it to rule a lever out (a bandwidth-invariant ratio is decisive), not
to rank the ones that survive.**

**What closes the lineage is neither degeneracy nor a bad configuration — it is uniform
smallness.** Across a 3× bandwidth range and the full hub↔mesh spectrum, mean additive-argmin
regret never exceeds **0.35%** against a 5% gate, and the contention term stays **0.7-1.9%** of
the link cost in every configuration tested. There is no setting in this family within reach of
the gate, so no corpus, cache, training run, or live gate was produced.

**Superseded pilot (same day, recorded so it is not re-run).** A first matched pilot used the
stock `shallow_v1` grid (n=16 × 3 arms: off / 5.0 / 1.5 MB/s) and failed on headroom alone —
regret 2.51% / 1.07% / 1.09%, i.e. the backbone made the corpus *more* separable. Cause:
`shallow_v1` keeps `per_client >= 1`, so many tasks run locally and never touch the network,
and a cost that prices only remoteness pushes the optimum toward the local corner the additive
fit already picks — the same shape as `netc_scarce_v1`. It was also underpowered (2-5
datasets with any regret). `netc_multihop_v1` sets `per_client = 0` and changes nothing else;
server spread deliberately stays at the 0.6 floor, unlike `netc_hotspot_v1`, which moved
`replica_server_percentage` and `per_client` together and whose "cliff" turned out to be one
node-occupancy integer over the only 2 hosts that existed. Frozen reports:
`simulation_data/separability_netc_multihop_{pilot,v1}.json`,
`simulation_data/link_overlap_precheck_netc_multihop_v1.json`.

### link_contention_v1 — a real-trace A/B, at realistic concurrency (2026-08-21, 🔄 IN PROGRESS)

**Why.** The FALSIFIED verdict above ("uniform smallness … regret never exceeds 0.35%") was
measured entirely on 4-task co-sim sweeps, where at most 4 transfers can ever share a core
segment. That is a claim about the mechanism's magnitude *at that concurrency*, not a claim
about its magnitude at realistic load — a real trace at rps=150 presents five to six orders of
magnitude more simultaneous traffic. This does **not** re-decide the co-sim separability
claim (`total_rtt` has no additive-argmin regret to report) — it is new evidence on a
different, previously-untested question: does the backbone change live outcomes at real
concurrency?

**Design.** Matched A/B: identical parity-verified cells, identical trace
(`workload-150-100.json`), identical deployed checkpoint; the only difference is a
`network.backbone` block (`n_core=4, attach_degree=1, chord_count=0, bandwidth_mbps=1.5`) —
`n_core=4` because it is this lineage's own measured interior peak, the only configuration
whose max regret cleared the 5% gate (see the closing hub↔mesh sweep above). The no-backbone
arm is `siv1_full_corpus`'s `workload-150-100` retest (above) on the same cells — reused
directly, not duplicated.

**Blocker found and resolved: `build_core_backbone`'s jitter rng is offset by the
replica-reachability repair.** `generate_infrastructure.py`'s backbone build draws
`rng.sample`/`rng.uniform(-jitter, +jitter)` (`:372-375`) from the *same* rng stream the
reachability repair already consumed via `rng.shuffle` (`:768`), and the backbone is overlaid
*after* the repair (`:780`). A live run autoscales from zero and performs no repair, so it
reaches the backbone build at a different stream position — every access-link latency
diverges on exactly the cells with a non-empty repair set:

| cell | repair edges (corpus) | backbone parity |
|---|---|---|
| cell02 p=0.35 | 0/282 | PASS |
| cell04 p=0.50 | 0/380 | PASS |
| cell05 p=0.20 | 12/172 | FAIL — 12 corpus-only edges |
| cell01 p=0.25 | 14/182 | FAIL — 14 corpus-only edges |
| cell03 p=0.15 | 34/174 | FAIL — 34 corpus-only edges |

Resolved with a narrowly-scoped, control-tested addition to `verify_live_infra_parity.py`:
`--allow-backbone-latency-divergence` downgrades exactly the two finding classes this causes
to notes, and **only** when a backbone is present on both the corpus and live sides (verified:
relaxes all 5 backbone cells to PASS; the same cells still FAIL 3/5 without the flag; a
non-backbone collection is unaffected by the flag either way). This is a live-vs-live-only
relaxation for this matched A/B — the corpus-side artifact exists only to satisfy the
preflight, and both live arms are self-consistent with each other. It does not paper over a
real mismatch on any collection that isn't deliberately using a backbone this way. Also see
GATE TOOLS below — `NetworkFabric.link_wait_total` / `task.link_wait_time` already measure
exactly the contention quantity this lineage needs and were never surfaced in the live result
JSON.

**Smoke result — the effect is not small at real concurrency, and decomposes cleanly.**
Knative, cell02 (p=0.35), `workload-150-100-30k.json` (30,000 events):

| arm | total_rtt | |
|---|---:|---|
| no backbone | 2,043,279.3 | |
| backbone @ 1000 MB/s (non-binding) | 1,451,938.7 | routing/path-sum effect only |
| backbone @ 1.5 MB/s (binding) | 7,867,634.7 | + transmission + contention |

Routing alone **improves** RTT by 28.9% (shorter/better-latency paths over the core vs. direct
one-hop); the binding bandwidth then adds +441.9%; net **+285.1%**, ~entirely bandwidth-driven.
This measures `total_rtt`, an absolute-cost effect — it does not by itself say whether the
backbone changes the *decision* (policy ordering), which is the full-trace A/B's actual
question and was still running when this entry was written.

**Full-scale result (2026-08-21): at real concurrency the backbone dominates absolute cost
AND changes the policy ordering in the GNN's favor.** All 15 runs complete
(`a1_backbone_bw1p5`, backbone `n_core=4, bw=1.5 MB/s`, vs the no-backbone arm
`a4_wl150100`, same cells/trace/checkpoints; local working tree, i.e. the fixed dims 9-11
live path — see the siv1 resolution subsection):

| cell | knative | mlp (vs kn) | gnn (vs kn) | kn backbone/no-backbone |
|---|---:|---:|---:|---:|
| cell01 (p=0.25) | 282,087,829.7 | 201,011,470.6 (−28.7%) | 192,457,679.0 (**−31.8%**) | 11.5× |
| cell02 (p=0.35) | 224,756,932.5 | 133,554,067.3 (−40.6%) | 169,969,881.1 (**−24.4%**) | 9.2× |
| cell03 (p=0.15) | 225,705,548.2 | 168,869,109.0 (−25.2%) | 165,408,019.3 (**−26.7%**) | 8.3× |
| cell04 (p=0.50) | 286,046,987.2 | 169,992,579.0 (−40.6%) | 260,550,047.3 (**−8.9%**) | 14.0× |
| cell05 (p=0.20) | 196,851,644.1 | 487,049,588.6 (+147.4%) | 141,739,988.9 (**−28.0%**) | 7.4× |

Three findings:
1. **Binding bandwidth is a 7–14× absolute-cost effect at real concurrency** — the co-sim
   FALSIFIED verdict's "regret never exceeds 0.35%" was a statement about 4-task sweeps,
   and does not describe live load. (This still does not reopen the co-sim separability
   claim, which is about a different statistic on different data.)
2. **The GNN's advantage over Knative widens under binding bandwidth**: mean margin −9.4%
   (0.9–14.8%) without the backbone → **−24.0% (8.9–31.8%) with it**, 5/5 both arms. This
   is the first live regime where the GNN's edge grows as network structure starts to
   bind — directionally what this lineage's physics was built to create, though from a
   checkpoint never trained on backbone corpora.
3. **MLP beats the GNN on 3/5 cells but keeps its catastrophic cell05 tail** (+147% here,
   +366% on `workload-175-100`, 6.1× on the no-backbone arm) — its wins don't survive its
   worst cell, and the GNN has no such tail on any of the six live sweeps run to date.

Margins are 25–300× the measured 0.1–0.3% local noise floor. Caveat: single trace, single
seed per cell, one backbone configuration; the rng-stream coupling in
`generate_infrastructure.py` (above) is still unfixed, so backbone cells still need
`--allow-backbone-latency-divergence` for parity. The rng bug and the parity-tool fix
stand regardless of this result.

#### ✅ The backbone win is REGIME-LEVEL, not draw luck — and the serving layout was confounding everything (2026-08-23)

**Why this ran.** The result above rests on one trace, one backbone config and **one training
draw**, and the 2026-08-22 variance control said a single-checkpoint verdict is a claim about
a draw. So: take the two draws that **lost** their no-backbone gates — `prefixctl` (the
variance control, 1/5) and `tempfix` (the corrected cache, 0/15 across three traces) — and run
them on the same parity-verified cells, same trace, under both conditions. 2×2×5, plus
`knative` and the deployed checkpoint re-run **in the same batch** so no arm's baseline comes
from a different venue or an unstamped working tree (jobs 710315 / 710335 / 710341).

**A confound had to be removed first, and it is the larger finding.** `prefixctl` and
`tempfix` declare `inference_feature_layout: null` in their sidecars, and
`load_gnn_model` defaulted an undeclared layout to **`atomic21`**, while the deployed
checkpoint's sidecar declares **`dim22`**. `task_dim=3 / platform_dim=14` is structurally
valid under both — they give the same platform columns different meanings (`dim22`
normalizes the queue features) — so nothing raised. **Every deployed-checkpoint gate served
`dim22`; both alternate-draw gates served `atomic21`.** Re-serving the same checkpoints on
the same cells under `dim22`:

| checkpoint | per-cell delta | mean | worst vs 0.1–0.4% noise floor |
|---|---|---:|---:|
| `prefixctl` | −1.2% … −14.8% | **−7.4%** | 37× |
| `tempfix` | −13.3% … −40.8% | **−29.8%** | **102×** |

`dim22` is uniformly better. This is **GNN-specific**: `mlp_scheduler` reads the layout from
its own checkpoint (and infers `dim22` from `input_dim=22` otherwise), so the MLP arms and
every MLP-vs-MLP comparison are unaffected.

**The gate, all arms at `layout=dim22`, `workload-150-100`, vs Knative per cell:**

| cell | deployed | prefixctl | tempfix |
|---|---:|---:|---:|
| *no backbone* | | | |
| cell01 (p=0.25) | −1.2% | +3.5% | −5.2% |
| cell02 (p=0.35) | −13.0% | −2.2% | −20.2% |
| cell03 (p=0.15) | −14.7% | −8.5% | −16.8% |
| cell04 (p=0.50) | −5.4% | +24.5% | −14.2% |
| cell05 (p=0.20) | −13.0% | −6.0% | −13.4% |
| **mean / wins** | **−9.4% · 5/5** | **+2.3% · 3/5** | **−14.0% · 5/5** |
| *backbone `n_core=4, bw=1.5`* | | | |
| cell01 | −31.8% | −24.4% | −32.9% |
| cell02 | −24.4% | −10.1% | −38.6% |
| cell03 | −26.7% | −24.2% | −30.0% |
| cell04 | −8.9% | +35.4% | −37.4% |
| cell05 | −28.2% | −19.8% | −31.8% |
| **mean / wins** | **−24.0% · 5/5** | **−8.6% · 4/5** | **−34.1% · 5/5** |

Four conclusions:

1. **`REGIME-LEVEL` — the backbone win survives draw variation.** Every draw improves
   markedly under binding bandwidth (−9.4→−24.0, +2.3→−8.6, −14.0→−34.1) and every one wins
   ≥4/5. The claim "under binding network contention the GNN beats Knative" no longer rests
   on the lucky checkpoint. It remains one trace and one backbone config.
2. **The `tempfix` 0/15 FAIL is FALSIFIED — it was the serving layout, not the cache.** At a
   fixed layout the corrected-cache checkpoint is the **best artifact on disk**: 5/5 in both
   conditions, beating the deployed draw on mean margin in both (−14.0% vs −9.4%, −34.1% vs
   −24.0%). The 2026-08-21 reading ("the mismatch was an accidental regularizer") is dead;
   the corrected cache is *better*, and the earlier gate could not see it.
3. **The lottery is real but much smaller than recorded.** `prefixctl` is genuinely the weak
   draw — cell04 is +24.5% / +35.4%, a draw-specific failure that persists at a fixed layout
   — but the 2026-08-22 table's 5–36% spread was draw **plus** layout. Ordering at a fixed
   layout: `tempfix` ≳ `deployed` > `prefixctl`.
4. **The venue/code axis is confirmed inert, again.** The deployed arm, re-run on datalab at
   a clean committed tree, reproduces its 2026-08-21 local numbers to
   −31.8/−24.4/−26.7/−8.9/−28.2 against the recorded −31.8/−24.4/−26.7/−8.9/−28.0.

**The second trace agrees (job 710366, `workload-175-100`, same three arms, same cells,
`layout=dim22`, 30/30 clean).** This was run precisely because the paragraph above demanded
it before any promotion:

| | deployed | tempfix |
|---|---:|---:|
| no backbone, mean / wins | −9.2% · 5/5 | **−15.6% · 5/5** |
| backbone, mean / wins | −23.9% · 5/5 | **−33.9% · 5/5** |

Against `workload-150-100`'s −9.4% / −24.0% and −14.0% / −34.1%, the two traces reproduce
each other to within ~1.6pp on every one of the four cells of that table. `tempfix` beats the
deployed checkpoint on **both traces in both conditions**, 20/20 cells beat Knative, and it
loses to deployed on only 2 of 20 individual cells (175-100 backbone cell03 and cell05, by
2.7 and 0.9pp). **`tempfix` is the promotion candidate**; what remains before swapping it in
is a re-gate on `workload-125-225` (the trace where the deployed checkpoint is weakest, 2W/1T/2L)
and `workload-200-200`.

**And the win is not specific to the tuned backbone config (job 710398, 30/30 clean).**
`n_core=4 / bw=1.5` was this lineage's own measured interior peak — chosen because the effect
was largest there — so a win visible only at that point would not be a claim about network
contention. Two further configurations, `workload-150-100`, same cells:

| backbone config | knative baseline | deployed | tempfix |
|---|---|---:|---:|
| `n_core=8, bw=1.5` (different core topology) | — | −22.4% · 5/5 | **−30.8% · 5/5** |
| `n_core=4, bw=0.5` (more binding bandwidth) | — | −24.4% · 5/5 | **−34.4% · 5/5** |
| `n_core=4, bw=1.5` (the original) | — | −24.0% · 5/5 | **−34.1% · 5/5** |

Mean margins move by ≤2pp across a doubled core tier and a 3× tighter bandwidth. **30/30
cells beat Knative across the three configurations.**

**These are also the first backbone gate cells in the repo that are parity-exact by
construction rather than by relaxation** — minted by `important/make_backbone_gate_cells.py`
with `rng_stream: independent_v1`, they pass `verify_live_infra_parity` with **no
`--allow-backbone-latency-divergence`**, including the three cells whose non-empty repair
sets forced the waiver on the legacy `a1` cells. The gate exports `PARITY_EXTRA_ARGS=""`
deliberately, so a regression in the rng fix would fail the job rather than be waived.

**Standing evidence for the GNN win, as of 2026-08-23:** 3 training draws × (backbone,
no-backbone); 2 traces × (backbone, no-backbone); 3 backbone configurations — every GNN arm
beats Knative on ≥4/5 cells under binding bandwidth, and every 5/5 except the weak
`prefixctl` draw. The remaining scope limits are honest and named: one topology family
(20c/20s sparse), and `workload-125-225` / `workload-200-200` not yet run under the
corrected layout. **The MLP baseline was added to all three gates on 2026-08-23 and changes
how this should be read — see the subsection immediately below.**

#### ⚠ The MLP baseline says the GNN's edge is RELIABILITY, not mean latency (2026-08-23)

Every gate above compared GNN draws against Knative only, which cannot distinguish "the graph
model wins here" from "any learned model wins here". The pointwise MLP
(`batch_edge_mlp_full_corpus_siv1_dim22_batchcache.pt`) was run as a fourth arm on all 30
cells of all three gates — same cells, same traces, same `dim22` layout, same
`node_disk_v2` physics (`datalab/mlp_arm_all_gates.sbatch`, jobs 710450/710451). The GNN and
Knative numbers did not move: the re-score is a pure addition to the three verdict JSONs.

Mean margin vs Knative · wins (a win is `< −0.4%`, the noise floor):

| gate / condition | deployed | tempfix | **mlp** |
|---|---|---|---|
| drawgate, no backbone | −9.4% · 5/5 | −14.0% · 5/5 | **+85.1% · 4/5** |
| drawgate, backbone | −24.0% · 5/5 | −34.1% · 5/5 | **+2.5% · 4/5** |
| promo175, no backbone | −9.2% · 5/5 | −15.6% · 5/5 | **+53.4% · 4/5** |
| promo175, backbone | −23.9% · 5/5 | −33.9% · 5/5 | **−35.1% · 5/5** |
| bbrob `n_core=8, bw=1.5` | −22.4% · 5/5 | −30.8% · 5/5 | **+38.5% · 3/5** |
| bbrob `n_core=4, bw=0.5` | −24.4% · 5/5 | −34.4% · 5/5 | **+11.3% · 3/5** |

**The MLP's mean margin is positive — worse than Knative — in 5 of 6 conditions, while every
GNN arm is negative in all 6.** But the mechanism is entirely a tail, and the per-cell record
is uncomfortable:

* **`tempfix` beats the MLP on only 17 of 30 cells; `deployed` on 13 of 30.** On the 23 cells
  where the MLP does not collapse it usually beats *both* GNN arms, often by 10pp or more
  (`promo175`/nobackbone/cell04: MLP −26.1% vs `tempfix` −20.7% vs `deployed` −10.7%).
* **7 of 30 cells collapse catastrophically** — `cell05` in five conditions (+509.8%, +365.5%,
  +195.4%, +147.4%, +119.1%) and `cell03` under both bbrob configs (+79.0%, +31.4%). One such
  cell is enough to swing a 5-cell mean by 100pp.
* The collapse is the `averageOccupation → ~1` packing failure recorded in
  `memory/herosim-mlp-collapse-is-occupation-collapse.md`, reproduced here on cells the MLP
  has never been gated on. It is a known failure mode, not a new one.

**So the defensible claim is narrower than "the GNN beats the pointwise baseline".** It is:
*the GNN is the only arm that beats Knative on every cell of every condition tested; the MLP
achieves a better typical cell and loses the regime on a fifth of them.* For a scheduler that
is a real advantage and it is exactly the advantage a graph-aware model should have — but a
paper claim of the form "GNN > MLP on latency" is **not** supported by these 30 cells and
should not be written. Two open questions this raises, neither answered here: whether the
collapse cells share a structural property the GNN exploits and the MLP cannot see, and
whether an MLP trained on the corrected cache (`..._tempfix`, on datalab since 2026-08-22,
deliberately **not** run as a second arm here) collapses on the same cells.

**Both of those questions are answered in the next subsection.**

#### ✅ The MLP collapse is ARCHITECTURAL — retraining relocates it without reducing it (2026-08-23)

The corrected-cache MLP (`..._batchcache_tempfix.pt`) was run as a fifth arm on the same 30
cells (`datalab/mlp_tempfix_arm_all_gates.sbatch`, jobs 710656/710657, commit `98b41e9`,
all 30 verified `dim22` / non-zero `total_rtt` / clean tree). Only the checkpoint and the
sweep dir differ from the `mlp` arm — cells, traces and parity waivers are byte-identical, so
this is an A/B on training data alone.

| gate / condition | mlp | **mlptempfix** |
|---|---|---|
| drawgate, no backbone | +85.1% · 4/5 | **+133.4% · 3/5** |
| drawgate, backbone | +2.5% · 4/5 | **+12.8% · 4/5** |
| promo175, no backbone | +53.4% · 4/5 | **+98.2% · 4/5** |
| promo175, backbone | −35.1% · 5/5 | **+4.3% · 4/5** |
| bbrob `n_core=8, bw=1.5` | +38.5% · 3/5 | **+28.6% · 4/5** |
| bbrob `n_core=4, bw=0.5` | +11.3% · 3/5 | **+10.5% · 4/5** |

**Exactly 7 of 30 cells collapse under each checkpoint — the same count, a different set.**
The corrected cache *fixed* `cell03` under both bbrob configs (+79.0% → −22.6%, +31.4% →
−21.6%) and *broke* two cells that were healthy before: `cell03` on drawgate/nobackbone
(−24.5% → **+187.6%**) and `cell05` on promo175/backbone (−35.0% → **+127.9%**), the one
condition where the first MLP's `cell05` had survived. Five `cell05` collapses are shared.
Mean margin is *worse* under the corrected cache in 4 of 6 conditions.

A training-data artifact would have been reduced by fixing the training data. An invariant
7/30 with a reshuffled victim list is the signature of an architectural failure whose victim
set is a function of the *weights*, not of the data or the graph. **The reliability claim
therefore hardens: across 120 scheduler runs (2 MLP × 2–3 GNN arms × 30 cells), all 14
collapse events are MLP arms and none is a GNN arm.**

**A detector that separates perfectly on all 120 runs.** `chosen_queue_vs_min` **p95** from
the `.decode_stats.json` sidecar: collapse 13,485–23,866, healthy 449–1,387 — a 9.7x gap with
no overlap. The *median* is normal in both (46–131 collapse vs 43–312 healthy), which is the
direct confirmation that this is a minority-of-decisions tail that compounds. The occupation
ratio to the same cell's Knative arm also holds on all 120 (collapse ≤0.33x, healthy ≥0.41x)
but with only a 1.24x gap, so prefer the p95.

#### ✅ The collapse cells share no STRUCTURE — it is a dispersal failure with two mechanisms (2026-08-23)

Analysis of artifacts already on disk (`extract_gate_stats_summary.py`,
`extract_platform_dispersal.py`; no new sims). The answer to "do the collapse cells share a
structure the GNN sees and the MLP cannot?" is **no**, and four independent checks kill it:

1. **Adjacency is byte-identical across all four cell sets** (`nobackbone`, `a1_backbone`,
   `bb_core8`, `bb_core4` differ only in link latencies, queues and fabric). Degree, choice-set
   size and nearest-replica-host concentration (HHI) do **not** separate collapse from healthy
   — `cell03` is *more* constrained than `cell05` (14 vs 11 clients with ≤2 reachable hosts for
   `dnn2`) yet collapses less often.
2. **It is not an initial-queue "bait".** The platforms that hog the load rank 26/54, 31/54 and
   51/54 by initial queue depth; platform 134 (rank 51/54, one of the *longest* initial queues)
   is the top hog in a collapse run and also the top platform in a healthy one.
3. **The trace that flips `cell05` is a different draw of the same distribution.**
   `workload-150-100` and `workload-175-100` are both 50/50 `dnn1`/`dnn2`, uniform over 20
   sources (4.8–5.2% each), same 100 s duration; they differ in rate and random draw only. On
   identical infrastructure and checkpoint: 6 platforms busy >1% (collapse) vs 83 (healthy).
4. **The victim set moves when only the weights move** (previous subsection). A structural
   property would collapse the same cells under both checkpoints; 4 of the 9 distinct
   (cell, condition) collapse events are checkpoint-specific.

What *does* separate them is dispersal — how widely the scheduler spread load — and there are
**two distinct failure mechanisms** hiding under one `averageOccupation` symptom:

* **Platform-side packing** (12 of 14 events): top-3 platforms hold 43–65% of all busy time,
  2–33 platforms busy >1%. `cell05`/nobackbone MLP puts 63% of busy time on 3 platforms while
  the GNN spreads over 109.
* **Link-side starvation** (2 of 14: `cell03` under both bbrob configs): dispersal looks
  *normal* (top-3 share 16–17%, in the healthy range) and every platform is nearly idle
  (max utilisation **5.6%** and **2.2%**) while RTT is +79.0% / +31.4%. The fabric is empty and
  the tasks still wait.

**Why the link-side one is invisible in the existing metrics, and why `link_wait_total` is the
right fix.** `averageCommunicationsTime` is pinned at ~16.7 ms across all 150 runs (range
0.016662–0.016668) even where `total_rtt` swings 10x between backbone and no-backbone — it
does not measure link queueing at all. The wait is taken *inside* the replica's serving loop
(`infrastructure.py:1082`, `with self.node.fabric.pipe(...).request()`), so it blocks the
replica and surfaces as **queue time**: `averageQueueTime / averageElapsedTime` is 0.9990–1.0000
in every one of the 150 runs. Serialising `link_wait_total` / `linkWaitTime` (gate-tools row,
2026-08-21) would separate these two mechanisms directly instead of by inference from
`max_busy_pct`; it remains a reporting-only change.

**Consequence for the research goal.** The GNN's advantage here is not that it reads a
topological property the MLP is blind to — no such property distinguishes these cells. It is
that it *disperses*, and dispersal is what keeps a metastable queueing instability from
igniting. That is still a graph-aware advantage (a pointwise scorer cannot condition on where
its peers are going), but it should be stated as a dispersal/reliability argument, not as
"the GNN exploits topology `P`".

### mp_parity — outcomes (2026-08-17)

**Root cause.** `train_near_rtt.py` fitted `self.gin(x, data.edge_index)` (bipartite only)
while the serving copy in `src/policy/gnn/gnn_model.py` concatenated every same-node
platform↔platform edge — ~26:1 more edges than bipartite on the full-corpus cache. The
served model ran message passing on a graph its weights had never seen. Fixed by making
same-node edges opt-in, and structurally by deleting the second copy of the model: the
trainer now imports the one definition.

**Baseline gate** (`normal_sim_sweeps/gnn_mp_parity_gate_20260816`, deployed checkpoint
with parity fix, 3 configs × 5 seeds):

| config | GNN/Kn | MLP/Kn | GNN cell wins | p99 winner |
|---|---|---|---|---|
| sparse_p25 | 1.14x | 0.83x | 0/5 | mlp |
| sparse_p25_skew | 0.84x | 2.27x | 3/5 | **gnn** (71.0s vs MLP 498.5s) |
| sparse_p35 | 1.02x | 0.77x | 0/5 | mlp |

Pre-registered PRIMARY (GNN > MLP on total_rtt in ≥2 of 3 configs) = 1/3 **FAILED**.
TAIL (same on p99) = 1/3 **FAILED**. The parity fix removes the 12.4x catastrophe but the
fixed baseline still loses to MLP on the two large-RTT configs. It does reproduce the
pre-registered *collision cliff* on `sparse_p25_skew`, where the MLP is catastrophically
unreliable (2.27x Knative) and the GNN is not — a bounded claim, not a general win.

**`FALSIFIED` — same-node edges.** Arm B (`full_corpus_siv1_gnn_mp_residual_node_edges`)
trained *with* candidate-restricted same-node edges (0.37x bipartite, present on 80% of
graphs, recorded in the checkpoint sidecar) and was worse than Arm A on every metric:
val acc 62.6% vs 65.6%, test greedy regret 0.4944s vs 0.2621s. Co-location coupling is
not the signal the GNN was missing. Do not re-try this without new evidence.

**`FALSIFIED` — the GIN residual, with one instructive exception.** Arm A
(`full_corpus_siv1_gnn_mp_residual`) more than halves offline greedy regret vs the
deployed baseline (**0.5682s → 0.2621s, −54%**; top-5 0.0346 → 0.0239) and learns
`mp_gate` = 1.08, i.e. it leans on message passing *more* once MP augments rather than
replaces the per-node encoding. **None of that transferred.** Live re-gate
(`normal_sim_sweeps/mp_residual_gate_20260817`, 15/15, `compare.json` + `manifest.json`):

| config | baseline GNN/Kn | Arm A GNN/Kn | delta | MLP/Kn |
|---|---|---|---|---|
| sparse_p25 | 1.14x | 1.18x | +3.5% | 0.83x |
| sparse_p35 | 1.02x | 1.12x | +9.9% | 0.77x |
| sparse_p25_skew | 0.84x | **0.80x** | **−4.3%** | 2.27x |

PRIMARY **1/3**, TAIL **1/3** — both still FAILED, paired wins identical to baseline
(GNN 3/15 · MLP 10/15 · Kn 2/15), SUM 1.05x → 1.12x. **The sign flips with coupling:**
the residual costs RTT on the two near-pointwise configs and pays only on
`sparse_p25_skew` (also p99 71.0s → 63.1s), the one config with real interaction. That is
the `graph_structure_physics` prediction measured directly — graph capacity is wasted, and
actively harmful, wherever the target is additive. A mid-session read that
`logit_tied_rate` was the discriminator is **wrong**: it rose on all three configs,
including the one that improved.

⇒ Model-side work on this corpus is closed. The next lever is physics (Phase 1
`node_contention_v3`), not architecture.

**Two reproducibility traps found, both still open.**
1. `run_provenance` records neither the git commit nor `OMP_NUM_THREADS`. The
   2026-08-16 ablation figure of 0.88x Knative on `sparse_p35/s42` is **not reproducible**
   — the current gate gives 1.04x for the same cell/seed/model/config. That arm ran with
   `GNN_DROP_NODE_EDGES=1`, a variable implemented nowhere in the tree today, so the code
   that produced it no longer exists. Treat the gate as the baseline of record.
2. `logit_tied_rate ≈ 0.54` — the scoring head's top-2 margin is under 0.1 on half of all
   live decisions. A model that indifferent is sensitive to FP reduction order, which is
   why thread count matters. If the residual does not move this, the next lever is the
   ranking loss or edge features, **not** the encoder.

---

### cosim_deepdive_v1 — live-state separability + pipeline integrity census (2026-08-23)

**The question.** Every additivity measurement to date ran on synthetic co-sim states
(4 tasks at t=0, seeded warm queues, forced placements). Live, the GNN is the only arm
that never collapses — so the open root-cause question was whether the co-sim target's
additivity is an artifact of that snapshot regime, i.e. whether **live-visited** states
(autoscaled, queue-loaded, mid-collapse) carry non-additive placement structure the
corpus never captures. If yes, the co-sim strategy pivots to live-regime data; if no,
no corpus design can teach the GNN's live edge through a single-batch regret target.

**Method (pre-registered before the sweeps ran).** Snapshot capture was ported from
`knative_network(_batch)` into the GNN scheduler family (`src/placement/live_audit.py`,
same JSONL schema), then:
1. `knative_network_batch` re-run on all **30 gate cells** (drawgate/promo175/bbrob ×
   2 conditions × 5 cells, byte-matched block mapping) with `LIVE_AUDIT_*` capture,
   stride 31/37 so snapshots span the whole trace (t≈9→116s; caps never hit at 2000) —
   job **710774**. The **mlp/mlptempfix arms** re-run on their **14 collapse events**
   with the same capture — job **710775** — so true collapse-moment states are in the
   test set, not just a Knative proxy.
2. Per cell, 100 time-stratified snapshots swept through the co-sim oracle physics
   (`live_snapshot_cosim_oracle`, cell topology seed — never an override) over the
   Cartesian product of each task's **top-K=6** candidates by schedule-time ECT
   (≤1296 combos/snapshot, collisions allowed; 1.27 ms/combo measured by calibration
   job 710754). Written as pseudo-dataset dirs so `separability_diagnostic.py` M4 runs
   **unmodified** — metric code identical to the corpus baseline. Jobs **710818/710819**;
   zero failed combos in 4,400 sweeps.
3. Verdict pre-registered: "live regime is non-additive" iff median
   `additive_choice_regret_rel` > 0.02 AND median one-integer repair < 0.8, n ≥ 100.

**Result: live states are additive. The snapshot-regime hypothesis is rejected.**

| stratum | n | additive R² med | choice regret med | regret>2% |
|---|---|---|---|---|
| ALL | 4,400 | 0.99999 | 0.00000 | 3.3% |
| knative-proxy states | 3,000 | 0.99999 | 0.00000 | 3.9% |
| mlp collapse-trajectory states | 700 | 1.00000 | 0.00000 | 2.7% |
| mlptempfix collapse-trajectory | 700 | 1.00000 | 0.00000 | 1.4% |
| collapse cells (03+05) | 2,600 | 1.00000 | 0.00000 | 2.7% |
| early / mid / late trace | ~1,470 ea | 0.99995→1.0 | 0.00000 | 6.5 / 1.7 / 1.7% |

K=4 pruning-sensitivity: identical (median regret 0.000). The verdict statistic:
median regret 0.0, `non_additive: false`.

**The 3.3% tail is link-shaped, not count-shaped.** Of the 146 snapshots with regret
> 2%, 112 sit in backbone-capable cells; the one-integer (node-count) column repairs a
median **18%** of their regret while the link-excess control repairs **40%** — the same
signature, at the same small magnitude, that `link_contention_v1` measured in co-sim
(0.08–0.35% mean regret). The live regime confirms rather than overturns that lineage.

**Consequence (the D3 strategy fork, first branch).** Even at the MLP's own collapse
moments the one-step placement surface is pointwise-separable — the collapse is the
*integral* of many individually near-indifferent decisions (consistent with
`chosen_queue_vs_min` median being normal in every collapsed run). Therefore:
1. **Stop investing in co-sim corpus design aimed at making the GNN beat the MLP on the
   supervised objective.** Five physics mechanisms plus, now, the live state
   distribution all say the single-batch target is additive.
2. The paper's GNN claim stays what the gate data supports: dispersal/reliability, an
   emergent closed-loop property. A pointwise scorer cannot condition on its batch
   peers at decode time; nothing in a (state, placement) → RTT target can teach it to.
3. Productive directions if training signal is wanted for the dispersal property:
   trajectory-level / closed-loop objectives, live-snapshot-labeled training
   (`label_live_snapshots_for_training.py` path), or oracle dynamics (autoscaler +
   sustained arrivals inside the horizon). Caveat, pre-registered: this experiment
   tested the state-distribution axis; the dynamics axis (arrivals during the horizon)
   remains untested — but the collapse-trajectory result caps how much could hide there.

**Pipeline integrity census (same campaign, all on the 6 training collections).**

| check | result |
|---|---|
| Sweep truncation | `contention_v2`: **50/900 datasets truncated, 465,282 rows missing** (worst: ds_00769 has 5.5% of its sweep); warmth_1060: 2 truncated + 9 missing JSONL; v3/v4/sparse/highq complete. **25 affected datasets are in the training cache** (0.94% of 2,651). Root cause instrumented (worker exceptions/timeouts silently omitted rows — counters + `COSIM_PLACEMENT_TIMEOUT_S` landed; `sweep_complete` now in `placement_metadata.json`). |
| Additivity robustness | Clean-subset re-run: R² 0.98992 vs 0.99011, regret 0.00821 vs 0.00824 — **the additivity conclusion is robust to the truncation bug**. No retrain triggered (D1 clear). |
| Corrupt rows | 5 corrupt `placements.jsonl` (v2 ds_00751 — torn line, also only 1,923/7,840 rows; warmth ds_00236/240/250/376). `separability_diagnostic --skip-corrupt` added (diagnose-only). |
| Label provenance | Both training caches (`full_corpus_siv1_dim14`, `_tempfix`) label from `placements.jsonl_sweep_minimum` — the 2026-08-04 label bug does **not** touch them. But `optimal_result.json` disagrees with the sweep minimum on **43.2% of warmth_1060 and 47.3% of sparse_warmth** (5-6% on v2/v3): `optimal_result.json`/`best.json` must not be treated as "the optimum" on warmth corpora. |
| Collapse-event contracts | All 60 MLP gate runs served dim22 / scale_invariant_v1 / node_disk_v2, identical across collapsed and healthy cells → **no contract artifact; the ARCHITECTURAL claim stands** (D2 clear). MLP serve path now has the same guards as the GNN path (layout conflict raises; topology + warmth/corpus checks wired). |
| Per-collection separability | The corpus is not uniformly additive: contention v2/v3 R² 0.990/0.993, v4/highq ≈ 1.0, but **warmth_1060 / sparse_warmth (31.1% of the cache) have additive-argmin-optimal only 51.7%/56.1%** at ~2.2% mean regret. The "fully additive corpus" summary was contention-series-shaped. |
| Reproducibility | Infra regen from recorded seeds: **byte-identical** on all 6 training collections (+highq); netc/topo divergence is schema-additive only (`rng_stream` field). Workload regen is **undecidable** — `template_idx` was never recorded per dataset. Queue-dist × task-mix confound: **refuted** (cross-tab exactly balanced 30/30/40). |
| Registry orphans | 16 collections / 4,139 datasets outside `REGISTRY.json` (topo_transfer_v1 3,815; netc series). `netc_pilot_longexec_*` (32 ds, task type `rf`) were generated with **no platform-compatibility repair** (hardcoded `['dnn1','dnn2']`, `generate_infrastructure.py:200`). |
| Skip-reason conflation | Over-limit (`MAX_PLACEMENT_COMBINATIONS_SKIP`) and zero-candidate infeasibility were logged identically — a retroactive census is impossible. `skip_reason.json` now recorded at generation. |

Instrumentation landed with the campaign (commits d88278c…dd3827b): link/ingress/contention
aggregates in both stats paths (`totalLinkWaitTime`, `fabricLinkWaitTotal` — closes the
2026-08-21 gate-tools row; the link-side collapse mechanism is now directly measurable),
`placement_seed` + SLURM identity + topology/network contract env in `run_provenance`,
`topology_feature_contract` in cache metadata and sidecars (same bug class as the 40.8%
layout confound, one field over), `queue_norm_mode` in the sidecar, live-audit capture on
the ML serve paths, `warmth_physics` persisted into `infrastructure.json` metadata.

### program_verdict_v1 — can the co-sim → GNN program ever work? (2026-08-24)

**Closed. Read-only investigation (no new sims, no retrains) answering the generalised D3
fork: is there any path by which a GNN trained here beats MLP and Knative on a live gate.**

**Verdict, in two halves.**
1. **On the win condition the evidence already supports — regime win + reliability — the
   program has already worked, but the evidence is post-hoc, not gate-grade.** `tempfix`
   beats Knative on 30/30 backbone cells across 3 training draws, 2 traces, 3 backbone
   configs (jobs 710315/710335/710341/710366/710398), and beats the MLP's *aggregate*
   margin in 5 of 6 gate conditions because 7/30 MLP cells collapse (jobs
   710450/710451/710656/710657); 0 of 120 GNN runs ever collapsed. **Provenance caveat,
   verified 2026-08-24:** those cells were minted for the `link_contention_v1` real-trace
   A/B, the "< −0.4%" win rule appears at scoring time, and no pre-registration language
   exists anywhere in that campaign's span — each follow-up run was motivated by the
   previous result. Internally well-controlled (same-batch Knative baselines, parity-exact
   cells, measured noise floors), but a reviewer will correctly call it exploratory. The
   claim to publish — after one pre-registered confirmatory gate (below).
2. **On per-cell mean latency vs the MLP, through any single-batch supervised co-sim
   target, the program is terminally closed** — five physics mechanisms
   (`graph_structure_physics` → `link_contention_v1`) plus the live state distribution
   (`cosim_deepdive_v1`, incl. all 14 collapse trajectories) all measure the target as
   pointwise-separable. No corpus design can reopen this; only a change of objective can.

**Path verdicts** (mechanism-level; citations in the session record):
- **P1 closed-loop RL/DAgger — EXPENSIVE-BUT-VIABLE.** A rollout episode exists today
  (`executesimulation.py` + `run_simulation.py`); wall-clock **measured** from
  `logs_sim/`: GNN 754.9 s on the 301k-event `workload-150-100`, 1001.8 s on `175-100`
  (~2.9 GB RSS/worker). ~10³ episodes ≈ 5K CPU-h at gate scale (~500 on the 30k trace)
  plus a 1–2 week build; prior 0.15–0.25 of beating the MLP's healthy-cell packing margin.
  The only path whose objective matches the known live edge (closed-loop dispersal).
- **P2 live-snapshot labels — RULED OUT.** `label_live_snapshots_for_training.py` imports
  `oracle_choice_cosim` — it *is* the one-step oracle `cosim_deepdive_v1` already tested
  on exactly those states. Any non-one-step labelling is P1 or P3 by definition. Flip
  condition: exhibit a single-state labelling that is neither one-step RTT nor a
  horizon/trajectory return and encodes dispersal value — none is known.
- **P3 dynamics inside the oracle horizon — VIABLE, the highest-upside measurement.**
  The two-line scaling test does **not** kill it: node-mediated interaction under arrivals
  is occupancy-count-shaped (empirical rule, 5 mechanisms), but the *link* term is the one
  mechanism that escaped the count control, and it scales with concurrency (0.08–0.35%
  regret at 4-task vs 7–14× cost and ordering changes at rps=150). Pre-registered pilot
  (thresholds fixed 2026-08-24, before any build): extend `live_snapshot_cosim_oracle` to
  continue trace arrivals for a ~10 s horizon on backbone cells; M4 unmodified. Because
  the hypothesised mechanism is link-shaped and link effects surfaced at t=0 as a **3.3%
  tail**, a median-only criterion cannot resolve its own hypothesis — co-primaries:
  (a) median `additive_choice_regret_rel` > 0.02, **or** (b) fraction of snapshots with
  regret > 2% ≥ 15% (≥ 4× the 3.3% t=0 base rate, binomial-testable at the stated n) —
  either fires only with node-count repair < 0.5 AND link repair < 0.5 **on the affected
  stratum**. n ≥ 300 snapshots (≥ 2 backbone cells, K=4 ≈ 256 combos), so the tail holds
  ≥ 10 states under H0 and ~45 under H1. **Cost calibrated 2026-08-24, then corrected the
  same day when the trace-rate naming was measured** (see the propagation note below):
  `workload-150-100` runs ~3,000 events/s at steady state (t≈20–100 s; "rps=150" is
  *per client node*, ×20 clients) with ~20 s ramps at both ends, and the first
  calibration slice (first 10 s, 4,609 events) sat in the ramp. Measured floor: one
  `knative_network_batch` episode on that slice is 7.3 s wall of which 5.1 s is process
  startup ⇒ ~0.48 ms marginal per event ⇒ **~1.4 s per horizon-second per combo at
  steady state**: h=10 s ⇒ ~14 s/combo (~300 CPU-h at 300×256), h=5 s ⇒ ~7 s
  (~154 CPU-h), h=3 s ⇒ ~4.3 s (~92 CPU-h), h=2 s ⇒ ~2.9 s (~62 CPU-h) — before
  backbone-fabric overhead (assumed 1.5–2×). **The horizon length is therefore a
  registered design parameter, not a default**: the in-harness calibration (3 snapshots,
  h ∈ {2, 5, 10}) must (a) confirm combos genuinely run in-process — if each combo pays
  the 5.1 s startup the budget quadruples and the pilot is re-scoped before queueing —
  and (b) fix h at the largest value the stated budget affords, registered before the
  array. Honesty cost of a short horizon, stated up front: queue-runaway ignition may
  need sustained load, so a null at h ≤ 3 s is terminal only for *short-horizon*
  dynamics; terminality for the axis as a whole requires h ≥ 5 s. ~1–2 days build.
  Prior 0.2–0.35.
- **P4 held-duration node contention — RULED OUT on the empirical rule, corrected
  2026-08-24 (the first write-up of this entry overstated it).** What exists holds the
  node slot only around exec (`infrastructure.py:1213-1218` wraps
  `yield timeout(task_duration)` alone; exec ≈ 0.024 s), and `memory/memory.md:79`
  correctly recorded in 2026-08-17 both why that measured `nodeContentionTime ≡ 0.0`
  (backlogs drain in one timeout; placed tasks never overlap) and the unbuilt candidates
  that would couple: a hold across the whole residency (cold starts reach 38 s) or a warm
  lifetime. So the residency-hold variant is *not implemented here*, not already
  falsified. It stays ruled out because its contended object is still node-indexed ⇒ the
  interaction is a co-residency count ⇒ the throughline predicts one-integer degeneracy —
  an invocation of the empirical rule, not a measurement. Flip condition unchanged: a
  slot-contention config with additive-argmin regret > 5% and node-occupancy repair < 50%
  under `--spread-plans-only`.
- **P5a reliability win condition — VIABLE, needs one pre-registered gate** (see the
  provenance caveat in half 1: the 30/30 record is exploratory). The gate must
  co-register the **MLP arm**, not just Knative — the paper's claim is collapse-freedom
  *relative to the MLP*, and the 14/120 collapse count comes from the same exploratory
  campaign as the 30/30. Fresh cells must be minted by a **new** script that does not
  inherit the A/B design (`make_backbone_gate_cells.py`'s cells are the ones the
  "< −0.4%" rule was written against). Remaining: register win condition + thresholds
  (incl. the collapse detector, `chosen_queue_vs_min` p95 with the measured 9.7×
  no-overlap gap) *before* running `tempfix` + MLP + Knative on
  `workload-125-225`/`200-200` and the fresh cells, then promote.
- **P5b batch conditioning — claim must be re-worded before publication.** The deployed
  gates run an *identical* decode for GNN and MLP arms (`mlp_scheduler.py:5-7` inherits it;
  in `argmax` mode `chosen_idx = gnn_idx` unconditionally, `seq_decode.py:719-728` — the
  queue roll-forward feeds stats only). The separation is therefore *score-side
  set-conditioning* (message passing sees the candidate context; a pointwise edge score
  cannot), *not* decode-time peer conditioning. Required control before the paper: MLP +
  candidate-relative queue feature (rank/z-score), retrain + 30-cell re-gate (~1–2 days).
  If that MLP stops collapsing, the honest claim shrinks to feature engineering.
- **P5c topology transfer — stays FAILED for the supervised objective** (its own 5-seed
  gate; note that gate scored the additive target, so the FAIL is *predicted by*
  additivity). Reopening evidence: a live *reliability* gate across sizes (~14 GPU-h
  partial gate, already unblocked). Only worth it as an extension of P5a's claim.
- **P6 freeze — the recommended frame.** Publish (1) the terminal negative — single-batch
  placement targets in this simulator class are pointwise-separable, with
  `separability_diagnostic.py` + the one-integer/link repair controls as the reusable
  artifact — and (2) the reliability result (half 1). Residual reviewer risks, named: one
  topology family (20c/20s); two traces not yet re-gated under the corrected layout; the
  P5b control unrun (the largest); backbone physics authored, not trace-calibrated;
  "GNN > MLP on latency" must not be written.

- **P7 — the least-additive stratum, measured (2026-08-24): the terminal statement holds
  unqualified.** `cosim_deepdive_v1`'s census left `warmth_1060`/`sparse_warmth` (31.1%
  of the training cache, additive-argmin-optimal only 51.7%/56.1%) as the one stratum
  where the one-integer and `--spread-plans-only` controls had never run. Pre-registered
  (STRENGTHEN iff median repair ≥ 0.5 or spread-plans R² ≥ 0.999 with ~zero spread
  regret; QUALIFY iff repair < 0.5 and spread regret > 1%), then run at n=150 per
  collection: `1060_warmth` regret 2.01% → 1.15% under the node-count column (repairs 51%,
  n=67 with any regret, `!! DEGENERATE`); `sparse_warmth` 2.12% → 0.49% (repairs 68%,
  n=64, `!! DEGENERATE`). Decisively, **`--spread-plans-only` takes both collections to
  additive R² = 1.00000 exactly, 0.00% regret, 100% optimal** (sparse: 142/150 fitted,
  8 underdetermined by too few spread plans) — identical to the base-physics isolation
  result. The warmth stratum's non-additivity is entirely the collision term. Frozen
  reports: `simulation_data/separability_{warmth_1060,sparse_warmth}_n150{,_spread}.json`.
  **The spread control itself was then audited for mechanical saturation** (R² = 1.00000
  exactly is also the signature of params ≈ observations, and the control is load-bearing
  for the throughline's "no reservoir" sentence too). Pre-registered (suspect if median
  rows/params < 2; survives iff median held-out R² ≥ 0.999 on a seed-0 half split), then
  measured with `scripts_cosim/audit_spread_fit_saturation.py`: `warmth_1060` ratio
  median 97.3, held-out R² = 1.00000 on 150/150; **`mh_off` (the throughline's base
  corpus) ratio median 7.8 (min 2.0), held-out R² = 1.00000 on 48/48**; `sparse_warmth`
  ratio median 14.7, held-out 1.00000 on 137/144, with 7 failures all at ratio ≤ 2.29 /
  ≤ 16 spread rows — unresolvable at their own n, excluded from the claim's basis rather
  than counted as counter-evidence. Held-out *exactness* is the strong form: overfitting
  cannot produce exact out-of-sample predictions, so the spread-plans conclusion — and
  the "no reservoir of non-count coupling" sentence it underwrites — now rests on
  out-of-sample evidence, not in-sample R². **The claim's basis is 137/144 sparse
  datasets plus 150/150 warmth_1060 plus 48/48 mh_off — not 144/144**; the 7 excluded
  sweeps are unresolved, full stop. The one-integer repair fractions (51%/68%, with 51%
  one point over the DEGENERATE threshold) are secondary; the spread control is the
  criterion this entry leans on.

- **Trace-rate naming propagation check (2026-08-24).** Measured on
  `workload-150-100.json`: the `rps` field is **per client node** — 20 sources ×
  ~15,067 events each, steady state ~3,000 events/s over t≈20–100 s, ~20 s ramps at both
  ends (ts max 118.7 s). Every claim framed "at rps=150" therefore describes an
  operating point whose *system* arrival rate is 20× the label — most prominently
  `link_contention_v1`'s real-trace A/B ("the effect is not small at rps=150", 7–14×
  Knative cost): the measured numbers are untouched, but the concurrency they occurred
  at is ~3,000 events/s, and any external-realism judgement ("is 150 rps a realistic
  load?") made against the label is off by that factor. The same naming corrupted this
  entry's first P3 cost estimate (calibrated on the ramp; corrected above). Paper text
  quoting an "rps" operating point must state the per-client convention and the system
  rate.

**Recommended sequence (reordered 2026-08-24 — downside protection before upside):**
P5b control first (1–2 d; if a candidate-relative queue feature stops the MLP collapsing,
every subsequent P5a gate would have been wasted) → P5a pre-registered gate + re-gates
(2–3 d CPU) → P3 pilot (2–3 d) → **if P3 nulls, the residency-hold scaling test on paper
and, if it demands one, its 16-dataset pilot (~0.5 d) before any P1 spend** — cold start
at 38 s vs 0.024 s exec is a ~1,500× longer hold, so mechanism #1's ≡ 0.0 says nothing
about it, and unlike link bandwidth (where additive `hops×T` and interaction
`crossings×T` scale together, ratio-invariant) a longer hold flips overlap from *never*
to *always* — a threshold, not a ratio. The two-line prediction: interaction =
`(co-residents − slots)⁺ × residency` — a node-occupancy count times a scalar — so the
**expected** outcome is a sixth confirmation of the empirical rule: coupling *with
teeth* that the one-integer column repairs. The pilot is still worth its 0.5 day
because it converts the weakest ruling in the set (a rule invocation) into a
measurement before any P1 spend. Pre-register **both** controls before generating:
the node-count repair column *and* `--spread-plans-only` — the spread control is the
comparison that would actually surprise (residual regret on all-distinct-node plans
under residency holds would be the first non-link escape), and registering only the
count column would leave the surprising outcome ungated → P1 only if P3 (or that
pilot) finds non-count signal (P3's horizon labels are also DAgger targets).

---

### p5b_candidate_relative — PRE-REGISTRATION (written 2026-08-24, before any gate run)

**Everything in this subsection was committed before the training or gate jobs were
submitted.** `program_verdict_v1` closed by finding that its own headline positive — GNN
beats Knative 30/30, MLP collapses 14/120, GNN 0/120 — was scored under a win rule written
at scoring time. Repeating that mistake in the control that tests it would make the whole
exercise worthless, so the rule is fixed here first and
`important/score_p5b_collapse_pairs.py` takes no threshold arguments.

**The question.** The 14/120 collapse was read as architectural because retraining on a
corrected cache moved the victim set (7/30 → a different 7/30) without shrinking it. The
untested cheaper explanation: the MLP's 22 features describe one (task, platform) edge in
absolute terms and never say that this platform's queue is the deepest in the task's own
candidate set. Message passing gives the GNN that comparison for free. So hand it to the
MLP directly and re-run the same cells.

**The intervention.** `dim25cr` = dim22 + 3 columns per (task, candidate) edge, over the
same normalized queue column the model already reads (platform index 7):
`x_22 = q - min(q_cand)`, `x_23 = avg_rank(q)/max(1, n-1)`, `x_24 = (q - mean)/std`
(std == 0 → 0). Shift-invariant; rank and z are also scale-invariant; a single-candidate
task gets zeros. One shared definition
(`reduced_features.candidate_relative_queue_columns`) is imported by both the training
extractor and `MLPBatchScheduler.build_feature_matrix` — a second copy of the formula is
how train/serve skew recurs. **The strongest reasonable version of the feature was chosen
deliberately**: this control exists to break our own result, and a weak version that fails
would be uninformative.

**Design.** Paired, 30 `(cell, condition)` pairs per arm; two arms, each against the
baseline trained on the *same* cache, so the pairing is an A/B on the feature set alone:

| arm | cache | baseline |
|---|---|---|
| `mlpcandrel` | `graphs_cache_full_corpus_siv1_dim14` | `mlp` |
| `mlpcandreltf` | `graphs_cache_full_corpus_siv1_dim14_tempfix` | `mlptempfix` |

Cells, traces, hyperparameters (hidden 64, lr 1e-3, seed 42, test-size 0.2, epochs 100,
patience 10), physics and parity waivers are copied verbatim from
`mlp_tempfix_arm_all_gates.sbatch`.

**Collapse detector.** `chosen_queue_vs_min.p95` from the `.decode_stats.json` sidecar —
the detector measured to separate all 120 prior runs with no overlap (collapse
13,485–23,866, healthy 449–1,387; the *median* is normal in both, which is the direct
evidence that collapse is a compounding minority-of-decisions tail). **Collapse iff
p95 ≥ 5,000.** A cell landing in the never-observed band (1,387, 13,485) is reported as
such: the separation is an empirical fact about 120 runs, not a law.

**Primary test.** Exact one-sided McNemar on the paired collapse indicator. `b` = baseline
collapsed / candrel healthy, `c` = baseline healthy / candrel collapsed; under H0,
`b | b+c ~ Binom(b+c, ½)`.
- **REFUTE** — one-sided `p ≤ 0.05` with `b > c`, in **both** arms. Against a 7/30
  baseline that means `b=6, c=0` (p = 0.0156) or `b=7, c≤1` (p = 0.0352).
- **HARDEN** — ≥ 5 of the baseline's collapses still collapse, arm total ≥ 5/30, and
  `p > 0.05`, in **both** arms.
- **INDETERMINATE** — anything else, *including* the two arms disagreeing. Reported as
  indeterminate; not read as either result.

**Threshold sensitivity is part of the rule, not a robustness afterthought.** Each arm's
verdict is recomputed at 2,000 / 5,000 / 10,000; if it is not identical at all three, that
arm is INDETERMINATE regardless of its p-value at 5,000.

**Secondary, reported always, never the basis of the verdict.** Mean margin vs the
same-condition Knative arm and per-cell wins at the measured `< −0.4%` noise floor.

**Validity gates — both must pass before the 60 gate runs are submitted.** A null from a
model that ignored the new columns is evidence about nothing.
1. Held-out test edge accuracy ≥ its own baseline − 0.01 (`mlp` baseline: 0.8842).
2. Zeroing `x_22..x_24` moves ≥ 5% of held-out argmaxes.

Both are computed by the trainer, recorded in the checkpoint `.meta.json`, and asserted by
`fc_siv1_mlp_candrel.sbatch`, which refuses to finish if either fails. **If (2) fails the
feature is inert: fix the representation and retrain — do not report the null.** Measured
locally at 2 epochs on the `dim14` cache before submission: **29.1%**, so the feature is
demonstrably load-bearing.

**What each outcome does to the paper.**
- REFUTE → the reliability separation is feature engineering, not architecture. "The GNN
  is the only learned scheduler that never collapses" must be restated as a statement
  about a feature the baseline lacked, and P5a's pre-registered gate is not worth running
  in its current form.
- HARDEN → the architectural reading stands, now against the strongest hand-engineered
  version of the missing feature rather than against its absence. Proceed to P5a.

**Provenance.** Layout served as `dim25cr` and declared in the sbatch; a conflicting
declaration is a hard load error, verified before submission (undeclared → pins from the
sidecar; `dim22` declared → refuses). Checkpoints carry `inference_feature_layout`,
`candidate_relative`, the column spec, `code_provenance` and `python_env`. Scoring must
pass `--expect-layouts` to `score_live_gate_matrix.py`: the arms differ in layout **by
design** here, and the guard was changed from an equality check to a declaration check
precisely so that an *unintended* mixture still fails loud.

### p5b_candidate_relative — OUTCOME: **INDETERMINATE**, and the reason is the finding (2026-08-24)

**Ran as registered.** Retrains job `711675` (both validity gates passed: accuracy vs own
baseline +0.0000 / −0.0038; CR-ablation argmax change 21.3% / 28.8%). Gate array `711679`,
60/60 COMPLETED, 30 results + 30 decode-stats sidecars per arm, no failures, all runs
`commit=886f5593 dirty=False torch=2.5.1+cu121`. Verdict artifact:
`simulation_data/p5b_verdict.json`.

**Registered verdict: INDETERMINATE — the two paired arms moved in opposite directions.**

| arm | cache | collapses (registered detector) | fixed / broken / both | p (1-sided) | pair verdict |
|---|---|---|---|---|---|
| `mlpcandrel` | `dim14` | 7/30 → **17/30** | 2 / 12 / 5 | 0.9991 | HARDEN |
| `mlpcandreltf` | `dim14_tempfix` | 7/30 → **2/30** | 5 / 0 / 2 | 0.0312 | INDETERMINATE (threshold-unstable: 2,000 → p=0.062) |

**The split is not noise, and it survives dropping the registered detector entirely.**
Re-scored post-hoc on `total_rtt` vs the same-cell Knative arm, at +30/+50/+100%:
`mlpcandrel` goes 7→13, 6→12, 5→11 (p = 0.989 / 0.996 / 1.000); `mlpcandreltf` goes
7→2 at **all three** thresholds (p = 0.0312 each, i.e. *more* stable than under the
registered detector). Registered secondary — mean margin vs same-condition Knative:
`mlpcandrel` blows out to **+436.1%** and **+509.9%** on the two nobackbone blocks, while
`mlpcandreltf` turns **negative in 4 of 6 conditions** (−23.2%, −20.0%, −26.8%, −33.8%)
and wins 26/30 cells — the first MLP arm in this repo to approach the GNN's record.

**What this does and does not license.**
- **"A candidate-relative feature fixes the MLP collapse" is NOT supported** — it fixed
  one arm and made the other roughly twice as bad.
- **"A pointwise scorer collapses *because* it cannot condition on the candidate set" is
  also no longer supported**, and this is the load-bearing correction: `mlpcandreltf` has
  exactly that conditioning, uses it heavily (28.8% of argmaxes move when it is ablated),
  and largely stops collapsing. The mechanism sentence in the 2026-08-23 subsection above
  ("a pointwise scorer cannot condition on where its peers are going") must not be written
  as the explanation of the reliability gap.
- **The GNN's 0/120 record is untouched** by this lineage — no GNN arm was re-run.
- What the split *does* support is the existing architectural reading in its weaker,
  weights-not-data form (`memory/herosim-mlp-collapse-is-occupation-collapse.md`): all
  four MLP checkpoints sit within ±0.004 test edge accuracy of each other while their live
  collapse counts range 2/30 → 17/30. **Supervised accuracy does not constrain live
  reliability at all**, and the same feature added to two caches moves reliability in
  opposite directions. Pointwise reliability here is a property of the draw.

**Blocking confound, unresolved: cache or seed?** Both candrel arms used `--random-state
42`, so "cache" and "training draw" are perfectly confounded — exactly the confound
`memory/herosim-live-quality-is-a-training-draw-lottery.md` was written about. The 7→17
and 7→2 results cannot be attributed to the corpus difference until the same checkpoints
are retrained at ≥3 seeds per cache and re-gated. Cheap: ~4 min per train, 30 gate runs
per checkpoint. **No claim about which cache is "better" may be written before that.**

**Methodological finding — the pre-registered detector is confounded by this specific
intervention.** `chosen_queue_vs_min` p95 agrees with catastrophic RTT on 29/30 (`mlp`),
30/30 (`mlptempfix`) and 30/30 (`mlpcandreltf`) — but only **25/30 for `mlpcandrel`**,
where it fires on five cells whose RTT is fine, including one at **−2.5% (a win)**. The
cause is structural, not statistical: the CR columns make choosing a non-minimum-queue
candidate a *deliberate learned behaviour*, so the detector partly measures the
intervention itself. Its errors are one-directional in all 180 runs (fires-but-healthy,
never quiet-but-collapsed), so it remains sound as a *negative* test. **Any future gate on
a candidate-relative arm must score collapse on RTT, not on this detector** — and the
registered verdict above would be INDETERMINATE either way, so nothing is being rescued
by saying so after the fact.

**Status: CLOSED as INDETERMINATE.** It did its job: it was registered first, it ran
clean, and it falsified the mechanism sentence the paper was going to lean on. The
sequence does **not** proceed to P5a as written — P5a's win condition assumed the MLP arm
was the reliability foil, and one MLP arm now beats Knative in 4 of 6 conditions. Resolve
the seed/cache confound first — see the draw study below.

### p5b_draw_study — PRE-REGISTRATION (written 2026-08-24, before any run)

**🔴 First, a defect that reframes every MLP result in this repo.** Nothing ever called
`torch.manual_seed`. `--random-state` seeded the parent split
(`split_by_parent_three_way`) and the batch order (`random.Random`) — **the model's weight
init came from OS entropy**. Verified by construction: two identical invocations of the
trainer produce different first-layer weights; with an explicit seed they are bit-identical.

Consequences, stated plainly:
- **Every MLP checkpoint produced before 2026-08-24 is an unreproducible draw.** Re-running
  its exact command cannot recover it. `random_state: 42` in those `.meta.json` files
  describes the split, not the model.
- The 2026-08-23 subsection above describes the `mlp` vs `mlptempfix` comparison as
  "an A/B on training data alone" and concludes the collapse is architectural because
  "only the checkpoint and the sweep dir differ". **That description is wrong**: the two
  checkpoints differ by cache *and* by an uncontrolled weight init. The observation
  (7/30 each, different victims) stands; the attribution to the cache does not.
- P5b's own confound is therefore not merely "cache vs seed" but "cache vs an
  uncontrolled draw", which is worse and cannot be resolved by re-reading anything.

`torch.manual_seed(args.random_state)` + `np.random.seed` are now wired in, and
checkpoints record `torch_seeded: true` so a seeded checkpoint can be told from a drawn
one. Verified: same seed → bit-identical weights.

**The study.** A full grid, so variance can finally be attributed instead of assumed:
`{dim14, dim14_tempfix} × {dim22, dim25cr} × seeds {1,2,3,4}` = 16 checkpoints × the same
30 backbone cells = 480 gate runs (`p5b_draw_study_{train,gate}.sbatch`). The `dim22` arms
are **not** padding: they measure what a *fixed* cache and layout does across draws, which
is the quantity every reliability claim in this program has silently assumed to be small.
Existing s42 checkpoints are retained as a 5th, unseeded draw and are never mixed into the
seeded statistics.

**Criterion, registered.** Collapse = `total_rtt` ≥ **+50%** vs the same-cell Knative arm.
Chosen over the P5b detector because `chosen_queue_vs_min` p95 is *measured* invalid for
candidate-relative arms (fires on 5/30 healthy `mlpcandrel` cells, one a −2.5% win).
Sensitivity at +30% and +100%; a verdict that does not hold at all three is INDETERMINATE.

**Q1 — is pointwise reliability a draw lottery?** Per condition, the range of collapse
counts across its 4 seeds.
- **LOTTERY** iff the largest within-condition range ≥ **5**/30.
- **STABLE** iff every condition's range ≤ **2**/30.
- **PARTIAL** otherwise.
Rationale for 5: P5b's headline effect was 7→2 and 7→17. If a *fixed* cache and layout
swings ≥ 5 cells on the seed alone, the feature effect was never distinguishable from a
draw, and neither was the 7/30-vs-7/30 result the architectural reading rests on.

**Q2 — does the candrel effect have a cache-determined sign?** Per (cache, seed),
`delta = collapses(dim25cr) − collapses(dim22)` at the same cache and seed: 8 deltas.
- **CACHE-DETERMINED** iff all 4 deltas of one cache are > 0 and all 4 of the other < 0
  (perfect sign separation; coin-flip null p = 2 × 2⁻⁸ = 0.0078).
- **DRAW-DOMINATED** iff the sign is mixed within either cache.
- Ties (delta = 0) count against separation.

**Q3 — descriptive, no threshold.** Mean and range of collapse count per layout, pooled
over caches and seeds (8 checkpoints each). Reported whatever Q1 says, but **not**
interpreted as a feature effect if Q1 returns LOTTERY.

**Validity gate 2 still applies** to every `dim25cr` checkpoint (CR-ablation argmax change
≥ 5%), asserted in the training sbatch, which refuses to finish otherwise.

**What closes the story.** Q1 = LOTTERY would mean the honest headline is *"pointwise
reliability on this benchmark is a property of the draw; the GNN's 0/120 is the only
claim that survives, and it needs its own multi-seed check"*. Q1 = STABLE with Q2 =
CACHE-DETERMINED would restore a real, attributable feature/corpus effect. Either way the
`p5b_candidate_relative` INDETERMINATE resolves into a statement that can be written down.

### p5b_draw_study — OUTCOME: **Q1 = LOTTERY, Q2 = DRAW-DOMINATED** (2026-08-24)

Trains job `711758` (16/16; every `dim25cr` arm passed validity gate 2, 0.179–0.291); gate
array `711774`, **480/480 COMPLETED**, 96 sweep dirs, zero failures. Artifact:
`simulation_data/p5b_draw_study_verdict.json`.

**Collapse counts, `total_rtt` ≥ +50% vs the same-cell Knative arm:**

| condition | s1 | s2 | s3 | s4 | range |
|---|---|---|---|---|---|
| `dim14 / dim22` | 0/30 | 0/30 | 8/30 | 10/30 | **10** |
| `dim14 / dim25cr` | 5/30 | 3/30 | 0/30 | 11/30 | **11** |
| `tempfix / dim22` | 0/30 | 0/30 | 21/30 | 16/30 | **21** |
| `tempfix / dim25cr` | 26/30 | 0/30 | 0/30 | 7/30 | **26** |

**Q1 = LOTTERY** (worst range 26/30 against a threshold of 5). **Q2 = DRAW-DOMINATED** —
paired deltas are `dim14: +5, +3, −8, +1` and `tempfix: +26, 0, −21, −9`; the sign is mixed
*within* both caches, so cache separation is arithmetically impossible. **Both verdicts
hold at +30%, +50% and +100%**, as the registered rule requires. Q3 (descriptive): pooled
over caches and seeds the two layouts have the **same median, 4.0/30** — the
candidate-relative feature has no average effect whatsoever.

**This retires three claims that were load-bearing in this file.**
1. **"The MLP collapses 7/30" was one draw.** The same configuration
   (`tempfix / dim22` — the corrected-cache baseline) gives **0, 0, 21, 16** across seeds.
   Two of four draws never collapse on any of the 30 cells.
2. **"Exactly 7 of 30 under each checkpoint — same count, different set" is a
   coincidence**, not the signature of an architectural failure. It was two samples from a
   distribution whose range is 21.
3. **P5b's 7→2 / 7→17 split was noise.** It sits well inside the baseline's own draw
   spread, and the feature's pooled median effect is zero.

**The seeding defect is MLP-specific.** `src/notebooks/train_near_rtt.py:104-107` seeds
`random`/`numpy`/`torch`/`torch.cuda`; the MLP trainer seeded none of them until today. The
GNN's checkpoints were always reproducible and its arms are genuine distinct draws — the
asymmetry went unnoticed because nobody compared the two trainers' seeding.

**What survives, and how strong it actually is.** On these 30 cells both GNN arms are 0/30
collapses with mean margins −18.9% (`deployed`) and −27.1% (`tempfix`), against MLP arms at
6/30, 7/30, 12/30, 2/30 and margins +25.9%, +48.0%, +203.8%, −6.9%. Real — **but it is 2–3
GNN draws against a now-measured MLP draw distribution.** Under the MLP's own draw-level
rate (4 of 8 seeded draws collapse at least once), three clean GNN draws is
p ≈ 0.5³ = **0.125 — not significant**. *The GNN's reliability advantage is not
established; it is unfalsified.* Establishing it requires exactly what was just done to the
MLP: ≥ 8 seeded GNN draws × 30 cells, pre-registered.

**Consequences for the program.**
- **Do not write "the GNN never collapses and the MLP does."** Write, if anything: *across
  the draws tested, every GNN draw was collapse-free while MLP draws collapsed in 4 of 8 —
  a difference this evidence cannot separate from a 1-in-8 outcome.*
- **P5a is superseded, not merely unrunnable.** Any reliability gate on this benchmark must
  compare *draw distributions*; one checkpoint per arm measures nothing. This is the most
  important design change to come out of `program_verdict_v1`.
- The terminal negative from `program_verdict_v1` (single-batch co-sim targets are
  pointwise-separable) is **untouched** — it never rested on any checkpoint.
- Every past per-checkpoint live-gate margin in this file inherits the caveat. The numbers
  are correct; what they measure is one draw.

**Status: CLOSED.** `p5b_candidate_relative`'s INDETERMINATE is resolved: the feature did
nothing, the cache did nothing, and the variable that moved was the one nobody was
controlling.

---

### serving_speed_v1 — the episode cost was `Data.to()`, not the device (2026-08-25)

Groundwork for the `gnn_draw_study_v1` gate below: 240 gate episodes are worth speeding up
first, and the draws must be trained under a settled regime.

**What was measured.** One 30k-event episode, `cell01_p25_s9001` × `workload-150-100-30k`,
GNN policy, deployed checkpoint, same box:

| arm | wall |
|---|---:|
| before (whole-graph `Data.to()`, cuda) | **93 s** |
| after (tensor-only move, cuda) | **72 s** |
| after (tensor-only move, cpu) | **86 s** |

**The predicted lever was the wrong one.** `PROGRAM_VERDICT`'s profiling attributed ~26% of
the episode to `Data.to(device)` and concluded "a ~3–4× is free: run inference on CPU."
Half right. The 26% was real and is recovered (93 s → 72 s, 22.6%), but it was never a
*transfer* cost — it is PyG's `Data.to()` recursing through every stored attribute,
including the `queue_snapshot` and `task_logit_to_placement` dicts the scheduler attaches
before the move, which the forward pass never reads. Moving only tensors is **174× faster
per call** and the win is the same on CPU. Serving on CPU is *slower* than cuda here (86 s
vs 72 s); there is no 3–4×, on either device.

**Changes.** `move_graph_tensors_` in `src/policy/gnn/scheduler.py` replaces `graph.to()`
at all three decode call sites (the field list comes from `verify_venue_parity.py`'s
`GRAPH_TENSOR_FIELDS`, which had already worked this out for the fixture path).
`resolve_serving_device()` in `executesimulation.py` reads `HEROSIM_GNN_DEVICE`
(`cpu` default | `cuda` | `auto` = old behavior), and the resolved device is now stamped
into `run_provenance` — `env_fingerprint` previously recorded only `cuda_available`, which
describes the box, not what served. Not added to `STRICT_KEYS`, so existing fingerprints
stand.

**Why cpu is the default given cuda is faster.** For parity, not speed. `cuda` is the only
axis `PARITY.md` finds that moves GNN logits at all (1.9e-5), and it is visible end to end:
the same cell's `total_rtt` differs by **4.6e-6 relative** between the two devices. A cpu
default makes a local run and a datalab `CPU-amd` gate resolve to the same device. The cost
is ~19% on a GPU box and **zero on the partition every gate actually runs on**. Set
`HEROSIM_GNN_DEVICE=auto` to restore the old behavior.

**Status: ACTIVE.** 366 tests pass, `test_venue_parity` included — no fixture re-baseline.

---

### trainer_determinism_v1 — the seed fix reached 1 of 4 trainers (2026-08-25)

`p5b_draw_study` fixed `train_mlp_dim22_from_batch.py` and stopped there.
`train_mlp.py`, `train_mlp_ce_reduced.py` and `train_mlp_dim22_from_seq.py` still had the
identical defect — split and batch order seeded, weight init from OS entropy — so
"every MLP checkpoint before 2026-08-24 is an unreproducible draw" was still *true going
forward* for three of the four trainers. All three now seed torch and stamp `torch_seeded`.

`torch.use_deterministic_algorithms(True, warn_only=True)` added to all four notebook GNN
trainers (`train_near_rtt`, `train`, `train_ram`, `train_seq`), default-on, escape hatch
`NEAR_RTT_NONDETERMINISTIC=1`. `gnn_necessity_ablation.py` has had this since 2026-08-19;
the trainers that produce deployable checkpoints did not. **This changes training numerics**
— a different algorithm is selected — so checkpoints trained from here are not bit-comparable
to earlier ones. Deliberate, and it lands *before* the draw study trains.

**`tests/test_trainer_determinism.py`** (10 tests, ~12 s, no GPU). Two runs at one seed must
give bit-identical weights. Both dynamic arms are verified to have teeth: remove
`torch.manual_seed` and the MLP arm fails 2 ways, the GNN arm diverges 28/31 tensors.

**What the test does not cover, measured rather than assumed.** It cannot catch removal of
`use_deterministic_algorithms`. The GIN nondeterminism recorded on 2026-08-19 **did not
reproduce on this box at any size tried** — 12 to 200 graphs, 2 to 5 epochs, node edges on
and off, flag on and off, all bit-identical. So the dynamic test cannot discriminate, and
the guard for that half is a *static* assertion that the line is present in every trainer.
Absence on one box is not evidence the op is gone; it is build- and hardware-dependent.

**Status: ACTIVE.** No CI exists in this repo — the command is documented in `CLAUDE.md`
and is manual.

---

### gnn_draw_study_v1 — PRE-REGISTRATION (written 2026-08-25, before any run)

**The only open empirical question in the program.** `p5b_draw_study` retired the MLP's
reliability record by showing it is a draw lottery. The GNN's record — both arms 0/30, mean
margins −18.9% and −27.1% — survived only because nobody had varied its seed. It is 2–3
draws against a measured distribution in which **7 of 16** MLP draws are clean, so a clean
GNN draw is roughly a coin flip and p ≈ 0.125. Writing "the GNN never collapses and the MLP
does" on that evidence repeats exactly the error `p5b_draw_study` was run to find.

**Design.** 8 seeded draws of the **deployed** config
(`graphs_cache_full_corpus_siv1_dim14`, seeds 1–8, `NEAR_RTT_TRAIN_SEED`), gated on the same
30 cells (5 topology cells × 6 blocks) with the same Knative arms already in
`gate_stats_summary.json`. 8 × 30 = 240 gate tasks. One config, not four: the claim under
test is about the checkpoint that is deployed, and splitting the budget would halve the
power on the only question asked. Nothing may vary across arms but the seed — the split
stays fixed at `random_state=42` inside the trainer, and `NEAR_RTT_MP_RESIDUAL` /
`GNN_MP_NODE_EDGES` are asserted unset.

**Entry points.** `datalab/gnn_draw_study_{train,gate}.sbatch` →
`important/score_gnn_draw_study.py` (a sibling of `score_p5b_draw_study.py`, not an edit).

**The rule, registered now.** Collapse is `total_rtt ≥ +50%` vs the same-cell Knative arm,
sensitivity at +30/+50/+100, verdict INDETERMINATE if it does not hold at all three —
identical to `p5b_draw_study`, deliberately, because the two distributions are being
compared and a different threshold would compare different questions. A draw is **clean**
when it collapses on zero of its 30 cells.

- **Q1** — within-condition range across the 8 seeds: `LOTTERY` if ≥ 5, `STABLE` if ≤ 2.
- **Q2** — one-sided Fisher exact, GNN clean draws vs the frozen MLP 7/16, α = 0.05. The
  MLP side is **read from the same summary at scoring time**, not typed in.

**Power, computed before the run, and the reason there is an escalation clause.** Against
7/16:

| GNN clean | Fisher p | |
|---|---:|---|
| 8/8 | **0.0087** | significant |
| 7/8 | **0.0507** | misses, by 0.0007 |
| 6/8 | 0.1557 | |

At n=8 **one unclean draw takes the study from decisive to not-established.** That is not a
licence to read 7/8 as a positive afterwards; it is why the response is fixed in advance,
in the same shape as `gate_statistics.py`'s tier ladder and for the same stated reason ("a
run below the power table is VOID, not FAIL"):

> **7/8 clean → `ESCALATE`, verdict VOID**, train seeds 9–12 and re-gate (existing draws
> stay valid, nothing is re-run); at n=12 the rule tolerates 2 unclean draws
> (10/12 → p = 0.0398). **≤ 6/8 clean → `NOT-ESTABLISHED`**, and that is a real negative.

**What each outcome means.** `STABLE` + `GNN-MORE-RELIABLE` is the one result that lets the
reliability claim be written. `LOTTERY` closes the last open empirical question in favour of
the program's terminal negative: the GNN's 0/30 arms were lucky draws and the claim retires
the way the MLP's did. Anything else is reported as a table and summarised as neither.

**Status: PRE-REGISTERED.** Outcome row to follow.

---

### gnn_draw_study_v1 — OUTCOME: **INDETERMINATE**, and "the GNN never collapses" is FALSIFIED (2026-08-25)

240/240 gate tasks, 8 arms × exactly 30 cells, 720 steps COMPLETED, zero failures.
Job 712381 (train, 8×GPU) → 712389 (gate, 240 tasks). Artifacts:
`simulation_data/gnn_draw_study_verdict.json`, `gate_stats_summary.json` (930 results).

**Collapse counts at the primary threshold (`total_rtt ≥ +50%` vs same-cell Knative):**

| s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | range |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | **1** | 0 | 0 | **3** | 0 | 0 | 0 | 3 |

MLP comparison group, same cells, same rule: `0 0 0 0 0 0 0 3 5 7 8 10 11 16 21 26` → 7/16 clean.

**Verdict: Q1 INDETERMINATE, Q2 INDETERMINATE.** Neither survives the registered sensitivity
requirement — the verdict must hold at +30/+50/+100 and it does not:

| threshold | GNN range | Q1 | clean | Fisher p | Q2 |
|---|---:|---|---|---:|---|
| +30% | 10 | LOTTERY | 5/8 vs 6/16 | 0.235 | NOT-ESTABLISHED |
| +50% | 3 | PARTIAL | 6/8 vs 7/16 | 0.156 | NOT-ESTABLISHED |
| +100% | 0 | STABLE | 8/8 vs 7/16 | **0.0087** | GNN-MORE-RELIABLE |

The whole verdict is a function of where the collapse line is drawn, which is exactly what
the sensitivity clause exists to catch. Reporting the +100% row alone would be threshold
shopping; it was pre-registered as one of three, not as the answer.

**What IS established, and it is the point of the study:**

1. **"The GNN never collapses" is dead.** It rested on 2–3 draws reading 0/30. With the seed
   varied, **2 of 8 draws collapse cells** at the registered threshold (s5 worst cell
   +63.2%, s2 +51.5%). The claim the program was going to publish does not survive its own
   first multi-seed test — the same fate as the MLP claim, for the same reason.
2. **The GNN is nonetheless far better behaved than the MLP.** Worst GNN draw is 3/30; the
   MLP's worst is 26/30, and its range across seeds is 26 against the GNN's 3. The
   *direction* is not in doubt at any threshold; only whether it clears α=0.05 is.
3. **The gap is real but under-powered at n=8.** p=0.156 at the primary threshold. The
   registered escalation (7/8 → 12 draws) did **not** fire: 6/8 is below its floor, so this
   is a genuine NOT-ESTABLISHED at +50%, not a VOID awaiting more draws.

**The honest one-liner:** GNN reliability is not a coin flip the way MLP reliability is, but
it is not the invariant the 0/30 record implied either — it is a *tight* distribution with a
tail, and the program may not describe it as "never collapses".

**Do not** re-run this at a different threshold hoping for a cleaner verdict; the three
thresholds and α were fixed in `score_gnn_draw_study.py` before any draw was gated, and the
INDETERMINATE is the registered answer. A future study wanting to resolve Q2 needs more
draws (≥12 at the +50% line), not a different rule.

**Status: CLOSED.** The last open empirical question in `PROGRAM_VERDICT` is answered: the
GNN reliability claim cannot be written as stated. The terminal negative
(`program_verdict_v1`) is untouched — it never rested on any checkpoint.

**Addendum (2026-08-25): a POST-HOC EXPLORATORY rank statistic, checked against a
pre-written decision rule, does not reopen this.** The registered dichotomy (clean/not-clean)
discards the collapse-count magnitude — a 26-cell draw scores identically to a 3-cell one —
so an exact permutation rank-sum test was run on the full collapse-count vectors (same
`collapse_counts()` rule, same cells, same three thresholds) as a check on whether that loss
of power was hiding a real effect. It was pre-decided that this would only be actionable if
the rank statistic cleared α=0.05 at **all three** thresholds, matching the registered
sensitivity requirement:

| threshold | GNN counts (s1–s8) | rank-sum | exact p | binary Fisher p |
|---|---|---:|---:|---:|
| +30% | 0, 8, 0, 0, 10, 2, 0, 0 | 80.0 | 0.1041 | 0.235 |
| +50% | 0, 1, 0, 0, 3, 0, 0, 0 | 71.5 | 0.0274 | 0.156 |
| +100% | all zero | — | 0.0087 | 0.0087 |

The rank statistic clears α at +50% and +100% but **not at +30%** (p=0.1041), so it fails the
same sensitivity requirement the registered dichotomy failed, for the same reason: whichever
statistic is used, the verdict is not stable across the pre-registered threshold ladder. No
new registration, no escalation to n=12. This was the one outstanding number from
`gnn_draw_study_verdict.json`'s Q2 (the local summary predated job 712389 and only carried
clean/not-clean counts, not full vectors); it is now filled from the synced 930-result
`gate_stats_summary.json` and closes the question the same way. CLOSED stands.

---

### route_a_v1 — the five blockers, cleared (2026-08-25)

**The hypothesis.** `program_verdict_v1` closed the supervised route by theorem: with
per-task costs separable and placements freely chosen, the componentwise minimiser is
optimal under **any** monotone aggregation, so no objective, target or scoring rule can
create structure — and co-location coupling cannot supply it either, because co-residents
are exchangeable within type and every symmetric function of a multiset is a function of
counts. Route A is the one structure that defeats both barriers at once: **a child's input
read priced by the network distance from its parent's node**, which is a pairwise term over
two *jointly decided* placements, between tasks playing *different structural roles*.

The trap that governs the whole design: if the parent is already placed when the child is
decided, "distance from the parent's node" is just another edge feature and a pointwise
model recovers optimality. Non-separability requires parent and child in the **same
jointly-decided set** — which co-sim already gives, since a `placement_plan` fixes every
task's platform before the episode runs.

**This entry is the groundwork, not the experiment.** No corpus has been generated and no
gate has been run. What it records is that the simulator can now express the hypothesis at
all — it could not before, in five separate ways, none of which could ever fire while every
application in every corpus is a single-node dag:

| # | Blocker | State |
|---|---|---|
| 1 | `workflow_process` dispatched a **linearization**, so `A → {B,C,D}` ran as a depth-3 chain — siblings never overlapped and were never co-decidable | fixed, 5 tests |
| 2 | `DeterminedScheduler._collect_task_batch` blocked without a timeout ⇒ **deadlock** on any DAG (`batch_timeout` was configured and never read) | fixed |
| 3 | Placement enumerator walked `dag.keys()` while the simulator assigns ids by `static_order()` ⇒ silent `forced_placements` mis-assignment | fixed |
| 4 | **No server↔server distance exists anywhere** — `network_map` and backbone routes are client↔server only, so a parent and child on two servers have no distance and no path | `build_server_mesh`, opt-in |
| 5 | `prepare_graphs_cache` hardcoded dnn1/dnn2 candidates ⇒ any other task type gets **zero** candidates, label `-1`, contract-5.5 failure for the whole cache | fixed, filter only |

**The new physics.** `Platform._dependency_transfer_time` charges each *remote* parent's
`stateSize[...]["output"]` over this node's bandwidth plus the parent→child network latency.
It reads **all** parents, closing the `dependencies[-1]` FIXME that silently drops every
parent but one on a fan-in. Gated on `HEROSIM_DATA_LOCALITY=1` and inert without
dependencies. Missing reachability **raises** rather than charging 0.0 — a free bad
placement is the signal inverted.

`HEROSIM_STATE_SIZE_BYTES` scales the input `stateSize` in memory. It is not a convenience:
`data/nofs-ids/task-types.json` is shared by *every* corpus and is never copied per dataset,
so editing the welded 153,600 B would rewrite the physics of every existing collection. It
is also the lever — the coupled term scales with `stateSize` while queue work does not, so
unlike link bandwidth (where both scale as `1/bandwidth` and the ratio is invariant) **the
ratio moves**. At the welded value the dependency read is ~1.2% of the queue term.

**Zero-diff is the load-bearing claim here**, since all of this touches shared physics: a
30k-event GNN episode on `cell01` reproduces `total_rtt = 1375056.421447831` bit-identically
before and after, *including with `HEROSIM_DATA_LOCALITY=1` set* — the term cannot fire
without dependencies, and no corpus has any. 385 tests pass. New: `tests/test_dag_dispatch.py`
(5), `tests/test_data_locality_cost.py` (14); the DAG-specific dispatch tests are verified to
fail against the old implementation.

**Not done, and required before any claim** — the pilot itself: DAG workload templates +
grid preset (§9 route A), the `stateSize` scaling probe with its **go/no-go** (if no
plausible `stateSize` makes additive-argmin regret non-zero on spread plans, stop — that is
the cheap intended failure point), pre-registration, the n≥200 corpus, the k-integer and
parent-node-identity repair controls in `separability_diagnostic.py`, and set-valued labels
(makespan optima tie 2–34 deep, and `audit_label_provenance` asserts a unique minimum).

**Status: SUPERSEDED by the two probe entries below — route_a_v1 is CLOSED (NO-GO).** This
row documents the *machinery*, which stands and is reusable: DAG dispatch, fan-in, the
server mesh, the path-bandwidth transfer term, and the makespan channel are all
prerequisites for §9 route B, which is where the closing entry points next. Nothing here
is evidence for or against route A; the verdict is below.

---

### route_a_v1 — scaling probe: **NO-GO**, and the reason is a defect in the term, not a verdict on route A (2026-08-25)

The pre-registered go/no-go before spending an n≥200 corpus
(`scripts_cosim/score_route_a_scaling_probe.py`, thresholds fixed before any arm ran):
proceed only if spread-plan **additive-argmin regret > 5%** *and* it **rises with
`stateSize`**. Artifacts: `simulation_data/route_a_scaling_probe_final_{rtt,makespan}.json`.

**4 arms × 6 datasets (23 with sweeps), `stateSize` spanning 100,000× — 8 KB to 800 MB
transfer payloads:**

| `stateSize` | transfer payload | best RTT | spread-plan regret (rtt) | (makespan) |
|---:|---:|---:|---:|---:|
| 153,600 | 8 KB | 9.75 s | **0.000%** | **0.000%** |
| 15,360,000 | 800 KB | 10.34 s | **0.000%** | **0.000%** |
| 153,600,000 | 8 MB | 15.18 s | **0.000%** | **0.000%** |
| 15,360,000,000 | 800 MB | **950 s** | **0.000%** | **0.000%** |

Zero in every arm, every dataset, both objectives — `nonzero_frac = 0.00` throughout. Not a
marginal miss.

**Everything the probe needed was verified present**, which is what makes the diagnosis
below trustworthy rather than a shrug:
- the DAG is real — retained `task_times` show both children dispatching at the parent's
  completion (0.204) and the join dispatching at `max(0.891, 0.505)`;
- the mesh is real — 380 server↔server edges, **190 distinct latencies**, 4.8× spread;
- the term is material — at the top arm it *dominates*, taking RTT from ~10 s to ~950 s.

**So why exactly zero? The implemented term's magnitude-carrying half is separable by
construction.** `_dependency_transfer_time` charges

```
payload / bandwidth(CHILD's node)   +   latency(parent, child)
```

The first term is indexed by the **child alone** — a per-task cost, exactly the shape a
pointwise model already fits. Only `latency(parent, child)` is pairwise, and latency does
**not** scale with `stateSize`: it stays bounded at 0.031–0.149 s while the separable half
grows to hundreds of seconds. Raising `stateSize` therefore drove the *additive* term and
left the coupled term pinned. This is the same class of error as the `input`/`output` field
mismatch found earlier in the same probe (the lever initially scaled `input` while the
transfer reads the parent's `output`) — one level deeper.

**What this does and does not establish.**
- It does **not** show route A's hypothesis is false. The hypothesis — a child's cost
  depending on *where its parent went* — was never actually exercised at magnitude.
- It does show the term **as implemented cannot express that hypothesis**, and that
  `stateSize` is the wrong lever for it.

**What a real test needs:** the payload must divide by a **path** bandwidth between parent
and child (minimum link bandwidth along the route), not the child's local NIC — so that
distance carries magnitude rather than only a small additive latency. The machinery exists
(`NetworkFabric.route_links`, per-link bandwidth), but it requires the backbone enabled for
server↔server routes, which `build_core_backbone` now emits and `ROUTE_A_PILOT_V1_GRID`
does not yet turn on.

**Status of that first probe: NO-GO on that term, superseded below.** It was re-run against
a path-bandwidth term, as it said to.

---

### route_a_v1 — **NO-GO, now properly tested. Breaking separability is NECESSARY BUT NOT SUFFICIENT** (2026-08-25)

The probe above was rejected as a test of route A because the term's magnitude was
child-indexed. Both defects were fixed and it was re-run twice.

**Fix 1 — the payload is now pairwise.** `_payload_transfer_time` does store-and-forward
over the parent→child route (`n_hops × payload / bottleneck_bandwidth`), the same model the
ingress path already uses, instead of dividing by the child's own NIC. The preset now
requires a backbone, which yields **server↔server hop counts of 2–8** — so distance carries
magnitude, with a 4× spread between the nearest and farthest server pair.

**Fix 2 — `HEROSIM_OUTPUT_SIZE_BYTES`**, scaling the transfer payload *alone*.
`HEROSIM_STATE_SIZE_BYTES` moves `input` too, and `input` is a per-task storage read: at the
extreme arm it drove the episode to ~1000 s while the pairwise variation was ~4.6 s, burying
the coupled term **200:1**. Same error as the first probe, one level up.

**The properly isolated measurement** — baseline input, transfer payload 8/80/800 MB:

| payload | pairwise cost per remote parent | episode RTT | spread-plan regret (rtt / makespan) |
|---:|---:|---:|---:|
| 8 MB | 0.02–0.06 s | 10–100 s | 0.000% / 0.000% |
| 80 MB | 0.15–0.61 s | — | 0.000% / 0.000% |
| **800 MB** | **1.53–6.10 s** (4× by hop count) | 58–135 s | **0.000% / 0.000%** |

At the top arm the coupled term is roughly **10–30% of episode cost and varies 4× with the
parent/child pair** — and the componentwise minimiser is *still* exactly optimal, on every
dataset, both objectives. The scorer was also corrected mid-probe: it had been *dropping*
datasets where the componentwise plan is infeasible (the strongest coupling signal there
is), and now scores them by masked greedy decode, as a real pointwise scheduler would.

**The finding, and it is sharper than "route A failed":**

> **Non-separability is necessary but not sufficient.** `f_child(p_child, p_parent)` is
> genuinely pairwise here — the composition theorem's *hypothesis* is violated — and its
> *conclusion* still holds empirically. Breaking separability does not make the
> componentwise minimiser suboptimal. For that, the tasks' individually-best placements
> must **conflict**, so that one has to yield. Dependency + distance creates coupling
> without creating competition: every task can take its own favourite, and does.

That is why the five earlier co-location mechanisms and this one fail for the *same*
underlying reason, and it identifies what the program has never actually tried:
**contention for a scarce resource** — hard capacity, anti-affinity, exclusive GPUs — i.e.
§9's route B, which attacks the theorem's *free choice* hypothesis rather than its
separability hypothesis. The one hint already on record points the same way: the M3 pilot's
only non-collision-shaped escape (17.25% regret) came from the **distinct-node matching
constraint**, which is a feasibility restriction, not a physics term.

**Route B's stated caveat still applies and must be handled**: a grouped-argmax pointwise
decoder cannot represent a matching at all, so "GNN beats MLP under constraints" would be a
*decoder* result unless compared against a constraint-aware sequential pointwise decoder.

**Status: CLOSED — NO-GO.** Route A is tested and does not clear its pre-registered bar.
Do not soften the 5% threshold, and do not retry it with a larger payload: the term was
taken to 30% of episode cost with 4× pairwise variation and produced exactly zero. The
groundwork (DAG dispatch, fan-in, server mesh, path transfer, makespan channel) stands and
is reusable — route B needs all of it.

**Retro-audit against the mid-episode scale-down defect (2026-08-25, during route_b_v1
step (c)).** The defect that fix `42627d8` closes — `KEEP_ALIVE=30s` silently evicting a
task's FORCED replica before dispatch, truncating the sweep — was fixed 8 hours *after*
this NO-GO closed (`17792db` at 02:37 vs `42627d8` at 10:46), on the same physics
(locality on, DAG dispatch) that produces the truncation. This was a real, not
hypothetical, risk to the verdict above, and `score_route_a_scaling_probe.py` never
checked `placement_metadata.json`'s `sweep_complete` — only whether the file existed
(`n_missing_sweeps`) — so the original NO-GO could not have caught it either way. Checked
directly: regenerated 6 datasets from `ROUTE_A_PILOT_V1_GRID` at the 800 MB point, once
under `KEEP_ALIVE=30s` (route A's original condition) and once under the fix. **4 of 6
datasets were truncated under the original condition** (one losing 59% of its sweep,
33/56 rows) — confirming the defect was live during route A's own probe, not merely
theoretically possible. **Both corpora score identically: 0.000% spread-plan regret, 0
nonzero, NO-GO on both.** The truncation did not change the verdict on this rerun. This
is reassuring but is a 6-dataset spot-check, not a re-run of route A's original n≥200
scale corpus (which no longer exists on disk — gitignored, never persisted) — **the
NO-GO stands, now with the defect risk checked rather than merely disclosed, but the
original probe's own datasets were never re-verified because they no longer exist.**

---

### route_b_v1 — PRE-REGISTRATION (written 2026-08-25, before any route B corpus exists)

**The hypothesis.** Route A violated the composition theorem's *separability* hypothesis
and its conclusion still held: coupling without competition leaves every task free to take
its individual favourite. This lineage attacks the other hypothesis, **free choice**:
contention for a scarce resource, so that some task must yield. Mechanism: **node-memory
knapsack** — a plan is feasible iff every node's co-resident demand
`Σ memReq[task_type][platform_type] ≤ cap_node(α)`, with
`cap_node(α) = α × max single candidate demand on that node`. Demands are the welded
`task-types.json` values (type-asymmetric on GPU: dnn1/dnn2 0.9, rf 1.5, cnn 1.3);
**the file is not edited** — the scarcity knob is per-node capacity. Memory occupancy does
not change episode physics, so the constraint is applied to the full enumerated sweep **at
scoring time**; one corpus serves the whole tightness ladder, and stage-1 zero-diff is
structural. Stacked design, two arms differing in exactly one flag: **Arm S** (primary) =
diamond4 DAG, distinct types, server mesh + backbone, `HEROSIM_DATA_LOCALITY=1`, payload
800 MB (the point where route A measured the pairwise term at 10–30% of episode cost);
**Arm B0** = identical, locality OFF. Rationale: under competition with *separable* costs
any regret is decoder myopia and a perfect decoder erases it — only competition **plus**
coupling forces the score itself to be joint. B0's predicted-zero is the built-in
instrumentation control.

**Statistics** (`scripts_cosim/score_route_b_contention.py`, per dataset / α / objective):
`R_greedy` = feasibility-masked sequential greedy over min-marginals (deployable pointwise
scheduler) vs constrained sweep optimum. **`R_exact` (primary)** = feasible-set exhaustive
argmin of the min-marginal-sum surrogate `Σ_t m_t(p_t)` — on separable physics
`m_t(p) = c_t(p) + const`, so `R_exact ≡ 0` under ANY feasibility restriction; nonzero
constrained `R_exact` can be neither a decoder nor an LS-fitting artifact. Repairs =
`y ~ a + b·Σm + counts` (one-integer: node-occupancy excess sharing, the program's
established collision column; k-integer: per-node×type counts, the constraint's own
sufficient statistic), fit on the full sweep, **refused as `saturated` when rows < 2×params**
— never silently reported. Views: memory-feasible (primary), memory-feasible ∩ spread
(secondary). A full indicator-LS surrogate is a sensitivity row only: measured on the m3
pilot it fires ~12% even *unconstrained* (collision channel + argmin tie noise) where the
registered statistic measures the established 0.000% — it is not the gate.

**Deviation from the phase-1 plan text, recorded honestly and made before any route B
corpus existed:** the plan named the LS surrogate as `R_exact`'s fit; measurement on the
m3 pilot showed that statistic broken as a gate (12% unconstrained false-fire), and it was
replaced by the min-marginal-sum form, which is *stronger for the pointwise side* (exactly
optimal wherever physics is separable). Control 1's expectation was also re-derived from
395.45% to 450% when the rig arithmetic was corrected to min-over-totals marginals.

**Positive controls, frozen (`tests/test_route_b_positive_controls.py`, 12 tests, all
passing before this entry):** Control 1 (separable, hot-node cap, wrong-task yield):
`R_greedy = 450.000000%` exactly, `R_exact = 0`; cap removed → both 0. Control 2 (pairwise
matching-shaped costs, 3×3): `R_exact = R_greedy = 150.000000%` exactly, 1int repair
**cannot** clean it (150% in every LS branch), kint repair **refused as saturated** at rig
scale; cap removed → 0. The guard exists because the 4-row rig caught the scorer's first
version reporting interpolated repairs as 0.0 — kept as a regression test. **Any control
failure makes route B runs VOID, not NO-GO**, and controls re-run after any scorer edit.
Still owed before corpus scoring: the end-to-end rigged dataset through the real
generation→sweep→scorer path (predicted `R_greedy ≥ 50%`, cross-checked by an independent
reader of the produced placements.jsonl).

**Pre-probe — run 2026-08-25, route B SURVIVES.** Registered kill condition (at the
tightest non-degenerate α on the existing m3 pilot n=200: `R_greedy>1%` on <5% of datasets
AND max < 17.25%): does **not** fire — measured 7% firing with max **92.10%** (rtt) /
**157.81%** (makespan) at α=1.0; `R_exact` stays ≈0 (≤1.46% max, 1% firing), as the
theorem predicts on separable physics. The M3 matching hint amplifies 5×. Calibration
finding: α=1.0 leaves the free-choice plan infeasible in only 10% of datasets — the corpus
grid needs scarcer candidates (`per_client=0`, fewer server hosts) to reach the 30–70%
band. Frozen: `simulation_data/route_b_preprobe_{rtt,makespan}.json`.

**THE GATE — Arm S, registered tightness, n=200, both objectives scored, rtt primary:**
- **PASS** iff ALL of: (1) fraction of datasets with `R_exact > 5%` is ≥ 10% with the 95%
  binomial CI excluding 10% from below; (2) repair fraction < 0.5 for BOTH count repairs
  (medians over firing datasets, saturated repairs excluded and counted); (3) the firing
  fraction rises monotonically along ∞ → loose → tight; (4) the spread-view firing
  fraction is nonzero.
- **FAIL** iff the CI excludes 10% from above, or condition (2) fails (count-shaped ⇒
  sixth confirmation of the empirical rule; route B closed as a GNN argument).
- **VOID** iff any positive control fails, or Arm B0 fires materially
  (`R_exact > 1%` on > 2% of datasets — theorem says ~0, so that is instrumentation), or
  the CI straddles 10% → escalate n = 200 → 400 → 800; ladder exhausted →
  **VOID-UNDERPOWERED**, never FAIL. Arm-vs-arm comparisons go through
  `gate_statistics.paired_regret_comparison` / `pooled_phase4_verdict`.
- **Tightness two-step:** three α values chosen from the smoke corpus so "tight" makes the
  free-choice plan infeasible in 30–70% of datasets, then **frozen here before the n=200
  corpus is scored**. Zero-feasible datasets are counted (`no_feasible_rows`), never
  dropped.

**The thresholds above may not be revised after data exists. A near-miss is a FAIL or a
VOID per the rules; there is no third option.**

**Stage 2 (conditional on PASS), binding constraints registered now:** any "GNN beats MLP
under constraints" claim requires (a) ONE shared constraint-aware sequential
feasibility-masked decoder used by both models (scarcity-pressure order, single
implementation both models plug scores into); (b) the MLP arm at its strongest (dim25cr +
the k-integer features); (c) an exact-assignment decode arm on the MLP's scores. Labels
become any-of-K tied-optimal sets; `audit_label_provenance` gains a tie-tolerant mode; the
cache carries the feasibility mask + capacity map (one contract, sidecar rule). Grouped
argmax is not an arm. Scope exclusions: no edits to `task-types.json`, no episode-physics
changes beyond route A's landed term, no training/checkpoints/live gates/datalab in stage
1, no new `train_*.py` ever.

**Status: PRE-REGISTERED.** Outcome row to follow.

---

### route_b_v1 — calibration freeze + three registration amendments (2026-08-25, before any gated corpus exists)

**What the smoke (12 matched datasets per arm, designated calibration data in the
registered two-step) established, and two harness defects it caught first:**

1. **Mid-episode replica scale-down corrupted DAG sweeps and their substrate.**
   `KEEP_ALIVE = 30 s` evicts idle replicas; under Arm S physics parents run past 30 s,
   so children's *forced* replicas were scaled down before dispatch — 66–72/240 rows
   lost per dataset, nondeterministically (the unstable scale-down victim sort), with
   `sweep_complete: false` recorded and nothing reading it. Worse: the same eviction ran
   during the *warmup capture*, so the enumerator's candidate substrate itself varied
   with physics speed — Arm B0 and Arm S got different enumerations (270 vs 576 plans on
   the same seed) until fixed. Fixes: `cosim_keep_alive()` env override
   (`HEROSIM_COSIM_KEEP_ALIVE`, unset = bit-identical); workers now append tracebacks to
   `placement_errors.log` (preserved next to `placement_metadata.json` — a truncated
   sweep without its error log is undebuggable); the route B scorer **refuses** truncated
   or metadata-less sweeps. Both corpus arms generate with the override set; enumeration
   counts verified identical across arms on all 12 smoke seeds.
2. **The smoke result itself, matched arms:** Arm B0 `R_exact` max 2.07% (the known
   collision/link residue; `R_greedy` up to 3578% — greedy myopia under scarcity is
   catastrophic but decoder-shaped). Arm S `R_exact` max **42.0%**, firing 25–33% at the
   >5% level, count repairs closing **nothing** — the joint signature the lineage
   predicts, 20× the B0 residue. Makespan channel fires too (max 19.6%).

**Amendments, each disclosed with what had been seen when it was made.** No gated
(n=200) data exists; the smoke's 12 datasets/arm had been scored. The PASS fraction
(≥10% at `R_exact > 5%`), the magnitude bar, the repair threshold (<0.5), α ladder
values, and the power ladder are all UNTOUCHED from the blind registration.

- **(A1) Arm B0 VOID trigger, was: `R_exact > 1%` on > 2% of datasets.** Premise error,
  visible in the registration's own text ("separable costs ⇒ surrogate = truth"): a
  backbone corpus is NOT separable — the collision channel and the link channel are
  real, known, count-shaped-or-thin couplings that produce exactly the 1–2% B0 residue
  measured. As registered, VOID would trip on real physics, not instrumentation.
  **Now: VOID iff B0 shows `R_exact > 5%` (the material bar) on > 2% of datasets.**
  Direction: loosens a validity check, does not touch the claim gate.
- **(A2) Tightness calibration, was: "tight" = free-choice plan infeasible in 30–70% of
  datasets.** Unsatisfiable: the α response is cliff-shaped (0.92 → 0.00 between α 3.2
  and 3.4 on the smoke) because CPU demands are near-equal; no α lands in the band.
  The band was a proxy for "binding but not degenerate" — replaced by the direct
  criteria: `no_feasible_rows = 0`, `greedy_stuck = 0`, mean feasible rows ≥ 50, and
  cw-infeasible ≥ 30%. **Frozen ladder: α ∈ {∞, 3.0 (loose), 2.0 (tight)}, tight = 2.0
  primary.** On the smoke: cw-infeasible 0.75–1.00, feasible rows 388–584, zero stuck,
  zero empty at both binding rungs.
- **(A3) PASS condition 3, was: firing fraction rises monotonically ∞ → loose → tight.**
  Two defects: (i) transplanted from route A, where the lever scaled a physics term —
  here the lever restricts a feasible set and the within-binding-regime gradient is
  flat/noisy (smoke: 0.33 at loose vs 0.25 at tight — one dataset's difference at
  n=12); (ii) as registered, "conditions 1,2,4 hold but 3 fails" lands in NONE of
  PASS/FAIL/VOID — an undefined outcome cell. **Now: (3′) the unconstrained rung fires
  `R_exact > 5%` on < 2% of datasets AND each binding rung fires above the unconstrained
  rung** — the free-choice attribution the condition was always meant to capture.
  Disclosed plainly: this amendment was made after seeing the 12-dataset smoke values;
  a reader may discount condition 3′ accordingly. Conditions 1, 2, 4 stand as
  registered blind. Outcome-cell closure, fixed before any corpus scoring: if the CI
  clears the PASS bar but condition 3′ or 4 fails, the verdict is **FAIL** with the
  failed condition named — the effect exists but is not attributable as registered;
  there is no fourth outcome. `score_route_b_gate.py` implements exactly this mapping
  and takes no threshold arguments.

**End-to-end control, form finalized:** the registered "rigged dataset" is superseded by
something stronger — `verify_route_b_scorer_agreement.py`, an independent from-scratch
recomputation of `R_greedy` and `R_exact` from `placements.jsonl` (no imports from the
scorer), which must agree within 1e-9 on **every** corpus dataset; plus the measured fact
that the real generation → sweep → scorer path fires at rig-scale magnitudes on the smoke
(`R_exact` 42%, `R_greedy` 151% on Arm S) — the "predicted ≥ 50% end-to-end fire" is
satisfied by measurement. Any verifier disagreement ⇒ VOID.

**Status: CALIBRATION FROZEN.** Next: zero-diff proof, corpus generation (2 arms ×
n≈200, same seeds, env-matched keep-alive), verifier, gate.

---

### route_b_v1 — OUTCOME: **PASS (stage 1).** Contention + coupling produces the non-pointwise structure five mechanisms and route A could not (2026-08-25)

**The registered row** (`score_route_b_gate.py`, no threshold arguments; Arm S, tight
α=2.0, rtt, n=204, zero truncated sweeps, enumerations bit-matched across arms):

> **35/204 = 17.2%** of datasets with `R_exact > 5%`, Wilson 95% CI **[0.126, 0.229]**
> — excludes 0.10 from below (condition 1 ✓). Median repair fraction **0.000** for BOTH
> the one-integer excess-sharing column and the k-integer per-node×type count vector,
> over all 35 firing datasets, none saturated (condition 2 ✓). Attribution:
> **0/204 fire unconstrained**; both binding rungs fire at 0.172 (condition 3′ ✓).
> Spread view: 14/109 firing — not collision-channel-only (condition 4 ✓). Arm B0
> validity: **0/204** above the material bar (max 2.49%) ✓. Independent verifier:
> 612 + 612 (dataset, α) cells agree to 1e-9 ✓. Positive controls 13/13 ✓.
> **VERDICT: PASS.**

**Full cell table** (`frac(R_exact > 5%)` / max `R_exact`):

| arm | objective | α=2.0 (tight) | α=3.0 (loose) | ∞ |
|---|---|---|---|---|
| **S** (coupling+competition) | rtt | **0.172** / 53.5% | 0.172 / 48.7% | 0.000 / 0 |
| **S** | makespan | 0.162 / 32.1% | 0.118 / 27.4% | 0.000 / 0 |
| **B0** (competition only) | rtt | 0.000 / 2.5% | 0.000 / 2.0% | 0.000 / 0 |
| **B0** | makespan | 0.000 / 1.6% | 0.000 / 1.7% | 0.000 / 0 |

**What this establishes.**
1. **The composition theorem's free-choice hypothesis is the load-bearing one, and
   violating BOTH hypotheses at once is what creates structure.** Under the memory
   knapsack + the 800 MB pairwise transfer, the best additive surrogate *with a perfect
   decoder* is suboptimal by up to 53% on 17% of datasets — a target no pointwise
   scorer can express regardless of decode.
2. **The effect is not LINEARLY count-shaped.** *(Amended in place 2026-08-25 by the §9b
   block ablation — see the correction paragraph below. The original text read "The
   effect is not count-shaped … does NOT extend here", which is wrong as written.)* The
   empirical rule that killed five co-location mechanisms ("every escape collapses to an
   occupancy integer") does not extend here **in its linear form**: the constraint's own
   sufficient statistic (per-node×type counts), entered linearly, repairs a median of
   exactly nothing. (Honest detail: the k-integer repair does pull ~1/3 of firing
   datasets under the 5% bar — 0.172 → 0.118 — but the median closure is 0.000 and the
   registered condition is decisive.)

   **The correction.** kint is *linear* in the counts. Adding the per-type quadratic
   co-residency sums Σ_t occ_{node(t)}[k] — the SAME statistic, entered nonlinearly, with
   no parent-placement or network columns at all — takes the median closure from 0.000 to
   **0.843**, and adding load/cap and the over-cap count takes it to **0.892**. So the
   occupancy rule DOES extend; a linear repair simply could not see it, and reporting
   "not count-shaped" on the strength of a linear fit was an overreach. What the stage-1
   PASS actually established is narrower and still stands: *the linear* count repair
   closes nothing, which is what registered condition 2 tested and what the 17.2% firing
   rate is measured against. Neither the PASS nor any of its four gate conditions moves —
   condition 2 was registered with the 1int/kint linear repairs and both still close a
   median of 0.000. Reproduced by `route_b_coefficient_transfer.py` (arms `kint`,
   `kint+quad`, `occupancy`), independently recomputed by
   `verify_route_b_scorer_agreement.py --check-blocks`, 315/315 arm-values to 1e-9.
3. **Competition alone is not sufficient either — the stacking argument was right.**
   Arm B0's score-side structure never crosses 2.5%, while its *greedy* regret reaches
   3578%: scarcity without coupling produces only decoder-shaped error, which a better
   decoder erases. Coupling decides *who should yield*; that is the graph question.
4. Both objectives fire; makespan is slightly weaker (0.162/0.118) but the same shape.

**What is NOT established, stated before anyone asks.** No model has been trained;
nothing here says a GNN can *learn* this structure, and nothing compares GNN to MLP —
that is stage 2, valid only with the registered decoder discipline (one shared
constraint-aware sequential masked decoder, dim25cr+k-integer MLP arm, exact-assignment
decode arm). Nothing about live serving. One topology family (6 servers, per_client=0,
diamond4 over dnn1/dnn2/rf/cnn), one demand table (the welded task-types.json), one
frozen α ladder. The 17.2% firing fraction is a property of this grid, not a universal
rate.

**Falsified along the way:** the B0-as-separable premise in the original registration
(a backbone corpus carries the collision + link channels; amendment A1); the 30–70%
tightness band (cliff-shaped α response; A2); the monotone-in-α firing condition (A3);
and the first scorer's LS-surrogate `R_exact` (12% false-fire unconstrained) plus its
unguarded repair fits (Control 2 caught interpolation at rig scale).

**Artifacts.** Corpora (local, gitignored): `gnn_datasets_dag4_route_b_pilot_v1_arm_{s,b0}`
(204 each; regenerable from `ROUTE_B_PILOT_V1_GRID` seeds 901–917 with
`HEROSIM_COSIM_KEEP_ALIVE=1000000 HEROSIM_RETAIN_TASK_TIMES=1`, Arm S adding
`HEROSIM_DATA_LOCALITY=1 HEROSIM_OUTPUT_SIZE_BYTES=800000000`). Frozen reports:
`simulation_data/route_b_pilot_v1_arm_{s,b0}_{rtt,makespan}.json`,
`route_b_preprobe_{rtt,makespan}.json`. Tools: `score_route_b_contention.py`,
`score_route_b_gate.py`, `verify_route_b_scorer_agreement.py`,
`tests/test_route_b_positive_controls.py` (13 tests).

**Post-PASS scrutiny (2026-08-25, same day, before anyone else read the result): a
clean confirmation gets the same suspicion a clean zero got all session.** Four checks,
run against the standing worry that a result matching the hypothesis this precisely is
exactly the one nobody feels the urge to re-check.

1. **Gate condition 2 (repairs close nothing) independently reconfirmed, with an
   honest caveat found along the way.** Extended `verify_route_b_scorer_agreement.py`
   with `--check-repairs`: a from-scratch LS fit (pure Python, no numpy — a hand-rolled
   Gaussian-elimination normal-equations solve) recomputing both repairs directly from
   each dataset's files. First run disagreed with the scorer on one dataset (10.6% vs
   42.0%) — traced to normal equations squaring the design matrix's condition number
   across columns of wildly different scale (intercept, RTT-magnitude sums, 0/1 counts);
   fixed by standardizing columns before solving (confirmed against numpy/SVD:
   coefficients now agree to 1e-13). One dataset (`ds_00008`, Arm S, α=2.0) still
   disagrees even after the fix — traced *at the time* to a "genuine near-tie": 4+
   feasible plans with materially different true costs (78.1s, 60.8s, 58.2s) predicted
   equal to ~13 significant figures by the fitted surrogate.

   > **RETRACTED 2026-08-25 (same day, later session).** This was **not** a genuine tie
   > and was not real: it was an artifact of *two* verifier bugs compounding — the
   > standardized-normal-equations solver still not reaching the true LS optimum on the
   > wider t1 matrix, and the verifier's 1int column computing `max` node-occupancy
   > excess where the registration says `sum`. Each masked the other. With the MGS-QR
   > solver and the correct column, scorer and verifier agree outright on `ds_00008` and
   > **all 612 repair values agree with zero tie-acceptances**. The original sentence
   > below — "This is real, not an artifact" — was wrong, and is struck rather than
   > silently deleted, because it was recorded here as an established finding and read
   > that way. Detail in the stage-2 pre-probe entry's defect 2. (Prediction ties DO
   > occur in this machinery and are real where they occur — see §9b's tie bands, where
   > 22/35 firing datasets tie at the full-T1 argmin — but `ds_00008` under the registered
   > columns was not one of them.)

   ~~**This is real, not an artifact — reported, not hidden.**~~
   Independently recomputing repair_fraction for **all 35 firing datasets from scratch**
   (ignoring the scorer's numbers entirely): **median 0.000 for 1int, median 0.000 for
   kint** (kint mean 0.357, max 1.0 — a few datasets ARE fully repaired by kint, but the
   *median*, the registered statistic, is exactly what the scorer reported). **Gate
   condition 2 holds under independent, from-scratch recomputation.**
2. **The amendment to condition 3 is disclosed with its own counterfactual, not just
   its rationale.** Route A's own precedent for "rising" (`score_route_a_scaling_probe.py`
   `rising = means[-1] > means[0] + 1e-9`, endpoints only) would read the observed
   sequence 0.000 (∞) → 0.172 (loose) → 0.172 (tight) as rising (0.172 > 0.000) — **the
   original wording, read consistently with the one existing precedent in this
   codebase, would ALSO pass.** Read literally step-wise (every adjacent pair strictly
   increasing), the flat loose→tight step (0.172 = 0.172) would **fail** it. Both
   readings are stated because the wording is genuinely ambiguous and the amendment was
   made after seeing this exact number — a reader is free to prefer either. What is not
   ambiguous: no reading of the original condition, applied honestly, changes the
   PASS verdict, because it was never the swing condition — condition 1 (the CI) and
   condition 2 (repairs) carry the result.
3. **Firing rate reported above the B0 noise floor, not just against the 5% bar.**
   B0's own residue tops out at 2.49% (α=2.0) — a floor under every Arm S number. Arm S
   firing fraction at `R_exact >` 5.0% / 7.5% / 10.0%: **0.172 / 0.157 / 0.123.** The
   effect does not thin out approaching a threshold four times the B0 floor; at >10% it
   still clears the registered 10% PASS bar on its own. Not floor-sensitive.
4. **Alpha provenance.** The frozen ladder (α ∈ {∞, 3.0, 2.0}) was calibrated on route
   B's own smoke corpus (`gnn_datasets_dag4_route_b_smoke_{s,b0}`, 12 datasets/arm on
   `ROUTE_B_PILOT_V1_GRID`, matching the gated corpus's topology and physics exactly),
   not on the unrelated m3 pilot — see amendment A2 above. The freeze commit
   (`2c3ebbc`… through the calibration-freeze entry) predates the n=204 generation run.
   Realised componentwise-plan-infeasible fraction at the frozen tight rung: 0.44–0.50
   across the corpus (table above) — comfortably binding, not degenerate.

**Route A cross-reference.** Arm S's unconstrained cell (α=∞: locality on, 800 MB
payload, DAG dispatch, on the keep-alive-fixed harness) is physics-adjacent to route
A's own condition but **not a re-run of route A's grid** (route B's grid uses 6 servers /
`per_client=0`; route A's used a different server count and replica config) — it should
be read as corroborating evidence at n=204, not as route A's own probe repeated. The
literal re-verification of route A's condition is the 6-dataset retro-check recorded in
`route_a_v1` above, which is the one that actually reused route A's grid.

**Status: PASS — stage 1 CLOSED, and re-checked. Stage 2 (can a GNN learn it and beat the
constraint-aware pointwise baseline?) requires its own pre-registration before any
training run.**

### route_b_v1 — stage 2 pre-probe zero: **NO-GO-PREPROBE-T1** (2026-08-25)

Stage 2's pre-registration was drafted (`ROUTE_B_STAGE2_PREREGISTRATION.md`, this commit)
and, on review, the user identified its load-bearing hole before sign-off: the registered
strongest-MLP arm (T1 = dim25cr + k-integer + partial-assignment state, including
parent-placement/hop/transfer columns) is **not a pointwise baseline** — its plan-level
score is non-separable, and stage 1's `R_exact` (a *separable* surrogate) and its
median-0.000 count-repair result say nothing about it. The doc's §9a registered an offline
kill test with the reading fixed **before** the number existed: recompute `R_exact` on the
stage-1 204 (Arm S, α=2.0, rtt) with the surrogate augmented by the full T1 plan-level
column set — kint + per-type quadratic co-residency + load/cap + over-cap count + min/max
parent-hop sums + `Σ_edges hops/bottleneck` + `Σ_edges latency` + same-node-parent count,
the last three computed from each dataset's own `link_topology.routes`, i.e. exactly what
`_dependency_transfer_time` charges (uniform 800 MB payload absorbed by the LS
coefficient, so the columns **span the charged coupling term exactly**). Registered
reading: median repair fraction ≥ 0.5 over the stage-1 firing datasets ⇒ the architecture
claim is pre-falsified and stage 2 does not run as registered.

**Result: median T1 repair fraction 1.000** (mean 0.730; 26/35 firing datasets closed
≥ 0.5; kint comparison: median 0.000, mean 0.357, matching the stage-1 scrutiny to the
digit). `frac(R_exact > 5%)` falls 0.172 → **0.054** (11/204 residual, max 22.2%).
Attribution ablation over the firing 35: the parent-coupling block alone closes at median
1.000, the occupancy block alone at median 0.892 — two largely redundant routes to the
same closure.

> **AMENDED 2026-08-25 by §9b, which put this ablation in code for the first time.** Both
> numbers reproduce exactly (1.000 and 0.892), but "two largely redundant routes" is
> wrong: **both blocks contained `kint`**, and `kint` is the shared ingredient. Membership,
> now unambiguous — occupancy = `kint+quad+cap`, parent-coupling *as originally run* =
> `kint+hop+coupling` (PREREG:406's parenthetical "kint + cols 33–35 analogues" was
> literal). Stripped of `kint`, the parent block alone closes only **0.392**, and `quad`
> alone closes **0.000**. The honest decomposition is that neither block is a route on its
> own: `kint` alone closes 0.000, and it is `kint` *combined with* either the quadratics
> (0.843) or the parent columns (1.000) that closes the effect. When this prose was
> written, no committed code computed it — the fitted coefficients were discarded at every
> solver call site — so it could not be checked. It can now:
> `route_b_coefficient_transfer.py`, verified 315/315 arm-values to 1e-9.

Scorer: `score_route_b_contention.py` (`t1` repair, constrained rungs only);
report frozen at `simulation_data/route_b_stage2_preprobe_t1_rtt.json`. Verification:
`verify_route_b_scorer_agreement.py --check-repairs` agrees on **all 204 cells and all
612 repair values (1int, kint, t1), zero tie-acceptances**.

**Two verifier defects found and fixed en route — both matter to the stage-1 record:**

1. **The pure-Python solver did not reach the true LS optimum on the wider t1 matrix.**
   Standardized normal equations (the stage-1 scrutiny's own fix) produced fitted values
   diverging from the unique LS projection on ds_00008 (fitted values on fit rows are
   solver-independent, so this is a numerical failure, not a tie). Replaced with
   hand-rolled MGS QR (one re-orthogonalization pass, dependent-column dropping) — still
   no numpy, still zero scorer imports; verified against numpy to 1.6e-13.
2. **The verifier's 1int column was `max` node-occupancy excess where the registration
   (and `separability_diagnostic._excess_sharing`) says `sum`.** A real bug, masked by
   defect 1: the imprecise fits happened to argmin onto the same plans on every dataset
   previously checked. The QR solver exposed it (ds_00019: scorer 1.036 vs
   verifier-with-max 9.633). With both fixes, **ds_00008's recorded "genuine
   floating-point-level tie" dissolves** — scorer and verifier now agree outright there;
   that scrutiny interpretation is superseded (the disagreement was the verifier's wrong
   column plus solver imprecision, not an inherent tie). Stage 1's gate verdict is
   untouched (the gate consumed the scorer's statistics, which were correct and are now
   re-verified under the fixed verifier), but the stage-1 claim "1int independently
   confirmed" was, until today, confirmed against a different column definition. It is
   now actually confirmed: 612/612 repair values agree.

**What this establishes:** on this grid, a pointwise score given partial-assignment state
(exactly what a sequential masked decoder exposes for free) is sufficient at the
surrogate-expressiveness level to close the median stage-1 firing dataset completely. The
"GNN beats strongest-MLP under constraints" claim is pre-falsified before any cache,
decoder, model, or corpus was built — for the price of one scorer run. **What is NOT
established:** the T1 repair is a per-dataset LS fit on the dataset's own sweep — an
expressiveness upper bound, not a trained cross-dataset model; whether a *trained*
pointwise-plus-state model realizes this bound is the reduced V5-shaped question
("decoder-state features suffice — no graph needed"), which needs no GNN and its own
registration if pursued. The 11-dataset residual stratum (5.4%, below stage 1's 10% bar,
max 22.2%) is real but does not clear the program's own materiality standard. The α=∞
rung's 0/204 remains corroborating evidence for route A's conclusion at n=204 (not a
literal grid re-run — see the stage-1 outcome entry's caveat).

**Status: PROVISIONAL — NO-GO-PREPROBE-T1 was RETRACTED AS MEASURED on 2026-08-25 by §9c
(entry below). Do not read this entry as settled.** The T1 column set used here includes
`kint`, one free coefficient per `(node, task_type)` — an identity-indexed per-dataset
lookup table with **no corresponding column in the registration's own §2 `dim36crk`
table**. So the kill test was run with a surrogate strictly more expressive than the T1 arm
it stands in for. Stripped of that block, the closure of the actually-registered feature set
is 0.392–0.648 depending on a tie rule §4 never specified, and §9c(a) measured the block's
coefficients to be unrecoverable from node features (held-out R² 0.014). **Stage 2's
architecture question is reopened and requires re-registration with a corrected T1
definition before anything is built.** The one thing this entry establishes unconditionally
is the reverse-direction result: whatever closes the effect, it is *not* message passing
that is needed — see §9c's exploratory pooled `krank` (0.790 under one coefficient set).

---

### route_b_v1 — STAGE 2 PRE-REGISTRATION (recorded retroactively 2026-08-25)

`ROUTE_B_STAGE2_PREREGISTRATION.md` at commit `df9971e` is the registration under which
the §9a pre-probe's reading was fixed before its number existed. Its own header (line 6)
required this row and it was never written — recorded now, late, rather than left absent.
§9b was added to that file 2026-08-25 (this commit) and is disclosed as a post-outcome
deviation in its §11.

---

### route_b_v1 — §9b coefficient transfer: **VOID-KINT-CONFOUNDED**, and the §9a bound is confirmed tie-robust (2026-08-25)

**The question.** §9a's T1 repair fits fresh coefficients on *every dataset's own sweep*;
a trained cross-dataset model gets **one** set. So NO-GO-PREPROBE-T1 rests on a bound that
may not transfer. Registered in `ROUTE_B_STAGE2_PREREGISTRATION.md` §9b before the number
existed, with three cells so that "cost of dropping kint" could never be confused with
"cost of pooling", and with the VOID condition written in advance.

**The obstruction, found while designing and stated before measuring:** `kint` **cannot be
pooled at all.** Its columns are one per `(node, task_type)` pair *in that dataset's own
demand*, so vocabulary and width both vary (K ∈ 8…13, X widths 21–26 over this corpus).
There is no cross-dataset coefficient vector to fit.

| cell | fit | median | tie-band | ≥0.5 |
|---|---|---|---|---|
| A | per-dataset, full T1 | **1.0000** | **[1.0000, 1.0000]** | 26/35 |
| B | per-dataset, T1 − kint | 0.3922 | [0.3922, 1.0000] | 17/35 |
| C | **pooled**, T1 − kint | 0.0000 | [0.0000, 1.0000] | 16/35 |
| C′ | pooled, equal dataset weight (sensitivity) | 0.0000 | [0.0000, 1.0000] | 15/35 |

**VERDICT: VOID-KINT-CONFOUNDED**, the registered branch — cell B is already below 0.5, so
cell C cannot be read as a test of *pooling*: the drop is attributable to dropping `kint`,
which no single coefficient set can carry anyway. **§9b does not weaken NO-GO-PREPROBE-T1
and does not strengthen it. The V5 question stays open and stays empirical.**

**What §9b did establish, and it is the more useful half:**

1. **The §9a statistic is tie-robust, which nobody had checked.** Cell A's median is 1.0000
   whether prediction ties at the argmin are resolved optimistically, pessimistically, or
   by the registered plan-key tie-break — even though **22/35 firing datasets do tie**
   (max group 8). A NO-GO resting on a tie-break would have been worth exactly as much as
   `ds_00008`'s retracted "genuine tie". It does not.
2. **Cell B, by contrast, is genuinely indeterminate** — band [0.392, 1.000] straddles the
   0.5 threshold, with ties up to 16 plans wide. This is not float noise: stripped of
   `kint`, the 9 node-agnostic columns **cannot separate up to 16 feasible plans at all**.
   That is the finding, not a nuisance. The independent verifier surfaced it first, as
   three "TIE (accepted)" lines on cell B where scorer and verifier picked different plans
   from the same tied group (0.000 vs 1.000) — a disagreement that is real and that the
   band now reports as first-class output rather than a footnote.
3. **The block attribution now exists in code** and both prose numbers reproduce — but
   their interpretation was wrong; see the amendment in the §9a entry above and the
   in-place correction of stage-1 finding #2.

| arm | blocks | median | ≥0.5 | residual >5% |
|---|---|---|---|---|
| kint | linear counts | 0.0000 | 13/35 | 24 |
| quad | quadratic counts only | 0.0000 | 7/35 | 32 |
| **kint+quad** | **counts, nonlinearly** | **0.8429** | 23/35 | 15 |
| occupancy | kint+quad+cap | 0.8924 | 24/35 | 14 |
| parent-coupling | hop+coupling | 0.3922 | 17/35 | 22 |
| parent-coupling incl kint | kint+hop+coupling | 1.0000 | 27/35 | 10 |
| full T1 | all five | 1.0000 | 26/35 | 11 |

**Coefficients (descriptive, as registered — the repair fraction is the decisive statistic
and this is not).** The pooled `transfer` coefficient is 330.96 against the registered
physical prediction of 762.939453125 (`800e6 / 1024²`), and pooled `latency_sum` is −38.4
against a predicted 1.0. Per-dataset dispersion is enormous (transfer mean −277, sd 8106).
**None of this is evidence about the physics**, and the registration said so in advance:
the cell-B/C fits are mis-specified by construction (they omit the block that does the
work), 9/35 per-dataset designs are rank-deficient, and `same_node_edges = 4 −
remote_edge_count` is collinear with the hop block on `diamond4`. Recorded because it was
registered, and because a *correctly specified* pooled fit would be the place to test the
762.94 prediction properly.

**Re-derived 2026-08-25 for the 8-task probe.** The "4" here is `diamond4`'s total edge
count (4 parent-child pairs), not a hardcoded constant in the scorer — `score_route_b_contention.py`'s
`same_node_edges`/`transfer` loop (`fn`, around line 596) sums over `parents_of` for
whatever edges the plan's DAG actually has, so no code change is needed. For two diamond4
instances co-decided in one episode (8 tasks, 8 edges total, 4 per instance), the identity
becomes `same_node_edges = 8 − remote_edge_count` and the collinearity with the hop block
persists at the new constant — the mechanism (total edge count is fixed per dataset, so
same-node and remote edge counts are complementary) is unchanged by doubling, only the
number is.

**Verification.** `verify_route_b_scorer_agreement.py --check-blocks` — an independent
pure-Python/QR recomputation from each dataset's raw files — agrees on **315/315 (dataset,
arm) repair fractions to 1e-9** across 35 datasets and 9 arms, with the three cell-B tie
acceptances described above. Cell A reproduces §9a exactly (median 1.0000, mean 0.7302,
26/35, 11 residual). The verifier itself is, as of this commit, backstopped by
`tests/test_route_b_repair_fixtures.py`: 16 closed-form fixtures (29 with the positive
controls) covering `solve_least_squares` against textbook OLS, the `sum`-vs-`max` 1int
distinction that survived three rounds of checking, the t1 columns hand-computed on a
4-node toy, the scorer/verifier cap-convention divergence on an uncapped node, and the
saturation guard. Refactor safety: `t1_cols`'s new block registry is proven **byte-identical**
on the frozen §9a report (204 datasets × 3 α).

**Residual stratum (11 datasets, DESCRIPTIVE — 5.4% is below the program's 10% materiality
bar and this is not a claim).** No separating structure found. Medians, residual vs closed:
feasible fraction 0.643/0.643, distinct nodes in the optimum 2/2, same-node edges 2/2,
max load/cap 0.921/0.921, kint width 11/11. The only gaps are small and in the direction
you would expect from their larger regret: R_exact 16.8 vs 14.7, transfer in the optimum
0.0140 vs 0.0100, hop sum 11 vs 10, RTT spread CV 0.197 vs 0.175. **There is no "a graph is
needed when X" sentence here** — at n=11 with no separating feature, the residual reads as
the tail of the same distribution, not a distinct stratum. The edge closes cleanly.

**Artifacts:** `simulation_data/route_b_coefficient_transfer.json`;
`scripts_cosim/route_b_coefficient_transfer.py`; `--check-blocks` in the verifier;
`tests/test_route_b_repair_fixtures.py`.

**Status: §9b returned VOID on its own question — whether the bound survives one coefficient
set is NOT answered by this method, because the block carrying the closure is not poolable.
Superseded in part by §9c below, which asked why that block was in the column set at all.**

---

### route_b_v1 — §9c: `kint` is not a T1 feature — **NO-GO-PREPROBE-T1 RETRACTED AS MEASURED** (2026-08-25)

**The objection, against §9a itself.** §9a's T1 set includes `kint`: one free coefficient per
`(node, task_type)`, a per-dataset lookup table over node **identities**. **No column of the
registration's own §2 `dim36crk` table is identity-indexed** — cols 25–28 are per-type
occupancy on *the candidate's own node*: anonymous, fixed width, four columns. §2's verbatim
rule cuts both ways, and a feature the MLP cannot have must not be credited to it. So §9a's
kill test may have been run with a surrogate strictly more expressive than the arm it stands
in for. Registered in `ROUTE_B_STAGE2_PREREGISTRATION.md` §9c before either number existed.

**The load-bearing observation, and it needed no new code.** The scorer's `quad` block is
*exactly* the plan-level rendering of cols 25–28:
`quad[k] = Σ_n tot[n]·occ[n][k] = Σ_t occ_{node(t)}[k]`. Likewise `load_over_cap` = col 29,
`overcap_tasks` = col 31, `min/max_hop_sum` = 33–34, `transfer` = 35. **T1 − kint is
precisely the dim36crk-expressible set — so §9b's cell B already WAS the anonymous closure
measurement**, at 0.392. `kint` is the only T1 block with no §2 column.

| measurement | result | registered reading |
|---|---|---|
| **(a)** kint coefficients regressed on node features, held out by dataset | **R² = 0.0138** (in-sample 0.0974) | < 0.5 ⇒ **identity-memorized; §9a does not bound the T1 arm** |
| **(b)** anonymous (dim36crk) closure | `mean_tied` **0.648** vs `registered`/`pessimistic` **0.392** | directions disagree ⇒ **VOID-TIE-INDETERMINATE** |

**(a) is decisive and (b) is a specification gap.** On (a), the in-sample R² is the telling
figure: node features barely explain these coefficients even without a generalization gap,
so it is not a small-sample artifact — the block is genuinely a per-dataset identity lookup.
On (b), the readings disagree because tie groups run up to **16 plans wide** and the
sorted-plan-key rule lands *worse than an average tie-break* on this corpus. **§4's decoder
never specified what to do with tied scores, and the anonymous verdict flips on that choice.**
That is a real hole in the registration, not a numerical nuisance.

**A second registration defect, found the same way (recorded 2026-08-25):** §4 pinned
"scarcity-pressure order" to `greedy_masked_plan`'s ascending `(best available marginal,
task_id)`. On this corpus that order does not exist: in **all 204 datasets the four
min-marginal minima are exactly tied** — every task's best placement lies in the globally
best plan, so `min_p m_t(p)` equals the global minimum RTT for every task — and the tie-break
collapses the order to `task_id`, which is the DAG's topological order. Measured
consequences: **0 of 816 DAG edges decode child-before-parent** (§2's hedge that "parents are
not guaranteed to precede children" never fires) and **0 of 816 steps** have a task's best
choice already taken by an earlier task; only capacity ever blocks the top choice, on
167/816 = 20.5% of steps. The registered order therefore carries no scarcity information
whatsoever. This is the same class of hole as the unspecified tie rule: a registration naming
a discriminator that is constant on its own corpus. **A corrected stage 2 must fix both**, or
it repeats the error under a new name.

Also measured: **T1 ≡ T0 at decode step 0** (all eleven partial-state columns are zero when
nothing is placed), and the prefix-oracle curve (7.78 → 9.84 → 1.98 → 0.31) puts essentially
all decoder myopia in the first two of four steps. **Write-up rule, binding: wherever the
four-task limit is doing the work, the sentence is "the corpus is too small to test the
architecture claim", never "the architecture claim is false."** These two measurements
support the first reading, not the second.

**Consequence: NO-GO-PREPROBE-T1 is retracted as measured, and stage 2's architecture
question is reopened.** Per §9c this is explicitly **not** a licence to start the build queue:
the corrected T1 definition — including a tie rule — gets re-registered first. §9a's purpose
was to kill cheaply; a corrected §9a that fails to kill changes the registration, not the
discipline.

**Exploratory (NOT registered, NOT independently verified, no verdict read from it) — and
it is why the reopening may be short.** Replacing `kint` with `krank`, occupancy indexed by
identity-free node **rank** (ascending capacity, then mean hop, padded to a common width):

| arm | median | ≥0.5 |
|---|---|---|
| krank + dim36crk, per dataset | **1.000** | 26/35 |
| **krank + dim36crk, ONE pooled coefficient set** | **0.790** (mean_tied 0.824) | 20/35 |

So the closure never needed node *identity* — it needed per-node occupancy **resolution**,
which dim36crk's four candidate-local columns do not supply. And unlike `kint`, `krank`
pools: a single coefficient set over identity-free columns closes the median firing dataset
at 0.79, with no message passing. **That is the follow-up the §9b VOID named, and it points
where §9a did: the structure looks reachable by a pointwise scorer, just not by the one
stage 2 registered.** A hypothesis for the corrected registration, not a result.

**Verification.** Cells and all ablation arms: `--check-blocks`, 315/315 to 1e-9. **The §9c(a)
regression and both `krank` arms are single implementations and are NOT independently
verified** — stated rather than implied. Gauge note: within a dataset the `kint` columns for
a given type sum to exactly 1 (each type has one task in `diamond4`), so the fit is
rank-deficient by one dimension per type and the coefficients are defined only up to a
per-type shift; `lstsq` returns the minimum-norm representative, a convention. (a) therefore
scores coefficients **centered within (dataset, task_type)**, the gauge-invariant content.

**Status: NO-GO-PREPROBE-T1 retracted as measured. Stage 2 REOPENED, pending re-registration
with (i) a T1 definition that either justifies identity-indexed columns or replaces them with
an anonymous per-node-resolution block, and (ii) a decoder tie rule. Nothing is built until
that registration exists. The 8-task probe of the exploratory pooled result is §9d below.**

### route_b_v1 — §9d: 8-task probe — pooled `krank` closure SURVIVES the doubling, attenuated (2026-08-26)

**The question.** §9c's exploratory pooled result — ONE identity-free coefficient set
(`krank` + dim36crk) closing the median firing dataset at 0.790 — was measured on 4-task
episodes, where the joint decision is small enough that a lookup-table-shaped fit is cheap.
Does it survive doubling the joint decision to 8 tasks (2 `diamond4` DAG instances per
episode, independently drawn client nodes, byte-identical infrastructure per index)?

**Corpus.** `gnn_datasets_dag4_route_b_pilot_v1_8task`, 204 datasets, generated on datalab
(job 713673): 204/204 complete, 0 silent skips, 0 truncated sweeps, 0 worker failures; sweep
sizes 27,648–516,096 (~41M sims, ~23 GB). Venue measured not to be a variable twice over:
the 4-task identity gate (job 713654, 16/16 artifact hashes match the frozen local corpus)
plus an 8-task spot check (cluster `ds_00002` vs the validated local smoke run —
`best.json`/`workload.json` byte-identical, 161,280-row `placements.jsonl` identical as a
set, `infrastructure.json` differing only in `metadata.{generation_time,config_file}`).
Generation recipe is the full Arm S env block — job 713615 failed for lack of
`HEROSIM_COSIM_KEEP_ALIVE`/`HEROSIM_RETAIN_TASK_TIMES`, and the skip threshold had to be
raised to 2,000,000 against the *pre*-uniqueness `total_possible` (hard bound 1,248² =
1,557,504) after 1,000,000 silently dropped `ds_00026`; both are documented in
`route_b_8task_probe.sbatch`.

**Alpha correspondence — registered a priori, not searched.** `cap_node = alpha ·
max_single_demand` has no task-count term, so 8 tasks against the same cap is ~2× tighter;
the equal-tightness match to `TIGHT_ALPHA = 2.0` is its double, **4.0** (ladder
3.0/4.0/5.0/6.0 covers both doubled rungs plus the response curve). At the primary 4.0 the
silent-bias counters are clean: `greedy_stuck = 0`, `no_feasible_rows = 0` (at 3.0: 76
stuck, 2 no-feasible — the tight end is real, and it is not the primary).

**Firing: 33/204 = 16.2%** at `r_exact_pct > 5.0`, vs the 4-task 35/204 = 17.2% — the
a-priori doubling landed on matched power (pooled statistic rests on 19 closed cells vs 20),
so the two closures are directly comparable. Firing `r_exact_pct` spans 5.3–63.7%, median
14.3%.

| arm (exploratory, NOT registered, no verdict read from it) | 4-task (§9c) | 8-task |
|---|---|---|
| krank + dim36crk, per dataset | 1.000 (26/35) | **0.988** (20/33) |
| **krank + dim36crk, ONE pooled coefficient set** | 0.790 (mean_tied 0.824, 20/35) | **0.617** (mean_tied 0.617, 19/33) |

**Reading: the layout hypothesis survives the doubling, attenuated.** A single pooled,
identity-free coefficient set still closes the median firing dataset above half
(0.790 → 0.617); per-dataset closure is essentially unchanged (1.000 → 0.988). Per §9c's
own framing this remains **evidence about the corrected-registration hypothesis (per-node
occupancy *resolution*, no identity, no message passing), not a gate** — it feeds the
stage-2 re-registration and changes no verdict.

**Registered readings of §9c(a)/(b), applied to this corpus — both land opposite their
4-task values:**

- **(a)** `kint` coefficients regressed on node features, held out by dataset:
  **R² = 0.607** (in-sample 0.644; 390 coefficients, 33 datasets) — ≥ 0.5 reads
  *feature-representable*, where the 4-task corpus measured 0.0138 (*identity-memorized*).
  With 2 tasks of each type per dataset the per-type gauge degeneracy §9c's verification
  note describes is also broken, so the fit is better-posed here, not just luckier.
- **(b)** anonymous (dim36crk-expressible) closure: **all three tie readings agree at
  0.988** (optimistic upper bound 0.997) — the 4-task VOID-TIE-INDETERMINATE does not recur
  at 8 tasks; the script's registered rule prints `NO-GO-PREPROBE-T1-STANDS`, and the
  transfer's top-level verdict is `BOUND-TRANSFERS`.
- Decomposition against the registered physical predictions: the closure is carried by the
  **parent-coupling block** (hop+coupling pooled median 0.997, 23/33); the occupancy blocks
  (`kint`/`quad`/cap) pool to median **0.000**. At 8 tasks the pooled structure is
  parent-coupling-shaped, not occupancy-shaped.

**Provenance.** Corpus job 713673, scorer job 713793, transfer job 713794 (41 min), all
CPU-amd, repo at `72d75e7` both venues. Artifacts (both venues):
`simulation_data/route_b_8task_rtt.json` (frozen report, `--include-per-dataset`),
`simulation_data/route_b_8task_coefficient_transfer.json`. Harnesses:
`scripts_cosim/datalab/route_b_8task_{probe,score,transfer}.sbatch`,
`route_b_venue_identity_gate.sbatch`.

**Status: route_b_v1 item 4 CLOSED — probe complete, outcome recorded. The anonymous
per-node-resolution layout hypothesis survives the 4→8 task doubling at matched firing
power (0.790 → 0.617 pooled, per-dataset ~1.0 both). Exploratory throughout; the stage-2
re-registration required by §9c remains the gating step.**

### route_c_link_transfer_v1 — SCREEN REGISTRATION (2026-08-26, registered BEFORE generation)

**Question.** §9b/§9d say node-memory contention is pointwise-closable with the right
feature layout. The one mechanism pointwise link controls could NOT repair is link
contention (`link_contention_v1`, FALSIFIED on magnitude only: 0.08–0.35% regret). Can an
environment where link waiting is a *material* share of RTT resist a fairly-armed pointwise
competitor? If yes, the stage-2 re-registration is rewritten against that environment
(Branch B of the 2026-08-26 handover); if no, Branch A proceeds on the current corpus.
The lineage name is reserved; it is **named only if this screen passes**.

**Physics facts the screen rests on** (verified in code 2026-08-26): link waiting accrues
ONLY on the client→executing-node ingress transmission of a task's INPUT
(`infrastructure.py` store-and-forward loop; payload = `stateSize[app]["input"]` via
`scheduling_cost.transfer_time`). The parent→child dependency payload (`output`, Arm S's
800 MB lever) never touches the fabric — `_payload_transfer_time` charges hop-count time
with no pipes. Hence the new input-only lever `HEROSIM_INPUT_SIZE_BYTES`
(`executecosimulation.apply_state_size_override`; mutually exclusive with
`HEROSIM_STATE_SIZE_BYTES`), and hence the competitor block counts co-use on ingress
routes, not DAG-edge routes.

**Instruments** (all changes landed and validated before this registration):
- `score_route_b_contention.py` with the opt-in `linkrank` block: fixed-width order
  statistics of per-link co-use over the plan's ingress routes (top-4 counts, excess
  Σ(c−1)+ all/core, shared-link counts all/core; 8 columns, identity-free, poolable).
  Registered §9a statistics are proven unchanged (default blocks =
  `T1_REGISTERED_BLOCKS`; old-vs-new report diff identical on 24 Arm S datasets; one
  plan's co-use hand-verified against `infrastructure.json` routes). New per-dataset
  arms: `r_exact_repaired_lnk_pct` (all rows incl. unconstrained), `r_exact_repaired_t1lnk_pct`
  (constrained rows).
- `route_b_coefficient_transfer.py --add-linkrank`: extends ONLY the exploratory krank
  pooled arm with `linkrank` (registered §9b cells untouched; without the flag the frozen
  §9c artifacts reproduce exactly: pooled 0.7898, mean_tied 0.8242, R² 0.0138 — re-run
  2026-08-26).
- `HEROSIM_RETAIN_LINK_STATS=1`: every `placements.jsonl` row carries `link_wait_total`,
  `link_transfer_avg`, `fabric_link_wait_total` (fail-loud, `HEROSIM_RETAIN_TASK_TIMES`
  precedent) — the manipulation-check statistic.

**Registered reading, per rung.** Scorer alphas `2.0,6.0` (+ auto `None`); objective rtt.
Two channels: the **unconstrained row** is candidate 1 (pure link coupling — anchor: on
Arm S today unconstrained R_exact = 0.000 across 24/24 datasets, so any nonzero here is
the fabric), and **α=2.0** is candidate 2 (link + memory contention; α=6.0 is the
near-unconstrained cross-check for the pooled machinery, which needs finite caps).
A rung is GNN-promising on a channel iff ALL of:
1. *Manipulation check*: median over datasets of (Σ per-plan `link_wait_total` share of
   summed rtt) ≥ **10%** — otherwise the rung failed to make contention material and is
   INVALID (no verdict read from it, ladder continues).
2. *Firing*: ≥ **15%** of datasets at `r_exact_pct > 5.0` on that channel.
3. *Closure*: pooled anonymous closure (krank + dim36crk-expressible + linkrank, ONE
   coefficient set; `--add-linkrank`) median fraction < **0.5** on the firing set,
   reading `mean_tied`, with registered and optimistic readings agreeing within 0.1
   (disagreement ⇒ AMBIGUOUS: widen the ladder, never lower the bar). Per-dataset
   `r_exact_repaired_lnk_pct` / `t1lnk` are reported as diagnostics, no verdict.
Screen PASSES iff at least one VALID rung is GNN-promising on either channel. §9c's
write-up rule binds: a failed screen reads "this environment class is pointwise-closable;
the corpus is the limit", never "the GNN is falsified".

**Disclosure.** During instrument validation (before this registration was written) the
`--add-linkrank` pooled arm was run once on the existing Arm S corpus: pooled median
0.648 (mean_tied 0.790) vs 0.790 (0.824) without. No verdict is read from it — rung 0
has no manipulation check (old jsonl lacks link fields) and its bandwidth (1000 MB/s)
makes contention ~0 by the anchor above.

**Rungs** (4-task, local, `ROUTE_B_PILOT_V1_GRID` physics + Arm S env
`HEROSIM_DATA_LOCALITY=1 HEROSIM_OUTPUT_SIZE_BYTES=800000000` +
`HEROSIM_COSIM_KEEP_ALIVE=1000000 HEROSIM_RETAIN_TASK_TIMES=1 HEROSIM_RETAIN_LINK_STATS=1`,
24–48 datasets/rung; backbone via preset `backbone_defaults`):
- R1: n_core=4, attach_degree=1, chord_count=0 (the measured `link_contention_v1`
  coupling peak), bandwidth 1000 MB/s, `HEROSIM_INPUT_SIZE_BYTES=157286400` (150 MB).
- R2+: lower bandwidth (100, then 25 MB/s) at the same topology until the manipulation
  check passes; bandwidth moves link cost's share of RTT even though it cannot move the
  wait/transfer ratio (the 2026-08-18 null-lever result — the ratio lever is crossings,
  which n_core=4 maximizes).
- Contingency (only if AMBIGUOUS): 8-task (2× diamond4, α ladder ×2 per §9d equal
  tightness) at the best 4-task rung, on datalab via the `route_b_8task_probe.sbatch`
  pattern — concurrency is the known 7–14× amplifier (real-trace A/B).
- Control anchor per rung family: the existing Arm S corpus itself (bandwidth 1000,
  input 153,600 B) with its measured unconstrained 0.000.

**Decision gate** (write the outcome here): PASS → name the lineage, register its full
gate before any full corpus, rewrite the stage-2 registration against the new
environment; the GNN's claim-to-beat is the linkrank-augmented pointwise model. FAIL on
all valid rungs → Branch A on the current corpus with the honest sentence above.
AMBIGUOUS → widen the ladder.

**4-task ladder outcome (2026-08-26): ALL THREE RUNGS INVALID — AMBIGUOUS, contingency
invoked.** R1/R2/R3 (24 datasets each, local, corpora
`gnn_datasets_dag4_route_c_link_screen_r{1_bw1000,2_bw100,3_bw25}`): link-wait share of
rtt median 0.0007 / 0.0045 / 0.0104 against the 0.10 bar
(`simulation_data/route_c_screen_manipulation.json`). The failure is **structural, not a
tuning miss**: the bandwidth-free ceiling wait/(wait+transfer) — the share link waiting
reaches even if link cost consumed ALL of rtt — is median 4–6%, max 8.8%, because one
client and a diamond DAG cap concurrent transfers at 2 (root and sink transfer alone; the
two mids always share the client trunk). No bandwidth or payload value can pass the
manipulation check in this family — the 2026-08-18 null-lever result reappearing at the
rtt level. Corroborating diagnostic (no verdict — rungs invalid): unconstrained R_exact
is **exactly 0.000 on all 72 datasets across the 40× bandwidth range**
(`simulation_data/route_c_screen_4task_rtt.json`), i.e. at this concurrency link waiting
never moves the argmin at all; the α=2.0 firing seen is route_b's known memory channel.
Per the registered gate: widen the ladder along the only remaining lever, concurrency —
the 8-task contingency rung (grid `route_c_link_screen_8task`, 2 diamond4 instances from
independent clients, bw=25, input 150 MB, α ladder ×2), generated on datalab via the
`route_b_8task_probe.sbatch` pattern.

---

## RETIRED

| Lineage | Status | Archive | Files | Outcome |
|---|---|---|---|---|
| **pre_gnn_herosim** | `PAPER` | `archive/pre_gnn_herosim/` | 145 | The original HeROsim proactive-autoscaling paper: XGBoost/GPR demand prediction, Bayesian optimisation over infrastructure, LHS sampling, `scenario-*.sh`, and the HRO/HRC/proactive-Knative policies. Superseded by the GNN co-simulation work; kept because the paper cites it. |
| **regime_b** | `FALSIFIED` | `archive/regime_b/` | 38 | Cold-burst regime with `platform_reuse_v1` physics. Phases 0–3.1 closed the gap 125→31 only by distilling `ect_pull`, and `ect_pull` itself lands at Knative level on the coupled trio — so it was never a ceiling to chase. CLAUDE.md already marks Regime B outdated. |
| **soft_combo** | `FALSIFIED` | `archive/soft_combo/` | 6 | Joint combination scoring (`soft_combo`, `soft_combo_conc`) gave no gain over CE on `oracle_split_v1` (commit `d6f1999`). The **loss functions stay live** in `non_unique_lib/soft_combo_loss.py` — `train_near_rtt.py` still imports them; only the experiment wrappers are archived. |
| **warmth_sparse** | `SUPERSEDED` | `archive/warmth_sparse/` | 110 | The warmth/sparse/skew-merged/hub9 series, plus its regen, repair and health-monitor tooling. Superseded by the contention series, which produces the contention the GNN actually needs. Largest single lineage. |
| **model_sweeps** | `SUPERSEDED` | `archive/model_sweeps/` | 38 | One-off sweeps named after their wandb run (`woven_totem`, `silvery_sun`, `ethereal_lake`, `worthy_bush`, `ssc_trash`, `clean_1230`, `mitrix`) plus `atomic21`, `dim14_1060`, `mega_matrix`, `reviewer_triangle`. **The reason the naming convention changed:** a filename encoding a run name tells you nothing about the hypothesis. |
| **topology_sweeps** | `SUPERSEDED` | `archive/topology_sweeps/` | 32 | `tiered_hub` and `bipartite_coordination` topology experiments. |
| **strategic_merge** | `SUPERSEDED` | `archive/strategic_merge/` | 10 | Merged-corpus training strategy, replaced by the siv1 full-corpus approach. |
| **decode_ablations** | `SUPERSEDED` | `archive/decode_ablations/` | 6 | `seqblend`, `seq_reforward`, pull-decode ablations. The decode paths themselves remain in `src/policy/gnn/seq_decode.py`; only the sweep scripts are archived. |
| **hetero_training** | `SUPERSEDED` | `archive/hetero_training/` | 4 | Heterogeneous-graph GNN training. The **`gnn_hetero` policy stays live** (`executesimulation.py` dispatches it); only the training/caching scripts are archived — nothing invoked them. |
| **exact_rtt** | `SUPERSEDED` | `archive/exact_rtt/` | 2 | Exact-RTT regression objective, replaced by near-RTT + CE. |
| **live_finetune** | `SUPERSEDED` | `archive/live_finetune/` | 3 | Live-trajectory finetuning experiment. |
| **old_scripts** | `SUPERSEDED` | `archive/old_scripts_{idk_big,old_bash}/` | 11 | Pre-`generate_gnn_datasets_fast.py` bash pipeline and one-off state-discrepancy analyses. |

---

## GATE TOOLS — correctness fixes

Facts about the *gates themselves*, kept out of the lineage narratives on purpose. A gate
that lies is worse than no gate, and someone re-running one of these in six months needs to
find out what changed about the tool without reading a lineage's story to get there.

| date | tool | what was wrong | now |
|---|---|---|---|
| 2026-08-19 | `verify_cache_live_feature_parity.py` | **Compared platform rows by position.** The cache enumerates platforms from `stats.nodeResults`, live from `config.infrastructure.nodes`, and those orders differ in **18/18 collections** (up to 229 rows). The gate therefore reported ~20 failures on every reordered corpus — false, since the model has no per-position parameter — which buried the one real failure and made the gate unrunnable on the whole netc family. | Compares **by platform identity** `(node_name, platform_id)`: features via a permutation, edges / candidates / same-node edges as identity-keyed sets, candidate lists as sets (decoding is by identity). Reordering prints as a `note:`. Object-dtype arrays compare exactly. On `netc_multihop_v1_core4/ds_00000`: 20 findings → 1 real → 0 after the fix below. |
| 2026-08-19 | `verify_cache_live_feature_parity.py` | Coerced every temporal value with `float()`, so newer SSC files carrying `previous_task_type_name` crashed it before any comparison ran. | Non-numerics pass through unchanged. |
| 2026-08-19 | `refresh_optimal_full_stats.py` (usage, not code) | The netc corpora ship without `system_state_captured_unique.json`, so the parity gate and `prepare_graphs_cache.py` both fail on them out of the box. | Run `--repair` (or `--rewrite-ssc`) on a collection before gating or caching it. **Fourth** time this has bitten — the `topo_transfer_v1` probe datasets ship without it too, so the Phase 4 corpus needs the repair pass before caching. |

| 2026-08-19 | `gnn_necessity_ablation.py` (gate design, not a code bug) | **Gated on statistics that drift with sweep size.** `regret_mean`/`p90`/`max` move 2.6–6.6× across sweep-size bins *when the decision rule is held constant*, so the topology_transfer_v1 PASS condition ("the gap widens monotonically") is satisfiable by landscape alone. Trimming (2.27) and log-scaling (2.64) do not help — the problem is aggregation order, not outliers. Separately, at n=30 the minimum detectable regret gap is 0.149 against observed gaps of 0.003–0.02, so the harness could not resolve its own effect — the real cause of the seed-44 verdict reversal. | Primary gate is now per-dataset **paired** `win_rate` (null = 0.5 at every size) + `regret_ratio_mean`, in `scripts_cosim/gate_statistics.py` (15 tests). The harness prints the paired block, a bootstrap CI, an exact sign test, and a loud `!! UNDERPOWERED` line when the gap is below the noise floor; frozen reports gain `paired_comparisons` (schema 3 → 4). |

| 2026-08-19 | `gnn_necessity_ablation.py` | **GIN training was not reproducible at a fixed seed, and the cause was misattributed to CUDA.** Two identical CPU commands gave different `gnn_base` results (`regret_mean` 103.41% vs 104.36%, `win_rate` 0.517 vs 0.550) while `pointwise` was bit-identical; the *training loss* diverged, so it is the GIN autograd path, not eval. Ruled out: intra-op threading (`OMP_NUM_THREADS=1`), `PYTHONHASHSEED`. Consequence: every "N seeds" result in this file that involves a GIN conflates seed effects with run-to-run noise, and the noise was the larger term. | `torch.use_deterministic_algorithms(True, warn_only=True)`, **on by default** (`--nondeterministic` opts out), verified bit-identical across repeated runs on all three models and every reported statistic. Note it also *changes* values (a different algorithm is selected), so it does not reproduce pre-fix numbers. |

| 2026-08-19 | `gnn_necessity_ablation.py` / `gate_statistics.py` (gate design) | **The v2 gate called a straddling CI a FAIL.** At the effect sizes this lineage actually expects (\|`win_rate` − 0.5\| = 0.017–0.033) a CI containing 0.5 is the *expected* outcome at tier 0.02, so the criterion would have converted an under-powered run into "topology transfer failed" — a false negative manufactured by corpus size, the same class of error as v1's landscape-drift PASS. Also: the tier names are in regret-gap units while the primary statistic is a proportion, so "MDG 0.02" and "resolves a 0.02 win_rate effect" are **not** the same claim. | Pre-registered `PHASE4_TIERS` + `phase4_verdict()` / `escalation_note()` (fixed *before* any corpus): `PASS` only when the CI excludes 0.5 above, `FAIL` only when it excludes 0.5 below, otherwise `ESCALATE` to the smallest tier that resolves the observed effect, or `INCONCLUSIVE_LADDER_EXHAUSTED` with the required n. Power arithmetic redone in win_rate units, which exposed that a 0.017 effect needs ~3,400/size and **no registered tier covers it**. Wired into the harness (`--power-tier`, printed per model); frozen reports gain `phase4_verdicts` (schema 4 → 5). 10 new tests, 25 total. |

| 2026-08-19 | `gate_statistics.py` (gate design) | **The v2 gate's "co-primary statistics disagree in sign" FAIL condition was only safe evaluated PER SEED.** Seed 44 on `shallow_v1` has `win_rate` 0.533 (GNN ahead) and `regret_ratio_mean` 1.999 (GNN far behind on magnitude) simultaneously — a real bimodal distribution (ratio ≈0.99 in calm seeds, ≈2.0 in a blow-up seed caused by one cliff-shaped dataset, `ds_00157`), not gate noise. A per-seed rule lets exactly one unlucky seed veto a lineage that the other four seeds agree on. **Before this fix**: a seed-44-shaped outcome among an otherwise-clean set of seeds would fail the lineage. | `pool_seed_comparisons()` pools `win_rate` as mean-of-seeds with a seed-level CI (mean ± 1.96·sd/√n_seeds — this is the *same* formula behind the frozen 5-seed calibration's reported CI) and `regret_ratio_mean` as **median**-of-seeds (not mean), specifically because a median resists exactly one seed landing in the blow-up mode. `co_primary_sign_agree()` compares the sign of those two pooled statistics **once**, at the end. `pooled_phase4_verdict()` runs the existing escalation ladder on the pooled CI and downgrades an otherwise-PASS to FAIL only if the pooled signs disagree. **After this fix**: the same seed-44-shaped outcome does not fail the lineage — verified in `test_pooled_verdict_one_blowup_seed_does_not_veto_an_otherwise_clean_pass` — unless the *pooled median* ratio actually disagrees with the *pooled mean* win_rate (`test_pooled_verdict_downgrades_pass_to_fail_on_sign_disagreement`). Also registered `tier_launch` (900/held-out size) between `tier_0.02` (360) and `tier_0.01` (1600) — covers the stronger of the two observed effects (0.033, needs ~880) at roughly `tier_0.02`'s already-budgeted cost, and is the tier Phase 4 actually launches at on escalation; the weaker effect (0.017, ~3,400/size) stays off the ladder as a documented, sized, not-yet-approved escalation path. 12 new tests, 37 total. |

| 2026-08-20 | `gnn_necessity_ablation.py` (harness bug, not gate design) | **Two independent uncommitted working trees silently dropped each other's fixes on merge.** Datalab's checkout had grown network-entity model support and a label-provenance preflight audit; local's had grown the `topology_size` split. Merging datalab's copy as the base regressed the eval loop's "predicted plan absent from the sweep" check — it stopped distinguishing an expected task-collision miss from a real corpus/harness bug and crashed `RuntimeError` on any predicted plan that merely collided two tasks onto one platform, which any undertrained model does routinely. First submission (job 705777, `topology_transfer_v1` Phase 4) burned ~50 min across all 5 array tasks before failing. This is the same failure class as `mp_parity`'s train/serve split: one file, two uncommitted forks, no diff review before running. | `eval_regret` now returns separate `n_missing_collided` (expected, silent) and `n_missing_clean` (real bug, still fails loud) counts; only `n_missing_clean` raises. When two machines have each extended the same script without committing, diff before merging, do not just take one side as "the base" — verified by the corrected rerun (job 705834) completing cleanly. |

| 2026-08-20 | `important/run_contention_v2_873_sealed_holdout.sh` and every live-gate runner | **The live gate ran infrastructure the models never trained on, and nothing said so.** The sealed holdout's four configs are 40/40 p50, 50/50 p60, 50/35 p50 and 35/50 p50; every dataset in the `legacy_v0_node_disk_v2_4task` corpora is **20 clients / 20 servers**. Node count doubled and connection probability doubled between train and serve. Neither the runner, `executesimulation.py`, nor any comparator compared the two — checkpoint sidecars recorded feature contracts but never the *infrastructure*, so there was nothing to compare against. Every live number from these gates is an out-of-distribution measurement that was read as an in-distribution one. | Checkpoint sidecars now carry a `corpus` block derived from the cache's own `dataset_ids` (`src/placement/corpus_provenance.py`), and `load_gnn_model` prints a loud `!! INFRA MISMATCH` with the trained-vs-live numbers for cluster size, topology type and connection probability. Deliberately a **warning, not an error**: both model classes are candidate-pair based and *can* run at another size — that is `topology_transfer_v1`'s question — but it must never happen unnoticed. Warmth-physics mismatch **does** raise (it changes the cost model, so the RTT is not comparable at all). New gate `scripts_cosim/verify_live_infra_parity.py` + 13 tests. |

| 2026-08-20 | `src/executesimulation.py` (`--seed`) | **`--seed` overrides the config's topology seed, so every "multi-seed" live gate was a multi-*topology* gate.** `prepare_infrastructure_for_real_simulation(space_config, seed=seed)` uses the CLI seed for topology generation and only falls back to `network.topology.seed` when it is absent (`:842-854`). Measured against `contention_v2/ds_00000`'s own config: `--seed 42` yields **144/210 directed edges different and 64/64 shared edges disagreeing on latency**. Runs reported as "seeds 42..46 on config X" therefore varied the infrastructure and the simulation randomness together, and any per-seed spread conflates the two — the same confound class as the GIN non-determinism row above, in the data rather than the optimizer. | `run_full_corpus_siv1_live_gate.sh` passes **no `--seed`**, so topology comes from each cell's config; replication comes from distinct parity-verified cells instead. `verify_live_infra_parity.py --seed N` reproduces the divergence on demand, and `test_seed_override_fails_parity` pins it. Pre-existing gate results are not retracted by this — they were internally consistent — but their per-seed variance should not be read as simulation-seed variance. |

| 2026-08-20 | any new `*.sbatch` (env activation) | **`micromamba shell hook --bash` fails on compute nodes but works on login nodes.** Compute nodes run a newer micromamba that requires `--shell bash` and rejects the old spelling with `The following argument was not expected: --bash`. Job 707292 (`siv1_full_corpus` live gate) died with all 15 array tasks failing in ~1s. The trap is that a **login-node sanity check cannot catch this** — the login node accepts the old form, so the pre-submit validation passed while every compute node rejected it. | Use `eval "$(micromamba shell hook --shell bash)"`; every other script in `scripts_cosim/datalab/` already did, and the broken one was a hand-written outlier. When adding an sbatch, diff its env-activation preamble against a neighbouring script rather than trusting a login-node dry run. |

| 2026-08-21 | `{knative,gnn,knative_network}/autoscaler.py` (all 3, `mlp_batch` inherits `GNNAutoscaler`) | **`eb6d131`'s tie-break fix was incomplete, and `PYTHONHASHSEED` does not cover this class of bug at all.** All three scale-down sites do `sorted(function_replicas, key=lambda couple: len(couple[1].queue.items))` — a **non-total** key over a `Set[Tuple[Node, Platform]]`. `sorted()` is stable, so any tie (two idle replicas with equal queue length) keeps the underlying set's iteration order, which for a set of *objects* is `id()`-derived — Python only randomizes `str`/`bytes` hashing, so `PYTHONHASHSEED` genuinely does nothing here. Proved directly: 3 processes under identical `PYTHONHASHSEED=0` iterate a set of 8 objects in 3 different orders, while a control set of 8 strings is identical across all 3. Measured consequence: `knative_network`/cell03 p=0.15, 4 independent processes, identical inputs, `PYTHONHASHSEED=0` pinned — `total_rtt` took 4 distinct values, 0.05% spread (sd 0.0235%). Small relative to any recorded gate margin (239-816x below the siv1 gate's 12-41%), but real and previously unmeasured — the `PYTHONHASHSEED=0` "defense-in-depth" in the live-gate scripts was assumed to cover this and does not. | **Not yet fixed** (found mid live-gate run this session; deferred to avoid splitting one gate's results across two code versions). Fix is a total key: `key=lambda couple: (len(couple[1].queue.items), couple[0].id, couple[1].id)`. See `memory/herosim-pythonhashseed-tiebreak-nondeterminism.md` for the full empirical writeup. |
| 2026-08-21 | `src/executesimulation.py` / live result JSON (reporting gap, not a bug) | **`NetworkFabric.link_wait_total` (`network_fabric.py:133`) and `task.link_wait_time` (`infrastructure.py:196`, serialized as `linkWaitTime`) accumulate real link-contention waiting and are never surfaced outside the test suite.** The live result JSON's `stats` block has no link field, so the simulator computes exactly the quantity `link_contention_v1`'s real-trace A/B needs (splitting a backbone's total_rtt delta into transmission vs. contention) and discards it. | **Fixed in two stages.** Stats-level: since `d88278c` (2026-08-23) both stats paths emit `averageLinkWaitTime` / `totalLinkWaitTime` / `averageLinkTransferTime` / `fabricLinkWaitTotal` next to `total_rtt` (`orchestrator.py` `_stats_low_memory()` and `stats()`), so live result JSONs and co-sim `best.json` carry them. Per-plan: 2026-08-26, opt-in `HEROSIM_RETAIN_LINK_STATS=1` writes `link_wait_total` / `link_transfer_avg` / `fabric_link_wait_total` into every `placements.jsonl` row (`executecosimulation.py`, same fail-loud contract as `HEROSIM_RETAIN_TASK_TIMES`) — the sweep-wide decomposition the route-C screen needs. |
| 2026-08-21 | `generate_infrastructure.py` (`build_core_backbone`) via `verify_live_infra_parity.py` | **The backbone's access-link jitter is drawn from the same `rng` stream the replica-reachability repair already consumed**, and the backbone is built *after* the repair (`:768` then `:780`). A live run performs no repair (it autoscales from zero), so it reaches the backbone build at a different stream position than the corpus generator did — every access-link latency diverges on any cell with a non-empty repair set (measured: 3/5 siv1 gate cells FAIL when a backbone is added, matching exactly the 3 cells with nonzero repair-edge counts; the 2 with zero repair edges PASS). | New `--allow-backbone-latency-divergence` flag on `verify_live_infra_parity.py`, scoped narrowly: downgrades exactly the two affected finding classes to notes, and only fires when a backbone is present on **both** the corpus and live sides — verified it cannot mask a mismatch on a non-backbone collection, and that without the flag the same cells still correctly FAIL. Root rng coupling itself is not fixed (would require an independent substream and would break bit-reproducibility of `gnn_datasets_4tasks_topo_transfer_v1`'s existing 3,744-dataset corpus from its seed — a bigger call than this session's scope; the flag is the practical unblock for a live-vs-live A/B where the corpus-side artifact is only a preflight fixture). |
| 2026-08-21 | live-gate protocol / `run_provenance` (via `datalab/siv1_env_probe_{gpu,cpu}.sbatch`) | **A live gate can silently measure an uncommitted code diff instead of the model.** `models/` syncs by rsync but `src/` syncs by git, so the dims 9-11 live-feature fix (working tree 2026-08-19, uncommitted) ran locally but not in datalab's job 708549 — 23.3% of `total_rtt` on `gnn/cell01`, flipping the gate verdict. `run_provenance` records env vars and contracts but **not the git commit or working-tree state**, so nothing in either side's result JSON could reveal the split. Root-caused by a 7-run probe matrix (see the siv1 resolution subsection): the two new probe sbatch files re-run one gate cell on the recorded node (GPU ×2) and CPU-forced, establishing datalab-side noise floors (±0.04% GPU run-to-run, ±0.03% GPU↔CPU) as a by-product. | Protocol: `git status --short src/ scripts_cosim/` must be clean before any datalab gate, and local+datalab must be at the same commit. Code fix planned (deferred until no sweep is mid-run): record `git describe --dirty --always` + a hash of `git diff` in `run_provenance`. |
| 2026-08-21 | `scripts_cosim/important/run_*_live_gate*.sh` (50 `pipenv run python3` call sites in 18 files) + new `scripts_cosim/verify_venue_parity.py`, `src/placement/env_fingerprint.py` | **Every datalab gate ran in an undeclared environment, and nothing recorded which.** `micromamba activate gnn` followed by `pipenv run python3` does not run in the `gnn` env — `pipenv run` resolves its own venv and shells past the activation, so the cluster silently used `~/.local/share/virtualenvs/gnn-herosim-2TQKssTQ` (**torch 2.12.0+cu130**) while `CLAUDE.md`, the sbatch header and this file all asserted the `gnn` env (torch 2.5.1+cu121). Cost: three sessions attributing an 11-26% GNN gap to the venue. **Measured and closed the same day:** a committed 64-graph fixture (256 decisions / 1,738 scored edges) forwarded through the deployed checkpoint gives max|Δlogit| = **exactly 0.0** and **0/256 argmax flips** between local (torch 2.5.1+cu121 / numpy 2.3.0 / PyG 2.6.1), the `gnn` env (numpy 1.26.4 / PyG 2.7.0) and the rogue venv (torch 2.12.0+cu130 / PyG 2.8.0), at 1 and 4 threads. A CPU→CUDA negative control on the *same* env gives 1.9e-5 and still 0 flips, proving the probe is sensitive and the zeros are real. So the library-version axis contributes nothing; only the accelerator does, below the flip threshold. | `verify_venue_parity.py --mode logits --assert` runs in ~6 s (login-node safe) and is the preflight; `--mode run` does the same end-to-end on one cell, keeping a `knative` control arm because Knative never touches `build_inference_feature_bundle` and a Knative-only cross-check is structurally blind to this bug class. Protocol + comparability checklist in **`PARITY.md`**; hard rules in `CLAUDE.md`; datalab-pitfalls **#8**. Leak fix itself (`${HEROSIM_PY:-pipenv run python3}` + `export HEROSIM_PY=python3`) deferred while `a4_wl200200` is mid-run — it is hygiene, not a numerical correction. |
| 2026-08-21 | `src/policy/{gnn,knative,knative_network}/autoscaler.py` (`remove_replica`) | **Scale-down tie-break was a non-total key**, so the residual 0.05–0.1% run-to-run spread survived `eb6d131`. `sorted(function_replicas, key=len(queue.items))` ties at 0 for every eligible candidate — scale-down only removes empty queues — and the stable sort then returns `Set[Tuple[Node, Platform]]` iteration order, which follows id()-based object hashes that `PYTHONHASHSEED` does **not** pin (it randomizes str/bytes hashing only). Enough to swamp a sub-percent gate margin. | **Fixed.** Key is now `(len(queue.items), node.id, platform.id)` in all three; `mlp_batch` inherits the gnn autoscaler and is covered. `scripts_cosim/test_autoscaler_scaledown_determinism.py` went 3 failed → 9 passed. The other seven policies (`herocache*`, `roundrobin_network`, `offload_network`, `gnn_hetero`, `evaluator`) still carry the old key — deliberately untouched, since changing them would break comparability with their own recorded results and none is in an active gate. |
| 2026-08-21 | `src/executesimulation.py` `build_run_provenance` + `src/placement/env_fingerprint.py` | **A result JSON could not say what code produced it.** The planned fix from the row above, now implemented: `describe_code_provenance()` records commit / branch / dirty / `diff_sha256` / `changed_files`, and the full `python_env` + `env_fingerprint` ride along, so two disagreeing results are triaged as *different commit* / *same commit, different working tree* / *identical code* without access to either machine. Both banners print at run start. | **Fixed**, 7 tests in `tests/test_code_provenance.py` driving real throwaway repos. The dirty flag is scoped to `CODE_PATHS` (`src`, `scripts_cosim`, `experiments`, `run_experiment.py`) on purpose — a repo-wide flag would fire on every `simulation_data/REGISTRY.json` refresh and be ignored within a week. A non-git tree is reported, not fatal. **Results produced before this commit carry no code stamp; that is what makes 708549-vs-709163 a story rather than a lookup.** |
| 2026-08-21 | 52 `pipenv run python3` call sites / 18 shell scripts + 3 `.sbatch` | The environment leak from the row above, **closed at the source**. | **Fixed.** All call sites read `${HEROSIM_PY:-pipenv run python3}`; the three sbatch files that activate micromamba `export HEROSIM_PY=python3` immediately after. Locally `HEROSIM_PY` is unset so behaviour is byte-identical. Also adds **`envs/herosim-lock.txt`**, which `PARITY.md` and `CLAUDE.md` had both called the single canonical spec while it did not exist — the same defect class as the leak. Note the cluster `gnn` env lacks `orjson`, `pytest` and a working `scikit-learn` (broken against its scipy 1.17.1); the live-gate and recache import closures were verified clean under it before the flip, and none of the three is on those paths. |

| 2026-08-23 | `generate_infrastructure.py` (`build_core_backbone`) + `verify_live_infra_parity.py` | The 2026-08-21 row above deferred the **root** rng coupling as out of scope ("would break bit-reproducibility of `gnn_datasets_4tasks_topo_transfer_v1`'s existing 3,744-dataset corpus from its seed"). That objection is answerable without giving up either property. Separately, a **second** finding class was being waived and had a different cause: under a backbone a genuine repair edge's latency is a **route sum**, not `base_latency`, so `_classify_corpus_only_edges` could not recognize it and reported real repair edges as "unexplained" — parity printed `repair=0/174` on a cell with 34 of them. | **Both fixed.** `network.backbone.rng_stream` selects the jitter stream: `independent_v1` derives it from `Random(f"{seed}:backbone_v1")`, independent of whatever consumed the shared stream earlier; `legacy_v0` remains the **default in `build_core_backbone`**, so every existing corpus still regenerates byte-identically from its seed, while `generate_gnn_datasets_fast.py` defaults **new** corpora to `independent_v1` (`--backbone-rng-stream`). `rng_stream` is stamped into `link_topology.params`. The classifier now takes the fabric and checks each corpus-only edge against **its own recorded route** — stricter than the `base_latency` signature, not a relaxation. Verified end-to-end on cell03 (34 repair edges, the worst offender): `legacy_v0` reproduces the recorded `infrastructure.json` links exactly and still diverges 140/140 shared edges live; `independent_v1` gives identical links, live routes a corpus subset with identical paths, and **PASSES with no waiver flag**, reporting `repair=34/174`. Control holds — legacy cell01/cell03 still FAIL without the waiver, cell02 (0 repair edges) still PASSes. 5 tests. The waiver flag stays for the existing legacy backbone cells, whose recorded arms must remain comparable. |
| 2026-08-23 | `src/executesimulation.py` `load_gnn_model` (`INFERENCE_FEATURE_LAYOUT`) | **A `task_dim=3 / platform_dim=14` checkpoint was served under a silently guessed feature layout.** That shape is structurally valid under **both** `atomic21` and `dim22`, which assign different meanings to the same platform columns (`dim22` normalizes the queue features via `use_norm_queue`), so `load_state_dict` succeeds either way and the forward pass raises nothing. `load_gnn_model:614` defaulted an undeclared layout to `atomic21`. Consequence: the `prefixctl` (variance control) and `tempfix` (corrected-cache) gates served **atomic21** — their sidecars declare `inference_feature_layout: null` and `run_full_corpus_siv1_live_gate.sh` deliberately does not export the variable — while **every** deployed-checkpoint gate served **dim22** from its sidecar. Both gates' result JSONs record `INFERENCE_FEATURE_LAYOUT: None` against the deployed runs' `dim22`, and the loader banner ("Using atomic21 inference layout with task_dim=3 checkpoint") was the only signal. **This sits underneath the 2026-08-22 lottery table**, whose deployed-vs-prefixctl comparison was read as a pure training-draw effect. | **Fixed:** on that ambiguous shape, declaring neither a sidecar layout nor an env var now **raises** instead of picking one. Sidecar declaration still works (so the deployed checkpoint and the gate scripts that rely on it are unaffected) and an explicit env declaration still works. 3 tests in `tests/test_inference_layout_contract.py`; full suite 337 passed. Magnitude of the effect on live `total_rtt` is being measured separately (`important/run_layout_confound_probe.sh`) — until that lands, the lottery table should be read as *draw + serving layout*, not draw alone. |

New tool from the same work: **`scripts_cosim/audit_cache_live_divergence.py`** — measures
cache↔live disagreement across every collection from `optimal_result.json` (+ SSC where
present). No cache, no training, ~seconds. Run it before trusting a model trained on a
collection you have not gated.

## Conventions

**A lineage is not finished until it has a row in this table with an outcome.** A sweep
whose result was never written down will be re-run by someone in three months.

**A gate tool's own correctness is a recorded fact, not folklore.** When a gate turns out to
have been measuring the wrong thing, it goes in the GATE TOOLS table above — not inside
whichever lineage happened to trip over it. Two of this repo's near-misses came from a
tool-level fact being buried in a lineage narrative.

**Do not fork a training script per experiment.** That habit produced 40 near-identical
`train_near_rtt_v2_*.py` wrappers that differed only in cache dir and wandb name. New
experiments get a config, not a copy.

**Do not import from `archive/`.** The live tree is verified closed against it. Re-run
the gate after any move:

```bash
# no live file may reference an archive-only filename
python3 - <<'PY'
import subprocess, pathlib, re
all_f = subprocess.check_output(["git","ls-files"], text=True).split()
live  = [f for f in all_f if not f.startswith("archive/")]
arch  = {pathlib.Path(f).name for f in all_f if f.startswith("archive/")
         and f.endswith((".py",".sh",".sbatch"))}
arch -= {pathlib.Path(f).name for f in live}
bad = []
for f in live:
    if not f.endswith((".py",".sh",".sbatch")): continue
    for ln, line in enumerate(pathlib.Path(f).read_text(errors="ignore").split("\n"), 1):
        if "archive/" in line: continue
        for n in arch:
            if re.search(r'(?<![\w.-])'+re.escape(n)+r'(?![\w])', line):
                bad.append(f"{f}:{ln} -> {n}")
print("\n".join(bad) or "CLEAN"); print("broken:", len(bad))
PY
```

Known-benign hits: six comments in `src/placement/{orchestrator,simulation}.py` that
name `executeinitial.py` while explaining why a guard exists. They are prose about a
code path that no longer exists, not references to it.
