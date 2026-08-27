# cosim_deepdive_v1 — CLOSED

> **Status:** `CLOSED` &nbsp;·&nbsp; **Index:** [LINEAGES.md](../../LINEAGES.md) &nbsp;·&nbsp; **Record spans:** 2026-08-23 → 2026-08-23

**Outcome.** Does the target's additivity come from the synthetic t=0 snapshot regime? **No — live-visited states are equally additive** (4,400 swept live states, median additive R² 0.99999). The GNN's dispersal edge is a closed-loop property no single-batch regret target can express.

**Entry points:** `scripts_cosim/{audit_sweep_truncation,audit_regen_reproducibility,snapshot_separability_sweep,analyze_snapshot_separability}.py`, `datalab/live_audit_capture_{all_gates,mlp_collapse}.sbatch`, `datalab/snapshot_separability_sweep{,_mlpcollapse}.sbatch`

**Datasets:** `snapshot_sweeps{,_mlpcollapse}` (44 cells × 100 pseudo-datasets)

**Related:** [graph_structure_physics](graph_structure_physics.md) · [program_verdict_v1](program_verdict_v1.md)

## Standing (from the index table)

**Closed 2026-08-23.** Does the co-sim target's additivity come from the synthetic t=0 snapshot regime? **No — live-visited states are equally additive** (4,400 swept live states incl. all 14 MLP collapse trajectories: median additive R² 0.99999, median additive-choice regret 0.000; jobs 710774/710775/710818/710819). The GNN's dispersal edge is a closed-loop property no single-batch regret target can express under current physics. Plus a pipeline-integrity census (sweep truncation, label provenance, contract audit of the collapse events). **Outcomes below.**

> **Split note.** This node was carved out of a single 4,995-line `LINEAGES.md` on
> 2026-08-27; the section bodies below are byte-for-byte as written. An *above* / *below*
> pointer inside one may refer to a section that now lives in another node — the
> **Related** links say which.

## Record

Newest first; the sections themselves are in chronological order below.

- [cosim_deepdive_v1 — live-state separability + pipeline integrity census (2026-08-23)](#cosim-deepdive-v1-live-state-separability-pipeline-integrity-census-2026-08-23)

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
