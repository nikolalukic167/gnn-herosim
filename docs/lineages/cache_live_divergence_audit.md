# cache_live_divergence_audit — CLOSED

> **Status:** `CLOSED` &nbsp;·&nbsp; **Index:** [LINEAGES.md](../../LINEAGES.md) &nbsp;·&nbsp; **Record spans:** 2026-08-19 → 2026-08-19

**Outcome.** Platform reordering: **18/18 collections, BENIGN** (no recache, no asterisk). Dims 9-11 temporal estimate: **8/18, REAL**. The parity verifier now compares by platform identity instead of position.

**Entry points:** `scripts_cosim/audit_cache_live_divergence.py`, `scripts_cosim/verify_cache_live_feature_parity.py`

**Datasets:** all 18 collections with `optimal_result.json`

**Related:** [siv1_full_corpus](siv1_full_corpus.md) · `queue_feature_contract` (index only)

## Standing (from the index table)

Where do the cache and live feature builders actually disagree? **Platform reordering: 18/18 collections, BENIGN** — the model has no per-position parameter; logits agree to 3e-8 under the identity permutation, so no recache and no asterisk on any result. **Dims 9-11 temporal estimate: 8/18 collections, REAL** (incl. `shallow_v1`; live-gate corpora clean). Parity verifier now compares by platform identity. **Outcomes below.**

> **Split note.** This node was carved out of a single 4,995-line `LINEAGES.md` on
> 2026-08-27; the section bodies below are byte-for-byte as written. An *above* / *below*
> pointer inside one may refer to a section that now lives in another node — the
> **Related** links say which.

## Record

Newest first; the sections themselves are in chronological order below.

- [cache_live_divergence_audit — outcomes (2026-08-19)](#cache-live-divergence-audit-outcomes-2026-08-19)

---

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
