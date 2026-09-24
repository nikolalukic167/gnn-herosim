# shallow_v1 — SUPERSEDED

> **Status:** `SUPERSEDED` &nbsp;·&nbsp; **Index:** [LINEAGES.md](../../LINEAGES.md) &nbsp;·&nbsp; **Record spans:** 2026-08-17 → 2026-08-17

**Outcome.** Shallow queues lower the pointwise ceiling but **MISS the coupling gate**. Contains a **retraction**: the coupled(>1%) = 31.0% figure does not reproduce at full corpus size. Read the retraction before citing any number in this node.

**Related:** [graph_structure_physics](graph_structure_physics.md) · [shallow_longexec_v1](shallow_longexec_v1.md)

> **Split note.** This node was carved out of a single 4,995-line `LINEAGES.md` on
> 2026-08-27; the section bodies below are byte-for-byte as written. An *above* / *below*
> pointer inside one may refer to a section that now lives in another node — the
> **Related** links say which.

## Record

Newest first; the sections themselves are in chronological order below.

- [Label-provenance gate — the third gate (2026-08-17)](#label-provenance-gate-the-third-gate-2026-08-17)
- [⚠ RETRACTED — `shallow_v1` coupled(>1%) = 31.0% does not reproduce (2026-08-17)](#retracted-shallow-v1-coupled-1-31-0-does-not-reproduce-2026-08-17)
- [shallow_v1 ablation — reproduced with the label-provenance audit (2026-08-17, superseded same day)](#shallow-v1-ablation-reproduced-with-the-label-provenance-audit-2026-08-17-superseded-same-day)
- [`shallow_v1` — lowers the pointwise ceiling, but MISSES the coupling gate (2026-08-17)](#shallow-v1-lowers-the-pointwise-ceiling-but-misses-the-coupling-gate-2026-08-17)

---

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
