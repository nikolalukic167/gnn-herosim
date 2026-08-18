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
| **siv1_full_corpus** | `scripts_cosim/datalab/full_corpus_siv1_{recache,gnn_train,mlp_train}.sbatch` → `run_full_corpus_siv1_*.sh` | whole `legacy_v0_node_disk_v2_4task` group | Trains on the full corpus under `scale_invariant_v1`. GNN: `src/notebooks/train_near_rtt.py`. MLP: `src/policy/tabular/train_mlp_dim22_from_batch.py`. Recache: `src/notebooks/prepare_graphs_cache.py`. **Outcome 2026-08-17 — see `mp_parity` below.** |
| **mp_parity** | `scripts_cosim/test_train_serve_mp_parity.py`, `experiments/full_corpus_siv1_gnn_mp_residual{,_node_edges}.yaml`, `datalab/mp_arm_gnn_train.sbatch` | full corpus siv1 | Train/serve message-passing parity, and what to do about it. Outcomes below. |
| **graph_structure_physics** | `scripts_cosim/separability_diagnostic.py` (M4 + `--gate-additive-r2`) | all co-sim collections | Does the simulator produce a target a GNN could ever beat a pointwise MLP on? **Outcome 2026-08-17 below: no, not today.** Phases 1-4 (node contention, congestible links, fan-out DAGs, batch size) planned against that measurement. |
| **contention_v4_v5** | `scripts_cosim/datalab/contention_v4_deepq_cosim.sbatch`, `contention_v5_quick_test.sbatch` | `contention_v4_pilot`, `contention_v5_quick_test` | Deep queues + coupling optimisation — the attempt at giving the GNN real graph structure to exploit. **`FALSIFIED` 2026-08-17: moved the corpus the wrong way** (additive R² 0.988 → 0.9997). See `graph_structure_physics`. |
| **contention_v2_v3** | `important/run_contention_v{2,3}_train_and_live_gate_nohup.sh`, `important/compare_contention_v2_live_gate.py` | `contention_v2{,_verify}`, `contention_v3` | Baseline contention series the v4/v5 work is measured against. Trainers: `train_near_rtt_v2_contention_v{2,3}_dim14_ce_only.py`, `train_mlp_contention_v{2,3}_dim22_batchcache.py`. |
| **sealed_holdout** | `important/run_contention_v2_873_sealed_holdout{,_rebaseline}.sh`, `compare_sealed_live_holdout.py`, `datalab/sealed_holdout_gpu.sbatch` | `contention_v2` | The honest generalisation gate. |
| **coupled_trio** | `important/run_contention_v2_873_coupled_trio.sh`, `chain_coupled_trio_then_rebaseline.sh` | `contention_v2` | See memory note: ECT is not a ceiling. |
| **encoder_ablation** | `important/run_gnn_encoder_ablation.sh`, `compare_encoder_ablation.py` | contention series | Is the graph encoder doing work, or is it the features? |
| **seed_variance** | `scripts_cosim/run_gnn_seed_variance_siv1.sh` | contention_v2 | Uses `train_near_rtt_v2_contention_v2_dim14_ce_only.py`. |
| **queue_feature_contract** | `src/placement/queue_features.py`, `scripts_cosim/test_queue_features.py`, `verify_cache_live_feature_parity.py` | all | `legacy_v0` vs `scale_invariant_v1`. See CLAUDE.md. |
| **dataset_metadata** | `scripts_cosim/{extract_dataset_metadata,validate_dataset_collection,compute_compatibility_matrix}.py` | all | Produces `REGISTRY.json`, `METADATA.json`, `COMPATIBILITY_MATRIX.json`. |

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

## Conventions

**A lineage is not finished until it has a row in this table with an outcome.** A sweep
whose result was never written down will be re-run by someone in three months.

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
