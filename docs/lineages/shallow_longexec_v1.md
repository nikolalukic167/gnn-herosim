# shallow_longexec_v1 — FALSIFIED

> **Status:** `FALSIFIED` &nbsp;·&nbsp; **Index:** [LINEAGES.md](../../LINEAGES.md) &nbsp;·&nbsp; **Record spans:** 2026-08-17 → 2026-08-17

**Outcome.** The last physics attempt before the paper pivot, and the **fourth independent confirmation** of the throughline: one-integer repair 100%.

**Related:** [throughline](throughline.md) · [shallow_v1](shallow_v1.md)

> **Split note.** This node was carved out of a single 4,995-line `LINEAGES.md` on
> 2026-08-27; the section bodies below are byte-for-byte as written. An *above* / *below*
> pointer inside one may refer to a section that now lives in another node — the
> **Related** links say which.

## Record

Newest first; the sections themselves are in chronological order below.

- [`shallow_longexec_v1` — unblocked (2026-08-17)](#shallow-longexec-v1-unblocked-2026-08-17)

---

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
