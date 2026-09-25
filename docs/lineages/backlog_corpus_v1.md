# backlog_corpus_v1 — does training on states that carry backlog teach the burst-seat GNN to load-balance?

**Status:** `REGISTERED` (2026-09-25). Corpus generation launched; every bar below was signed before
any datum of this corpus existed.

**Parents:** [`load_repr_v1`](load_repr_v1.md) (the `partial_state_v4` load columns; its standing risk,
measured there: only 0.5 % of jb2 replica specs carry backlog > 0) and [`cd_gap_v1`](cd_gap_v1.md) (the
gap to CD is all queue). `docs/hard-stops.md` (`peer_affinity_warm_v1`) names a drainable served regime
as a new environment question, which this is.

## Why the jb2 states are empty

- **Capture defect.** `Task.started` fires before the input + peer-exchange stage
  (`Platform.platform_process`), and the temporal capture prices an in-flight task as execution minus
  time since `started`. Exchange is ~4 s per task, so a platform still exchanging reads idle. In a
  102-dataset jb2 sample, 0 of 23,358 platform specs carried an in-flight task.
- **Seat dynamics.** A burst arrives every ~22 s and a group drains in ~8 s, so little queue carries over.

## Design

- **Synthetic backlog** (`live_snapshot_seed.inject_synthetic_backlog`, `make_warm_corpus.py
  --synthetic-backlog-*`). Per dataset a rung is drawn from {0, 3, 8, 20} s. Each platform of the
  replayed cluster is busy with probability 0.5; a busy one gets k = 1 + Poisson(rung/4.5 − 1) fake
  queued tasks, each draining Gamma(2, 2.25) s (the live execution + exchange scale). The fields
  `synthetic_queue_length` / `synthetic_backlog_seconds` enter:
  - the co-sim replay: the platform is busy that long before it serves the batch, so every sweep
    plan's RTT includes the wait, and `queue_length()` counts the fake tasks (the legacy count column);
  - the `v4` backlog columns, through the same `seeded_backlog_seconds`;
  - the drift label's B_p, on a new **seeded** clock (`NEAR_RTT_LABEL_BACKLOG_CLOCK=seeded`, label
    string `…,backlog=seeded`). A synthetic corpus on the old `counts` clock raises.
- **Corpus.** The jb2 recipe verbatim (same snapshot shards, traces, cells, candidate draw seed 9001,
  `--no-cap-filter`, min-choice 0.2), two passes with backlog seeds 7001 and 7002. Every dataset's
  no-backlog twin is the jb2 dataset of the same snapshot. Up to ~5,000 groups.
- **Cache.** `load_repr_v1`'s recipe (`partial_state_v4`, alpha `inf`, `rtt_drift:1` at 0.46/s,
  measured drain table) with the seeded backlog clock.
- **Split.** Test = held-out cells 9223/9224. Val = whole training cells (`--val-group-by cell`), so
  no val group has a same-run neighbour in train (jb2: 387/407 did).
- **Capture fix for serving** (`HEROSIM_INFLIGHT_CAPTURE=service_end_v1`, default `legacy`): the
  platform records when its in-flight service ends (`Platform.inflight_service_end`) and
  `live_audit.temporal_state_of` reads it, for both snapshot capture and `v4` serving. The default
  path is bit-identical. Every arm of this lineage is served with `service_end_v1`.

## Bars (signed 2026-09-25, before any datum)

Offline (orders only, rule 6), from `scripts_cosim/backlog_corpus_v1_audit.py`:

| read | fires as |
|---|---|
| **O1** backlog reaches the features | > 20 % of candidate replicas carry v4 backlog > 0 (asserted by the cache job) |
| **O2** backlog reaches the label | at rungs > 0, the raw-RTT optimum moves vs its jb2 twin in ≥ 20 % of groups, and the optimum puts less work on busy replicas than a random plan |

Live, on the `fresh_topo_burst_v1` study (12 topologies × 4 windows, base rate), 4 seeds per arm:
`bc1load` = the jb2 `gnnedge0` recipe with `partial_state_v4` on this cache; `bc1mpoff` = its MP-OFF
twin; comparators `load_repr_v1` `v4load` (same architecture, jb2 corpus), CD and self-predict. All
learned arms are served with `service_end_v1`.

| read | contrast | fires as |
|---|---|---|
| **L1** | `bc1load` vs `v4load`, paired | `CORPUS-HELPS` (≤ −5 %, p < 0.05) / `CORPUS-HELPS (direction only)` (p < 0.05) / `NOT-SEPARATED` / `CORPUS-HURTS` |
| **L2** | `bc1load` vs CD | `CLOSES-GAP` if not slower (median ≤ 0 or p ≥ 0.05), else `NARROWS` if L1 helps, else `NO-EFFECT` |
| L3 | `bc1load` vs `bc1mpoff` | reported; a backlog term is pointwise-expressible, so this is not a GNN-vs-MLP lever |

**Expectations, written before data.** O2 fires at rungs 8/20: 80 %. L1 `CORPUS-HELPS` (either form)
45 %, `NOT-SEPARATED` 45 %. L2 `CLOSES-GAP` 15 %. **Standing risk:** part of the live queue is
in-batch stacking, which the jb2 sweep already simulates; the inherited-vs-self-inflicted split of the
live queue has not been measured.

## Entry points

- Injection and replay: `src/placement/live_snapshot_seed.py` (`inject_synthetic_backlog`,
  `seeded_backlog_seconds`), `scripts_cosim/make_warm_corpus.py`.
- Label: `scripts_cosim/drift_label.py` (`BACKLOG_CLOCK_ENV`, `_seeded_backlog_by_pid`).
- Capture: `src/placement/infrastructure.py` (`inflight_service_end`), `src/placement/live_audit.py`
  (`inflight_capture_mode`, `temporal_state_of`).
- Jobs: `scripts_cosim/datalab/backlog_corpus_v1_{corpus,cache}.sbatch`; audit
  `scripts_cosim/backlog_corpus_v1_audit.py`; tests `tests/test_backlog_corpus.py`.
- Datasets: `simulation_data/gnn_datasets_backlog_corpus_v1_{train,heldout}`, cache
  `graphs_cache_backlog_corpus_v1_psv4_inf`, split `experiments/backlog_corpus_v1_split.json`.

## Record (newest first)

- 2026-09-25 — Registered. The trainer's served-decode logging (`val/mt_task_acc_choice`,
  `val/mt_plan_exact`, `val/regret_masked_topo_median`, the `readout/*` summary keys, and the fixed
  `checkpoint_metric` for teacher-forced runs) lands on this branch. `tests/test_backlog_corpus.py`
  11/11; trainer determinism, drift label, warm corpus and `partial_state_v4` suites pass.
