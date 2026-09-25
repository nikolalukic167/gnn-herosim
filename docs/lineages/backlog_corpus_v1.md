# backlog_corpus_v1 — does training on states that carry backlog teach the burst-seat GNN to load-balance?

**Status:** `ACTIVE` (2026-09-25). Corpus and cache built; **O1 and O2 fire**. Amendment 1 (serve-only
`v4load_se` arm, L1 read at fixed capture, parity check P1) signed; training, P1 and the live gate
(L1–L3) are next. Every bar below was signed before any datum of this corpus existed.

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
| **L1** | `bc1load` vs `v4load`, paired (Amendment 1: vs `v4load_se`, same capture) | `CORPUS-HELPS` (≤ −5 %, p < 0.05) / `CORPUS-HELPS (direction only)` (p < 0.05) / `NOT-SEPARATED` / `CORPUS-HURTS` |
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

- 2026-09-25 — **Amendment 3 (signed before any datum of the arm): `bc1load_selfref`.** This is
  `bc1load` served with `GNN_PREFIX_SELF_REFINE=3`: after the id-order decode, 3 passes re-score each
  task with every other batch-mate committed, on the model's own v4 score. It uses the four `bc1load`
  checkpoints, `service_end_v1`, gate phase `bc1selfref` and no new training.
  - **Reads** (`backlog_corpus_v1_read.py`, the same statistic and drop rule):

    | read | contrast | fires as |
    |---|---|---|
    | **S1** | `bc1load_selfref` vs CD | `CLOSES-GAP` if not slower (median ≤ 0 or p ≥ 0.05); else `CD-FASTER` (with the direction-only qualifier if \|median\| < 5 %) |
    | **S2** | `bc1load_selfref` vs `bc1load` (same seed) | reported |
    | S3 | `bc1load_selfref` vs self-predict | reported |

  - **Why:** the diagnosis below found that one-pass id-order decode is about half the queue gap.
    A 3-environment × 2-seed probe (job 808107) gave self-refine −2.3 to −6.9 % vs `bc1load` and
    −1.8 to +5.1 % vs CD, recovering 32–66 % of CD's refine gain from the same seed (`gnnedge0`
    in `cd_gap_v1` D6: 8.9 %). It is a direction only; the probe environments are inside the
    study.
  - **Caveat:** self-refine scores with later batch-mates committed, a context absent from training.
  - **Expectations:** S1 `CLOSES-GAP` 35 %, `CD-FASTER` (direction only) 50 %, `CD-FASTER` 15 %.
    S2 faster 90 %.

- 2026-09-25 — **Diagnosis of the L-read** (five read-only investigations; scratch under
  `/tmp/claude-0/agent_*`, cluster copies in `simulation_data/backlog_corpus_v1/debug_*`).
  - **The live queue gap to CD is in-batch stacking.** It is 86 % of the gap over 5 probe runs
    (job 808083): tasks of one burst wait 5–7 s behind a batch-mate's peer exchange on the same
    platform.
    - Inherited backlog is 14 %, and only as a knock-on of the model's own piling.
    - On bursts that start with more than a quarter of candidates busy (8–15 % of bursts), the model
      is already faster than CD by 0.8–4.3 s/task. The corpus fixed the regime it aimed at, and that
      regime is small.
    - CD's shadow refine would change 53–58 % of the model's batches, un-stacking and trading a little
      exchange for less queue.
  - **Not the cause:**
    - The label: on the held-out states, a CD replay is worse than `bc1load` on raw RTT, on the label
      and on residual busy time.
    - Replay mechanics: offline, a stacked task waits as long as it does live (4.6–5.6 s median). A
      stacking penalty of 1–4 s moves the label's same-platform share only 0.671 → 0.672.
    - Train/serve code: every test passes against the base commit, the load inputs are identical,
      and the `service_end_v1` predicted ends match the actual ends within 1 ms on 2,999/3,000
      tasks.
  - **Real mismatches, minor:**
    - Live batches are not always complete: 27 % of tasks are in batches smaller than 10.
    - Training states are much more loaded: 49 % of candidates busy vs 13 % live.
    - A replica busy only from its in-flight task (queue count 0, backlog > 0) never appears in
      training.
    - Platform column 8 `node_cold_frac` is identically 0 in training and ~1 live; zeroing it
      changes 0.9 % of plans.
    - The capture mode is not in the checkpoint sidecar.
    - Rung "3" is 4.5 s in effect: with the Poisson rate clamped, k = 1.
  - **Levers named:**
    - (a) The decode order, tested by Amendment 3.
    - (b) An on-policy corpus from the model's own live states (it stacks 0.70 offline and 0.80 live,
      against CD's 0.62 live).
    - (c) Training on full-batch context, so that self-refine is in distribution.

- 2026-09-25 — **Live read (L1–L3, C1): `NOT-SEPARATED` / `NO-EFFECT`.** Job 807878 at 0f9c9c9:
  569/576 runs, and the 7 timeouts are all 9423 w3 (4 × `bc1load`, 3 × `v4load_se`), so 9423 drops
  as it did for `v4load`. [Read](backlog_corpus_v1/bc1_read.json).

  | read | median | p | first faster | label |
  |---|---|---|---|---|
  | **L1** `bc1load` vs `v4load_se` | +0.83 % | 0.067 | 2/11 | `NOT-SEPARATED` |
  | **L2** `bc1load` vs CD | +6.59 % | 0.002 | 0/10 | `NO-EFFECT` |
  | L3 `bc1load` vs `bc1mpoff` | **−5.03 %** | 0.001 | 11/11 | reported |
  | C1 `v4load_se` vs `v4load` | −0.82 % | 0.001 | 11/11 | reported (direction) |
  | `bc1load` vs self-predict | −7.55 % | 0.005 | 9/11 | reported |
  | `bc1load` vs `v4load` | −0.56 % | 0.70 | 8/11 | reported |
  | `bc1mpoff` vs CD | +10.29 % | 0.001 | 0/11 | reported |
  | `v4load_se` vs CD | +5.82 % | 0.004 | 1/10 | reported |

  - **Queue per task:** `bc1load` 3.57 s, `v4load_se` 3.53 s, CD 2.99 s. Exchange is flat at
    4.1–4.2 s.
  - The corpus did not move the live queue.
  - MP beats its twin at the 5 % magnitude bar for the first time in the burst seat. This is 4 seeds
    on one corpus; queue is 3.57 vs 3.88 s and exchange 4.18 vs 4.32 s.

- 2026-09-25 — **Trained; P1 passes; O3 mixed (offline, orders nothing).** The live gate (job 807878,
  phase `bc1`) is running.
  - **Training:** job 807608 at 0f9c9c9, all 8 tasks passed their sidecar checks, W&B project
    `gnn-peer-affinity-v1`, tag `backlog-corpus-v1`. Best val served regret (`val/regret_masked_topo`,
    mean s, 1,186 whole-cell val groups):

    | arm | s1 | s2 | s3 | s4 |
    |---|---|---|---|---|
    | `bc1load` | 6.81 | 7.07 | 6.96 | 6.92 |
    | `bc1mpoff` | 7.57 | 7.50 | 7.74 | 7.71 |

    On validation, every `bc1load` seed is ahead of every `bc1mpoff` seed. The 8 failed W&B runs of
    the same names are the Amendment-2 crash (807583).
  - **P1 passes** (job 807618). `bc1load` s1 and `bc1mpoff` s1, served with `service_end_v1`, are
    each 60/60 bit-identical on held-out batches: fields including `backlog_s`/`service_s`, the
    decoded plan and the sweep RTT. 46 of the 60 carry synthetic backlog (rungs 3/8/20: 14/21/11).
  - **O3** (job 807617; 960 test groups, cells 9223/9224; registered regret %):

    | arm | median per seed | mean per seed |
    |---|---|---|
    | `bc1load` | 4.7 / 4.7 / 4.5 / 4.0 | 25.1 / 24.8 / 26.3 / 20.7 |
    | `bc1mpoff` | 5.3 / 5.4 / 5.6 / 5.6 | 22.6 / 20.1 / 23.4 / 23.8 |

    - **Paired** (same seed): `bc1load` wins 913, loses 689 and ties 2,238, with mean +1.75 pp. The
      median favours message passing and the mean, through a heavy tail, does not.
    - **By rung** (median / mean %):

      | rung | `bc1load` | `bc1mpoff` |
      |---|---|---|
      | 0 | 5.9 / 46.0 | 7.8 / 42.1 |
      | 3 | 6.3 / 21.5 | 6.5 / 18.7 |
      | 8 | 4.1 / 15.2 | 5.0 / 15.5 |
      | 20 | 2.2 / 16.5 | 2.9 / 15.6 |

    - Not comparable to `load_repr_v1` O1: different test groups and a different label clock.

- 2026-09-25 — **Amendment 2 (signed before any trained weights exist).** The first training array
  (807583, f5edef6) failed on all 8 tasks at the first training batch: `partial_state_v4 requires a
  peer_exchange corpus`.
  - **Cause:** 4 training groups have every candidate on one node, so the exchange table is zero and
    the cache writes `peer_norm` 0. They are two snapshots, `pgb_cc40s9203_w0` #18 and
    `pgb_cc40s9204_w0` #8, one per backlog pass (`ds_00807`, `ds_01201`, `ds_20807`, `ds_21201`).
    Each is a 16-plan sweep on one server. These are most likely the 4 groups the audit found with
    no jb2 twin. No val or test group is affected.
  - **Fix:**
    - The 4 dataset dirs move to `simulation_data/backlog_corpus_v1/quarantine_single_node/`.
    - The cache, split and audit are rebuilt without them.
    - The cache job now asserts `peer_norm > 0` on every graph.
    - The old split (sha 2fccc095…) is retired and the new one is committed before training.
  - **Default-path witness** (`backlog_corpus_v1_witness.sbatch`, runs alongside the rebuild): CD s0
    and `v4load` s1 on 9119 w1 must reproduce their earlier summaries to the digit at this branch's
    code, and a `v4load_se` s1 smoke must record `service_end_v1`. Without it, the reuse of CD,
    self-predict and `v4load` gate runs is not licensed.
  - Bars and labels are unchanged.
  - **Rebuilt** (job 807594, 75f9caa): 5,032 graphs, backlog > 0 on 51.3 % of candidate replicas,
    candidate guard 0 offenders. **Split** `experiments/backlog_corpus_v1_split.json` sha256
    9c459b16… is train 2,886, val 1,186 (the same cells, 9207/9208/9209/9213) and test 960. The audit
    now finds a jb2 twin for every group (`missing_twin` 0), so the 4 quarantined groups were exactly
    the twinless ones. O1/O2 are unchanged to the reported precision.
  - **Witness passed** (job 807595, 75f9caa), with 9119 w1 reproduced to the digit:
    - CD s0: 440055.157;
    - `v4load` s1: 452534.104;
    - `v4load_se` s1 ran clean with `HEROSIM_INFLIGHT_CAPTURE=service_end_v1` recorded (one
      environment, not a read).
  - The CD and self-predict runs of the fresh gate (1ae90af, 48 environments each) are copied
    md5-verified to datalab `simulation_data/backlog_corpus_v1/gates/ref_fresh_1ae90af/`. Reader:
    `scripts_cosim/backlog_corpus_v1_read.py`, which reproduces `cd_gap_v1`'s `gnnedge0` vs CD
    +11.42 % (11 topologies).

- 2026-09-25 — **Amendment 1 (signed before any model of this lineage is trained).** `load_repr_v1`
  closed `LOAD-HELPS / NARROWS` (`v4load` −5.3 % vs its twin, +6.7 % vs CD, gain and remaining gap
  both queue). Two facts from it change the live design:
  - **The capture mode confounds L1 as registered.** `v4load` was gated with the legacy capture;
    `bc1load` is served with `service_end_v1`. L1 would mix corpus with capture. CD is not affected:
    its score is `platform_queue_drain_seconds` (queued items + seeded warmup), which never reads the
    in-flight task.
  - **`v4load`'s backlog column was served populated but trained nearly empty.** Live (fixed-serving
    gate 807153) ~0.43 candidates per batch carry backlog > 0; in jb2, 0.5 % of replica specs do.
    The reading that `v4load`'s gain came mostly from the committed-service column is inference, not
    measured column by column.
  - **Design change:**
    - New serve-only arm **`v4load_se`**: the four `load_repr_v1` `v4load` checkpoints served with
      `HEROSIM_INFLIGHT_CAPTURE=service_end_v1`. No training.
    - **L1 is read as `bc1load` vs `v4load_se`** (capture held fixed; corpus is the only difference).
      Bars and labels unchanged.
    - **C1 (reported):** `v4load_se` vs `v4load` as gated in 807153 (capture effect on a jb2-trained
      model).
    - **P1, parity, required before the gate:** `peer_affinity_live_serve_check.py` on 60 held-out
      batches of this cache for `bc1load` s1, served with `service_end_v1`. `backlog_s` and
      `service_s` must match on 60/60 and the plan on ≥ 58/60, with any leftover a same-node sibling
      tie-break whose live RTT equals the sweep row (`load_repr_v1`'s standard). A failure blocks the
      gate and is fixed, never waived.
    - **O3 (orders nothing):** held-out regret on this lineage's test cells, both arms, A1's protocol.
  - **Training:** `experiments/backlog_corpus_v1_{bc1load,bc1mpoff}.yaml` are `v4load`'s recipe and
    jb2 mpoff's swap on this cache and split; `scripts_cosim/datalab/backlog_corpus_v1_train.sbatch`,
    seeds 1–4, sidecar checks as `load_repr_v1`.
  - **Disclosed, not controlled:** corpus size (2,890 train groups vs jb2's 1,629) and split
    (whole-cell val vs jb2's random val). Every quote names both corpora. A size-matched arm is not
    registered here. The serve-only zeroing of the backlog column alone is not run (it needs a new
    serving knob).
  - **Expectations:** C1 `NOT-SEPARATED` 50 %, `v4load_se` slower 30 %, faster 20 %. L1 and L2 as
    registered.

- 2026-09-25 — **Corpus and cache built; O1 and O2 fire** (offline, orders only).
  - **Corpus** (job 807182, 12 tasks × 8 units, code eb9fa58): 5,036 groups (4,076 train, 960
    held-out), 2 × jb2's 2,516; 132 snapshots rejected by the jb2 filters. Rungs: 0 s 1,160 · 3 s
    1,371 · 8 s 1,558 · 20 s 947.
  - **Cache** (job 807183) `graphs_cache_backlog_corpus_v1_psv4_inf`: label
    `rtt_drift:1@lambda=0.46,clock=measured,backlog=seeded`, 0 incomplete sweeps, 0 candidate-guard
    offenders, near-RTT sidecar 562,064 rows.
  - **Split** `experiments/backlog_corpus_v1_split.json` (sha256 2fccc095…): train 2,890, val 1,186
    (whole cells 9207, 9208, 9209, 9213; 29 % of training groups, since cells are indivisible),
    test 960 (cells 9223/9224).
  - **O1 fires.** 13,608 of 26,558 candidate replicas (51.2 %) carry v4 backlog > 0 (jb2: 0.5 %).
  - **O2 fires** (`backlog_corpus_v1/audit.json`). At rungs 3/8/20 the raw-RTT optimum moves vs its
    jb2 twin in 46 % / 57 % / 71 % of groups (bar 20 %). It puts 46 % / 42 % / 38 % of its tasks on
    busy replicas against 51 % / 50 % / 50 % for a random plan. Playing the twin's (no-backlog)
    optimum costs a mean +2.9 / +7.6 / +22.0 s (median 0.0 / 1.7 / 13.0 s).
  - **Rung 0 is the jb2 control**: identical optimum cost in every group (ratio 1.000, twin regret 0);
    the 7.5 % "moved" there are ties broken in a different row order. 4 datasets had no jb2 twin.
  - **Smoke defect fixed before the run** (e6a39f1): the measured drain table has no `xavierDla` rows,
    so seeding a synthetic backlog there tripped the table's coverage guard. A synthetic spec now
    seeds by its seconds. A rung-0 replay reproduced jb2's optimum to the digit (54.394 s).

- 2026-09-25 — Registered. The trainer's served-decode logging (`val/mt_task_acc_choice`,
  `val/mt_plan_exact`, `val/regret_masked_topo_median`, the `readout/*` summary keys, and the fixed
  `checkpoint_metric` for teacher-forced runs) lands on this branch. `tests/test_backlog_corpus.py`
  11/11; trainer determinism, drift label, warm corpus and `partial_state_v4` suites pass.
