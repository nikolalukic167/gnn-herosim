# fullctx_refine_v1 — does training the burst-seat GNN on full-batch context make its self-refine beat CD?

**Status:** `ACTIVE`. Registered 2026-09-25; the live gate was read 2026-09-26.
- **F1 `CLOSES-GAP`:** `fc1load_selfref` vs CD −1.76 %, p = 0.76, faster on 6/11 topologies.
- **F2 `TRAINING-HELPS` (direction only):** vs `bc1load_selfref` −2.23 %, p = 0.002, faster on 10/11.

This is the first learned burst-seat arm whose median sits on the fast side of CD. It is not
separated from CD. It stays behind CD on 9461 (+24 %), 9434 (+8 %) and 9456 (+8 %). Every bar was
signed before any datum of this arm existed.

**Parent:** [`backlog_corpus_v1`](backlog_corpus_v1.md). The diagnosis of its L-read found that 86 % of
`bc1load`'s live queue gap to CD is in-batch stacking. That lineage's Amendment 3 then showed that
self-refine on the same weights (`GNN_PREFIX_SELF_REFINE=3`), which re-scores each task with every
other batch-mate committed, is not separated from CD: +1.50 %, p = 0.15, 5/11, −3.4 % vs `bc1load`.
Self-refine scores a context the model never trained on. The prefix CE only ever commits EARLIER
tasks, so at refine time `committed_s` holds later tasks' service and the peer-mass column is ~0.
Earlier, [`cd_gap_v1`](cd_gap_v1.md) D6 gave self-refine 8.9 % of CD's refine gain for `gnnedge0`,
which had no committed-service column.

## Design

- **Arm `fc1load`** (4 seeds): `bc1load`'s recipe verbatim (the `backlog_corpus_v1` cache and split,
  `partial_state_v4`, the `gnnedge0` architecture) plus `NEAR_RTT_FULL_CONTEXT_CE_WEIGHT=1.0`.
  - The added term is a second any-of-K CE over the same tied plans, in which each task is scored with
    every other batch-mate committed where the plan puts it. That is exactly the context
    `prefix_serving._self_refine` scores in, through the same `make_partial_state_score_fn`.
  - Train batches only. Val CE and checkpoint selection stay on the one-pass prefix decode
    (`val/regret_masked_topo`). Disclosed: the selector does not see self-refine.
  - Config `experiments/fullctx_refine_v1_fc1load.yaml`, job
    `scripts_cosim/datalab/fullctx_refine_v1_train.sbatch`. The sidecar records
    `full_context_ce_weight`.
- **Served** with `service_end_v1`, as `fc1load_selfref` (3 passes, the registered arm) and plain
  `fc1load`, on the `fresh_topo_burst_v1` study (phase `fc1` of `backlog_corpus_v1_gate.sbatch`).
  CD, self-predict, `bc1load` and `bc1load_selfref` are the existing `backlog_corpus_v1` runs, at the
  same code for the default path.
- **Before the gate:** P1-style parity (`peer_affinity_live_serve_check.py`, 60 held-out batches,
  `fc1load` s1) must be clean.

## Bars (signed 2026-09-25, before any datum)

| read | contrast | fires as |
|---|---|---|
| **F1** | `fc1load_selfref` vs CD | `BEATS-CD` (median < 0, p < 0.05) / `CLOSES-GAP` (median ≤ 0 or p ≥ 0.05) / `CD-FASTER` |
| **F2** | `fc1load_selfref` vs `bc1load_selfref` (same seed) | `TRAINING-HELPS` (≤ −5 %, p < 0.05) / `TRAINING-HELPS (direction only)` / `NOT-SEPARATED` / `TRAINING-HURTS` |
| F3 | `fc1load` vs `bc1load` (one-pass decode) | reported |
| F4 | `fc1load_selfref` vs self-predict | reported |

- The reader is `scripts_cosim/backlog_corpus_v1_read.py`, with the same statistic and drop rule as
  its parent.
- **Expectations:**
  - F1: `BEATS-CD` 15 %, `CLOSES-GAP` 50 %, `CD-FASTER` 35 %.
  - F2: `TRAINING-HELPS` (either form) 45 %, `NOT-SEPARATED` 45 %, `TRAINING-HURTS` 10 %.
- **Standing risks:**
  - The parent's diagnosis puts the other half of the gap in the score itself. On topology 9461, CD
    reaches lower exchange *and* lower queue, and self-refine does not help there. Exchange enters
    as `committed_x / peer_norm`, which is normalised per batch.
  - Training on full-batch context cannot fix a score that mis-weights exchange against queue.

## Entry points

- Loss: `src/notebooks/train_near_rtt.py` (`loss_tied_teacher_forced_ce`, `full_context_weight`).
  Determinism: `tests/test_trainer_determinism.py::test_full_context_ce_is_reproducible_and_changes_the_fit`.
- Serving self-refine: `src/policy/gnn/prefix_serving.py` (`_self_refine`).

## Record (newest first)

- 2026-09-26 — **Live gate read** (job 808242, phase `fc1`, 47 min, at b2c395d).
  - **Runs:** 380 of 384 wrote summaries. The 4 misses are plain `fc1load` on `cc40s9423__w3`, seeds 1–4,
    all 1800 s timeouts; plain `bc1load` times out on the same cell. The reader's drop rule removes them.
  - **Attachment:** [`fc1_read.json`](fullctx_refine_v1/fc1_read.json).
  - **F1 `CLOSES-GAP`:** `fc1load_selfref` vs CD −1.76 %, p = 0.76, 6/11 faster, 9466 dropped (CD w3).
    - Per topology (%): 9119 −2.7, 9414 −1.8, 9420 +1.9, 9423 −3.2, 9434 +8.2, 9435 −5.5, 9444 −1.8,
      9446 +1.9, 9456 +7.8, 9461 +24.2, 9469 −2.7.
    - Pooled means: elapsed 7.43 vs 7.63 s; queue 3.09 vs 2.91 s; exchange 4.09 vs 4.18 s per task.
    - Where it wins, the win comes from lower exchange; the queue is still 0.18 s longer than CD's.
  - **F2 `TRAINING-HELPS` (direction only):** vs `bc1load_selfref` −2.23 %, p = 0.002, 10/11 faster
    (9434 +0.1).
    - The largest gains are on the slow topologies: 9461 −10.3 % and 9456 −7.3 %.
    - The gain is mostly exchange (4.21 → 4.09 s); queue moved 3.13 → 3.09 s.
  - **F3 (reported), plain `fc1load` vs `bc1load`:** −0.04 %, p = 0.90, 6/11. The full-context term
    does nothing for the one-pass decode. It pays only when the model revises, which is the context
    it was trained for.
  - **F4 (reported), `fc1load_selfref` vs self-predict:** −13.87 %, p = 0.0005, 12/12. This is the
    fastest learned arm against the burst-seat bar.
  - **Reading:** training on the revision context moves the self-refined GNN from +1.5 % to −1.8 % vs
    CD. That is a tie in the paired statistic, and the tail sits on the same three topologies.
    - 9461 improved from +44.6 % to +24.2 % but is still the dominant loss. This fits the standing
      risk: exchange enters as `committed_x / peer_norm`, normalised per batch.
    - Next lever, not registered: exchange in seconds rather than divided by `peer_norm`.

- 2026-09-26 — Trained (job 808175; 2 h 40 min to 4 h 52 min per seed). Best val `regret_masked_topo` was
  7.23 / 7.07 / 6.93 / 7.23 s.
  - **Offline one-pass decode** (the 5032-group eval; this is not the served arm):
    - median regret 5.74 / 5.25 / 5.47 / 5.68 %, against `bc1load`'s 4.82 / 5.32 / 4.92 / 4.40 %;
    - mean regret 25.9 / 24.5 / 25.2 / 24.3 %, against 24.5 / 24.3 / 25.2 / 21.8 %.
  - The full-context term costs the prefix decode about half a point. This is recorded, not a bar.
  - **P1 parity** (`fc1_servecheck_fc1load_s1.json`): 60/60 bit-identical.
  - Live gate phase `fc1` submitted as job 808242.

- 2026-09-25 — Registered. The default loss is unchanged: weight 0 skips the term, and the trainer
  determinism suite is 18/18 including the new case. Same seed gives identical weights with the term
  on, and different weights from the term off.
