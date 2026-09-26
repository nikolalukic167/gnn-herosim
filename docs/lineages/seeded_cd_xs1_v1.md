# seeded_cd_xs1_v1 — does CD refinement started from the strongest learned plan beat CD?

**Status:** `ACTIVE` (2026-09-26). Registered; the gate is running. Every bar below was signed before
any datum of this arm existed.

**Parents:**
- [`exchange_seconds_v1`](exchange_seconds_v1.md): `xs1load_selfref` ties CD (−0.94 %, p = 0.92). It is
  the strongest learned burst-seat arm, −14.6 % vs self-predict.
- [`cd_gap_v1`](cd_gap_v1.md) D5: CD's refine passes started from `gnnedge0`'s plan beat CD
  −3.83 % (10/10, direction only). About two thirds of that came from any learned seed: MP-OFF-seeded
  CD read −2.56 %.

Three levers on the learned score each moved the median a little, and every one tied CD. The remaining
loss sits on 9461, 9456 and 9434, where CD's refine passes find what a per-task score does not. This
lineage stops competing with CD and asks whether the learned plan is a better *start* for CD than CD's
own greedy pass, now that the seed is `xs1load` rather than `gnnedge0`.

## Design

- **Arm `xs1load_cdapply`** (4 seeds): the `exchange_seconds_v1` checkpoints, unchanged and served as
  in that gate (`service_end_v1`, `partial_state_v4`, exchange in seconds), with `GNN_CD_REFINE=apply`.
  - After the GNN decodes a batch, CD's refine greedy runs 3 passes from that plan on the same live
    state, and the refined plan is served (`GnnCdRefiner`, `cd_gap_v1` D5's code path). There is no
    self-refine, so the seed is the one-pass decode.
  - CD's refine prices the queue with `platform_queue_drain_seconds`, which never reads the in-flight
    capture mode. The seed is the only thing that differs from CD.
- **Seed control `bc1mpoff_cdapply`** (4 seeds, reported): the MP-OFF `bc1mpoff` checkpoints under the
  same refine.
  - Disclosed: this is not a same-recipe twin of `xs1load`. It lacks the full-context CE and the seconds
    encoding, and it has no message passing, so it separates "a learned seed" from "this seed" only
    loosely.
  - No MP-OFF `xs1load` exists. The read quoted is a learned-seed contrast, never a message-passing
    one (see `gnn_seeded_cd_mp_ablation_v1`).
- **Where it runs:** the `fresh_topo_burst_v1` study, phase `xs1cd` of `backlog_corpus_v1_gate.sbatch`.
  CD, self-predict and `xs1load_selfref` are the existing runs.
- No new training, so no new parity check: `xs1load` s1 passed P1 on 2026-09-26. The `cdapply` instrument
  (`cdr_batches > 0`) is checked per run by the driver.

## Bars (signed 2026-09-26, before any datum)

| read | contrast | fires as |
|---|---|---|
| **G1** | `xs1load_cdapply` vs CD | `BEATS-CD` (median ≤ −5 %, p < 0.05) / `BEATS-CD (direction only)` (median < 0, p < 0.05) / `TIES-CD` (p ≥ 0.05) / `CD-FASTER` (median > 0, p < 0.05) |
| G2 | `xs1load_cdapply` vs `xs1load_selfref` | reported: what CD's refine adds on top of the model's own revision |
| G3 | `xs1load_cdapply` vs `bc1mpoff_cdapply` | reported: the seed contrast, disclosed as confounded (above) |
| G4 | `bc1mpoff_cdapply` vs CD; `xs1load_cdapply` vs self-predict | reported |

- Also reported: the per-topology G1 on 9461, 9456 and 9434.
- The reader is `scripts_cosim/backlog_corpus_v1_read.py`, with the same statistic and drop rule as the
  parents.
- **Expectations:** G1 `BEATS-CD` 15 %, `BEATS-CD (direction only)` 55 %, `TIES-CD` 25 %, `CD-FASTER` 5 %.
- **Standing risks:**
  - This is not a learned policy on its own; the result is a learned *seed* for a hand search, the
    `gnn_seeded_cd_fixed_v1` framing.
  - 4 seeds on 11 topologies: quote a direction, not a magnitude, unless G1 clears the 5 % bar.
  - `cc40s9423__w3` has timed out for several learned arms at 1800 s. The drop rule applies.

## Record (newest first)

- 2026-09-26 — Registered.
