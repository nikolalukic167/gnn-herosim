# x11_confirm_v1 — does the learned arm's ×1.1 burst-load win over CD hold on the seeds nobody has seen?

**Status:** `ACTIVE` (2026-09-27). Registered; the gate is running. Every bar below was signed before any
datum of seeds 2–4 at this rung existed.

**Parent:** [`capacity_sweep_v1`](capacity_sweep_v1.md). At w0 ×1.1 (4 perturbed arrival draws; reactive
admissible on 11/12 topologies), `xs1load_selfref` **seed 1** beat CD −13.7 % (p = 0.042, 10/12). That is
the one burst-seat number against CD that clears both the 5 % bar and p < 0.05 at an admissible load, and
it rests on one learned seed. It was picked out of a ladder of rungs *after* being seen, so it carries a
winner's-curse risk.

## Design

- **New runs:** `xs1load_selfref` seeds 2, 3, 4 on the same `w0x11d1`–`d4` draws and the same 12 study
  topologies (144 runs), phase `x11confirm` of `backlog_corpus_v1_gate.sbatch`. The checkpoints are
  `exchange_seconds_v1`'s, unchanged.
- **Reused runs:** CD, reactive and seed 1 at those draws from `capacity_sweep_v1` (gate dir `capacity`).
- **Witness:** seed 1 on `w0x11d1` for 9119 and 9420, rerun at this lineage's commit, must equal the
  `capacity` runs to the digit. The code between them is a series of merges whose new flags are all off by
  default; the witness proves the served learned path did not move.
- **Statistic** (`scripts_cosim/x11_confirm_v1_read.py`): per topology, the median over every (seed, draw)
  of the paired % vs CD on the same draw; exact two-sided Wilcoxon over topologies; a topology missing any
  required run is dropped by name.

## Bars (signed 2026-09-27, before any datum)

| read | seeds | fires as |
|---|---|---|
| **Q1** | 2, 3, 4 (unseen) | `CONFIRMED` (median ≤ −5 %, p < 0.05) / `DIRECTION-ONLY` (median < 0, p < 0.05) / `NOT-CONFIRMED` |
| Q2 | 1–4 | reported |
| Q3 | each seed alone; reactive queue share per topology | reported |
| W | seed 1 witness | must pass, or the read is void |

- **Expectations:** Q1 `CONFIRMED` 35 %, `DIRECTION-ONLY` 35 %, `NOT-CONFIRMED` 30 %. The ×1 four-seed read
  was −10.9 % (p = 0.23, `replica_guard_v1` twin), dragged by 9434's seed-dependent collapse, and ×1.1 is
  closer to the load where that collapse bites.
- **Standing risks:**
  - The 9434 collapse is seed-dependent; seeds 2–4 may collapse where seed 1 did not, or the reverse.
  - Rule 6 is met: these are live runs. A `CONFIRMED` read is a quotable burst-seat result at one
    admissible rung of one window, on one checkpoint family — not a program-level verdict.

## Record (newest first)

- 2026-09-27 — Registered.
