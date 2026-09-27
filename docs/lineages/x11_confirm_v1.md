# x11_confirm_v1 — does the learned arm's ×1.1 burst-load win over CD hold on the seeds nobody has seen?

**Status:** `CLOSED` (2026-09-27) — **Q1 `NOT-CONFIRMED`.** Registered and gated 2026-09-27; every bar below
was signed before any datum of seeds 2–4 at this rung existed.
- **Q1 (seeds 2–4, unseen):** learned vs CD −11.47 %, p = 0.092, faster on 9/12. The magnitude replicates
  seed 1's −13.7 %; the significance does not.
- **Q2 (seeds 1–4):** −11.81 %, p = 0.064, 10/12. Each seed alone: −13.7 / −8.4 / −8.8 / −11.1 %, and only
  seed 1 reaches p < 0.05.
- **Why it misses:** 9434 collapses on every seed (+120 % to +278 % vs CD), and 9456 (+5 %) and 9466 (+5 %)
  trail. On the other 9 topologies the lead is −4 % to −34 %, consistent across seeds.
- **So:** a consistent ~−11 % median lead over CD at ×1.1 burst load, held back from significance by one
  topology's seed-independent collapse. `capacity_sweep_v1`'s ×1.1 number must not be quoted as significant.

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

- 2026-09-27 — **Gate read** (job 809663, phase `x11confirm`, 146/146 runs, 12 min, at 99f05ae).
  - **Attachment:** [`x11confirm_read.json`](x11_confirm_v1/x11confirm_read.json).
  - **Witness passes:** seed 1 on `w0x11d1` reruns to the digit on 9119 (8.7267 s) and 9420 (5.7419 s).
  - **Q1 `NOT-CONFIRMED`:** −11.47 %, p = 0.092, 9/12, none dropped. Per topology (%): 9119 −12.7,
    9414 −11.9, 9420 −4.1, 9423 −13.6, 9434 +159.3, 9435 −31.4, 9444 −11.1, 9446 −10.1, 9456 +5.4,
    9461 −34.2, 9466 +4.7, 9469 −15.6.
  - **Q2:** −11.81 %, p = 0.064, 10/12.
  - **Q3:** seed 1 −13.73 % (p = 0.042), seed 2 −8.38 % (p = 0.064), seed 3 −8.79 % (p = 0.092),
    seed 4 −11.05 % (p = 0.092). Reactive admissible on 11/12 (9423 at 0.81).
  - **Reading:** seed 1 was the favourable draw of four; the winner's-curse risk registered above is what
    happened to the p-value, not to the effect size. The single blocker is the 9434 replica-expiry collapse
    (`replica_guard_v1`), which now appears on every seed at this load. Fixing it through training is the
    lever; on the 9 non-collapsing, non-9456/9466 topologies the lead is large and seed-stable.
- 2026-09-27 — Registered.
