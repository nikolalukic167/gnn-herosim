# unsaturated_edge_v1 — the −20.8 % headline, at the power it was never read at

**Status:** `CLOSED` (2026-09-20) — **`HEADLINE-WAS-AN-UNPAIRED-STATISTIC` ·
`NO-LEARNED-ARM-BEATS-REACTIVE-AT-THE-EDGE-RUNG`.** Closed on a live gate (rule 6): 1,338 of
1,344 arms (6 hang deterministically, below). Registered 2026-09-19; bars, reader, 14 tests and
the expectation committed before any learned arm ran. **The registered expectation ("even odds
that E1 fires at 40 clients") was wrong**, and wrong for a reason neither alternative named.

**Outcome. The −20.8 % headline was never there.** Read PAIRED on the same cell, the four
w0 cells it came from give `gnnedge0` **+4.6 / +75.8 / +9.7 / −36.8 %** against Knative — it loses
three of four, one catastrophically, and wins only on `cc40s9005`, whose reactive queue share
(0.803) fails the admissibility bar this lineage applied. The old reader's `collapse_to_seed`
took the median of the arm's elapsed over cells (24.5 s) and divided by the median of Knative's
over cells (30.5 s): an **unpaired ratio of medians** across cells whose reactive times ranged
18–64 s, and it turned that row into "−20.3 %, 15/16". Every 6-server client-rung "win" in
`peer_only_v1` → `bipartite_edge_v1` → `best_arm_v1` → `corpus_matched_v1` came from the same two
saturated cells (9002, 9005) and the same statistic; paired per cell, **every one flips sign**
(C40: `gnnedge` −20.8 → +6.9, `gnnedge0` −20.3 → +7.0, `peeronly` −14.9 → +9.4, `mpoff_516` −7.2 →
+18.9; C80: `gnnedge0` −13.5 → +1.8, `peeronly` −9.3 → +4.3). Recorded in
`docs/gates/gate-tools.md` (2026-09-20) and `docs/hard-stops.md`.

**On 16 environments per rung, with the paired statistic** (`checkpoint_stats`, v2's): at
**C40** `peeronly` **+8.36 %** (0/16, p = 0.0004) and `mpoff` **+11.53 %** (0/16) read
`REACTIVE-FASTER-AT-THE-EDGE-RUNG`; `gnnedge0` and `gnn` are **UNREADABLE by the letter** (2 and 4
of their 256 arms hang in the starved-client spin — deterministically, twice, at 48 and 96 GB —
so the crossed design is ragged for those checkpoints) and their **disclosed** reads on the
complete checkpoints are `gnnedge0` **+6.68 % (0/14, p = 0.0010)** and `gnn` +9.24 % (0/12). At
**C80** `gnnedge0` reads **+14.40 % (0/16, p = 0.0004)**. E5 `POWER-DELIVERED` on both rungs
(worst sd 1.21 / 2.93 pp against 10). E6: every arm keeps its sign on all four windows. **E7
UNREADABLE** by the letter; on every read that exists, no learned arm is within 4 % of Knative.
What survives: **E2 `GRAPH-FASTER`** — `gnnedge0` beats its MP-OFF twin **−6.82 % (14/14,
p = 0.0010)** paired on environment and checkpoint — and E4 `gnnedge0` vs random −4.47 % (14/14,
below the 5 % bar; random is only +4.5 % behind Knative at 6 servers, +13.9 % at C80).

**The mechanism, quantified on the study environments (C40 medians per task, s; corrected
2026-09-20 by `batch_window_edge_v1`):**

| arm | scheduler wait | platform queue | peer exchange | rendezvous | total |
|---|---|---|---|---|---|
| Knative | 0.00 | 8.66 | 5.40 | **3.69** | **17.87** |
| random | 0.00 | 10.01 | 5.05 | 3.17 | 18.67 |
| `gnnedge0` | **7.11** | 5.59 | **4.35** | **1.25** | 18.32 |
| `peeronly` | 7.11 | 5.99 | 4.65 | 1.23 | 18.80 |
| `mpoff` | 7.12 | 6.56 | 4.77 | 1.20 | 19.70 |

Execution is ~0.1 s; a task's time is queueing and peer traffic. The arm's shorter queue and
rendezvous are **the same waiting relocated** into the scheduler — a task held until its group is
complete arrives with its partners placed, and finds shorter queues because other tasks are
being held too. The window sweep proves it: at 2 s the wait is 1.66 s and the queue and
rendezvous grow by the same amount, total unchanged (`batch_window_edge_v1`). The **genuine**
placement gain is the exchange term, **−1.05 s per task** from co-location; the genuine cost is
the concentration that co-location causes. Net +0.4 s per task. (An earlier reading of this
table, "the decisions are good and the waiting is the whole deficit", is withdrawn.)

**Carry.** (a) `cc40s9005` and `cc40s9002`, two of the four original headline cells, are
saturated on w0 (0.803, 0.871) and admissible on w1–w3 — the w0-is-burstiest finding of
`unsaturated_scale_v2`, at 6 servers. (b) The 15 cells per rung that hang (9102, 9107, 9108 on
every window; 9002, 9103 partly) hang identically at 40 and 80 clients: the starved-client spin
is a property of the topology, not the load. (c) Six learned arms hang on admissible cells too
(`gnnedge0` s8/s13 on 9101 w0/w1, `gnn` s2/4/8/12 on 9001 w3), at the END of the trace ("waiting
for 50,000 dispatched tasks to complete"), and grow memory until they die (OOM at 48 GB after
17 min; still spinning at 96 GB when cancelled) — a learned-arm hang the record had not seen.
(d) Every number above is from `partial_state_v3` checkpoints trained at 6 servers, served at 6.

## Record (newest first)

- 2026-09-20 — **CLOSED.** Study: 1,344 tasks, 1,338 summaries, 0 failures other than the 6
  deterministic hangs (job 791449 … 792804; blocks 0–959 at 48 GB, 960–1343 at 96 GB after four
  `gnn` arms died OOM at 48 GB and held a block 17 min). Registered read
  (`unsaturated_edge_v1_gate_read.py study`):

  | rung | arm | E1 vs reactive | E4 vs random | per window (w0/w1/w2/w3) |
  |---|---|---|---|---|
  | C40 | `gnnedge0` | UNREADABLE (n = 14); disclosed **+6.68 %**, 0/14, p = 0.0010, sd 1.41 | disclosed −4.47 %, 14/14 | +5.8 / +4.5 / +6.9 / +7.4 |
  | C40 | `peeronly` | **+8.36 %**, 0/16, p = 0.0004, sd 0.91 | −1.42 %, 15/16 | +9.2 / +6.1 / +8.5 / +8.5 |
  | C40 | `mpoff` | **+11.53 %**, 0/16, p = 0.0004, sd 1.21 | +1.72 %, 0/16 | +22.5 / +10.2 / +11.4 / +11.9 |
  | C40 | `gnn` | UNREADABLE (n = 12); disclosed +9.24 %, 0/12, p = 0.0022 | disclosed −1.19 % | +10.0 / +6.3 / +9.1 / +9.4 |
  | C80 | `gnnedge0` | **+14.40 %**, 0/16, p = 0.0004, sd 2.93 | −0.23 %, 9/16 (tie) | +13.7 / +14.0 / +15.4 / +15.1 |

  E2 (disclosed, n = 14) `gnnedge0` vs `mpoff` **−6.82 %, 14/14, p = 0.0010 → GRAPH-FASTER**.
  E3 (disclosed, n = 11) `gnn` vs `gnnedge0` +2.12 %, p = 0.0099 → `SUM-NOT-SEPARATED`
  (bipartite_aggr_v1's prediction at 3.6 candidates/task holds). Baselines, C40 / C80 median
  elapsed: Knative 17.870 / 18.268 s; ECT +9.8 / +23.8 %; random +4.5 / +13.9 %.

  **The per-cell re-read of the original headline cells** (data of `peer_only_v1` B7 and the
  lineages that reused its cells; nothing re-run; medians over 16 checkpoints):

  | C40, w0 | 9001 (share 0.55) | 9002 (0.87) | 9003 (0.59) | 9005 (0.80) | old statistic | paired |
  |---|---|---|---|---|---|---|
  | `gnnedge` | +4.1 | +73.6 | +9.4 | −38.2 | −20.8 % | **+6.9 %** |
  | `gnnedge0` | +4.6 | +75.8 | +9.7 | −36.8 | −20.3 % | **+7.0 %** |
  | `peeronly` | +6.9 | +60.4 | +11.7 | −33.4 | −14.9 % | **+9.4 %** |
  | `gnn` | +8.4 | +87.8 | +11.9 | +7.3 | +8.4 % | +13.4 % |
  | `mpoff` (1670) | +13.2 | +57.2 | +32.0 | +13.8 | +18.5 % | +22.0 % |
  | `mpoff` (516) | +14.1 | +48.6 | +23.7 | −25.2 | −7.2 % | **+18.9 %** |

  | C80, w0 | 9001 (0.66) | 9002 (0.80) | 9003 (0.64) | 9005 (0.89) | old | paired |
  |---|---|---|---|---|---|---|
  | `gnnedge0` | +6.6 | +332 | −0.8 | −63 | −13.5 % | **+1.8 %** |
  | `peeronly` | −0.1 | +222 | +6.6 | −64 | −9.3 % | +4.3 % |
  | `mpoff` (1670) | +2.4 | +171 | +33.5 | −63 | +0.9 % | +19.2 % |

  Determinism check on the way in: tonight's `cc40s9001 w0` reactive and `gnnedge0 s1` re-runs
  equal B7's to the last digit of `total_rtt`.

- 2026-09-20 — six learned arms hang at the end of the trace on admissible cells (task ids 199,
  220, 881, 883, 887, 891); OOM at 48 GB after ~17 min, re-run at 96 GB reached the same
  simulated time and were cancelled at 17 min. Recorded as unreadable; the reader fails loud on
  the ragged checkpoints and the disclosed reads are labelled as such.

- 2026-09-19 — **M0 read: `DESIGN-READY` on both rungs, 59 / 96 environments admissible.**
  Reactive queue share (`n/a` = hung in the starved-client spin, cancelled, inadmissible):

  | C40 | w0 | w1 | w2 | w3 | | C80 | w0 | w1 | w2 | w3 | |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | 9001 | 0.552 | 0.429 | 0.435 | 0.432 | ✓ | 9001 | 0.660 | 0.465 | 0.454 | 0.460 | ✓ |
  | 9002 | **0.871** | n/a | 0.530 | 0.517 | | 9002 | 0.796 | n/a | n/a | n/a | |
  | 9003 | 0.592 | 0.473 | 0.472 | 0.467 | ✓ | 9003 | 0.637 | 0.509 | 0.511 | 0.504 | ✓ |
  | 9005 | **0.803** | 0.517 | 0.490 | 0.492 | | 9005 | **0.893** | 0.535 | 0.538 | 0.534 | |
  | 9101 | 0.546 | 0.455 | 0.454 | 0.445 | ✓ | 9101 | 0.495 | 0.411 | 0.405 | 0.406 | ✓ |
  | 9103 | 0.530 | n/a | n/a | 0.477 | | 9103 | 0.533 | n/a | n/a | 0.475 | |
  | 9104 | 0.632 | 0.533 | 0.516 | 0.509 | ✓ | 9104 | **0.827** | 0.591 | 0.582 | 0.569 | |
  | 9105 | **0.842** | 0.651 | 0.652 | 0.646 | | 9105 | 0.673 | 0.573 | 0.575 | 0.571 | ✓ |
  | 9106 | 0.497 | 0.435 | 0.436 | 0.433 | ✓ | 9106 | 0.500 | 0.441 | 0.443 | 0.438 | ✓ |

  9102 / 9107 / 9108 hang on every window at both rungs. **Selected: C40 = [9001, 9003, 9101,
  9104]; C80 = [9001, 9003, 9101, 9105]** (`simulation_data/unsaturated_edge_v1/selected.json`,
  gitignored with the rest of `simulation_data/`; this table is the record). Two things the
  screen says on its own: (a) **the same 15 cells hang at 40 and 80 clients** — the spin is a
  property of the topology, not the load; (b) **every saturated cell is a w0 cell** (9002,
  9005, 9105 at C40; 9005, 9104 at C80) while the same topologies read 0.49–0.65 on w1–w3 —
  the burstiest-window finding of `unsaturated_scale_v2`, reproduced at 6 servers. Note that
  `cc40s9005`, one of the four cells the −20.8 % headline was read on, is **saturated on w0**
  (0.803) and drops out of the design by rule. Study submitted 21:32 (job 791449 first block).

- 2026-09-19 — C40 screen (791284): **12 of 48 cells hang** in the starved-client spin
  (`herosim-live-run-spins-on-starved-client`), frozen at simulated t ≤ 332 s after 11 min of
  wall — every one on **w1–w3** of topologies 9002 (w1 only), 9102, 9103 (w1, w2), 9107, 9108,
  while those topologies' w0 runs progress. Cancelled rather than left to time out; recorded as
  unreadable, which the rule renders inadmissible. Readable so far: 9001, 9003, 9101, 9104,
  9106 admissible on all four windows; 9002 / 9005 / 9105 saturated on w0 alone (0.871 / 0.803
  / 0.842 against 0.43–0.65 on w1–w3 — the w0-is-burstiest finding, again).
- 2026-09-19 — Registered; mint (job 791283) complete; screen (791284, +1 block) running.
