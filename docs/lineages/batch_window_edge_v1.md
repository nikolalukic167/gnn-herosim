# batch_window_edge_v1 — the batch window where the learned arm is competitive

**Status:** `CLOSED` (2026-09-20) — **`WINDOW-NOT-THE-LEVER` · `BATCHING-RELOCATES-WAITING`.**
Closed on the registered screen (rule 6 satisfied: 189 live arms on the screen topology; the
confirmation phase does not run because the incumbent won the screen). Registered 2026-09-19;
bars, reader, 6 tests and the expectation committed before any arm at a new window ran. **The
registered expectation (4 or 8 s chosen; W2 fires at 70 %) was wrong.**

**Outcome. Shortening the window removes the scheduler wait and puts the same time back as
platform queue and rendezvous; the total does not move.** `gnnedge0` on the screen topology
(cc40s9001 × 4 windows, 16 checkpoints; median checkpoint statistic vs Knative):

| window | batch wait / task | median vs Knative | w0 / w1 / w2 / w3 | n |
|---|---|---|---|---|
| 2 s | 1.66 s | +8.77 % | +9.3 / +8.9 / +8.6 / +8.5 | 16 |
| 4 s | 2.65 s | +7.87 % (disclosed) | +5.7 / +8.9 / +8.0 / +7.7 | 15 |
| 8 s | 4.30 s | +8.04 % (disclosed) | +4.0 / +8.7 / +7.9 / +8.1 | 14 |
| **16 s** (incumbent) | 7.15 s | **+7.70 %** | +4.6 / +8.5 / +7.4 / +8.2 | 16 |

The registered rule (`select_window`: lowest screen median among windows carrying all 16
checkpoints) picks **16 s** — 4 s and 8 s are ineligible because one and two of their arms hang
in the starved-client spin (the same checkpoints that hang at 16 s on another topology) — and
the disclosed reads on the complete checkpoints agree: no window is better than the incumbent by
more than 0.3 pp, and the span across all four windows is 1.1 pp. **W6's premise held** (the wait
fell 7.15 → 1.66 s) **and it bought nothing**: at 2 s the platform queue rose 4.8 → 8.2 s and
rendezvous 1.25 → 3.3 s per task. Nothing to confirm; the held-out phase was not run.

**What this settles, with `unsaturated_edge_v1`'s decomposition** (C40, 16 environments,
medians per task): Knative pays 8.66 s of platform queue, 5.40 s of peer exchange and **3.69 s of
rendezvous** (waiting at the platform for a partner not yet placed); `gnnedge0` pays 7.11 s of
scheduler wait, 5.59 s queue, 4.35 s exchange and **1.25 s rendezvous**. Execution is ~0.1 s.
The arm's smaller queue and rendezvous are the same waiting **relocated** into the scheduler — a
task that has waited for its whole group arrives with its partners placed and finds shorter
queues because the other tasks are still being held. The **genuine** placement gain is the
exchange term, **−1.05 s per task from co-location**; the genuine cost is that co-location
concentrates load. The net is +0.4 s (+2.5 % raw, +6.7 % paired), and the window cannot move it
because it only chooses where the waiting is booked. `unsaturated_edge_v1`'s "the decisions are
good; the waiting is the whole deficit" is **corrected** to this reading in its head.

**Carry.** (a) The hang follows the checkpoint: `gnnedge0` s13 hangs at 16 s on 9101 w1 and at
4 s on 9001 w1; s4 and s8 at 8 s on 9001 w0. A learned-arm hang at the end of the trace is a
property of (checkpoint, cell), reproducible, and grows memory until killed. (b) Removing
batching outright is closed elsewhere (`drainable_serving_config_v1`: −1731 %); this lineage
closes the interior. (c) The one lever left that this record has not measured is decoding each
arrival **immediately, conditioned on the partners already placed** (no group wait, no
rendezvous relocation) — the prefix decoder supports it in form; it has never been served.

**Question (as registered).** Every learned arm collects a task's whole **peer group (10 tasks)** before it
decodes, waiting up to `scheduler.batch_timeout` = **16 s** for members not yet arrived. That
wait is **6.8 s per task** at every operating point ever measured, and reactive Knative pays
**0.000 s** of it: at 80 servers it is 37 % of reactive's elapsed and the arms end ±2 % of
reactive (`unsaturated_scale_v2`); at 6 servers / 40 clients on w0 the arm halves reactive's
queue and keeps −20 % after the tax. Group arrival spans are **median 13.8 s, p90 32.3 s**, so
16 s closes about half the groups complete and times out on the rest (2,911 incomplete of
8,327 batches on `cc40s9001`). The window is a **policy constant** set once
(`drainable_regime_v1`) and swept once, at the 20-client rung where every learned arm loses by
≥ 70 % at every window (`drainable_serving_config_v1`: 8 s VOID on C4, 16/24 s TIE vs the twin,
no window within 2× of reactive). **Nobody has moved it where the arm is competitive.** A
shorter window closes groups earlier and decodes late members in a later batch conditioned on
the prefix already placed (`partial_state_v3`): less wait, more peers placed apart. Is the trade
a net win against reactive?

**Design — screen / confirm, so the window is chosen on environments the verdict never sees.**
- **Screen:** `gnnedge0` (be1670) at **2 / 4 / 8 s** on the **lowest-seed** study topology × 4
  windows × 16 checkpoints (192 arms); the 16 s arm is the main study's. The window with the
  lowest median checkpoint statistic vs reactive wins (`select_window`); a window that cannot
  deliver all 16 checkpoints is ineligible. **16 s winning closes the lineage
  `WINDOW-NOT-THE-LEVER`.**
- **Confirm:** the chosen window on the **12 held-out** environments (the other 3 topologies × 4
  windows): `gnnedge0` and its MP-OFF twin `mpoff` (1670), 16 checkpoints each (384 arms).
  Reactive and random do not batch; the study's runs are the reference unchanged.
- Cells: the study's `cc40s{seed}.json` with **only** `scheduler.batch_timeout` moved
  (`cc40s{seed}_bt{2,4,8}.json`, minted by the sbatch, refused if any other field differs).
  Poll interval unchanged at 1 ms. Nothing retrained.

**Bars — the chain's** (`scripts_cosim/batch_window_edge_v1_read.py`): unit = checkpoint,
median over the 12 held-out environments of the checkpoint's relative % on the same
environment, signed-rank against zero, |median| ≥ 5 %, p < 0.05, n ≥ 16.

| read | what (held-out unless stated) | fires as |
|---|---|---|
| W1 | `gnnedge0`@chosen vs reactive | `SHORTER-WINDOW-BEATS-REACTIVE-HELD-OUT` / `REACTIVE-FASTER-HELD-OUT` / `NOT-SEPARATED` |
| W2 | `gnnedge0`@chosen vs `gnnedge0`@16 s, paired | `SHORTER-WINDOW-FASTER-THAN-16S-HELD-OUT` / `16S-FASTER…` / `NOT-SEPARATED` |
| W3 | `gnnedge0`@chosen vs `mpoff`@chosen, paired | `GRAPH-FASTER-AT-MATCHED-WINDOW` / `POINTWISE-FASTER…` / `NOT-SEPARATED` |
| W4 | `gnnedge0`@chosen vs `random_network` | `SHORTER-WINDOW-BEATS-RANDOM-HELD-OUT` / … |
| W5 | sign per arrival window (descriptive) | `SIGN-CONSISTENT…` / `SIGN-FLIPS…` |
| W6 | mechanism: median batch wait fell vs 16 s on the same keys | `BATCH-WAIT-FELL` / `BATCH-WAIT-DID-NOT-FALL` — if it did not fall, W1 is not a batching result |

**Registered expectation (signed 2026-09-19).** The screen picks **4 s or 8 s**, not 2 s:
at 2 s most groups split into singletons and the graph arm decodes singletons badly
(`drainable_serving_config_v1` A: no batching = −1731 % vs reactive). W6 fires. W2 fires
(the shorter window is faster than 16 s on the held-out set — this is the mechanism claim, and
I put it at 70 %). **W1 is the real question and I put it at 50 %**: the wait falls from 6.8 s
to ~2–4 s, but part of that is given back as peers placed apart (rendezvous + exchange) and as
queue the decoder no longer sees. W3 fires GRAPH-FASTER (the twin has never beaten the graph
arm at matched corpus at this rung). W4 fires.

**Consequences, signed in advance.**
- W1 fires and W6 fires: the standing answer gains a **held-out** learned-arm win over reactive
  at a healthy 6-server rung with the window named; the paper quotes W1 with W2/W5.
- W1 does not fire, W2 fires: the window buys speed but not enough; the tax is only part of the
  margin, and the remaining lever is the decode itself (out of scope for these checkpoints).
- `WINDOW-NOT-THE-LEVER` or W2 fails: the 16 s window is retained and the batching tax is
  recorded as intrinsic to peer-group placement, not a knob.
- W6 does not fire: the read is VOID as a batching result whatever W1 says.

**Cost.** 576 arms × ~165 s ≈ 45 min at the 48-task ceiling.
Script: `scripts_cosim/datalab/batch_window_edge_v1.sbatch`; read:
`scripts_cosim/batch_window_edge_v1_gate_read.py {screen,confirm}`. Results under
`simulation_data/peer_affinity_live_gate/results/bw_v1/`.

**Datasets.** None new; the 1,670-dataset checkpoints of `bipartite_edge_v1` / `peer_only_v1`
served as-is.

## Record (newest first)

- 2026-09-20 — **CLOSED on the screen.** Jobs 792810 / 792858 / 792907 / 792xxx: 192 tasks,
  189 summaries, 3 hangs cancelled by a 20-minute watchdog (tasks 92, 131, 135). Registered
  read (`batch_window_edge_v1_gate_read.py screen`): 2 s +8.77 % (n = 16, wait 1.663 s); 4 s
  UNREADABLE (seed 13 hangs on w1); 8 s UNREADABLE (seed 4 on w0); 16 s +7.70 % (n = 16, wait
  7.147 s) → `WINDOW-NOT-THE-LEVER`. Disclosed on complete checkpoints: 4 s +7.87 % (n = 15),
  8 s +8.04 % (n = 14). Per-task decomposition on the screen topology at 2 / 4 / 8 / 16 s:
  wait 1.66 / 2.65 / 4.30 / 7.15; queue (w1) 8.19 / 7.34 / (8 s: n/a) / 4.78; rendezvous (w1)
  3.34 / 3.20 / — / 1.2; exchange ≈ 5.0 at every window on w1–w3, 4.1–4.2 on w0.

- 2026-09-19 — Registered.
