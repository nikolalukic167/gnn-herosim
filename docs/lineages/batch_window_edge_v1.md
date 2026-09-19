# batch_window_edge_v1 — the batch window where the learned arm is competitive

**Status:** `REGISTERED` (2026-09-19) — bars, reader, 6 tests and this expectation committed
before any arm at a new window is served. Runs after `unsaturated_edge_v1`'s study, on its
16 study environments at 6 servers / 40 clients, whatever that lineage's verdict; its
BELOW-BAR consequence makes this lineage mandatory, its HOLDS consequence makes it the next
margin.

**Question.** Every learned arm collects a task's whole **peer group (10 tasks)** before it
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

- 2026-09-19 — Registered.
