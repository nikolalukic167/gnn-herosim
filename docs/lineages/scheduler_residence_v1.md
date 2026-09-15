# scheduler_residence_v1 — where the learned arms' scheduler-side time actually goes

**Status:** `REGISTERED` (2026-09-15). Every bar below is a module constant in
`scripts_cosim/scheduler_residence_v1_read.py`, committed **before** the arms it reads exist.
Amend by dated amendment only.

**Parents:** `queue_range_v1` (CLOSED today — its per-term decomposition is what raised this),
`drainable_serving_config_v1` (CLOSED — the window sweep this must not repeat),
`serving_stability_v1` (the early advantage).

---

## The claim under test, and the mis-attribution that motivates it

`queue_range_v1` measured, on `cell_s9001` — the one cell where placement works — that the
learned arm **beats reactive Knative on every term it was designed to win and loses the cell
anyway**: queue −2.67 s, peer exchange −0.85 s, peer rendezvous −1.85 s, `averageWaitTime`
**+6.87 s**, net **+1.43 s**. That +6.87 s is flat at 6.7–6.9 s on all three cells and both
policies.

**It was written up as "batch wait". That is wrong and this lineage exists to correct it.**
`averageWaitTime` is `scheduled_time − dispatched_time` (`src/placement/infrastructure.py:389`)
— the *whole* scheduler-side residence. It contains at least four things:

1. **Head-of-line blocking.** The scheduler is one serial process
   (`src/policy/gnn/scheduler.py:381-388`). A ready task that is not a peer of the batch
   currently being collected waits out that entire 16 s window, and a run has ~8,300 batches.
2. The collection window itself (`scheduler.py:443-454`).
3. Re-queue after deferral (`scheduler.py:811-812`) plus `create_first_replica`.
4. Store / mutex / platform serialization.

**The split has never been measured**, because the collector's own `actual_wait_time` is
computed, printed and thrown away (`scheduler.py:474`). (1) and (2) are different problems with
different fixes, and only (2) is covered by the window sweep.

> **The claim:** most of the 6.87 s is (1), not (2) — the arms are not paying for their own
> batching, they are paying for the batching of tasks they have nothing to do with.

## Why no existing stop covers this

* **It is not the window.** `drainable_serving_config_v1` swept 0 / 8 / 16 / 24 / 80 s and
  closed `BATCHING-DOES-NOT-EXPLAIN`; zero window is **−1731 %** because the graph arm cannot
  decode singletons, and 16 s is the interior optimum. **No bar here varies the window**, and
  R2's intervention leaves group composition byte-identical.
* **It is not "dispatch when the group is complete".** That is already implemented —
  `scheduler.py:443` exits the loop the instant `remaining` is empty.
* **Nothing is retrained**, so the count theorem, the horizon-return stop, the closed-loop PG
  stop and the warm-corpus stop do not reach it.
* `docs/hard-stops.md` stops the K = 3 queue guardrail and the platform cap; neither touches
  when a task leaves the scheduler's store.

## Bars

**R0 — the decomposition (blocking, and it can close the lineage).**
Per placed task the scheduler now records `[dispatched, head_of_line, collection, placement]`,
which reconstructs that task's `waitTime` exactly by construction.

* `R0_RECONSTRUCTION_TOL = 0.01` — mean(head_of_line + collection + placement) must equal
  `averageWaitTime` to within 1 %, and `residence_unstamped` must be **0**. A decomposition
  that cannot be reconciled against the number it decomposes is a story, not a measurement.
  **Blocking: failing this makes the cell VOID, not negative.**
* `R0_HOL_MIN_S = 2.0`, `R0_MIN_CELLS = 2`, `R0_MIN_SEEDS = 12`.
  **`HEAD-OF-LINE-DOMINATES`** if median head-of-line time ≥ `R0_HOL_MIN_S` on ≥ 2 of 3 cells
  **and** head-of-line exceeds collection on those cells. Otherwise **`COLLECTION-DOMINATES`**,
  which the window sweep has already closed, and the lineage ends there and says so.
* **Registered expectation: UNCERTAIN.** The argument for head-of-line is arithmetic (a serial
  scheduler, a 16 s window, ~8,300 batches); the argument against is that peer-group collection
  at 0.46 arrivals/s is genuinely slow and may account for all of it. This bar is placed first
  because it is the one that decides whether anything else here is worth running.

**R1 — what separates the good cell (no simulation; reads the three configs).**
`cell_s9001` has **no** queue problem under identical policy, trace, load and checkpoints,
while s7901 and s9002 carry +23.3 s and +37.2 s. The difference is therefore in the cell.
Statistics compared: server (replica-hosting) node count, client count, mean and minimum
reachable-server fan-out per client, replicas per task type, hosting-node spread.

* `R1_MIN_SEPARATING = 1` — **`STRUCTURE-SEPARATES`** if ≥ 1 statistic puts s9001 strictly
  outside the range spanned by s7901 and s9002. Otherwise **`STRUCTURE-DOES-NOT-SEPARATE`**
  and the cause is dynamic, which R1 reports rather than guesses at.
* **Registered expectation: NEGATIVE.** Stated in advance, and it is also the user's. R1 is run
  because it costs minutes, not because it is likely.

**R2 — the live gate (rule 6), conditional on R0.**
Runs **only** if R0 reads `HEAD-OF-LINE-DOMINATES`. The intervention: a ready task that no
batch currently being collected is waiting for must not be held by that batch's window.
Group composition is unchanged — the knob may not alter which tasks end up in a batch, only
when a batch that is not waiting for them releases the store.

* `R2_CONTROL_MAX_BATCH_DELTA = 0.05` — **blocking**: mean batch size and peer-pair retention
  within 5 % of control. A knob that shrank the groups is `nobatch` wearing a new name and must
  read **VOID**, not negative.
* `R2_ALPHA = 0.05`, `R2_HOLM_N = 3` (cells), `R2_MIN_SEEDS = 12`, `R2_MIN_CELLS = 2`.
  Paired **Wilcoxon signed-rank** — the arms share seeds (`docs/gates/gate-tools.md`,
  2026-09-15). **`HEAD-OF-LINE-IS-A-LEVER`** on ≥ 2 of 3 cells for `gnn`.
* `R2_BEATS_REACTIVE_MIN_CELLS = 2`. **Registered expectation: fires on `cell_s9001` and
  nowhere else.** s9001 is +1.43 s behind against a recoverable 6.87 s; s7901 and s9002 are
  +28.9 s and +42.0 s behind, which no scheduler-residence change reaches.

`mpoff` is carried as the pointwise control throughout and **no GNN-vs-pointwise claim may be
founded here** — a scheduling-concurrency change is policy-agnostic and must help both.

## Declared in advance

* **`gnn` seed 3's checkpoint deterministically livelocks the simulator** on `cell_s7901`
  (`drainable_objective_v1` Phase C, reproduced byte-for-byte). Excluded by name.
* **BURNED and excluded:** the `jam_probe` scoping read's `gnn` seeds **8** and **14** on
  `cell_s7901`.
* **The instrument must be inert.** One control arm must reproduce `queue_range_v1`'s `plain`
  medians — **54.819 / 22.350 / 65.401** — to three decimals, the same parity check that
  lineage passed across a builder/scheduler/orchestrator change.
* **`GNN_BATCH_POLL_INTERVAL` is not exported by the parent gate**, so its arms polled at 1 ms
  against a 16 s window. R0 and R2 keep that exactly as it was; changing it would change the
  control.

## Not in scope

Re-sweeping the batch window; removing batching; dispatch-on-group-complete (implemented);
the queue blow-up on s7901/s9002 (real, unexplained, separate lineage); retraining anything;
the cluster-size axis (`cluster_scale_v1`, which follows this).

## Record

*(dated entries appended below as the lineage runs)*
