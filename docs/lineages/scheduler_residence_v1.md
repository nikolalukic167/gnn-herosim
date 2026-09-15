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

### 2026-09-15 — R0 read (smoke, n = 1/cell): **COLLECTION-DOMINATES**. The claim is refuted.

Job 768927, three arms, one per cell, `gnn` seed 1, same cells/window/checkpoints/knobs as
`queue_range_v1`'s `plain` arms.

| cell | `waitTime` | head-of-line | collection | placement | reconstruction error |
|---|---|---|---|---|---|
| s7901 | 6.890 | **0.766** | **6.124** | 0.000 | 0.000000 |
| s9001 | 6.832 | **0.733** | **6.099** | 0.000 | 0.000000 |
| s9002 | 6.721 | **0.607** | **6.115** | 0.000 | 0.000000 |

**Head-of-line blocking is 9–11 % of the scheduler-side wait, against a 2.0 s bar and against
a collection term 8–10× larger.** The registered claim — "the arms are paying for the batching
of tasks they have nothing to do with" — is **wrong**. The verdict is `COLLECTION-DOMINATES`,
which `drainable_serving_config_v1` has already closed (0 s window is −1731 %; 16 s is the
interior optimum), so **R2 does not run**, exactly as registered.

The decomposition reconciles to **six decimal places** on every cell and `residence_unstamped`
is 0, so this is not an artefact of how the split was defined. Two consequences worth keeping:

* **`queue_range_v1`'s "batch wait" label was right after all.** The correction written into
  that node on the same day — that `averageWaitTime` is whole-scheduler residence and might be
  mostly head-of-line — was over-cautious. It is 89 % peer-group collection, now measured.
  Placement (decode, mutex, platform enqueue) is **0.000 s**: the decoder costs no sim time.
* **The 6.87 s is therefore the price of assembling a peer group at 0.46 arrivals/s**, where a
  10-task group takes ~21.7 s to co-arrive. The only lever on it that is not the window is the
  arrival rate — which is `cluster_scale_v1`.

Full 96-arm confirmation at the registered n running as job 768931.

### 2026-09-15 — R1 read: **STRUCTURE-SEPARATES**, against a registered NEGATIVE expectation

`simulation_data/scheduler_residence_v1/r1.json`. No simulation; the topologies are built
through the simulator's own `prepare_infrastructure_for_real_simulation`, so the measurement
and the run share one code path.

**First, the fact that made R1 worth doing at all:** the three cells' configs are identical in
**149 of 150 flattened fields**. The only difference is `network.topology.seed`
(7901 / 9001 / 9002). Everything downstream is what that seed drew.

| statistic | **s9001 (no queue blow-up)** | s7901 (+23.3 s) | s9002 (+37.2 s) | separates |
|---|---|---|---|---|
| server nodes | 6 | 6 | 6 | no |
| client nodes | 20 | 20 | 20 | no |
| mean reachable servers per client | 3.450 | 3.050 | 3.500 | no |
| **min reachable servers per client** | **2** | **1** | **1** | **yes** |
| mean clients per server | 11.500 | 10.167 | 11.667 | no |
| **clients-per-server imbalance (max − min)** | **3** | **8** | **7** | **yes** |

Two statistics put s9001 strictly outside the range the other two span, against a bar of
`R1_MIN_SEPARATING = 1`. **On both cells with a queue blow-up there is a client that can reach
exactly one server** — it has no placement choice at all — **and the reachability load is 2.3–
2.7× more lopsided.** The mean fan-out separates nothing; it is the worst case and the spread
that do, which is why an average over a topology is the wrong summary.

**This is an ordering on three cells.** That is the exact shape that killed the stability
mechanism in `serving_stability_v1` (three aggregates, ρ = +0.088 within cells). R1 fires the
bar as signed — strictly outside, committed before the measurement — and it is **not a claim
until it survives a sample size.** R3 below.

---

## Amendment 1 (2026-09-15) — R2 VOID, R1 gets a powered replication and the live gate

**Signed before any R3 cell has been served a task.** R0 and R1's bars are untouched and their
verdicts stand as read.

* **R2 is VOID.** It was registered as conditional on R0 firing. R0 read `COLLECTION-DOMINATES`,
  so the head-of-line intervention is not built and not gated.
* **R3 replaces it as this lineage's live gate** (rule 6). It tests R1's finding at a sample
  size, on the same physics, and it ends live.

### R3 — does topology lopsidedness predict the queue blow-up?

**R3-a — mint and screen (no simulation, free).**
`scripts_cosim/scheduler_residence_v1_r3_cells.py` writes cells that differ from
`cell_s9001_f4000_pg16` in **exactly one flattened field**, `network.topology.seed` — asserted
field-by-field per cell, refusing to write one that differs in anything else. 40 draws are
measured; `R3_N_CELLS = 12` are selected **spanning `min_reachable_servers`** (tiebreak: the
clients-per-server imbalance).

**The selection is on the independent variable only, and the ordering is the control:** cells
are chosen before any of them has been served a single task, so no latency can influence which
cells are read. Recorded here because a selection made after seeing the outcome is choosing
the answer, and the manifest is committed with the seeds in it.

**R3-b — the live gate.** Per cell: 1 reactive `knative_network`, `R3_SEEDS_PER_ARM = 4`
`gnn` and 4 `mpoff`. 12 cells × 9 = **108 arms**, ~5 min each. Same trace, window, checkpoints
and physics as every other reading in this family.

The unit of replication is the **cell**, not the seed: the dependent variable is the cell's
median **excess queue over its own reactive arm**, which removes whatever the topology costs
everybody.

* `R3_PRIMARY = spearman(min_reachable_servers, excess_queue_s)` across the 12 cells.
  `R3_MIN_ABS_RHO = 0.60`, `R3_ALPHA = 0.05`, `R3_MIN_CELLS_READ = 10` (a cell whose arms fail
  is dropped and the read says so). **`LOPSIDEDNESS-PREDICTS`** if ρ is negative — fewer
  reachable servers, more excess — and clears both. Otherwise **`LOPSIDEDNESS-DOES-NOT-PREDICT`**.
* **`R3_CONTROL`, blocking in interpretation, not in verdict.** The same correlation against
  **reactive's own** queue. If lopsidedness predicts reactive's queue just as strongly, it is a
  fact about the environment and not about the learned arm, and the primary is reported
  `CONFOUNDED-ENVIRONMENT` whatever its ρ. `drainable_objective_v1`'s C0 and
  `peer_affinity_v1`'s H5 both died of a control that moved with the treatment; this one is
  registered in advance.
* **`R3_SECOND = spearman(hosting_node_spread, excess_queue_s)`**, R1's other separator,
  reported beside the primary. Holm over `R3_HOLM_N = 2`.
* `mpoff` is carried throughout and **no GNN-vs-pointwise claim may be founded here** — a
  topology effect is policy-agnostic and must appear in both.
* **Registered expectation: UNCERTAIN, leaning positive.** The mechanism is concrete (a client
  with one reachable server cannot be spread, and a server every client can reach is a queue
  sink) but it rests on three points, and three-point orderings have failed twice in this
  program.

**What R3 cannot do:** it cannot say the lopsidedness *causes* the blow-up, only that it
predicts it across independent topology draws with the control attached. A causal read would
hold the draw fixed and edit the reachability graph, which is a separate registration.

### 2026-09-15 — R3-a: the 40 draws, and a power limitation disclosed before the gate runs

`simulation_data/scheduler_residence_v1/r3_cells.json`. 40 topology draws from seeds
9100–9139, each verified field-by-field to differ from `cell_s9001_f4000_pg16` in exactly one
flattened field. `min_reachable_servers` is **coarse**: across 40 draws it takes only three
values — **1 (17 draws), 2 (21), 3 (2)**. The 12 selected span 1 / 2 / 3 with 5 / 6 / 1 cells.

**Disclosed, and the bar is not moved:** a three-level independent variable caps the Spearman
a perfect monotone relationship can reach, so `R3_MIN_ABS_RHO = 0.60` is demanding on the
primary. It was committed before the draws were measured and it stays. `hosting_node_spread`,
the registered second statistic, is much richer across the same draws (range 1–10) and is read
beside the primary under Holm over the registered family of 2. If the primary fails on
granularity while the second fires, the read says exactly that rather than promoting the
second.

Selected cells (structure measured before any was served a task):

| cell | min reachable servers | clients-per-server imbalance |
|---|---|---|
| `cell_r3s9135` | 1 | 2 |
| `cell_r3s9120` | 1 | 4 |
| `cell_r3s9107` | 1 | 6 |
| `cell_r3s9103` | 1 | 7 |
| `cell_r3s9132` | 1 | 7 |
| `cell_r3s9125` | 2 | 1 |
| `cell_r3s9124` | 2 | 3 |
| `cell_r3s9114` | 2 | 4 |
| `cell_r3s9108` | 2 | 5 |
| `cell_r3s9112` | 2 | 6 |
| `cell_r3s9104` | 2 | 9 |
| `cell_r3s9127` | 3 | 6 |

**Also declared:** R3 runs `gnn`/`mpoff` seeds **1, 2, 4, 5**. Seed 3's checkpoint
deterministically livelocks the simulator on `cell_s7901`, and whether it does so on a fresh
topology draw is unknown — it is excluded by name rather than risked, so a livelock cannot
silently drop a cell below `R3_MIN_SEEDS_PER_CELL`.
