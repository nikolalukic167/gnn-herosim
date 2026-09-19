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

### 2026-09-15 — R0 CONFIRMED at the registered n: **COLLECTION-DOMINATES, 0/3 cells**

Job 768931, 95 of 96 arms (the one missing is `gnn` seed 3 on `cell_s7901`, the declared
livelock). `simulation_data/scheduler_residence_v1/r0.json`.

| cell | n | `waitTime` | head-of-line | collection | placement | reconstruction error |
|---|---|---|---|---|---|---|
| s7901 | 13 | 6.887 | **0.762** (11.1 %) | **6.128** (89.0 %) | 0.000 | 0.00000 |
| s9001 | 15 | 6.872 | **0.762** (11.1 %) | **6.096** (88.7 %) | 0.000 | 0.00000 |
| s9002 | 15 | 6.718 | **0.591** (8.8 %) | **6.125** (91.2 %) | 0.000 | 0.00000 |

The smoke reproduces at n = 13–15 to three decimals. **The registered claim is refuted on every
cell, not narrowly**: head-of-line blocking would have to be 2.6–3.4× larger to clear the bar
and 8–10× larger to exceed collection.

**Instrument inertness — the registered check, exceeded.** The bar asked for the control's
medians to three decimals. Measured seed-by-seed against `queue_range_v1`'s `plain` arms over
all 95 shared arms, the **worst per-seed difference is 0.000000000 s**: 54.819 / 51.403 /
22.380 / 22.523 / 65.508 / 73.998 on both sides, bit-identical. A wrapped collector, two new
call sites in the scheduling paths and two orchestrator whitelist entries changed nothing about
what the simulator does.

**What R0 settles.** `placement` — decode, mutex, node and platform acquisition, queue enqueue —
is **0.000 s**, so the GNN's inference costs no simulated time and none of the loss is decode
cost. The 6.87 s is **89 % peer-group collection**: the price of assembling a 10-task group at
0.46 arrivals/s, where the group takes ~21.7 s to co-arrive. `drainable_serving_config_v1`
swept the only knob on that (the window: 0 / 8 / 16 / 24 / 80 s) and found 16 s the interior
optimum, so **within this environment the term is already minimised**. The remaining lever is
the arrival rate, which is `cluster_scale_v1`.

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

### 2026-09-15/16 — R3 batch 1 read: **VOID-TOO-FEW-CELLS**, and the bar is what caught it

Job 769048, 72 of 108 arms. `simulation_data/scheduler_residence_v1/r3.json`.
**8 of 12 cells readable against `R3_MIN_CELLS_READ = 10` ⇒ VOID.** The registered floor was
committed before the cells existed and it fired; the eight cells' numbers are recorded below
and are **not** a result. "Could not measure" is not "nothing there".

**Why four cells are missing: they hang, on all NINE arms including `knative_network`.**
Tasks 18–26, 45–53, 54–62 and 72–80 — three complete cells each — TIMEOUT at 30 min with the
documented signature: 100 % CPU, no log growth, frozen on `Gateway: Processing event 20001
(30000 remaining)`. That is the **starved-client spin** already in the record
(`docs/gates/gate-tools.md`, 2026-09-14, 5 of 28 W1 capture cells): a postponed task retried
every batch when its client's reachable servers are memory-full. **The reactive baseline hangs
too, so this is a property of the topology draw, not of any policy** — and at 4 of 12 fresh
draws it is more common here (33 %) than in the W1 capture (18 %).

**Does the hang bias the sample? Measured, not assumed.**

| statistic | hung cells | readable cells | hung range inside readable range? |
|---|---|---|---|
| `min_reachable_servers` (**primary**) | 1, 2, 2, 2 (mean 1.75) | 1,1,1,1,2,2,2,3 (mean 1.62) | **yes** |
| `mean_reachable_servers` | 3.75–4.10 (mean 3.97) | 3.45–4.10 (mean 3.91) | **yes** |
| `hosting_node_spread` (**second**) | 1, 3, 5, 6 (mean 3.75) | 2,4,4,6,6,7,7,9 (mean 5.62) | **no — the low end is truncated** |

So the loss is **not** selecting on the primary, and **is** thinning the low-imbalance end of
the second. That is disclosed here and carried into the read; it is not corrected for.

| cell | min reach | imbalance | reactive queue | `gnn` queue | `gnn` excess | `mpoff` excess |
|---|---|---|---|---|---|---|
| `r3s9135` | 1 | 2 | 14.097 | 16.329 | +2.232 | +6.252 |
| `r3s9120` | 1 | 4 | 10.544 | 7.863 | **−2.681** | −2.798 |
| `r3s9103` | 1 | 7 | 15.172 | 22.968 | +7.796 | +8.210 |
| `r3s9132` | 1 | 7 | 11.690 | 26.939 | +15.248 | +10.763 |
| `r3s9114` | 2 | 4 | 13.696 | 65.270 | +51.574 | +27.744 |
| `r3s9112` | 2 | 6 | 38.590 | 13.900 | **−24.690** | −24.369 |
| `r3s9104` | 2 | 9 | 29.142 | 49.815 | +20.673 | +33.704 |
| `r3s9127` | 3 | 6 | 11.815 | 46.065 | +34.249 | +22.682 |

One thing worth noting even from a VOID read, because it is a fact about the environment and
not about the bar: **the learned arms beat reactive outright on 2 of these 8 fresh topology
draws** (`r3s9120` −2.68 s, `r3s9112` −24.69 s of queue), which the three original cells never
showed. Whatever governs that is not `min_reachable_servers` in any obvious way.

---

## Amendment 2 (2026-09-16) — batch 2, to reach the registered cell count

**Signed before batch 2 is served.** No bar moves. R3's verdict on batch 1 stands as VOID.

Eight more cells, selected from the **same 40-draw manifest** by the **same deterministic
spanning rule** (`--exclude-seeds` over the 12 already spent), so the selection remains on the
independent variable and no cell is hand-picked:

| cell | min reach | imbalance |
|---|---|---|
| `cell_r3s9101` | 1 | 3 |
| `cell_r3s9123` | 1 | 5 |
| `cell_r3s9118` | 1 | 7 |
| `cell_r3s9126` | 1 | 9 |
| `cell_r3s9131` | 2 | 3 |
| `cell_r3s9119` | 2 | 4 |
| `cell_r3s9129` | 2 | 7 |
| `cell_r3s9139` | 3 | 5 |

The read pools both batches and reports the total readable count against the unchanged
`R3_MIN_CELLS_READ = 10`. **If batch 2 hangs at the same 33 % rate the pooled total is ~13**,
which clears it; if it does not, R3 stays VOID and the lineage records that this environment
cannot supply 10 servable topology draws rather than inventing a result from 8.

**Declared:** `--time` drops from 30 min to 15 min. A servable arm here takes ~5 min, so a
hung one is identifiable in a third of the wall clock; batch 1 spent ~18 CPU-hours waiting for
spins that were never going to finish (`gate-tools.md`, 2026-09-14: cancel and re-chain rather
than wait for TIMEOUT). Nothing else about the arms changes.

**Also declared:** whichever way R3 lands, the hang rate itself is now a recorded property of
this environment — **4 of 12 independent topology draws are unservable by every policy** — and
any future work that mints cells from seeds must budget for it.

### 2026-09-16 — R3 READ, pooled: **LOPSIDEDNESS-DOES-NOT-PREDICT**. R1 was a three-point coincidence.

Jobs 769048 + 769227, 135 arms, **15 of 20 cells readable** against the registered floor of 10.
`simulation_data/scheduler_residence_v1/r3.json`. Batch 2 hung on 1 of 8 cells (`r3s9123`)
against batch 1's 4 of 12, so the pooled hang rate is **5 of 20 (25 %)**.

| statistic | ρ vs `gnn` excess queue | p | bar | reactive control | `mpoff` control |
|---|---|---|---|---|---|
| `min_reachable_servers` (**primary**) | **+0.041** | 0.885 | \|ρ\| ≥ 0.60, sign − | +0.316 (p 0.25) | +0.020 (p 0.94) |
| `hosting_node_spread` (second) | **+0.029** | 0.918 | same | +0.120 (p 0.67) | +0.183 (p 0.51) |

**Both are nulls, and both have the wrong sign.** The registered claim was that fewer reachable
servers means more excess queue; the measured ρ is +0.041 against a bar of 0.60 — not a weak
effect, no effect. The pointwise `mpoff` control agrees (+0.020), which is what a
policy-agnostic topology effect should do and is the one prediction that held.

**R1's separation on three cells does not replicate on fifteen.** Its two statistics put
`cell_s9001` strictly outside the other two with no overlap, and both were committed bars. On
15 independent topology draws from the same generator they carry no information about the
excess at all. This is the third time in this program that a clean ordering on three aggregates
has evaporated at a sample size (`serving_stability_v1`'s stability mechanism, this session's
own 3-aggregate over-read, and now R1).

**Disclosed: the printed verdict is `CONFOUNDED-ENVIRONMENT`, and that label is degenerate
here.** The registered control rule fires when |control ρ| ≥ 0.75 × |primary ρ| with matching
sign. At a primary ρ of **+0.041** that threshold is 0.031, so *any* non-trivial control ρ trips
it — the rule fired by arithmetic, not because a confound obscured a real effect. **The bar is
not moved**; the substantive reading is `LOPSIDEDNESS-DOES-NOT-PREDICT`, both are recorded, and
the rule defect is filed in `docs/gates/gate-tools.md`. A ratio control needs a floor on the
primary's own magnitude below which it reports NOT-APPLICABLE instead of CONFOUNDED.

#### Descriptive, unregistered, and the most interesting thing in the read

The gate incidentally produced the widest topology sample this program has ever run at a
drainable load — 15 independent draws, each with its own reactive arm — and on it:

| statistic | `gnn` | `mpoff` |
|---|---|---|
| beats reactive on **queue** | **7 / 15** | 7 / 15 |
| beats reactive on **elapsed** | **3 / 15** | **4 / 15** |

Against **0 / 3** on the three cells this whole family of lineages has been run on. The margins
are not small where they land: −20.3 s, −24.0 s and −14.0 s of mean elapsed on `r3s9112`,
`r3s9129` and `r3s9139`.

And they are not randomly placed. **The learned arms win on the cells where reactive itself is
slow**: those three carry reactive elapsed of 46.6 / 69.8 / 163.8 s against 15.5–37.2 s
everywhere else. Post hoc, ρ(reactive elapsed, `gnn` excess) = **−0.446, p = 0.095, n = 15** —
directionally consistent, **not significant, and not a bar this lineage registered**. It clears
nothing. It is recorded because it names a testable claim that no existing stop covers: *the
learned arms may be relatively better precisely where the reactive rule does badly*, which is a
different question from every "does the arm beat reactive on this cell" gate in the record, and
it needs its own registration with a pre-declared difficulty measure.

**What this does NOT license.** These are queue and elapsed medians over 4 seeds per cell with
no paired test, on cells selected to span a statistic that turned out to be irrelevant, and the
3/15 is not corrected for anything. Quoting it as "the GNN beats Knative on 20 % of topologies"
would be exactly the error `drainable_regime_v1` was created to stop.

---

## Outcome: **RESIDENCE-IS-COLLECTION · LOPSIDEDNESS-DOES-NOT-REPLICATE**

**CLOSED 2026-09-16.**

* **R0** (n = 13–15, 0/3 cells): the scheduler-side wait is **89 % peer-group collection**,
  9–11 % head-of-line, and **0.000 s placement**. The registered claim is refuted. The
  instrument is bit-identical to the control on all 95 shared arms.
* **R1** (3 cells): fired against a NEGATIVE expectation.
* **R3** (15 cells, the live gate): R1 does not replicate. ρ = +0.041.
* **R2**: VOID, registered as conditional on R0.

**Closed precisely:** head-of-line blocking is not what the learned arms pay, and neither
`min_reachable_servers` nor the clients-per-server imbalance predicts a cell's excess queue
over its own reactive arm, at the x4000 rung, on 15 independent topology draws from this
generator. **Not closed:** what *does* make one topology draw a disaster and another a win —
the excess ranges from −29.2 s to +51.6 s across draws and nothing measured here explains it;
and the descriptive pattern above, which is a new registration if pursued.
