# drainable_regime_v1 — does the peer-affinity environment carry its own mechanism at a servable load?

**Status:** `CLOSED` 2026-09-14 on its **S1 live gate** (rule 6) — **POINTWISE-BETTER,
REACTIVE-WINS**. At ρ ≈ 0.16, the only rung on this ladder a reviewer would accept as a load
level, `gnn` is 4.9× slower than its own MP-OFF twin (1/16 seeds) and 18.6× slower than reactive
Knative (0/16); `mpoff` is 3.2× slower than Knative. The mechanism is autoscaler churn
(`scaleEventCount` 1,036 → 1,774 → 20,400), not the peer objective, which is within 9 % on all
four arms. **The x200 p2 GNN-NEEDED reading and the platform cap's +21.9 % are both properties
of the 940× overload.** Registered 2026-09-14 with every bar committed before any rung ran.

Two S0 pass-2b arms (the x4000/x8000 *batch* rungs, job 763577) were still running when the live
gate landed; they inform **S4**, a screen bar about batch alignment, and cannot change the live
verdict. Their reading is appended to the S0 record when it arrives.

**Parent:** [`peer_affinity_v1`](peer_affinity_v1.md) (the environment, the cell, the trace) and
[`peer_affinity_warm_v1`](peer_affinity_warm_v1.md) (whose W0 record names "a drainable served regime"
as the open environment question this lineage takes up).

## Why this exists — the measurement that forces it

`peer_affinity_v1` was built so the cost is indexed by *pairs of task instances*, which is the one
mechanism the count theorem does not flatten and the reason the lineage escaped every neighbouring
hard stop. On the served gate that mechanism is **0.01 % of the number the gate scores.**

Measured on the landed x200 p2 capped gate (16 seeds per arm, `cell_s7901`,
`workload-150-150-peer_p2_x200.json`, 450,729 tasks), medians over seeds:

| statistic | `gnn` | `mpoff` | `knative_network` |
|---|---|---|---|
| average task latency, s | 34,893.4 | 35,646.1 | 44,663.0 |
| of which queue, s | 34,890.1 | 35,642.7 | 44,657.6 |
| of which communications, s | 3.43 | 3.46 | 5.35 |
| of which execution, s | 0.0315 | 0.0305 | 0.0387 |
| **queue share of latency** | **99.9903 %** | **99.9906 %** | **99.9879 %** |
| **`totalPeerExchangeTime` / `total_rtt`** | **0.0098 %** | **0.0096 %** | **0.0119 %** |

Capacity and load, same runs: Knative drains 450,729 tasks by `endTime` 159,215 s = **2.83 tasks/s**
against **2,659 arrivals/s** — a **940×** overload. Mean service time is 3.46 s, so the cluster is
about **10 parallel channels**. The supervised target and the live statistic are therefore both
~99.99 % queue drain, which is a sum of per-(task, platform) terms and is exactly what
`peer_affinity_warm_v1` W0.b measured as pointwise-recoverable (median regret 0.0 %, 0/100 above 2 %).

**No architecture can win a contest whose objective is 99.99 % a quantity a pointwise scorer reads off
one column.** This lineage asks whether any arrival rate exists at which the environment's own
mechanism is a material share of its own objective. It changes no physics, no payload, no topology and
no policy: the only moving part is the timestamp scale.

## Stage 1 — S0, the live characterisation screen (submitted 2026-09-14)

`scripts_cosim/rescale_workload_arrivals.py` multiplies every event timestamp by `--factor` and
touches nothing else: same events, same types, same peer groups, same 200 MB payloads, same demand
scales, same seed. Four rungs on `workload-150-150-peer_p2_x200.json`, `cell_s7901`, policy
`knative_network_batch` with `KNATIVE_BATCH_BY_PEER_GROUP=1` (the corpus bridge's own source) and
`LIVE_AUDIT` snapshots on, one SLURM task per rung:

| rung | factor | arrivals/s | nominal ρ vs 2.83/s | role |
|---|---|---|---|---|
| R0 | 300 | 8.86 | 3.13 | control: still overloaded |
| R1 | 1,000 | 2.66 | 0.94 | near critical |
| R2 | 2,000 | 1.33 | 0.47 | drainable |
| R3 | 4,000 | 0.66 | 0.23 | slack |

**Amendment 1 (2026-09-14, signed before any rung produced a number).** The rungs' *realised*
arrival rates came out below the projection: the first 50,000 events of the trace arrive at
1,842/s, not the whole trace's 2,659/s, so the ladder landed at 6.14 / 1.84 / 0.92 / 0.46
arrivals/s = ρ 2.17 / 0.65 / 0.33 / 0.16. That leaves the critical band ρ ∈ [0.8, 1.0]
unsampled, and it is exactly the band where the two guards can fight: peer-group alignment (S4)
degrades as the rate falls, while the mechanism (S1) only appears once it has. Two rungs are
added, submitted before any rung's numbers were read and with no bar changed:

| rung | factor | arrivals/s | realised ρ | role |
|---|---|---|---|---|
| R1b | 500 | 3.68 | 1.30 | just over capacity |
| R1c | 700 | 2.63 | 0.93 | the critical band |

The screen is therefore 6 rungs, 12 arms. Bars S1–S5 are untouched; the control stays R0.

**Bars, fixed here before the data exists.** A rung PASSES only if all four of S1–S4 hold on it.

- **S1 — mechanism (primary).** `totalPeerExchangeTime / total_rtt` ≥ **20 %**. Below that the
  peer term cannot move the scored statistic no matter how well it is predicted.
- **S2 — trained range (primary).** live dim-7 queue column (`legacy_v0`) p90 ≤ **42**, the maximum
  over all 516 cold training datasets. Above it the served queue feature is out of the range every
  checkpoint was fitted on (`herosim-queue-column-is-broken-live`), which is the defect that
  contaminated every previous live reading.
- **S3 — placement sensitivity (guard).** `totalPeerRendezvousWait / total_rtt` < **50 %**. At low
  arrival rates a peer group's members no longer co-arrive, so the pair cost can turn into *waiting
  for a partner to exist*, which is indexed by arrival time and not by placement. A rung where
  rendezvous dominates is not usable even if S1 passes on it.
- **S4 — corpus feasibility (guard).** ≥ **12** peer-group-aligned `LIVE_AUDIT` snapshots on the rung.
  The bridge cuts datasets from aligned batches; a rung that cannot form them cannot host a corpus.
- **S5 — control (VOID condition, not a pass bar).** R0 must reproduce the overloaded regime: peer
  share < 1 % and queue share > 99 %. If R0 does not, the rescale moved something other than the
  arrival rate and the whole screen is VOID rather than read.

**GO** — at least one rung passes S1–S4 with S5 holding. The GO names that rung and orders stage 2:
a co-sim corpus cut at it, the T1b recipe at 16 seeds per arm, and the registered live gate. Stage 2
is a separate registration written before its own data.

**NO-GO** — no rung passes. Then the peer-affinity environment does not carry its own mechanism at
any servable arrival rate on this cluster, and the direction closes in `docs/hard-stops.md`. This is a
*live* measurement on the live simulator, so closing on it honours CLAUDE.md rule 6; what rule 6
forbids is closing on an offline read, and nothing here is offline.

**Descriptive readings, recorded but not bars:** `averageOccupation`, `endTime`, `scaleEventCount`,
the latency decomposition of the table above at each rung, realised batch sizes and
`peer_group_incomplete_batches`, and whether `knative_network` and `knative_network_batch` separate at
all at each rung (if two reactive policies are indistinguishable on a rung, placement has no room
there regardless of what S1 says).

**Cost:** 12 CPU tasks (6 rungs x 2 policies), 12 h limit each, no GPU, no training. Nothing downstream is chained to it.

## What this can and cannot establish

It can establish that a servable load exists at which the environment's pair-indexed cost is a
material share of its own objective, with the queue feature inside its trained range. That is a
precondition for any graph-reasoning claim on this simulator, and it has never been checked.

It cannot establish that message passing wins there. Every prior stop still applies at the new rung:
the label may still be pointwise-recoverable (W0.b's statistic is the screen stage 2 must pass), and
the offline/live reversal of `peer_affinity_v1` remains unexplained. A GO buys the right to build a
corpus, nothing more.

## Record

### 2026-09-14 — registered and submitted

Bars above fixed before any rung ran. Screen implemented as
`scripts_cosim/datalab/drainable_regime_v1_s0_screen.sbatch`, read tool
`scripts_cosim/drainable_regime_s0_read.py` (8 tests), truncating rescaler (5 tests), all green
before submission. Submitted at f65edb2: traces **763543** (COMPLETED, 7 s), screen **763544**
(8 arms, R0/R1/R2/R3), read **763545**. Amendment 1's rungs R1b/R1c submitted as traces **763553** +
screen **763554** (4 arms); the original read 763545 was cancelled and re-chained behind both
arrays as **763555** (`afterok:763544:763554`). Realised ladder from the
traces job: x300 6.139 arrivals/s, x500 3.684, x700 2.631, x1000 1.842, x2000 0.921, x4000 0.460;
50,000 events and 88,894 peer pairs per rung, identical across rungs as the rescale requires.

### 2026-09-14 — S0 pass 1 read (763544 + 763554, read 763555): **NO-GO as run, and S4's failure is a screen artifact**

Twelve arms, 25–50 s each (a drainable cluster simulates far faster than an overloaded one).
S5 control HOLDS: the x300 rung reproduces the overloaded regime (peer share 0.32 %, queue share
99.67 %), so the rescale moved the arrival rate and nothing else, and the ladder is readable.

| rung | arrivals/s | ρ | peer % of `total_rtt` | rendezvous % | queue % of latency | dim-7 busy p90 | aligned batches | mean batch size |
|---|---|---|---|---|---|---|---|---|
| x300 | 6.139 | 2.17 | 0.32 | 0.00 | 99.674 | 391.0 | 0 | 1.09 |
| x500 | 3.683 | 1.30 | 0.77 | 0.01 | 99.213 | 195.2 | 0 | 1.07 |
| x700 | 2.631 | 0.93 | 1.31 | 0.02 | 98.654 | 267.0 | 0 | 1.03 |
| x1000 | 1.842 | 0.65 | 1.80 | 0.04 | 98.136 | 187.0 | 0 | 1.01 |
| x2000 | 0.921 | 0.33 | 3.24 | 0.38 | 96.338 | 78.7 | 0 | 1.01 |
| **x4000** | 0.460 | 0.16 | **21.15** | 10.10 | 68.458 | **6.0** | 0 | 1.01 |

**Verdict as run: NO-GO** — no rung clears all four bars. **What it establishes anyway:**

- **The mechanism is rate-controlled, and it is large.** Peer exchange rises monotonically from
  0.32 % of `total_rtt` to **21.15 %**, against **0.0098 %** on the landed gate. S1 passes at x4000.
- **The queue feature comes back into its trained range at the same rung.** dim-7 busy p90 falls
  391 → 6.0 against the cold-corpus maximum of 42; x2000 is still out of range at 78.7. S2 passes
  only at x4000, the same rung as S1. Both primaries fire together, at ρ ≈ 0.16.
- **S4 fails on every rung, including the overloaded control, and that is a defect in the screen,
  not a reading.** Mean batch size is **1.01–1.09 everywhere**: `KNATIVE_BATCH_TIMEOUT` was held at
  0.02 s while inter-arrival times stretched from 0.0004 s to 2.17 s, so the batching window shrank
  by the same factor as the rate and no peer group could ever assemble. The identical policy at the
  landed gate's 2,659 arrivals/s aligns 149 of 150 batches. `peer_group_incomplete_batches` is
  39,071–44,991 of 50,000 on every rung. The bar cannot distinguish "peer groups do not co-arrive
  at this rate" from "the batch window was 4,000× too short", so it must be re-run before it is read.
- **S3 is the same tension, honestly measured.** Rendezvous wait climbs 0.00 % → 10.10 % as the
  rate falls. Assembling or awaiting a peer group costs more the slower the arrivals, which is the
  real force S4 was trying to see. At x4000 a 10-task group takes ~21.7 s to arrive.

### 2026-09-14 — Amendment 2 (signed before pass 2 produces a number)

`KNATIVE_BATCH_TIMEOUT` is a policy time constant, not physics, and holding it fixed across a
4,000× stretch is the confound above. Pass 2 re-runs the ladder with the window scaled per rung
(`DRAIN_SCALE_BATCH_TIMEOUT=1`, window = 0.02 s × factor) and adds **x8000** (ρ ≈ 0.08) to find
where the S3 guard closes the window from below. The batch loop exits the moment the group is
complete, so a longer window costs nothing when the group arrives sooner. **No bar changes**; pass
1's numbers stand as recorded and pass 2 is written to `s0_pass2/` so neither overwrites the other.
Pass 1's S1/S2 readings are unaffected by the batch window (they are per-task statistics), so the
open question pass 2 answers is exactly S4 and S3 at a comparable batching policy.

### 2026-09-14 — Amendment 2a: the poll interval is a policy time constant too

Pass 2's first submission (763560) reproduced S4 correctly on the five rungs that finished —
mean batch size **9.82** and **143/150 aligned** at x300 through x2000, against 1.01–1.09 and
0/150 in pass 1 — which confirms pass 1's S4 failure was the fixed batch window and nothing else.
Its x4000 and x8000 batch arms were then cancelled as stalled. **That diagnosis was wrong and is
corrected here.** Two samples of the simulated clock taken minutes apart differed by 4 s, and I
attributed it to `_collect_batch`'s hard-coded `poll_interval = 0.001` s (an 80 s window would cost
80,000 simpy timeout events per batch). The resubmitted arms carry the scaled poll — `poll=4.0` at
x4000, `poll=8.0` at x8000, confirmed in their logs — and **run at the same speed**: 36,448
simulated seconds after 29 min, against 37,718 after 30 min before the fix. The arms were never
stalled; they are simply slow, about a third of their span per half hour, ~85–170 min each, and
cancelling them was premature.

`py-spy` on the x4000 arm (1,751 samples) gives the real cost, and it is neither the poll nor the
batch window: **59.3 % of samples are inside `knative_network/autoscaler.py:create_first_replica`**
— the starved-client retry loop recorded in `docs/gates/gate-tools.md` on 2026-09-14 — and
**48 % of total samples are `logging.debug` building `LogRecord`s that the ERROR-level handler then
discards** (`makeRecord` 23 %, `findCaller` 4 %, `_is_internal_frame` 2.6 %). `_collect_task_batch`
is **5.1 %**. The root logger is configured at `DEBUG` while its only handler is at `ERROR`
(`simulation.py:656`), so every `logging.debug` call in the simulator pays full record construction
including a stack walk, for output nobody ever sees. That is ~45 % of the runtime of a long run, and
it is label-invariant to fix. **Not fixed here** — it touches the shared simulator mid-lineage and
no bar depends on it; recorded for a separate change.

**Amendment 2a stands as a change but not as a diagnosis.** The poll interval genuinely is a policy
time constant that must scale with the window, and leaving it at 1 ms inside an 80 s window is a
latent hazard worth closing; it simply was not what made these two arms slow.

`KNATIVE_BATCH_POLL_INTERVAL` and `GNN_BATCH_POLL_INTERVAL` now set it explicitly and fail loud on
a non-number or a non-positive value; **unset, the expression is exactly what it was, so every run
in the record stays bit-identical.** The ladder scales both together, holding the landed gate's
ratio of 20 polls per window at every rung. Pass 2 is resubmitted from the top with this in place.
No bar changes.

## Stage 2 — S1, the live gate at the candidate rung (registered 2026-09-14, before any arm is read)

**Rule 6: this gate runs whatever S0 returns.** S0 orders the work; it does not close it. If S0
returns NO-GO on every rung, S1 still runs at x4000 and its verdict is recorded with that NO-GO
named alongside it.

### Why x4000 is the rung

From the pass-2 screen, reading only the arms that had completed (the two x4000/x8000 *batch* arms
were still running and are not used to choose the rung):

| rung | arrivals/s | peer % of `total_rtt` | rendezvous % of peer | queue % of latency | dim-7 p90 | cool-down |
|---|---|---|---|---|---|---|
| landed gate (x200) | 2,659 | 0.0098 | — | 99.99 | ~574 | **99.89 %** |
| x700 | 2.631 | 1.33 | 1.42 | 98.64 | 143.0 | 9.56 % |
| x1000 | 1.842 | 1.55 | 2.54 | 98.40 | 240.9 | 6.25 % |
| x2000 | 0.921 | 3.69 | 12.34 | 95.82 | 105.0 | 3.61 % |
| **x4000** | **0.460** | **21.19** | **47.86** | **68.44** | **6.0** | **0.07 %** |
| x8000 | 0.230 | 20.49 | **136.71** | 51.20 | — | 0.03 % |

x4000 is the only rung that clears S1's 20 % peer share while staying inside S3's 50 % rendezvous
guard, and it is the only rung whose served queue column (dim-7 p90 = 6.0) falls inside the range the
corpus actually contains (cold max 42). x8000 is excluded: its rendezvous wait **exceeds** its
exchange time, so tasks spend longer waiting for a peer to appear than exchanging with one.

### Slate

The stage 3 gate slate, unchanged, on the drainable trace — `peer_affinity_v1_stage3_live_gate.sbatch`
with `WORKLOAD=drainable_f4000_n50000.json`, `CFG=cell_s7901.json`:

- `knative_network` — reactive, per arrival, peer-blind
- `knative_network_batch` — reactive, 10-task window at the real 0.02 s
- `gnn` × 16 seeds — T1b `lr2e3`, `masked_topo`, `GNN_BATCH_BY_PEER_GROUP=1`
- `mpoff` × 16 seeds — identical, `PeerConv` never runs

Run twice: `GNN_PREFIX_PLATFORM_CAP=1` (the capped headline config) and unset. 68 arms. Physics,
seeds, warmth, `PYTHONHASHSEED` and device are the stage 3 gate's, unchanged. **No retraining** —
these are the same checkpoints the x200/x800 gates served, so the only moving part between this gate
and the landed one is the arrival rate.

### Bars

- **B1 — primary, `gnn` vs `mpoff`.** Paired over 16 seeds on `total_rtt`, Wilcoxon signed-rank.
  **GNN-NEEDED** if `gnn` is faster at p < 0.05 with ≥ 12/16 seeds; **POINTWISE-BETTER** if `mpoff`
  is, on the same test; **TIE** otherwise.
- **B2 — `gnn` vs reactive.** Same test against the faster of the two Knative arms.
  **LEARNED-WINS** / **REACTIVE-WINS** / **TIE**.
- **B3 — cap dependence.** B1 and B2 are read separately capped and uncapped. If the two disagree on
  either verdict, the headline is recorded **CAP-CONTINGENT**, as the x200 gate's was.
- **B4 — regime control, VOID if it fails.** On `knative_network`'s own run: cool-down share < 5 %
  **and** peer share ≥ 20 % of `total_rtt`. If the gate does not reproduce the screen's regime it did
  not measure what this stage claims, and B1–B3 are not read.
- **B5 — queue-range control.** dim-7 p90 over busy platforms ≤ 42. Above it, the served queue column
  is again outside the trained range and B1/B2 are recorded **OUT-OF-RANGE** rather than as a
  model-class result.

### What each outcome means

A **TIE** on B1 here, against the x200 p2 gate's GNN-NEEDED (+3.94 %, p = 0.0076, 13/16), is not a
null result — it says the one positive live reading in the program is a property of a 940× overload.
A **GNN-NEEDED** on B1 with B2 LEARNED-WINS would be the first measurement in this program where a
graph arm beats both its pointwise twin and reactive Knative, and it would be at the only load the
literature would accept. Both are recorded; neither is preferred.

### 2026-09-14 — S1 LIVE GATE READ: **POINTWISE-BETTER, REACTIVE-WINS**

Jobs 764300 (capped) / 764301 (uncapped), 34 arms each, all 68 COMPLETED; 764303 the B5
snapshot capture. Trace `drainable_f4000_n50000.json` on `cell_s7901`, T1b `lr2e3`
checkpoints, no retraining. Read by `scripts_cosim/drainable_regime_s1_read.py`;
attachment `drainable_regime_v1/s1_live_read.json`.

**Both controls hold, and B5 holds for the first time in this program.**

| bar | value | bar | holds |
|---|---|---|---|
| B4 peer share of `total_rtt` | 21.19 % | ≥ 20 | yes |
| B4 cool-down share | 0.07 % | < 5 | yes |
| B5 dim-7 p90, 427 busy platform-queues over 150 snapshots | **5** | ≤ 42 | **yes** |

The landed x200 gate's equivalents are 0.0098 %, 99.89 % and ~574. **This is the first live
measurement in the program where the served queue column is inside the range the corpus
contains**, so for once B1 and B2 are model-class readings and not extrapolations.

| bar | capped | uncapped |
|---|---|---|
| **B1 `gnn` vs `mpoff`** | **−488.78 %**, p = 9.2e-05, 1/16 → **POINTWISE-BETTER** | −471.97 %, p = 1.5e-04, 1/16 → POINTWISE-BETTER |
| **B2 `gnn` vs `knative_network`** | **−1763.40 %**, p = 3.1e-05, 0/16 → **REACTIVE-WINS** | −1698.52 %, p = 3.1e-05, 0/16 → REACTIVE-WINS |
| `mpoff` vs `knative_network` (descriptive) | −216.49 %, p = 3.1e-05, 0/16 | −214.44 %, p = 3.1e-05, 0/16 |
| **B3** | not cap-contingent — the platform cap changes nothing at this rate | |

**The cap is a property of the overload, not of the decoder.** Its +21.9 %-over-Knative
headline was measured at ρ ≈ 940; at ρ ≈ 0.16 capped and uncapped are indistinguishable on
both bars. `GNN_PREFIX_PLATFORM_CAP` was a concentration control for a cluster whose capacity
mask could not bind; with slack everywhere there is nothing for it to control.

#### Where the time actually goes (medians over 16 seeds, capped)

| statistic | `knative_network` | `knative_network_batch` | `mpoff` | `gnn` |
|---|---|---|---|---|
| average task latency, s | 25.95 | 26.03 | 80.43 | **474.5** |
| of which queue, s | 17.76 | 17.82 | 71.92 | **466.1** |
| of which batch wait, s | 0 | 0.0201 | 0.0182 | 0.0182 |
| `totalPeerExchangeTime`, s | 2.750e5 | 2.753e5 | 2.977e5 | 2.972e5 |
| `totalPeerRendezvousWait`, s | 1.316e5 | 1.314e5 | 1.217e5 | 1.181e5 |
| average pull, s | 6.855 | 6.854 | 5.641 | **13.63** |
| cold-start proportion | 0.89 | 0.892 | 1.653 | 1.509 |
| **`scaleEventCount`** | 1,036 | 1,060 | 1,774 | **20,400** |

Three things this rules out and one it establishes.

- **Not the peer-group batch wait.** `averageWaitTime` is 0.018 s for both learned arms. The
  obvious hypothesis — that waiting ~21.7 s for a 10-task group to arrive at 0.46 arrivals/s
  is what sinks the learned arms — is refuted by the arms' own counters.
- **Not the peer objective.** `totalPeerExchangeTime` is 2.75–2.98e5 s on **all four** arms, a
  9 % spread. The learned arms do not reduce the quantity they were trained to reduce; they do
  not increase it either. Rendezvous wait is actually *lowest* on `gnn`.
- **Not the cap, and not message passing alone.** B3 is flat, and `mpoff` — the same
  architecture with `PeerConv` switched off — loses to reactive Knative by 216 % on its own.
- **It is autoscaler churn.** `scaleEventCount` goes 1,036 → 1,774 → **20,400** across
  reactive, pointwise and graph. Average pull time doubles on `gnn`. The learned decoders were
  fitted on states captured from a cluster in 940× overload, where every platform was saturated
  and placement could not make things worse. Served with slack everywhere, they thrash the
  replica lifecycle, and the queue they create is 26× the reactive arm's.

#### What this closes

The one positive live reading in the program — x200 p2 capped, `gnn` vs `mpoff` **+3.94 %**,
p = 0.0076, 13/16, GNN-NEEDED — **is a property of the 940× overload**. At the only rung on
this ladder that a reviewer would accept as a load level, the same checkpoints, the same cell
and the same physics give **−488.78 %, 1/16, in the opposite direction**, with the queue column
inside its trained range for the first time. The program's offline/live anti-correlation is
therefore not the whole story: there is a third venue, a defensible load, where both learned
arms lose to a reactive baseline by 2–18×.

**No measurement in this program has a graph arm beating both its pointwise twin and reactive
Knative, and the search has now covered the overloaded regime, the warm-state corpus and the
drainable regime.**

### 2026-09-14 — gate-tool correction: B5 read the wrong key and passed vacuously

The first run of the S1 read reported `B5 dim-7 p90 over 0 busy candidates = 0.0 -> holds=True`.
`LIVE_AUDIT` snapshots carry queue depth in `full_queue_snapshot` (`queue_key` → depth); the
reader looked for `candidates[].queue_length`, which is the **co-sim dataset** schema and does
not exist in a live capture. It found nothing, took the p90 of an empty list as 0.0, and passed.
**A bar that cannot find its input must not pass.** Fixed the same session, before the verdict
was recorded: B5 now reads `full_queue_snapshot`, fails loud when no snapshot parses, and
records NOT-APPLICABLE (`holds: null`) when the snapshots parse but nothing ever queues. The
corrected read is the one above (p90 = 5 over 427 busy platform-queues). Regression test in
`tests/test_drainable_regime_s1_read.py`.
