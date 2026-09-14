# drainable_regime_v1 — does the peer-affinity environment carry its own mechanism at a servable load?

**Status:** `REGISTERED` — signed off 2026-09-14, **before any rung has been run**. Stage 1 (S0, the
live characterisation screen) is submitted with the bars below already committed.

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

**Cost:** 4 CPU tasks, 12 h limit each, no GPU, no training. Nothing downstream is chained to it.

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
`scripts_cosim/drainable_regime_s0_read.py`. Job ids recorded here on submission.
