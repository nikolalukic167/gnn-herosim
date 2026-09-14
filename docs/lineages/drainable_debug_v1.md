# drainable_debug_v1 — why do both learned arms lose to reactive Knative at ρ ≈ 0.16?

**Status:** `REGISTERED` 2026-09-14 — bars below are signed before any datum exists.

**Parents:** [`drainable_regime_v1`](drainable_regime_v1.md) (S1 live gate: `gnn` −226.04 % vs
`knative_network`, 0/16) and [`drainable_serving_config_v1`](drainable_serving_config_v1.md)
(no batch policy rescues it; the best window still leaves both learned arms ~2× slower).

## The question

Two lineages have now measured **that** the learned arms lose at a drainable load. Neither
measured **why**, and the three candidate causes imply three different next experiments:

| # | hypothesis | the lever it implies |
|---|---|---|
| **H-env** | at this load the one-step label has nothing over shortest-queue; the environment, not the model, is the ceiling | the load / cluster regime (Phase 2) |
| **H-gap** | the checkpoint fails to reach its own label's optimum on served states | more data or more capacity (Phase 3) |
| **H-myopia** | the checkpoint *wins* one-step and still loses live | the objective (cross-batch state), neither of the above |

This lineage is a **diagnostic**, not a comparison: it runs four reads (D1, D2, D3, D5) whose
job is to say which of the three is true, so that the expensive Phase 2 / Phase 3 registrations
are ordered by a measurement rather than by taste.

## What the decomposition already says (parent data, no new runs)

Medians over 16 seeds at the best window (config E, 16 s), from `drainable_serving_config_v1`:

| arm | latency | platform queue | batch wait | init (cold + rendezvous) | scale events |
|---|---|---|---|---|---|
| `knative_network` | 25.95 | 17.76 | 0.00 | 2.64 | 1,036 |
| `gnn` | 53.45 | 39.59 | 7.22 | 1.28 | 6,091 |
| `mpoff` | 51.32 | 37.41 | 7.22 | 1.27 | 6,382 |

Three facts about the code that this table has to be read against, established 2026-09-14 and
recorded here because they are not in the parents:

1. **`queueTime` is platform FIFO wait.** A `Platform` serves one task at a time
   (`src/placement/infrastructure.py:1240-1251`), and `queue_time = arrived − scheduled`
   (`:333`). Batch wait is a separate column (`waitTime`, `:332`). So the learned arms' excess
   is 7.2 s of batch wait **plus ~22 s of extra serialization behind other tasks**, while they
   *win* on cold start and rendezvous (init 1.28 s against 2.64 s).
2. **The autoscaler is the same class for every arm** — same Knative formula, reconcile 1 s,
   keep-alive 30 s, target concurrency 100 (`src/placement/autoscaler.py`, `constants.py:10-12`);
   the gate sbatch exports no autoscaler knob per arm. The 6× `scaleEventCount` is a
   *consequence* of deeper queues, amplified by one scaling delta being applied to each of
   5 platform types × 4 task types per tick (`autoscaler.py:129-158`).
3. **The cluster is ~98.5 % idle at this rung** (`averageOccupation` 0.014–0.019,
   `unusedPlatforms` 87–94 %, 20 of `cell_s7901`'s 26 nodes are clients). Scale-up fires only at
   `ceil(total_queued / 100) > replicas`, so every arm packs onto ~1–2 replicas per type and
   placement reduces to queue balancing — the regime shortest-queue is built for. Every co-sim
   corpus state was captured at the same target concurrency 100
   (`src/executecosimulation.py:1348,1601,2532,2667`).

## Design

Everything fixed at the parent's settings: trace `drainable_f4000_n50000.json`, cell `cell_s7901`
at the 16 s window (config E), T1b `lr2e3` checkpoints, uncapped, `HEROSIM_PEER_EXCHANGE=1`,
server-only replicas, `node_disk_v2`, CPU, no `--seed`. Four reads:

- **D1 one-step diagnostic.** Capture ≥ 40 peer-group-aligned live batch states per source
  (`knative_network`, `gnn`, `mpoff`), cut them into brute-forced co-sim datasets with the
  `peer_affinity_warm_v1` W0 pipeline, and score four plans per state: the sweep optimum, the
  **shortest-queue plan** recomputed offline from the snapshot (a pure function of the snapshot,
  so it exists for every source), and the `gnn` / `mpoff` checkpoint decodes.
- **D2 per-task read.** Re-run 6 arms with the raw result retained and read the per-task records:
  where queue time comes from, how concentrated placement is, and whether peers are co-located.
- **D3 target-concurrency probe.** `knative_network` alone at
  `HEROSIM_QUEUE_LENGTH ∈ {1, 2, 4, 10, 100}` at x4000 and x2000. This asks whether the
  "drainable" regime is itself an artifact of a target concurrency of 100 against serial replicas.
- **D5 serving-knob screen** (1 seed, exploratory). `GNN_PREFIX_CONCURRENCY_PENALTY ∈
  {0, 0.5, 1, 2}` × {`gnn`, `mpoff`} — the decoder's standing-load control, unset in every gate
  so far (`src/policy/gnn/prefix_serving.py:273-275`).

## Bars (signed before any datum)

Sign convention as elsewhere: regret % = 100 · (plan − optimum) / optimum. All D1 statistics are
medians over the datasets of one source, and a source is read only with ≥ 40 datasets.

| bar | statistic | threshold | what it decides |
|---|---|---|---|
| **D1a** | shortest-queue plan regret vs the sweep optimum, `knative_network` source | ≤ **3.0 %** ⇒ H-env supported | if the reactive rule is already near-optimal one-step, no supervised arm can win here |
| **D1b** | `gnn` checkpoint decode regret on the *same* states | ≥ **10.0 %** and ≥ D1a ⇒ H-gap supported | the model does not reach its own label's optimum |
| **D1c** | `gnn` regret < shortest-queue regret on its own source, while the live gate has it losing | ⇒ H-myopia supported | one-step quality is not the live problem |
| **D2** | share of the learned arms' *excess* queue time (vs `knative_network`) attributable to same-batch predecessors on the same platform | ≥ **50 %** ⇒ the loss is self-inflicted co-location | names serialization, not the environment |
| **D2b** | `chosen_queue_vs_min` (mean), `gnn` vs `knative_network` | reported, no threshold | the `queue_features.py:13-21` statistic at this rung |
| **D3** | `knative_network` mean latency at the best Q ≤ 4 as a fraction of its Q = 100 latency | < **50 %** ⇒ the rung's regime is a target-concurrency artifact | every Phase 2 contrast is then run at both Q values |
| **D5** | best knob value's `gnn` latency on seed 1, as a fraction of the gap to `knative_network` closed | ≥ **30 %** ⇒ promote to a 16-seed registered arm | exploratory; a single seed never closes anything |

**Two floors added to D2 before any D2 datum existed (2026-09-14, same day, while the read
tool was being smoke-tested on an unrelated x200 result).** A share of a negligible excess is
noise dressed as a finding: the first smoke printed "167.6 % of a 2.49 s excess" against a
reference queue of 1,334 s per task, i.e. 0.19 %. D2 therefore decomposes an arm's excess only
when it is **both ≥ 1.0 s per task and ≥ 5 % of the reference's queue time**; below either, the
verdict is NO-EXCESS and no share is computed. Neither floor touches the case the bar exists
for — at the drainable rung `gnn` carries 39.6 s against a 17.8 s reference, an excess of 122 %.
The 50 % threshold itself is unchanged.

**Controls that VOID a read.** D1 is VOID for a source with < 40 datasets, or if the shortest-queue
plan or a checkpoint plan is absent from that dataset's enumerated sweep (fail loud, never
substitute). D2 is VOID if `taskResults` is empty (the low-memory stats path) or if
`queueSnapshotAtScheduling` is null where `chosen_queue_vs_min` is reported. D3 is VOID if
`run_provenance` does not echo the intended `HEROSIM_QUEUE_LENGTH`.

**This lineage closes on its reads, and it orders the live gates that follow; it does not
replace them.** Phase 2 (load ladder) and Phase 3 (corpus ladder) are separate registrations,
each ending in a live gate per rule 6.

## Amendment to the 2026-09-14 arrival-rate hard stop

`docs/hard-stops.md` § "The peer-affinity environment at a servable load" forbids re-running the
peer-affinity checkpoints at another arrival rate, on the premise that "a learned arm fitted on
states from a 940× overload does not transfer to a cluster with headroom". **That premise is false
for the cold T1b checkpoints**, which is what the stop's own gate served: the cold corpus states
come from the generator's seeded queue draws (`shallow_pois2`, `deepvar_uniform0_12`,
`deepvar_pois4`; dim-7 max 42 over 516 datasets), not from the live overload — it was the *warm*
corpus that was cut from served states, and the *gate trace*, not the corpus, that carried the
940× overload. The stop stands as written for warm checkpoints and for "run it again and hope".
It is amended here to permit a **registered ladder with a control that reads the arms' own
counters**, which is what Phase 2 is. The measurement that closes or confirms it is that ladder.

## Record

### 2026-09-14 — Phase 0 landed (tooling only, no experimental datum)

Five engineering changes, each verified label-invariant before anything else was built on it.

**E1 — the root logger was below its own handler.** `simulation.py` configured `level=DEBUG`
with a single `ERROR` handler, so every `logging.info` in the event loop built a full
`LogRecord` — including the stack walk `%(funcName)s` forces — and then had it discarded.
py-spy had already put 48 % of samples there (`drainable_regime_v1`, 2026-09-14). Root level
now matches the handler. Verified: `total_rtt`, `num_tasks`, `scaleEventCount`, `endTime`,
`averageQueueTime`, `averageElapsedTime` and `totalPeerExchangeTime` byte-identical before and
after on the 3k smoke, for `knative_network` and `knative_network_batch`.

**E2 — no gate in this program could be read per task.** `Orchestrator._use_low_memory_stats`
switches to the streaming path above 10,000 events, which writes `taskResults: []`, and the
gate sbatch deleted the raw result after summarising. `KEEP_RAW=1` now exports
`SIM_FORCE_FULL_STATS=1` (optionally `GNN_CAPTURE_DATASET_STATE` via `KEEP_RAW_QSNAP=1`) and
keeps the raw file. Both hooks are off by default.

**E2b — two schedulers never recorded the per-task candidate snapshot.** The masked_topo
prefix path and the `knative_network_batch` subclass both omitted the
`queue_snapshot_at_scheduling` capture that the per-arrival Knative scheduler and the GNN
argmax path have always written, so `chosen_queue_vs_min` — the statistic
`queue_features.py:13-21` names as the live failure mode — was unreadable for precisely the
arms this lineage gates. Both now capture it behind the existing `GNN_CAPTURE_DATASET_STATE`
flag. Verified inert with the flag off (byte-identical `total_rtt` on the 3k smoke) and
label-invariant with it on, in both batching modes.

**E3 — target concurrency is now visible.** `HEROSIM_QUEUE_LENGTH` is echoed into the job log
and the resolved value is carried into the summary, which is all that survives a gate run.

**E4 — the D2 read exists, with 21 tests, before any record it will read.** One of those tests
caught a real defect in the read: `float(rec.get("scheduledTime") or 0.0)` treats a legitimate
simulated time of 0.0 as absent, so a batch committed at t = 0 counted as "no timestamp" and
the same-batch attribution silently returned zero.

A sanity reading on an unrelated x200 smoke behaves as the overload regime predicts: 98.1 % of
`knative_network`'s queue time is behind *some* predecessor but **0.0 %** behind a same-batch
one — at 940× overload the queue is standing load, not self-inflicted — and 25.5 % of tasks had
only one legal replica, so a quarter of that trace cannot distinguish two schedulers at all.


### 2026-09-14 — D3 read: the rung is real, but the ladder below it was not

`knative_network` alone, `HEROSIM_QUEUE_LENGTH ∈ {1, 2, 4, 10, 100}`, two rungs, ten runs
(jobs 766030–766039). Read tool `scripts_cosim/drainable_debug_d3_read.py` (5 tests, bar
committed first); reading `simulation_data/drainable_regime_v1/d3_read.json`.

| rung | Q=1 | Q=2 | Q=4 | Q=10 | Q=100 | best low-Q as % of Q=100 |
|---|---|---|---|---|---|---|
| **x4000** latency s | 18.40 | 19.78 | 21.52 | 25.49 | **25.95** | **70.9 %** |
| **x2000** latency s | 14.39 | 16.07 | 19.67 | 28.69 | **141.73** | **10.1 %** |

**x4000: REGIME-IS-REAL.** The rung both parent lineages gated at does not clear the 50 % bar.
At the default target concurrency Knative is within 29 % of its own best, so its −226 % win over
the graph arm is not an artifact of that constant.

**x2000: TARGET-CONCURRENCY-ARTIFACT.** One rung down, the default is **9.8× worse** than Q = 1.
Scale-up fires at `ceil(total_queued / Q) > replicas`, so at Q = 100 a replica must reach a
hundred queued tasks before a second appears; at ρ ≈ 0.33 the arrivals never build that backlog
fast enough and the cluster runs on too few replicas for most of the trace. **Every latency
figure on the S0 ladder between the overload and x4000 is therefore a reading about
`QUEUE_LENGTH = 100`, not about the rung** — including the x2000 row of the S0 pass-2 table in
`drainable_regime_v1`. The peer-share and queue-column columns that chose x4000 are unaffected
(they are per-task ratios, not throughput), so the rung choice stands; the latency column does
not. Consequence for Phase 2, as registered: every contrast on the load ladder runs at Q = 100
**and** at the low Q, and both are reported.

A side observation that closes an old story for good: at x4000 Q = 1 the reactive arm issues
**75,805** scale events — 12× the graph arm's 6,091 at the same rung — and is **29 % faster**.
Autoscaler churn is not a cost in this simulator. The "churn" mechanism withdrawn from
`drainable_regime_v1`'s first read should not be revived in any form.

### 2026-09-14 — D2 read: the excess queue is real serialization, and it is NOT self-inflicted batching

Six arms at x4000, 16 s window, uncapped, raw results retained (job 766040; read job 766052).
50,000 task records per arm. Reading `simulation_data/drainable_regime_v1/d2_read.json`.

| arm | elapsed | queue | batch wait | init | behind **any** predecessor | behind **same-batch** | effective platforms |
|---|---|---|---|---|---|---|---|
| `knative_network` | 25.95 | 17.76 | 0.00 | 2.64 | 99.2 % | **0.0 %** | 5.13 |
| `knative_network_batch` | 26.03 | 17.82 | 0.02 | 2.63 | 99.2 % | 0.1 % | 5.14 |
| `gnn_s1` | 48.60 | 35.09 | 6.94 | 1.20 | 99.2 % | **11.3 %** | 5.86 |
| `gnn_s2` | 73.78 | 60.28 | 6.94 | 1.18 | 99.3 % | 6.7 % | 5.72 |
| `mpoff_s1` | 59.66 | 46.11 | 6.92 | 1.19 | 99.3 % | 8.8 % | 6.05 |
| `mpoff_s2` | 60.17 | 46.69 | 6.92 | 1.20 | 99.3 % | 8.7 % | 5.83 |

**D2 does not fire: NOT-SELF-INFLICTED on all four learned arms** (9.6–22.9 % of the excess
against a 50 % bar). Queue time is essentially all serialization — 99.2–99.3 % of it is
provably behind another task on the same platform, for *every* arm including the reactive one —
but only a tenth of the learned arms' excess is behind a task their own decode placed. **The
excess is cross-batch: they put tasks onto replicas that earlier decodes had already loaded.**

The statistic that says why, `chosen_queue_vs_min` — how much deeper than the shallowest legal
replica each placement went:

| arm | mean | p95 | max | placements above the minimum |
|---|---|---|---|---|
| `knative_network` | 0.000 | 0 | 1 | **0.005 %** |
| `gnn_s1` | 3.65 | 19 | 100 | **35.5 %** |
| `gnn_s2` | 6.86 | 43 | 183 | 36.0 % |
| `mpoff_s1` | 5.00 | 18 | 158 | 36.3 % |
| `mpoff_s2` | 5.07 | 19 | 165 | 35.2 % |

Knative is a scale-free `min` re-read per task, so its number is 0 by construction. The learned
arms take a deeper replica in **~36 % of the placements where a choice existed**, and at the
95th percentile that replica carries 18–43 more queued tasks. This is `queue_features.py:13-21`
measured at a drainable load, and it is the same for the graph arm and its pointwise twin.

**What the learned arms buy with it: almost nothing.** Peer co-location, the environment's own
objective, is flat across every arm — same-platform pairs 20.77 % (reactive) against 21.4–21.8 %
(learned), and free exchange bytes 34.0 % against 34.0–35.3 %. A 0.7 pp co-location gain is paid
for with 19 extra queued tasks at p95. The learned arms do win the terms the parents already
credited them with (init 1.20 s against 2.64 s, covering cold start and rendezvous), and they
spread *more*, not less (effective platforms 5.7–6.1 against 5.13) — consistent with
`serving_gap_v1`'s refutation of the herding hypothesis.

**Two facts about the cell that bound what any scheduler can do here.** 53–58 % of tasks have
exactly **one** legal replica, so more than half the trace cannot distinguish two schedulers at
all; and seed variance inside one arm is larger than the gap between arms (`gnn_s1` 48.60 s
against `gnn_s2` 73.78 s), which is why B1-style contrasts need their 16 seeds.

### 2026-09-14 — D5 screen: the decoder's standing-load knob helps, and not nearly enough

`GNN_PREFIX_CONCURRENCY_PENALTY ∈ {0, 0.5, 1, 2}` × {`gnn`, `mpoff`}, seed 1, x4000, 16 s
window, uncapped (jobs 766085–766087; penalty 0 is the D2 run). Registered as exploratory: a
single seed never closes anything, and the bar is a promotion criterion — 30 % of the gap to
`knative_network` — not a verdict.

| penalty | `gnn` latency | gap closed | `mpoff` latency | gap closed |
|---|---|---|---|---|
| 0 | 48.60 | — | 59.66 | — |
| 0.5 | 46.41 | 9.7 % | 58.47 | 3.5 % |
| **1** | **44.89** | **16.4 %** | 58.38 | 3.8 % |
| 2 | 57.80 | −40.6 % | 60.05 | −1.1 % |

**D5 does not fire (16.4 % against 30 %); the knob is not promoted to a registered arm.** It
does move in the direction D2 predicts — the one control that lets the decoder see standing
load buys back a sixth of the graph arm's deficit — and it over-corrects at 2, which is what a
penalty fighting a scale-free `min` looks like. It is not the missing term. Read with an ad-hoc
script rather than a registered tool, which is disclosed here because it is allowed only by
D5's exploratory standing; the threshold itself was committed in this node before the runs.

Also recorded: at penalty 0 the `mpoff` arm reproduces its D2 latency to the digit (59.66 s),
so the two submissions are the same experiment.
