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

*(dated entries appended as each read lands)*
