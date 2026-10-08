# kpa_scaleout_v1 — Knative-faithful scale-out as the third physics factor

**Status:** `REGISTERED` (no runs). Depends on: `transfer_physics_v1` (read, `e76862a2`).
Plan: [`reference_physics_programme.md`](reference_physics_programme.md) (shared rules apply).


**Pre-run amendments (2026-10-07, before any `kpa` cell; decided by the coordinator).** Three elements the factor
table left out, added for fidelity to Knative and not chosen by any outcome:
- **A1 buffered demand.** Tasks that arrived and are not yet on a replica (held in a batching scheduler's peer-group
  buffer, or postponed for want of a reachable replica) count as in-flight demand for their type, as Knative's
  activator reports the requests it buffers. Without it CD and the batched arms hide up to 16 s of demand.
- **A2 rate limits.** Knative's defaults `max-scale-down-rate` 2.0 (at most halve per decision) and
  `max-scale-up-rate` 1000. Recorded in `scaleOut`.
- **A3 a never-served replica is not idle.** A reachability-created replica that has not started a task since its
  creation is not a scale-down candidate, for at most one stable window (in Knative the request that triggers a pod
  is already in flight, so the pod cannot be removed before serving it).

Implementation and checks: `src/placement/scaleout.py`, `src/placement/autoscaler.py`; `tests/test_kpa_scaleout.py`
(16 tests). Local legacy replay on `cc40s9001` (4 arms, 4,000 events) identical after the amendments. A one-cell
smoke (not a result, not the test workload) moved short-lived replicas from 66 % to 46 % for CD and 37 % to 34 % for
reactive; CD's reachability creations still outnumber load creations (2,191 : 1,017).
- **A4 one autoscaler for every arm (2026-10-08, before any read).** The first `kpa` launch (`c9e5ed6c`) timed out
  only on `reactive` and `selfpredict`, the two arms on the `knative_network` autoscaler: 40–310 runs per condition,
  every topology, rising with load. A trace of `9483 g2 ×5 selfpredict` showed the clock frozen at 4,900 s with no
  memory refusal and no unplaced task. Under `kpa` every arm now runs the GNN-family autoscaler (starved-type eviction,
  rendezvous release), and the starved-task defer is shared (`src/placement/starved_defer.py`), so arms differ only in
  their scheduler. The traced cell then completes in 55 s. Legacy keeps the historical pairing (local replay
  identical). The `c9e5ed6c` outputs are kept as `*_c9e5` and are not read; every condition and the legacy replay
  rerun at `71d9cbcb`.

## Question
Does replacing the current scale-out rule (target 100, queued-only concurrency, reachability-driven
replica creation) with a Knative-faithful autoscaler change queue share, rankings or the gap of every
arm to CD, under each of the four transfer/release conditions?

## Why
Instrumented exact replay (topology 9483, g0, ×2, single origin) showed the load formula asked to scale
up in 0.07 % / 0.01 % of checks; replica counts tracked reachability, not load. The simulator serves
one task per replica (`containerConcurrency: 1`) but scales as if 100 were allowed. This is the third
modelling choice behind old-physics queue dominance and behind `unsaturated_scale_v1`'s
"capacity doesn't grow with servers".

## Factor definition (new flag; proposed name `HEROSIM_SCALEOUT=kpa`, default `legacy`)
| Element | legacy (current) | kpa (new) |
|---|---|---|
| Target concurrency per replica | 100 (int) | 0.7 (float; = 1 × 70 % utilisation) |
| Concurrency signal | queued tasks only | in flight = queued + in service (+ mid-transfer when replicas are released) |
| Averaging | 1 s samples, none | stable window 60 s; panic window 6 s |
| Panic mode | none | when panic-window demand ≥ 200 % of current capacity; no scale-down in panic |
| Scale to zero | 30 s keep-alive | last replica removed only after a full stable window with no traffic |
| Reachability-triggered creation | yes | **kept** (needed for a type's first replica on an unreachable client); logged separately |
| Memory caps | node.available_memory on eviction path | **verified on the scale-up path before registration of results**; replicas never exceed node memory |

Defaults are from the KPA defaults page (fetched in session). `_resolve_queue_length` must accept a float.
`HEROSIM_POLICY_TIME_SCALE` scales the windows exactly as it scales keep-alive; recorded in provenance.

## Design
- Factorial: transfer ∈ {store-and-forward, pipelined} × release ∈ {0, 1} × scale-out ∈ {legacy, kpa}.
  The four `legacy` cells are the existing `transfer_physics_v1` runs (no rerun); four `kpa` cells are new.
- Cells: 19 single-origin server-only topologies, loads ×2/×3/×5 (old ladder, kept only for continuity;
  `load_recalibration_v1` replaces it), seeds as in `transfer_physics_v1`.
- Arms: CD, locality-first, one-pass greedy, self-predict, Knative, `so1load` (zero-shot, context only).
- Replay: `legacy` path with the new code reproduces `transfer_physics_v1` field-for-field before any `kpa` read.
- Instrumentation on every `kpa` run: scale-up decisions by cause (load vs reachability), replicas per type
  over time, cold starts, panic-mode entries, memory-cap refusals.

## Primary family (Holm across 5 rule/Knative arms × 3 loads × 4 transfer/release cells = 60 tests)
Each arm vs CD, paired, under `kpa`.

## Predictions (written before any `kpa` run)
1. Load-driven scale-ups become the majority cause of replica creation (> 50 % of creations) in every cell.
2. Under store-and-forward + held replicas, queue share falls from ~0.8 to < 0.5 at ×2; latency falls ≥ 40 %.
3. Under pipelined + released, queue share falls further (< 0.15) but cold starts per task rise ≥ 2×.
4. CD remains first in every cell; rule order CD > {locality-first, greedy} > self-predict holds.
5. Knative's gap to CD shrinks in every cell versus `legacy` (it was throttled by the target).
6. Panic mode fires mainly at group bursts (inter-arrival CV 2.7); oscillation (scale-up then scale-down
   within one stable window) occurs in < 10 % of replica lifetimes.
7. `so1load` (zero-shot) stays behind CD by ≥ 10 %.

## Outcomes and what follows
- If predictions 1–2 hold: `kpa` becomes part of candidate reference physics R1 for `physics_audit_v1`.
- If load-driven scale-up still rarely fires: the signal or windows are wrong — fix and amend, do not proceed.
- If oscillation > 10 %: report; consider KPA scale-down delay before freezing R1 (amendment).
- If CD loses its lead anywhere: that is a finding; report it, do not tune the physics to restore it.

## Cost
4 new conditions × (arms × cells × seeds) as in `transfer_physics_v1` (~1,600 runs each), plus replay.
