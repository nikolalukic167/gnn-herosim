# kpa_scaleout_v1 — Knative-faithful scale-out as the third physics factor

**Status:** `CLOSED` (2026-10-08) — **SELFPREDICT-BEATS-CD-UNDER-RELEASE**. Registered 2026-10-07; amendments A1–A4 were
made before any `kpa` cell was read. Depends on: `transfer_physics_v1` (read, `e76862a2`).
Plan: [`reference_physics_programme.md`](reference_physics_programme.md) (shared rules apply).

**Outcome (read 2026-10-08, code `71d9cbcb`, 4 × 1,596 runs, 0 failed).**
- **Under KPA scale-out with released replicas, the self-predict rule beats CD**: −6.5 to −9.5 %, faster on 19 / 19
  topologies, Holm-confirmed at all 3 rungs of both `release` and `pipe_release` (6 of the 60 tests). It is the first
  cell in this programme where CD is not first. Mechanism: per task, self-predict now exchanges as little as CD
  (1.06 vs 1.11 s at ×2, `release`) and pays none of CD's 0.11 s peer-group batching wait.
- **Elsewhere CD stays first**: 50 tests CD-FASTER, 4 NOT-SEPARATED (self-predict at `sf_held` ×5 and `pipe` ×3 / ×5;
  zero-shot `so1load` at `sf_held` ×5). Locality-first and the one-pass greedy trail CD by 3.0–11.9 % everywhere.
- **KPA, not the autoscaler swap, causes the shift.** A control with legacy scale-out and every arm on the shared
  autoscaler moves reactive and self-predict by ≤ 1.2 % and leaves `transfer_physics_v1`'s "CD first everywhere"
  intact; KPA itself speeds self-predict up by 15–96 %, more than CD.
- **Scale-out still behaves partly like reachability, and churns.** Load-caused creations for CD are 33–75 % (below
  half in `pipe` ×2 and `pipe_release` ×2 / ×3); 26–61 % of replicas die within one stable window.
- **Predictions:** 1 FAIL, 2 PARTIAL (latency −64 %, queue share 0.56 not < 0.5), 3 FAIL on queue share (cold starts
  not in the gate summaries, unscored), 4 FAIL (CD not first; self-predict now above locality and greedy), 5 holds in
  11 / 12 cells (Knative's gap grows at `release` ×5), 6 FAIL (oscillation 26–61 %, not < 10 %), 7 holds in `pipe` and
  `pipe_release`, fails in `sf_held` and `release`. Excluding 9485 changes no label.
- **Decision:** KPA enters candidate R1 for `physics_audit_v1` (the legacy target of 100 per replica is known to be
  unfaithful); I5 there decides how the reachability share is reported, and the churn is a stated property, not tuned
  away (Knative's `scale-down-delay` default is 0). For learned arms under R1 the bar is CD **and** self-predict.


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

## Record

### 2026-10-08 — read (code `71d9cbcb`; control `38cd802f`)
- **Replay.** Legacy scale-out at `71d9cbcb` reproduces `transfer_physics_v1` field for field: 23 / 23, 1,589 / 1,589,
  1,592 / 1,592, 1,591 / 1,591 (only the runs that failed in the original gate are absent;
  [`legacy_replay_*_check.json`](kpa_scaleout_v1/)). Checker: `scripts_cosim/kpa_legacy_replay_check.py`.
- **First launch discarded unread.** At `c9e5ed6c` the two arms on the `knative_network` autoscaler froze the clock
  (A4). Outputs kept on datalab as `kpa_scaleout_v1/*_c9e5`.
- **Read.** `scripts_cosim/kpa_scaleout_v1_read.py` → [`kpa_read.json`](kpa_scaleout_v1/kpa_read.json), sensitivity
  without 9485 → [`kpa_read_no9485.json`](kpa_scaleout_v1/kpa_read_no9485.json). Family: reactive, self-predict,
  locality, batched and zero-shot `so1load` vs CD × 3 rungs × 4 conditions = 60, Holm (the registered "5 rule/Knative
  arms" counts the zero-shot arm as the fifth; it is labelled zero-shot and makes no claim).

| CD latency (s), queue share | ×2 | ×3 | ×5 | self-predict vs CD ×2 / ×3 / ×5 |
|---|---|---|---|---|
| `sf_held` | 3.06, 0.56 | 2.70, 0.58 | 2.29, 0.61 | +4.5 / +3.5 / +1.0 % (n.s.) |
| `pipe` | 1.69, 0.57 | 1.37, 0.58 | 1.12, 0.60 | +1.3 / +0.8 (n.s.) / −0.9 % (n.s.) |
| `release` | 1.75, 0.09 | 1.52, 0.11 | 1.31, 0.13 | **−7.4 / −7.5 / −6.5 %**, 19 / 19 |
| `pipe_release` | 0.99, 0.17 | 0.85, 0.20 | 0.73, 0.24 | **−8.8 / −9.2 / −9.5 %**, 19 / 19 |

Legacy CD latency for comparison: `sf_held` 8.62 / 13.32 / 56.99 s, `pipe_release` 0.97 / 0.86 / 0.78 s. Latency still
falls from ×2 to ×5 under every condition, so the old ladder does not order load (`load_recalibration_v1`).
- **Control.** `scripts_cosim/kpa_shared_control_read.py` → [`shared_control_read.json`](kpa_scaleout_v1/shared_control_read.json):
  `HEROSIM_SHARED_AUTOSCALER=1` under legacy scale-out, reactive and self-predict only (4 × 456 runs, 0 failed).
- **Not measured.** Cold starts per task, P95 / P99 and replica counts over time are not in the gate summaries; the
  gate summary should carry them before the audit's runs.

