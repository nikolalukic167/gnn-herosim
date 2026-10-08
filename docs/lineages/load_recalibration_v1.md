# load_recalibration_v1 — define load rungs on R1 + WF1 by measured queue share

**Status:** `REGISTERED` (no runs). Depends on: `workload_fix_v1` (WF1 frozen). Plan: [`reference_physics_programme.md`](reference_physics_programme.md).
Revision 2026-10-08 (before any run): calibration now runs on the fixed workload WF1, not today's payloads.

**Pre-run amendments (2026-10-08, coordinator, before any run; from the `workload_fix_v1` close).**
- **Calibration set: 9601, 9602, 9607, 9608** (9603–9606 are infeasible on the live topology; `workload_fix_v1`). This
  replaces "9601–9604" below.
- **Code and workload:** R1.1 + the reservation fix (`rp/starve-w4fix` `97269192`), WF1 (W2 + W3 + W4), window 1 s.
- **Guard additions, every rung, every arm run for the guard (CD and Knative):** request failures ≤ 1 % of tasks, **p95
  per-task latency ≤ 300 s**, and the run ends within 1.25× the last arrival time. Under WF1 at ×11.61, Knative's queue
  share was 0.001 while its p95 was 5,366 s, so queue share alone can't see a pre-execution collapse. A rung where
  Knative fails a guard is still allowed (Knative is context), but the failure is reported with the rung. A rung where CD
  fails one isn't allowed.
- **Instrumentation first:** add an arrival-to-placement time per task (mean, p95, max) to the gate summary, behind an
  identity check on 3 cells. Without it, the guard can't attribute a pre-execution wait.
- **Guard definitions (2026-10-08, before the bisection's data; code `rp/recal` `8286024c`, identity 4/4 cells).**
  - **Busy fraction:** total execution time ÷ (time-mean replicas × end time).
  - **Heavy stability:** per-task backlog = arrival-to-placement wait + queue time, by arrival quarter. The worst cell's
    last-quarter mean must be ≤ 2× the mean of the middle two quarters, **and** tasks in the system at 3/4 of the last
    arrival ÷ tasks in the system at 1/2 must be ≤ 2. The ratio binds only when at least 20 tasks are in the system at 1/2;
    below that it's a ratio of small integers, so it's reported, not applied.
  - The backlog guard replaces a queue-time-only version that would have missed the unplaced wait. The job that used it
    (843483) was cancelled after about 10 minutes. Bisection job: 843490.

## Question
Which arrival-rate multipliers on R1 + WF1 produce light, moderate and heavy load, defined by a policy-independent
measure rather than the old ×2/×3/×5 ladder?

## Definition (fixed now)
| Rung | CD median queue share target | Guard |
|---|---|---|
| light | 0.05–0.10 | mean per-replica busy fraction < 0.3 |
| moderate | 0.20–0.30 | — |
| heavy | 0.40–0.50 | stability: end-of-run backlog ≤ 2× mid-run backlog for every arm |

CD is the reference because it is the strongest fixed policy; once fixed, a rung is a property of the environment.

## Calibration protocol
- 4 calibration topologies **not** among the 19 test topologies (same generator, different seeds), 2 windows each.
  **Named 2026-10-08 (before any run): ids 9601–9604**, minted with the 9473–9568 generator and template; see `workload_fix_v1`.
  The 19 test topologies are never used for calibration.
- Bisection on the arrival multiplier (inter-arrival gaps scaled; burst shape and sibling offsets preserved), max 8
  steps per rung. Policy time scale stays at the R1 value for every rung (invariant I12).
- If the heavy band is unreachable or unstable: heavy = the highest stable multiplier, recorded with its achieved
  queue share. This is reported as a property of R1, not worked around.

## Also recorded
The old ladder (×2/×3/×5) and the provisional rungs of `workload_fix_v1`, mapped onto the new rungs.

## Predictions
1. All three multipliers are well above the old ×5.
2. Heavy load is unreachable or unstable under released replicas; if so, R1 is predominantly a light-to-moderate
   regime, and the paper says so (learned-scheduler gains in the literature concentrate at high load, e.g. Decima).

## Outcomes
Rungs fixed and committed; no policy comparison is read in this node.

## Cost
≈ 3 rungs × 8 steps × 8 calibration cells × CD (+ Knative for the stability guard): under 400 runs.
