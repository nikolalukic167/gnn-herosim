# load_recalibration_v1 — define load rungs on R1 + WF1 by measured queue share

**Status:** `REGISTERED` (no runs). Depends on: `workload_fix_v1` (WF1 frozen). Plan: [`reference_physics_programme.md`](reference_physics_programme.md).
Revision 2026-10-08 (before any run): calibration now runs on the fixed workload WF1, not today's payloads.

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
