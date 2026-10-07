# physics_audit_v1 — scrutinise and freeze reference physics R1

**Status:** `REGISTERED` (no runs). Depends on: `kpa_scaleout_v1` (read). Plan: [`reference_physics_programme.md`](reference_physics_programme.md).
Revision 2026-10-08 (before any run): added I11, I12 and the I5 fallback.

## Question
Does the candidate reference physics behave like the system it claims to model, checked by invariants that do
not depend on which policy wins? Only physics that passes is frozen as **R1** and used by every later node.

## Candidate R1 (declared now)
`HEROSIM_TRANSFER_MODEL=pipelined`, `HEROSIM_REPLICA_RELEASE=1`, `HEROSIM_SCALEOUT=kpa` (proposed name),
peer exchange on, single origin, current workload (unchanged until `workload_fix_v1`), `bandwidth_mbps` documented
as MB/s, one policy time scale for every rung (value recorded here at freeze). Any deviation needs an amendment with
a reason that is not "a policy did better".

## Invariant checks (each pass/fail; thresholds fixed now)
| # | Check | Pass condition |
|---|---|---|
| I1 | Little's law per platform: L = λ·W, with "in system" = arrived and not yet completed (queued, in rendezvous, in transfer or executing) | within 5 % on every platform with ≥ 200 tasks |
| I2 | Transfer time vs analytic | simulated = size ÷ bottleneck + route latency, within 1 %, sampled 1,000 transfers |
| I3 | Store-and-forward off everywhere | no transfer charged hops × size anywhere (ingress, peer, parent→child) |
| I4 | Released replicas | compute never overlaps on one replica; I/O overlap observed; sandbox warmth preserved |
| I5 | Scale-out causality | load-caused creations > 50 % of all creations **and** Spearman ≥ 0.5 between load-caused replica count and in-flight concurrency over time |
| I6 | Memory | no replica ever exceeds node memory; refusals logged |
| I7 | Conservation | every arrived task completes or is logged as failed; no silent drops |
| I8 | Determinism | same seed → identical summary, two runs, 12 cells |
| I9 | Cold-start accounting | cold starts counted once per sandbox creation; warm starts never charged |
| I10 | Decision time | recorded outside simulated time for every arm |
| I11 | Co-simulation fidelity | for 200 captured states (spread over topologies and loads), the co-simulated latency of the chosen plan matches the same plan executed live from that state within 1 % (median) and 5 % (p95). Snapshot must capture released replicas mid-transfer, KPA window history, and eagerly created replicas. |
| I12 | Load-rung configuration | policy time scale, keep-alive and reconcile interval identical across all load rungs; cold starts per task logged per rung |

**I5 fallback (pre-stated).** If load-caused creations are ≤ 50 % because reachability-triggered creation dominates,
I5 is recorded as FAIL-WITH-CAUSE, not as a physics bug. R1 may still freeze, but the paper must state that replica
counts are partly reachability-driven, and `replica_placement_v1` is closed as NO-LEVER.

**I11 is blocking** for `r1_attribution_v1`. If it fails, labels cannot be generated under R1; fix the snapshot and
replay, then re-run I11.

**I12 also explains a known anomaly.** Under the new physics, CD's latency fell from ×2 to ×5 in `transfer_physics_v1`.
Report whether that came from rung-dependent time scale, rising warmth (fewer cold starts at higher arrival rates),
or something else.

## Descriptive outputs (no tests, reported once for R1)
Per-replica busy fraction distribution (median, p90, max) per rung; utilisation per platform type; queue, exchange,
execution and cold-start share of latency; replica counts over time.

## Concentration vs saturation check
On R1 and on old physics, compare per-replica busy fraction: concentration predicts a few replicas near 1 with a low
mean; saturation predicts most replicas high. Report both; no bar.

## Outcomes
- All of I1–I12 pass (or I5 fallback) → freeze R1 (commit hash recorded here); later nodes cite it.
- Any other fail → fix, replay, rerun the audit; R1 is not frozen until a full pass.
- No policy comparison is read in this node.

## Cost
Instrumented runs on 6 topologies × 3 rungs × CD and Knative, plus I8 duplicates and 200 I11 replays.
