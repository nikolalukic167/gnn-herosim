# physics_audit_v1 — scrutinise and freeze reference physics R1

**Status:** `ACTIVE` (2026-10-08) — audit pass 1 read; R1 **not frozen** (I12 fails by construction, fixed by amendment B1; pass 2 runs at the fixed time scale). Registered 2026-10-08. Depends on: `kpa_scaleout_v1` (read). Plan: [`reference_physics_programme.md`](reference_physics_programme.md).
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


## Record (newest first)

### 2026-10-08 — audit pass 1 read; amendments B1–B4 before pass 2

Code: `rp/audit` merged at `2366ad31` (opt-in trace `HEROSIM_AUDIT_TRACE`, snapshot fidelity block
`HEROSIM_SNAPSHOT_FIDELITY=1`, checkers `scripts_cosim/physics_audit/check_invariants.py`, I11 harness
`scripts_cosim/physics_audit/i11_replay.py`, 29 tests in `tests/test_physics_audit.py`, each checker with a
seeded-defect negative control). Default path unchanged: 36/36 R1 cells and 6/6 legacy cells identical to
`99fac565` (wall clock masked); trace on, and trace + snapshots + fidelity on, identical to trace off; repo suite
26 failures before and after.

Pass 1 ran candidate R1 with the per-rung time scale the gates used (0.5 / 0.333 / 0.2 on ×2/×3/×5): 6 topologies
(9483, 9491, 9506, 9533, 9550, 9568) × 3 rungs × {CD, reactive}, 12,000 arrivals each, workload g0; I8 on 12 cells × 2.

| # | Pass 1 | Note |
|---|---|---|
| I1 | PASS 36/36 | counter vs task rows pointwise, 0 violations; L = λW worst gap 0.75 % |
| I2 | PASS 36/36 | 1,000 ingress transfers per run, max error 3e-11 |
| I3, I7, I9 | PASS 36/36 | |
| I4 | PASS 36/36 | 0 compute overlaps; ~3,200 overlapping I/O pairs per run |
| I5 | **FAIL** | load-caused share CD median 0.44 (8/18 runs > 0.5; 10 FAIL-WITH-CAUSE), reactive 0.77; Spearman vs in-flight (10 s bins) CD −0.09, reactive +0.12 |
| I6 | PASS 36/36 | peak 4.5 % of node memory, no refusal: **refusal path untested** |
| I8 | PASS 12/12 | another `PYTHONHASHSEED` also identical |
| I10 | PASS 36/36 | on the two instrumented scheduler paths only; others NOT-TESTED |
| I11 | **PASS** after the fix | 216 states (6 topologies × 3 rungs × 12, CD, first 3,000 arrivals): 0 failed, median 0.00 %, p95 0.08 %, max 0.98 %. The pre-fix replay on the same states: 93/216 failed, median 95.7 % |
| I12 | **FAIL** | time scale differs per rung by construction |

**I11, what was wrong.** The old snapshot had no KPA history or tick phase, replayed with target 100 against 0.7
live, dropped tasks held by released replicas, marked every replica ready and warm, compressed queued tasks into a
backlog, dropped partners outside the batch, and seeded free platforms as initialised. All nine are in the fidelity
block. Not independent draws: start-up states repeat across cells. CD only; 5 states with a busy link pipe; no state
with an unplaced partner.

**I11, a property of R1.** Under released replicas a placed batch's latency depends on later arrivals (a task holds
the compute lock, then waits in output for node storage an image pull holds). Against the continuing live run the
same replays miss > 1 % in 36/216 states, all in the first 27 s; from 300 s on all 104 states match it (max 0.59 %).

**I12's anomaly (CD faster at ×5).** Pass-1 cells: CD 0.678 / 0.649 / 0.650 s; cold starts per task 0.229 / 0.206 /
0.182, exchange 0.319 / 0.272 / 0.235 s, queueing flat. Consistent with the rung-scaled constants and rising warmth;
not settled (12,000 arrivals, 6 topologies).

**Amendments (coordinator, 2026-10-08, after pass 1 and before pass 2; B3 is post-data and labelled so).**
- **B1 — one time scale: `HEROSIM_POLICY_TIME_SCALE=1.0` on every rung.** Knative's keep-alive, windows and tick are
  wall-clock constants that do not change with load; scaling them with the rung tied policy constants to the load
  ladder. Value recorded for R1. Results measured at the per-rung scale (`kpa_scaleout_v1`) stand as measured and are
  not R1 numbers.
- **B2 — I11 reference.** The reference is the live run cut at the decision (no later arrivals), which is what "the
  same plan executed live from that state" and a co-sim label mean; the continuing-run error is a second column,
  reported, not scored. Labels for `r1_attribution_v1` come only from decisions at t ≥ 2× the latest continuing-run
  miss in pass 2, rounded up to the minute (pass 1 would give 60 s).
- **B3 — I5 (post-data).** The verdict stays as registered. The in-flight clause compares replicas with
  instantaneous concurrency; KPA, like Knative, scales on its stable-window average, against which pass 1 reads
  ρ +0.64 (CD) / +0.72 (reactive). I treat that clause as mis-specified for the modelled system, not as a physics
  bug, so I5 failing on it alone does not block the freeze; both forms are reported. The load-share clause keeps its
  registered fallback, scored on CD (the arm `replica_placement_v1` holds fixed): CD's median load-caused share over
  the pass-2 runs ≤ 0.50 → FAIL-WITH-CAUSE and `replica_placement_v1` closes NO-LEVER, as registered.
- **B4 — pass 2 scope.** Rerun I1–I12 and the 216-state I11 at B1; add a reduced-memory cell so I6's refusal path
  fires; produce the descriptive outputs and the concentration-vs-saturation check (both missing from pass 1); add
  P95/P99 latency and replica counts over time to gate summaries. Freeze if every check passes apart from I5's
  in-flight clause (B3).
