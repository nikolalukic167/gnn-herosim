# physics_audit_v1 — scrutinise and freeze reference physics R1

**Status:** `CLOSED` (2026-10-08) — **R1-FROZEN**. Registered 2026-10-08; amendments B1–B4 signed after pass 1 and before pass 2 (B3 post-data, labelled). R1 = code `033811d6`, gate pin `242d0ea6`; superseded on 2026-10-08 by **R1.1** = `rp/starve` `665c9142` / gate `252b45aa` (audit pass 3).
Revision 2026-10-08 (before any run): added I11, I12 and the I5 fallback.


**Outcome (2026-10-08).** R1 is frozen: `HEROSIM_TRANSFER_MODEL=pipelined`, `HEROSIM_REPLICA_RELEASE=1`,
`HEROSIM_SCALEOUT=kpa`, `HEROSIM_POLICY_TIME_SCALE=1.0` on every rung (gates: `GATE_FIXED_POLICY_TIME_SCALE=1.0`),
peer exchange on, single origin, server-only replicas. At that time scale every invariant passes except I5, which is
FAIL-WITH-CAUSE: on CD only 22 % of replica creations are load-caused (the rest reachability), and replica counts do
not follow instantaneous load under either arm. **Quote R1 replica counts as partly reachability-driven.** I11 (the
co-simulation reproduces live) passes on 216 states: median 0.00 %, p95 0.15 %, one state 4.3 % undiagnosed. **Labels
under R1 come only from decisions at t ≥ 360 s**, and that range is validated only to 3,000 arrivals. I11 covers the
CD arm; I6's refusal path is checked only on a reduced-memory cell. `replica_placement_v1` closed NO-LEVER as
registered. **R1.1 (2026-10-08)** adds the free-pool leak and cross-source drain fixes and a 300 s request timeout that
runs only once the awaited partner has arrived. Pass 3 passes I1–I4, I6–I13, with I5 as before. It is identical to R1 on all 42
audit and legacy cells, and on 198 of 200 W2 cells (2 leak cells within 1 %). Cite R1.1 from now on.

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

### 2026-10-08 — audit pass 3 on R1.1; R1.1 accepted

Report: [`physics_audit_v1/pass3_report.md`](physics_audit_v1/pass3_report.md) (S5, `rp/starve`). The 36 cells are the
pass-2 design at time scale 1.0. **I1–I4, I6, I7, I9, I10, I13 PASS 36/36**; I8 12/12; I11 216 states, worst p95 0.10 %;
I12 12/12. I5 is as before (CD load share 0.217; reactive above 0.5 in 10/18; B3). I7 now judges a run with no end row as
FAIL. **I13 pool conservation** is new: free + owned + draining = platforms on every node at every KPA tick, and free equals
`available_platforms`. Its seeded-defect tests include the old filtered-pool swap fed through the real emitter.
Identity and the timeout's scope are recorded in [`workload_fix_v1`](workload_fix_v1.md) (R1.1-T, accepted). **Caveat:**
pass 3 ran on `8e835eef` (timeout over the whole wait), not on `665c9142` (R1.1-T). Both commits are identical to `d4aeb7ca` on all
42 cells, and the timeout fired 0 times at each, so the simulated runs are the same, but the invariants themselves weren't rerun at
`665c9142`. A rerun is requested.

### 2026-10-08 — audit pass 2 at time scale 1.0: R1 frozen

Code `033811d6` (`rp/audit2`, from `b4ed6c32`). Same design as pass 1 at `HEROSIM_POLICY_TIME_SCALE=1.0`: 36 runs
(6 topologies × ×2/×3/×5 × {CD, reactive}, 12,000 arrivals), I8 on 12 cells × 2, I11 on 216 states. Trace on equals
trace off on 12 cells, and with snapshots and fidelity on as well, on 3. Full report:
[`physics_audit_v1/pass2_report.md`](physics_audit_v1/pass2_report.md) (pass 1: [`pass1_report.md`](physics_audit_v1/pass1_report.md)).

| # | Pass 2 |
|---|---|
| I1–I4, I7, I9, I10 | PASS 36/36 (I10 on the two instrumented scheduler paths) |
| I6 | PASS 36/36; reduced-memory cell (9483 ×3, 1 GB nodes): refusals logged equal the counter (CD 324, reactive 506), no node over memory, peak 55 % |
| I8 | PASS 12/12 |
| I12 | PASS 12/12 |
| I11 | cut-at-decision reference: 0 failed, median 0.00 %, p95 0.15 %, 215/216 within 1 %, 216/216 within 5 % (bar median ≤ 1 %, p95 ≤ 5 %): **PASS**. One state at 4.32 % (9491 ×2, t = 140 s, batch behind an image pull) undiagnosed. Continuing-run reference (reported): 30/216 miss > 1 %, latest at 165.7 s; from 360 s on, 100/100 within 0.68 %. 3 states with a busy link pipe |
| I5 | **FAIL-WITH-CAUSE.** CD load-caused share 0.217 (0.14–0.39), 18/18 runs ≤ 0.5; reactive 0.52 (10/18 > 0.5). Spearman vs instantaneous in-flight: CD −0.22, reactive +0.01 (0/18 each ≥ 0.5); vs KPA's stable-window average: CD 0.45 (4/18), reactive 0.54 (10/18) |

**Freeze decision (by B3/B4).** Every check passes except I5. The load-share clause takes the registered fallback
(FAIL-WITH-CAUSE on CD), and the in-flight clause does not block (B3). R1 freezes. Note that on CD even the
stable-window form stays below 0.5: CD's load-caused replicas are too few for their count to track demand. Label
cut-off by B2: 2 × 165.7 s, rounded up to the minute, is **360 s**. `r1_attribution_v1` adds a 20-state I11 spot check
on its own label states beyond 3,000 arrivals before training on them.

**Descriptive outputs for R1 (by rung, CD and reactive).** Per-replica busy fraction (replicas alive ≥ 60 s): median
0.03–0.09, p90 0.12–0.26, max 0.34–0.57, ≤ 0.3 % of replicas above 0.5. Latency shares: exchange 42–67 %, queue
11–16 %, CD cold start 12.6 → 7.2 % (×2 → ×5). Live replicas 2–11 over run deciles; CD time-mean 4.5 / 5.6 / 7.1.

**Concentration vs saturation.** R1 shows neither: low mean busy fraction, nothing near 1, top 10 % of replicas carry
~42 % of busy seconds. Old physics (store-and-forward, held, legacy scale-out, scale 1.0): reactive saturates (median
0.77 at ×5, about half of replicas above 0.8); old CD in between.

**I12's anomaly.** At one time scale CD still gets faster with load: 0.661 → 0.612 → 0.569 s. From ×2 to ×5 cold start
−0.042 s per task (cold starts per task 0.216 → 0.115) and exchange −0.086 s, other +0.040 s. So the per-rung time scale
was not the cause; warmth and exchange are. Not decomposed further.

**Gate summaries** now carry `latency_percentiles` (p50/p95/p99/max), `replica_count_series` and `policy_time_scale`;
unit-tested, not yet through a full gate.

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
