# load_recalibration_v1 — define load rungs on R1 + WF1 by measured queue share

**Status:** `CLOSED` (2026-10-09) — **RUNGS-FIXED**. Registered 2026-10-08; the bands and guards were signed before any
run, and every amendment below is dated before the data it governs. Depends on: `workload_fix_v1` (WF1 frozen). Plan:
[`reference_physics_programme.md`](reference_physics_programme.md).

**Outcome.** On R1.1 + WF1 (code `rp/recal` `54fddd0d`; calibration topologies 9601, 9602, 9607, 9608 × 2 windows):
- **Light = ×1.2584, moderate = ×5.9402, heavy = ×11.6139.** CD's effective queue shares are 0.080, 0.270 and 0.440,
  all inside their bands. Every CD guard passes at every rung, with 8/8 cells finished.
- **The measure is the effective queue share**: (queue + compute-lock wait) ÷ elapsed. Under `HEROSIM_REPLICA_RELEASE=1`,
  the FIFO backlog on a saturated replica is filed as compute time, so the plain queue share is blind to it. It reads
  0.0003 for Knative at heavy, where the effective share is 0.995.
- **Knative**, reported and not binding: healthy at light (0.092, p95 44 s). Collapsed at moderate (0.981, p95 6,230 s,
  one cell timed out) and at heavy (0.995, median latency 837 s, run end 2.8×). The collapse is lock wait (826 of 837 s),
  not pre-placement wait.
- **Predictions:**
  - Prediction 1 ("all multipliers well above ×5") is FALSIFIED: light is ×1.26.
  - Prediction 2 ("heavy unreachable or unstable") is FALSIFIED: heavy is reachable and stable for CD.
- **Not quotable without:** "effective" next to any share; the queue share of any earlier R1 read is understated
  wherever replicas saturate.
- **Old ladder on WF1:** ×2 ≈ 0.10 and ×3 ≈ 0.14 (between light and moderate); ×5 ≈ 0.23 (moderate). ×0.2666 gives
  0.046 (CD 7/8). Addenda complete.

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
    (843483) was cancelled after about 10 minutes, and 843490 before any evaluation (the in-system rule must steer the search, not only the read). Bisection job: 843494, code `0f6e7832` (only the bisect script and its test differ from `8286024c`).
  - **Bracket amendment (2026-10-09, after the first context evaluation and before any midpoint).** ×11.61 gives a CD
    median share of 0.104 (Knative 0.0003), so every rung lies at about ×10 or above. The ×0.05 end costs hours (simulated time
    grows as 1/m) without moving any rung. The lower end becomes the already-evaluated ×0.2666, and ×11.61 for
    moderate and heavy, whose bands sit above 0.104. The upper end stays ×64, and no midpoint below ×5 is evaluated
    unless the bracket requires it. Rung definitions, guards and the step cap are unchanged.
    Job 843586, `rp/recal` `64a545cd`, with linear midpoints and the cached ends read from disk. ×11.61: CD 8/8, share 0.1045, all guards pass.
    ×0.2666: CD 7/8 (the 8th, 9608 g1, hit the wall limit and its rerun was cancelled), share 0.0251. A seeded end places the
    bracket by its share alone and can never be the answer (accepted).
  - **Steering amendment (2026-10-09 04:05, before any midpoint result was read).** Each step was waiting up to about 3 h
    for 3× reruns: ×5.94 on one Knative cell, ×64 on 3 CD and 1 Knative cell. Changes:
    - the bisection steers on CD alone;
    - during steering, a CD cell that hits the 2,700 s first-pass limit counts as unfinished, so the step fails, with no
      rerun. At m ≥ 5 that limit is ≥ 5× a normal cell's wall time;
    - Knative and the 3× reruns run only at the final chosen rungs and the mapped context;
    - if a final rung fails a CD guard there, it moves to the highest passing step.

    Guards and bands are unchanged.
  - **Observation and decision (2026-10-09 06:15, mid-bisection, before any final rung).** ×24.71 fails: CD finishes
    7/8 (9602 g0 hits the 2,700 s limit), and run end ÷ last arrival is 1.39, 1.30 and 1.63 on three cells. CD's median
    queue share is **not monotone in m**: 0.104 at ×11.61, then < 0.04 at ×24.7–64. The added latency sits in
    compute time (done − started: 2.9 s → 42–71 s), while execution plus communications is only 1.2–3.8 s, and queue,
    placement wait and rendezvous stay ≈ 0. So above about ×12, queue share and placement wait can't measure load; p95 and
    run end ÷ last arrival are the guards that detect the collapse. Decision: the bisection continues unchanged. Moderate
    and heavy are expected to fall back to the highest stable step (protocol fallback) with their achieved share. The
    unattributed compute-time component goes to S5 for attribution (physics or accounting artifact). The heavy rung does
    not feed `r1_attribution_v1` until that is answered.
  - **Measure amendment (2026-10-09 06:40, before any final rung).**
    - **Attribution** (S5, job 843730: a rerun of 9601 g0 ×24.71 CD at `204c0aa1`, reproducing done − started
      41.980 s exactly). 40.734 s of it is the wait for the replica's `compute_lock` after the input stage.
    - **Why it was hidden.** Under `HEROSIM_REPLICA_RELEASE=1` the platform queue pops at once, and `started` is stamped
      before `compute_lock.request()` in `_serve_task`. So the FIFO backlog on a saturated replica is filed as compute
      time. It is a tail (p50 0 s, p95 80 s, max 1,704 s) on a few replicas. Load scale-ups fail 16,229 times in that
      cell.
    - **Physics or artifact.** It is real physics; queue_share misreads it (0.010 reported, about 0.97 with the wait
      counted).
    - **Decision.** The band measure becomes **effective queue share = (queue time + lock wait) / elapsed**, with lock
      wait = compute_start − io_end per task. The summary gains its mean, p95 and max, and keeps the old queue_share
      for continuity.
    - **Checks.** An identity check on 3 cells, plus S5's cell reproducing 40.734 s.
    - **Re-bisection.** Moderate and heavy re-bisect on the effective share, with a fresh step cap. Light is re-checked
      at ×5.94 and re-bisects only if it leaves the band. Guards, bands and CD-only steering are unchanged.
    - **Unblocked.** I11 runs at ×24.71 (9601 g0) before the heavy rung feeds node 5.
  - **I11 under backlog (2026-10-09, S5, job 843731, `204c0aa1` plus the rp/i11-wf1 drivers; results in
    `load_recalibration_v1/i11_m24/`).**
    - **Errors.** Uniform 40 states: median 0.000 %, p95 0.60 %, max 3.2 %. Backlog > 100 s, 39 states: median
      0.000 %, p95 4.2 %, max 11.4 %.
    - **The one exceedance** is a 0.27 s batch on an uncongested replica: an absolute miss of 0.03 s. The 13 batches
      that themselves wait 100–1,838 s behind the backlog reproduce within 0.34 %.
    - **Mechanism.** The snapshot carries the compute-lock waiters as `lock_wait` ghosts (`snapshot_fidelity.py:108`),
      and the replay re-requests the lock in the live order.
    - **Verdict: PASS-WITH-CAUSE.**
    - **Caveats.** One cell, captured to t ≤ 1,700 s of 6,093 s. The registered re-check of 20 *late* states at the
      final heavy rung still runs; it covers the late-run backlog this capture did not reach.
  - **Effective shares and rung choice (2026-10-09 07:25, before the final stage).**
    - **Runs.** Code `aa4b10c6`. Identity 843733: 3 cells, 0 differences. Measurement 843734: CD only, the 7 cached points.
    - **Effective CD median share, monotone in m:** 0.046 (×0.2666), 0.270 (×5.94), 0.440 (×11.61), 0.602 (×18.16,
      7/8), 0.877 (×24.71), 0.969 (×37.8), 0.978 (×64).
    - **Heavy = ×11.61, provisional.** It is inside the band on the evaluated point.
    - **Moderate = ×5.94, provisional.** It is inside the band on the evaluated point.
    - **Light** re-bisects between ×0.2666 and ×5.94, on log midpoints, with a fresh cap.
    - **Guard amendment.** The heavy stability backlog becomes placement_wait + queue + lock wait.
    - **Prediction 1** ("all multipliers well above ×5") is falsified for light.
    - **Job 843742** (`rp/recal` `54fddd0d`; identity check 843741) runs the light search and then the finals.
      - The finals write the amended backlog to `backlog_profile_v2`.
      - The earlier ×11.61 backlog ratio of 1.13 used the old definition, so it does not count.
      - Lock wait on S5's cell is 40.733508 s, which matches S5's figure (reported to 3 decimals).
      - **Light = ×1.2584, provisional:** CD effective share 0.0797, in band on the first log midpoint.
      - **Identity check 843741 FAILED:** the comparator's new-key whitelist did not list `backlog_profile_v2`, so it
        failed on keys, not on values. A rerun is required, and the finals are not accepted until identity passes.
      - **Identity passed** after the comparator fix (`293fd8c4`): 0 differences in the other fields on 3 cells.
      - **Finals, light ×1.2584:** CD 8/8, share 0.0797, p95 28 s, run end 1.0006×, 0 failures, backlog ratio 1.08,
        busy fraction 0.023. All CD guards pass. Knative share 0.092.
      - **Finals, heavy ×11.61:** CD 8/8, share 0.440, p95 15.5 s, run end 1.008×, 0 failures, amended backlog ratio
        0.985. All CD guards pass. Knative share 0.995.
      - The in-system ratio is not applied at either rung, because in-system(1/2) < 20.
      - **Decision (08:50):** heavy feeds node 5 now (B2 dry run, heavy portion; I11 late-state re-check). Moderate
        waits for its finals.
      - **Decision (09:10):** the ×0.2666 mapped context is supplementary. The rungs close on the three finals, and
        ×0.2666 is added later as an addendum.

## 2026-10-09 — Finals and close (RUNGS-FIXED)
Job 843742, code `54fddd0d` (identity comparator fix `293fd8c4`; identity checked on 3 cells with 0 differences). Fresh
directories; CD and Knative on 8 cells per rung; 3× reruns on timeouts. Full numbers in datalab
`simulation_data/load_recalibration_v1/finals_report.json`.

| Rung | ×m | CD eff. share | CD worst p95 | Run end ÷ last arrival | Backlog ratio | Busy (median) | Knative eff. share / median latency |
|---|---|---|---|---|---|---|---|
| light | 1.2584 | 0.080 | 28.3 s | 1.001 | 1.08 | 0.023 | 0.092 / 5.5 s |
| moderate | 5.9402 | 0.270 | 12.7 s | 1.004 | 1.13 | 0.044 | 0.981 / 231 s (7/8; 9602 g1 timed out at 3×) |
| heavy | 11.6139 | 0.440 | 15.5 s | 1.008 | 0.98 | 0.064 | 0.995 / 837 s (run end 2.8×) |

- CD has 0 request failures at every rung.
- The in-system ratio is not applied: in-system(1/2) < 20 at every rung (it reads ∞, ∞ and 4.5).
- No rung needed the fallback.
- Light was found on its first log midpoint. Moderate and heavy came from measurement pass 843734 and were confirmed
  in the finals.
- Knative's wait is lock wait: 826 of 837 s at heavy, and 227 of 231 s at moderate. Placement wait is 0.006 s.

## 2026-10-09 — Addendum: old ladder on WF1
Job 843900, code `5d52cd49`. CD only, the same 8 cells, 2,700 s steering limit, no rerun. Saved in
`simulation_data/load_recalibration_v1/old_ladder_read.json`.

| ×m | CD cells | CD eff. share | Old queue share | Worst p95 | Run end ÷ last arrival | Backlog ratio | Where it sits |
|---|---|---|---|---|---|---|---|
| 2 | 8/8 | 0.104 | 0.046 | 23.0 s | 1.001 | 1.05 | just above light |
| 3 | 8/8 | 0.145 | 0.057 | 18.3 s | 1.001 | 1.07 | between light and moderate |
| 5 | 7/8 | 0.234 | 0.070 | 13.5 s | 1.002 | 1.12 | moderate |

- At ×5, the 9607 g1 cell is labelled **stalled, not a guard result**: a 43 min busy loop against 44–54 s for its
  siblings, with an empty log. It is the spin under diagnosis in `r1_attribution_v1`. The cell finishes at ×5.94.
- The old ×2/×3/×5 ladder is therefore light-to-moderate on WF1. No old rung reaches heavy.
- **×0.2666 at full treatment** (job 843742, `54fddd0d`):
  - CD 7/8, effective share 0.046, just below light. Knative 8/8 at 0.065.
  - Every CD guard passes apart from finished.
  - CD 9608 g1 timed out at 1× and at 3× (8,100 s) and ends in request-timeout starvation. It is unfinished, not a measured
    collapse; a likely case of the R1.1-T hole, rechecked under the fix by job 849879.
  - The point is supplementary. The light rung stays ×1.2584.

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
Rungs fixed and committed; no policy comparison is read in this node. See the head.

## Cost
≈ 3 rungs × 8 steps × 8 calibration cells × CD (+ Knative for the stability guard): under 400 runs.
