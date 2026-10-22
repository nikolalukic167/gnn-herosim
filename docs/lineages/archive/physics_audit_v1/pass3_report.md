# physics_audit_v1 pass 3 on R1.1 (S5, `rp/starve`)

Code under audit: `8e835eef` (R1.1: free-pool leak fix, cross-source drain release, batch-target reservation, rate-limited starved log,
fail-loud `StarvedForeverError`, 300 s request timeout whole-wait scope, starved-path counters, I13, I7 FAIL on unfinished run).
R1.1-T (`665c9142`) changes only *when* the timeout may fire; pass 3 was not rerun on it (no new runs requested). The timeout fired
0 times in all 42 identity cells on `665c9142`, and pass-3 cells have the same shape (12,000 arrivals, ×2/×3/×5), so I do not expect
a change, but that is an inference, not a measurement.

Setup: 6 topologies (9483, 9491, 9506, 9533, 9550, 9568) × 3 rungs (x20/x30/x50) × {CD, reactive}, workload g0, 12,000 arrivals,
`HEROSIM_POLICY_TIME_SCALE=1.0` on every rung (B1), R1 flags, traces on. 36 cells.

## Invariants

| # | Result | Note |
|---|---|---|
| I1 | PASS 36/36 | |
| I2 | PASS 36/36 | |
| I3 | PASS 36/36 | |
| I4 | PASS 36/36 | |
| I5 | FAIL 10 (all reactive, load-caused share > 0.5), FAIL-WITH-CAUSE 26 | CD median load share 0.217 (min 0.139, max 0.389, > 0.5 in 0/18); reactive 0.520 (0.359–0.770, > 0.5 in 10/18). rho vs in-flight (instantaneous): CD −0.22, reactive +0.01; vs KPA stable window: CD +0.45, reactive +0.54. Same verdict class as passes 1–2 (B3 treats the in-flight clause as mis-specified). |
| I6 | PASS 36/36; reduced-memory cell PASS | 9483 x30, 1 GB nodes, 12,000 arrivals: refusals logged = counter (CD 324, reactive 506), 0 violations, peak 55 % — identical to pass 2. |
| I7 | PASS 36/36 | Now FAIL (not NOT-TESTED) when the trace has no end row; unit-tested. |
| I8 | PASS 12/12 | Two same-seed runs of 12 cells (CD, x20/x50 × 6 topologies). A third run with `PYTHONHASHSEED=1` differs only in the recorded seed value (provenance field), nothing simulated. |
| I9 | PASS 36/36 | |
| I10 | PASS 36/36 | Instrumented scheduler paths only, as before. |
| I11 | PASS | 216 states (18 cells × 12, CD, first 3,000 arrivals): 0 failed, worst per-cell median error 0.00 %, worst per-cell p95 0.10 %. |
| I12 | PASS 12/12 | Time scale 1.0 on every rung. Cold starts per task fall with load (e.g. 9483 CD 0.114 / 0.093 / 0.065 at x20 / x30 / x50). |
| **I13** | **PASS 36/36** (+ both reduced-memory cells) | New. free + owned + draining = platforms on every node at every KPA tick, and free = the node's `available_platforms` counter. |

## I13 seeded-defect tests (`tests/test_physics_audit.py`)
- A platform that is neither free, owned nor draining from t ≥ 120 (the pool leak): FAIL.
- The node's `available_platforms` counter disagreeing with the free set: FAIL.
- A platform counted twice (draining and owned): FAIL.
- No pool rows in the trace: NOT-TESTED.
- Through the real emitter: `AuditRecorder.pool` fed the state the old filtered-pool swap left behind (the node's free set missing from
  `available_resources`, its platforms owned by nobody) gives FAIL with free = 0; the same state intact gives PASS.
- I7 on a trace with no end row: FAIL.
The checker caught the defect only because the emitter records raw per-node numbers; the checker holds the node platform counts from
the header.

## Identity, R1.1 vs `d4aeb7ca` (src byte-identical to `690ba384`)

Wall-clock fields masked and set-ordered lists sorted; everything else compared.

| Check | whole-wait timeout (`8e835eef`) | partner-arrived scope, R1.1-T (`665c9142`) |
|---|---|---|
| 36 audit R1 cells + 6 legacy cells (local; legacy = physics-old, 9483/9506/9568 x20, CD + reactive) | 42/42 identical, timeout fired 0 | 42/42 identical, timeout fired 0 |
| W2 datalab sample, 200 cells (5 topologies 9483/9568/9484/9550/9538 × 5 arms × 2 rungs × 4 windows) | 123 identical, 77 differ | **198 identical, 2 differ** |
| cause of the differing cells | 75 cells: the timeout fires at ×0.2666 g1–g3 (136 / 105 / 113 failed tasks per cell, same for every arm and topology; mean latency −45…−49 %). 2 cells: the leak. | 2 cells: the leak (9550 g2 ×11.61, batched −1.00 % avg elapsed 0.6363 → 0.6299 s; CD +0.40 % 0.6262 → 0.6287 s; 0 failed requests) |
| W4 9565 g2 CD | completes, 50,000 tasks, 313 failed requests (earlier variant with the sticky starved flag) / 5 on the final whole-wait code | completes, 50,000 tasks, end 9412 s, 5 failed requests; with the timeout disabled the same cell stalls at t = 453 (2.8 M deferrals, 0 successful evictions) |

Leak attribution (old code instrumented): one `_release_replica` per cell landed while `create_first_replica`'s filtered-pool swap was active,
with its node missing from the filtered pool, in 9550 g2 ×11.61 CD and batched; 0 in every identical cell. Restoring only the old swap in the
new tree makes both cells bit-identical to the old code. Overlapping `create_first_replica` calls: 0 on all 42 cells; cross-source
releases: 0 on all 42 cells.

Process lesson recorded: a timeout implemented as a SimPy event changed 29 of 42 identity cells even though it never fired (the autoscaler
tick's `env.step()` consumes the next queued event). The deadline is evaluated in the tick, never scheduled. A "sticky starved" retry
optimisation changed event order in completed runs and was removed.

Not covered: pass 3 on `665c9142`; the other 35 stuck W4 cells; children of a failed task in multi-task DAGs.
