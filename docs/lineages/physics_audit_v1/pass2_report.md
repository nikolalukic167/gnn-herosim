# physics_audit_v1 pass 2 (branch rp/audit2, from reference-physics b4ed6c32)

Setup: candidate R1 with `HEROSIM_POLICY_TIME_SCALE=1.0` on every rung (B1). 6 topologies (9483 9491 9506 9533 9550 9568) x x2/x3/x5 x {CD, reactive}, 12,000 arrivals, workload g0.
Scratch: `p2/audit` (36 traced runs), `p2/det` (I8), `p2/m/<cell>` (I11), `p2/mem` (I6 cell), `p2/old` (old physics), `p2/checks.json`, `p2/desc_r1.json`, `p2/desc_old.json`.

## 1. I1-I10, I12
I1 I2 I3 I4 I6 I7 I9 I10: PASS 36/36. I12: PASS 12/12 (6 topologies x 2 arms; scale 1.0, keep-alive and reconcile interval identical across rungs).
I8: PASS, 12 cells x 2 runs identical (wall-clock fields and the code-dirty marker masked). A third run at PYTHONHASHSEED=1 differs only in the recorded env var.
I5: see 3.
Trace on = trace off: 12 cells (trace) and 3 cells (trace + snapshots + fidelity), result JSONs identical (wall clock and code-dirty marker masked).

## 2. I11 at 1.0 (216 states: 6 topologies x 3 rungs x 12; 6 uniform + 6 targeted per cell; CD, first 3,000 arrivals)
Scored (cut at the decision): 0 failed; median 0.00 %, p95 0.15 %, max 4.32 %; 215/216 within 1 %, 216/216 within 5 %. By rung: x2 max 4.32 % (p95 0.08 %), x3 max 0.55 %, x5 max 0.68 %.
The one state over 1 %: 9491 x2, t = 140.1 s, batch 118-127, an uninit_replica state (a pull with 10.37 s remaining; 5 of the 10 tasks wait on it). Replay is +0.98 s on the first task there and +0.38 s on the other four; the 5 tasks not on that platform are exact. Cause not identified; the pull-remaining value matches the live first-task latency (10.366 s) to 4 digits, so the extra is after the pull.
Busy link pipe: only 3 states this time (5 at pass 1); open peers 0 in all 216.
Reported, not scored (continuing-run reference): median 0.00 %, p95 54.0 %, max 71.1 %, 30 of 216 miss by > 1 %. Latest miss at t = 165.7 s (9483 x3, -60.8 % against the continuing run, -0.18 % against the cut run); next latest 140.1 s (the 4.32 % state above). 2 x 165.7 = 331.4 s -> label cut-off 360 s (B2). From 360 s on: 100 states, continuing-run max error 0.68 %, cut-run max 0.68 %.
Caveat: the continuing run ends at 3,000 arrivals (t up to 2,532 s), so "no miss after 165.7 s" covers states up to that horizon only.

## 3. I5
Verdict as registered (load share > 0.5 and rho >= 0.5 vs instantaneous in-flight):
- CD: 18/18 FAIL-WITH-CAUSE. Load-caused share median 0.217 (range 0.14-0.39); 0/18 above 0.5.
- reactive: share median 0.520 (0.36-0.77), > 0.5 in 10/18; those 10 FAIL on the in-flight clause, the other 8 FAIL-WITH-CAUSE.
Both correlation forms (Spearman over 10 s bins of the load-caused replica count against ...):
| arm | instantaneous in-flight (median, range, runs >= 0.5) | KPA 60 s stable-window average (median, range, runs >= 0.5) |
|---|---|---|
| CD | -0.22 (-0.29 .. 0.04), 0/18 | 0.45 (0.33 .. 0.64), 4/18 |
| reactive | 0.01 (-0.09 .. 0.36), 0/18 | 0.54 (0.33 .. 0.64), 10/18 |
Under B3's registered fallback, CD median share 0.217 <= 0.50 -> FAIL-WITH-CAUSE and replica_placement_v1 closes NO-LEVER. Note the stable-window form does not clear 0.5 for CD either (median 0.45).

## 4. I6 refusal cell
Cell 9483 x3, node memory capped at 1.0 GB (from 8 / 32 GB; `run_reduced_memory.py` wrapper, not a flag), 12,000 arrivals.
CD: 324 refusals logged = the result's memory_cap_refusals counter (324); 4 scale-ups blocked by memory; 0 nodes over memory; 0 negative free memory; peak 55 % of node memory. Reactive: 506 logged = 506 counter; 28 blocked; same zeros.
Refusals count candidate (node, platform) pairs per scale-up step, not failed creations.

## 5. Descriptive outputs (median over 6 topologies)
Definitions in `descriptives.py`. Busy = fraction of a replica's life (rep_up to rep_down) with a task past its pop and not done ("occupied"); "compute" is the compute lock only (exec is ~5 % of latency, so it is small everywhere).
### R1, all replica lives and lives >= 60 s
| rung/arm | lives (>=60 s) | occupied median / p90 / max, all | occupied median / p90 / max, >= 60 s | >= 60 s: share >0.5, top-10 % share of busy seconds | alive mean / peak |
|---|---|---|---|---|---|
| x2 CD | 2720 (539) | .215 / .791 / .999 | .031 / .119 / .418 | 0.000, 0.42 | 4.5 / 14 |
| x2 reactive | 1368 (622) | .109 / .464 / .992 | .062 / .167 / .570 | 0.003, 0.42 | 4.6 / 11 |
| x3 CD | 2070 (488) | .212 / .785 / .999 | .037 / .133 / .342 | 0.000, 0.41 | 5.6 / 14 |
| x3 reactive | 1096 (530) | .112 / .477 / .988 | .073 / .212 / .412 | 0.000, 0.42 | 5.7 / 11 |
| x5 CD | 1446 (406) | .203 / .778 / .997 | .051 / .160 / .411 | 0.000, 0.42 | 7.1 / 28 |
| x5 reactive | 708 (404) | .101 / .395 / .977 | .087 / .262 / .487 | 0.000, 0.50 | 7.4 / 28 |
Time-weighted mean occupied: 0.085-0.14 in every R1 cell. Compute-lock busy fraction, all lives: median <= 0.005, p90 0.03-0.10, max 0.30-0.48.
### Utilisation per platform type (occupied / compute-lock), x2 CD and x5 reactive
x2 CD: pynqFpga .135/.0001, rpiCpu .154/.028, xavierCpu .059/.003, xavierGpu .123/.004. x5 reactive: pynqFpga .218/.0002, rpiCpu .342/.094, xavierCpu .103/.004. (All rungs/arms in `desc_r1.json`.)
### Latency shares (seconds per task; % of the latency)
| | latency | sched | queue | ingress | cold | rendezvous | exchange | exec | other |
|---|---|---|---|---|---|---|---|---|---|
| x2 CD | 0.661 | .019 (2.9) | .094 (14.2) | .052 (7.9) | .083 (12.6) | .006 (0.9) | .334 (50.5) | .037 (5.6) | .055 (8.3) |
| x3 CD | 0.612 | .013 | .085 | .051 | .061 | .004 | .259 | .032 | .111 |
| x5 CD | 0.569 | .008 | .088 | .052 | .041 | .002 | .248 | .029 | .095 |
| x2 reactive | 0.846 | 0 | .114 | .054 | .044 | .008 | .548 | .048 | .032 |
| x3 reactive | 0.785 | 0 | .093 | .052 | .031 | .005 | .479 | .045 | .064 |
| x5 reactive | 0.784 | 0 | .086 | .053 | .017 | .003 | .523 | .045 | .053 |
`other` = storage waits, input/output I/O beyond exchange, output.
### Replica counts over time
Time series (60 points per run) in `desc_r1.json` -> per_run -> replica_counts. Alive replicas (median of topologies) at the deciles of the run stay between 2 and 11 for every rung/arm; KPA scales to zero between bursts (17 scale-to-zero events in 9483 x3 CD). Time-mean 4.5 / 5.6 / 7.1 (CD x2/x3/x5), peak 14 / 14 / 28.
### I12's anomaly at scale 1.0 (CD faster at x5)
CD latency still falls 0.661 -> 0.612 -> 0.569 s (-14 %) at one time scale, so the rung-dependent time scale was not the (only) cause. Cold starts per task median 0.216 -> 0.164 -> 0.115 (reactive 0.105 -> 0.084 -> 0.052): rising warmth. By component, x2 -> x5: cold -0.042 s, exchange -0.086 s, `other` +0.040 s, queue -0.006 s. Reactive latency is flat (0.846 -> 0.784) while its cold starts also fall. Not further decomposed.

## 6. Concentration vs saturation (occupied fraction, lives >= 60 s, median over 6 topologies; no bar)
| | R1 CD x2/x3/x5 | R1 reactive | old physics CD | old physics reactive |
|---|---|---|---|---|
| median | .031/.037/.051 | .062/.073/.087 | .111/.154/.311 | .480/.612/.768 |
| p90 | .119/.133/.160 | .167/.212/.262 | .456/.506/.753 | .735/.885/.959 |
| max | .418/.342/.411 | .570/.412/.487 | .948/.935/.918 | .914/.966/.983 |
| share > 0.5 | 0/0/0 | .003/0/0 | .067/.117/.301 | .453/.670/.658 |
| share > 0.8 | 0/0/0 | 0/0/0 | .005/.026/.083 | .051/.172/.494 |
| top-10 % share of busy seconds | .42/.41/.42 | .42/.42/.50 | .63/.57/.57 | .48/.43/.21 |
| lives (>= 60 s), x2/x5 | 2720 (539)/1446 (406) | 1368 (622)/708 (404) | 682 (288)/50 (36) | 216 (133)/13 (13) |
Old physics = defaults (no R1 flags), same traces, `HEROSIM_POLICY_TIME_SCALE=1.0`, 12,000 arrivals, same cells; old latency 10.5-22.5 s (CD) and 22.8-74.3 s (reactive), 85-96 % of it queue.
Reading, no bar: R1 has no long-lived replica above 0.57 and a low mean (0.09-0.14), with the top 10 % of replicas carrying ~42 % of busy seconds: neither "a few near 1" nor "most high". Old reactive at x3/x5 is the saturation pattern (median 0.61-0.77, half of replicas > 0.8 at x5); old CD sits between.

## 7. Gate summaries
`fresh_topo_burst_v1_gate.py`: two new keys per summary, `latency_percentiles` {n, p50, p95, p99, max; nearest rank over done - dispatched} and `replica_count_series` {t, total, by_function, peak, time_mean; 121-point grid from the one-second systemEvents}. Existing keys untouched. On the 9483 x3 CD result: p50 0.183, p95 1.970, p99 6.305, max 37.36 s; peak 28 replicas, matching the trace (time-mean 5.09 against 5.15 from the trace: the series samples once a second).
Not run through a full gate (needs the cluster inputs); the two helpers are unit-tested and were run on a real result JSON.

## Tests
`tests/test_physics_audit.py`: 30 passed, 1 skipped (integration test needs HEROSIM_AUDIT_TEST_CFG). With test_capacity_sweep_v1, test_decima_wfair: no failures. tests/test_record_hygiene.py: 14 failures on the unchanged record (docs untouched in this pass).
