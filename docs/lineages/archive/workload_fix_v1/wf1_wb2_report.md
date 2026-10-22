# workload_fix_v1 amendment WB2 — tuned batching window (S4)

Commits: log cap cc13d1a4 (sent to S6 and S7); tune job 2c26c53b; reader --tune 5438b9c5; NO_RERUN flag 942fcd2b (run code for step 3). Outputs: simulation_data/workload_fix_v1/stage_w2_wb2/ (tune/, gate/, tune_read.json, wb2_read.json; copies in scratchpad).

## Quota: 50 MB log cap (cc13d1a4)
First 40 MB + last 10 MB of each gate run's combined output, dropped bytes noted; HEROSIM_GATE_LOG_CAP_MB (0 = uncapped). Summaries unchanged: 5 R1 cells replayed through the capped driver identical to the earlier replay on every simulated field. /home 249.7 GB -> 183-185 GB, steady through 616 further runs. 51 tests pass.

## Tuning (CD, calibration 9601/9602/9607/9608, W2 workload, 4 windows, both provisional rungs; 160 runs, 0 hung)
Median CD latency over 16 cells per rung (s): window 1: 2.772 / 0.541, geomean 1.2251; 2: 2.774 / 0.547, 1.2321; 4: 2.778 / 0.603, 1.2940; 8: 2.796 / 0.819, 1.5134; 16: 2.842 / 1.631, 2.1528.
Minimum: **1 s**. Rule from the coordinator: no hung run in the winner or in the runner-up (2 s, +0.57 %) -> the choice stands. 1 s is the lowest value on the grid, so the true optimum may be lower; the light rung is nearly flat in the window (2.77-2.84 s), the heavy rung carries the difference.

## W2 rerun at 1 s (cd, batched, locality; 19 x 2 x 4 = 456; 456/456 done, 0 hung, no rerun needed)
Unchanged arms (reactive, self-predict) from the first W2 run. Per-topology-median paired % vs CD, Holm over {selfpredict, locality, batched} x 2 rungs:
| rung | CD s | selfpredict | locality | batched | reactive (context) |
|---|---|---|---|---|---|
| x0.2666 | 2.36 | +2.4 % (0/19, CD-FASTER) | +0.7 % (CD-FASTER) | +0.1 % (CD-FASTER) | +2.5 % |
| x11.61 | 0.50 | -6.1 % (19/19, CONFIRMED) | +1.5 % (CD-FASTER) | +0.5 % (CD-FASTER) | +5.6 % |
CD is first at the light rung and not at the heavy one: self-predict beats CD by 6.1 % there, the same as under the ladder's 1.4 s window (-6.1 %). Prediction 1 holds at the light rung only. Across the three windows the light-rung order flips with the window (60 s: self-predict -14.9 %; 16 s: tie; 1 s: CD ahead by 2.4 %).

## CD latency by stage, three windows (s per task)
| rung | window | latency | batching wait | queue | exchange | rendezvous | cold start | other |
|---|---|---|---|---|---|---|---|---|
| lo | 60 s (ladder) | 2.847 | 0.696 | 0.180 | 0.073 | 1.582 | 0.210 | 0.113 |
| lo | 16 s | 2.417 | 0.148 | 0.181 | 0.075 | 1.691 | 0.210 | 0.114 |
| lo | 1 s | 2.362 | 0.035 | 0.182 | 0.075 | 1.745 | 0.211 | 0.115 |
| hi | 1.4 s (ladder) | 0.498 | 0.017 | 0.188 | 0.050 | 0.036 | 0.033 | 0.175 |
| hi | 16 s | 1.351 | 0.932 | 0.178 | 0.033 | 0.011 | 0.024 | 0.173 |
| hi | 1 s | 0.498 | 0.011 | 0.188 | 0.051 | 0.037 | 0.033 | 0.175 |
Shrinking the window moves wait into rendezvous at the light rung (0.696 -> 0.035 wait; rendezvous 1.58 -> 1.75), so the net saving is 0.49 s of 2.85. At the heavy rung 1 s and 1.4 s give the same decomposition to three digits.

## Not done / caveats
- Hung cells: none at 1 s on the test topologies (the three that hung at 16 s did not at 1 s). The starvation fix (S5) has not been needed for this stage; nothing was rerun.
- The calibration topologies chose 1 s, the grid edge. The tuned window is a hyperparameter chosen on 4 topologies; the test read at the heavy rung is insensitive to 1 vs 1.4 s.
- docs/ and LINEAGES.md not edited.
