# workload_fix_v1 amendment WB — fixed batching window, stage W2 rerun (branch rp/wf1-run)

Commits: builder change **b401b656** (pushed; S6/S7 need this one); reader tweak 612a7686 (scripts only). Runs from the pinned detached worktree wf1_b401b656 on datalab, R1 env, outputs simulation_data/workload_fix_v1/stage_w2_wb/ (inputs/, gate/, gate_ladder_cd/, gate_nonreaders/, wb_read.json; copy in scratchpad).

## 1. Builder
`workload_fix_v1_build.py --batch-timeout-fixed 16`: scheduler.batch_timeout = 16 on every rung, timestamps still scale, ladder value recorded in the cfg (workload_fix_v1_batch_timeout {fixed_s, ladder_value_s}). Workload bytes identical to the default build. Without the flag nothing changes (tests: default cfg has no record; workload bytes equal; only batch_timeout + the record differ). Built rungs: lo ladder value 60.015 s -> 16; hi 1.378 s -> 16.

## 2. Arms that read batch_timeout (from code)
Read it: cd (PeerGreedyNetworkCDScheduler), batched (PeerGreedyNetworkBatchScheduler), locality (= the batched policy with HEROSIM_PG_EXCHANGE_SCALE) — all on GNNOrchestrator -> GNNScheduler peer-group batching; src/policy/gnn/orchestrator.py:56-71 applies cfg scheduler.batch_timeout.
Do not read it: reactive (KnativeNetworkScheduler), selfpredict (PeerGreedySelfPredictNetworkScheduler, per arrival on KnativeNetworkOrchestrator). Confirmed empirically: 48 reactive/selfpredict runs (topologies 9483, 9484, 9487 x 2 rungs x 4 windows) with the fixed window are identical to the first-run summaries on every field (wallclock/code/inference-time excluded).
Also checked: the ladder-window CD rerun (152 runs, run to add cold-start time) reproduces the first W2 run on all 152 cells, every field.

## 3. Rerun (cd, batched, locality; 19 x 2 x 4 = 456): 453 done, 3 hung
Hung at the heavy rung, window g2, with the 16 s window: cd and batched on 9565, locality on 9538. Signature: simulated clock frozen (2936.80 s / 1944.59 s), "No compatible hardware available for dnn2 ... client_nodeN" repeated (28,777 times in the last 4 MB of one log) — the starved-client spin of calibration topologies 9603/9604. I cancelled the job at ~39 min of the 45-min first-pass timeout because the logs were growing ~30 MB/min each (0.2-0.9 GB each) and /home was at 249 of 268 GB. **The registered one-rerun-at-3x-timeout was NOT run for these 3** (a frozen clock is a hang, not slowness; the rerun would write several GB per run). They count as failures for their arm: that (topology, window) drops out of that arm's paired tests; the sensitivity block excludes 9565 and 9538 for every arm. Log excerpts (first and last 1 MB) kept as *.log.excerpt next to failed.json. If you want the 3x rerun anyway, it needs a file-size cap on the log.
Note: the same three cells finished in the first W2 run (1.4 s ladder window at hi) — the 16 s window induces these hangs.

## 4. W2 family re-read (unchanged arms from the first W2 run; Holm over selfpredict/locality/batched x 2 rungs; reactive context)
| rung | CD s | selfpredict vs CD | locality vs CD | batched vs CD | reactive (context) |
|---|---|---|---|---|---|
| lo, window 16 s (was 60 s) | 2.42 (was 2.85) | +0.2 % (6/19 faster, Holm 0.096, NOT-SEPARATED) | +0.7 % (0/19, CD-FASTER) | +0.1 % (0/19, CD-FASTER) | +0.3 % (4/19) |
| hi, window 16 s (was 1.4 s) | 1.35 (was 0.50) | -67.3 % (19/19, Holm 2e-5, CONFIRMED) | +0.5 % (1/19, CD-FASTER) | +0.5 % (0/19, CD-FASTER) | -63.8 % (19/19) |
Light rung: CD is first again (self-predict no longer separable); the 60 s window explained the first-read light-rung loss. Heavy rung: the window grew 1.4 -> 16 s and CD's latency rose 0.50 -> 1.35 s; self-predict and reactive (no window) are 64-67 % faster. So the fix is not in CD's favour at the heavy rung, as the amendment anticipated; CD / locality / batched stay within 0.5-0.7 % of each other.
Sensitivity excluding 9565 and 9538 for every arm: lo self-predict +0.19 % reads CD-FASTER (Holm-significant) instead of NOT-SEPARATED; hi unchanged in sign and label (-67.5 %). CD first at lo, not at hi in both reads.

## 5. CD latency by stage (mean s per task; median over 19 topologies of per-topology means; "other" = latency minus the five listed)
| rung, window | latency | batching wait (dispatch->scheduled) | queue | exchange | rendezvous | cold start | other |
|---|---|---|---|---|---|---|---|
| lo, ladder 60 s | 2.847 | 0.696 | 0.180 | 0.073 | 1.582 | 0.210 | 0.113 |
| lo, fixed 16 s | 2.417 | 0.148 | 0.181 | 0.075 | 1.691 | 0.210 | 0.114 |
| hi, ladder 1.4 s | 0.498 | 0.017 | 0.188 | 0.050 | 0.036 | 0.033 | 0.175 |
| hi, fixed 16 s | 1.351 | 0.932 | 0.178 | 0.033 | 0.011 | 0.024 | 0.173 |
(fixed hi uses 18 topologies' worth of g2 for 9565; excluding 9565 and 9538: lo 2.846 -> 2.413, hi 0.494 -> 1.351, same pattern.) Reading: at lo the 60 s window cost 0.55 s of batching wait (0.696 -> 0.148) while rendezvous rose 0.11 s; at hi the 16 s window adds 0.92 s of batching wait. Queue, exchange and cold start barely move between windows at lo; at hi exchange and rendezvous fall as the batch grows (0.050 -> 0.033; 0.036 -> 0.011). The rendezvous (1.6-1.7 s) is CD's largest single component at lo under both windows.

## Housekeeping
- /home is at ~249.7 GB of 268 GB after I removed my three giant logs; other sessions' jobs on the account (wf1w23wb, wf1fix, wf1fix2, wf1w234) are still writing. Worth watching.
- My monitor "fixed-window rerun completes" exited 1 because I cancelled the job it was watching.
- docs/ and LINEAGES.md not edited.
