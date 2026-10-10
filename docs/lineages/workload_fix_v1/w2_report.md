# workload_fix_v1 run report — stage W2 (branch rp/wf1-run from 690ba384; pushed; run code f22b2e7e)

R1 on every run: HEROSIM_TRANSFER_MODEL=pipelined, HEROSIM_REPLICA_RELEASE=1, HEROSIM_SCALEOUT=kpa, GATE_FIXED_POLICY_TIME_SCALE=1.0 (the driver refuses any other env; summaries re-check it). Outputs: datalab simulation_data/workload_fix_v1/.
Pinned detached worktrees ~/gnn-herosim-wt/wf1_<hash>; src/ identical across f44cf9cf (replay) .. f22b2e7e (run): 0 diff lines.

## Default path replays
- 5 R1 cells (9483 g0 x20; CD, batched, reactive, selfpredict, locality) on 690ba384 vs the pinned code: every simulated field identical; only total_rtt_plus_inference differs (adds measured decision wall-clock, 1-3 s; total_rtt equal).
- Legacy build chain reproduces client_local_v1/wl_so/grounded_x20 byte for byte, 4/4 windows.
- Re-running the x0.5 calibration step reproduced median queue share 0.1172019077024211 to all digits.
- Full pytest: 26 failures, all present on a pristine 690ba384 export; 0 new. 44 new tests pass.

## 1. Calibration topologies
- Minted 9601-9604 from the pool template (identical to a pool config except network.topology.seed). W4 static check 4/4 pass.
- BUT 9603, 9604 time out (900 s) on all 8 cells (CD, Knative x 4 windows) at x2. The static reachability check does NOT predict this (it passes them); the cause is runtime ("No compatible hardware available for dnn2 ... client_node11"), a capacity/eligibility hang, not reachability.
- Per your decision: replaced by the next ids passing the same screen. Screened 9605-9610 (x2, CD+Knative, 4 windows, 900 s): 9605, 9606 hang; 9607, 9608, 9609, 9610 finish 8/8. Admitted the first two in id order: **9607, 9608**. Calibration set = 9601, 9602, 9607, 9608. 9603-9606 excluded; 9609/9610 unused. (4 of 10 minted ids hang.) The screen covers x2 only.

## 2. Provisional load (CD median queue share over 4 topologies x g0,g1; legacy payloads; single origin; ladder protocol: timestamps and batch_timeout x 1/m)
Target 0.3 (8 evaluations, bracket [0.5,24]): x0.5 0.1172; x24 0.3427; x3.464 0.2131; x9.118 0.2847; x14.793 0.3100; x11.614 0.3007; x10.291 0.2953; x10.932 0.2971. Closest: **m = 11.6139 (0.3007)**; bracket [10.93, 11.61].
Target 0.1: bracket [0.5,24] did not bracket (0.1172 at the low end > 0.1; search stopped at 2 steps as pre-stated). Labelled extension on [0.1,0.5], new 8-step cap: x0.1 0.0668; x0.5 0.1172; x0.2236 0.0944; x0.3344 0.1061; x0.2734 0.1011; x0.2472 0.0974; x0.26 0.0996; x0.2666 0.1001. Closest: **m = 0.2666 (0.1001)**; bracket [0.26, 0.2666].
All 8 cells finished at every evaluation. The light rung is 0.27x the base trace rate (0.12 tasks/s), so its batch_timeout is 16 s x 3.75 = 60 s.

## 3. Stage W2 (wf1_v1 payloads; 19 test topologies x 2 rungs x 4 windows x 5 arms = 760 runs; 760/760 done, 0 failed, no reruns needed)
W2 payloads: mean 27.2 MB (legacy 429.2), median 4.77 MB (197.2), 69.6 % of pairs < 10 MB; pair graph and events identical to the legacy workload.
Per-topology-median paired % vs CD (latency); Holm over {selfpredict, locality, batched} x 2 rungs = 6 tests; reactive = context, outside the family.

| rung | CD lat s | CD queue share | selfpredict | locality | batched | reactive (context) |
|---|---|---|---|---|---|---|
| lo (x0.2666) | 2.85 | 0.09 | -14.9 % (19/19 faster, Holm 2.3e-5) | +0.6 % (0/19, CD-FASTER) | +0.1 % (0/19, CD-FASTER) | -14.9 % (19/19) |
| hi (x11.61)  | 0.50 | 0.37 | -6.1 % (19/19, Holm 2.3e-5) | +1.4 % (1/19, CD-FASTER) | +0.9 % (2/19, CD-FASTER) | +4.4 % (1/19) |

Exchange per task / share of latency (CD): lo 0.07 s / 3.9 %; hi 0.05 s / 10.5 %. selfpredict hi 0.05 s / 11.1 %. (kpa_scaleout_v1 legacy payloads, release x2: ~1.1 s per task.)
Prediction 1 (CD first at every stage): FAILS at W2, both rungs; self-predict is faster, consistent with kpa_scaleout_v1. Locality and one-pass greedy trail CD by <= 1.4 % (smaller than the 3-12 % under legacy payloads).
Caveats: (a) at the light rung the reactive/self-predict advantage coincides with 60 s batching for CD; the ladder protocol scales batch_timeout with the rung, so rung and batching wait move together. (b) Calibrated queue shares (0.10/0.30, legacy payloads) are not the W2 shares (CD 0.09 / 0.37): the rungs are provisional by design. (c) Prediction 2 needs access classes: exchange_by_access_class is "unavailable (uniform links)" in W2 by construction; the reader splits wired/wifi/cellular pairs once W3 summaries carry peerExchangeByAccessClass (tested on synthetic summaries only, not yet on a real run). (d) Queue share and cold-start read from summaries; cold_start_pct is in each summary.
Read: simulation_data/workload_fix_v1/w2_read.json (copy in scratchpad).

## Not mine
Two jobs from another session (wf1w23, wf1w234, from worktrees wf1w4_2c286008 / gnn-herosim-pin-2b49065c) were running on the account while W2 finished; untouched.
Stopped after W2 as instructed. docs/ and LINEAGES.md not edited.
Mistakes caught on the way: an sbatch --export comma split ran only the lo rung first (cancelled, resubmitted; the 128 valid lo summaries were kept); a first calibration job was cancelled after 9603/9604 hung.
