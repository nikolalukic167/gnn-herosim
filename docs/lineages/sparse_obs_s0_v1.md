# sparse_obs_s0_v1 — how much of the hand scheduler's lead comes from information a real scheduler does not have?

**Status:** `REGISTERED` (2026-10-12, signed by the coordinator at 827718cf with four rulings, applied below) — Phase 0, hand arms only, live. Owner S5. Parent question (the user's): *does our
scheduler get information real systems do not, which makes the placement problem pointwise?* Reference arm A0 = `cd_exactS_ib0`
(`ib0_confirm_v1`, the current hand bar). Parents read: `ib0_confirm_v1`, `decision_cost_s0_v1`.

## Why

Every arm in the program prices a candidate with the simulator's own state: the exact queue walk, the exact execution time per platform type, the
exact warmth, the exact partner nodes. A real scheduler sees counts (Knative in-flight), resource requests (k8s) or counts plus estimates (Decima).
If the hand bar's lead over reactive baselines, and the pointwise-separable finding, come from that surplus, then (a) the bar is over-informed and (b) a
learned scorer has room only where the observation is poor. This node measures the surplus, channel by channel, before anything is trained.

## Audit: what `peer_greedy_network` (CD, `cd_exactS`, ib0) reads at decision time

Line numbers are at `fb577a8a` (`rp/decision-cost`). "Exact-in-sim" means the value is the simulator's ground truth at the decision instant.

| # | Quantity | Read at | Exact-in-sim? | A real scheduler (k8s / Knative / Decima) has |
|---|---|---|---|---|
| 1 | Queue drain seconds of a platform: walks `platform.queue.items`, charges each queued task its exec on THIS platform type, storage I/O, source→platform latency and every partner transfer to a known partner node | `live_audit.py:46` (`platform_queue_drain_seconds`), used at `scheduler.py:496`, `:1047`, `:1260` | **Yes**, per-task exact | in-flight / queue **counts**; per-type mean service from history |
| 2 | In-flight remaining seconds (`inflight_service_end − now`) when `HEROSIM_PG_INFLIGHT=1` | `live_audit.py:135`, `scheduler.py:499`, `:1049`, `:1262` | **Yes** (the recorded service end) | whether a task is running (Knative), not when it ends |
| 3 | Execution time per (task type, platform type) | `task.type["executionTime"]`, `scheduler.py:504`, `:1035`, `:1263`, and inside the drain walk | **Yes** (table, deterministic; `table_v0`) | a per-type estimate, noisy, not per hardware type unless profiled |
| 4 | Cold start for the incoming task: 0 if the platform is busy or its previous task has the same type, else the table `coldStartDuration` | `scheduling_cost.py:71`, `scheduler.py:505`, `:1050`, `:1263` | **Yes**, and the same predicate the sim uses (`infrastructure.py` `sandbox_is_warm`) | warm-pool membership is usually visible (Knative); exact reuse state is not |
| 5 | Source→node network latency | `scheduling_cost.py:86`, `scheduler.py:506`, `:1051`, `:1263` | **Yes** (`network_map`) | measured / configured latency; near-exact |
| 6 | Payload sizes, storage I/O (`_approx_comm`) | `live_snapshot_seed.py:17`, `scheduler.py:489`, `:1036` | **Yes** (the app's declared state sizes) | declared sizes, available for workflows with typed edges |
| 7 | Partner table `orch.peer_exchange[task] = {partner: bytes}` | `scheduler.py:428` (`_pg_peer_nodes`), `:431` | **Yes**: the full DAG-edge table, including partners that have not arrived | the DAG is known only for submitted workflows; unarrived partners are not |
| 8 | Partner nodes: placed (`peer.platform`), planned by this batch's pass/plan | `scheduler.py:428–452`, `:1027` (plan), exchange seconds `:453` | **Yes** | placed partners only; a batch's own plan exists only for a batch scheduler |
| 9 | Batch composition and in-batch stacking (`committed_service`; ib0 zeroes its weight) | `scheduler.py:507–508`, `:1053` | **Yes** (the batch window is the scheduler's own) | not applicable to per-arrival schedulers |
| 10 | Replica eligibility / reachability (`_get_valid_replicas`) | `scheduler.py:1008` | **Yes** | yes (the control plane's own state) |
| 11 | Exchange time through links (`platform._payload_transfer_time`, `peer_link_latency`) | `scheduler.py:453–470` | **Yes** (the same method the sim charges) | topology + bandwidth estimate |

Not read: queue wait of the sim's own future arrivals, node CPU/memory contention beyond the queue, link contention (`link_bound_s0_v1`: links never bind). The learned arms read the same
quantities through features (`partial_state_v5`), so a degradation here is a degradation of what both the hand rule and a learned scorer may use.

## Degradations (single flags, default off; flag-off byte-identical, test per flag)

The physics is untouched; only what the scheduler observes changes. Applied at the five score sites above and inside the drain walk, through one
observation module (`src/policy/peer_greedy_network/observation.py`).

| Flag | Observation |
|---|---|
| `HEROSIM_OBS_QUEUE=count` | rows 1–2: drain = (queued count + 1 if a task is in flight) × a service estimate (below); no per-task exec, I/O, latency or partner transfer |
| `HEROSIM_OBS_EXEC_SIGMA=0.3` | row 3: every exec estimate × a log-normal factor drawn once per (task type, platform type) from `HEROSIM_OBS_SEED` (the cell seed), fixed for the run |
| `HEROSIM_OBS_WARM=hidden` | row 4: cold priced at (1 − p̂_warm) × table cold; p̂ from the scheduler's own past placements (below). The disclosed arm: real systems mostly know warmth |
| `HEROSIM_OBS_PARTNERS=arrival` | rows 7–8: a partner's node is known only if the partner has already been placed; the batch's own plan and unarrived partners are invisible (exchange terms dropped for them) |
| BUNDLE | queue=count + exec σ=0.3 + partners=arrival, warmth visible |

## Arms (two same-observation hand arms per degradation, fixed before any read)

Both are `cd_exactS_ib0`'s exact search on the degraded S. Only the estimator differs.
- **NAIVE**: service per queued task = the global mean of exec over all types and platform types; cold priced at the global warm rate.
- **EXPECTATION-CORRECT (EC)**: service per queued task = the per-type mean over the visible queued type mix (types the scheduler can see in the queue); warmth prior from keep-alive age; partners (bundle only) priced by the self-predict rule (`peer_greedy_lookahead`, the unarrived partner at the node the rule would give it now).
- **A0** = `cd_exactS_ib0`, full information.

Reads: **VoI(D) = NAIVE(D) vs A0** (how much the channel is worth); **R(D) = EC(D) vs A0** (what a hand rule cannot recover, the residual a learned scorer could take).
D ∈ {queue, exec, warm, partners, BUNDLE}. Order: BUNDLE and queue first; exec, warm and partners after if slots allow.

## Seat and cells

The `ib0_confirm_v1` seat (160 clients × 24 servers, p 0.6, `fastest_compatible`, WF1, rungs moderate and heavy, pipelined, replica release, KPA). 12 fresh
topologies **19440–19451**. Disjointness: `git grep` of the 194xx–199xx tokens over every remote branch finds 19400, 19404–19417, 19420–19431
(`slate_fix_v1`), 19490 and 19516+; no `/share` name matches. To be re-run at signature.
- Stage 1: A0 + NAIVE/EC for queue and BUNDLE = 5 arms × 12 topologies × 2 rungs × **g0** = **120 cells**; adding g1 doubles it (240).
- Stage 2 (as ruled, see "Stage 2 design"): warmth only, A0+ / W1-NAIVE / W1-EC = 72 cells (g0); exec and partners not run. Walls of the `cd_exactS_ib0` class (about 3 ks median): about 0.4 Mcore-s per 120 cells; one 24-shard array each, about 1 h.
- Queue: at most 24 of mine. The topologies are new, so their inputs (`wf1_*`) must be minted first (one CPU job, as `ib0_confirm_v1` did).

## Rulings applied (2026-10-12) and how they were implemented

- (a) Stage 1 at g0 only: A0 + NAIVE/EC for queue and BUNDLE = 120 cells. Stage 2 (exec, partners, warmth) after the stage-1 read.
- (b) Topologies 19440–19451 (`scripts_cosim/sparse_obs_s0_v1_selected.json`), disjointness re-checked at launch.
- (c) A0 stays the batch exact search. BUNDLE arms search the visible batch exactly with partners known only once placed (`planned_node_name` / `platform`); the batch's own plan is invisible.
  EC prices a partner with no known node at the node of its bytes-heaviest known partner (the lookahead rule's two-step estimate, `PeerGreedyLookaheadNetworkScheduler._pg_predict_node`). **Deviation:**
  the event-index self-predict (`PeerGreedySelfPredictNetworkScheduler`) needs the per-arrival orchestrator's event list and is not available inside the batched CD search, so EC uses the two-step estimate. The seat gap (batch search vs per-arrival serving) is reported with the result.
- (d) Row 2 (in-flight remaining) is **not degraded**, ruled after the draft: the reference A0 does not read it (`HEROSIM_PG_INFLIGHT` is unset for `cd_exactS_ib0`), so a term there would be information A0 lacks and EC could beat A0 for the wrong reason. The BUNDLE degrades rows 1, 3, 7, 8. The `HEROSIM_OBS_INFLIGHT` flag exists in the module, is tested, and is used by no arm.
- Warmth (row 4) stays exact in stage 1 and in the bundle; the `HEROSIM_OBS_WARM` flag is a stage-2 item and is not implemented yet.
- Exec: one estimator for both arms (table value × the seeded log-normal factor per (task type, platform type)); NAIVE/EC differ only in queue, in-flight and partners.

## Implementation (flags default off, `src/policy/peer_greedy_network/observation.py`)

`HEROSIM_OBS_ESTIMATOR=naive|ec` (required with any channel), `HEROSIM_OBS_QUEUE=count`, `HEROSIM_OBS_EXEC_SIGMA=<s>`, `HEROSIM_OBS_INFLIGHT=running`, `HEROSIM_OBS_PARTNERS=arrival`, `HEROSIM_OBS_SEED` (the cell seed).
With none set `Observation.from_env()` is None and every score site runs its registered arithmetic (tests/test_sparse_obs.py compares the flag-off terms to the registered functions). Sites: `_pg_choose`, `_pg_plan_cost`, `_pg_standalone`
(drain, exec; the running term is 0 for every arm) and `_pg_peer_nodes` (partners); the exec channel also reaches the exact queue walk (`live_audit.platform_queue_drain_seconds(exec_of=...)`, default None = unchanged).
Books (`schedulerCounters`): `obs_drain_degraded`, `obs_drain_exact`, `obs_exec_degraded`, `obs_inflight_degraded`, `obs_partner_hidden`, `obs_partner_predicted`. The gate checks that a degraded channel's book is non-zero, that the in-flight book is 0, that the exact queue walk
is never used by a queue-degraded arm, and that every other arm's books are zero. Arms: `cd_obs_queue_{naive,ec}`, `cd_obs_bundle_{naive,ec}` (stage 1), `cd_obs_exec`, `cd_obs_partners_{naive,ec}` (stage 2).

## Rulings after the smoke (2026-10-12)

- **Partner row = NOT-TESTABLE-IN-SEAT.** The scheduler batches whole peer groups: cross-batch pairs are 438 (moderate) / 86 (heavy) against about 66,000 in-batch pairs in the smoke cells (19405 g0). Nothing is "unarrived and known" in this seat, so the EC two-step estimate has no inputs (`obs_partner_predicted` is 158,388 / 112 evaluations of those few pairs) and EC = NAIVE on partners by construction.
  The arrival view still hides the batch plan the exact search prices in-batch partners from: it removes A0's in-batch co-location signal. The bundle is therefore a **mixed read dominated by the hidden batch plan**; its VoI is reported as the measured price of removing the batch plan (the smoke cell reads about +100 % at moderate), not as a pass/kill input. The stronger EC (a batch-derived estimate for an unplaced in-batch partner) is not built (ruled).
- **The kill rule (VoI < 10 % or R < 5 %) applies per channel**: drain on the queue arms in stage 1, warmth (W1) in stage 2. Exec is not run (see Stage 2).

## Stage 2 design (ruled by the coordinator 2026-10-12; signature after the stage-1 drain read)

Design by SCIENTIST; lines at `2d4e15c4` unless named. Stage 2 runs **warmth only**: A0+, W1-NAIVE, W1-EC = **72 cells** (12 topologies × 2 rungs × g0).

**Physics both channels rest on.**
- Exec is a deterministic table per (type, platform type), `data/nofs-ids/task-types.json`, read at `scheduler.py:433`. dnn1/dnn2/rf run 0.000–0.17 s; cnn runs 0.10 (xavierGpu) / 0.56 (xavierDla) / 0.71 (xavierCpu) / 3.09 s (rpiCpu).
- Cold is a table in the same file: 0.05–1.16 s on CPUs, 6.1–12.7 s on xavierGpu, 38.0 s for cnn on xavierDla.
- A sandbox is warm iff the previous task on the platform has the same type (`warmth.py:197–202`), applied at `infrastructure.py:1616–1628`. Under `REPLICA_RELEASE`, `previous_task` is set at pop (`infrastructure.py:1651`). Replica removal resets it only under `node_disk_v2` (`autoscaler.py:606–607`).
- Cold is 12.6 → 7.2 % of CD latency in the WF1 seat (×2 → ×5); cold starts per task fall 0.216 → 0.115 with load (`physics_audit_v1.md:110–118`).

**Exec: not run; the reason is recorded instead of a measurement.** Under `table_v0` exec is deterministic, so any estimator with history (EWMA, per-(type, platform-type) mean) equals the table after one completion per pair. A realistic EC is therefore A0 within minutes, and R ≡ 0 by construction.
- The only degraded exec observations that persist are no hardware resolution (per-type mean across platforms) or a bias that never self-corrects (`HEROSIM_OBS_EXEC_SIGMA`). Neither is how a system with history behaves.
- σ = 0.3 moves S by about 0.35 × exec at most, under 0.06 s outside cnn, and cannot flip cnn's 5–30× platform gaps.
- `hidden_exec_s0_v1` closed NO-HEADROOM even with stochastic hidden exec physics. That was another seat (6 servers, exec 0.05 s of 14.6 s), so it is supporting evidence, not a carry-over.
- The descriptive no-profile arm (E-NOPROF: exec = per-type mean, `observation.py:92–101`) is **not run**.

**Warmth: what a real system observes.**
- Knative sees ready pods per revision, i.e. per function; cold seconds are not priced.
- OpenWhisk invokers keep containers for a fixed 10 minutes, and the controller's load balancer sees invoker health, capacity and the history of prior executions (Shahrad et al. ATC'20, `/root/papers_text/03_workloads_and_traces/Shahrad_ServerlessInTheWild_ATC20.txt:104`, `:580–590`).
- AWS/Azure keep containers for 10/20 minutes (same file, `:63–66`).
- HeROsim has one sandbox per platform, so "which function this platform holds" is exactly the Knative-visible bit, and A0 reads it. The realistic degradation hides the cold **seconds**, not the bit.
- Hiding the bit (W2) is recoverable exactly: the warm state is a deterministic function of the scheduler's own FIFO placements plus autoscaler removals, both control-plane facts. **W2 is not run.** Its EC would be that bookkeeping, R ≡ 0 by construction.

**A0 is not a full-information warmth reference.** A0's cold term is `incoming_cold_start_time` (`scheduling_cost.py:71–83`), called at `scheduler.py:527` and `:1077`. It has two defects:
1. It reads `previous_task` at decision time. The task actually starts against the type at the queue tail, or the in-batch task the plan stacks ahead of it. The drain walk charges queued tasks no cold (`live_audit.py:94–100`).
2. "`current_task` set → cold 0" (`scheduling_cost.py:73`) is wrong under release. `current_task` is set only during compute (`infrastructure.py:1857`), and a different-type task still cold-starts.

A degraded arm with a better predictor could therefore beat A0 legitimately. Every warmth contrast uses the corrected reference **A0+**.

**Arms (24 cells each).**
- **A0+** (full information, corrected; a hand-bar correction in its own right). Cold is priced against the type the task will follow: the queue tail, or its in-batch predecessor under the plan. The `current_task` → 0 rule is dropped under release.
  - Own contrast: **A0+ vs A0**, paired per topology, Holm, 12 topologies, both rungs.
  - **Before coding A0+, S5 confirms which autoscaler path resets `previous_task` in the WF1 seat** (KPA path vs `autoscaler.py:606–607`), so A0+ predicts resets the sim actually performs.
- **W1-NAIVE**: A0+'s warm bit; cold seconds = the global mean of the cold table.
- **W1-EC**: A0+'s warm bit; cold seconds = the per-platform-type mean over task types (a runtime-class prior). Within-class spread is small except xavierGpu (6.1–12.7 s); xavierDla has only cnn.

**Reads, kill and leak checks.**
- VoI_W1 = W1-NAIVE vs A0+; R_W1 = W1-EC vs A0+. Same statistic as below.
- Kill: VoI_W1 < 10 % or R_W1 < 5 % at both rungs → `HAND-RECOVERS-UNDER-SPARSITY` for warmth.
- Books `obs_cold_degraded` > 0 and `obs_cold_exact` = 0 in W1 arms, and no W1 call reaches `incoming_cold_start_time`.
- A W1 arm faster than A0+ by ≥ 1 % (Holm) is audited before VoI is read.
- Flag-off identity test per new flag. Stage-1 A0 cells are reused only at the same pinned head, with the 2-cell identity check.

**Priors (SCIENTIST).** A0+ ≥ 1 % faster than A0: ~35 %. VoI_W1 ≥ 10 %: ~10 % heavy, ~20 % moderate. R_W1 ≥ 5 %: ~5 %. A Phase-1 pass from exec or warmth: ~3 %. The likely yield is the A0+ hand-bar check and the credibility numbers.

**Correction on record.** The "62/49 %" in `slate_fix_v1.md:221–231` (rp/slate-fix) is the share of the 417 train / 49 held-out *improved* labels whose new best used a candidate the declared slate dropped by its cold = 0 error. That is about 14.5 % / 9.3 % of datasets, and it is a defect in the declared slate's pricing, not in what A0 observes.

## Stage 2 implementation and the previous_task reset path (confirmed before coding A0+, 2026-10-12)

- **Reset path.** Every replica removal in this seat goes through `Autoscaler._release_replica` (`autoscaler.py:581`): the KPA scale-down (`_kpa_scale_down`, `autoscaler.py:335`) and the legacy `scale_down` (`:579`) both call it, and it is the only place `previous_task` is cleared on removal (`autoscaler.py:606–607`). The clear is gated on `warmth_physics == NODE_DISK_V2`.
  The seat declares none: the minted `cfg` carries no `warmth_physics`, the space config does not set it, and `r1a_gate.sbatch` unsets every `HEROSIM_*` variable before exporting its own, so `resolve_warmth_physics` returns the default `platform_reuse_v1` (`warmth.py:123–145`).
  **Under `platform_reuse_v1` a removal does not reset `previous_task`**: the sandbox state survives scale-down and scale-up, so A0+ has no autoscaler resets to predict. This is derived from the code and the inputs; the cell summaries do not record `warmth_physics`, and the per-task raw results that would are deleted after summarising.
  `previous_task` is set at pop under `HEROSIM_REPLICA_RELEASE` (`infrastructure.py:1651`) and by warm-up (`:1418`).
- **A0+ (`HEROSIM_PG_COLD_TAIL=1`, `scheduler.py` `_pg_cold`).** Cold is priced against the type the task will follow: the platform's queue tail, else its `previous_task`; inside a whole plan (`_pg_plan_cost`: the exact search, the expansion, the kept pass), the task the plan stacks ahead of it on the platform in batch order. "`current_task` set -> cold 0" is dropped.
  **Approximation:** the single-task greedy passes (`_pg_choose`, `_pg_standalone`) price cold from the queue tail only and do not see the in-batch tasks already committed; only the plan-level scoring does. The flag is default off and byte-identical off.
- **W1 (`HEROSIM_OBS_WARM=seconds`, needs `HEROSIM_PG_COLD_TAIL=1`).** The warm bit is A0+'s; the seconds are the global mean of the cold table (naive) or the mean over task types for the platform's type (ec). Books: `pg_cold_tail_calls`, `obs_cold_degraded`, `obs_cold_exact` (0 by construction: a W1 call never reaches `incoming_cold_start_time`).
- Arms `cd_obs_a0plus`, `cd_obs_w1_naive`, `cd_obs_w1_ec`; reader `scripts_cosim/sparse_obs_v2_gate_read.py` (A0+ vs A0 primary; VoI_W1, R_W1; the leak check; the books).

## Rules, written before data

- Statistic: the paired % of mean latency per topology (median over windows), median over topologies, exact Wilcoxon, Holm within VoI and within R (the `prune_g8_gate_read.py` shape). Negative = the first arm faster; here "VoI" = NAIVE slower than A0, reported as a positive loss.
- **Kill:** VoI < 10 % **or** R < 5 % at both rungs for every D and the bundle → `CLOSE HAND-RECOVERS-UNDER-SPARSITY`.
- **Pass:** goes to Phase 1 (registered successor, not part of this node): GNN, MP-OFF twin and a hand-feature MLP trained on degraded observations with true-RTT labels, vs the EC hand rule; WIN ≥ 3 % at both rungs; an MP claim only against the twin; a learned arm that beats A0 is treated as an information leak and audited.
- **Disclosed up front:** a pass most likely yields a learned-scorer result, not an MP result. This seat is more realistic than the current one (it removes an over-informed baseline). Warmth hidden is a disclosed arm, not the realistic one.
- **Prediction (SCIENTIST's priors):** BUNDLE VoI ≥ 10 % at heavy ~40 %; R ≥ 5 % ~20 %.
- Flag-off identity: a test per flag that the scheduler's scores are unchanged with the flag off; the live gate adds a 2-cell identity check at the pinned head.

## Record

- 2026-10-12 — **Stage 1 read (119 of 120 cells; array 864053, pin 8392fbd2, reader 2d4e15c4, output `sparse_obs/read_s1.json`).** 19442 heavy `cd_obs_bundle_ec` still running (progress file live) at the read; 0 failed.
  Identity (smoke 19405 g0, `cd_exactS_ib0` at 8392fbd2 vs d192081f): total_rtt, endTime, averageElapsedTime and every shared counter equal at both rungs. Instruments OK (no exact queue walk in a queue-degraded arm, in-flight book 0, A0 books 0).
  - **Drain channel (queue count × service): VoI +0.73 % moderate / +0.34 % heavy** (NAIVE vs A0; Holm .41 both); R: −0.74 % / +1.49 % (heavy p .043, Holm .085); EC vs NAIVE −0.72 % / +0.42 %. Both far under the 10 % bar: STAGE1_PASS False. The exact queue walk is worth about nothing over a count at this seat; no residual for a learned scorer on drain.
  - **Exec channel:** not run (stage 2). **Warmth:** flag not written (stage 2).
  - **Bundle (mixed read, not a pass/kill input): VoI +113.2 % moderate / +169.6 % heavy** (0/12, Holm .002), R +124.3 % / +181.7 % (n=11 heavy), EC vs NAIVE +6.7 % / +2.4 %. The bundle hides the batch plan the exact search prices in-batch partners from; this is the measured price of removing the batch plan (how much of the margin is in-batch co-location). It also contains the queue count and exec noise (exec noise was not isolated in stage 1).
  - **Partner row: NOT-TESTABLE-IN-SEAT.** Cross-batch pairs are 0.1–0.4 % of pairs, so hiding unarrived partners removes nothing from either degraded arm; the EC two-step estimate has no inputs.
  - Prediction: bundle VoI ≥ 10 % at heavy (~40 %) held; R ≥ 5 % (~20 %) held formally on a vacuous EC. Decision time not scored.
