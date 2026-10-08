# workload_fix_v1 — payloads, access-link classes and task types on R1 (freeze workload WF1)

**Status:** `ACTIVE` (2026-10-08) — W2 and W3 read on R1.1. W3 makes exchange the dominant cost; CD then leads at light and the batching arms at heavy, and self-predict's heavy lead shrinks to −2.9 % (DIRECTION). `kpa_scaleout_v1` gates are being re-measured on R1.1; W4 is read next. Depends on: `physics_audit_v1` (R1 frozen). Plan: [`reference_physics_programme.md`](reference_physics_programme.md).
Created 2026-10-08 from W2–W4 of the withdrawn draft `workload_redesign_v1` (never committed) (W1 moved to `call_graph_pairing_v1`).


**Pre-run amendments (2026-10-07, decided by the coordinator from a code check).**
- **W3 applies to every node's access links, servers included.** Peer exchange runs server to server
  (`Platform._peer_exchange_time`, `src/placement/infrastructure.py`), and replicas are server-only, so a client's
  access link carries only the ingress input (12.8–153.6 KB per task, `data/nofs-ids/task-types.json`): at 4 MB/s
  about 0.04 s, against about 1 s of latency. Classes on client links alone could not move exchange. Each node
  (server or client) draws its class once per topology, mix 40 / 40 / 20 %. Our servers are Raspberry Pi and Jetson
  Xavier edge devices, so wireless edge servers are in scope.
- **Cellular is directional:** 4 MB/s out of the node, 75 MB/s into it. Links today carry one bandwidth for both
  directions (`network_fabric.link_key`), so the fabric gains a per-direction bandwidth; uniform-link topologies are
  unchanged.
- **Prediction 2 reworded:** exchange is under 10 % of latency for pairs whose two servers are both wired, and not
  when either server is Wi-Fi or cellular.
- **W2 scale, stated before the read:** mean payload per pair falls from about 430 MB to about 27 MB, and about 72 %
  of the new bytes sit in the heavy tier, so co-location value will come mostly from heavy pairs and route latency.
- **W4 cause found:** the generator's reachability repair skips any type with no initial replica placement
  (`src/generate_infrastructure.py`, `if not replica_servers: continue`), so `rf` / `cnn` can be unreachable. The fix
  repairs every type the workload uses; the static check then runs over all test and calibration topologies.


**Implementation and W4 precondition (2026-10-08, before any W4 run).** Code at `28411eb3` (all opt-in; with the
flags off, 96 of 96 configs and the minter replay byte for byte): `src/placement/workload_payloads.py` (W2,
`--payload-sampler wf1_v1` on `grounded_workload_v1_mint.py` only), per-node access classes and direction-aware
bandwidth in `src/placement/network_fabric.py` (W3; every bandwidth reader, including CD's cost model and the GNN
features, goes through `fabric.hops` or `route_hops_and_bottleneck`), and `network.reachability_repair` (W4), which
fails loud on a used type with no server replica instead of skipping it. Static check
(`scripts_cosim/workload_fix_v1_reachability_check.py`): **19 / 19 test topologies pass without the repair**
([`reach_19.json`](workload_fix_v1/reach_19.json)) and 96 / 96 configs of the pool pass with it
([`reach_all_repair.json`](workload_fix_v1/reach_all_repair.json)), so the `rf` / `cnn` defect is latent in these
configs; the 4 calibration topologies are checked when minted. W3 draws on the 96 configs: 40.1 / 40.5 / 19.4 %.
**Calibration topologies: ids 9601–9604**, minted with the same generator and config template as the 9473–9568
pool (unused anywhere in `simulation_data/` or the record, checked 2026-10-08). The 77 non-admitted ids of that pool
are not used: they failed admission under old physics.


**Pre-run amendment (2026-10-08, coordinator, before any W2+W3 or W2+W3+W4 cell): stages run in parallel, and
are read in order.** The W2+W3 and W2+W3+W4 cells may run while W2 is running. The registered order still governs
reading. No arm comparison of a stage is opened before the previous stage is read. Before that, only completion,
failure and hang counts may be looked at. If reading an earlier stage forces a change to shared code, configs or rungs,
every later stage's cells are discarded and rerun. All stages use the same provisional rungs (from the W2 bisection) and
the same 19 test topologies. Each stage writes to its own directory under `simulation_data/workload_fix_v1/`.
Reason: W3 and W4 add new code paths (direction-aware links; `rf`/`cnn`), and hangs surfaced as stuck cells on
calibration topologies 9603/9604. Finding such failures while W2 runs costs nothing in attribution, since each
stage still differs from the previous one by one factor and is read after it.

## Question
Which classical rankings on R1 survive replacing the synthetic payloads, uniform links and two-type task mix with
grounded or explicitly labelled values? The result fixes workload **WF1**, on which all learned arms are trained.

## Factors (introduced one at a time, then combined)
| Factor | Current | Fixed value | Source class |
|---|---|---|---|
| W2 Payload | 200 MB × 10^U(−1,1) (20 MB–2 GB) per partner pair | log-normal per pair with median 4 MB (σ = 1.2 in natural-log space), plus a labelled heavy tier: 10 % of pairs at 50–500 MB log-uniform. Overall ≈ 70 % of pairs under 10 MB (0.9 × Φ(ln 2.5 / 1.2) ≈ 0.70), approximately matching Eismann's 69 % (theirs is data volume per application, not per pair; stated as an approximation) | cited: Eismann et al., IEEE Software 2021 (69 % of apps under 10 MB); platform caps (Step Functions ≤ 256 KB direct, Lambda 6 MB); heavy tier = assumed and swept |
| W3 Access links | every link 1000 MB/s | per-client class drawn once per topology: wired 117 MB/s; Wi-Fi 7–14 MB/s; cellular 4 MB/s up and 75 MB/s down. Mix 40 / 40 / 20 %. Core links unchanged. | weak: practitioner Pi benchmarks; one 5G measurement (Muzaffar et al., 2020) |
| W4 Task types | 2 occurring (`dnn1`, `dnn2`) | all 4 defined types, equal mix | HeROsim task profiles (Lannurien et al., IEEE Internet Computing 2024) |

All rates in MB/s (unit documented in `gate-tools.md`). Values are fixed in this node and will not change after any read.

**W4 precondition (code, before any W4 run).** The generator must guarantee that every task type has at least one
reachable eligible server in every topology (today `rf`/`cnn` can starve and hang). Verified by a static check over
all test and calibration topologies, recorded here before W4 runs.

## Design
- Physics: R1. Policy time scale fixed as frozen in R1.
- Provisional load: two multipliers found on the 4 held-out calibration topologies (protocol of
  `load_recalibration_v1`, max 8 bisection steps), targeting CD queue share ≈ 0.1 and ≈ 0.3 on the current workload.
  Final rungs come from `load_recalibration_v1` on WF1.
- Arms: classical only (CD, locality-first, one-pass greedy, self-predict, Knative as context). **No learned arms.**
- Stages: W2 → W2+W3 → W2+W3+W4. Each stage is read before the next runs.
- Output: WF1 = the final stage's configuration (committed hash), plus a parameter-provenance table for the paper,
  with every parameter marked trace / measured / cited / assumed-and-swept.

## Primary family
CD vs each other classical arm at each stage and provisional rung (Holm within stage).

## Predictions
1. CD remains first at every stage.
2. W2 makes exchange a small share of latency (< 10 %) for clients on wired links, but not on Wi-Fi/cellular links.
3. W3 widens the spread of per-task cost; locality-first's gap to CD grows (hop distance stops proxying cost).
4. W4 increases cold-start share (more types → more sandboxes).

## Outcomes
Rankings that survive all stages are reported as robust; flips are sensitivity findings. WF1 is frozen regardless
of which arm benefits.


## Record (newest first)

### 2026-10-08 — W3 (access-link classes) on R1.1: the stage read

Job 843086, code `rp/wf1-w3-r11` `ebd673e9`, 760/760 runs, 0 failed, 0 hung, **0 request failures**. Read by S4's reader
(`0ebeca16`) through S6's wrapper (`rp/wf1-w3-r11-read` `79d381ef`): [`w23_r11_read.json`](workload_fix_v1/w23_r11_read.json).
Paired % vs CD, median over 19 topologies, Holm over 6:

| rung | CD latency | self-predict | locality | batched | reactive (context) |
|---|---|---|---|---|---|
| light ×0.2666 | 4.29 s | +11.9 % (0/19, CD-FASTER) | +2.4 % (0/19, CD-FASTER) | +2.4 % (0/19, CD-FASTER) | +22.1 % |
| heavy ×11.61 | 0.94 s | −2.9 % (16/19, Holm 0.0039, DIRECTION) | +22.4 % (0/19, CD-FASTER) | +22.8 % (0/19, CD-FASTER) | +120.4 % |

- **Prediction 2 holds** (pooled descriptive): transfers between two wired servers cost about 0.25 s each and come to 1.3 / 2.6 %
  of summed task time; any non-wired pair costs 2.5–3.0 s per transfer (cellular–cellular 5–6 s) and comes to 49 / 65 %. The
  denominator is summed elapsed time, not a strict share, because a task can make several transfers.
- **W3 makes exchange the dominant cost.** CD at light: exchange 1.99 of 4.29 s per task, rendezvous 1.75. At heavy: exchange 0.52
  of 0.94. W3 minus W2 (paired): every arm is slower on 19/19 topologies, CD least (+70 / +90 %), reactive most (+106 / +289 %).
- **Against W2:** self-predict's heavy-load lead over CD shrinks from −6.1 % (CONFIRMED) to −2.9 % (DIRECTION), and it loses
  +11.9 % at light. The batching arms fall 22 % behind CD at heavy. CD's joint search gains the most from the new link
  heterogeneity.
- Wired+wired share of each arm's transfers: 21–25 % at light and 23–35 % at heavy (reactive 22.7 %, self-predict 35.4 %, CD 31.3 %).
  Whether any arm routes around slow links needs the share a random placement would give; requested.

### 2026-10-08 — W2 on R1.1: the stage read (window 1 s)

Job 843070, code `rp/wf1-r11` `1021c310` (on `rp/starve-gate` `252b45aa`): 760/760 runs, 0 failed, 0 hung, 0 missing, no
reruns; **0 request failures** in every arm × rung group. Read: [`w2_r11_read.json`](workload_fix_v1/w2_r11_read.json)
(with the request-failure column; reader `0ebeca16`), [`w2_r11_request_failures.json`](workload_fix_v1/w2_r11_request_failures.json).
Paired % vs CD, median over 19 topologies, Holm over 6:

| rung | CD latency | self-predict | locality | batched | reactive (context) |
|---|---|---|---|---|---|
| light ×0.2666 | 2.36 s | +2.4 % (0/19, CD-FASTER) | +0.7 % (0/19) | +0.1 % (1/19) | +2.5 % |
| heavy ×11.61 | 0.50 s | **−6.1 % (19/19, Holm 2.3e-5, CONFIRMED)** | +1.3 % (5/19, Holm 0.0013, CD-FASTER) | +0.5 % (4/19, Holm 0.032, CD-FASTER) | +5.6 % |

This replaces the provisional 1 s read. Only locality at heavy moved (+1.5 → +1.3 %), as the 2/200 leak sample predicted.
**W2 under R1.1: at light load CD leads, and every other arm sits within 2.5 %. At heavy load self-predict beats CD by 6 % on
every topology, and the batching arms trail CD by about 1 %.** CD by stage (s per task; latency / batching wait / queue /
exchange / rendezvous / cold): light 2.362 / 0.035 / 0.182 / 0.075 / 1.745 / 0.211, where rendezvous (waiting for a partner
to arrive) dominates; heavy 0.498 / 0.011 / 0.188 / 0.051 / 0.037 / 0.033. W3 (S6) and W4 (S7) are read next, in order.

### 2026-10-08 — WB2: window 1 s; W2 re-read; the hang's three causes; decisions for R1.1

**WB2 tuning** (CD, calibration 9601/9602/9607/9608 × 2 rungs × 4 windows, W2; 160 runs, 0 hung). Median CD latency
light / heavy, geometric mean: 1 s 2.772 / 0.541, 1.225; 2 s 2.774 / 0.547, 1.232; 4 s 1.294; 8 s 1.513; 16 s 2.153.
**Window = 1 s** for CD, batched and locality at every rung and stage. No hang touches the top two. It's the grid edge;
the batching wait left at 1 s is 0.035 s (light) and 0.011 s (heavy), about 1.5 % and 2 % of CD's latency, which bounds
what a smaller window could still gain. Report: [`wf1_wb2_report.md`](workload_fix_v1/wf1_wb2_report.md).

**W2 at 1 s** (456 batching-arm runs, 0 hung; reactive and self-predict from the first run):

| rung | CD | self-predict vs CD | locality | batched | reactive (context) |
|---|---|---|---|---|---|
| ×0.2666 | 2.36 s | +2.4 % (CD-FASTER) | +0.7 % (CD-FASTER) | +0.1 % (CD-FASTER) | +2.5 % |
| ×11.61 | 0.50 s | −6.1 % (19/19, CONFIRMED) | +1.5 % (CD-FASTER) | +0.5 % (CD-FASTER) | +5.6 % |

CD is first at the light rung, and self-predict at the heavy one. Self-predict's lead exceeds the bound on what a smaller
window could give CD. CD at 1 s: batching wait 0.035 s, rendezvous 1.75 s (light). The light-rung order moved with the
window (60 s −14.9 %, 16 s tie, 1 s +2.4 %). **Provisional:** the free-pool leak below may affect completed runs,
so this read stands only if the R1.1 identity check shows the W2 cells unchanged.

**The hang's causes** (S5, `rp/starve` `2b701c73`, single local cells, log-capped):
1. **Free-pool leak.** `create_first_replica` in both GNN-family autoscalers (KPA's shared autoscaler, so every arm)
   swapped `available_resources` for a filtered dict across a `yield`. Overlapping calls restored each other's
   copy, and nodes dropped out of the free pool for good. Fixed: `scale_up(reachable_nodes=…)`, no swap.
2. **Cross-source drain cycle.** A drain released only rendezvous peers matching its own source. Fixed: it also
   releases starved same-type peers from other sources, and skips victims whose tasks it can't release.
3. **Mixed-type hold-and-wait (open; W4 9565 g2).** Draining platforms hold tasks waiting on unplaced peers of
   other types that need those platforms.
4. **Infeasible cells.** On the live topology, 9603–9607 and test topologies 9484, 9548, 9568 each have a client with
   no reachable compatible server for some type. The static check validated a different generator
   (`generate_deterministic_infrastructure`), not the live one. 9603 under Knative spun silently (710k deferrals).

Also fixed: batch-decoded targets reserved against eviction (a latent race); per-(type, source) rate limit on the
starvation lines (log 487 MB → 10 KB); a task that can never reach a compatible platform raises `StarvedForeverError`.
W2-WB 9565 g2 CD now completes 50,000 events. 11 tests (`tests/test_starved_eviction.py`). Eviction succeeded 3
times in 14,770 calls before the fix; A1/A3 protection played no part. **Audit gap:** I7 returns NOT-TESTED for a run with
no end row, so a hung run was never judged; no invariant checked pool conservation.

**Decisions (coordinator, 2026-10-08; R1.1, before any W3/W4 read):**
- **Cause 3: a request timeout of 300 s** (Knative's default revision `timeoutSeconds`), from the moment a task is
  placed on its platform. A timed-out task fails, frees its platform and is logged. It enters latency at its elapsed
  time, and failures per arm are reported next to every latency. Chosen because it's the mechanism the modelled system
  has, and it breaks any hold-and-wait cycle without choosing which one. A rendezvous that doesn't hold the platform
  would be a new physics; un-draining doesn't free capacity for other types.
- **Feasibility rule for cells.** A (topology, window) cell is infeasible if, **on the live topology**, a client the
  window uses can't reach a compatible server for a type it sends. Infeasible cells are excluded for every arm (paired)
  and listed; the static check moves onto the live generator.
- **R1.1 = R1 + the fixes above + the timeout.** It's accepted only if (a) the default path is identical on the 36 audit
  and 6 legacy cells wherever no starved state or timeout occurs, and (b) a sample of completed W2 cells is identical.
  If (b) fails, every R1 number since `kpa_scaleout_v1` is re-measured. Audit pass 3 reruns I1–I12 on R1.1, with I7
  judging an unfinished run as FAIL, and a new **I13 pool conservation** (free + owned + draining = platforms, per node,
  at every KPA tick).

**Identity result and amendment R1.1-T (coordinator, 2026-10-08, after S5's identity data; `rp/starve` `8e835eef`).**
- (a) passes: 42/42 identical (36 audit + 6 legacy: physics-old on 9483, 9506, 9568 at ×20, CD and reactive); the
  timeout fired 0 times; 7 CD cells reach the starved path and are identical. Audit pass 3 on the final code: I1–I4,
  I6, I7, I9, I10, I13 PASS 36/36; I8 12/12; I11 216 states, worst p95 0.1 %; I12 12/12; I5 as in pass 2.
- (b) fails, for two separate reasons, over 200 W2 cells (5 topologies × 5 arms × 2 rungs × 4 windows): 123 identical.
  **Timeout, 75 cells** (every arm, ×0.2666, g1–g3): 105–136 failed tasks per cell, the same for every arm, all
  tasks whose partner *arrives* more than 300 s later (137 such tasks in g1; up to 1,095 s, the rung's 3.75× time
  stretch). Mean latency drops 45–49 %. **Leak, 2 cells** (9550 g2 ×11.61, batched −1.0 %, CD +0.41 %): one drain
  release per cell landed during the old filtered-pool swap; restoring only the swap reproduces the old numbers bit for bit.
- **Amendment R1.1-T: the timeout covers a wait on a partner that has arrived, never on one that hasn't.** The timeout
  exists to break hold-and-wait, which needs both tasks in the system; a partner not yet arrived holds nothing. Failing
  those 137 tasks measured the workload's stretch, not any arm (identical counts across arms), and erased the wait
  every arm pays. The clock starts at the later of placement and the partner's arrival. Recorded before the
  rescoped code exists; it's judged by the same gates: the 75 timeout cells must return to identical, W4 9565 g2 CD
  must still complete, and the 42 identity cells must stay identical.
- **The leak triggers the pre-registered rule, and it stands:** every R1 number since `kpa_scaleout_v1` is labelled
  pre-R1.1 and re-measured on R1.1. Expected size from the sample: 2 of 200 cells, ≤ 1 %; that expectation doesn't
  replace the re-measure. Order: W2 at 1 s on R1.1 first (it replaces the provisional W2 read), then the
  `kpa_scaleout_v1` gates. `physics_audit_v1` needs no rerun (42/42 identical; pass 3 ran on the final code).
- **Live-topology feasibility (S4, `rp/wf1-run` `7437a0f1`, `workload_fix_v1_reachability_live.py`):** built through
  `prepare_infrastructure_for_real_simulation` (no replica plan, no repair). Infeasible: 9603, 9604, 9605, 9606, every
  window (one sending client per topology reaches no dnn2-capable server platform; the dnn repair counts the client's
  own rpi). All 19 test topologies pass, 76/76 cells; calibration 9601, 9602, 9607–9610 pass. S5's other unreachable
  clients (9484, 9548, 9568, 9607, 9610) never send. **Decision:** the rule applies as written; 9603–9606 stay out,
  no repair is added to the live generator (it would change topologies, and the calibration set needs none of them).
  A pass means no structural block, not that a run completes: the 9565/9538 starvation is dynamic, and that's what
  the timeout handles.
- **Scope of the re-measure (coordinator, 2026-10-08).** Its question is whether the published numbers survive the
  R1.1 code, so each gate reruns at its **published settings**, with only the code changed: the `kpa_scaleout_v1` gates
  keep the per-rung policy time scale (0.5 / 0.33 / 0.2), and the result pairs directly with `kpa_read.json`.
  Measurements at a time scale of 1.0 on the new ladder belong to `load_recalibration_v1`, not here. Outputs go to new `*_r11`
  directories with `OUT` passed explicitly; the published directories are never written. Order:
  W2 at 1 s on R1.1 (`rp/wf1-r11` `1021c310`), then `gate_pipe_release_kpa` (R1's condition), then the other three
  conditions, then the four `*_legacy_shared` controls (they use the changed GNN-family autoscaler). The `legacy_replay_*`
  identity checks are not rerun, because the legacy path's cells were identical.
- **R1.1 ACCEPTED (2026-10-08), at `rp/starve` `665c9142` / `rp/starve-gate` `252b45aa`.** With R1.1-T, the 42
  audit/legacy cells are identical to `d4aeb7ca` (0 timeouts). 198 of 200 W2 sample cells are identical, including all 75
  cells the first timeout moved. The 2 that differ are the leak cells (9550 g2 ×11.61: batched −1.00 %, CD +0.40 %, 0
  failures). W4 9565 g2 CD completes 50,000 tasks with 5 failed requests. The re-measure above proceeds on this code.

### 2026-10-08 — W2 rerun under amendment WB; amendment WB2 (tuned window); the starved-replica hang

Builder `b401b656` (`--batch-timeout-fixed 16`). Arms that read `batch_timeout`: CD, batched, locality (one
peer-group batching scheduler, `gnn/orchestrator.py`). Reactive and self-predict never read it: 48 spot-check runs
are identical on every field. The ladder-window CD rerun (152) reproduces the first run exactly. Report:
[`wb_report.md`](workload_fix_v1/wb_report.md), read: [`wb_read.json`](workload_fix_v1/wb_read.json). 453/456 finished; **3 hung**
(CD and batched on 9565 ×11.61 g2, locality on 9538 ×11.61 g2). The same cells finished under the 1.4 s window.

| rung | window | CD latency | batching wait | rendezvous | self-predict vs CD | locality | batched |
|---|---|---|---|---|---|---|---|
| ×0.2666 | 60 s (ladder) | 2.85 s | 0.70 | 1.58 | −14.9 % | +0.6 % | +0.1 % |
| ×0.2666 | 16 s | 2.42 s | 0.15 | 1.69 | +0.2 % (not separated; CD-FASTER without the hung topologies) | +0.7 % | +0.1 % |
| ×11.61 | 1.4 s (ladder) | 0.50 s | 0.02 | 0.04 | −6.1 % | +1.4 % | +0.9 % |
| ×11.61 | 16 s | 1.35 s | 0.93 | 0.01 | −67 % | +0.5 % | +0.5 % |

**The W2 ranking depends on the batching window, in both directions.** The 60 s window cost CD its light-rung place,
and 16 s costs it the heavy rung. Self-predict's lead is real at the ladder's heavy window (−6.1 %, 1.4 s), the best
window CD has had at that rung.

**Amendment WB2 (coordinator, 2026-10-08, post-data, labelled; supersedes WB's value, keeps its rule).** The
window is a hyperparameter of the batching policies, not a platform constant, and the W2 ranking turns on it. The bar
is CD at its own best single wall-clock window. That value is chosen **once**, on the calibration topologies (9601,
9602, 9607, 9608; never the 19 test topologies), from {1, 2, 4, 8, 16} s, at both provisional rungs, 4 windows, on
the W2 workload. It minimises the geometric mean over the two rungs of CD's median latency; a hung run counts as
infinite latency. The same value serves batched and locality (same scheduler) and every later stage, and any later
learned arm that batches gets the same protocol. The W2 family is re-read with it; the 60 s and 16 s reads stay
recorded as run.

**Starved-replica hang (blocks W4; an R1 defect).** Every stuck run (W2-WB: 3; W2+W3-WB: at least 1 so far; W4: most
of the unfinished cells, under both windows; calibration ids 9603–9606) loops in `create_first_replica()`: "No
compatible hardware available for <type> on nodes with connectivity to <client>". It logs about 900 lines per
simulated second while the clock creeps or stops. The static reachability check passes these topologies, so the cause
is runtime capacity: no free eligible platform, and nothing frees one. The W4 logs reached ~45 GB and threatened the
`/home` quota. The coordinator cancelled S7's four W4 jobs (main pass 710/760 summaries written) and cut 41 logs to
1 MB head + tail excerpts. `physics_audit_v1`'s I7 passed because no audit cell reached this state. The W4 stage
waits for a diagnosis and fix, and a fix to R1 code needs the audit's default-path identity check.

### 2026-10-08 — stage W2 read; calibration set; provisional rungs; amendment WB (batching window)

Run code `f22b2e7e` (`rp/wf1-run`; `src/` identical to the replay-tested `f44cf9cf`), R1 with
`GATE_FIXED_POLICY_TIME_SCALE=1.0`. Default path: 5 R1 cells identical to `690ba384` on every simulated field; the
legacy build chain reproduces `grounded_x20` byte for byte. Report: [`w2_report.md`](workload_fix_v1/w2_report.md),
read: [`w2_read.json`](workload_fix_v1/w2_read.json).

**Calibration set: 9601, 9602, 9607, 9608.** 9603 and 9604 time out on every cell at ×2 for CD and Knative ("No
compatible hardware available" for one client). That's a runtime capacity hang that the static W4 check passes. Replaced
by the first ids from 9605–9610 that finish all 8 screen cells; 9605 and 9606 hang too (4 of 10 minted ids). The screen
covers ×2 only.

**Provisional rungs** (CD median queue share, 4 calibration topologies × g0/g1, legacy payloads): **×11.61 → 0.3007**
(8 steps); **×0.2666 → 0.1001**. The registered bracket [×0.5, ×24] did not reach 0.1 (0.117 at ×0.5), so a labelled
8-step extension on [×0.1, ×0.5] found it. On W2 payloads CD reads 0.09 / 0.37, as expected of provisional rungs.

**Stage W2** (`wf1_v1` payloads: mean per pair 27.2 MB against 429.2, median 4.8 MB, 70 % of pairs < 10 MB; pair graph
and events unchanged): 19 topologies × 2 rungs × 4 windows × 5 arms, 760/760, 0 failed.

| rung | CD latency | self-predict vs CD | locality vs CD | one-pass greedy vs CD | reactive (context) |
|---|---|---|---|---|---|
| ×0.2666 | 2.85 s | −14.9 % (19/19, Holm 2e-5) | +0.6 % (CD-FASTER) | +0.1 % (CD-FASTER) | −14.9 % |
| ×11.61 | 0.50 s | −6.1 % (19/19, Holm 2e-5) | +1.4 % (CD-FASTER) | +0.9 % (CD-FASTER) | +4.4 % |

Prediction 1 (CD first at every stage) **fails at W2**: self-predict is first at both rungs. Exchange is 3.9 % / 10.5 %
of CD's latency; the wired/wireless split of prediction 2 needs W3.

**Amendment WB (coordinator, 2026-10-08, after the W2 read: post-data, labelled).** The ladder protocol scales the
cells' `batch_timeout` with the rung: 16 s × 3.75 = **60 s** at ×0.2666 and 1.4 s at ×11.61. That's a 43× change in a
policy constant across rungs. B1 (`physics_audit_v1`) froze one set of policy constants in wall-clock seconds on every
rung, and the batching window is one of them; I12 didn't list it, so it slipped through. CD's latency is 2.85 s at the
light rung against 0.50 s at the heavy one, and reactive (no batching) ties self-predict there, so the 60 s window
plausibly explains much of CD's light-rung loss. For WF1 and every later node, **`batch_timeout` is 16 s (its ×1 value) at
every rung**. The fix isn't obviously in CD's favour: at ×11.61 the window grows from 1.4 s to 16 s. The W2 read above
stays recorded as run. The batching arms (CD, one-pass greedy, and any other arm that reads `batch_timeout`) are rerun on
W2, W3 and W4 with the fixed window; arms that don't read it stand. The W2 primary family is re-read on the rerun, with a
decomposition of CD's latency (batching wait, queue, exchange, cold start) under both windows.
