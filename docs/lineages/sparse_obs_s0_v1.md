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
- Stage 2: exec, warm, partners = 6 arms × 24 = 144 cells (g0). Walls of the `cd_exactS_ib0` class (about 3 ks median): about 0.4 Mcore-s per 120 cells; one 24-shard array each, about 1 h.
- Queue: at most 24 of mine. The topologies are new, so their inputs (`wf1_*`) must be minted first (one CPU job, as `ib0_confirm_v1` did).

## Rulings applied (2026-10-12) and how they were implemented

- (a) Stage 1 at g0 only: A0 + NAIVE/EC for queue and BUNDLE = 120 cells. Stage 2 (exec, partners, warmth) after the stage-1 read.
- (b) Topologies 19440–19451 (`scripts_cosim/sparse_obs_s0_v1_selected.json`), disjointness re-checked at launch.
- (c) A0 stays the batch exact search. BUNDLE arms search the visible batch exactly with partners known only once placed (`planned_node_name` / `platform`); the batch's own plan is invisible.
  EC prices a partner with no known node at the node of its bytes-heaviest known partner (the lookahead rule's two-step estimate, `PeerGreedyLookaheadNetworkScheduler._pg_predict_node`). **Deviation:**
  the event-index self-predict (`PeerGreedySelfPredictNetworkScheduler`) needs the per-arrival orchestrator's event list and is not available inside the batched CD search, so EC uses the two-step estimate. The seat gap (batch search vs per-arrival serving) is reported with the result.
- (d) BUNDLE degrades row 2 to "running or not": NAIVE charges nothing, EC charges half the running task's type mean service. **Disclosed:** A0 does not read row 2 at all (`HEROSIM_PG_INFLIGHT` is unset for `cd_exactS_ib0`), so for NAIVE this channel is a no-op against A0 and for EC it ADDS a term A0 lacks. `HEROSIM_OBS_INFLIGHT` fails loudly if `HEROSIM_PG_INFLIGHT=1`.
- Warmth (row 4) stays exact in stage 1 and in the bundle; the `HEROSIM_OBS_WARM` flag is a stage-2 item and is not implemented yet.
- Exec: one estimator for both arms (table value × the seeded log-normal factor per (task type, platform type)); NAIVE/EC differ only in queue, in-flight and partners.

## Implementation (flags default off, `src/policy/peer_greedy_network/observation.py`)

`HEROSIM_OBS_ESTIMATOR=naive|ec` (required with any channel), `HEROSIM_OBS_QUEUE=count`, `HEROSIM_OBS_EXEC_SIGMA=<s>`, `HEROSIM_OBS_INFLIGHT=running`, `HEROSIM_OBS_PARTNERS=arrival`, `HEROSIM_OBS_SEED` (the cell seed).
With none set `Observation.from_env()` is None and every score site runs its registered arithmetic (tests/test_sparse_obs.py compares the flag-off terms to the registered functions). Sites: `_pg_choose`, `_pg_plan_cost`, `_pg_standalone`
(drain, running, exec) and `_pg_peer_nodes` (partners); the exec channel also reaches the exact queue walk (`live_audit.platform_queue_drain_seconds(exec_of=...)`, default None = unchanged).
Books (`schedulerCounters`): `obs_drain_degraded`, `obs_drain_exact`, `obs_exec_degraded`, `obs_inflight_degraded`, `obs_partner_hidden`, `obs_partner_predicted`. The gate checks that a degraded channel's book is non-zero, that the exact queue walk
is never used by a queue-degraded arm, and that every other arm's books are zero. Arms: `cd_obs_queue_{naive,ec}`, `cd_obs_bundle_{naive,ec}` (stage 1), `cd_obs_exec`, `cd_obs_partners_{naive,ec}` (stage 2).

## Rules, written before data

- Statistic: the paired % of mean latency per topology (median over windows), median over topologies, exact Wilcoxon, Holm within VoI and within R (the `prune_g8_gate_read.py` shape). Negative = the first arm faster; here "VoI" = NAIVE slower than A0, reported as a positive loss.
- **Kill:** VoI < 10 % **or** R < 5 % at both rungs for every D and the bundle → `CLOSE HAND-RECOVERS-UNDER-SPARSITY`.
- **Pass:** goes to Phase 1 (registered successor, not part of this node): GNN, MP-OFF twin and a hand-feature MLP trained on degraded observations with true-RTT labels, vs the EC hand rule; WIN ≥ 3 % at both rungs; an MP claim only against the twin; a learned arm that beats A0 is treated as an information leak and audited.
- **Disclosed up front:** a pass most likely yields a learned-scorer result, not an MP result. This seat is more realistic than the current one (it removes an over-informed baseline). Warmth hidden is a disclosed arm, not the realistic one.
- **Prediction (SCIENTIST's priors):** BUNDLE VoI ≥ 10 % at heavy ~40 %; R ≥ 5 % ~20 %.
- Flag-off identity: a test per flag that the scheduler's scores are unchanged with the flag off; the live gate adds a 2-cell identity check at the pinned head.

## Record
