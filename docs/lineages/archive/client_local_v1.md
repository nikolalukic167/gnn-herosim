# client_local_v1 — may a call run on its own client, and does the origin model matter?

**Status:** `CLOSED` (2026-10-07) — **CLIENT-EXECUTION-NO-GAIN / ORIGIN-MODEL-MATTERS**. Exploratory: no bars were
registered before the runs; the follow-up retrain was registered as `small_batch_so_v1`.

**Outcome.** (1) **Client execution does not pay.** Only 2–7 of the 20 issuing clients can host `dnn1`/`dnn2` at all and
the autoscaler scales those replicas away, so every policy ran 0.1–2 % of calls locally; client-enabled cells are 2–15 %
slower than server-only for the rules (ties at ×5) and within −2 to +4 % for the learned arms, and local-first equals
Knative. (2) **The origin model matters a great deal.** The grounded mint scatters a peer group over clients (97 % of
multi-task groups; a trace request has one caller). With one origin per group every rule runs 26–82 % faster, CD remains
the strongest rule (locality-first and the one-pass greedy tie it at ×5), and the learned arms trained on scattered
origins lose their lead: `sb1load` vs CD +12.2 / +3.8 / −7.0 % (×2 CD faster, 0/19), still −41 / −54 / −65 % vs Knative;
`lf1gnn` vs CD +21 / +18 / +16 % while beating `lf1mlp` −11 / −14 / −22 %. Retraining on single-origin groups does not
recover the win (`small_batch_so_v1`). **Every grounded-workload result before 2026-10-05 was measured under scattered
origins** — name the origin model wherever one is quoted. (3) Running single origin needed three simulator fixes, each
verified replay-identical on every previously completed run (119–120 / 119–120): the frozen-clock deferral loop, a
starved-type eviction (idle, then drain), and release of a rendezvous on a starved peer.

**Question.** Every grounded gate ran with `HEROSIM_SERVER_ONLY_REPLICAS=1` and `replicas.*.per_client = 0`, so every
call was offloaded. Is that hiding a gain from running a call on its own client (the own-device-or-offload choice of the
edge-offloading literature)? And is the scatter of a group's tasks across clients a faithful model?

**What changed in the simulator** (all opt-in; existing cells replay bit-identically):
- `network.backbone.exchange_routes: all_pairs` (`src/generate_infrastructure.py`, `build_core_backbone`) adds a
  Dijkstra backbone route for every node pair to `routes` only; `network_maps`, and with them which replicas a task may
  use, are unchanged.
- The peer exchange charges that route's latency when two nodes share no logical edge
  (`Platform.peer_link_latency`, `src/placement/infrastructure.py`; CD's estimate in
  `src/policy/peer_greedy_network/scheduler.py` uses the same call).
- `local_first_network` (`src/policy/offload_network/scheduler.py`): the source client when it hosts a replica, else
  shortest queue. `offload_network` now sorts its candidates (it iterated a set).
- GNN serving: routes are stored one way (client → server); `route_hops_and_bottleneck`
  (`src/placement/dag_workload.py`) and the exchange-latency column in `src/policy/gnn/prefix_serving.py` now read them
  the way the fabric already charges them (reverse path; backbone-route latency). Only lookups that used to raise
  change. Tests: `tests/test_peer_exchange_cost.py`.

**Cells.** `small_batch_confirm_v1`'s 19 topologies × grounded windows g0–g3 × {×2, ×3, ×5}, each changed in three ways
only: `replicas.<type>.per_client = 1`, `exchange_routes = all_pairs`, `"client_local_v1": true` (the gate then serves
`HEROSIM_SERVER_ONLY_REPLICAS=0`). Caveat: adding client replicas changes start-up seeding, so server replica placement
can differ from the server-only cell; a client-vs-server contrast is between environments.

**Workload facts found on the way** (x2 g0, `small_batch_confirm_v1/inputs/grounded_x20/wl`): only **20 of the 40
clients** issue tasks, and only **`dnn1` and `dnn2`** occur — the two types the topology generator guarantees a
reachable server (`src/generate_infrastructure.py`, compatibility pass); `rf` and `cnn` are not guaranteed for any
client, which is where the deterministic starved-client hangs (e.g. 9538 g2 ×2, CD and one-pass greedy) come from.

**Entry points.** Gates: `scripts_cosim/datalab/client_local_v1_gate.sbatch` (part A / A2, phase `cl1` of
`scripts_cosim/fresh_topo_burst_v1_gate.py`), `scripts_cosim/datalab/client_local_v1_so.sbatch` (part B). Workload
rewrite: `scripts_cosim/client_local_v1_single_origin.py`. Reader: `scripts_cosim/client_local_v1_read.py`. Data:
`simulation_data/client_local_v1/` on datalab (`gate/`, `gate_so_server/`, `gate_so_client/`, `wl_so/`).

## Record

- 2026-10-07 — **Part B read; CLOSED.** Rules on single-origin cells (server-only 1,138 / 1,140 runs, client-enabled
  1,596 / 1,596; the two missing are Knative and self-predict on 9485 g1 ×2, which time out at 2,700 s and again at
  10,800 s): single origin vs original −26 to −82 % for every rule, 18–19 / 19; client-enabled vs server-only under single
  origin +2 to +15 % (×2 / ×3, Holm REF-FASTER), ties at ×5; vs single-origin CD: locality-first +9.8 / +7.3 / +1.0 %,
  one-pass greedy +7.5 / +6.3 / −3.3 %, self-predict +52 / +54 / +8 %, Knative +90 / +126 / +151 %. Learned arms zero-shot
  on single-origin server cells (seeds 1–4, 1,824 runs per pair, 0 failed): `sb1load` vs CD +12.15 (0/19) / +3.79 (5/19) /
  −6.99 % (13/19); vs twin −0.1 / +0.4 / −1.4 %; `lf1gnn` vs CD +21.1 / +18.0 / +15.7 %, vs `lf1mlp` −11.1 / −13.6 / −22.1 %.
- 2026-10-06 — **Starved-type eviction, drain and rendezvous release** (`src/policy/gnn/autoscaler.py`
  `evict_idle_for` / `_release_when_drained` / `_release_starved_rendezvous`, `src/placement/infrastructure.py`
  `STARVED_RENDEZVOUS`; commits a4fe061c, 8fe4a51f, dd456b5d; `tests/test_starved_eviction.py`). Reached only from the
  starved-spin path. Idle-only eviction left one 9491 run starved (four `dnn1` replicas on the only reachable node were
  never idle); the drain then deadlocked on a `dnn1` task waiting for its starved `dnn2` partner, which the release
  resolves by planning that partner onto the drained node. Verified: 119 / 119 completed reference runs identical at each
  commit, 36 / 36 single-origin 9491 runs complete; a merged hidden_exec_s0_v1 commit later replayed 120 / 120 identical.
- 2026-10-06 — **Part A2 read: learned arms zero-shot on the client-enabled cells** (seeds 1–4, 910–912 / 912 runs
  each; same caveat as part A — clients barely host replicas, so this tests serving, not local execution). Median paired
  % over 19 topologies, ×2 / ×3 / ×5, exact Wilcoxon unadjusted. Vs client-enabled CD: `sb1load` −10.1 / −31.9 / −20.4 %
  (18–19 / 19), `sb1mpoff` −9.6 / −31.2 / −16.7 %, `lf1gnn` +14.3 / +35.7 / +8.9 %, `lf1mlp` +33.2 / +86.2 / +35.7 %.
  Vs client-enabled Knative: `sb1load` −41.6 / −85.9 / −39.0 %, `lf1gnn` −25.1 / −51.8 / −19.8 %, `lf1mlp` −14.3 / −24.4 /
  −2.8 % (×5 not separated). GNN vs twin: `lf1gnn` vs `lf1mlp` −12.3 / −30.7 / −18.4 % (18–19 / 19); `sb1load` vs
  `sb1mpoff` −0.6 / −2.4 / −0.6 % (not a message-passing win, as server-only). Vs each arm's own server-only runs: within
  −2 to +4 %. Local share 0.1–1.5 %. The server-only rankings carry over unchanged.
- 2026-10-06 — **Hang cause found and partly fixed** (`src/policy/gnn/scheduler.py`, 7b2098d2): a deferred task went
  straight back into the queue and the batch collector took it again at the same instant, so when no replica could be
  created the batch rules (CD, locality, one-pass greedy, GNNs) looped forever with the clock frozen. After 50 deferrals
  of one task at one instant it now waits 1 s (`HEROSIM_DEFER_RETRY_S`). Verified: 47 / 47 completed CD and locality runs
  replay to the digit; the old 9538 g2 ×2 CD hang completes; 4 of 7 single-origin 9491 CD hangs complete. The other 3
  are a real starvation: a new replica needs a platform hosting no replica and nothing evicts one, so under single
  origin `dnn1` replicas hold every `dnn2`-capable platform client 2 reaches and its `dnn2` tasks never run. Part B's
  server-only half lost 96–108 runs each of CD, locality and one-pass greedy to timeouts; not readable until resolved.
- 2026-10-05 — **Part A read, all seven rules (gate 828478 + local-first rerun 828629); it does not test local
  execution.** Local-first ran 1.42 / 0.39 / 0.13 % of calls locally, the same as Knative (1.41 / 0.35 / 0.11 %), and
  its paired % vs client-enabled CD equals Knative's at ×2 (+56.27 %) and ×3 (+290.81 %); at ×5 +37.7 vs +34.5 %.
  Cause, from the run logs: only 7 of the 20 issuing clients ever hold a replica on 9483 and 9502, 2 on 9550 (the
  others have no platform that runs `dnn1`/`dnn2`), and the autoscaler removes those (590 client-replica removals in
  9483 g0 ×2, 389 of them on client_node0). Always-offload vs CD: +46.2 / +768.4 / +195.4 %. Missing runs: five timeouts
  (9538 g2 ×2 CD and one-pass greedy; 9568 g1–g3 ×2 one-pass greedy, which completes server-only).
- 2026-10-05 — **Relaunch after three faults; nothing below had produced an outcome.** (1) `local_first_network` was
  registered in `simulation.py` but not in `model.scheduling_strategies`, so all 228 local-first runs died at start-up
  (KeyError); fixed, with `tests/test_policy_registry_names.py` checking every registered policy has a short name.
  (2) The home quota filled at 19:09 (three gates at 62 parallel, each holding ~250 MB raw results, plus training):
  runs ended with rc 120 (stdout flush failed) or silently; those records were removed and the runs requeued. The
  gate now writes raw results to node-local disk (`HEROSIM_RAW_DIR`, `docs/gates/gate-tools.md`). (3) The reader's
  twin loop shadowed its arguments; fixed. Learned-arm smoke on 9483 (seed 1): all 12 `sb1load` runs served on the
  client cells; the other failures in it were the quota. New hang caused by client execution: 9568 g1–g3 ×2 one-pass
  greedy times out with client replicas and completes server-only. Jobs: local-first 828629, learned 828630 / 828631,
  single-origin reruns queued after 828619.
- 2026-10-05 — **Part B launched: single-origin groups.** `client_local_v1_single_origin.py` moves every task of a peer
  group to the client of the group's earliest task; nothing else changes. At x2 g0 it moves 34,563 of 50,000 tasks;
  the (client, type) pairs are identical before and after, so no client issues a type it could not before. The rules
  run on the rewritten workloads twice — server-only cells and client-enabled cells — so the reader can separate "does
  the origin model move the rules?" (single-origin vs original, server-only) from "does client execution pay once a
  group has one origin?" (client-enabled vs server-only, single-origin). Jobs 828617 (build), 828618 (server), 828619
  (client).
- 2026-10-05 — **Workload defect: scattered origins.** `scripts_cosim/grounded_workload_v1_mint.py` (lines 11, 99) gives
  each task the client of the k-th base event. At x2 g0, 11,660 of 11,974 multi-task groups span clients, 3.56 distinct
  clients per group on average. The trace (`cluster-trace-microservices-v2021`) carries no client field; a trace id is
  one request, so one origin per group is the natural reading and the scatter was never a modelling decision. A
  literature check (pasted report, 2026-10-05) found multi-source tasks in collaborative edge computing and D2D helper
  offloading, but in both a group still has one requester; the scatter here matches neither.
- 2026-10-05 — **Part A2: learned arms zero-shot.** `sb1load`, `sb1mpoff`, `lf1gnn`, `lf1mlp` (seeds 1–4) served on the
  client-enabled cells, trained only on server candidates. The first two submissions crashed in GNN serving (one-way
  route, then exchange latency without a logical edge; fixed as above); 508 crashed records moved to
  `client_local_v1/crash_route_bug/`, not counted as outcomes. Smoke job 828621 (one topology, seed 1, 48 runs).
- 2026-10-05 — **Part A read, five of seven rules** (gate 828478; always-offload and local-first still running).
  Client-enabled vs the same rule server-only, median paired % over 19 topologies (exact Wilcoxon, unadjusted):
  reactive −0.1 / +1.3 / −1.2 % (ties); CD +4.8 % (p = 0.002) / +2.4 / +2.5 %; locality-first +3.1 / +2.0 / +4.2 %;
  self-predict +1.3 / +1.1 / +1.3 %; one-pass greedy +3.8 / +3.7 / +3.9 %. Local share 1.3–1.6 / 0.4–0.7 / 0.1–0.3 %.
  Within client-enabled, vs CD: locality-first −20.3 % at ×3 and −6.0 % at ×5, self-predict −12.3 % at ×3. The −48 %
  reactive gain seen on cell 9483 ×2 alone did not survive 19 topologies. Gate disk: hung runs write ~1.4 GB logs; a
  watcher kept excerpts and truncated logs over 300 MB.
