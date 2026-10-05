# client_local_v1 — may a call run on its own client, and does the origin model matter?

**Status:** `ACTIVE` (2026-10-05) — part A (rules, scattered origins) read on six of seven rules, local-first rerunning;
part A2 (learned arms zero-shot) running; part B (single-origin groups) running. Exploratory: no bars were registered before the runs.

**Outcome so far.** With every client hosting one replica per function and allowed to run its own calls, the five rules
read so far ran **1.3–1.6 % of calls locally at ×2, 0.4–0.7 % at ×3 and 0.1–0.3 % at ×5**, and enabling it **helped
none of them**: Knative ties its server-only run at every rung, and CD, locality-first, self-predict and the one-pass
greedy are 1–5 % *slower* (CD +4.8 % at ×2, p = 0.002). **That reading is conditional on a workload defect found the
same day:** the grounded mint gives each task its own client, so **97 % of multi-task peer groups span several
clients** (3.56 distinct clients per 4.09-task group), whereas in the Alibaba trace a group is one request from one
caller. Running a call locally therefore usually separates it from its partners. Part B re-runs the rules with one
origin per group. **Every grounded-workload result before 2026-10-05 was measured under scattered origins** — not
wrong, but to be disclosed wherever it is quoted.

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
