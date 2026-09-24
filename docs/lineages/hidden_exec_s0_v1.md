# hidden_exec_s0_v1 — does hiding execution time from the rule leave headroom?

**Status:** `REGISTERED` (2026-09-24). Bars below signed before the gate's data; one smoke
environment was read first and is disclosed.

## Why this lineage exists

faas-sim (Raith et al. 2023, `edgerun/faas-sim`, read 2026-09-24) gives a function a
per-invocation lognormal execution time per (device, image), a same-replica concurrency factor, and a
co-location slowdown predicted by a regressor over the resource usage of every call overlapping the
execution. Its scheduler samples its own independent draw and never sees the realized value or the
slowdown. HeROsim reads a fixed `executionTime[platform]` table (`data/nofs-ids/task-types.json`, one
number per type × platform, no recorded provenance) in the engine AND in every policy, so every rule
here has perfect execution-time knowledge. The question: if the physics hides execution time,
does a rule that knew it gain enough that a learned arm could have something to recover?

What this cannot reopen: faas-sim's slowdown is a function of what is co-resident on the node,
which is node-indexed, so `docs/hard-stops.md` (co-tenancy interference, count theorem) already
closes it for GNN-vs-MLP. Hiding it changes what a *rule* knows, not what a model class can
express. This lineage asks only the learned-vs-rule question, and asks it first as a headroom screen.

faas-sim has no co-sim → train → evaluate pipeline to port (its only ML is the offline degradation
regressor, trained on real testbed data), so the physics idea is the only thing borrowed.

## Physics (`src/placement/exec_physics.py`)

`HEROSIM_EXEC_PHYSICS=table_v0` (default, bit-identical: selfpredict on `cc40s9001 w0` reproduces
the `selfpredict_bar_v1` gate's `total_rtt` 706132.6793659842 to the digit) or `hidden_node_v1`:
realized = table × node speed × (1 + β × other platforms executing on the node at start) × noise.
Node speed is a mean-one lognormal (σ 0.5), β ~ U(0, 0.5), and noise is a mean-one lognormal (CV 0.2),
with every draw a hash of `HEROSIM_EXEC_SEED` (= the topology seed) and its key. Synthetic magnitudes,
chosen before data (no traces). Policies keep reading the table; `HEROSIM_PG_EXEC_KNOWLEDGE=oracle`
lets the peer-greedy rules read the node constants and current co-execution for their own exec
term and for each queued task in the drain estimate.

## Bars (signed 2026-09-24, before the gate's data)

Arms, fresh at one commit, under `hidden_node_v1` on the 16 + 16 `unsaturated_edge_v1`
environments (6 servers, 0.460393 arrivals/s, the `selfpredict_bar_v1` serving env verbatim):
`table` = `peer_greedy_selfpredict_network`; `oracle` = the same plus
`HEROSIM_PG_EXEC_KNOWLEDGE=oracle`. Paired per (rung, env) on `total_rtt`, Δ = (oracle − table)/table.

- **H (headroom), per rung:** `HEADROOM` iff median Δ ≤ −5 % and Wilcoxon p < 0.05 (n = 16);
  `ORACLE-SLOWER` iff median Δ ≥ +5 % and p < 0.05; else `NO-HEADROOM`.
- **Verdict:** `GO-P0B` (build the `estimate` arm: EWMA of observed completions, the model's
  information, and register its recovery bar) iff `HEADROOM` on both rungs; otherwise
  `NO-HEADROOM`, the lineage closes at S0, and nothing is trained.
- Disclosed: what the hidden physics costs the rule (`table` here vs the same arm under
  `table_v0` in the `selfpredict_bar_v1` gate), and mean realized exec per task.

Disclosed before data: (1) execution is **0.058 s of a 14.12 s per-task latency (0.4 %)** in
this regime (selfpredict, `cc40s9001 w0`), and exchange (3.0 s) and queue (7.3 s) dominate, so a
large headroom would need magnitudes far above these; (2) smoke on `cc40s9001 w0`: hidden physics
raises exec 0.058 → 0.084 s and `total_rtt` +0.46 %; `oracle` vs `table` **+0.53 %**; two repeats
identical to the digit.

## Entry points

- Physics: `src/placement/exec_physics.py`; engine hook `Platform._execute` in
  `src/placement/infrastructure.py`; oracle read `_PeerGreedyCore._pg_xf` in
  `src/policy/peer_greedy_network/scheduler.py`; `exec_scale` in `live_audit.platform_queue_drain_seconds`.
- Gate + reader: `scripts_cosim/hidden_exec_s0_v1_gate.sh`, `scripts_cosim/hidden_exec_s0_v1_read.py`.

## Record (newest first)

- 2026-09-24 — **Registered.** Physics built, table_v0 bit-identity and seed determinism checked.
