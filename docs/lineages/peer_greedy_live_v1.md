# peer_greedy_live_v1 — a hand rule with the learned arms' information, on the live gate

**Status:** `REGISTERED` (2026-09-20) — bars, reader, 7 tests, sbatch and the expectation below
committed before any rule arm ran on the gate. One local smoke cell was run first and is
disclosed under the expectation.

**Parents:** [`unsaturated_edge_v1`](unsaturated_edge_v1.md) (the environments, the baselines,
`gnnedge0`'s 16 checkpoints, the paired statistic) and
[`batch_window_edge_v1`](batch_window_edge_v1.md) (the decomposition: the learned arm's only
placement gain is −1.05 s of peer exchange per task from co-location; its deficit is the
7.1 s it waits to see a peer group). Offline ancestor: `peer_affinity_v1`'s B3/B5 greedy bars,
which a hand greedy on the true marginals passed in every cell — offline, never served.

**Question.** The learned arms have never been compared with a rule that uses the same
information. Two questions, one design: **(a)** does a two-line peer-aware rule that decides
per arrival — no group wait — beat reactive Knative where every learned arm loses? **(b)** served
exactly as the learned arms are (peer-group batching, 16 s window), does the rule match
`gnnedge0`'s placement, i.e. has the model learned anything the rule does not encode?

**The rule** (`src/policy/peer_greedy_network/scheduler.py`). For task i and candidate
replica (node n, platform p), in seconds, every term the physics charges:

    S(n, p) = drain(p) + cold(p) + exec_i(p) + latency(src_i → n) + X_i(n)

`drain(p)` is `platform_queue_drain_seconds` (the audit's estimator: each queued task's
execution, storage I/O, source latency and known-peer transfers). `X_i(n)` is the exchange the
simulator will charge at i's input stage for every peer whose node is already known — placed,
planned by a batch pre-pass, or committed earlier in the same batch — by the same
`Platform._payload_transfer_time` + latency, 0 for a co-located peer; unknown peers contribute 0.
Argmin over Knative's candidate set (reachable replicas, initialized preferred) with Knative's
tie-break. There is no free constant: a task joins a partner's node unless the queue there is
longer, in seconds, than the transfer it would save. `knative_network_ect` is NOT this rule: its
queue term is `len(queue) × exec` (~0.1 s per queued task, blind to the ~5 s of exchange each
queued task carries) and it reads +12.8 % behind shortest-queue on these environments.

| arm | stack | wait | what it isolates |
|---|---|---|---|
| `peer_greedy_network` | knative_network (per arrival) | none | the rule, item-1's competitor |
| `drain_greedy_network` | knative_network (per arrival) | none | the same with X ≡ 0: the exchange term's ablation |
| `peer_greedy_network_batch` | gnn (peer-group batching, `masked_topo` batch path, `planned_node_name` pre-pass) | 16 s | the rule in `gnnedge0`'s seat, greedy in task-id order with in-batch commitments |

**Design.** The 16 (topology, window) environments per rung that `unsaturated_edge_v1`
selected (`simulation_data/unsaturated_edge_v1/selected.json`), 6 servers, C40 primary and C80
secondary; 3 arms × 16 × 2 = 96 runs. Reactive, ECT, random and `gnnedge0` are the study's own
runs, unchanged. Same trace windows, same cell configs (`scheduler.batch_timeout` 16 s,
`batch_size` 10 asserted), same environment variables as the study sbatch.

**Bars** (`scripts_cosim/peer_greedy_live_v1_read.py`, chain's: |median| ≥ 5 %, p < 0.05,
two-sided signed-rank). A rule has no training draw, so a rule-vs-baseline read has one value
per environment: **unit = environment, n = 16**. A rule-vs-`gnnedge0` read keeps the chain's unit,
the checkpoint, by pairing the rule's value against every checkpoint on the same environment.

| read | what | fires as |
|---|---|---|
| **G1** | `peer_greedy_network` vs reactive (env) | `IMMEDIATE-RULE-BEATS-REACTIVE` / `REACTIVE-FASTER-THAN-IMMEDIATE-RULE` / `NOT-SEPARATED` |
| G2 | `peer_greedy_network_batch` vs reactive (env) | `BATCHED-RULE-BEATS-REACTIVE` / `REACTIVE-FASTER-THAN-BATCHED-RULE` / `NOT-SEPARATED` |
| **G3** | `peer_greedy_network_batch` vs `gnnedge0`, paired on env and checkpoint, RULE FIRST | `RULE-FASTER-THAN-GRAPH-ARM` / `GRAPH-ARM-FASTER-THAN-RULE` / `RULE-MATCHES-GRAPH-ARM` (the tie IS the "matches" verdict) |
| G4 | `peer_greedy_network` vs random (env) | `IMMEDIATE-RULE-BEATS-RANDOM` / … |
| G5 | `peer_greedy_network` vs `peer_greedy_network_batch` (env) | `IMMEDIATE-FASTER-THAN-BATCHED` / `BATCHED-FASTER-THAN-IMMEDIATE` / `NOT-SEPARATED` — the same score with and without the wait; predicts the no-wait decoder before it is built |
| G6 | `peer_greedy_network` vs `drain_greedy_network` (env) | `EXCHANGE-TERM-HELPS` / `EXCHANGE-TERM-HURTS` / `EXCHANGE-TERM-NOT-SEPARATED` |
| G7 | mechanism: exchange per task fell vs reactive on ≥ 12/16 environments; queue delta carried | `CO-LOCATION-DELIVERED` / `CO-LOCATION-NOT-DELIVERED` (descriptive) |
| G8 | composite on G1, carrying G3 and G5 | `HAND-RULE-BEATS-REACTIVE-WITHOUT-WAITING` / `…-TIES-…` / `…-LOSES-TO-…` |

A `gnnedge0` checkpoint that hangs (2 of 256 at C40, `unsaturated_edge_v1`) makes G3 UNREADABLE
by the letter; the read on the complete checkpoints is printed as **disclosed**, as before.

**Registered expectation (signed 2026-09-20, before the gate; one smoke cell seen).** The three
arms were smoke-tested locally on `cc40s9001` w0 before signing — the reactive reference
reproduced datalab's `total_rtt` to the last digit and a repeated rule run was bit-identical —
and they read, on that ONE cell, immediate −14.3 %, batched −7.5 %, drain-only +0.2 % vs Knative,
with exchange 4.71 → 3.66 s per task and queue 10.0 → 8.2 s. That is w0, the burstiest window,
on one topology; the record's own lesson is that a direction from one cell survives and a
magnitude does not. So: **G1 fires BEATS at 70 %** (NOT-SEP 25 %, REACTIVE-FASTER 5 %); G2
BEATS 45 % / NOT-SEP 40 %; **G3 RULE-FASTER 55 %**, MATCHES 35 %, GRAPH-FASTER 10 %; G4 fires;
**G5 IMMEDIATE-FASTER 85 %**; G6 HELPS 90 %; G7 DELIVERED 90 %. At C80 the same signs with
smaller margins.

**Consequences, signed in advance.**
- G1 fires: the first live win over a healthy reactive baseline in this programme is a
  two-line rule. The learned arms' claim reduces to "learns what the rule encodes, minus the
  wait"; the no-wait decoder (the next lineage) has the rule, not Knative, as its bar, and the
  paper's mechanism claim is the rule's decomposition.
- G1 does not fire and G7 is DELIVERED: co-location's gain is eaten by concentration even
  with no wait at all; the no-wait decoder is closed before it is built (`docs/hard-stops.md`).
- G3 RULE-FASTER or MATCHES: at matched serving the graph arm has learned nothing beyond the
  rule; "a graph model learns co-location" becomes "a graph model recovers a rule", a hard stop
  on the model-class claim at this operating point.
- G3 GRAPH-FASTER: the model encodes something the rule does not; the gap is the thing to
  name, and the no-wait decoder is worth building.
- G6 does not fire: the rule's gain, if any, is the drain estimator (a better shortest-queue),
  not co-location; G1 is not a peer-affinity result.

**Cost.** 96 runs; the per-arrival arms take ~1 min, the batched ~3 min. Two blocks of 48.
Script `scripts_cosim/datalab/peer_greedy_live_v1.sbatch`; read
`scripts_cosim/peer_greedy_live_v1_gate_read.py`; results under
`simulation_data/peer_affinity_live_gate/results/pg_v1/`.

**Datasets.** None; nothing trained.

## Record (newest first)

- 2026-09-20 — Registered. Policies, registry entries (`src/placement/simulation.py`,
  `src/placement/model.py`, `src/executesimulation.py`), the `pg_*` counter whitelist in
  `Orchestrator._scheduler_counters`, reader, tests, gate read and sbatch committed; local smoke
  as disclosed above.
