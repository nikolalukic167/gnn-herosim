# peer_greedy_live_v1 — a hand rule with the learned arms' information, on the live gate

**Status:** `CLOSED` (2026-09-20) — **`HAND-RULE-BEATS-REACTIVE-WITHOUT-WAITING` ·
`RULE-FASTER-THAN-GRAPH-ARM`.** Closed on a live gate (rule 6): 95 of 96 arms (one batched-rule
arm hangs at the end of the trace, below). Registered 2026-09-20; bars, reader, 7 tests and the
expectation committed before the gate, one local smoke cell disclosed. **The registered
expectation was right in direction on every read and too cautious in magnitude on all of them.**

**Outcome. The first live win over a healthy reactive baseline in this programme is a two-line
rule, and it beats the learned arm too.** `peer_greedy_network` — Knative's candidate set scored
in seconds as *queue drain + cold + exec + latency + exchange to the partners already placed*,
per arrival, no group wait — reads against reactive Knative on the 16 admissible (topology,
window) environments of `unsaturated_edge_v1`:

| rung | G1 rule vs reactive (env, n = 16) | G4 vs random | vs ECT | G6 vs the same rule with exchange off |
|---|---|---|---|---|
| **C40** | **−12.96 %, 16/16, p = 0.0004** | −22.64 %, 16/16 | −28.68 %, 16/16 | **−14.80 %, 15/16** |
| **C80** | **−16.01 %, 16/16, p = 0.0004** | −21.12 %, 16/16 | −31.98 %, 16/16 | −13.33 %, 16/16 |

The exchange-off twin (`drain_greedy_network`) ties Knative (−0.45 % / +0.09 %, 8/16 each), so
the whole margin is the exchange term: the rule joins a partner's node on **41–43 % of
decisions**, the exchange term moves the argmin on 16–17 %, and per task (C40 medians) exchange
falls **5.40 → 3.91 s** and the platform queue **8.66 → 7.80 s** — G7 `CO-LOCATION-DELIVERED`
on 16/16 environments, with the queue *shorter*, not longer: co-location does not cost
concentration here, it shortens the queues because co-located peers do not hold a platform for
a remote transfer. Rendezvous is unchanged (3.69 → 3.84 s), as it must be for a per-arrival rule.

**Against the learned arm (G3, GRAPH `gnnedge0`, paired on environment and checkpoint):** at C80
the batched rule is **−13.18 % (16/16)** and the immediate rule **−22.90 % (16/16)** faster than
`gnnedge0`; at C40 the registered slot is UNREADABLE by the letter (the batched rule hangs on
`cc40s9003` w2, `gnnedge0` s8 on `cc40s9101` w0) and the disclosed reads are **−8.26 % (14/14)**
and **−18.00 % (14/14)**. **Served identically** — the same peer-group batching at 16 s, the same
`masked_topo` batch path, greedy in id order instead of the decoder — the rule pays the same
7.09 s wait and still beats the model: `peer_greedy_network_batch` vs reactive is −4.86 %
(C80, 10/16, `NOT-SEPARATED`) and −7.55 % (C40 disclosed, 12/15) where `gnnedge0` reads +6.68 %
and +14.40 %. G5: no wait beats the wait by **−8.30 % (16/16)** at C80 and −7.30 % (15/15,
disclosed) at C40 — the number the no-wait decoder was going to be built to find.

**What this settles.** (1) The environment *does* reward peer-aware placement at a healthy
load, by 13–16 % over shortest-queue, and a rule with no free constant collects it; the
learned arms' loss to Knative was never evidence that co-location does not pay. (2) The
learned arm has not learned what the rule encodes: at matched serving it is 8–13 % behind a
greedy on the physics it was trained to approximate, so "a graph model learns co-location" is,
at this operating point, "a graph model recovers part of a rule". (3) Item 1 of the plan (a
no-wait decoder) now has the rule as its bar, not Knative, and G5 says removing the wait is
worth ~8 % on top of whatever the decoder learns. (4) ECT's +12.8 / +23.7 % loss to
shortest-queue is its queue term (`len × exec`, 0.1 s per queued task), not its physics
awareness — the same physics with the queue in seconds ties Knative.

**Carry.** The rule's queue term is `platform_queue_drain_seconds`, which charges known-peer
transfers of queued tasks and not their rendezvous; it is a better shortest-queue, not an
oracle. Every number here is at 200 MB payload scale on a 1 Gbps backbone
(`payload_scale_v1`, `backbone_sparsity_v1` move that) and with groups spread over 13.8 s
(`burst_groups_v1`). The batched rule inherits the learned arms' end-of-trace hang on one cell
(deterministic; 48 and 96 GB) — a property of the batch path, not of any model.

**Question (as registered).** The learned arms have never been compared with a rule that uses
the same information. **(a)** Does a two-line peer-aware rule that decides per arrival — no
group wait — beat reactive Knative where every learned arm loses? **(b)** Served exactly as the
learned arms are (peer-group batching, 16 s window), does the rule match `gnnedge0`'s placement?

**The rule** (`src/policy/peer_greedy_network/scheduler.py`). For task i and candidate replica
(node n, platform p), in seconds, every term the physics charges:

    S(n, p) = drain(p) + cold(p) + exec_i(p) + latency(src_i → n) + X_i(n)

`drain(p)` is `platform_queue_drain_seconds` (the audit's estimator). `X_i(n)` is the exchange
the simulator will charge at i's input stage for every peer whose node is already known —
placed, planned by a batch pre-pass, or committed earlier in the same batch — by the same
`Platform._payload_transfer_time` + latency, 0 for a co-located peer; unknown peers contribute
0. Argmin over Knative's candidate set with Knative's tie-break; no free constant.
`knative_network_ect` is not this rule (queue term `len(queue) × exec`).

| arm | stack | wait | what it isolates |
|---|---|---|---|
| `peer_greedy_network` | knative_network (per arrival) | none | the rule |
| `drain_greedy_network` | knative_network (per arrival) | none | X ≡ 0: the exchange term's ablation |
| `peer_greedy_network_batch` | gnn (peer-group batching, `masked_topo` batch path, `planned_node_name` pre-pass) | 16 s | the rule in `gnnedge0`'s seat |

**Design.** The 16 environments per rung of `unsaturated_edge_v1` (`selected.json`), 6 servers,
C40 primary and C80 secondary; 3 arms × 16 × 2 = 96 runs. Reactive, ECT, random and `gnnedge0`
are the study's own runs. Same cell configs (`batch_timeout` 16 s, `batch_size` 10 asserted),
same environment variables.

**Bars** (`scripts_cosim/peer_greedy_live_v1_read.py`; |median| ≥ 5 %, p < 0.05, two-sided
signed-rank). Rule-vs-baseline: **unit = environment, n = 16**. Rule-vs-`gnnedge0`: unit =
checkpoint, the rule's value paired against every checkpoint on the same environment.

| read | what | fires as |
|---|---|---|
| **G1** | `peer_greedy_network` vs reactive (env) | `IMMEDIATE-RULE-BEATS-REACTIVE` / `REACTIVE-FASTER-THAN-IMMEDIATE-RULE` / `NOT-SEPARATED` |
| G2 | `peer_greedy_network_batch` vs reactive (env) | `BATCHED-RULE-BEATS-REACTIVE` / … |
| **G3** | batched rule vs `gnnedge0`, RULE FIRST (ckpt) | `RULE-FASTER-THAN-GRAPH-ARM` / `GRAPH-ARM-FASTER-THAN-RULE` / `RULE-MATCHES-GRAPH-ARM` |
| G4 | immediate rule vs random (env) | `IMMEDIATE-RULE-BEATS-RANDOM` / … |
| G5 | immediate vs batched rule (env) | `IMMEDIATE-FASTER-THAN-BATCHED` / … |
| G6 | immediate vs drain-only (env) | `EXCHANGE-TERM-HELPS` / `EXCHANGE-TERM-HURTS` / … |
| G7 | mechanism: exchange fell on ≥ 12/16 env; queue delta carried | `CO-LOCATION-DELIVERED` / … |
| G8 | composite on G1, carrying G3 and G5 | `HAND-RULE-BEATS-REACTIVE-WITHOUT-WAITING` / `…-TIES-…` / `…-LOSES-TO-…` |

**Registered expectation (signed 2026-09-20, before the gate; one smoke cell seen).** Smoke on
`cc40s9001` w0: reactive reproduced datalab's `total_rtt` to the digit, a repeated rule run was
bit-identical, and the cell read immediate −14.3 %, batched −7.5 %, drain-only +0.2 %. Signed:
G1 BEATS 70 % (NOT-SEP 25 %, REACTIVE-FASTER 5 %); G2 BEATS 45 % / NOT-SEP 40 %; G3 RULE-FASTER
55 %, MATCHES 35 %, GRAPH-FASTER 10 %; G4 fires; G5 IMMEDIATE-FASTER 85 %; G6 HELPS 90 %; G7
DELIVERED 90 %; at C80 the same signs with smaller margins. **Read:** every sign as signed; the
C80 margins are *larger*, not smaller.

**Consequences (as signed; the ones that fired in bold).**
- **G1 fires:** the first live win over a healthy reactive baseline is a two-line rule; the
  no-wait decoder has the rule, not Knative, as its bar; the paper's mechanism claim is the
  rule's decomposition.
- G1 does not fire and G7 delivered: co-location eaten by concentration — did not happen.
- **G3 RULE-FASTER:** at matched serving the graph arm has learned less than the rule encodes; a
  hard stop on "a graph model learns co-location" at this operating point (`docs/hard-stops.md`).
- G6 does not fire: did not happen — the drain-only twin ties Knative, the exchange term is the
  whole margin.

**Cost.** 96 runs; per-arrival arms ~1 min, batched ~3 min; two blocks of 48 plus one re-run.
Script `scripts_cosim/datalab/peer_greedy_live_v1.sbatch`; read
`scripts_cosim/peer_greedy_live_v1_gate_read.py`; results under
`simulation_data/peer_affinity_live_gate/results/pg_v1/`.

**Datasets.** None; nothing trained.

## Record (newest first)

- 2026-09-20 — **CLOSED on the gate.** Jobs 793097 (C40, 47/48: task 38 = batched rule on
  `cc40s9003` w2 OUT_OF_MEMORY at 48 GB at the end of the trace), 793149 (C80, 48/48), re-run
  793198 at 96 GB cancelled by the 20-minute watchdog at the same point (deterministic; the read discloses it). Read (`peer_greedy_live_v1_gate_read.py`, disclosed variants
  added after the data for the two hung arms, registered slots untouched): C40 G1 −12.96 %
  (16/16), G2 UNREADABLE / disclosed −7.55 % (12/15), G3 UNREADABLE / disclosed −8.26 % (14/14),
  immediate vs `gnnedge0` disclosed −18.00 % (14/14), G4 −22.64 %, G5 disclosed −7.30 % (15/15),
  G6 −14.80 % (15/16), vs ECT −28.68 %, G7 DELIVERED 16/16 (exchange −1.297 s, queue −1.213 s);
  C80 G1 −16.01 % (16/16), G2 −4.86 % NOT-SEP (10/16), G3 −13.18 % (16/16), immediate vs
  `gnnedge0` −22.90 % (16/16), G4 −21.12 %, G5 −8.30 % (16/16), G6 −13.33 % (16/16), vs ECT
  −31.98 %, G7 DELIVERED 16/16 (exchange −1.282 s, queue −1.865 s). Per-task medians C40:
  Knative 0 / 8.66 / 5.40 / 3.69 / 17.87; rule 0 / 7.80 / 3.91 / 3.84 / 15.59; drain-only
  0 / 8.53 / 5.25 / 3.50 / 17.47. C80 batched rule 7.09 / 4.99 / 3.76 / 1.24 / 17.16.
  Counters: joined a partner's node on 41.3 % (C40) / 42.9 % (C80) of decisions; exchange moved
  the argmin on 16.0 / 16.7 %; the batched rule 43.7 % / 17.3 %, 10,028 batches at C80.

- 2026-09-20 — Registered. Policies, registry entries (`src/placement/simulation.py`,
  `src/placement/model.py`, `src/executesimulation.py`), the `pg_*` counter whitelist in
  `Orchestrator._scheduler_counters`, reader, tests, gate read and sbatch committed; local smoke
  as disclosed above.
