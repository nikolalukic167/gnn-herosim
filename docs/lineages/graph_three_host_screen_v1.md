# graph_three_host_screen_v1

> Produced on local-record-2026-10-06 at afea04bd; code not merged; old physics.

**Status:** `CLOSED` (2026-09-21) — SMALL-THREE-HOST / NO-TRAINING.
Diagnostic configuration and live seeds fixed before this run; not a powered model gate.

**Outcome.** On 16 generated nine-task graphs with three hosts and three pinned
terminal tasks, nine-round hand min-sum and the multi-start coordinate-descent
portfolio both match the exact exchange optimum on 16/16. Exhaustive search over
729 placements takes 0.34–0.88 ms. Full live sweeps on two preselected graphs
confirm the objective and exact-control placements. No GNN training is justified
on this small construction. This is not a result about larger groups or learned
model classes.

**Parent:** [graph_decision_witness_v1](graph_decision_witness_v1.md).
Both that binary witness and [peer_lookahead_v1](peer_lookahead_v1.md) stay closed.

**Entry points:** [runner](../../scripts_cosim/graph_three_host_screen.py),
[configuration](../../experiments/graph_three_host_screen_v1.json),
[tests](../../tests/test_graph_three_host_screen.py).
Artifacts: `simulation_data/graph_three_host_screen_v1/` holds the pre-run
protocol, explicit substrate, overall read, per-seed reads, and full live
`placements/placements.jsonl` sweeps for seeds 46200 and 46201.

## Record

- [2026-09-21 — Artifact audit and auxiliary reporting repair](#2026-09-21--artifact-audit-and-auxiliary-reporting-repair)
- [2026-09-21 — Third-host challenge and serving-input audit](#2026-09-21--third-host-challenge-and-serving-input-audit)

## 2026-09-21 — Artifact audit and auxiliary reporting repair

Independent recomputation checked all 1,458 saved placement runs (13,122 task
rows): actual execution hosts, arrived prefixes, per-task completion minus dispatch,
summed RTT, exchange, zero queues, contract selection and each optimum. All pass.
Recorded source hashes matched the files before this repair. Scalar enumeration
also reproduced the analytical optimum and control costs on all 16 graphs.
The receipt is `simulation_data/graph_three_host_screen_v1/audit_2026-09-21.json`.

**Reporting defect found and fixed:** the shared original probe's `group_rtt`
means its first eight tasks. Reusing that helper for nine tasks left this auxiliary
field short by task 8 in all saved placement rows. The constructed-witness wrapper
now explicitly makes `group_rtt` equal the complete group's `total_rtt`. The
primary result always used `total_rtt`, independently reconciled against all nine
tasks, so the optima, comparisons and verdict do not change. Historical artifacts
are preserved; their auxiliary `group_rtt` must not be used as a nine-task total.
A live regression checks all nine tasks for two different plans.

The original exhaustive-search timer stopped before argmin and plan extraction.
It measured enumeration and scoring, not the complete selection call. The timer
now includes selection. A fresh 320-call check including selection measured
0.332–0.519 ms (median 0.360 ms); these are local timings, not a production latency
guarantee. Source fingerprints in the historical run intentionally predate these
repairs. Neither timing observation establishes performance at larger group sizes.

## 2026-09-21 — Third-host challenge and serving-input audit

The binary witness's private-platform substrate was extended to three hosts.
Tasks 6, 7 and 8 have physical hardware compatibility only with hosts 0, 1 and 2,
respectively; the other six tasks can use any host. All graph edges and candidate
domains are declared before the first arrival. Each seeded graph contains a chain
for connectivity plus random additional edges, with integer weights from 1 to 8.
The 16 graph draws are unique but share **one physical substrate**. They are not
16 independent infrastructure draws. No new shared-resource physics was introduced.

| Control | Exact exchange optimum, out of 16 | Observed wall time per graph |
|---|---:|---:|
| Immediate analytical greedy | 4 | 0.07–0.14 ms |
| Peer mass | 5 | 0.28–0.47 ms |
| Three-round conditional hand min-sum | 14 | 5.13–10.42 ms |
| Nine-round conditional hand min-sum | 16 | 13.39–28.22 ms |
| Multi-start coordinate-descent portfolio | 16 | 19.33–39.92 ms |
| Exhaustive vectorized search | 16 | 0.34–0.88 ms |

The portfolio starts from immediate, peer-mass, both min-sum solutions and three
constant-host initializations with anchors respected. Its time includes computing
those starting solutions. Its success is therefore **not independent of min-sum**.
Exhaustive search includes constructing all 729 feasible plans and evaluating
their exchange objectives. It is practical at this size; no scaling claim follows.
The configured 50 ms search allowance is a diagnostic assumption, not a measured
production scheduling deadline. No inference time is charged to simulated RTT.

The preselected live seeds 46200 and 46201 each receive 729 complete-placement
simulations, one actual `availability_v2` immediate-rule run, and two exchange-off
controls: **1,464 live runs**, completed in 42.48 seconds excluding process startup.
All live runs use 0.25-second arrivals. Actions enter only when their task arrives;
the arrived-prefix assertion prevents accidental exposure of future task objects.
No future placement table is passed into the simulator.

For every live placement, queue time is zero, measured exchange matches the
analytical objective, and total RTT minus that objective is constant within its
graph. Exchange-off controls have zero exchange/rendezvous and equal RTT. These
checks justify the exact objective on the two live-tested graphs. Their optimum
total RTTs are 55.65608 and 34.59608 seconds; the actual immediate rule obtains
55.65608 and 50.61608 seconds. Both exact-search and portfolio placements attain
the live optimum. The remaining 14 graphs are analytical screens, not live gates.

Source SHA256 fingerprints were captured before the run and rechecked afterward.
An independent read verified all recorded fingerprints, sweep hashes, 729 distinct
plans per live seed, and reported optima. Focused binary/three-host witness and
live-probe tests pass (20 tests); record hygiene is checked after writing this node.
Historical artifacts are preserved. The shared substrate helper now accepts a
host count with the historical default of two; historical fingerprint records
remain records of the earlier source, not hashes of the extended helper.

**Serving audit.** `src/policy/gnn/scheduler.py` builds the inference graph from
`batch_tasks`. In `src/policy/gnn/prefix_serving.py`, `attach_live_prefix_block`
maps peer IDs only to that batch, counts outside peers, and excludes their edges.
The existing prefix-serving contract thus does not expose unarrived peers as
nodes carrying their candidate domains. A proposed future-lookahead GNN needs an
explicit, tested future-group representation and matching training contract;
this audit does not establish that any current checkpoint can distinguish these
cases. No serving code or checkpoint was changed.

**Decision.** Stop this nine-task candidate before training. Increasing the host
count alone did not defeat cheap controls. Any larger-group investigation must
first establish an actual scheduling-time constraint, test stronger affordable
search at that scale, and verify model-visible information. This measurement
does not establish a general impossibility result for three-host scheduling.
