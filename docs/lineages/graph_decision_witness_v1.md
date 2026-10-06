# graph_decision_witness_v1

**Status:** `CLOSED` (2026-09-21) — GRAPH-INFORMATION-YES / NO-TRAINING.
Constructive diagnostic; protocol saved before the full run, after a two-cell smoke check.
Not a pre-registered statistical model comparison.

**Outcome.** Distant rewiring changes the optimal first placement while preserving
its two-hop observation in all six constructed pairs. However, both three-round
hand min-sum and exact binary min-cut attain the exhaustive live optimum in all
12 cells. This construction warrants no GNN training. It demonstrates a need for
graph information, not a need to learn graph processing. No GPU was used.

**Parent:** [peer_lookahead_v1](peer_lookahead_v1.md), which remains closed.
The availability estimator repair is maintenance, not evidence for this result.

**Entry points:** [runner](../../scripts_cosim/graph_decision_witness.py),
[configuration](../../experiments/graph_decision_witness_v1.json),
[tests](../../tests/test_graph_decision_witness.py).
Artifacts: `simulation_data/graph_decision_witness_v1/`: `protocol_before_run.json`,
`substrate.json`, `read.json`, and each cell's `placements/placements.jsonl`,
`read.json`, and simulator log. Historical artifacts are untouched.

## Record

- [2026-09-21 — Live constructive witness and cheap-control challenge](#2026-09-21--live-constructive-witness-and-cheap-control-challenge)

## 2026-09-21 — Live constructive witness and cheap-control challenge

Eight tasks have private platforms on two hosts. Terminal task 3 can run only on
host 0; terminal task 7 only on host 1. Physical hardware compatibility enforces
these anchors. Swapping the distant terminal edges changes which anchor is
connected to first task 0. Its two-hop induced graph, candidate domains, weighted
degrees and two-round min-sum columns remain identical. The optimum flips from
host 0 to host 1. All group types, candidate forecasts and peer edges are declared
at the first arrival; future placements and task queues are not supplied.

Three graph families cover separate chains, a connected tree with a weak bridge,
and a connected cycle. Both simultaneous arrivals and 0.25-second trickle arrivals
are evaluated, with both rewiring variants: 12 cells, six diagnostic pairs on one
synthetic physical substrate, **not 12 independent infrastructure replicates**.

Each cell runs all 64 feasible complete placements through the actual per-arrival
simulator, plus the live immediate rule and two exchange-off controls: **804 live
simulations**. Each planned action is released only when its task arrives;
the simulator never receives the future placement table. Trickle runs assert the
observed task IDs are exactly the arrived prefix. The live rule uses
`availability_v2`. Every placement sweep is retained.

| Control | Cells attaining exhaustive minimum total RTT |
|---|---:|
| Immediate live rule | 0 / 12 |
| Peer mass | 0 / 12 |
| Two-round conditional hand min-sum | 0 / 12 |
| Three-round conditional hand min-sum | 12 / 12 |
| Eight-round conditional hand min-sum | 12 / 12 |
| Exact hand min-cut | 12 / 12 |

The minimum mean root-completion regret of choosing the same root host in both
variants ranges from 22.84% to 435.42%, exceeding the configured 5% materiality
screen in every pair. This uses optimal continuations under each fixed first
choice; it is not a comparison of trained policies. Large percentages reflect
the deliberately strong exchange costs and small private-platform execution cost.

The runner verifies in every cell that total RTT minus the binary pairwise
exchange objective is constant across all 64 placements, exchange accounting
matches that objective, and queue time is zero. The two exchange-off checks have
zero exchange and rendezvous time and equal RTT. Thus the live objective here is
a constant plus attractive binary edge costs with pinned terminals: an s-t cut
solves it exactly. Min-cut took approximately 0.54–1.02 ms per constructed graph;
the complete run took 11.23 seconds, excluding process startup. These are local
diagnostic timings, not production latency benchmarks.

The artifact audit rechecked actual script/dependency SHA256 fingerprints, every
sweep hash, uniqueness of all 64 plans per cell, and each reported optimum.
The focused witness, lookahead, availability, snapshot and record tests passed
(146 tests before adding this record).

**Scope and next gate.** Three rounds of explicit messages distinguish the root
cases; this is a constructive representational witness, not verification of a
trained GNN or its current serving graph. Current inference would need the declared
future-group representation before it could exploit this information. Because
the affordable controls already attain the optimum, there is no reason to build
or train that model for this construction. A future proposal must first show a
live advantage surviving affordable graph processing and search, and audit the
actual proposed GNN input representation. Two hosts, private platforms and
attractive exchange costs are substantive restrictions. This result does not
close shared-resource scheduling, arbitrary graph objectives, or GNNs generally.
