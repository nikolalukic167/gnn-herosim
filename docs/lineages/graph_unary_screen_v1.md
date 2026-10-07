# graph_unary_screen_v1

> Produced on local-record-2026-10-06 at afea04bd; code not merged; old physics.

**Status:** `CLOSED` (2026-09-21) — REAL-TRADE-OFF / NO-TRAINING.
Exploratory search followed by selected-case and fresh-seed live checks; no powered
model comparison or pre-registered discovery claim.

**Outcome.** Heterogeneous execution costs create a live placement trade-off between
fast individual execution and peer co-location. The optimum need not concentrate
on one host. But a coordinate-descent plus graph-cut expansion control leaves at
most 0.844% objective improvement in 60 exploratory graphs, and at most 0.774%
RTT regret in eight completed live cases. Exact MILP solves every tested case.
This candidate does not meet the program's 5% useful-headroom threshold against
strong controls. No GNN training is justified on these draws.

**Parent:** [graph_size_screen_v1](graph_size_screen_v1.md).
**Entry points:** [search](../../scripts_cosim/graph_unary_screen.py),
[live checker](../../scripts_cosim/graph_unary_live_check.py),
[tests](../../tests/test_graph_unary_screen.py).
Artifacts: `simulation_data/graph_unary_screen_v1/` contains the two exploratory
reads, incomplete `live/` and `live_retry/` attempts, and the authoritative complete
`live_compact/` run. Each completed cell retains inputs, selected-plan
`placements/placements.jsonl`, logs and a read. No file is an exhaustive live sweep.

## Record

- [2026-09-21 — Heterogeneous execution versus peer exchange](#2026-09-21--heterogeneous-execution-versus-peer-exchange)

## 2026-09-21 — Heterogeneous execution versus peer exchange

The mechanism uses existing task-by-platform execution tables, not new contention
physics or an imposed balancing constraint. Tasks have host-specific execution
costs; peer edges charge the existing remote exchange. Each task has a private
replica on each of three hosts. All group edges and execution forecasts are declared
before arrivals. The costs are synthetic draws, not measured production workloads.

The exploratory screen contains 48 graphs at 24/48 tasks (three execution-cost
scales × eight seeds) and 12 graphs at 128 tasks (three scales × four seeds).
Costs are nonnegative multiples of 0.1 seconds; the complete objective sums unary
execution cost and attractive pairwise exchange. Controls include independently
fastest placement, best constant-host placement, multi-start CD initialized from
those plans, and alpha-expansion initialized from CD. Each expansion solves an
exact binary keep-or-switch move by min-cut. A one-second MILP supplies an optimum
certificate on all 60 draws. The largest remaining improvement over expansion
is 0.84355% of its analytical cost, at 48 tasks; the largest at 128 is 0.43761%.
These exploratory reads were not pre-registered and do not support significance claims.

For live validation, select the two largest exploratory gaps in each size block,
then add four fresh fixed 48-task seeds (50600–50603). Save explicit cases and source
hashes before the live run. Tiny floating-point ties influenced the second selected
48-task case; it is a selected example, not independent evidence.

**Concrete trade-off.** Fresh seed 50601 uses 48 tasks. Its certified solution
places 9/12/27 tasks on the hosts. In live RTT it beats independent fastest-host
placement by 7.70% and best constant-host placement by 13.32%. Graph-cut expansion
attains the same optimum. Selected seed 48400 splits 30/12/6 and improves those
baselines by 8.58% and 6.35%, respectively; expansion is only 0.77319% above its
live optimum. This demonstrates a graph-dependent optimization benefit, not a
benefit specific to learned message passing.

**Live outcome.** The complete run has **49 simulations across eight cases**,
including distinct selected plans, the actual immediate rule and an exchange-off
control per case. Full task accounting confirms the execution tables and exchange
costs, zero queues and a placement-independent RTT residual for replayed plans.
Exchange-off runs have zero exchange and rendezvous. The largest live expansion
regret is 0.77319%; among the four fresh cases it is 0.37815%. The solver closes all
of those gaps. Solver wall times in this live check range from 11 to 185 ms;
these are local observations, not a production deadline guarantee. The analytical
certificate does not prove the reduction for unexecuted live placements.

**Harness defect and repair.** The initial generator assigned a separate hardware
alias to every task-host pair, creating 384 types at 128 tasks. Autoscaling scans
all types for each task type and performed excessive futile scale-down checks.
The first 128-task replay exceeded 10 seconds; a bounded 30-second retry also failed.
Both failed artifact directories are retained, not treated as valid runs.
The final harness uses three hardware types, one per host, with the same per-task
execution tables and deterministic private replica assignments. Every selected
plan and actual immediate-rule RTT in the two previously completed 48-task cases
matches the old harness to 1e-8. No production simulator physics was edited.

**Checks.** The completed run's source fingerprints and placement-sweep hashes match.
An independent audit reconciled all task completion-minus-dispatch sums. Tests
compare unary MILP and vectorized enumeration against exhaustive scalar costs,
check that expansion leaves no improving binary move on a small graph, and check
physical execution tables. The focused witness/solver suite passes (24 tests);
record hygiene is checked after writing this node. Shared helpers gained optional
unary costs with unchanged defaults; earlier artifact fingerprints are historical.

**Decision.** This search found a real execution-versus-exchange trade-off, but
not one that survives affordable graph processing by a material margin. Do not
promote a win over independent-fastest or constant-host placement into a GNN claim.
Current future-peer serving representation remains an additional prerequisite.
No result here establishes impossibility for all heterogeneous scheduling problems;
no training or larger sweep is authorized by the mere existence of this trade-off.
