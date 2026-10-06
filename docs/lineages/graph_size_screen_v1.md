# graph_size_screen_v1

**Status:** `CLOSED` (2026-09-21) — SIZE-ALONE / NO-TRAINING.
Diagnostic configuration fixed before the run; not a powered model comparison.

**Outcome.** Increasing the three-host private-platform construction to 12, 24
and 48 tasks did not produce a hard case. Multi-start coordinate descent without
message-passing starts matches the MILP-certified optimum on all 12 graphs in
1.40–13.46 ms. All 12 solver runs certify optimality within the one-second solver
limit. Selected placements pass 57 live simulator runs. At 48 tasks, every optimum
co-locates 46 tasks on one host. No GNN training or further size-only scaling is
justified for this construction; this is not a result about arbitrary larger graphs.

**Parent:** [graph_three_host_screen_v1](graph_three_host_screen_v1.md).
**Entry points:** [runner](../../scripts_cosim/graph_size_screen.py),
[configuration](../../experiments/graph_size_screen_v1.json),
[tests](../../tests/test_graph_size_screen.py).
Artifacts: `simulation_data/graph_size_screen_v1/` contains the pre-run protocol,
overall read, per-cell explicit inputs, reads, logs and selected-plan
`placements/placements.jsonl` files. These files are **not exhaustive sweeps**.

## Record

- [2026-09-21 — Bounded size screen with optimization certificates](#2026-09-21--bounded-size-screen-with-optimization-certificates)

## 2026-09-21 — Bounded size screen with optimization certificates

Three sizes × four graph seeds produce 12 diagnostic cases. Each connected graph
has a chain plus random edges with probability 4/(n−1), keeping expected degree
roughly constant. Edge weights are 1–8. The last three tasks are pinned to distinct
hosts; all others have three eligible private platforms. All graph information is
declared before arrivals, which occur at 0.25-second intervals. The four graphs at
each size share one physical substrate; these are not 12 independent infrastructure
draws. There is no shared CPU slot or platform queue constraint forcing load balance.

The solver encodes one binary host-assignment variable per task and host. For each
edge, a nonnegative cut variable bounds both directions of the assignment difference
for every host. Minimizing positive exchange weights charges exactly one remote-edge
cost when endpoints differ. Anchors are fixed through bounds. A one-second MILP
limit is explicit; timeout would leave optimality unresolved, even with an incumbent.
Here all runs return optimal status and matching incumbent/lower bound. Solver
wall time includes matrix construction and extraction; its internal time limit does
not include those steps. Tests compare certificates against exhaustive enumeration
on three nine-task graphs and check an edgeless pinned case.

| Tasks | Immediate exact / 4 | Three-round min-sum exact / 4 | Nine-round min-sum exact / 4 | CD without MP exact / 4 | CD wall time | Solver wall time |
|---|---:|---:|---:|---:|---:|---:|
| 12 | 2 | 3 | 4 | 4 | 1.40–1.89 ms | 3.43–22.27 ms |
| 24 | 2 | 3 | 3 | 4 | 4.28–6.02 ms | 5.74–7.78 ms |
| 48 | 1 | 2 | 3 | 4 | 9.79–13.46 ms | 11.74–13.23 ms |

CD starts from immediate, peer mass, and three constant-host assignments with anchors
respected. It does **not** use min-sum or solver starts. Its timings include computing
immediate and peer-mass starts. Conditional nine-round min-sum takes 47–68 ms at 12,
229–250 ms at 24 and 939–1,244 ms at 48 in this Python implementation. This compares
these implementations, not the inherent speed of learned GNN inference. The 50 ms
allowance remains a diagnostic assumption, not an actual production deadline.

All distinct control placements were replayed live, together with an actual
`availability_v2` immediate-rule run and an exchange-off control per graph:
**57 simulations**, 29.66 seconds total excluding process startup. For selected plans,
measured exchange matches the analytical cost, queues are zero, and RTT minus
exchange is constant within each graph. The immediate rule shares this constant;
exchange-off has zero exchange/rendezvous. The live checks validate sampled plans;
unlike the small parent, they do not enumerate every possible placement. Certificates
are for the analytical objective, with live agreement on the replayed controls.

Source hashes match the pre-run record. The audit rechecked sweep hashes, complete
per-task RTT sums and repaired complete-group RTT fields. The focused solver and
witness tests pass (21 tests). Historical artifacts and closed parent results remain
unchanged. No GPU training or learned-model evaluation occurred.

**Interpretation.** At 48 tasks the four certified solutions have host occupancies
46/1/1 up to relabeling. Cheap constant-host starts already approach the favorable
structure. These fixtures remain easy because remote exchange is penalized while
co-location has no balancing cost. A larger n is therefore not evidence of useful
learning difficulty. Further work requires a justified placement trade-off surviving
strong cheap controls, a measured decision-time requirement, and a GNN input contract
that exposes the distinguishing information. Existing stops on artificial contention
and current serving-input omissions still apply.
