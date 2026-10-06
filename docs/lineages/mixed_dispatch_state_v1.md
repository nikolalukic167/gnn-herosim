# mixed_dispatch_state_v1

**Status:** `CLOSED` (2026-09-22) — READY-SET-IMITATION-NEGATIVE. Registered
before fresh data generation; completed the required live gate.

**Outcome.** Replacing unstable total ranks with state-dependent feasible actions
does not recover the scheduling headroom. Across nine models and 384 actual
HeROsim runs, GNN ties MP-OFF (−0.71% median) and the hand-graph model (+0.42%),
loses 9.74% to `srpt:128`, and takes 15.47–15.69 ms p95 to serve. The target
repair is correct, but this imitation recipe neither establishes a
message-passing advantage nor meets the 5 ms contract.

**Parent:** [mixed_dispatch_v1](mixed_dispatch_v1.md).
**Protocol:** [registration](../../experiments/mixed_dispatch_state_v1_protocol.json).

The total-rank child taught 48 absolute ranks even when two operations never
competed. Four equally budgeted searches on its 32 held-out problems produce a
median normalized rank difference of 0.328; even the 42 teacher pairs within 1%
objective differ by 0.329 and no pair has the same realized start order. The
model's 0.313–0.328 rank error is therefore on the scale of disagreement between
valid teachers.

This child replaces total-rank regression with a choice among operations that
are feasible now. A GNN computes static operation embeddings once. A small head
combines those embeddings with current setup, resource occupancy, ready time and
remaining-work state at each decision. MP-OFF receives identical local inputs;
the hand arm receives fixed typed one/two-hop summaries. The full decision path,
including every dynamic head call, must fit the same 5 ms budget.

Before training, fresh data had to pass three checks: reconstructed state choices
exactly replay the teacher objective, decision sets contain meaningful choices,
and the teacher retains at least 5% headroom over `srpt:128`. Passing those
checks permitted the registered small matched pilot; it was not a GNN result.

## Record

- [2026-09-22 — Nine-model live result](#2026-09-22--nine-model-live-result)
- [2026-09-22 — Registration and qualification](#2026-09-22--registration-and-qualification)

## 2026-09-22 — Registration and qualification

Four independently seeded 8,192-proposal searches exposed the preceding target
defect. On the old 32-case holdout, teacher-pair normalized rank MAE is 0.328
and realized start-order inversion fraction is 0.280. Among 42 teacher pairs
within 1% objective, rank MAE remains 0.329; no pair has identical start order.

The successor was registered on disjoint seeds before generation. Its 32 fresh
validation workloads passed qualification: the teacher beats `srpt:128` by
10.10% mean, 66.93% of action states have multiple feasible choices, candidate
sets average 2.65 operations, and all state traces exactly reproduce their
teacher objectives. The resulting corpus contains 128 train, 32 validation and
32 sealed test workloads. The GNN, separately trained MP-OFF twin and hand
one/two-hop model share hidden width 16, two layers, cross-entropy action labels,
three draws and actual served validation-cost selection.

## 2026-09-22 — Nine-model live result

The registered gate fails. Validation GNN means are 2124.78 / 2177.03 /
2147.88 ms, 8.78–11.46% worse than `srpt:128`; MP-OFF and hand-graph arms occupy
the same range. The sealed test runs 32 physical workloads × 12 arms = **384
actual HeROsim simulations and 18,432 completed operations**. Every objective,
arm count and completion count passes the independent artifact audit.

| GNN comparison | Median gain | Mean paired gain | Bootstrap median 95% interval |
|---|---:|---:|---:|
| MP-OFF | −0.709% | −0.124% | [−3.293%, 3.460%] |
| Hand-graph model | +0.421% | −0.141% | [−2.425%, 3.393%] |
| `srpt:128` | −9.744% | −10.273% | [−13.848%, −6.599%] |
| Feasible teacher | −22.557% | −23.140% | [−27.099%, −18.275%] |

GNN mean costs are 2146.09 / 2135.81 / 2133.84 ms, compared with 1944.94 ms
for `srpt:128`, 2176.50 ms for plain SRPT, and 1737.94 ms for the teacher. The
learned policy improves over plain SRPT but fails to reproduce the hand rule's
cheap 128-swap refinement. GNN p95 decision times are 15.47–15.69 ms; all learned
controls are equally slow. One static graph encoding did not solve latency
because the Python/PyTorch boundary is crossed once for each of 48 dependent
actions.

This distinguishes two failures. The first child used an unstable total-order
target. This child repairs that target and still finds no message-passing edge:
local dynamic state plus fixed graph summaries are enough for the learned
behavior it recovers. Compiling or batching the dynamic head could repair the
time budget, but cannot supply the missing 9.74% objective gain or separate GNN
from its controls.

Artifacts are under `simulation_data/gnn_environment_search_v1/` in
`mixed_dispatch_state_screen/`, `mixed_dispatch_state_corpus/`,
`mixed_dispatch_state_models/`, `mixed_dispatch_state_training_logs/`, and
`mixed_dispatch_state_gate/`. Teacher instability diagnostics are in the prior
gate as `teacher_stability_diagnostic_v2.json`.

**Hard stop.** Do not scale unchanged teacher-trajectory cross-entropy with the
hidden16/two-layer ready-set decoder. A successor needs outcome-aware action
values or on-policy correction, a compiled event loop, and evidence before
training that message passing predicts residual value beyond the hand graph
control.
