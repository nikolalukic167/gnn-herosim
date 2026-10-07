# mixed_dispatch_value_v1

> Produced on local-record-2026-10-06 at afea04bd; code not merged; old physics.

**Status:** `CLOSED` (2026-09-22) — ACTION-VALUE-FULL-POLICY-NEGATIVE.
Registered before fresh data generation; completed the required live gate.

**Outcome.** Outcome-aware targets move GNN in the intended direction but miss
every claim bar. Across nine models and 384 actual HeROsim runs, GNN has a
non-significant +1.90% median edge over MP-OFF and +1.68% over hand-graph, while
losing 8.98% to `srpt:128`. NumPy head serving confirms the repeated framework
calls were costly, yet the remaining Python state loop leaves GNN at
11.30–11.45 ms p95. Close unguarded full-policy action-value decode; a guarded
residual over the hand rule remains a distinct decoder question.

**Parent:** [mixed_dispatch_v1](mixed_dispatch_v1.md).
**Protocol:** [registration](../../experiments/mixed_dispatch_value_v1_protocol.json).

The total-rank child penalized arbitrary global order and the ready-set child
still treated one teacher action as uniquely correct. This successor labels
every feasible current action with the exact final cost of forcing that action
under the retained continuation order. Its listwise target preserves near-ties
and weights harmful choices by their measured consequence.

Training states come from both the strong teacher and `srpt:128` trajectories to
reduce single-policy state bias. The GNN, separately trained MP-OFF twin and
fixed one/two-hop hand-graph model remain matched. Serving computes graph
embeddings once, extracts the small dynamic head's weights, and scores subsequent
states in NumPy. This tests the objective change without repeating the previous
48-call PyTorch overhead.

A positive claim still requires a fresh 384-run HeROsim gate, at least 5% over
both learned controls and `srpt:128`, positive paired intervals, and the complete
GNN decision path within 5 ms.

## Record

- [2026-09-22 — Post-close guarded decoder screens](#2026-09-22--post-close-guarded-decoder-screens)
- [2026-09-22 — Nine-model live result](#2026-09-22--nine-model-live-result)
- [2026-09-22 — Registration](#2026-09-22--registration)

## 2026-09-22 — Registration

The protocol froze 128 training, 32 validation and 32 sealed test seeds, three
draws each of GNN, separately trained MP-OFF and hand one/two-hop controls, and
a 5% / 5 ms claim bar. Each workload contributes both teacher and `srpt:128`
trajectories. At each of 96 states, every feasible action is forced and replayed
to completion under the retained continuation order; the listwise target is a
soft distribution over measured final costs. All 192 inputs are unique and all
18,432 trajectory states pass exact baseline-continuation checks.

## 2026-09-22 — Nine-model live result

Validation GNN means are 2093.44 / 2140.81 / 2098.41 ms against 1968.19 ms for
`srpt:128`. MP-OFF means are 2095.78 / 2105.09 / 2107.84 ms and hand-graph means
are 2128.06 / 2119.13 / 2106.56 ms. No learned arm passes the validation bar.

The mandatory live gate evaluates 32 sealed workloads × 12 arms = **384 actual
HeROsim simulations and 18,432 completed operations**. The audit verifies every
arm count, objective and completion count.

| GNN comparison | Median gain | Mean paired gain | Bootstrap median 95% interval |
|---|---:|---:|---:|
| MP-OFF | +1.897% | +1.037% | [−2.697%, 4.706%] |
| Hand-graph model | +1.680% | +0.519% | [−2.970%, 3.927%] |
| `srpt:128` | −8.984% | −9.165% | [−12.337%, −6.014%] |
| Feasible teacher | −19.862% | −20.659% | [−24.546%, −16.352%] |

GNN mean costs are 2045.66 / 2077.00 / 2125.59 ms, while `srpt:128` is 1910.06
ms, plain SRPT is 2185.22 ms, and the teacher is 1727.63 ms. Extracting the
dynamic head into NumPy reduces the previous ready-set path from 15.47–15.69 ms
to 11.30–11.45 ms p95. Comparable timings across GNN, MP-OFF and hand-graph
locate the remaining cost in Python state construction and event iteration,
rather than message passing.

Outcome values therefore help but do not make the full learned policy competitive.
The small directionally positive MP contrast justifies testing a guarded residual
decoder; it is below 5%, uncertain, and cannot be reported as a GNN win.

Artifacts are under `simulation_data/gnn_environment_search_v1/` in
`mixed_dispatch_value_corpus/`, `mixed_dispatch_value_models/`,
`mixed_dispatch_value_training_logs/`, and `mixed_dispatch_value_gate/`.

**Hard stop.** Do not scale the unchanged unguarded action-value policy. A
successor may use the frozen outcome-value models only as selective residuals
over `srpt:128`, and must establish objective headroom before compiling the
state loop or opening a fresh test set.

## 2026-09-22 — Post-close guarded decoder screens

Two validation-only screens use the frozen checkpoints and touch no new holdout.
First, bounded confidence-gated overrides fail: the best GNN setting averages
0.09 overrides per workload and is 0.69% worse than leaving `srpt:128` intact;
the best MP-OFF setting is −0.02%. The exact single-override ceiling itself is
only 1.21% mean on validation, so a perfect residual classifier cannot meet the
5% bar.

Second, exact-accepted coordinated proposal search evaluates 4–32 model-ranked
schedule changes and accepts improvements only. Its best GNN gains 0.77% over
`srpt:128`, compared with 0.79% for MP-OFF and 0.94% for hand-graph. It therefore
fails the signed 5% / learned-control qualification and no fresh live gate is
opened. These screens close the obvious decoder rescues for the frozen value
models; they do not make a claim about a new training objective.

Artifacts: `mixed_dispatch_value_gate/residual_screen.json` and
`mixed_dispatch_value_gate/proposal_search_screen.json`.
