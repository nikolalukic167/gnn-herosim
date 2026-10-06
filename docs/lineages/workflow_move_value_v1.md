# workflow_move_value_v1

**Status:** `CLOSED` (2026-09-22) — EXACT-MOVE-TARGET-ALREADY-AFFORDABLE.
Registered before preflight; completed the mandatory fresh live gate.

**Outcome.** Do not train this one-move value predictor. Constructing ECT and
exactly scoring all 80 single-operation host flips takes 0.199 ms p95 on validation
and 0.204 ms on fresh test cases, well inside the 2 ms planning budget. Repeated
exact best-improvement search costs 1.825 ms p95 and beats the frozen GNN/ECT
hybrids by median 4.04% RTT on 32 fresh workflows. The 224 live runs and 17,920
completed task records agree with the exact evaluator. This closes the proposed
single-flip ranking target in this fixed workflow family, not multi-step lookahead
or larger graph families. No move-value GNN was trained.

**Parents:** [workflow_proposal_v1](workflow_proposal_v1.md),
[workflow_amortized_v1](workflow_amortized_v1.md).
**Protocol:** [registration](../../experiments/workflow_move_value_v1.json).
Artifacts: `simulation_data/gnn_environment_search_v1/workflow_move_value_v1/`.

## Record

- [2026-09-22 — Exact-control result and fresh live verification](#2026-09-22--exact-control-result-and-fresh-live-verification)
- [2026-09-22 — Registered exact-control preflight](#2026-09-22--registered-exact-control-preflight)

## 2026-09-22 — Registered exact-control preflight

The proposed GNN target is the full-workflow RTT change of moving one operation
to its other eligible host, conditioned on the current complete schedule. This
would replace imitation of individual teacher assignments. The first required
control is an exact evaluator of all 80 such moves. Compile the existing FIFO
semantics without fast-math and compare every result with independent Python
replay. Include Python wrapper and input conversion in steady-state timings;
exclude one-time compilation and model loading symmetrically.

Use 32 existing validation inputs (63032–63063). If constructing ECT and scoring
all moves fits a 2 ms p95 budget, stop this target's training because the exact
rank and best move are already affordable. Otherwise require useful full-plan
headroom before proceeding to matched GNN, MLP and hand-guided training.
In either outcome, run the fixed fresh 89000–89031 live gate with the same
infrastructure, exact search, retained GNN/ECT hybrids and actual Knative.
This registration does not claim exact local search is globally optimal.


## 2026-09-22 — Exact-control result and fresh live verification

**What changed.** The hand comparator now uses a compiled C++ implementation of
the existing event-driven FIFO objective. It does not introduce different physics
or peek at future placements. At each event, the earliest-ready job is processed,
with job-index tie breaking matching the Python heap. Each single-flip candidate
is replayed to completion. Python validates eligibility and shapes before calling
the native library through ctypes. Compilation uses `-O3 -std=c++17 -shared -fPIC`,
without fast-math; compiler, source and binary fingerprints are retained.

**Preflight.** On validation seeds 63032–63063, the p95 per-case-median cost of
building ECT and checking every possible single flip is **0.199081 ms**, against
the pre-registered 2 ms stop. That provides the exact supervised target, its rank
and the best single move more cheaply than inference with the existing GNN.
No GPU training, new move-value corpus or model ablation is justified on this
specific target after this stop. This is a control result, not a measurement of
an untrained hypothetical move-value model.

**Fresh live gate.** Seeds 89000–89031 are fixed independently of earlier test sets;
actual source identities are checked for reuse before simulation. Seven arms run
on each of 32 cases: native ECT, best exact single move, repeated exact descent,
three frozen parent GNN/ECT hybrids, and actual KnativeNetwork. All 224 simulations
complete. The GNN hybrids also use native ECT construction, so the compiler speedup
is not withheld from their common starting-schedule construction. Their probabilities
and frozen weights are unchanged; this is a reference comparison, not retraining.

| Planner | Median planning time | p95 planning time |
|---|---:|---:|
| Native ECT | 0.026 ms | 0.028 ms |
| ECT + exact best of 80 flips | 0.196 ms | 0.204 ms |
| Repeated exact best improvement | 0.950 ms | 1.825 ms |
| GNN + ECT guard, three draws | 1.288–1.296 ms | 1.296–1.309 ms |

Timing includes input checks, conversions, wrapper calls and plan construction;
one-time compilation and model loading are excluded from both sides. Eleven
repetitions per case supply a median, then p95 is taken across cases. The largest
case median for exact descent is 1.940 ms; none exceeds 2 ms in this sample.
This is observed steady-state timing, not a worst-case execution-time guarantee.
Descent uses at most 11 rounds in the test sample, including its stopping check;
the registered cap is 64. Single-flip local optimality does not mean global
optimality, and none is claimed.

| Comparator | Median exact-descent RTT gain | Hierarchical 95% interval |
|---|---:|---:|
| Frozen GNN/ECT hybrids | +4.04% | [+2.35%, +5.08%] |
| ECT alone | +4.44% | [+2.75%, +6.38%] |
| Actual KnativeNetwork | +59.10% | [+57.61%, +60.42%] |

GNN comparisons average paired percentages across three frozen training draws
within each environment before taking the median. Training draws are not counted
as additional independent environments. The mean full-workflow RTT is 1860.89 ms
for descent, 1951.51 for ECT, 1936.86–1945.23 for the GNN hybrids and 4587.05 for
Knative. The Knative comparison retains the parent's information asymmetry and
private-replica fixture limitation; it is not a production superiority claim.
Planning wall time remains separate from simulated task RTT.

**Independent checks.** Every single-flip score at the initial and final test
schedules is compared with independent scalar Python replay. The retained trace
contains 14,711 unique complete search plans; the artifact auditor recomputes all
their scalar costs. It also verifies original physical input tables, executed
hosts and durations, complete task/decision coverage, completion-minus-dispatch
sums, live/objective agreement, source hashes, compiled binary and frozen weights.
All 17,920 task records pass. Unit tests include tied release/duration cases,
alternate assignments, native/scalar parity, trace neutrality, local optimality
and invalid native-call inputs.

**Artifacts and entry points.** `read.json`, `preflight.json`,
`protocol_before_run.json`, `source_validation.json`, `AUDIT.json`, the compiled
library under `build/`, and per-case inputs, live records and searched-plan traces
live under the artifact root above. Code:
[compiled kernel](../../src/placement/native/workflow.cpp),
[checked wrapper](../../src/placement/workflow_exact.py),
[gate](../../scripts_cosim/workflow_move_value_gate.py),
[auditor](../../scripts_cosim/audit_workflow_move_values.py),
[tests](../../tests/test_workflow_exact.py).

**Decision and boundary.** Close the proposed learned scorer for single-flip
whole-plan values on this 80-operation family. A GNN cannot supply a more accurate
value than the exact same objective, and this exact neighborhood already fits
the time budget. This does not prove impossibility for a learned multi-step policy,
different action neighborhoods, larger instances or physics not covered by this
replay. Such a successor must first exhibit a useful budget-constrained opening
against the compiled control, rather than restore a Python-only weak baseline.
