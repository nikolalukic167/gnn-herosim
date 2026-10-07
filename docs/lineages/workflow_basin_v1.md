# workflow_basin_v1

> Produced on local-record-2026-10-06 at afea04bd; code not merged; old physics.

**Status:** `CLOSED` (2026-09-22) — INSUFFICIENT-INITIALIZER-SELECTION-HEADROOM.
Registered before measurements; completed the fresh live gate.

**Outcome.** Even perfect selection among twelve refined starting schedules gives
only 2.58% median RTT improvement on qualification and 1.60% on the fresh live
gate, below the registered 5% training bar. A calibrated hand selector fits the
2 ms budget. No GNN was trained for this target. The 128 live simulations confirm
this finite-portfolio result; it does not rule out learning new starting schedules.

**Parents:** [workflow_move_value_v1](workflow_move_value_v1.md),
[workflow_proposal_v1](workflow_proposal_v1.md).
**Protocol:** [registration](../../experiments/workflow_basin_v1.json).
Artifacts: `simulation_data/gnn_environment_search_v1/workflow_basin_v1/`.

## Record

- [2026-09-22 — Insufficient initializer-selection headroom; live gate completed](#2026-09-22--insufficient-initializer-selection-headroom-live-gate-completed)

- [2026-09-22 — Registered initializer-selection screen](#2026-09-22--registered-initializer-selection-screen)

## 2026-09-22 — Registered initializer-selection screen

Build twelve initial schedules: four remaining-work-aware greedy variants and
eight deterministic perturbations of ECT, with probabilities and seeds fixed
in the protocol. Run the same compiled exact single-flip descent from each.
The hypothetical model predicts which completed search will be best; it is not
being asked to approximate cheap exact one-move costs.

Use 64 existing validation cases for calibration, then 64 different validation
cases for the headroom screen. Choose a globally fixed initializer and a deployable
hand selection rule on calibration, including minimum initial RTT and two-start
search. Measure all plan construction, initial scoring and refinement costs.
The best final result across all twelve searches is an oracle ceiling with its
actual expense reported, not a free per-instance hand selector.

Require at least 5% median oracle improvement with a positive bootstrap lower
bound before training. A GNN cannot exceed that portfolio's best result; a small
oracle gap stops the proposed learning target cheaply. Regardless of the screen,
complete the fresh 32-case live gate before closing. Earlier test sets are not
reused for selection or this live gate.

Calibration implementation note: the first attempt stopped before qualification
because no Python-loop selector met 2 ms. A direct diagnostic measured ECT descent
p95 2.4805 ms on the calibration cases. The same strict best-improvement loop is
now compiled for every arm, with exact assignment/cost/round-count parity against
the traced Python loop. No portfolio, budget, split or threshold changed. The
initial attempt is preserved at the original artifact path; the complete rerun
uses `workflow_basin_v1_native/`. A fixed initializer computes only its own start.

## 2026-09-22 — Insufficient initializer-selection headroom; live gate completed

The frozen executable control chooses the start with lowest initial RTT, then
runs exact descent once. Calibration selected this rule at mean RTT 1808.01 ms
and p95 planning time 1.799 ms. The fixed initializer (remaining-work weight .25)
was worse at 1827.09 ms. Two-start search was better at 1789.86 ms but exceeded
the budget at 2.699 ms p95. ECT descent cost 1844.72 ms. These are calibration
comparisons; the selector was frozen before qualification and fresh live testing.

On 64 qualification workflows, a perfect selector among all twelve refined
starts improves median paired RTT by **2.584%**, bootstrap 95% interval
**[1.429%, 4.164%]**. It misses the registered 5% threshold even when granted
free oracle selection. No GNN, MP-OFF or hand-feature selector was trained.
This is a target-headroom rejection, not a trained-model comparison.

The mandatory fresh live gate uses seeds 90000–90031: four arms per environment
(frozen hand selector, ECT descent, portfolio oracle, actual KnativeNetwork),
**128 simulations and 10,240 completed tasks**. Median paired oracle improvement
over the hand selector is **1.597% [0.393%, 2.701%]**. The hand selector improves
RTT over Knative by **58.214% [55.946%, 59.499%]**, without a GNN. The latter
comparison uses the inherited full-workflow information advantage and private
replicas sharing node compute slots; it is not a production Knative benchmark.

| Planner | Median planning time | p95 planning time |
|---|---:|---:|
| Frozen minimum-initial-RTT selector + descent | 1.412 ms | 1.817 ms |
| Compiled ECT descent | 0.593 ms | 0.990 ms |
| Full twelve-start oracle search | 12.734 ms | 15.150 ms |

Times include start construction, scoring, selection, checks and refinement,
excluding one-time compilation. Each case uses the median of five runs; p95 is
across cases. Planning time is reported separately from simulated RTT. The oracle
cannot actually be deployed for free: its actual measured search time is above.
No per-case hindsight best-hand selector was used as the control.

The new compiled loop preserves the original single-flip neighborhood, strict
improvement and index tie breaking. Timed and traced search are checked for
agreement. Original native and Python planner files remain unchanged, preserving
historical fingerprints. Source/cache/physical-problem checks precede simulations;
new problems are checked against the parent corpus and both preceding live gates.
The trace retains **319,363 unique complete scored plans**, individually checked
against scalar replay, plus all actual task placements, durations and RTTs.

Authoritative artifacts: `simulation_data/gnn_environment_search_v1/workflow_basin_v1_native/`:
`calibration.json`, `qualification.json`, `decision.json`, `read.json`,
`protocol_before_run.json`, `parent_validation.json`, `source_validation.json`,
`AUDIT.json`, and per-case source inputs, live records and searched-plan traces.
Entry points: [gate](../../scripts_cosim/workflow_basin_gate.py),
[auditor](../../scripts_cosim/audit_workflow_basins.py),
[portfolio](../../src/placement/workflow_basins.py),
[native extension](../../src/placement/native/workflow_starts.cpp),
[tests](../../tests/test_workflow_basins.py).

This closes learning to select from this finite initializer portfolio on this
fixed family at the registered 5% bar. It does not close models that construct
new starts outside the portfolio, adaptive multi-step search, or larger workflows.
A successor needs distinct measured headroom against the compiled controls before
training; expanding the portfolio until these test cases look positive is not a
fresh test.
