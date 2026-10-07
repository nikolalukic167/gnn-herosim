# dag_reservation_learning_v1

> Produced on local-record-2026-10-06 at afea04bd; code not merged; old physics.

**Status:** `CLOSED` (2026-09-22) — GUARDED ONE-SHOT DAG PROPOSALS NEGATIVE.
The user explicitly requested training after the 8% funding gate failed; this
overrides that spending decision for this experiment. It does not change the
earlier measurement or its stated scope.

**Outcome.** Four matched CPU models completed training and a fresh **96-run**
HeROsim gate. GNN ties MP-OFF at median 0.00% and loses 1.81% to selected
hand search, despite all planners meeting 250 ms. All 64 learned one-shot
proposals lose the initial-cost guard; the served learned arms therefore
test a shared hand start plus search. Both full schedule and resource-event
audits pass. This closes this proposal/guard recipe, not graph-conditioned
stepwise reservations or GNN scheduling generally.

**Question.** Can a graph model propose complete placement and reservation
orders that improve summed job completion within 250 ms, compared with a
separately trained pointwise twin and the strong hand search?

**Parent:** [mixed_dispatch_v1](mixed_dispatch_v1.md).
**Protocol:** [frozen training and gate](../../experiments/dag_reservation_learning_v1_protocol.json).

## Record

- [2026-09-22 — Exploratory trained live result](#2026-09-22--exploratory-trained-live-result)
- [2026-09-22 — Corpus and matched training receipt](#2026-09-22--corpus-and-matched-training-receipt)
- [2026-09-22 — Registration before corpus and training](#2026-09-22--registration-before-corpus-and-training)

## 2026-09-22 — Registration before corpus and training

The corpus uses 64 training, 16 validation and 16 sealed test seeds, all new
and disjoint from prior DAG screens. Training labels are the best feasible
schedules found by the full heuristic baseline and two fixed 8,192-step
searches. They are not optimal schedules. Every saved training plan is
replayable; this is a searched-plan corpus, not a co-simulation enumeration.
Test inputs are generated and fingerprinted before training, but their costs
and labels are not opened until checkpoints are selected.

Both models receive the same scalar features and explicit fixed 1/2/4-hop
hand-graph columns. Only the GNN propagates learned hidden states across DAG,
lock and host edges. Two separately initialized seeds per arm are trained on
the same split and selected by complete-workflow validation cost. Training
runs through `run_experiment.py`, uses deterministic CPU PyTorch and logs
offline to W&B. All checkpoints require architecture/data/source sidecars.

The served learner proposes a complete placement and operation rank, tries
both scheduling decoders, and keeps the better plan or full heuristic
initializer before equal-budget compiled search. Inference, hand initialization,
conversion and search all count against 250 ms. The selected hand rule from
the predecessor screen is the primary control. A stronger per-input hand
envelope and the larger feasible search reference remain visible.

All four learned runs (two GNN, two MP-OFF) plus hand and reference are
registered for each of sixteen sealed test workflows: **96 actual HeROsim runs
and 24,576 operations**, even if offline learning looks poor. Native, independent
SimPy, actual task completions and resource events must agree. Report paired
medians and intervals over environments, treating training draws as draws,
and the complete planner latency. A GNN win requires positive intervals
against MP-OFF and hand control. This two-seed/16-environment trial is
exploratory; a favorable result needs a larger registered replication.

## 2026-09-22 — Corpus and matched training receipt

The generated corpus has **96 unique inputs**: 64 training, 16 validation and
16 sealed test cases. Independent validation replays each saved candidate,
checks input and array fingerprints, and confirms the sealed cases contain
no labels or search plans. It passes. The required trainer suite passes **37
tests**; the new trainer's graph/feasibility and same-seed weight tests pass
three tests. CPU smoke training passed before the full runs.

Four full CPU trainings completed through `run_experiment.py`, each logging an
offline W&B run and saving weights with a `.contract.json` sidecar under
`models/dag_reservation_learning_v1/`. The selected validation mean summed
job completions are GNN seed 101 **9664.5 ms** (epoch 13), GNN seed 102
**9672.1875 ms** (epoch 2), MP-OFF seed 101 **9664.3125 ms** (epoch 6),
and MP-OFF seed 102 **9740.625 ms** (epoch 11). Reloaded weights reproduce
all four figures exactly. Validation is therefore essentially tied; no
model-class conclusion follows from it. The source, corpus and checkpoint
fingerprints are carried into the fresh live gate, which is underway.

## 2026-09-22 — Exploratory trained live result

All **96 registered actual HeROsim runs / 24,576 operations** complete on
sixteen sealed workloads. All four learned paths meet the 250 ms budget;
maximum observed times are 244.87–244.93 ms. The separate resource-event
audit **PASSES** all **49,152** starts/completions and their placements,
durations, setups and release constraints. The full fixed-step reference
reproduction audit is running.

Mean summed job completions are GNN 8201.13 ms averaged over its two draws,
MP-OFF 8200.66 ms, the calibration-selected hand control 8014.75 ms, the
stronger per-input budget-eligible hand envelope 7886.25 ms, and the slow
feasible reference 7614.31 ms. Paired across workloads, GNN's median gain
over MP-OFF is **0.00%** (interval [0.00%, 0.00%]); against selected hand
search it is **−1.81%** (interval [−2.74%, −0.68%]); against the stronger hand
envelope it is **−3.62%**. Against the single equal-budget `portfolio_cool`
hand arm, which uses the same mixed-decoder search after the same full hand
initializer, GNN's median gain is **0.00%** and thirteen of sixteen costs match.
The gap to the calibration-selected hand arm also reflects its different
non-delay search mode, rather than an accepted learned action. These intervals
resample environments at the fixed two trained draws; the trial is
exploratory, not a powered model-class claim.

The guarded learner chooses the hand initializer on **all 16/16 test inputs
for all four checkpoints**. Post-hoc replay of raw proposals confirms zero
accepted per model: GNN raw schedules are median 11.30%/9.12% worse than
that initializer, and MP-OFF schedules 11.44%/11.36% worse. After the same
search, all four learned arms yield the same placement and rank within each
of the sixteen inputs; decoder choice changes the cost on one input. This
explains the GNN/MP-OFF tie and locates the failed learned contribution in
one-shot complete-plan imitation and its guard, rather than
showing a harmful message-passing effect in accepted actions.

Artifacts are under `simulation_data/gnn_environment_search_v1/` in
`dag_reservation_learning_corpus_v1/`, `dag_reservation_learning_gate_v1/`
and `dag_reservation_learning_proposal_diagnostic_v1.json`; checkpoints and
sidecars are under `models/dag_reservation_learning_v1/`. The model did train
and complete a fresh live gate, but **no GNN win** occurred. The earlier
8% funding gate was waived only by the user's explicit request. The action
space remains open: this trial neither trains a sequential reservation policy
nor tests residual graph information after a strong hand proposal model.

The full fixed-step audit **PASSES**: all **2,359,296** search proposals
reproduce, source/input/artifact/native/checkpoint fingerprints match, and
all 96 served plans match independent SimPy. The focused new and predecessor
DAG suites pass 22 tests; the trainer determinism gate passes 37.
