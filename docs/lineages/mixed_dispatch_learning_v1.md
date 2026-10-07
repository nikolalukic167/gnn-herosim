# mixed_dispatch_learning_v1

> Produced on local-record-2026-10-06 at afea04bd; code not merged; old physics.

**Status:** `CLOSED` (2026-09-22) — TOTAL-RANK-IMITATION-NEGATIVE.
Registered before data generation and training; completed the required live gate.

**Outcome.** Nine matched models on 192 new workloads fail the 5 ms / 5% pilot.
GNN is 19.03% slower than the budgeted hand rule, 0.91% slower than MP-OFF,
and 1.59% slower than the hand-graph model by the registered paired median.
Inference fits at 3.86–3.88 ms p95, so timing is not the cause. All 384 actual
HeROsim evaluations pass audit. Close normalized total-rank regression with a
direct priority decoder, not the ordering environment or state-dependent
dispatch learning.

**Parent:** [mixed_dispatch_v1](mixed_dispatch_v1.md).
**Protocol:** [registration](../../experiments/mixed_dispatch_learning_v1_protocol.json).
**Artifacts:** `simulation_data/gnn_environment_search_v1/mixed_dispatch_corpus/`,
`mixed_dispatch_models/`, `mixed_dispatch_training_logs_retry/`,
`mixed_dispatch_curves/`, and `mixed_dispatch_model_gate/`.

## Record

- [2026-09-22 — Nine-model live result](#2026-09-22--nine-model-live-result)
- [2026-09-22 — Registration](#2026-09-22--registration)

## 2026-09-22 — Registration

The user authorized this pilot even though the constructive witness is equally
distinguishable by hand graph features. The protocol freezes 128 training
workloads (129000–129127), 32 validation workloads (130000–130031), and 32
sealed live-test workloads (131000–131031). Three seeds each train a GNN, a
separately initialized MP-OFF model, and an MLP with fixed typed one/two-hop
summaries. All arms share the same fixed affinity/descent64 placement and
teacher labels; there is no architecture or objective sweep.

The teacher is the best final feasible schedule among the budgeted `srpt:128`
control and 8192 priority-swap proposals from SPT and SRPT. It is not a certified
optimum. Models regress normalized teacher ranks with SmoothL1 and serve one
direct predicted priority vector without post-model priority search. Full
serving includes placement construction, features, inference, priority sorting,
and native scoring. Checkpoint selection uses mean directly served validation
objective, including the untrained epoch. Every run logs to offline W&B and
retains weights plus data/source/architecture contracts.

Validation must improve mean objective by at least 5% over `srpt:128`, with all
GNN draws at or below 5 ms p95. Regardless of validation, all nine selected
models are frozen before the 384-run live gate on the sealed test set: nine
learned arms, `srpt:128`, teacher, and plain SRPT on each of 32 workflows. A GNN
claim requires at least 5% median paired improvement over MP-OFF, the hand-graph
model, and the hand control, positive bootstrap lower bounds, and all GNN paths
within 5 ms. Training draws are paired within physical environments.

## 2026-09-22 — Nine-model live result

**The pilot fails validation and the sealed live gate.** The corpus audit passes
192 unique physical workloads, disjoint splits, reproducible features and
targets, retained final teacher/control plans, and all source/data fingerprints.
The teacher mean on validation is 1774.69 ms and `srpt:128` is 1959.22 ms.
Selected GNN means are 2232.66 / 2204.53 / 2229.91 ms: 12.52–13.96% slower
than the hand rule. MP-OFF is 13.45–17.19% slower; hand-graph is 10.76–11.54%
slower. All validation GNN timings fit at 4.08–4.10 ms p95, but none approaches
the objective bar.

Selected epochs are GNN 24/17/23, MP-OFF 37/19/34, and hand-graph 34/28/30.
All nine offline W&B transaction logs were read. Training rank loss decreases,
but selected validation rank MAE remains 0.313–0.328 of the complete 48-operation
rank range. Checkpoints are selected by actual validation objective, so mismatch
between rank error and scheduling value cannot silently choose the wrong epoch.
The first GNN launch failed before any gradient step because batched rank vectors
were not reshaped to 8×6 for the checked native API. Its log is preserved in
`mixed_dispatch_training_logs/`; a focused regression test covers the repaired
interface, and the complete pilot restarted from scratch in the retry directory.

The nine checkpoints, data and runtime sources were frozen before test access.
The live gate runs **384 actual HeROsim simulations and 18,432 completed
operations**. Every arm uses the same physical workload and placement contract;
every task placement, completion time, aggregate objective and arm/environment
count passes independent SimPy and artifact audit.

| GNN comparison | Median gain | Mean paired gain | Bootstrap median 95% interval |
|---|---:|---:|---:|
| MP-OFF | −0.907% | −0.836% | [−4.511%, 3.195%] |
| Hand-graph model | −1.595% | −2.281% | [−5.372%, 1.392%] |
| `srpt:128` | −19.025% | −17.799% | [−21.715%, −14.818%] |
| Feasible teacher | −28.553% | −31.452% | [−34.791%, −26.979%] |

GNN mean costs are 2273.00 / 2269.53 / 2321.41 ms; MP-OFF means are
2323.75 / 2214.94 / 2298.38; hand-graph means are 2269.03 / 2244.66 /
2230.75. The hand rule is 1945.53 ms, plain SRPT 2204.09 ms, and the teacher
1743.75 ms. GNN full serving takes 3.860–3.881 ms p95, MP-OFF 3.823–3.841,
hand-graph 3.900–3.937, and `srpt:128` 4.208. Decision wall time is measured
separately from simulated execution time, as in the parent gates.

### Debugging the decoder

The stored target priority exactly reproduces every retained teacher objective,
and predicted ranks are valid permutations accepted by native, independent
SimPy and HeROsim replay. There is no inversion, shape, label-index, feature
leakage, or checkpoint mismatch after the repaired pre-training shape defect.

A posthoc frozen-weight diagnostic adds 64 exact priority swaps after each
prediction. It is not a registered live arm or positive claim. GNN means improve
to 2020.19 / 2008.88 / 2045.09 ms and usually fit 5 ms p95, but remain
3.26–5.12% slower than `srpt:128`. MP-OFF remains 3.72–6.23% slower and
hand-graph 3.38–3.74% slower. Small refinement therefore does not rescue the
recipe or reveal a control-surviving message-passing advantage.

The technical failure is meaningful: a teacher's complete priority permutation
contains many relative orderings between operations that never compete in the
same ready set. Smooth rank regression spends capacity matching those arbitrary
relations, while small rank errors can change lock/setup arbitration sharply.
The result does not close learning state-dependent choices over the current
ready set, outcome-aware ranking losses, or joint placement/order policies.
Those are materially different objectives and serving contracts.

### Implementation and checks

- `src/policy/dispatch_priority/model.py` and `train.py`: matched encoders,
  deterministic training, direct priority serving, and actual-objective selection.
- `scripts_cosim/generate_dispatch_data.py`: corpus generation and full audit.
- `scripts_cosim/run_dispatch_pilot.py`: config-driven offline-W&B launcher.
- `scripts_cosim/evaluate_dispatch_models.py`: validation, timing, live gate,
  source/checkpoint freeze and independent audit.
- `scripts_cosim/diagnose_dispatch_models.py`: frozen 64-swap diagnostic.
- `tests/test_dispatch_priority.py`: rank permutation, batch reshape,
  declared-input isolation, model sensitivity and deterministic training checks.

The repository-wide trainer determinism suite passes 28 checks; focused
dispatch/model/record suites pass. Historical artifacts remain unchanged.

**Hard stop.** Do not scale the unchanged hidden16/two-layer, 40-epoch
normalized-total-rank SmoothL1 model with direct priority decode on this holdout.
A later ordering learner needs a state/action target or loss aligned to actual
ready-set decisions, matched MP-OFF and hand-graph controls, full 5 ms serving,
and a fresh live gate. A timing-only change cannot repair the measured objective
loss.
