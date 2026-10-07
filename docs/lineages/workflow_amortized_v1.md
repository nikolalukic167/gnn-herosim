# workflow_amortized_v1

> Produced on local-record-2026-10-06 at afea04bd; code not merged; old physics.

**Status:** `CLOSED` (2026-09-21) — HEADROOM-REAL / LEARNED-PILOT-NEGATIVE.
Registered before qualification and training; the checkpoints were frozen before the
256-case held-out gate.

**Outcome.** A complete-workflow environment passed the CPU headroom gate, but
nine trained checkpoints did not produce a GNN win. Across 256 fresh workflow
instances and 3,001 full HeROsim replays, GNN RTT is 0.39% higher than its MP-off
twin (inconclusive difference), 2.54% higher than the hand-feature MLP, and 7.15%
higher than the best tested hand control within 2 ms, using paired medians averaged
over training draws. GNN planning fits the budget. The independent artifact audit
reconciles all 240,080 completed task records. This closes this one-shot supervised
pilot, not the environment family or GNN scheduling generally.

**Question.** Can a learned graph planner improve placement quality at matched
planning latency? This is narrower than beating unrestricted search: slow controls
remain visible, and the 2 ms budget is a laboratory constraint, not a production SLA.
The earlier private-platform search nodes remain closed.

**Entry points:** [protocol](../../experiments/workflow_amortized_v1_protocol.json),
[planning engine](../../src/placement/workflow_planning.py),
[real-engine adapter](../../scripts_cosim/workflow_live.py),
[qualification](../../scripts_cosim/workflow_qualification.py),
[data generation](../../scripts_cosim/generate_workflow_data.py),
[model](../../src/policy/workflow/model.py),
[trainer](../../src/policy/workflow/train.py),
[validator](../../scripts_cosim/validate_workflow_data.py),
[evaluator](../../scripts_cosim/evaluate_workflow.py).
Artifacts live under `simulation_data/gnn_environment_search_v1/`.

## Record

- [2026-09-21 — Frozen pilot and complete live gate](#2026-09-21--frozen-pilot-and-complete-live-gate)
- [2026-09-21 — Qualified complete-workflow pilot](#2026-09-21--qualified-complete-workflow-pilot)

## 2026-09-21 — Qualified complete-workflow pilot

**Environment.** Eight jobs, ten serial operations per job, four machines, two
eligible machines per operation. Operation durations are 1–30 ms with small seeded
jitter. A complete group manifest is available before planning; no future placement
or queue state is provided. All roots arrive at time zero. HeROsim uses existing
single compute slots with deterministic private replicas and zero transfer/cold-start
costs. Its FIFO execution realizes the event-driven planning engine. Summed task RTT
and summed complete-workflow completion times coincide for these serial chains.
There is no fixed-horizon return, no added co-tenancy dilation, and no link-fabric
physics. This changes the environment from the earlier pointwise co-simulation target.

**Initial adapter defect.** The first attempt reused physical platform IDs in the
initial replica map; the initializer allocated each ID only once, leaving most task
types without replicas. That attempt timed out and produced no valid evidence.
The adapter now gives each eligible operation a private replica and shares the
node's single compute slot. Real-engine and analytical costs match on tested plans.

**Gate before training.** Seeds 60520–60535 were fixed in the protocol before the
run. Hand controls include ECT, four remaining-work-aware greedy variants, beam
widths 1/2/4/16/64 and full-completion rollout search. The teacher takes the best
feasible complete plan across those methods and annealing. A wider beam is not
assumed to dominate a smaller one; every returned plan is independently replayed.
The gate compares against an intentionally strong per-instance envelope of tested
hand methods whose measured runtime fits 2 ms. It grants free algorithm selection
within that envelope, favoring the control. Timing noise near the cutoff remains a
limitation; the final evaluation must repeat timings and report the whole frontier.

Median teacher gain is 11.3463%, bootstrap median lower bound 7.0316%, against a
pre-training 5% bar. The selected ECT, budget-control and teacher plans are replayed
in HeROsim; both complete-workflow JCT and total task RTT match the analytical
values. The source hashes and inputs are in `workflow_gate/`. Gate methods were
profiled without search-trace I/O; corpus generation timings include tracing and
must not substitute for inference timing measurements.

**Representation check.** `workflow_graph_witness.json` contains a small exact
example: swapping two operations in other jobs preserves the first operation's
pointwise features but flips its optimal host from 3 to 2. Exhaustive enumeration
covers 512 feasible assignments in each version. This establishes an information
need relative to that feature contract, not an impossibility for arbitrary hand
features. Model tests verify that MP can receive other operations' information and
the MP-off path remains pointwise. The hand arm adds ECT/beam-1 plan columns.

**Data and training contract.** Fixed disjoint seed ranges produce 512 training,
128 validation and 256 sealed test instances. Every complete evaluated search plan
is retained in each case's `placements/placements.jsonl`; these are searched-plan
traces, **not exhaustive co-simulation sweeps**. Labels are best-observed complete
plans, never called certified optima. Source-instance identity and file hashes are
verified. No test instance is used for training or checkpoint selection.

Three arms: a new `WorkflowAssignmentNet` with message passing, its separately
trained MP-off twin on identical features, and an MP-off hand-feature arm augmented
with cheap plan columns. These are not the old `jb2_gnnedge0` or rollout checkpoints.
All use the same mean complete-validation-RTT checkpoint selector. Initial training
seeds are 101/102/103; every run uses `run_experiment.py`, logs offline to WandB and
saves a contract sidecar. The user authorized sufficient compute for the full
investigation; budget does not lower the evidence bar.

**Registered interpretation.** Evaluate raw RTT and measured planning latency on
one CPU thread, including feature building and decoding. Report the 1/2/5/10/50/100/
1000 ms frontier and unrestricted strong-search quality. A pilot GNN win requires
at least 5% lower RTT than both learned controls and the best applicable hand control,
with positive paired uncertainty bounds; otherwise report the result as negative
or inconclusive. Three training seeds are a pilot, not a powered model-class claim.
No post-hoc threshold reduction or test-driven tuning is allowed.

**Logging amendment (2026-09-21).** Automatic approval review rejected online W&B uploads because the destination and payload were not explicitly approved. All pilot runs retain offline W&B logs; no upload is attempted. The 896-case corpus passed actual source, trace, cache and cross-split uniqueness checks.

**Training receipt.** SLURM array 800138 completed all nine runs on NVIDIA A40,
using isolated Git snapshot `5c46fc72d3daa0ebb20edb27489a01ddc4105025`.
The existing cluster checkout was not modified. All three arms passed repeated
same-seed CUDA training checks in every job. The required trainer suite passed
22 tests; workflow/runner/record checks passed 443 tests before the pilot.
Actual source hashes and all training/validation cache hashes matched before
submission. All nine selected checkpoints reproduce their recorded GPU validation
mean RTT on the local CPU within 1e-6 ms, with matching model/planner/trainer sources.

GNN selected epochs are 13/15/17; MP-off selects 91/64/71; hand MLP selects 1/75/1.
GNN validation assignment accuracy is approximately 84%, versus a 50% choice floor.
Read that with complete-plan RTT: its validation means are 1955/1959/1968 ms,
versus 1957/1951/1962 for MP-off and 1931/1958/1944 for the hand arm.
The seed-101 selected GNN also loses to ECT on its own training cases
(1969 versus 1929 ms), so simply calling this a data-generalization failure
is unsupported. Lower cross entropy alone does not establish better joint plans.

Artifacts: `workflow_training_receipt.json`, `workflow_checkpoint_audit.json`,
`workflow_learning_audit.json`, `workflow_training_v1.bundle`, `workflow_wandb/`,
and `workflow_training_logs/` under the experiment artifact root. Deployable pilot
weights and sidecars are under `models/workflow_amortized_v1/`.


## 2026-09-21 — Frozen pilot and complete live gate

**Protocol and coverage.** The 896 unique generated problems split into 512 train,
128 validation and 256 held-out test cases. All actual problem, trace, cache and
source hashes passed validation; declared seeds alone were not trusted as proof
of uniqueness. Models use seeds 101/102/103 in all three arms. All nine selected
weights and contracts were frozen before opening test data for evaluation.
The final gate contains 3,001 live simulations with 80 completed operations each,
covering every learned arm on every test instance plus ECT, the timed control
winner and the strongest tested search plan. Analytical and live costs agree.

**Paired outcome.** Positive values in the following table would favor the GNN.
For each environment, average the paired percentage gain across three matched
training draws, then take the median across 256 environments. The interval
resamples both environments and training draws; repeated training draws are not
counted as additional independent environments.

| Comparator | Median GNN gain | Hierarchical 95% interval |
|---|---:|---:|
| MP-off, same features | −0.39% | [−1.71%, +1.04%] |
| Hand-feature MLP | −2.54% | [−4.16%, −0.71%] |
| Best tested hand control within 2 ms | −7.15% | [−8.32%, −5.87%] |

All comparisons miss the registered +5% bar. The arithmetic mean RTT across
instances and draws is 1986.83 ms for GNN, 1989.53 ms for MP-off and 1944.94 ms
for hand MLP. The first two raw means slightly reverse the paired-median ordering;
do not substitute a ratio of pooled means for the registered paired statistic.
These three training draws constitute a pilot, not a powered model-class theorem.

**Planning cost.** On one CPU thread of the local AMD EPYC 7313 host, five repeated
complete planning calls include feature construction, forward pass and argmax.
GNN median time is 1.14–1.17 ms across seeds and its per-case-median p95 is
1.16–1.19 ms. MP-off costs approximately 0.63 ms. Hand MLP costs 2.29 ms because
its engineered columns run ECT and beam-1; it is reported as a stronger information
control despite exceeding the 2 ms budget. GNN still loses to the budget-compliant
hand search envelope, independently of that slower learned control.

| Hand planning budget | Mean full-workflow RTT |
|---|---:|
| 1 ms | 1854.81 ms |
| 2 ms | 1849.62 ms |
| 5 ms | 1744.76 ms |
| 10 ms | 1744.76 ms |
| 50 ms | 1643.78 ms |
| 100 ms | 1643.78 ms |
| 1000 ms | 1634.20 ms |

Every frontier point covers all 256 cases. This is the best **tested** per-instance
hand method, granting free method selection to the control. It is not a claim
about optimal compiled solvers or every possible hand algorithm. The unrestricted
best-observed teacher also averages 1634.20 ms and is not a certified optimum.
Timing and quality are separate measurements: planning wall time is not inserted
as a simulated arrival delay. The fixed four-machine, eight-by-ten workflow family
is an IID-instance test, not topology-size transfer or production trace validation.

**Debugging and verification.** The small exact witness preserves current-task
pointwise features while changing its optimal first host; the MP-on representation
responds to the changed graph. Separate tests verify live FIFO replay, repeated
training determinism, source mismatch rejection and cross-split actual-instance
duplicate rejection. The final audit reconstructs every input, checks every served
placement and operation execution time, and independently sums completion minus
dispatch time. Source fingerprints and frozen checkpoint hashes still match.
The generic curve reader initially ignored the new flat workflow metric names;
it now reads offline configuration records, maps these metrics, displays the
recorded 50% chance floor and identifies the selected RTT epoch. This reader fix
did not change training, checkpoints or evaluation. Its regression tests pass.

**Artifacts.** `workflow_test_gate/read.json` is the authoritative comparison;
`AUDIT.json` reconciles 256 instances, 3,001 live runs and 240,080 task records.
`frozen_checkpoints.json`, `protocol_before_run.json`, `venue.json`, per-instance
inputs, reads, logs and `placements/placements.jsonl` retain the evidence.
`workflow_corpus/FINAL_VALIDATION_RECEIPT.json` is the reproducible full corpus
audit. The evaluator and artifact auditor are
[scripts_cosim/evaluate_workflow.py](../../scripts_cosim/evaluate_workflow.py) and
[scripts_cosim/audit_workflow_evaluation.py](../../scripts_cosim/audit_workflow_evaluation.py).
Offline W&B transaction logs and their curve reads are retained locally.

**Decision.** Close this supervised one-shot assignment pilot as negative. The
CPU gate established planning headroom; it did not establish that this model and
loss can learn to exploit it. Do not enlarge this same recipe on the strength of
teacher headroom or assignment accuracy alone. Any successor needs a demonstrated
training/validation full-plan improvement, the same strong controls and measured
planning cost, followed by a new sealed test set. The consumed test cases must not
be reused for tuning. Earlier closed peer-lookahead lineages remain closed.
