# workflow_proposal_v1

**Status:** `CLOSED` (2026-09-22) — TIMED-PROPOSAL-PILOT-NEGATIVE.
Registered before validation selection and the fresh live gate.

**Outcome.** Exact whole-plan guarding does not establish a GNN advantage at the
registered 2 ms budget. On 32 fresh workflows and 384 full live simulations,
the guarded GNN is essentially tied with MP-off and has 4.97% higher median paired
RTT than the best tested budget-compliant hand control. It beats actual Knative
by 56.92%, while the hand control beats Knative by 59.50%. These are fixture-specific,
information-asymmetric comparisons, not proof that message passing provides the gain.
All 30,720 completed task records and retained proposal costs pass the artifact audit.

**Parent:** [workflow_amortized_v1](workflow_amortized_v1.md).
**Registration:** [protocol](../../experiments/workflow_proposal_v1.json).
Artifacts: `simulation_data/gnn_environment_search_v1/workflow_proposal_v1/`
retains validation and the failed first live attempt. The authoritative completed
gate is `simulation_data/gnn_environment_search_v1/workflow_proposal_v1_retry/`.
**Entry points:** [gate](../../scripts_cosim/workflow_proposal_gate.py),
[search](../../src/placement/workflow_proposals.py),
[Knative observer](../../scripts_cosim/workflow_knative.py),
[auditor](../../scripts_cosim/audit_workflow_proposals.py).

## Record

- [2026-09-22 — Completed fresh live gate](#2026-09-22--completed-fresh-live-gate)
- [2026-09-22 — Registered proposal-search gate](#2026-09-22--registered-proposal-search-gate)

## 2026-09-22 — Registered proposal-search gate

Reuse all nine parent checkpoints and their contracts. Start from ECT; accept the
model's complete argmax only if its exact full-plan RTT is better. Propose one,
two or four assignment flips, weighted by the model's probability of the alternate
host, and accept only whole-plan improvements. The hand controls use the same
search code and uniform or duration-derived probabilities, with the same available
workflow manifest. Include the parent's ECT, load-aware and beam controls.

Select the proposal count separately for each checkpoint on 32 already available
validation cases, using complete RTT subject to a 2 ms single-thread p95 planning
budget. Feature construction, model inference, all scoring and search are timed.
If no count fits, retain the fastest solely for comparison and fail its budget gate.
Freeze all selections before generating 32 fresh seeds 88000–88031. The parent's
consumed test set is excluded. Verify source bytes, checkpoint hashes and actual
problem uniqueness before simulation. Preserve every scored complete proposal.

The primary remains a 5% GNN gain over MP-off, hand MLP and the best tested hand
control fitting the budget, with positive paired uncertainty bounds. Report the
1/2/5/10 ms hand frontier. Actual KnativeNetwork is an additional live baseline;
it lacks the declared full-workflow lookahead used by the planning controls, so a
win over Knative alone cannot establish a GNN-specific advantage.

Run the full fresh live gate even if validation is negative. This is a small
serving experiment with frozen checkpoints, not a new training run; no GPU or
W&B upload is needed. A negative result stops this decoder recipe, not all
structured decoding or the workflow environment family.


## 2026-09-22 — Completed fresh live gate

**Execution.** Reused all nine frozen parent checkpoints; no training or GPU job
was run. Validation uses seeds 63000–63031. Search-count selections are frozen
before constructing seeds 88000–88031, and actual source identities were checked
against all 896 parent inputs. The parent's consumed test performance was not
used for tuning. The live gate runs nine selected learned configurations, ECT,
the best tested timed hand control, and actual KnativeNetwork on every fresh case.
All 384 runs complete; selected analytical plans agree with HeROsim costs.
The registered 5% bar and 2 ms budget were unchanged.

**Budget finding.** All GNN draws select zero additional perturbations: feature
construction, inference, ECT construction and exact whole-plan guarding consume
about 1.80 ms, with test p95 1.83–1.84 ms. No tested positive proposal count met
the validation p95 budget. MP-off fits four perturbations at a similar total time.
The hand-feature MLP cannot meet 2 ms even at zero proposals (about 2.90 ms);
its fastest configuration is retained as the registered stronger information
control, explicitly marked non-qualifying. Thus this result primarily rejects
this proposal-search recipe **at 2 ms**, not guided search with arbitrary latency.

| Comparator | Median paired GNN RTT gain | Hierarchical 95% interval |
|---|---:|---:|
| MP-off with four proposals | −0.16% | [−0.91%, 0.00%] |
| Guarded hand-feature MLP | +0.08% | [0.00%, +0.98%] |
| Best tested hand control within 2 ms | −4.97% | [−7.19%, −3.48%] |
| Actual KnativeNetwork | +56.92% | [+53.06%, +59.90%] |

Positive gains favor the GNN. Each environment's paired percentage is averaged
across the three training draws before taking the median. Bootstrap intervals
resample environments and training draws; these are 32 independent workflow
instances, not 96 independent environments. This is a small decoder pilot.
The hand control independently beats Knative by a median 59.50%.

**Where the apparent Knative win comes from.** GNN guarding takes the better
complete plan of ECT and model argmax. It improves ECT in only 11/13/10 of 32 cases
for seeds 101/102/103, with zero median gain in each draw; mean improvements are
2.05%/2.05%/1.32%. Most cases therefore use the hand incumbent or tie its cost.
Do not call the +56.92% Knative contrast a pure neural-model or MP advantage.
The unmodified Knative policy has per-arrival shortest-platform-queue information,
whereas the planning arms see the declared full workflow and execution tables.
Private replicas and shared node compute slots make platform queue lengths a weak
load signal in this fixture. No claim of production superiority follows.

**Broader hand frontier.** Mean full-workflow RTT is 1854.20 ms at 1 ms planning,
1806.23 at 2 ms, 1749.36 at 5 ms, and 1743.36 at 10 ms, with all 32 cases represented.
The 5/10 ms points are hand-search comparisons, not fresh tests of new GNN settings.
Model-averaged RTT is 1896.92 ms for the guarded GNN, 1878.57 for MP-off,
1918.56 for hand MLP and 4456.05 for Knative. Planning time is measured separately
from simulated RTT. Do not compare these fresh-case means directly to the
parent's different 256-case test set to claim an improvement.

**Harness faults, not results.** The initial Knative smoke call used the registry
policy name where the co-simulation adapter requires `kn_network_kn_network`;
it failed before producing a result and was corrected. The first full gate then
completed validation but its first replay failed the peer scheduler's explicit
physics guard because the wrapper omitted `HEROSIM_PEER_EXCHANGE=1`. There are
no peer edges in these inputs; the flag is required by the fixed-plan replay
wrapper. The original failed directory is retained. The retry restores the
flag and reuses byte-identical frozen selections and checkpoint hashes; no
validation reselection or test-driven tuning occurred. The environment regression
test now exercises both the actual Knative and forced-plan paths.

**Verification.** The independent auditor checks frozen weights/contracts, retry
selection identity, actual source inputs, all executed hosts and durations,
completion-minus-dispatch sums, complete operation coverage and every retained
searched-plan cost. It passes 32 cases, 384 live runs and 30,720 completed tasks.
Timed runs exclude trace I/O; an untimed deterministic replay records the same
unique proposals and is checked against the timed output. Proposal counts,
eligibility, incumbent preservation, trace neutrality and both live adapters
have focused regression tests. Record-hygiene migration handling is documented
in [gate-tools](../gates/gate-tools.md#2026-09-22--research-guidance-after-the-agentsmd-migration).

**Decision.** Close this exact checkpoint/decoder/budget combination. It improves
robustness by retaining ECT, and it answers the Knative comparison for this hybrid,
but it does not clear the controlled GNN bar. Further work needs a distinct way
to allocate planning compute or exploit joint structure, with its own validation
and fresh sealed gate; another run of these fixed settings is not justified.
