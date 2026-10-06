# dag_resume_s0_v1 — fixed-history continuation feasibility

**Status:** `CLOSED` (2026-09-23) — RESUME CONTRACT PASSES; CONTINUATION RECIPE NO-GO.
The agent fixed the protocol before qualification generation under the user's
authorization to test the recommendation and make one bounded adjustment if it
fails. The numerical bars are agent-selected, not separately user-signed.

**Outcome.** Resumable fixed-history continuation passes independent and actual
HeROsim replay, but the one-decision search and its predeclared two-decision retry
both achieve **0% median gain over matched hand search**, across sixteen fresh
workflows each. The larger bounded reference also has 0% median gain in both
phases. All 128 fresh live runs and 3,677 evaluated continuations audit. This
closes the tested four-rule continuation recipe, not all stepwise reservations,
the resumable implementation or the scheduling optimum. No model was trained;
materialized live replay is not an online policy callback.

**Parent:** [mixed_dispatch_v1](mixed_dispatch_v1.md).
**Protocol:** [dag_resume_s0_v1.json](../../experiments/dag_resume_s0_v1.json).
**Implementation:** [dag_resume.py](../../src/placement/radical/dag_resume.py),
[screen runner](../../scripts_cosim/dag_resume_s0.py).

## Record

- [2026-09-23 — Primary and predeclared retry results](#2026-09-23--primary-and-predeclared-retry-results)
- [2026-09-23 — Bounded feasibility protocol](#2026-09-23--bounded-feasibility-protocol)

## 2026-09-23 — Primary and predeclared retry results

The [primary report](../../simulation_data/gnn_environment_search_v1/dag_resume_s0_v1_primary/report.json)
covers seeds 272000–272015. Candidate and matched hand retain the incumbent on
all sixteen cases, so candidate gain is zero throughout. The larger reference
improves one case from 4,691 to 4,649 (0.8953%); its median gain is still zero,
below the 8% bar. The 3% candidate bar also fails. The primary reference exhausts
its generated continuation family before its 128-evaluation cap, so this result
cannot be attributed solely to the served wall-clock limit.

The [retry report](../../simulation_data/gnn_environment_search_v1/dag_resume_s0_v1_retry/report.json)
covers disjoint seeds 273000–273015. Every served beam scores four first-stage
proposals and three or four second-stage proposals, retaining four distinct beam
schedules. This is a real second-decision test. Candidate and reference tie the
matched hand on all sixteen cases: both median gains remain zero. All three
search arms improve the incumbent on seed 273015 from 4,020 to 3,863 (3.9055%);
they retain it on the other fifteen. The isolated gain is shared with the hand
control and supplies no candidate advantage.

Candidate and hand therefore retain the common incumbent on **31/32** fresh
workflows. Raw worse proposals are rejected by the incumbent guard; selected-arm
ties must not be interpreted as raw proposals all having equal quality. Releasing
unstarted reservations and changing their continuation often makes the existing
strong schedule worse. This finite family exposes no material additional gain;
it does not bound the true optimum or establish general training futility.

The [primary audit](../../simulation_data/gnn_environment_search_v1/dag_resume_s0_v1_primary/AUDIT.json)
passes 1,054 feasible evaluated continuations and 64 actual-HeROsim runs. The
[retry audit](../../simulation_data/gnn_environment_search_v1/dag_resume_s0_v1_retry/AUDIT.json)
passes 2,623 continuations and another 64 live runs. Together, qualification
checks **3,677 continuations, 128 live runs and 16,384 executed operations** on
32 unique physical workflows. Every final start, completion, placement and setup
event matches; source hashes stay unchanged and both phase reports mark timing
valid. The final [calibration audit](../../simulation_data/gnn_environment_search_v1/dag_resume_s0_v1_calibration_v3/AUDIT.json)
adds 355 continuations and eight live runs on two separate inputs, giving 4,032
rechecks and 136 live runs overall. Calibration is not qualification evidence.
The final focused regression run passes 73 tests.

The Python implementation spends substantial time validating each continuation,
so served search evaluates few proposals; its timing is not an optimized-serving
claim. The experiment uses fully known deterministic, zero-I/O workflows and
materializes each chosen continuation for live replay. It establishes executable
history-preserving plans, not a live online replanning callback, transfer-aware
planning, or a GNN advantage.

**Decision.** Stop the unchanged one/two-decision, four-rule continuation recipe
under this physics. Keep the tested resumable-state machinery available. Further
work needs a distinct decision mechanism and independently demonstrated useful
headroom; compilation or learning to rank this unchanged family is not supported
by these measurements. The parent environment stays active.

## 2026-09-23 — Bounded feasibility protocol

This tests a new resumable-state contract, rather than extending the stopped
[static block-pool recipe](dag_block_proposer_128_s0_v1.md). A checkpoint retains
executed events and resource commitments while proposals change future
placements, priorities and waiting decisions. The objective is summed job
completion time. Workflows are deterministic, fully announced, zero-I/O per-job
DAGs: resuming supplies no previously unavailable information.

The machine-readable protocol fixes 16 jobs × 8 operations on 12 hosts, a common
250 ms starting-plan budget, and one checkpoint at 25% completed operations.
The checkpoint is after completions and before new dispatch at that timestamp.
Calibration seeds 271000–271001 are excluded from qualification. Primary seeds
272000–272015 compare branching on a ready operation and eligible host, or waiting
for the next event, with a hand search over the same state and information.
Continuation rules are incumbent order, short job, critical path and graph2.
Both served arms have a 32-evaluation and 250 ms decision limit; construction and
resume revalidation count. A larger feasible reference has 128 evaluations and
includes incumbent and served solutions. It is not an optimality certificate.

The bars are median paired improvements across physical workflows versus the
matched hand arm: at least 8% for the bounded reference and 3% for the candidate.
Improvement over the common incumbent is reported separately. Passing only
qualifies further diagnostics, never training. If either bar fails, exactly one
predeclared algorithm retry uses a width-four beam,
branches again after the next completion and evaluates on disjoint seeds
273000–273015. Served limits stay unchanged; the retry reference allows 256
evaluations. No further algorithm retries or model training are in this screen.

Before qualification, engineering calibration on the same two calibration seeds
corrected the snapshot phase, validated resource-feasible actual next actions and
deduplicated beam schedules. A second calibration exposed zero second-stage
evaluations inside the served retry budget. The protocol therefore reserves
second-stage budget by limiting served first-stage scoring to four evaluations,
with diagonal action/rule ordering. These are disclosed calibration amendments,
not fresh population results. The retained stage counts must demonstrate that the
retry actually scores second decisions before it is described as a depth-two test.

Every feasible evaluated continuation must preserve history and match its full
from-zero native materialized replay. Final arms also undergo independent SimPy
comparison. Regardless of offline outcome, every qualification workflow replays
incumbent, hand, candidate and reference in actual HeROsim: 64 primary runs and
64 more if the retry is triggered. Audit every start, completion, placement and
setup event, preserve candidate traces, input identities, source hashes and
timings. Timing violations invalidate performance claims.

Materialized live replay checks execution of the selected continuation; it does
not implement or validate a live online GNN callback. Python timing is not an
optimized deployment result. The earlier six-operation, 720-order scratch
example is a constructed mechanism illustration, excluded from fresh evidence;
hand search already solves that example.
