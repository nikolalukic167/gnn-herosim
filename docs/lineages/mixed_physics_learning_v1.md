# mixed_physics_learning_v1

**Status:** `CLOSED` (2026-09-22) — NO-CONTROL-SURVIVING-GNN-WIN.
Registered before data generation and training; completed the real-HeROsim gate.

**Outcome.** The combined setup/lock/power environment is integrated and has real
planning headroom, but the nine-model pilot fails: GNN median gains are zero over
MP-OFF, the hand-graph MLP and the no-model decoder, and it is 4.97% slower than
the calibrated hand search. All 768 held-out live runs pass audit. GNN beats
Knative by 34.26% in this declared-workflow fixture; that does not establish a
GNN-specific advantage. Close this training/decoder recipe, not the environment.

**Parent:** [radical_physics_v1](radical_physics_v1.md).
**Protocol:** [learning registration](../../experiments/mixed_physics_learning_v1_protocol.json).
**Artifacts:** `simulation_data/gnn_environment_search_v1/mixed_physics_v1_gate/`;
training corpus will be `mixed_physics_corpus/` under the same parent.

## Record

- [2026-09-22 — Nine-model live pilot: no control-surviving GNN win](#2026-09-22--nine-model-live-pilot-no-control-surviving-gnn-win)

- [2026-09-22 — Qualification, integration and learning registration](#2026-09-22--qualification-integration-and-learning-registration)

## 2026-09-22 — Qualification, integration and learning registration

The prior generic scorer computed disabled thermal and energy terms. A specialized
C++ implementation removes that work and uses atomic lock and domain bitmasks;
search scores and chosen plans match the original exactly. A fresh calibration
compares 36 initializer/search combinations, selecting affinity plus 256 annealing
steps. On seeds 106000–106031, its p95 is 1.859 ms; a feasible two-start 8,192-step
reference leaves 12.437% median headroom [8.324%, 13.975%]. This is not a certified
optimality gap. The old source and artifacts are preserved.

A new `mixed_execution` infrastructure field activates atomic host/lock/domain
reservation and sequence-dependent setup in real HeROsim platform execution.
Default simulation keeps its existing environment and execution branch. Inputs
carry an explicit `mixed_execution_v1` contract; results carry `mixedExecution`
events and wait/setup totals. A priority barrier processes all zero-time arrivals
and completions before choosing FIFO machine heads; no batching delay is added.
The opt-in environment canonicalizes timestamps to a picosecond grid. The first
attempt exposed floating-point split completions and failed at seed 106006; it is
retained under `live/`. Quantizing timeout duration alone was insufficient because
addition reintroduced the split. Canonicalizing event timestamps fixed it; `live_v2/`
and `live_v2_READ.json` are authoritative. Every task completion time and served
assignment matches independent replay in all 128 runs. Two default-contract fixtures
match pre-change timings/placements/RTT exactly. This clock contract is experimental,
not a silent change to historical physics.

The typed graph connects predecessor/successor operations and operations sharing
locks or eligible power domains. Every arm receives processing times, eligibility,
setup classes, lock requirements, remaining job type/lock/work counts, setup matrix,
domain memberships and global counts. The hand MLP additionally receives fixed
one- and two-hop typed graph summaries, so it is not deliberately graph-blind.
MP-OFF is separately trained with the same base architecture and features.

Train on 512 new cases; validate on 64, test on 64, with disjoint seed ranges and
actual-identity checks. Teacher plans come from two 8,192-step searches and the
frozen hand control. Keep every evaluated teacher plan. Three draws per arm run
through `run_experiment.py`, with offline W&B logs, checkpoint sidecars and equal
selection on complete served validation RTT. Fifteen CE warmup epochs are followed
by self-critical full-plan cost training with a small teacher-CE auxiliary term.
This target is a complete finite announced workload, unlike the chaotic arbitrary
wall-clock horizon closed in `objective_pivot_v1`. Serving uses the same guarded
initializer and one exact descent round in every learned arm. The pilot must beat
both learned controls and the frozen rule by 5%, with positive intervals, and fit
2 ms including features and refinement. Finish with the registered real-HeROsim
gate even if offline results are negative.

## 2026-09-22 — Nine-model live pilot: no control-surviving GNN win

**Completed work.** Generated 512 training, 64 validation and 64 held-out workflows
with actual-identity separation. All 10,487,680 scored teacher placements are
retained in per-case `placements/placements.jsonl`. Validation checks source/cache
fingerprints, reconstructs every cached graph, checks every trace's row count and
minimum against its label, and independently scores trace boundaries and selected
plans. It does not independently re-simulate all ten million candidates. The
corpus's `all_warm_zero_io` metadata string denotes this fixed regime; actual
HeROsim configurations use `warmth_physics=node_disk_v2` with pre-warmed replicas
and zero I/O. This custom cache is not interchangeable with legacy model caches.

The architecture witness additionally preserves every job's future type/lock
histogram, so the entire base-feature row of the first task remains unchanged.
Three of 16 pairs flip its unique optimal host. The proposed untrained GNN
produces different root logits on all three, while MP-OFF's root output is
identical. The engineered hand-graph arm also distinguishes all three. This checks
representation, not learned accuracy or GNN necessity; it prevents claiming a win
solely over the graph-blind control.

Nine models were actually trained, through the three registered configs at seeds
201/202/203. All use offline W&B, deterministic CPU training, saved weights and
contract sidecars. GNN selected epochs are 43/35/8; MP-OFF 38/41/54; hand-graph
MLP 14/15/14. Selected validation accuracies are 63.4–63.8%, 63.7–64.0%, and
61.7–62.9%, respectively, against 50% chance. Selection is on served validation
RTT, not accuracy. Later policy-gradient epochs do not systematically improve the
selected result: one GNN and all hand-graph runs select during imitation warmup.
This is the registered bounded pilot, not a hyperparameter search or a power claim.

Training startup exposed a missing dispatcher `--wandb-project` argument, fixed
before any gradient step. The next attempt failed because the sandbox blocked
W&B's local logging socket. Both attempts are retained. The successful run uses
local IPC with explicit offline mode; no upload or GPU job was used. Checkpoints
live in the writable experiment artifact tree, leaving inherited model directories
untouched. The old determinism smoke harness's filesystem/worker repairs are
recorded separately in `docs/gates/gate-tools.md`.

**Held-out gate.** Before testing, all nine validation-selected checkpoints were
fingerprinted. A diagnostic amendment adds `guard_only` (fastest/affinity selection
plus one descent round, no network), without changing the primary criteria,
checkpoints or test cases. The 64 new environments run all nine learned arms,
the frozen affinity/256-step rule, guard-only, and actual Knative: **768 HeROsim
runs, 36,864 completed task records**. Every completion time agrees with independent
replay, in addition to aggregate RTT agreement. Infrastructure identities, source
and checkpoint bytes, served decisions and execution/setup accounting pass audit.

| Comparator | Median GNN RTT improvement | Mean paired improvement | Hierarchical median 95% interval |
|---|---:|---:|---:|
| Separately trained MP-OFF | 0.00% | +0.124% | [0.00%, 0.00%] |
| Hand-graph MLP | 0.00% | +0.365% | [0.00%, 0.00%] |
| Frozen affinity + annealing rule | **−4.97%** | −6.877% | [−7.930%, −3.534%] |
| Guard-only, no model | 0.00% | +0.725% | [0.00%, 0.00%] |
| Actual Knative | +34.26% | +34.592% | [+32.729%, +36.200%] |

Percentages average the three paired training-draw contrasts within each
infrastructure before taking the median. The zero-width median intervals reflect
a large exact tie mass, not proof that the learned functions are identical or
that every tail effect is zero. Knative lacks the complete-workflow forecast and
uses private replicas sharing node service; this is not a production SLA claim.

Full feature construction, inference, initial-plan scoring and refinement fit the
2 ms budget: GNN p95 1.708–1.727 ms, MP-OFF 1.667–1.676 ms, hand-graph MLP
1.759–1.762 ms. The rule costs 1.881 ms p95 and guard-only 1.106 ms. One-time
compilation/loading is excluded symmetrically. These are p95s of seven-repeat
case medians, not worst-case execution guarantees. Planning wall time is separate
from simulated task RTT. Mean workload RTT is 2551.14–2567.63 ms for GNN, versus
2396.05 ms for the frozen rule and 2578.73 ms for guard-only.

**What the network actually contributes.** At the three GNN seeds, the final
plan is identical to guard-only on 48/51/52 of 64 cases. It improves 11/10/7 and
worsens 5/3/5. The registered guard compares initial costs, so it does not guarantee
that the selected plan is better after refinement. A post-hoc bound grants a free
comparison of both final results: median gain remains zero over guard-only and
−4.893% against the frozen rule. Even granting free selection of the best of all
three GNN seeds plus guard-only still loses to the rule by 4.187% median. These
are generous diagnostic bounds from already-run plans, not new timed policies;
repairing that guard alone cannot rescue this pilot's win criterion.

**Decision.** Close this specific 32-hidden/two-layer, imitation-plus-self-critical,
guarded-one-refinement recipe. The integrated environment has real search headroom
and nonlocal dependencies, but this learned policy mostly recovers the fallback
and does not beat the strong hand scheduler. Do not turn the Knative gain or the
above-chance task accuracy into a GNN result. Do not repeat this recipe unchanged
with more draws. The broader environment, other radical mechanisms, alternate
representations and genuinely different learned search policies remain unproven,
not falsified by this pilot.

**Artifacts.** Under `simulation_data/gnn_environment_search_v1/`:
`mixed_physics_corpus/` (inputs, full traces, graph caches, metadata and validation),
`mixed_graph_witness_v1/` (exact pairs and representation check),
`mixed_physics_models/` (nine checkpoints, sidecars and histories),
`mixed_physics_training_logs_localipc/` (successful offline W&B runs),
`mixed_physics_curves/` (chance-floor curve reads), and
`mixed_physics_model_gate/` (`read.json`, `AUDIT.json`, frozen provenance,
`decoder_contribution.json`, `guard_upper_bound.json`, and every live record).
The qualification root contains `runtime_sources/` before/after snapshots and
legacy parity receipts; the reconstructed pre-change infrastructure hash matches
the preceding experiment's fingerprint.

Entry points: [trainer](../../src/policy/mixed/train.py),
[model and graph features](../../src/policy/mixed/model.py),
[data generator](../../scripts_cosim/generate_mixed_data.py),
[data validator](../../scripts_cosim/validate_mixed_data.py),
[real-engine gate](../../scripts_cosim/evaluate_mixed_models.py),
[auditor](../../scripts_cosim/audit_mixed_models.py),
[execution coordinator](../../src/placement/radical/coordinator.py).
The completed focused suites contain 25 determinism, 156 physics/availability/record,
and 386 runner/curve/snapshot checks (567 total); record hygiene is rerun at closure.
