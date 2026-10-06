# mixed_pair_learning_v1

**Status:** `CLOSED` (2026-09-22) — PAIR-SELECTOR-PILOT-NEGATIVE.
Registered before data generation and training; completed the required live gate.

**Outcome.** Nine trained models on 192 new workloads fail the 5 ms pilot:
GNN median gains are zero versus MP-OFF, the hand-graph model and the hand rule.
Its full decision path takes 5.20–5.24 ms p95. All 384 real-HeROsim evaluations
pass audit. The earlier narrow oracle qualification does not become a GNN win;
close this model/target/serving recipe, not all learned search or mixed physics.

**Parent:** [radical_physics_v1](radical_physics_v1.md).
**Protocol:** [pilot registration](../../experiments/mixed_pair_learning_v1_protocol.json).
**Artifacts:** `simulation_data/gnn_environment_search_v1/mixed_pair_corpus/`,
`mixed_pair_models/`, `mixed_pair_training_logs/`, `mixed_pair_curves/`, and
`mixed_pair_model_gate/` under the same parent directory.
Training configs retain the parent `radical_physics_v1` lineage tag; this node
owns the specific pilot outcome.

## Record

- [2026-09-22 — Trained pilot and independent live audit](#2026-09-22--trained-pilot-and-independent-live-audit)
- [2026-09-22 — Small pair-selector learning pilot registration](#2026-09-22--small-pair-selector-learning-pilot-registration)

## 2026-09-22 — Small pair-selector learning pilot registration

The user authorized the bounded pilot following the narrow 5 ms ceiling pass.
[Protocol](../../experiments/mixed_pair_learning_v1_protocol.json) fixes 128
training workflows (121000–121127), 32 validation (122000–122031) and 32 sealed
live-test workflows (123000–123031), with three training seeds for each of GNN,
MP-OFF and hand-graph MLP. Each label enumerates the 1,128 pair moves plus retaining
the incumbent, with the same two refinement rounds and final-cost incumbent guard.
No additional environment or objective changes are introduced.

The small two-layer/hidden-16 graph model scores symmetric operation pairs.
MP-OFF is retrained with message passing disabled on identical inputs; the hand
model additionally receives fixed typed one/two-hop means. All arms observe the
declared workflow, resources, computed incumbent hosts and completion times,
and explicit pair lock/job/host relations. Computing these observations is
charged at inference. Targets are soft distributions over exact guarded costs,
with temperature fixed to 1% of incumbent cost. Train 30 epochs, select minimum
mean validation served cost including epoch zero, log offline W&B, and preserve
weights with contracts. There is no architecture or temperature sweep.

Regardless of validation success, freeze all selected models before the full
384-run HeROsim gate (nine models, graph1, oracle and incumbent on 32 cases).
Check every task completion against independent SimPy. Training draws are paired
within physical environments. An advance needs GNN gains of at least 5% over
both learned controls and graph1 with positive bootstrap lower bounds, while
full model serving fits 5 ms p95. This is a small pilot, not a powered publication
claim. Failure stops scaling this recipe on this holdout.


## 2026-09-22 — Trained pilot and independent live audit

**No advance or scaling on this holdout.** The fixed 30-epoch CPU pilot completed
three independently seeded GNNs, three MP-OFF models and three hand-graph models
through `run_experiment.py` configs. Every run logged offline W&B and retained
weights, architecture/data/source contracts and per-epoch curves. No GPU and no
external W&B upload were used. The data audit passes 192 unique workflows and
216,768 complete candidate outcomes: 128 train, 32 validation and 32 test cases.
Every pair construction/refinement and cached feature was regenerated before
training; all input/source/label file fingerprints were checked.

### Validation and curves

GNN selected epochs are 10/16/29; MP-OFF 3/16/14; hand-graph 17/19/14. Mean GNN
served RTT is 2272.0625 / 2271.46875 / 2275.625 ms versus hand-rule 2286.5 ms and
oracle 2167.0625 ms. GNN gains are only 0.631%, 0.657% and 0.476%, below the 5%
validation advance bar. MP-OFF gains are 0.569%, 0.488%, 0.854%; hand-graph gains
0.335%, 0.559%, 0.622%. Full GNN validation p95 is 5.385–5.404 ms, also over budget.
The registered live gate ran despite this failure; no test-dependent selection
or additional training occurred.

All nine W&B transaction logs were read with `read_training_curves.py`.
Uniform soft-target cross-entropy is log(1129), about 7.029. The validation
probability that a uniformly chosen candidate is optimal is 0.321%, accounting
for tied optima rather than incorrectly assuming one correct pair. Selected GNN
optimal-candidate fractions are 9.375%, 3.125%, 3.125%, above chance but not close
to recovering the oracle's scheduling benefit. For example, seed301 train CE
falls from 7.028 to 6.557, while its best served validation cost is at epoch10
and ends worse. These small-data results do not prove that no different training
objective or larger corpus could learn the task; they fail this precommitted pilot.

### Fresh live result

Models and source/data/checkpoint hashes were frozen before opening test caches.
All nine models, `graph1`, the perfect selector and the incumbent run on the same
32 fresh workflows: **384 actual HeROsim simulations, 18,432 completed operations**.
Every task placement and completion time agrees with independently written
SimPy. The audit also checks complete arm/environment coverage and retained
artifact hashes. GNN/control comparisons average seed-paired gains within each
physical environment, then take the median; training seeds are not additional
independent infrastructures.

| GNN comparison | Median gain | Mean paired gain | Bootstrap median 95% interval |
|---|---:|---:|---:|
| MP-OFF | 0.000% | 0.243% | [0.000%, 0.000%] |
| Hand-graph model | 0.000% | 0.117% | [0.000%, 0.000%] |
| Hand rule `graph1` | 0.000% | 0.115% | [0.000%, 0.000%] |
| Unchanged incumbent | 0.000% | 0.465% | [0.000%, 0.165%] |

The zero-width median intervals reflect a large mass of exact ties, not proof
that all means or individual decisions are identical. GNN full serving p95 is
5.242 / 5.223 / 5.203 ms. MP-OFF is 5.170–5.190 ms; hand-graph is 5.091–5.277 ms;
`graph1` takes 4.282 ms and the incumbent 3.221 ms. Timing includes initial exact
descent, incumbent replay for completion features, all features, inference,
selected-pair refinement and the final-cost guard. Compilation/loading are
excluded. Wall-clock planning remains separate from simulated RTT. Even a
successful latency optimization alone would not repair the missing RTT gain.

The **perfect selector itself** gains 4.300% median over `graph1` on this fresh
set, CI [3.732%, 5.120%], mean 4.583%. The earlier 5.072% qualification was a
fragile sample threshold crossing, not guaranteed headroom on all new draws.
Neither number is a GNN measurement or a certified global scheduling optimum.

### Debugging the selected moves

`selection_diagnostic.json` recomputes each frozen model's candidate index from
the cached inputs, checks its guarded target against real served cost, and
reconstructs the exact final placement from the retained raw candidate. All
checks pass; there is no label-index or guard mismatch.

GNN seeds301/302/303 leave the incumbent plan unchanged on 22/28/30 of32 cases,
respectively, and improve it on 10/4/2. None explicitly chooses the no-move class;
13/15/13 raw proposed final outcomes are worse than the incumbent and are guarded
away, with further equal-cost outcomes also retaining the incumbent. MP-OFF
improves 5/3/5 cases; hand-graph improves 3/2/4. Message passing produces occasional
useful proposals but no consistent median advantage over the controls.

### Implementation, checks and scope

- `src/policy/pair_selector/model.py` and `train.py`: symmetric pair head,
  separately trained ablations, fixed objective and full serving path.
- `scripts_cosim/generate_pair_data.py`: exhaustive candidate generation and audit.
- `scripts_cosim/run_pair_pilot.py`: actual config-driven nine-run launcher.
- `scripts_cosim/evaluate_pair_models.py`: validation, timed serving, live gate
  and independent replay audit.
- `scripts_cosim/diagnose_pair_models.py`: candidate-index and guard reconciliation.
- `tests/test_pair_selector.py` and additions to
  `tests/test_trainer_determinism.py`: same-seed training equality, declared-input
  isolation, MP-OFF adjacency invariance and GNN representation sensitivity.

The full trainer-determinism and experiment-dispatcher suite passed (399 checks),
and focused model/physics/record checks passed. The curve reader now recognizes
pair success as a higher-is-better metric and reports its tied-optimum chance
floor; this reporting fix does not alter selected weights or gate decisions.

Stop this hidden16/two-layer, 30-epoch soft-target pair-selection recipe at the
5 ms / 5% bar. Preserve its complete corpus and checkpoints for inspection.
Reopening needs a materially different, validated learning/serving proposal;
more runs on this holdout or a timing-only fix are not a positive GNN result.
