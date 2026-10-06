# radical_physics_v1

**Status:** `ACTIVE` (2026-09-22) — exploratory sweep of eight new scheduling mechanisms.

**Outcome so far.** Eight new-mechanism sandbox screens and stronger-control
confirmation found planning headroom. The setup/locks/power follow-up is now
integrated into real HeROsim and has completed a nine-model training/live pilot:
[mixed_physics_learning_v1](mixed_physics_learning_v1.md) records its negative
GNN result. The two original pair-selector recipes failed their 2 ms gates.
A budget-matched follow-up narrowly qualified perfect pair selection at 5 ms,
but the resulting nine-model [pair-selector pilot](mixed_pair_learning_v1.md)
failed: zero median GNN gain over both learned controls and the hand rule,
with full inference slightly over 5 ms. The earlier one-shot model and this
pair-selector recipe remain closed. The active successor, [mixed_dispatch_v1](mixed_dispatch_v1.md), changes execution
order and passes two headroom screens, including adaptive hand controls. This
is an ordering/search opportunity, not a learned or GNN-specific result.

**Parent:** [workflow_basin_v1](workflow_basin_v1.md).
**Protocol:** [registration](../../experiments/radical_physics_v1.json).
**Artifacts:** `simulation_data/gnn_environment_search_v1/radical_physics_v1/`.

## Record

- [2026-09-22 — Ready-operation priority exploration registration](#2026-09-22--ready-operation-priority-exploration-registration)

- [2026-09-22 — Small pair-selector learning pilot registration](#2026-09-22--small-pair-selector-learning-pilot-registration)

- [2026-09-22 — Larger-budget results and live validation](#2026-09-22--larger-budget-results-and-live-validation)
- [2026-09-22 — Larger-budget timing-margin amendment](#2026-09-22--larger-budget-timing-margin-amendment)
- [2026-09-22 — Larger-budget pair rematch registration](#2026-09-22--larger-budget-pair-rematch-registration)

- [2026-09-22 — Pair-selector results and live audit](#2026-09-22--pair-selector-results-and-live-audit)
- [2026-09-22 — Shallow pair-selection follow-up registration](#2026-09-22--shallow-pair-selection-follow-up-registration)
- [2026-09-22 — Pair-selection exploration registration](#2026-09-22--pair-selection-exploration-registration)

- [2026-09-22 — Integrated child completed](#2026-09-22--integrated-child-completed)

- [2026-09-22 — Discovery, stronger-control confirmation and constructive witnesses](#2026-09-22--discovery-stronger-control-confirmation-and-constructive-witnesses)

- [2026-09-22 — Discovery registration](#2026-09-22--discovery-registration)

## 2026-09-22 — Discovery registration

This is an isolated experimental environment with independent compiled-search and
SimPy-event implementations. It does not change HeROsim's historical physics and
is not a production HeROsim live gate. Every shortlisted mechanism still requires
fresh confirmation, an equal-information graph witness, stronger hand controls
and integration validation before training. The eight-way search is exploratory;
bootstrap intervals do not supply a multiple-testing-corrected discovery claim.

Eight jobs of six operations run on four FIFO machines, with two eligible hosts
per operation. Full jobs and physics parameters are announced equally to all
planners at time zero; schedules are non-preemptive. Four setup classes have
sequence-dependent transition durations. Logical operations atomically acquire
one or two of six shared locks; no hold-and-wait deadlocks. Three overlapping
power domains permit at most two concurrent member hosts. Thermal state follows
exponential cooling and cross-host heat input; temperature at dispatch sets that
operation's execution dilation. Announced maintenance excludes crossing its
window. Deadlines add three times tardiness to summed job completion. The energy
case adds 0.2 times duration × operation power × dispatch-time price (a declared
reservation tariff, not an integral of changing grid prices). The mixed case
combines setup, locks and power. These are simplified synthetic mechanisms, not
calibrated hardware models.

Calibrate six executable hand controls on eight cases, choose one per mechanism
under 2 ms p95, then freeze all selectors before sixteen different cases. Compare
against a feasible expensive reference: best of controls and 8,192-step annealing.
No certified optimality is claimed. Fastest host, load balancing, setup affinity,
exact one-flip descent, and 128/1,024-step annealing challenge each mechanism.
Include full wrapper, construction and search time. Every selected, reference
and fastest plan is replayed through independent SimPy with full task events;
mechanism-off counterfactuals use the same plans. Compare completed operation
vectors, not only aggregate objectives. Actual inputs are fingerprinted and
unique before simulations; repeated mechanisms on the same substrate are paired,
not independent additional infrastructures.

## 2026-09-22 — Discovery, stronger-control confirmation and constructive witnesses

**Discovery completed.** All eight mechanisms left more than 5% improvement for
an expensive feasible search reference over the initially calibrated hand control.
That alone is insufficient: the mixed case initially selected simple load balancing
because the coarse search choices overshot 2 ms. Rather than call that a learning
opportunity, a separate registered confirmation challenged each mechanism with
24 start/search combinations, including fastest, load, affinity and event-driven
ECT starts; one or two descent rounds; and 64/128/192-step annealing. All selectors
were frozen on calibration before opening fresh seeds 102000–102015. Added a
no-new-physics arm to expose ordinary search headroom.

| Mechanism | Frozen hand control | Fresh median reference gain | Bootstrap 95% interval | Control p95 |
|---|---|---:|---:|---:|
| No new physics | fastest + 2 descent rounds | 5.05% | [3.24, 7.57]% | 0.806 ms |
| Sequence-dependent setup | load + 192 annealing steps | 11.81% | [9.76, 13.84]% | 1.528 ms |
| Shared atomic locks | fastest + 2 descent rounds | 5.18% | [3.90, 9.80]% | 1.349 ms |
| Overlapping power limits | fastest + 2 descent rounds | 6.56% | [3.10, 7.43]% | 1.699 ms |
| Thermal coupling | fastest + 2 descent rounds | 7.84% | [4.27, 9.33]% | 0.841 ms |
| Announced maintenance | fastest + 192 annealing steps | 5.69% | [3.45, 6.94]% | 1.964 ms |
| Deadlines / weighted tardiness | ECT + 192 annealing steps | 15.13% | [11.83, 20.34]% | 1.417 ms |
| Reservation energy tariff | fastest + 2 descent rounds | 5.75% | [3.79, 7.47]% | 0.805 ms |
| Setup + locks + power | fastest + 2 descent rounds | 14.43% | [7.84, 17.05]% | 1.890 ms |

Reference is the best feasible plan among the frozen control and two 8,192-step
annealing runs, initialized from fastest and load. Its compute is not credited
as a free deployable controller. Percentages concern each mechanism's objective:
completion time for the physical mechanisms, completion plus tardiness penalty
for deadlines, and completion plus energy charge for tariff. These are not all
latency percentages. Bootstrap units are 16 unique environments per stage, not
the repeated mechanisms, planning repetitions or simulated tasks. This is still
exploratory; no familywise error correction or powered model-class claim.

The initial lock gap dropped from 12.61% to 5.18% under stronger controls; its
initial apparent promise depended substantially on the starting heuristic.
The no-new-physics gap also passes 5%, so none of these gaps establishes graph
necessity. The neutral arm uses this general-purpose simulator/search kernel,
not every specialized historical workflow optimizer, and does not reopen the
closed workflow recipes. Full use of the budget, stronger mechanism-specific
rules and a future deployment timing check remain necessary.

**The mechanisms actually bind.** In the confirmation reference schedules,
setup contributes 3,306 duration units, locks cause 632 blocked-start observations,
power limits 443, and maintenance 353. Thermal dilation contributes 4,916.37
additional duration units. Combined schedules still pay 3,452 setup units,
648 lock-block observations and 152 power-block observations. Blocking observations
are checks, not elapsed wait times or independent events. Thus better placements
do not eliminate the mechanisms entirely. These are synthetic units and parameters.

**Constructive witnesses.** A separately registered small exact experiment uses
16 pairs each for setup and combined physics, at three jobs × three operations.
Permuting type and lock requirements only among future non-first operations keeps
all execution times, first-operation observations, deadlines and global type/lock
histograms unchanged. Nevertheless the uniquely best first host flips in **3/16
setup pairs** and **5/16 combined pairs**. Full enumeration retains all **32,768
placements** across both members of all pairs. Both conditional-root optima are
replayed in SimPy. These witnesses establish dependence on how future requirements
are arranged, not a message-passing or neural advantage: exact search solves these
small cases, and no claim is made that every engineered lookahead feature is blind.
A specific GNN's ability to represent and learn this distinction remains untested.

**Validation and coverage.** Discovery runs 640 primary SimPy replays (384 retained
full-event runs plus 256 mechanism-off same-plan counterfactuals). Confirmation
runs 432 full-event replays; witnesses add 128, for **1,200 primary replays**.
The retained full-event runs contain **40,320 completed operations**. Two independent
artifact audits recheck source identities, plans, native objective agreement,
precedence, durations and retained events. The witness audit independently replays
every enumerated placement in SimPy. Compiled/Python parity tests span every
mechanism, multiple seeds, deterministic optimization, locks, domain capacities,
and physics-off agreement with the pre-existing workflow evaluator. The focused
suite passed **145 tests**. No GPU training and no production runtime modifications.

**Standing decision.** Keep this lineage active. Prioritize the **combined
setup/locks/power** environment and **sequence-dependent setup** for a next
integration/control challenge; keep deadline-aware scheduling as an objective
alternative. The other mechanisms are screened, not falsified. No experiment is
closed on this sandbox screen, and no model training is authorized by these numbers
alone under this protocol. Main-HeROsim event-order/physics integration, a specified
graph encoder, affordable hand graph processing and a fresh model gate remain
outstanding before a GNN claim. The user's general authorization to continue work
remains in effect; this is an evidence gate, not an approval requirement.

**Entry points and artifacts.** [native/Python environment](../../src/placement/radical/environment.py),
[independent SimPy replay](../../src/placement/radical/live.py),
[discovery runner](../../scripts_cosim/radical_physics_screen.py),
[confirmation runner](../../scripts_cosim/radical_physics_confirm.py),
[witness runner](../../scripts_cosim/radical_physics_witness.py),
[artifact auditor](../../scripts_cosim/audit_radical_physics.py),
[witness auditor](../../scripts_cosim/audit_radical_witness.py).
Sibling artifact directories `radical_physics_v1_confirmation/` and
`radical_physics_v1_witness/` share the discovery root's parent. Each includes
pre-run source/compiler fingerprints, input/plan records, result summaries and
an `AUDIT.json` receipt. Registration files are
`experiments/radical_physics_v1_confirmation.json` and
`experiments/radical_physics_v1_witness.json`.

## 2026-09-22 — Integrated child completed

The combined-physics follow-up has progressed through specialized controls,
real-engine integration, data generation, matched training and a held-out live
gate. Its standing result and exact stop belong to
[mixed_physics_learning_v1](mixed_physics_learning_v1.md). Earlier sandbox-only
statements below describe the discovery stage, not the current child status.

## 2026-09-22 — Pair-selection exploration registration

The next question changes the learned decision from a complete assignment to
selecting a two-operation perturbation. The previous mixed model recipe stays
closed. [Protocol](../../experiments/mixed_pair_exploration_v1.json) fixes eight
new calibration and 32 fresh cases before measurement. Start at an affinity
placement refined by single-operation descent. Enumerate each unordered pair,
flip both hosts, and refine by two exact best-improvement rounds. Retaining the
incumbent makes the best final candidate an optimistic perfect-selector ceiling;
it is not a deployable GNN or an optimal-placement certificate.

Challenge this ceiling with the previous 256-step annealing rule, pure descent,
and graph-ranked or fixed-random portfolios of 4/8/16 pairs. Select the best
control satisfying 2 ms p95 on calibration, freeze it, then require at least 5%
median fresh improvement with a positive bootstrap lower bound. All planner
preparation and search time is charged to controls. Regardless of the screen,
run the frozen control, incumbent and ceiling plans in real HeROsim on all 32
fresh cases (96 runs), checking every task completion against independent SimPy.
This screen can reject an insufficient selector opportunity without training;
a pass alone cannot establish learnability or a GNN-specific advantage.

## 2026-09-22 — Shallow pair-selection follow-up registration

The first pair screen's perfect selector clears the headroom bar, but preparing
its converged starting placement alone exceeds the 2 ms budget. Before training,
[the shallow follow-up](../../experiments/mixed_pair_shallow_v1.json) changes that
recipe to two initial descent rounds and one refinement round per pair. It uses
new calibration seeds 115000–115007 and fresh seeds 116000–116031; the previous
holdout is retired. Graph ordering is vectorized without changing its ranking.
Controls consider one, two or four pairs, plus the original annealing rule.
The same 5% headroom and full 96-run real-HeROsim gate apply. The two screens are
exploratory, sequential decisions, not independent confirmations of one recipe.


## 2026-09-22 — Pair-selector results and live audit

**No training on either tested recipe.** Both registered screens and their full
real-HeROsim gates completed. These are two sequential exploratory screens on
separate workloads, not two replications of one model. No GNN was trained and
no GPU was allocated. The parent remains ACTIVE for materially different ideas.

| Recipe | Perfect-selector median gain vs frozen hand rule | Bootstrap 95% interval | Free-selector full-path p95 | Decision |
|---|---:|---:|---:|---|
| Up to 64 initial descent rounds, pair flip, two refinement rounds | 6.276% | [4.119%, 8.591%] | 3.945 ms | Exceeds 2 ms before any learned selection |
| Two initial rounds, pair flip, one refinement round | 2.632% | [2.183%, 6.185%] | 1.508 ms | Misses registered 5% median headroom bar |

Each oracle chooses the best **final** result among 1,128 pair perturbations and
the unchanged incumbent. This is an upper bound only for a selector choosing one
of these candidates; it does not bound larger neighborhoods, different starts,
multiple successive perturbations, or other implementations. Oracle decisions
use final outcomes and are not deployable policies. The reported posthoc cost
recomputes the starting placement and only the known winning candidate, granting
selection zero time. The deep implementation already required 2.791 ms p95 just
to construct its starting placement on calibration. The shallow implementation
has some timing room, but its measured objective ceiling fails the advance bar.
Its confidence interval includes 5%; this is failure of the registered screen,
not proof that a population-level 5% effect is impossible.

The hand control selected independently on each eight-case calibration was
`affinity:anneal256`. On the respective 32-case fresh gates it took 1.842 and
1.863 ms p95. The deep screen also challenged pure descent and graph-ranked or
random portfolios of 4/8/16 pairs. The shallow follow-up priced vectorized
same-information graph ordering and 1/2/4-pair portfolios. Both controls were
frozen before fresh outcomes were read. Existing mixed-model training/test cases
were not reused. Gains pair on workloads; there are no training-draw replicates.

**Validation.** Actual packed-input fingerprints and file/source hashes were
checked before simulation; duplicated inputs and changed source/input files are
covered by rejection tests. Every retained pair was scored again and every pair
construction/refinement was regenerated. Both artifact audits passed: 72,192
pair candidates total, plus retained incumbents. All 192 live runs (three arms
on each of 64 fresh workflows) completed 9,216 operations. Objective sums,
placements and every task completion matched independently written SimPy.
Small-fixture tests independently compare native pair refinement with individual
single-flip searches and SimPy. Focused pair, mixed-physics and record tests passed
(116 tests at this revision). Historical code, checkpoints and artifacts remain
unchanged; only new experiment files and current research records were added.

**Entry points and evidence.**

- `src/placement/radical/pairs.cpp` and `pairs.py`: checked native portfolio.
- `scripts_cosim/mixed_pair_exploration.py`: shared screen/live harness.
- `scripts_cosim/mixed_pair_shallow.py`: separately registered serving variant.
- `scripts_cosim/audit_mixed_pairs.py`: source/input, candidate and live-artifact audit.
- `simulation_data/gnn_environment_search_v1/mixed_pair_exploration_v1/` and
  `simulation_data/gnn_environment_search_v1/mixed_pair_shallow_v1/`: protocol/source
  snapshots, inputs, calibration, complete final-candidate JSONL, per-case live
  event records, `read.json`, `live_READ.json`, and `AUDIT.json`. `read.json` is the
  pre-live screen snapshot; completed live/audit receipts supersede its pending flag.

The next investigation should change the action space or mechanism, or first
show a faster implementation that preserves sufficient candidate headroom.
Merely training a better classifier over either unchanged portfolio is not
supported by these measurements. This does not reopen the closed one-shot mixed
model recipe and does not close all learned neighborhood search.

## 2026-09-22 — Larger-budget pair rematch registration

At the user's request, test the unchanged deep pair portfolio at **5 and 10 ms**,
giving hand search the same budgets. This is a new budget comparison, not a claim
that the old 2 ms implementation fits. The
[protocol](../../experiments/mixed_pair_budget_v1.json) fixes 16 new calibration
cases (117000–117015) and 32 fresh cases (118000–118031), shared between budgets.
For each budget freeze the lowest-mean-cost affordable control across three
initializers, 256–1536 annealing steps, 4/8/64 descent rounds and graph/random
pair portfolios. Price all preparation and search; vectorized graph ranking must
match the earlier definition. Preserve the original perfect-selector ceiling
and time its full path with selection free. Require 5% median headroom and a
positive bootstrap lower bound, plus both paths fitting the selected budget.
Regardless of the screen, run control5, control10 and oracle through all 32
real-HeROsim cases (96 runs), independently checking every task completion.
These are paired budget comparisons, not additional independent environments.

## 2026-09-22 — Larger-budget timing-margin amendment

The initial rematch selected `graph2` at 5 ms, but it took 5.184 ms p95 on the
fresh gate despite 4.920 ms on calibration. That is a failed timing check, not an
acceptable 5 ms comparator. Preserve the first attempt and complete its live
replay. Before new measurements, the
[margin amendment](../../experiments/mixed_pair_budget_margin_v1.json) selects
controls using only the original calibration outcomes, with a fixed 15% timing
reserve (calibration p95 <= 0.85 × budget), then evaluates new seeds
120000–120031 at the actual 5/10 ms budgets. The first holdout remains retired.
The deep oracle, headroom threshold and full 96-run live gate stay unchanged.
This amendment repairs timing feasibility; it does not retune against test RTT.


## 2026-09-22 — Larger-budget results and live validation

**The deep pair selector narrowly qualifies for a 5 ms learnability pilot; it is
not a measured GNN advantage.** Raising the budget also strengthens the hand
controls. The registered margin follow-up compares the unchanged perfect
selector with an affordable hand graph rule at 5 ms and a longer annealing rule
at 10 ms. No model was trained and no GPU was allocated in this comparison.

| Budget | Frozen hand control | Hand p95 | Perfect-selector path p95 | Median oracle gain | Bootstrap 95% interval | Registered screen |
|---|---|---:|---:|---:|---:|---|
| 5 ms | `graph1` | 4.220 ms | 4.038 ms | 5.072% | [3.933%, 6.316%] | PASS, narrow |
| 10 ms | `affinity:anneal1280` | 7.177 ms | 4.038 ms | 2.248% | [−0.135%, 5.310%] | NO-GO |

The 5 ms rule constructs the same affinity/descent incumbent, chooses one pair
by shared locks, same-job and same-host ranking, then runs two exact refinement
rounds. The oracle instead chooses the best **final** outcome from all 1,128
pairs plus the incumbent. Its reported time gives selection zero cost and
executes only the winning pair. An actual selector must add features/inference
within roughly the remaining millisecond and preserve nearly all the measured
headroom to clear the 5% bar. The confidence interval is not wholly above 5%;
the registered bar requires a 5% sample median and a positive lower bound.
This is a fragile pilot qualification, not a guarantee of a 5% population effect,
a GNN-specific advantage, or a production latency guarantee.

The 10 ms control was selected from all 35 registered candidates by calibration
mean objective among those meeting the timing margin. It uses less than the
maximum time because the longer tested options had worse calibration means;
this is the best tested qualifying control, not a globally optimal hand solver.
Both budgets use the same 32 environments, so they are paired comparisons, not
64 independent cases. Wall-clock decision time is measured separately and is
not added to simulated execution RTT, matching previous gates.

**First attempt retained.** Original calibration selected `graph2` (4.920 ms)
at 5 ms and `affinity:anneal1280` at 10 ms. On seeds 118000–118031, oracle gains
were 4.623% [2.938%, 5.661%] and 4.264% [0.393%, 5.353%], respectively.
The 5 ms control took 5.184 ms p95, failing the timing contract; the 10 ms
control took 7.232 ms. The free selector took 4.382 ms. Both screens failed.
These outcomes were not hidden or replaced by the follow-up. Its 15% timing
margin was registered before seeds 120000–120031 were measured, using only
original calibration outcomes to select `graph1` and the unchanged 10 ms rule.
The two attempts are sequential exploratory tests; the 5 ms follow-up is not a
replication of the original comparator, and its pass should not be pooled with
the failed original as though the protocol were identical.

**Validation and artifacts.** Both full 96-run live gates completed: 192 real
HeROsim simulations, 9,216 operations, with per-operation completion and
placement agreement against independent SimPy. Both candidate sweeps contain
36,096 pair plans plus incumbents; audits re-score every plan, regenerate every
pair/refinement, check source/input/artifact fingerprints and recompute paired
statistics. Inputs are physically unique against earlier source receipts and
all pre-existing holdouts remain retired. Focused budget, pair, mixed-physics
and record tests pass. Historical implementation files and earlier evidence
were preserved.

- `experiments/mixed_pair_budget_v1.json` and
  `experiments/mixed_pair_budget_margin_v1.json`: original and amended protocols.
- `scripts_cosim/mixed_pair_budget.py`: control calibration, budget summaries,
  screen and artifact audit, reusing the prior live harness.
- `scripts_cosim/mixed_pair_budget_margin.py`: timing-margin follow-up.
- `simulation_data/gnn_environment_search_v1/mixed_pair_budget_v1/` and
  `simulation_data/gnn_environment_search_v1/mixed_pair_budget_margin_v1/`:
  frozen source/protocol/input receipts, calibration outcomes, complete final
  candidate sweeps, live task events, screen reads, `live_READ.json` and
  `AUDIT.json`. Screen pending flags describe the pre-live snapshots only.

This qualification led to [mixed_pair_learning_v1](mixed_pair_learning_v1.md),
which owns the completed training/live answer. The qualification is not a GNN
result and does not supersede the later negative pilot. The 10 ms result did
not justify training this unchanged recipe.

## 2026-09-22 — Small pair-selector learning pilot registration

The pilot completed. Its registration, training and final live result now live
in [mixed_pair_learning_v1](mixed_pair_learning_v1.md).

## 2026-09-22 — Ready-operation priority exploration registration

The ordering screen and stronger adaptive-control challenge both completed.
Their registration, results and remaining gates are recorded in
[mixed_dispatch_v1](mixed_dispatch_v1.md).
