# mixed_dispatch_v1

**Status:** `ACTIVE` (2026-09-22) — ORDERING-ENVIRONMENT-ACTIVE;
DAG-RESERVATION-RECIPE-NOT-QUALIFIED. Earlier screening protocols preceded their
measurements; the new reservation example is explicitly constructed after
exploration, not a preregistered discovery. The live-control correction is disclosed below.

**Outcome so far.** The three earlier learned children are closed negative:
[total-rank imitation](mixed_dispatch_learning_v1.md),
[ready-set imitation](mixed_dispatch_state_v1.md), and
[unguarded action values](mixed_dispatch_value_v1.md). The scaled block/joint
screens expose 5.06%/0.91% reference headroom and four-operation region S0 finds
0.55%; these fail their registered 8% funding bar, not a theorem against learning.
All use non-delay dispatch, which cannot deliberately reserve an idle resource.
A new constructed four-operation example proves that restriction matters: the
best of all 384 placement/priority plans costs 30, whereas a reservation schedule
costs 27 and is certified optimal by machine-order enumeration. Four actual
HeROsim runs and independent audit confirm the 10% improvement. A simple hand
reservation also attains 27, so **this is action-space headroom, not a GNN win**.
The environment remains active; broad training futility, residual graph
information and a learned reservation advantage are not established.
The first irregular-DAG screen finds 3.94% reference headroom and passes all
66 live audits. Its successor corrects decoder initialization/translation and
finds **5.43%** on fresh inputs, below the 8% funding bar. All 58 successor live
runs, resource-event checks and full search-reproduction audit pass.
Reservations supply the best observed schedules on 11/16 inputs. The
[trained DAG successor](dag_reservation_learning_v1.md) also found no GNN
advantage; residual graph information remains unanswered. The
[resumable continuation child](dag_resume_s0_v1.md) passes its execution contract
but closes its bounded one/two-decision recipe without gain over matched hand
search. A [hidden-arrival successor](online_reservation_s0_v1.md) also finds no
gain for its two-decision search over a matched hand rollout portfolio after
fresh live replay. Broader stepwise reservation learning remains unanswered.
No GPU was allocated.

**Parent:** [radical_physics_v1](radical_physics_v1.md).
**Protocols:** [initial](../../experiments/mixed_dispatch_v1.json),
[adaptive challenge](../../experiments/mixed_dispatch_adaptive_v1.json).
**Artifacts:** `simulation_data/gnn_environment_search_v1/mixed_dispatch_v1/` and
`mixed_dispatch_adaptive_v1/` under the same parent directory.

## Record

- [2026-09-22 — Trained DAG successor](#2026-09-22--trained-dag-successor)
- [2026-09-22 — Schedule-translation successor result](#2026-09-22--schedule-translation-successor-result)
- [2026-09-22 — DAG result and schedule-translation successor](#2026-09-22--dag-result-and-schedule-translation-successor)
- [2026-09-22 — Irregular DAG reservation qualification protocol](#2026-09-22--irregular-dag-reservation-qualification-protocol)
- [2026-09-22 — Scope correction and certified reservation witness](#2026-09-22--scope-correction-and-certified-reservation-witness)
- [2026-09-22 — Exact region selection S0 result](#2026-09-22--exact-region-selection-s0-result)
- [2026-09-22 — Exact region selection S0 protocol](#2026-09-22--exact-region-selection-s0-protocol)
- [2026-09-22 — Joint placement and ordering result](#2026-09-22--joint-placement-and-ordering-result)
- [2026-09-22 — Joint placement and ordering control challenge](#2026-09-22--joint-placement-and-ordering-control-challenge)
- [2026-09-22 — Capacity-scaled block-search result](#2026-09-22--capacity-scaled-block-search-result)
- [2026-09-22 — Capacity and block-search qualification protocol](#2026-09-22--capacity-and-block-search-qualification-protocol)
- [2026-09-22 — Scaled CPU runtime calibration](#2026-09-22--scaled-cpu-runtime-calibration)
- [2026-09-22 — Matched priority-learning pilot completed](#2026-09-22--matched-priority-learning-pilot-completed)
- [2026-09-22 — Exhaustive ordering witness result](#2026-09-22--exhaustive-ordering-witness-result)
- [2026-09-22 — Constructive ordering witness registration](#2026-09-22--constructive-ordering-witness-registration)

- [2026-09-22 — Ordering and adaptive-control results](#2026-09-22--ordering-and-adaptive-control-results)
- [2026-09-22 — Adaptive dispatch control challenge registration](#2026-09-22--adaptive-dispatch-control-challenge-registration)
- [2026-09-22 — Ready-operation priority exploration registration](#2026-09-22--ready-operation-priority-exploration-registration)

## 2026-09-22 — Ready-operation priority exploration registration

After closing the learned pair selector, change the action to execution order.
[Protocol](../../experiments/mixed_dispatch_v1.json) fixes placements to the same
old affinity/descent plan for all arms, then permits priority arbitration among
all ready operations, bypassing resource-blocked candidates. The setup, atomic
locks and power-domain physics stay unchanged. Priorities are static per
operation in this first screen; this is not yet a state-adaptive learned policy.

Sixteen new calibration and 32 fresh workloads compare seven priority rules
(processing time, remaining work in both directions, type grouping, low/high
lock pressure and graph-weighted remaining work) with 0/64/128/256/512 priority
swap annealing steps. Charge the common placement calculation and all planning.
Select the best control below 4.25 ms calibration p95, test the actual 5 ms budget,
and compare with a feasible 8192-step two-start reference. Require 5% median
headroom with a positive bootstrap lower bound. All 96 registered HeROsim runs
execute even after an offline failure, checking every task against independent
SimPy. The adapter restores the historical coordinator after each run; original
physics/source files are preserved. A screen pass would still require stronger
state-adaptive dispatch controls and a graph representation challenge before
training, plus attention to joint placement/order optimization.

## 2026-09-22 — Adaptive dispatch control challenge registration

The first ordering screen passes at 8.80% median headroom, but its dispatch rules
are static. Before training, [the adaptive challenge](../../experiments/mixed_dispatch_adaptive_v1.json)
adds nine state-dependent hand rules based on current setup, remaining execution,
ready-lock overlaps and future lock pressure, plus short search from their
realized schedules. Their deterministic complete-workflow replay is charged and
materialized into equivalent static ranks for the same HeROsim adapter. Native
and independent SimPy checks must preserve all completion times. Use new
calibration seeds126000–126015 and fresh seeds127000–127031. The 5 ms budget,
15% calibration reserve, reference and full 96-run live gate stay fixed.


## 2026-09-22 — Ordering and adaptive-control results

The new action is a static operation-priority vector used whenever operations
are ready. At each completion time, dispatch considers all ready operations in
priority order and starts resource-feasible ones. It can bypass a blocked
higher-priority operation. This changes arbitration relative to the historical
FIFO/head-of-line model, while keeping sequence-dependent setup, atomic locks,
non-preemptive service and power-domain limits unchanged. The experiment is
conditional on fixed placements: every arm receives the same affinity/descent
assignment computed under the old placement physics. It does not establish
that joint placement/order optimization needs a GNN.

| Stage | Fresh workloads | Frozen control | Control p95 | Reference median gain | Bootstrap 95% interval |
|---|---:|---|---:|---:|---:|
| Initial static-priority screen | 32 | `srpt:128` | 4.903 ms | 8.801% | [6.910%, 9.404%] |
| Adaptive-control challenge | 32 new | `srpt:128` | 4.747 ms | 11.285% | [9.257%, 13.469%] |

`srpt:128` begins with shortest remaining assigned execution time per job,
then searches 128 operation-priority swaps. Both stages choose it on separate
16-workload calibrations among controls below 4.25 ms p95. Its calibration p95
is 4.036 ms initially and 4.085 ms in the challenge. The larger fresh p95s still
fit the actual 5 ms budget. Compute includes the common placement descent,
priority construction and search. Timings use seven runs per case and p95 over
per-case medians on this machine; loading/compilation are excluded. Wall-clock
planning time is not added to simulated execution time.

The reference is the best of the frozen control and 8192-step annealing from
SPT and SRPT priorities. It has more compute, is feasible, and is not claimed
optimal or deployable within 5 ms. All intermediate search proposals are not a
training corpus; retained final control/reference plans are complete for these
gates. No supervised dataset or model checkpoint is produced by this screen.

The adaptive challenge adds nine rules whose keys change with current host
setup, remaining assigned work, ready-operation lock overlaps or suffix lock
pressure. Each also faces 0/64/128/256 swaps, alongside four static starts: 52
candidate controls in total. No tested adaptive rule or its short refinement
beats the selected static-plus-search rule on calibration within the timing
reserve. Since the two screens have different fresh workloads, 11.28% versus
8.80% is not evidence that stronger controls increased headroom. The relevant
finding is that the positive gap survives the expanded control family.

For actual execution, a deterministic adaptive rule's realized operation start
order is converted into static priority ranks. The full adaptive replay is
charged. Native adaptive decisions match separately written SimPy dynamic keys,
start ranks and all completion times on the test fixtures; static replay of the
materialized ranks reproduces the schedule exactly. This equivalence relies on
the fully announced deterministic workflow, not arbitrary unknown arrivals.

### Validation and implementation

The two live gates total **192 HeROsim runs and 9,216 completed operations**
(control, reference and plain SRPT on 64 different fresh workloads). The actual
coordinator uses `mixed_ready_priority_v1` arbitration, recorded alongside the
physics contract and full priorities. Native, independent SimPy and HeROsim
objectives, placements and every task completion agree. Posthoc audits also
replay all 320 retained final plans, verify source/input/artifact hashes and
coverage, and recompute paired gains and p95 timings. Both audits pass.

Historical coordinator/simulator sources are unchanged. The experimental adapter
installs a subclass for one simulation and restores the original coordinator in
`finally`; regression tests subsequently run old FIFO simulation and recover its
original native score. Malformed priority vectors are rejected before native
execution. There are 43 focused dispatch tests, including all nine adaptive
rules against independent dynamic replay and real simulator restoration fixtures.

- `src/placement/radical/dispatch.cpp` and `dispatch.py`: static-priority replay,
  checked priority permutations and swap search.
- `src/placement/radical/dispatch_live.py`: independent static SimPy replay and
  actual HeROsim coordinator adapter.
- `src/placement/radical/dispatch_adaptive.cpp` and `dispatch_adaptive.py`:
  compiled adaptive controls; `dispatch_adaptive_reference.py` independently
  defines their dynamic choices in SimPy.
- `scripts_cosim/mixed_dispatch_gate.py` and `mixed_dispatch_adaptive_gate.py`:
  registered calibration, fresh screen and complete live gate.
- `scripts_cosim/audit_mixed_dispatch.py`: retained-plan/event/statistics audit.
- Both artifact roots contain protocol/source snapshots, exact unique inputs,
  calibration results, per-case priorities/placements, live event records,
  `read.json`, `AUDIT.json` and `INDEPENDENT_AUDIT.json`. The screen's pending
  flag is its pre-live snapshot, superseded by completed audit receipts.

### Remaining gates before a GNN claim

This is stronger headroom than the previous barely passing pair-selector screen,
but repeated prior pilots show that search headroom does not imply learnability.
The constructive witness below completes the equal-local-observation and
representation check; hand-graph features also distinguish those pairs. Next
test a small matched GNN/MP-OFF/hand-graph pilot with full serving-time accounting and fresh live evaluation. Keep the
fixed-placement limitation visible; an end-to-end scheduling claim also requires
controls that can optimize placement and execution order together. The previous
pair-selector and one-shot placement recipes remain closed.

## 2026-09-22 — Constructive ordering witness registration

Before training, [the witness protocol](../../experiments/mixed_dispatch_witness_v1.json)
fixes sixteen new two-job/three-operation pairs. Reverse the two future types
and lock masks within each job while preserving all first-operation and per-job
aggregate features. Use the same fastest-host assignment in each paired variant;
this is a smaller fixed-placement ordering witness, not a recreation of the
main gate's affinity/descent placement generation. Enumerate all720 static
priority permutations and compare exact optima conditioned on each root starting
first. Check the proposed PriorityNet's untrained root representations against
MP-OFF and hand one/two-hop features. Replay both conditional optima from every
variant through actual HeROsim: 64 runs even if no strict preference flips occur.


## 2026-09-22 — Exhaustive ordering witness result

**Two of sixteen pairs strictly reverse the optimal first dispatch despite
identical engineered base observations at both root operations.** All first
operations, processing times, per-job/global type and lock counts, resource
parameters and fixed placements are preserved; only the order of the two future
types/lock masks within each job changes. The root feature vectors include the
existing suffix aggregates and global counts; this is not merely hiding the
number of future peers. The smaller fixture uses fastest-host placements,
so this is a constructive ordering-seat witness, not a population estimate for
the main gate's affinity/descent placement generator.

For seed128002, the exact total-completion minima conditioned on job0/job1
starting first are [115,98] ms before the transformation and [111,118] ms after.
For seed128011, they are [92,103] then [101,84] ms. These are exhaustive optima
over the declared static-priority, non-idling arbitration family, not arbitrary
preemptive or intentionally idling schedules. Other pairs either retain their
preference or permit ties; a tie changing to a preference is not counted.

`PriorityNet` is the proposed two-layer/hidden16 typed node encoder with a scalar
priority head. With a fixed untrained seed, it distinguishes both strict flips.
MP-OFF root outputs are bit-identical across every paired transformation;
the hand one/two-hop graph-feature model also distinguishes both flips. This
proves neither learnability nor GNN necessity. A learned GNN must still beat
that stronger hand-feature control in the actual workload family.

The witness retains all **23,040 priority permutations** (16 pairs ×2 variants
×720), exact costs, root-conditioning and full live events. Both conditional
optima from every variant were replayed in actual HeROsim: **64 additional runs,
384 operations**. A separate audit independently replays every enumerated plan
through SimPy, checks conditional minima, verifies the actual first-started job
and every operation completion, and passes. Combined with the two main screens,
this investigation completes **256 HeROsim runs and 9,600 operations**.

Entry points: `experiments/mixed_dispatch_witness_v1.json`,
`scripts_cosim/mixed_dispatch_witness.py`, `scripts_cosim/audit_dispatch_witness.py`
and `src/policy/dispatch_priority/model.py`. Artifacts are under
`simulation_data/gnn_environment_search_v1/mixed_dispatch_witness_v1/`, including
input/source fingerprints, exhaustive placement/priority JSONL, live records,
`read.json` and `AUDIT.json`. The untrained encoder is a representation probe;
there is no trained checkpoint or GNN gain to deploy or report yet.

## 2026-09-22 — Matched priority-learning pilot completed

The registered pilot completed and is owned by
[mixed_dispatch_learning_v1](mixed_dispatch_learning_v1.md). Nine models and
384 fresh live runs reject normalized total-rank regression with direct decode:
GNN is 19.03% slower than `srpt:128`, slightly worse than both learned controls,
and comfortably within the time budget. This does not reverse the ordering
headroom screens; it shows that this supervised target fails to recover them.

## 2026-09-22 — State/action successor completed

[mixed_dispatch_state_v1](mixed_dispatch_state_v1.md) repaired the unstable
total-rank label by teaching choices only among currently feasible operations.
The fresh corpus retained 10.10% teacher headroom and meaningful choices, but
nine models and 384 live runs leave GNN tied with both learned controls and
9.74% behind `srpt:128`. The parent ordering environment remains active because
the feasible teacher still has 22.56% over GNN; unchanged supervised imitation
of either total ranks or teacher actions is closed.

## 2026-09-22 — Outcome-value successor completed

[mixed_dispatch_value_v1](mixed_dispatch_value_v1.md) labels all feasible
actions by counterfactual final cost and removes repeated PyTorch head calls.
The GNN develops a small but uncertain 1.90% edge over MP-OFF, yet remains 8.98%
behind `srpt:128` and over the serving budget across 384 fresh live runs. Full
learned replacement of the hand rule is closed for this recipe. A selective
residual decoder is still open because it preserves the strong control and asks
the model only to identify measured improvements.

## 2026-09-22 — Scaled CPU runtime calibration

**S0 timing is complete; environment qualification remains pending.** This is a
calibration extension of the active parent, not registration of a block-move
lineage or permission to train. The successor benchmark preserves all historical
runtime/model sources and reads no stored corpus, checkpoint, or test split.

The protocol was saved before generation and timing. It fixes eight workloads
per rung, seven timed repeats after warmup, one CPU thread, and 8×6/16×8/32×8/48×8
job/operation shapes with 4/8/16/24 hosts. Seeds 138000–138007, 138100–138107,
138200–138207, 138300–138307 and parity fixtures 138900–138901 are now used
calibration inputs, never a fresh holdout. Physical identities are checked for
uniqueness within these 34 inputs; this is not an audit of every historical corpus.

The measured proxy constructs the common affinity/descent64 placement and
`srpt:128` priorities, builds base features, runs an untrained hidden16/two-layer
encoder, and exactly scores 32 independent rank swaps. It includes tensor
conversion in the full path. It does not construct structured blocks or run a
learned block-ranking head, and is not a deployable trained policy. Startup and
compilation are excluded; planning wall time is not added to simulated latency.
All figures below are p95 over eight per-workload timing medians on this local
AMD EPYC 7313 CPU. Component percentiles need not sum to the full-path percentile.

| Operations / hosts | Placement descent (ms) | `srpt:128` total (ms) | Full proxy (ms) | Median active hosts |
|---|---:|---:|---:|---:|
| 48 / 4 | 3.192 | 4.354 | 9.865 | 1.779 |
| 128 / 8 | 95.560 | 100.258 | 107.168 | 2.060 |
| 256 / 16 | 1585.919 | 1605.178 | 1617.473 | 2.180 |
| 384 / 24 | 6709.462 | 6748.444 | 6767.053 | 2.197 |

| Operations | Checked dispatch score (ms) | Base features (ms) | Hand 1/2-hop features (ms) | Encoder forward (ms) | 32 exact scores (ms) |
|---|---:|---:|---:|---:|---:|
| 48 | 0.157 | 0.195 | 0.262 | 0.214 | 4.845 |
| 128 | 0.184 | 0.341 | 0.718 | 0.298 | 6.038 |
| 256 | 0.306 | 0.758 | 2.573 | 0.517 | 10.247 |
| 384 | 0.488 | 1.297 | 7.801 | 0.859 | 16.442 |

Before results, the budget ladder was fixed at 5/10/25/50/100/250 ms with a 15%
calibration reserve. Choose the largest rung below 85% of the ceiling, then the
smallest eligible budget: **128 operations / 250 ms**. This is a provisional
engineering budget, not a latency guarantee or objective qualification. The
48-operation proxy needs the 25 ms rung. No proxy fits 5 ms; the historical
48-operation hand control itself still does. A later comparison must retain
both equal proposal count and equal end-to-end wall time, charging placement
for every arm. Changing that placement contract requires a separate explicit
design and new calibration.

**More hosts did not scale the shared resource pools.** The generator retains
six exclusive locks and three power domains; every operation takes one or two
locks, bounding concurrency by six regardless of host count. Measured median
job completion/service stretch rises 4.71 → 7.14 → 13.38 → 17.91. Thus this
run cannot certify the handover's requirement to avoid a trivial congestion
cliff. Capacity scaling needs an explicit contract before a larger-workflow
headroom claim. These measurements do not falsify block moves or establish
residual graph information.

Two 48-operation fixtures passed native/independent SimPy/actual HeROsim parity
before the sweep. Every calibration plan passes independent replay; the first
128/256/384-operation plans also run in HeROsim, for **five live runs and 864
operations**. The separate audit replays all 34 inputs, verifies every live
placement/completion, checks source/artifact hashes, reconstructs `srpt:128`,
and recomputes timing quantiles and budget selection. **Audit: PASS** on all
34 distinct inputs. Dispatch regression tests pass 43/43; record hygiene passes
97/97. Fingerprinted runtime/model sources are unchanged across the run.

Entry points: `scripts_cosim/dispatch_scale_feasibility.py` and
`scripts_cosim/audit_dispatch_scale_feasibility.py`. Artifacts live under
`simulation_data/gnn_environment_search_v1/dispatch_scale_s0_v1/`, including
`protocol_before_run.json`, per-input plans/replays/timing samples, `read.json`,
`artifacts.json`, and `AUDIT.json`. There is no training corpus or checkpoint.
The block-move lineage remains unregistered. Structured moves, ≥8% median
reference headroom, hand search capturing under half that gain, and residual
nonlocal information beyond strong 1/2/4-hop controls are still owed before
training or GPU allocation.

## 2026-09-22 — Capacity and block-search qualification protocol

User-authorized continuation of S0–S2, frozen before calibration or qualification
inputs are generated. The [CPU protocol](../../experiments/mixed_dispatch_block_screen_v1.json)
scales exclusive lock count with hosts and assigns each host to one four-host
power domain (capacity two). Setup costs, atomic locking and dispatch semantics
are unchanged. This is a new capacity configuration, not a reuse of the earlier
fixed-global-pool measurements.

Common placement explicitly changes to affinity plus one full best-improvement
descent pass under the existing FIFO placement score. Every arm gets the same
placement and pays for constructing it; claims remain conditional on placement.
The historical 64-pass contract is not silently amortized or redefined.

Four fresh calibration inputs at each 48/128/256/384-operation rung select the
largest scale whose full proxy p95 fits 212.5 ms of a fixed 250 ms budget, whose
median host utilization is at least 20%, and whose median completion/service
stretch is at most twice the 48-operation rung. The proxy charges placement,
`srpt:128`, typed 1/2/4-hop features, an untrained encoder, block-pool construction,
hand ordering and 32 exact candidate scores. Two small three-way parity fixtures
precede calibration, and the first input at every scale is also replayed live.

At the selected scale, seeds 140000–140015 compare critical/job-chain, lock-chain,
setup-class gather moves, and disjoint priority-block swaps of lengths 2/4/8.
A deterministic seeded pool caps each family at 1,024 proposals. The feasible
reference runs four rounds of broad exact-scored best-improvement descent and
includes every hand arm's retained final outcome; it is not a certified oracle.
All structured accepted moves must strictly reduce the exact objective.

Controls include random, remaining-chain, low/high lock-pressure, setup and typed
1/2/4-hop ordering, plus compiled static/adaptive multistart priority-swap search.
Compare both 32 additional scores and full 250 ms wall time. Wall-time arms run
three times; use the worst objective and require every repeat within budget.
Advance only with ≥8% median reference headroom and the strongest fixed hand arm
capturing strictly under half of summed reference improvement under both budgets.
This is a conservative conditional screen, not a learned-model comparison.

Regardless of those offline bars, execute all **48 qualification live runs**:
baseline, reference and strongest fixed equal-wall-time arm on each environment,
checking every operation against native and independent SimPy. Estimated cost is
under a few local CPU-hours, with no cluster or GPU allocation. A failure stops
training, not the registered live replay. Only a pass opens a separately frozen,
fresh-input residual-information gate beyond all supplied local and strong hand
graph features. Only after that gate passes may a learning lineage, data splits,
matched training and a fresh learned-policy live gate be registered.

## 2026-09-22 — Capacity-scaled block-search result

**PRETRAINING-NO-GO.** The registered finite reference gains **5.0599% median**
over `srpt:128` at 256 operations, below the ≥8% advance bar. The screen completes
its mandatory live evaluation despite this offline failure. No residual probe,
training corpus, learned checkpoint, or GPU allocation follows. The active
ordering parent stays open; no block-move learning lineage is registered.

### Capacity and runtime

The new capacity and placement contracts pass the utilization/stretch checks at
all four scales. Larger workloads no longer retain the old fixed six-lock,
three-domain concurrency ceiling. Calibration uses four workloads per scale and
five timing repeats; p95 is over per-workload medians. It selects 32 jobs × eight
operations on 16 hosts at the fixed 250 ms budget with a 15% reserve.

| Operations / hosts | Full proxy p95 (ms) | Median host utilization | Median completion/service stretch |
|---|---:|---:|---:|
| 48 / 4 | 94.70 | 43.13% | 4.61 |
| 128 / 8 | 98.69 | 34.06% | 6.12 |
| 256 / 16 | 186.18 | 35.52% | 5.95 |
| 384 / 24 | 387.80 | 33.27% | 6.19 |

These are calibration results, not a serving guarantee. Capacity configuration
and common placement effort both changed relative to the previous S0; their
separate causal effects were not isolated. Historical 48-operation headroom
numbers are not paired comparisons with these new inputs.

### Qualification

On 16 distinct qualification environments, exact best-improvement search checks
**241,664 structured candidates** across up to four rounds, retaining every move,
cost/delta, rank fingerprint and round plan. Accepted moves strictly improve the
objective. The reference includes the retained final hand-arm outcomes. It is a
feasible finite reference, not an exhaustive oracle or an upper bound on what a
different proposal space could achieve. Per-environment reference gains range
from 1.31% to 9.31%; only one of sixteen clears 8%.

| Gate | Measurement | Outcome |
|---|---|---|
| Median reference headroom ≥8% | 5.0599% | FAIL |
| Best fixed 32-proposal hand capture <50% | High-lock-pressure ordering: 15.8040% | PASS |
| Measured fixed wall-time hand capture <50% | Random: 41.9424%, but timing-invalid | Not a qualified equal-budget result |
| Best timing-eligible fixed wall-time capture <50% | Native static/adaptive search: 32.7819% | PASS |
| Every wall-time repeat ≤250 ms | Multiple overruns; maximum 987.32 ms | FAIL |
| Residual information beyond strong hand features | Not run after failed prerequisite | NOT ESTABLISHED |

Capture is summed objective improvement divided by summed reference improvement,
paired on the same environments and placements. Wall-time objectives use each
arm's worst of three repeats, which disadvantages the hand controls; these
numbers alone are not evidence that a learned proposer would win. The original
timing-invalid selection and its correction are owned by the
[gate-tool record](../gates/gate-tools.md#2026-09-22--filter-timing-eligibility-before-choosing-a-live-control).
Only high-lock-pressure ordering and native search meet 250 ms on every repeat;
native search is the stronger of these, with a maximum measured 241.98 ms.
Some invalid proposal repeats finish pool construction/ordering after budget
before scoring any additional move. No timing outlier is silently discarded.

The original **48 qualification live runs / 12,288 operations** cover baseline,
reference and the originally selected hand arm. The corrected eligible native
control adds **16 live runs / 4,096 operations** on the same environments. Together
with six calibration/parity runs, this stage completes **70 actual HeROsim runs
and 17,224 operations**. The correction does not change reference headroom or the
NO-GO decision. This is a live validation of the CPU search screen, not a trained
policy live gate.

### Artifacts and scope

Sources: `src/placement/radical/block_moves.py`,
`scripts_cosim/mixed_dispatch_block_screen.py`,
`scripts_cosim/audit_dispatch_block_screen.py`, and
`scripts_cosim/complete_dispatch_block_budget_gate.py`.
Artifacts: `simulation_data/gnn_environment_search_v1/mixed_dispatch_block_screen_v1/`,
including calibration, all candidate JSONL, hand-search traces, final
`placements/placements.jsonl`, live events and separate original/corrected audit
receipts. All earlier fingerprinted dispatch/model sources remain unchanged.
Both audit receipts are **PASS**; the original audit also reexecutes 119,300
hand-search proposals and verifies 34 distinct physical inputs. Validation:
43 historical dispatch tests, seven block/eligibility tests, and all 97 record
hygiene checks pass.
Calibration seeds 139000–139003, 139100–139103, 139200–139203, 139300–139303,
parity seeds 139900–139901 and qualification seeds 140000–140015 are now used;
they cannot be advertised as fresh holdouts later.

The measured search-hardness screen passes against the eligible control, but
headroom and proposal-path timing do not. Repairing latency alone cannot supply
the missing reference gain. A successor needs a materially different, explicitly
specified move/reference or environment contract and fresh measured headroom
before reopening residual-information testing or training. This result does not
establish a GNN/MP-OFF equivalence theorem or close all block-move search.

## 2026-09-22 — Joint placement and ordering control challenge

The user requested further ideas and pursuit of a controlled GNN win. The next
CPU hypothesis removes the fixed-placement restriction: jointly change eligible
hosts and operation priorities. This is distinct from repeating the stopped
four-round priority-block screen. It preserves the scaled setup/locks/power
physics, complete-announcement information, summed job completion objective,
256-operation shape and 250 ms budget. Two other ideas remain unimplemented:
learned selection of larger regions for exact repair, and variable fork/join DAGs
in place of independent chains. Neither is evidence of a GNN advantage.

The [joint-action protocol](../../experiments/mixed_joint_dispatch_screen_v1.json)
is frozen before measurement. New successor C++/Python files implement placement
flips, priority swaps, coupled flips/swaps, and job-block rerouting/reordering.
The expensive reference and hand controls get the same actions. Compiled hand
annealing runs continuously to a wall deadline, with temperature tied to elapsed
budget and a measured atomic-step reserve; it does not pay the previous Python
pool-building overhead. All arms still pay common placement and initial ordering.

Seeds 142000–142003 calibrate pure placement, pure ordering, joint pair, joint
block and joint greedy controls. Select the lowest-cost fixed arm only among
those meeting every 212.5 ms calibration repeat. Freeze that selector before
seeds 143000–143015. Three repeats per fresh arm use its worst objective; report
every arm's timing validity. The reference runs 32,768 proposals for each of two
seeds from each of the selected control and fastest/SRPT starts, retaining the
best feasible result including controls. Require ≥8% median improvement over
the selected affordable hand control and a positive paired bootstrap lower
bound. Also disclose the stronger per-environment eligible-control envelope.

Two small fixtures first run three-way parity for placement, ordering and joint
actions. The 16 fresh workloads then run baseline/control/reference through
actual HeROsim irrespective of headroom outcome: 48 qualification live runs,
plus six small parity runs. Audit all final operation completions, input/source
fingerprints, selection arithmetic, and deterministically reexecute the complete
fixed-step reference search. Estimated cost is under a few local CPU-hours.

Even a headroom pass is only an entry condition for a fresh residual-information
and equal-budget hand-graph challenge. No training corpus, GPU work or GNN win is
authorized by headroom alone. Training, if later qualified, must still use matched
GNN/MP-OFF/hand-graph arms and a fresh live gate. Literature motivation for composite
actions is [GNN-EGDRL](https://doi.org/10.1016/j.cor.2025.107155); that paper's
reported gains do not establish this program's matched-control claim.

## 2026-09-22 — Joint placement and ordering result

**PRETRAINING-NO-GO.** Calibration selects `joint_greedy` before opening the
16 fresh inputs. The finite reference improves median summed job completion
by **0.9079%**, with a paired median-bootstrap 95% interval **[0.00%, 3.07%]**,
against the required 8% and strictly positive lower bound. None of the sixteen
inputs reaches 8%; five retain the affordable control itself as the reference.
The reference searches **2,097,152 proposals** in total: 32,768 per trajectory,
two random seeds, two starts, sixteen environments. This is a feasible reference,
not an exhaustive optimum or an upper bound on a different repair method.

All five control families are eligible on both calibration and fresh inputs.
Every fresh repeat fits 250 ms; the largest observed time is **247.5501 ms**.
The selected control scores 995–1,116 additional proposals per call after paying
for the common placement/order baseline. Each arm's reported objective is its
worst of three repeats, as registered; thus the reference margin does not rely
on taking the hand control's best timing draw.

| Affordable control | Median reference improvement |
|---|---:|
| Placement-only annealing | 4.6309% |
| Ordering-only annealing | 3.5824% |
| Joint-pair annealing | 4.6070% |
| Joint-block annealing | 5.6201% |
| Calibration-selected joint greedy | **0.9079%** |
| Per-environment best eligible arm, diagnostic | **0.9079%** |

The selected joint control beats ordering-only search by **1.6331% median**,
paired median-bootstrap 95% interval **[0.93%, 3.42%]**, and beats the common
baseline by 4.2170% median. Joint actions help the hand algorithm; they do not
establish a learnable residual or a message-passing advantage. The global
best-arm envelope is disclosed only as a stronger diagnostic, not used to
change the calibration-frozen primary comparison.

All **48 qualification HeROsim runs / 12,288 operations** complete, regardless
of the failed headroom screen. Six small parity runs add 96 operations, for
**54 live runs / 12,384 operations** overall. The independent audit **PASSES**:
22 distinct inputs, all 2,097,152 reference proposals deterministically
reexecuted, final hand plans checked, and every live placement and completion
compared against independent replay. All 73 focused dispatch tests and 97
record-hygiene checks pass.

Sources are `src/placement/radical/joint_dispatch.cpp`, its Python wrapper, and
`scripts_cosim/mixed_joint_dispatch_screen.py` (including `--audit-only`).
Artifacts are under
`simulation_data/gnn_environment_search_v1/mixed_joint_dispatch_screen_v1/`:
the pre-run protocol/source manifest, native compiler/library provenance,
calibration selector, all final search plans, live/replay events,
`placements/placements.jsonl`, report and artifact hashes. Historical
fingerprinted sources are unchanged. Calibration seeds 142000–142003, parity
seeds 142900–142901, and qualification seeds 143000–143015 are now consumed.

No training data, GPU allocation or learned-model result follows from this
screen. Residual graph qualification remains **NOT RUN** after the failed
prerequisite. Reopening requires a materially different specified reference or
environment with fresh qualification, not a weaker hand arm. The next distinct
candidate is selecting regions for coordinated exact repair; a successor must
first show that its repairs produce headroom and fit the budget, and must give
the same repair engine to strong hand selectors. Variable fork/join workflows
remain a separate, unimplemented environment hypothesis.

## 2026-09-22 — Exact region selection S0 protocol

The user authorized pursuit of further ideas. Before registering a learned
region selector, [this CPU S0](../../experiments/mixed_region_repair_s0_v1.json)
tests whether a bounded exact repair can be both useful and affordable. A region
contains four operations. Enumerate all 16 eligible placement combinations and
all 24 permutations of their existing priority ranks, holding every outside
decision fixed. This is exact only within that 384-plan neighborhood.

Freeze four new seeds, 144000–144003, at 256 operations / 16 hosts. A 140 ms
joint-greedy start leaves room for one repair; the comparison remains the same
joint-greedy control with the full 250 ms. Both starts pay the common baseline
and use the worst objective of three repeats. For each input, search 128 regions:
96 sampled using mixed precedence/resource adjacency and 32 random regions.
Give the selector the best region for free and allow the full-budget control as
an alternative. This is deliberately optimistic about selecting the right
region; no learned-model claim follows from it.

Require 8% median headroom even under this free selector, and require every
repeated winning repair plus the partial start, pool construction and a 10 ms
future selector reserve to fit 250 ms. This four-input S0 is a feasibility screen,
not confirmation. A pass requires a fresh sixteen-input hand-selector and
nonlocal-information challenge before any training. A fail stops this finite
four-operation repair recipe without bounding other region sizes or repairs.

Regardless of outcome, run all twelve HeROsim cases (partial start, full control,
best repaired plan per input), verify all operation placements/completions, and
independently audit the 196,608 exact candidates. Sources are new successor files
`src/placement/radical/region_dispatch.cpp`, its Python wrapper, and
`scripts_cosim/mixed_region_repair_s0.py`; all preceding sources stay frozen.

## 2026-09-22 — Exact region selection S0 result

**S0-NO-GO.** Free perfect selection from 128 exact four-operation repairs
finds **0.5456% median headroom** over the full-budget joint-greedy control,
against the fixed 8% bar. The four input gains are **0.00%, 1.0912%, 0.00%,
1.6740%**. This is an exploratory four-input feasibility result, not a powered
claim about all region search or a GNN/MP-OFF comparison.

Timing passes: the longest optimistic path, including the 140 ms start,
pool construction, winning exact repair and a 10 ms selector allowance, is
**220.1548 ms**. All measured full-control and partial-start calls meet their
respective 250/140 ms budgets. The favorable selection and runtime assumptions
still do not expose enough objective gain. This stops the registered serving
recipe before a learned selector or fresh residual-information gate.

The screen enumerates **196,608 candidates** (four inputs × 128 regions × 384
plans). All **12 HeROsim runs / 3,072 operations** finish, covering the partial
start, full-budget control, and best repair on every input. Independent audit
**PASSES**, reexecuting all 196,608 candidates and checking every live placement
and completion. Six new tests pass, including native repair versus independent
SimPy exhaustive enumeration for region sizes one through four. The previous
73 dispatch tests also pass; no fingerprinted predecessor source changed.

Artifacts and frozen protocol/source hashes live in
`simulation_data/gnn_environment_search_v1/mixed_region_repair_s0_v1/`, with
all optimized region plans, timing repeats, final `placements/placements.jsonl`,
live/replay events and an artifact manifest. Seeds 144000–144003 are consumed
exploration inputs, never fresh holdouts. No GPU was allocated and no training
corpus or model was produced.

The next unimplemented environment hypothesis is irregular fork/join precedence
in place of independent fixed-length chains, while preserving the setup/lock/
power mechanisms and objective. This does not reopen the stopped DAG-output
fabric contention direction. It first needs a native/SimPy/HeROsim parity
contract, capacity/runtime feasibility and fresh headroom against critical-path,
downstream-work, 1/2/4-hop and equal-budget search controls. Larger coordinated
repairs remain a second algorithmic hypothesis only if their full serving cost
and new headroom can be demonstrated. Neither is a promised GNN win.

## 2026-09-22 — Scope correction and certified reservation witness

The user challenged whether the recent screens justify not training in general.
**They do not.** Their original numerical results and recipe-specific funding
decisions stand. Three distinctions govern the interpretation:

- A feasible search reference provides an upper bound on optimal cost, hence a
  **lower bound on available improvement** over a hand control. Failure to find
  a large improvement is not a certificate that none exists. A bound excluding
  improvement needs an optimality certificate or a sufficiently tight lower
  bound on optimal cost.
- Perfect region selection is an upper limit only within the specified 128
  regions, four-operation neighborhood, fixed partial start and serving recipe.
  Four exploratory inputs do not establish a population-wide equivalence.
- The 8% advancement bar is a registered resource-allocation decision, not a
  mathematical prerequisite for learning or evidence that smaller gains cannot
  be valuable. It remains unchanged for the completed screens.

### Shared action restriction

The historical `priority_replay` dispatches every feasible ready operation in
priority order, then advances to the next completion. It has no intentional
waiting action. The block, joint and region references all inherit this same
restriction. More trajectories or alternative labels within that decoder cannot
reach schedules that require deliberate reservation.

The new [witness protocol](../../experiments/dispatch_reservation_witness_v1.json)
is a **constructed example after exploratory enumeration**, not an independent
sample from the earlier large-workflow distribution. Two jobs each have two
operations and both hosts are eligible for every operation. Processing durations
stay within the original 3–18 range. Setup types are identical, locks are distinct
between jobs, and the two-host power domain allows two concurrent operations.
Those extra constraints do not bind; the example isolates dispatch expressivity.

| Operation | Host 0 duration | Host 1 duration |
|---|---:|---:|
| Long job, first | 12 | 18 |
| Long job, second | 18 | 3 |
| Short job, first | 18 | 3 |
| Short job, second | 3 | 18 |

All **16 placements × 24 priority permutations = 384 plans** were scored by the
historical native engine and independent SimPy. The best objective is **30**:
both jobs complete at 15. Reserving host 0 until time 6 lets the short job finish
at 6 and the long job at 21, giving **27**, a **10% improvement**.

This fixture also admits a global certificate. Enumerate all placements and
per-host operation orders: **120 combinations, 52 acyclic** after adding job
precedence. Compute earliest starts in each acyclic order. Every feasible
nonpreemptive schedule induces such machine orders, and left shifting cannot
worsen this objective with zero active setup costs and no additional binding
constraints. The minimum is **27**. This certification is specific to the small
fixture, not a solver for arbitrary setup/lock/power cases.

### Live verification and implications

New `src/placement/radical/reservation_dispatch.py` adds opt-in not-before times
and wakeups, preserving all historical sources. Zero reservations reproduce the
old replay. `scripts_cosim/dispatch_reservation_witness.py` produces four live
runs: historical best **30**, zero-reservation parity **30**, certified schedule
**27**, and simple hand reservation **27**. All **16 operations** match their
independent replay placements and completion times. The audit **PASSES**,
reexecuting the complete non-delay sweep, the certificate and all live checks.
Artifacts, `placements/placements.jsonl`, source/native hashes and audit receipt
are in `simulation_data/gnn_environment_search_v1/dispatch_reservation_witness_v1/`.
The fixture uses seed 145900 as a reproducible substrate; seeds 145901–145903
are unit-test inputs. They are not fresh qualification splits.

**No GNN was trained and no GPU was allocated.** The hand reservation reaches
the same optimum, so the witness does not justify learning that trivial rule.
It instead demonstrates why a conclusion about all training from the old action
family is unsupported. The next reservation challenge must include the wait
action in every arm and beat hand reservation/lookahead plus equal-budget search
on fresh, nontrivial workloads before a message-passing claim is considered.

## 2026-09-22 — Irregular DAG reservation qualification protocol

The user authorized implementing reservations on irregular fork/join workflows.
The [CPU protocol](../../experiments/dag_reservation_screen_v1.json) is frozen
before scale calibration or qualification data. New successor files are
`src/placement/radical/dag_reservation.cpp`, its Python wrapper,
`src/placement/radical/dag_reservation_live.py`, and
`scripts_cosim/dag_reservation_screen.py`. Historical source fingerprints are
checked before and after the run. No training lineage or GPU allocation is
registered by this screen.

### Environment and actions

Each announced job has sixteen operations arranged in random-width layers,
additional forward edges, one source and one sink. Branches may execute
concurrently. Every operation retains two eligible hosts, heterogeneous durations,
sequence-dependent setups, atomic shared locks and capacity-scaled power domains.
This changes precedence and planning actions, not DAG-output network physics.
The objective remains summed job completion time; the distinct raw task-RTT
metric is covered by the [gate-tool correction](../gates/gate-tools.md#2026-09-22--read-dag-job-latency-from-terminal-completion-not-summed-task-rtt).

The non-delay decoder schedules feasible ready operations at event times. The
reservation decoder constructs a precedence-feasible sequence and books future
host, lock and power capacity. It may leave a resource idle for a later-ready
operation. Its resulting earliest-start constraints are served through actual
HeROsim, while native event replay and independent SimPy check the plan. All
hand controls receive reservation access where their action ablation permits it;
the primary control can be a portfolio of both decoders.

### CPU gates and controls

Two inputs each at 64/128/256/384 operations use seeds 146000–146001,
146100–146101, 146200–146201 and 146300–146301. Three repeats time complete hand
initialization plus 1,024 search proposals and final materialization. Select the
largest rung with p95 ≤200 ms and median host utilization ≥0.15. The full serving
budget is 250 ms; if none qualifies, complete the fourteen parity/scale live runs
and report scale not qualified.

Four calibration inputs, 146800–146803, select among nine fixed controls:
non-delay, reservation and mixed-decoder search, each greedy or annealed at
relative temperature 0.001 or 0.01. Every call evaluates fastest and affinity
placements with short-job, short-operation, critical-path, downstream-work,
lock-pressure and 1/2/4-hop hand-graph priorities. Every cost, including feature
construction and initialization, is charged. Unlike prior worst-repeat screens,
use the median objective of three repeats; retain every timing draw. Select the
lowest calibration mean only among arms meeting every 212.5 ms repeat, then
freeze that choice before qualification.

Sixteen new inputs, 147000–147015, compare all nine controls at 250 ms. The
reference runs eighteen 8,192-proposal trajectories: three decoder modes, three
temperatures and starts from both selected control and full heuristic portfolio.
It retains every final hand/reference winner. This is a diverse feasible
reference, **not** an optimality certificate. Report the calibration-frozen
primary and the stronger per-input timing-eligible envelope. Keep the 8% median
headroom and positive paired-bootstrap lower bound as advancement criteria,
without interpreting a failure as general learning futility.

### Live verification and conditional next work

Six small live runs first cover the certified reservation witness, a truly
concurrent diamond and a mixed-resource irregular DAG. All eight scale cases,
four selected calibration controls and 48 fresh baseline/control/reference cases
then run in actual HeROsim: **66 registered live runs** if scale qualifies.
Verify served edges, sink coverage, placements and all operation completion
times. Audit source/input/artifact hashes and deterministically reexecute every
fixed-step reference trajectory. Estimated cost is under a few local CPU-hours.

The preflight suite passes thirteen tests, including the old-chain reduction,
independent native/SimPy parity, search determinism, concurrent branches, actual
HeROsim diamond execution and the certified reservation witness. Test seeds
148801–148809 are not qualification inputs; parity also uses 148900–148901 and
the already-open constructed witness at 145900.

A headroom pass only permits a fresh hand-reservation/lookahead and residual
information challenge beyond strong trained 1/2/4-hop controls. Matched GNN,
MP-OFF and hand-graph training still requires those gates, deterministic trainers,
W&B records and a fresh live gate. Failed qualification stops this recipe's
advancement, not all reservation learning or all irregular DAGs.

## 2026-09-22 — DAG result and schedule-translation successor

**First result: RECIPE-NOT-QUALIFIED.** Complete initialization, 1,024 proposals
and materialization have p95 21.53/60.68/196.11/417.86 ms at 64/128/256/384
operations; 256 qualifies. Calibration selects `portfolio_greedy`. Across sixteen
fresh inputs, the finite reference improves this control by median **3.94%**,
paired-bootstrap interval **[2.34%, 5.05%]**. The stronger per-input hand envelope
leaves 2.36%. All nine arms meet their 250 ms budget, but the primary result misses
the registered 8% advancement bar. No training or GPU allocation follows.

Artifacts are in `simulation_data/gnn_environment_search_v1/dag_reservation_screen_v1/`.
The main audit **PASSES**: 31 inputs, **66 live runs / 15,024 operations** and
2,367,488 reproduced proposals. The separate resource-event audit **PASSES** all
30,048 start/completion events, placements, base/setup times and release constraints.
These comparisons use actual HeROsim sink completion, with raw task RTT separate.

A post-hoc representation diagnostic on the opened sixteen inputs found that
chronological ranks preserve every selected incumbent exactly, while reusing its
original event priorities in the serial reservation decoder increases median cost
by 201.17%. This is a search-initialization weakness, not a physical-parity failure;
its correction is owned by the [gate-tool entry](../gates/gate-tools.md#2026-09-22--translate-schedule-semantics-between-search-decoders).
Historical code, reports and conclusions about their measured costs remain frozen.
`scripts_cosim/diagnose_dag_reservation_translation.py` independently reproduces
that post-hoc check and writes the separate
`simulation_data/gnn_environment_search_v1/dag_reservation_bridge_diagnostic_v1.json`
receipt, outside the historical artifact manifest.

### Fresh successor protocol

Before new measurements, [dag_reservation_bridge_v1.json](../../experiments/dag_reservation_bridge_v1.json)
fixes four calibration seeds 149000–149003 and sixteen qualification seeds
150000–150015 at 256 operations. The nine controls, median of three timing repeats,
250 ms serving budget, 212.5 ms calibration allowance, eighteen 8,192-proposal
reference trajectories and 8% advancement bar remain unchanged. The new compiled
search translates the incumbent's chronological execution order before switching
decoders; reservation initialization also considers that translated non-delay
incumbent. Every conversion is charged to wall time. Proposal counts count mutations,
not additional materializations; fixed-step audits reproduce both.

New successor sources are `src/placement/radical/dag_reservation_bridge.cpp`, its
Python wrapper and `scripts_cosim/dag_reservation_bridge_screen.py`. Six regression
tests pass, including deterministic search and non-worsening initialization. Test
seeds 149901–149904 are excluded from qualification. Six small parity runs use
149900 and 149905. Four selected calibration plans and 48 fresh
baseline/control/reference plans bring the mandatory live gate to **58 runs /
13,408 operations**, regardless of headroom. Source/input/artifact audits and
complete resource-event checks follow. Expected cost remains a few CPU-hours.

A headroom pass permits the stronger hand-reservation/lookahead and residual
graph-information challenge, not immediate model training. A failure stops this
successor recipe; it does not establish general GNN or reservation futility.

## 2026-09-22 — Schedule-translation successor result

**RECIPE-NOT-QUALIFIED.** Calibration freezes `nondelay_cool`. Across sixteen
fresh 256-operation inputs, the larger feasible reference improves it by median
**5.4282%**, mean 5.2544%, with paired-bootstrap median interval
**[3.8809%, 6.4636%]**. The timing-eligible per-input hand envelope leaves 4.2834%.
Every arm passes every fresh 250 ms timing draw; the maximum over all arms and
repeats is 244.92 ms. Thus the timing gate passes and the registered 8% headroom
gate fails. Do not use the 10–14% gains against the weaker reservation-only arms
to claim qualification against the selected strong control.

There is a useful action-space signal: among the evaluated candidates, reservation
plans strictly beat every evaluated non-delay plan on **11/16** inputs. This is
a comparison of the finite observed candidates, not an optimality claim or a
GNN result. Descriptively, the selected 250 ms hand search captures median **63.35%**
of the measured baseline-to-reference improvement, where the baseline is the full
heuristic initializer and the per-input fraction is
`(baseline - control) / (baseline - reference)`. The stronger residual-information
challenge was not reached. The previous and successor screens use different
fresh inputs, so their 3.94% and 5.43% results do not isolate the correction's
causal effect.

The completed artifact directory is
`simulation_data/gnn_environment_search_v1/dag_reservation_bridge_v1/`.
All **58 live runs / 13,408 operations** pass; the resource-event audit checks
**26,816 events**, placements, durations, setups and not-before constraints.
The full fixed-step search-reproduction audit **PASSES**, reproducing all
**2,359,296** reference proposals across the sixteen fresh inputs and verifying
22 input identities, frozen sources, native binaries, saved plans and report
arithmetic. The focused
reservation/DAG/dispatch suite passes **67 tests**. Separate post-hoc diagnostics
are `dag_reservation_bridge_diagnostic_v1.json` and
`dag_reservation_bridge_search_diagnostic_v1.json` in the artifact directory's
parent; both carry input/source hashes and are not fresh qualification splits.

**No training and no GPU allocation.** This applies the predeclared spending
rule to this recipe. The feasible reference is not an upper bound on possible
improvement, the 8% threshold is not a necessity theorem for successful learning,
and the experiment does not reject all reservations, irregular DAGs or GNNs.
The parent remains ACTIVE. Any successor must keep the strong non-delay control,
charge schedule conversion, and use fresh inputs; residual graph information and
matched learned comparisons remain unanswered. Record hygiene passes **97 tests**.

## 2026-09-22 — Trained DAG successor

The user explicitly authorized training despite the CPU funding bar. The
[new child](dag_reservation_learning_v1.md) owns the complete data, training and
live-gate record. It closes guarded one-shot complete-plan proposals with no
GNN win. The parent stays ACTIVE for different reservation decisions; the
earlier CPU recipe verdict and its finite-reference caveat still stand.
