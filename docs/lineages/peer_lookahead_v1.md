# peer_lookahead_v1 — CLOSED

> Produced on local-record-2026-10-06 at afea04bd; code not merged; old physics.

> **Status:** `CLOSED` (2026-09-21) · **Index:** [LINEAGES.md](../../LINEAGES.md)

**Outcome (2026-09-21): DO-NOT-ADVANCE this hand-lookahead candidate.**
Fresh live simulation on 16 distinct infrastructures × two 1,024-task traces
finds median RTT gains of **0.021% against the immediate rule** and **2.823%
against peer mass** for first-task-only two-hop lookahead. Both fail the fixed
5% advance bar; both one-sided bootstrap lower bounds are zero. Baseline load
screens pass on every infrastructure. The larger validation supersedes the
small pilot's promising magnitude, without proving model-class equivalence.
No learned GNN or MLP was trained, and the conditional three-arm rematch is not
launched. Closure covers this tested uncapped cycle/manifest policy proposal,
not all partial-arrival environments. Criteria were saved before execution,
not committed/signed as a powered preregistration.

**Parents:** [joint_burst_v2](joint_burst_v2.md),
[rollout_imitation_v1](rollout_imitation_v1.md),
[peer_affinity_v1](peer_affinity_v1.md).

## Question and scope

Does the placement of a current task depend materially on downstream peer
structure after matching its local information, and does that dependence survive
cheap hand-computed lookahead and search? The K4 twin is a competent co-location
control; repairing the separate five-feature rollout scorer is not a prerequisite.

This lineage does not reopen the closed warm-corpus, horizon-return or node-count
physics directions. It measures an explicitly declared future-peer-information
contract first on frozen paper costs, then in complete live simulations. No existing checkpoint can be served under that
new contract without rebuilding its cache and retraining.

## Observability audit

The orchestrator constructs the full `peer_exchange` table at initialization, but
fills `task_by_id` only as the gateway creates applications. Peer IDs and payloads
are thus globally available inside the simulator; future Task objects, candidate
sets and future queue states are not current scheduler observations.

`attach_live_prefix_block` restricts peer edges to tasks in the current batch.
An outside peer increments `peers_outside_batch` and is skipped. The screen calls
this actual function with one arrived task and a known three-task peer chain.
The current contract must report one outside peer and zero decoder peer edges.

For a future rematch, declare an application manifest at first arrival: peer
structure, task types, source nodes, demands, and a candidate forecast derived
from current infrastructure. Use the same manifest and current state for all
arms; exclude future placements, future queues and realized completion times.
Distinguish forecast eligibility from replicas actually admitted when a task
arrives. The current serving path does not implement this manifest contract.

## Fixed exploratory screen

Entry points: `experiments/peer_lookahead_screen_v1.json` and
`scripts_cosim/peer_lookahead_screen.py`; output:
`simulation_data/peer_lookahead_v1/screen.json`.
The adjacent `screen.sweeps.npz` retains every enumerated paper cost, including
exchange-off controls, with a shared placement-index array. These are analytical
screen artifacts, not co-simulation snapshots or a training corpus. Source
snapshot sweeps remain in their original `placements/placements.jsonl` files.

- Select 24 sources, every eighth sorted snapshot from the existing `arm_b0`
  corpus. Fixed paper draws 42100–42123; no selection on results.
- Reuse the existing `peer_affinity_probe.Source`/`Paper` base costs and candidate
  construction: eight tasks, three candidates, uncapped. These are forecasts
  frozen at a source snapshot, not costs observed after future arrivals.
- Two eight-task cycles differ only in edges beyond task 0's two-hop view. Root
  neighbors, every task's weighted degree, types, candidates, base costs and
  demand stay fixed. All peer edges carry 200 MB. No extra memory cap.
- Enumerate all 3^8 placements. Cost is additive base + the existing paper
  count-shaped sharing approximation + peer transfers charged at BOTH endpoints.
  The latter matches runtime accounting, unlike counting an undirected transfer
  once. This is not simulator RTT: no arrival, rendezvous or draining dynamics.
- Root completion costs minimize over all other placements for each task-0
  candidate. Disjoint optimal candidate sets prove a local-information ambiguity.
  The best single choice across both matched variants gives an information-loss
  lower bound for a scorer restricted to those matched features.
- Controls: immediate greedy, sequential candidate-aware peer mass, peer min-sum candidate
  columns at 1/2/3/8 rounds, eight-round min-sum including sharing factors, and
  three-pass coordinate descent from one or multiple deterministic starts.
  Min-sum is graph computation supplied as engineered columns to a pointwise
  scorer, not evidence that graph computation is absent. Multi-start CD is a
  stronger search budget than the existing live CD rule; label it separately.
- Report both root-choice regret with oracle completion and full-plan regret.
  Neither is a measured performance estimate for an MLP or GNN. Compare the
  exchange-off matched pair to rule out topology changes in other costs.

Interpretation fixed before reading: 5% regret is the materiality reference, not a
statistical significance threshold. A zero median or a strong control that removes
the gap argues against a training campaign on this design. A residual only buys a
simulated partial-arrival pilot. Report tails and all cases, including ties.

## Conditional trained rematch — not launched after validation

Keep three trained arms: MP-on, separately trained MP-off, and a pointwise MLP
with the strongest engineered lookahead columns. Match information, corpus,
label, decoder, selector and tuning budget. Keep immediate greedy and CD as
policy references. Train through experiment configs with W&B and checkpoint
contracts, never inside this probe.

Before registration: implement and test the manifest contract, cache/serving
parity and feature invariance to forbidden future state; demonstrate finite
partial-arrival execution with real rendezvous/queue accounting; verify stable
labels across completion horizons and healthy reactive load. Use the actual
served decision as the label. This is a new environment/representation study,
not a flag change on K4 weights.

Proposed live bar: at least 5% lower paired total RTT against BOTH learned
controls, with a pre-specified joint decision rule and power calculation. Pair
within environment and training draw; aggregate repeated environments within
draw, or pre-specify a hierarchical analysis. Derive the number of fresh draws
from a separate pilot; 16 is a floor, not proof of power. Report uncertainty
against the 5% bar, rather than testing only a zero effect. Retain missing draws
and hung cells in an explicit failure rule. Queue, exchange, rendezvous and direct
co-location counters are secondary diagnostics, not independent causal effects.

Cost: this paper screen is CPU-only (24 × 2 × 6,561 placements), with no GPU
training. A 16-draw, 16-environment three-arm live gate would require at least
768 learned-policy simulations plus shared rule baselines, before power top-ups
or a second load rung. Measure per-run cost in the pilot before allocating it.
The conditional trained comparison was never registered. The lineage now closes
on the completed live-simulation validation below; it does not cancel a registered
trained gate or claim to have measured the three trained arms.

## Record

- 2026-09-21 — **Fresh-infrastructure live validation: DO-NOT-ADVANCE.**
  Entry points: `experiments/peer_lookahead_validation_v1.json` and
  `scripts_cosim/peer_lookahead_validate.py`. Artifacts:
  `simulation_data/peer_lookahead_v1/validation/read.json`, the adjacent
  `protocol_before_run.json`, and per-source `read.json`, workloads and logs.
  Protocol/config and relevant code hashes were captured before execution.

  **Independent units and information.** The source corpus's 204 snapshots
  represent 34 distinct infrastructures. Select 16 not used in the live pilots;
  ten are also unseen in the paper screen, six were paper-screen sources.
  Selection used infrastructure identity and prior-use metadata, not validation
  outcomes. Two new seeded traces per infrastructure each contain 1,024 tasks,
  with shuffled task types, randomized cycle choice and jittered spacing within
  128 eight-task peer groups. The manifest is exposed at the first group arrival;
  other future groups, realized future placements and future queues are hidden.
  These are synthetic workloads on existing infrastructures, not production data.

  **Load and controls.** Reactive and immediate baselines alone choose the first
  passing mean arrival gap from 1/2/4 seconds. Both seeds and both baselines must
  pass queue/RTT ≤0.8 and final-quarter mean RTT ≤1.2 × second-quarter mean +0.5 s.
  All 16 pass at 1 s; none is dropped or replaced. Candidate results never select
  the load. All 32 candidate traces also pass these diagnostics; finite-trace
  diagnostics do not prove steady-state stability. Run immediate, reactive,
  peer mass and first-task-only two-hop at the selected load: **128 complete
  simulations**, 131,072 task completions, approximately 76 s on two local workers.

  **Fixed advance rule and result.** First take the median paired percentage
  gain over the two traces of each infrastructure, then the median over 16
  infrastructures. Bootstrap infrastructure units 20,000 times for a one-sided
  95% lower bound. Require median and lower bound ≥5%, positive gain on ≥75%
  of infrastructures, and no individual trace regression >10%, against BOTH
  immediate and peer mass. No training draws or repeated traces count as
  independent infrastructures.

  | Control | Median RTT gain | Lower bound | Infrastructure wins | Worst trace regression |
  |---|---:|---:|---:|---:|
  | Immediate rule | 0.021% | 0.000% | 9/16 | 0.170% |
  | Peer mass | 2.823% | 0.000% | 9/16 | 7.653% |

  Both comparisons pass the fresh-trace regression limit but fail all three
  benefit requirements. The earlier 16-task counterexample remains in the
  record; passing the fresh-trace limit does not erase it. Median co-location
  is already 99.902% for the immediate rule, versus 100% for the candidate;
  exchange/task medians are 0.00119 s and zero. This offers little median
  exchange headroom in the tested uncapped regime. Peer mass itself is weaker
  on many environments, so beating it alone would not establish useful benefit.
  Wall-clock planning is measured separately and does not advance simulated
  time; the failed gain bar needs no optimistic overhead correction to fail.

  **Decision.** Do not promote this candidate or begin a GNN training campaign
  on the strength of its small pilot. The current policy/environment proposal
  fails its advance rule; this is not a powered equivalence result and does not
  test learned message passing. Reopening needs a materially different,
  justified environment with verified residual headroom, followed by the same
  information-matched three trained arms if the environment survives controls.
  Simply increasing seeds or training volume on this proposal is not the next step.

  **Artifact audit.** Independently recomputed the summary; verified all source
  file hashes and 16 distinct infrastructure hashes, all 128 full task/decision
  arrays, and baseline health from retained statistics. No incomplete run or
  source error was discarded.

- 2026-09-21 — **Complete-trace hand-control comparison.** Entry points:
  `experiments/peer_lookahead_trace_v1.json`, `scripts_cosim/peer_lookahead_trace.py`;
  output `simulation_data/peer_lookahead_v1/complete_trace/read.json` and adjacent
  logs. User authorized serving the two-hop control throughout complete traces,
  including the earlier failure case. No model training or production registry
  changes; the simulator policy wrapper is scoped to each run and restored.

  **Design.** Same three source infrastructures, two cycles, 0.1/1 s arrival
  gaps, eight task types and 200 MB edges. Each complete synthetic trace has
  32 consecutive eight-task groups (256 tasks), with peer edges only within a
  group. A group's metadata is announced when its first task is scheduled;
  controls cannot see other future groups. Known executed placements become
  fixed candidates; unplaced tasks use current replica/cost forecasts. The
  decision is recomputed from the actual live state at each relevant arrival.

  Five arms: immediate rule; committed exchange ×2 (no future-peer term);
  hand peer mass; two-round min-sum hand lookahead on every arrival; two-round
  lookahead only on each group's first task, immediate rule thereafter. The ×2
  ablation checks whether pricing both transfer endpoints alone explains a gain.
  All arms receive the same declared information. The committed-only control
  computes the common forecast scaffold for instrumentation but ignores its
  future terms, so its overhead is not an optimized rule implementation.

  **All 65 runs completed:** 12 matched traces × five arms, plus the exact earlier
  16-task failure on each arm; 15,440 task decisions, about 32 seconds locally.
  Comparison is paired within each trace. The two cycles/gaps reuse source
  infrastructures and are not independent training draws.

  | Arm | Median RTT change vs immediate, % | Wins /12 | With directly added decision wall time, % |
  |---|---:|---:|---:|
  | Committed exchange ×2 | 0.00 | 3 | +0.02 |
  | Peer mass | −17.31 | 9 | −17.28 |
  | Two-hop, every arrival | −23.83 | 10 | −23.80 |
  | Two-hop, first task per group | −23.83 | 9 | −23.82 |

  Every-arrival two-hop versus peer mass is −6.42% paired median (7/12 ahead),
  but its worst comparison is **+34.32%** (`ds_00064`, 0.1 s gaps, rewired cycle).
  On that same trace it is **+169.68%** slower than first-task-only two-hop.
  Versus first-task-only overall, it wins 2, loses 1 and ties 9; median difference
  is zero. Equal aggregate medians against the immediate rule must not hide
  these differences. The ×2 ablation's zero median means the measured lookahead
  margin is not explained by doubling known transfer costs alone.

  **Load disclosure.** At 0.1 s gaps the two-hop median versus rule is −61.36%
  (6/6); at 1 s gaps it is −22.25% (4/6, two ties). The rule's quarter-mean RTT
  on `ds_00064`, 0.1 s gaps, rises 1.81→5.25 s (base cycle) / 1.82→3.83 s
  (rewired), so its fast-rung losses include accumulating queues. Queue share
  alone (0.778/0.710) would not flag those traces at the historical 0.80 bar.
  This is not a steady-state healthy-baseline gate, and the −23.83% pooled median
  must not be promoted as a production or general unsaturated speedup.

  **Earlier failure retained.** Exact `ds_00064`, 1 s gaps, rewired eight-task
  group plus eight peer-free suffix tasks: immediate 23.219 s, committed ×2
  24.372 s, **peer mass 19.117 s**, both two-hop modes **24.718 s** total RTT.
  Peer mass co-locates 8/8 edges; two-hop co-locates 6/8 and pays 5.618 s exchange.
  The every-arrival version does not repair the original first-choice error.

  **Overhead.** Across traces, the median of per-trace median decision times is
  0.767 ms for every-arrival two-hop versus 0.062 ms for the immediate rule;
  the largest two-hop per-trace p95 is 2.387 ms. These measure this implementation
  locally. The adjusted column above adds measured decision wall time directly
  to summed RTT; it does **not** insert inference delay into the simulation clock
  or model queue amplification from scheduler latency. First-task-only's median
  decision time is dominated by its seven immediate decisions per group.

  **Interpretation.** A useful hand-lookahead direction survives actual feedback,
  but neither every-arrival two-hop nor the current synthetic design justifies
  a GNN training campaign. First-task-only is the cheaper, more stable candidate
  on this small set; peer mass must remain a control because it wins the original
  regression. A fresh, larger, load-controlled trace comparison is needed before
  claiming a robust policy improvement. No parameters were tuned to these results.

- 2026-09-21 — **Short-trace real-engine partial-arrival probe completed.** User
  authorized testing timing/rendezvous omitted by the frozen paper screen.
  Entry points: `experiments/peer_lookahead_live_probe_v1.json`,
  `scripts_cosim/peer_lookahead_live_probe.py`; result and per-run logs/workloads:
  `simulation_data/peer_lookahead_v1/live_probe/read.json` and adjacent files.
  This probe uses the actual simulator, not the earlier paper objective.

  **Design.** Three source infrastructures (`ds_00000`, `ds_00064`, `ds_00128`),
  gaps 0.1/1.0 seconds, both previously specified peer cycles, fixed eight task
  types, 200 MB per edge. Complete horizons of 8/12/16 arrivals; the additional
  arrivals have no peer edges, so no horizon truncates a peer component. Each
  source retains its seeded infrastructure/replicas and warmth physics. This is
  a finite, short workload, not an established steady-state unsaturated regime.

  At task 0, enumerate **all** live candidates (5/12/9 by source), force only
  that placement, and let `peer_greedy_network` choose every later placement.
  Primary: sum of RTT over every task in the completed horizon. The fixed
  eight-task group's RTT and queue/exchange/rendezvous/wait are retained too.
  Every forced action runs through the same per-arrival scheduler, with no
  predetermined future placements and no batch wait. This is a conditional
  optimum over the first action under that continuation, not a global policy
  optimum or an eight-task placement sweep.

  **Information control.** At task 0 exactly one Task exists. A probe-scoped
  wrapper reads an explicitly declared eight-task manifest and forecasts future
  candidates from the CURRENT replica set. It uses current drain/cold/exec/latency
  and known peer transfers; it does not read future Task objects, queues or
  placements. No production serving/checkpoint contract was modified. Forecast
  candidates and costs match across rewired variants and all three horizons;
  two-hop root columns also match across variants. Controls choose only task 0;
  the continuation policy is common to all of them.

  **Checks and cost.** 406 simulator runs in about 26 seconds locally, no failures.
  Includes 36 unforced baselines, 312 forced-action runs and 58 exchange-off runs.
  Forcing the baseline's own action reproduces its RTT in all 36 comparisons.
  Per-task RTT sums and peer accounting reconcile. All six exchange-off cells
  charge exactly zero exchange and rendezvous, and the two rewired inputs become
  identical. All workloads finish; logs, task rows, input hashes and code hashes
  are retained. No GPU work or training.

  **Results at the 16-task horizon**, regret against the best forced first action
  under the common greedy continuation. These are 12 correlated variants over
  six paired cases, not 12 training draws or a population estimate:

  | First-choice control | Median RTT regret, % | Maximum, % | Cases ≥5%, /12 |
  |---|---:|---:|---:|
  | Immediate rule | 28.513 | 182.469 | 8 |
  | Hand peer mass | 7.937 | 137.705 | 6 |
  | Hand peer min-sum, 2/3/8 rounds (same choices) | 0.046 | 29.297 | 1 |
  | Full sharing + peer min-sum, 8 rounds | 0.109 | 29.297 | 1 |
  | Three-pass static CD | 0.046 | 137.705 | 2 |
  | Three-pass static multi-start CD | 0.046 | 29.297 | 1 |
  | Eight-task simulator replay search | 0.000 | 0.000 | 0 |

  Peer min-sum's two rounds are hand graph computation usable as engineered
  pointwise columns, not learned MP. Increasing their depth to 3 or 8 changes no
  chosen action on this design. The static controls use the same forecast as the
  paper-style score; their choices are evaluated through the actual simulator.

  **Horizon stability.** The eight-task best choice remains optimal at 12 and 16
  tasks in all **24/24** comparisons (12 variants × two longer horizons). Rank
  Spearman median 1.0, minimum 0.815. The replay-search control selects on the
  declared eight-task manifest ONLY, then is evaluated at the longer horizons;
  its zero eight-task regret is tautological and is not counted as confirmation.
  Its planning budget is roughly **0.47 s median / 0.76 s maximum** for all
  first-action simulations locally, excluding process import overhead. This is
  an offline replay-search control, not a deployed scheduler: simulation RTT does
  not charge this wall-clock planning cost. The tested suffixes are a narrow
  stability check, not protection against arbitrary workload changes.

  **Mechanism and tail.** Rewiring leaves a common best first candidate in **6/6
  pairs at every horizon**. At the 16-task conditional optimum, all **8/8 peer
  edges are co-located in every variant** (directly counted from executed node
  identities, not inferred from absent scheduler counters). Rendezvous is real:
  median baseline total is 6.786 s, not zero. Distant structure affects timing and
  regret but has not required a different optimal first choice.

  The two-hop control's one ≥5% loss is `ds_00064`, 1 s gaps, rewired cycle: its
  choice gives 24.718 s total RTT versus 19.117 s for the best action. Exchange is
  5.618 s versus zero; queue is 1.412 versus 1.440 s and rendezvous 15.064 versus
  15.054 s. This is a material forecast/control miss, not erased by the median;
  eight-task replay search recovers it. The same paired base cycle still prefers
  the same best action, so the tail is not evidence of an unavoidable two-hop
  information ambiguity.

  **Decision.** Useful first-action lookahead is present; deeper learned MP is
  not yet motivated over the engineered controls. Do not promote the regret
  percentages to whole-trace policy speedups. No comparison of trained models,
  long-run saturation screen, deployable snapshot rollout, or inference-overhead
  gate was performed. The lineage stays ACTIVE, not closed by these probes.

- 2026-09-21 — **Exploratory screen completed**, 24 source pairs, 48 variants,
  6,561 placements per variant (314,928 exchange-on plans plus the exchange-off
  controls). Final execution took about 4 seconds locally. Config, script,
  dependency and source hashes are in the JSON; the working tree was dirty.

  The actual live prefix builder reports one outside peer and **zero** peer
  edges/peer pairs for the singleton batch. Global peer metadata does not reach
  the current decoder under its existing contract.

  Both variants have a common optimal root candidate in **24/24 pairs**. This
  does not mean the entire optimal candidate sets must be identical. The
  matched-feature information-loss lower bound is zero in every pair, not just
  at the median. Optimal total cost changes in only **2/24 pairs**; **38/48**
  variants have an optimum with zero exchange. Thus the intended manipulation
  mostly leaves the decision-relevant optimum untouched on this substrate.

  | Paper control | Root regret mean, % | Full-plan regret median, % | Full-plan regret ≥5%, /48 |
  |---|---:|---:|---:|
  | Immediate greedy | 12.43 | 14.58 | 30 |
  | Sequential peer-mass greedy | 12.94 | 0.13 | 14 |
  | Peer min-sum, 2 rounds | 7.85 | 12.47 | 30 |
  | Peer min-sum, 3 rounds | 2.12 | 12.16 | 27 |
  | Peer min-sum, 8 rounds | 1.27 | 1.23 | 18 |
  | Sharing + peer min-sum, 8 rounds | 0.81 | 31.55 | 31 |
  | Three-pass CD, immediate start | 8.02 | 1.38 | 22 |
  | Three-pass CD, multiple starts | 0.17 | 0.00 | 1 |

  Every control's median root regret is zero; means and tails prevent that
  statistic from hiding failures. Multi-start CD still has one root-choice loss
  of **8.34%**. The min-sum controls choose candidates independently from their
  beliefs; the sharing-factor control's bad full plans demonstrate why a good
  root choice cannot be reported as a good joint policy. CD evaluates/refines
  joint plans against the same explicit paper objective.

  **Disclosed implementation refinement:** after the first exploratory read,
  the peer-mass control was strengthened from independent candidate argmins to
  sequential greedy with exact committed costs and expected remaining-peer
  costs. Its root choice was unchanged; its full-plan median improved from
  26.44% to 0.13%. Multi-start CD also benefits from that stronger start. The
  table is the final implementation; this was not a registered confirmation.

  **Scope of the negative:** one equal-weight cycle-rewiring construction, one
  payload, eight tasks, three candidates, uncapped frozen source costs. The 48
  variants are not independent training draws. No arrival dynamics were run,
  no learned arm was trained, and there is no powered equivalence finding.
  Future partial-arrival dynamics could change the answer. This screen supplies
  no reason to incur that training campaign yet; it does not close the lineage
  or establish that all downstream lookahead is pointwise-recoverable.
