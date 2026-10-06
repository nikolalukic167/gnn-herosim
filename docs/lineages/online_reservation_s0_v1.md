# online_reservation_s0_v1 — hidden-arrival reservation screen

**Status:** `CLOSED` (2026-09-23) — TWO-DECISION-ONLINE-RECIPE-NO-GO.
The primary protocol was fixed before qualification; the one bounded shortlist
retry was specified after the primary result and before disjoint retry seeds.

**Outcome.** An online event-level planner sees the DAGs of three active
workflows and learns of a fourth only at its arrival. On sixteen fresh
workflows, a two-decision reservation beam gains **0% median** over a hand
portfolio with the same state, continuation rules, exact evaluator and
32-evaluation ceiling; it wins 5/16. A focused-shortlist retry on sixteen
disjoint workflows again gains **0% median** and wins 2/16. Both fail the
registered 3% and 12/16 advancement bars. The beam uses more evaluations
and about twice the planning time. Actual HeROsim executes all 96 materialized
plans with matching task and resource events. No GNN was trained; live replay
does not establish a live online callback or a general limit on online learning.

**Parent:** [mixed_dispatch_v1](mixed_dispatch_v1.md).
**Protocol:** [online_reservation_s0_v1.json](../../experiments/online_reservation_s0_v1.json).
**Implementation:** [screen](../../scripts_cosim/online_reservation_screen.py),
[staggered-arrival adapter](../../src/placement/radical/dag_reservation_live.py).
**Artifacts:** [primary](../../simulation_data/gnn_environment_search_v1/online_reservation_s0_v1/report.json),
[retry](../../simulation_data/gnn_environment_search_v1/online_reservation_s0_v1/retry_report.json).

## Record

- [2026-09-23 — Fresh screen and bounded retry](#2026-09-23--fresh-screen-and-bounded-retry)

## 2026-09-23 — Fresh screen and bounded retry

The event simulator uses four irregular four-operation DAGs on three hosts,
with host setup, atomic locks and domain capacity. Three workflows arrive at
zero. The fourth arrives at 4–12 ms and is excluded from both policies'
visible state and rollout objective until arrival. Both policies may start a
ready operation on an eligible host or wait for the next event. They replan
after execution events and arrivals. The outcome is the sum of all four job
completion times. A regression changes the hidden workflow's durations and
locks and confirms that neither policy's initial action changes.

The hand portfolio scores each feasible first action under four complete
visible-workflow continuations: short job, critical path, setup-aware and lock
pressure. The primary beam scores up to sixteen first-action continuations,
then spends the rest of its 32-score cap on a second decision from four
shortlisted first actions. Both arms use the same evaluator and state; the
beam's extra depth does not imply extra information. The local short-job rule
is a weak context control, not the advancement bar. Both policies can wait.

| Fresh phase | Seeds | Beam median gain vs hand | Beam wins / losses / ties | Hand portfolio gain vs local rule | Median hand / beam planning time |
|---|---:|---:|---:|---:|---:|
| Primary | 275000–275015 | 0.00% | 5 / 4 / 7 | 19.38% | 594 / 1269 ms |
| Shortlist retry | 276000–276015 | 0.00% | 2 / 7 / 7 | 17.16% | 520 / 1165 ms |

Paired-workflow bootstrap 95% intervals for the median beam gain are
[−0.27%, 4.58%] in the primary phase and [−2.26%, 0.00%] in the retry
(20,000 resamples, fixed seed 9421). These small samples cannot rule out
every modest advantage, but neither phase meets its predeclared bars.

The retry chooses up to four diverse first actions from the hand rules,
including wait where feasible, and uses the same 32-score cap. Its design was
fixed before the retry seeds. The 3% median and 12/16 win bars fail in both
phases. The reported planning times are Python diagnostic times, not an
optimized deployment estimate; the beam consumes roughly 780 exact rollouts
per workflow versus roughly 350 for the hand portfolio despite sharing the
same per-decision ceiling. Thus a tie in objective is especially unpromising.

All three arms on every fresh workflow were materialized from their online
decisions into final starts, hosts and reservations. Actual HeROsim replayed
them with the staggered application arrivals. The audit matches each task's
completion and placement and every execution start, completion, host and setup
event: **96 live runs, 1,536 operations and 3,072 resource events**. This
validates execution physics of the realized schedules. It does not validate
an online scheduler callback, because the live adapter executes the final
materialized decisions. The screen also does not model transfer costs, queue
measurement noise, or unknown DAG structure inside an arrived workflow.

**Decision.** Stop these one/two-decision, four-rule online reservation
searches and do not train a GNN on their action labels. The strong hand
portfolio's gain over local dispatch shows that future-aware decisions matter
in this small environment, but deeper search did not add robust headroom over
an equally informed hand planner. A later online learner needs a new source
of residual decision value and a real callback gate before a GNN claim.
