# dag_block_stepwise_s0_v1

> Produced on local-record-2026-10-06 at afea04bd; code not merged; old physics.

**Status:** `CLOSED` (2026-09-22) — FINITE BLOCK-POOL NO-GO. The
corrected protocol was fixed before its fresh cases were generated.

**Outcome.** At 384 operations, the best feasible reference improves the
250 ms hand search by 3.05% median on sixteen fresh workflows, below the
registered 8% training-funding bar. Hand search captures 50.3% of the
measured baseline-to-reference improvement, just over the under-half bar.
The broad block pool beats that hand search on zero of sixteen workflows.
All 52 HeROsim replays and the independent artifact audit pass. No GNN was
trained; this closes the finite block-pool and fixed proposal order, not
online stepwise reservations or all graph-conditioned search.

**Question.** On irregular mixed-resource DAGs, can exact-scored critical-chain,
lock-chain, setup-class and block-swap proposals expose material improvement
unrecovered by a matched 250 ms compiled hand search? This is a feasibility and
headroom experiment, not a trained GNN comparison.

**Parent:** [mixed_dispatch_v1](mixed_dispatch_v1.md). The previous guarded
one-shot learner is [dag_reservation_learning_v1](dag_reservation_learning_v1.md).
The [corrected protocol](../../experiments/dag_block_stepwise_s0_v1_corrected.json) fixes shapes,
unused seed ranges, a 250 ms primary budget, an 8% reference-headroom screen and
an under-half hand-capture screen before any qualification result is read.

The four scale rungs have 48, 128, 256 and 384 operations. Native full-plan
costs score each candidate. The broad reference includes the best of one-step
and repeated block moves plus three 8,192-step compiled searches; it is feasible,
not an optimum. The timing-matched controls include the established mixed-decoder
hand search and a fixed-order block proposer. Actual HeROsim replay checks one
case at every scale and all fresh hand, block and reference schedules. Passing
both screens would permit a separate residual-information test before any
training. A failure limits only this finite move recipe and proposal order.

**Entry points:** [move pool](../../src/placement/radical/dag_block_moves.py),
[runner](../../scripts_cosim/dag_block_stepwise_s0.py),
[auditor](../../scripts_cosim/audit_dag_block_stepwise_s0.py). Corrected artifacts
live under `simulation_data/gnn_environment_search_v1/dag_block_stepwise_s0_v1_corrected/`.

**2026-09-22 amendment before corrected fresh data.** The first partial run
stopped after six fresh cases: it passed non-delay priorities directly into
the serial reservation decoder before applying moves. That changes the
incumbent schedule even with no move, invalidating its move-cost comparison.
Its partial artifacts and source snapshots remain under the original artifact
directory as a defect record; none of those cases qualifies a result. The
[corrected protocol](../../experiments/dag_block_stepwise_s0_v1_corrected.json)
uses unused calibration seeds 161200–161201 and fresh seeds 161300–161315.
It requires chronological rank translation and a cost-preservation assertion
before generating reservation moves. The corrected output has its own directory.

## 2026-09-22 — Corrected CPU screen and live gate: NO-GO

The 48/128/256/384-operation calibration completed with 24-host,
384-operation workflows selected for the fresh qualification. The broad
candidate pool takes a median 0.59 s at that size, so its best answer is a
slow feasible reference rather than a budget-compliant control. The common
initializer takes about 76 ms. Both timed paths stay within 250 ms on every
fresh case: maximum hand 243.61 ms and block proposer 243.31 ms.

Sixteen new physical workflows, seeds 161300–161315, compare the common
initializer, compiled mixed-decoder hand search, fixed-order exact-scored
blocks, repeated block descent and three 8,192-step searches. The median
reference gain over hand is **3.053%**, with a descriptive environment
bootstrap interval **[1.837%, 4.279%]**. The median hand capture of the
baseline-to-reference gain is **0.5029**, interval **[0.3673, 0.6566]**.
Both registered gates fail by their point estimates. The capture estimate is
near the boundary and uncertain; headroom failure is the decisive stop.
The reference is a finite feasible search, not an optimality certificate.

Across the fresh cases, the broad pool contains **6,656** exact-scored
proposals; **20** strictly improve their own common initializer. A broad
block move improves that initializer on 8/16 inputs but beats the timed hand
control on **0/16**. Its median gain over hand is **0.0%**. Three rounds of
block descent and the slow searches are visible in every retained case;
none of the candidate costs is treated as a free served decision.

Actual HeROsim replay covers one calibration case at each scale and the
hand, broad-block and reference plans on every fresh case: **52 runs and
19,248 operations**. The independent audit passes source/input fingerprints,
all **9,941** calibration and fresh candidate costs, native/independent
replay and every live plan. The corrected report and `AUDIT.json` are in
the corrected artifact directory. The focused DAG/block suite passes, and
record hygiene is checked after the record edit.

**Decision.** Do not train a GNN or allocate GPU to this fixed proposal pool.
The result leaves genuinely online stepwise reservation, better move
generation, and value-specific memory lifetimes untested. The earlier
partial run's decoder mismatch is governed by the existing
[gate-tool rule](../gates/gate-tools.md#2026-09-22--translate-schedule-semantics-between-search-decoders).
