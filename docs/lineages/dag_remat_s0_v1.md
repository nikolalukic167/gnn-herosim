# dag_remat_s0_v1

> Produced on local-record-2026-10-06 at afea04bd; code not merged; old physics.

**Status:** `CLOSED` (2026-09-22) — ONE-LEVEL REMATERIALIZATION NO-GO.
Registered 2026-09-22; the corrected protocol used untouched seeds after
the first gate's lock-accounting defect was found.

**Outcome.** On 16 corrected fresh workflows, the feasible two-start
reference improves a 250 ms memory-aware hand search by 5.90% median
(descriptive bootstrap interval 3.01%–8.21%), short of the registered 8%
bar. Hand search captures 61.7% of the best-simple-to-reference gap versus
an under-half target. Cache capacity binds in every reference plan, but
actual recomputation occurs in only 2/16. All 49 actual-HeROsim runs and
the independent read-back audit pass. This closes the tested per-output,
one-level inline recomputation recipe, not all rematerialization designs.

**Question.** Does resource-accounted DAG output rematerialization create
material planning headroom and search hardness beyond a 250 ms memory-aware
hand search? This follows the [closed keep/spill learner](dag_memory_value_learning_v1.md),
whose negative result does not test recomputation.

The [corrected protocol](../../experiments/dag_remat_s0_v1_corrected.json) fixes calibration and
fresh seeds before their evaluation. The new opt-in contract retains the
prior 32 MB host cache, FIFO eviction, durable reads and peer reads. When a
direct parent output is absent, a binary choice may recompute that parent on
the consumer's host. It reads the parent's direct inputs from existing cache
or durable storage, executes for the parent's declared host duration, and
occupies the consumer host and union of parent/consumer locks. This is
one-level inline recomputation: it neither recursively recomputes ancestors
nor caches the new copy. Ineligible host computations are rejected. Existing
keep/spill sources and their fingerprinted results remain untouched.

The screen first chooses between budgeted greedy and annealing hand searches
on eight calibration inputs. Sixteen disjoint fresh inputs then compare the
selected 250 ms hand control with a two-start, 2,048-step-per-start feasible
reference. Advance to representation testing only if the reference gains at
least 8% median over hand, hand captures under half the best-simple-to-reference
gain, and memory binding and actual recomputation both appear in at least half
the reference schedules. The full live gate executes regardless of those
bars. A finite reference is not an optimality certificate, and no GNN claim
or GPU allocation comes from this screen alone.

**Entry points:** [physics](../../src/placement/radical/dag_remat.py),
[actual execution](../../src/placement/radical/dag_remat_live.py),
[protocol](../../experiments/dag_remat_s0_v1_corrected.json).

**2026-09-22 gate correction before corrected fresh inputs.** The original
164100/164200 gate completed, but a targeted ablation found that a recompute
bit could change lock scheduling when its output was cached and no recompute
work happened. Five reference plans showed apparent marginal recompute
benefit with zero recompute events. That violates the declared contract, so
the original 49 live runs and all objective numbers are **invalid for
qualification**. The original artifact and protocol are retained as a
diagnostic. Replay and actual HeROsim now acquire the parent's lock only
when its output is absent at dispatch. A regression test checks that a
cached output cannot acquire a phantom recompute lock. The corrected protocol
uses untouched calibration seeds 164300–164307 and fresh seeds 164400–164415
with the same bars and live gate.

## Record

- [2026-09-22 — corrected one-level rematerialization screen and live gate: NO-GO](#2026-09-22--corrected-one-level-rematerialization-screen-and-live-gate-no-go)

### 2026-09-22 — corrected one-level rematerialization screen and live gate: NO-GO

Eight calibration cases selected budgeted greedy search over annealing.
The complete path includes baseline placement/rank construction, eight
simple-rule evaluations, and search; all sixteen fresh hand decisions fit
250 ms (maximum 234.69 ms). The slow feasible reference takes two 2,048-step
annealing starts, one from the best simple plan and one from timed hand.
Its median paired gain over hand is **5.901%**, with an environment-bootstrap
95% interval **[3.011%, 8.209%]**. Median hand capture of the simple-to-reference
gap is **0.6165**, interval **[0.3753, 0.7284]**. The point estimates fail
both pretraining bars. The reference is finite and does not certify an optimum.

Capacity binds in **16/16** reference plans, but recomputation actually runs
in only **2/16** (one event each), below the registered half-case mechanism
bar. Only one reference plan improves when its recompute bits are kept
versus setting all recompute bits to zero at fixed retention; the other
fifteen have zero marginal recomputation gain. Only about **16%** of consumed
outputs are eligible for a per-output recompute action at every consumer
host, because host execution support is sparse. This is an action-contract
bottleneck, not evidence that all recomputation is useless. Mean simple,
timed-hand and reference summed job completion times are respectively
3407.88, 3110.34 and 2928.70 ms.

The [corrected result](../../simulation_data/gnn_environment_search_v1/dag_remat_s0_v1_corrected/read.json)
contains 16 fresh physical identities, complete plans, timings and source
fingerprints. Its live gate ran one calibration plan and three plans per
fresh workflow: **49 actual-HeROsim runs and 3,136 operations**. Every
operation completion and cache, read, eviction and rematerialization count
matched the independent event replay. The separate
[audit](../../simulation_data/gnn_environment_search_v1/dag_remat_s0_v1_corrected/AUDIT.json)
read back all live records, checked all source and input identities, and
verified 13 no-op recompute plans cannot alter objective through phantom
locks. The original artifact remains invalid and is not pooled with this
result.

**Decision.** Do not train a GNN or allocate GPU for this one-level,
per-output action contract. A successor worth screening would make
recomputation a per-consumer-edge choice and schedule it as explicit work,
so an output can be recomputed on one eligible consumer host without
requiring every consumer host to support the same choice. It still needs
fresh headroom, equal-budget hand-search, mechanism-use, graph-residual
and actual-HeROsim gates before a learned-model claim.
