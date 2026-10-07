# dag_memory_lifetime_s0_v1

> Produced on local-record-2026-10-06 at afea04bd; code not merged; old physics.

**Status:** `CLOSED` (2026-09-22) — CACHE-RETENTION RECIPE NOT QUALIFIED.
The corrected protocol was fixed before its fresh cases were generated.

**Outcome.** The new output-cache physics binds in all sixteen fresh
reference schedules and matches actual HeROsim operation timings, but its
finite two-start search improves the 250 ms memory-aware hand control by
only 6.32% median, below the registered 8% training bar. The hand search
captures 55.7% of the measured simple-rule-to-reference improvement, just
above the under-half target. The 49-run live gate and independent re-execution
audit pass. No GNN was trained. This closes the tested keep/spill contract and
retention-search recipe, not rematerialization or all memory-aware scheduling.

**Question.** Do value-specific DAG output lifetimes under a binding host-memory
limit leave material, search-hard improvement beyond memory-aware hand rules?
This is a new opt-in environment, not a GNN comparison. The prior block-pool
screen is [closed](dag_block_stepwise_s0_v1.md) and remains unchanged.

**Contract.** Every completed operation has a distinct 8–24 MB output backed
by durable storage. A chosen retained output occupies its execution host's
32 MB cache until its last consumer starts or capacity eviction removes it.
Retention decisions are per output. An input copied from the same host's cache
has zero read delay, a remote cached input charges 0.25 ms/MB, and a durable
store read charges 0.75 ms/MB. Input preparation occupies the child's host,
lock and domain before its execution. Capacity eviction is FIFO. Backing-store
write cost is included in the operation's declared base duration; this screen
adds no shared store-bandwidth contention. All arms know the whole workflow.
An opt-in actual HeROsim execution adapter and an independent SimPy replay must
agree on every operation completion and cache event count.

**Registered gate.** The [corrected protocol](../../experiments/dag_memory_lifetime_s0_v1_corrected.json)
fixes new calibration seeds 162300–162307 and fresh seeds 162400–162415,
eight jobs × eight operations × eight hosts, the 250 ms hand-search budget,
five simple retention controls, and a larger two-start 2,048-step feasible
reference. Advance only if reference plans show capacity binding on at least
half the fresh inputs, reference gain is at least 8% median over the selected
timed hand control, and hand search captures under half of the improvement
from the best simple rule to the feasible reference. A passing screen would
still need a stronger placement/order control, a residual graph-information
test after 1/2/4-hop hand features, and matched learned comparisons before a
GNN claim. No GPU is allocated at this stage.

This first contract tests keep/spill. Recomputation requires its own operation
and resource accounting and is deliberately a later extension if this screen
qualifies; no result here can reject rematerialization as a mechanism.

**Entry points:** [physics and replay](../../src/placement/radical/dag_memory.py),
[actual execution](../../src/placement/radical/dag_memory_live.py),
[screen](../../scripts_cosim/dag_memory_lifetime_s0.py),
[auditor](../../scripts_cosim/audit_dag_memory_lifetime_s0.py).

**2026-09-22 timing amendment before fresh inputs.** The original calibration
on seeds 162100–162107 failed loudly: both search modes exceeded 250 ms, and
their timer excluded baseline and simple-rule preparation. No fresh case was
generated. The failed artifacts and source snapshot remain under the original
artifact directory. The corrected protocol charges all preparation, adds a
search reserve, and uses unused calibration and fresh seeds. Only that
corrected run may qualify the gate.

## 2026-09-22 — Corrected output-memory screen and live gate: NO-GO

Eight calibration inputs selected full-path greedy bit-flip search over a
time-matched annealing control; both include the common no-memory placement
and rank initialization plus evaluation of all five simple retention rules.
On all sixteen fresh inputs the selected hand path fits 250 ms, with a
maximum of **234.11 ms** and median preparation time **16.54 ms**. No
post-hoc mode choice is credited to it. The slow reference takes the best
feasible result of 2,048-step annealing from both the best simple rule and
the timed hand result. It is not an optimality certificate.

The median reference gain over hand is **6.319%**, with a descriptive
environment-bootstrap interval **[5.020%, 7.011%]**; reference is strictly
better on 16/16 inputs. Median hand capture of the simple-rule-to-reference
gain is **0.5567**, interval **[0.3946, 0.6825]**. Thus the registered 8%
headroom bar fails, and the registered under-half search-hardness bar also
fails by point estimate. The latter is near its boundary and uncertain;
the headroom failure is decisive for this recipe.

The mechanism itself is real within its declared contract. Every reference
plan has at least one capacity-binding completion, with a median of **five**
such completions and **five** evictions. The best simple retention rule is
`keep_all` on 14/16 inputs and `reuse` on two. Turning both read charges
off with ample memory reproduces the historical DAG schedule exactly in a
regression test; the default opt-in adapter does not alter historical physics.

The registered live gate executes one calibration plan and three plans on
each fresh input: **49 actual HeROsim runs and 3,136 operations**. Every
operation completion and the cache/read/eviction counts agree with the
independent event replay. The separate artifact audit passes all **24**
physical input identities, saved plan costs, source and native fingerprints,
live records, and re-executes all **65,536** reference search proposals.
Authoritative results and `AUDIT.json` live under
`simulation_data/gnn_environment_search_v1/dag_memory_lifetime_s0_v1_corrected/`.

**Decision.** Do not generate a training corpus or allocate GPU for this
fixed cache-retention recipe. No residual graph-information claim was
tested after the failed pretraining bars. Recomputation remains open because
this contract provides durable reads only; adding it requires explicit
resource-accounted recompute events and a new live gate. More extensive
joint placement/order and cache search could outperform this finite
reference, so the 6.32% is not a bound on the environment.
