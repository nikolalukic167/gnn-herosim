# dag_memory_value_learning_v1

> Produced on local-record-2026-10-06 at afea04bd; code not merged; old physics.

**Status:** `CLOSED` (2026-09-22) — NO GNN-SPECIFIC WIN. Registered
2026-09-22 before data generation; the user explicitly authorized this pilot
despite its parent screen's failed funding bars.

**Outcome.** Six CPU checkpoints trained with offline W&B on 64 workflows,
selected on 16 disjoint workflows, then faced 16 sealed workflows in 144
actual-HeROsim runs. The GNN's median paired gain over MP-OFF is +0.07%
(environment-bootstrap 95% interval −3.38% to +0.61%), and it trails the
250 ms hand search by 2.75% median. The graph-conditioned ranking has no
supported advantage over learned controls or hand search. This closes the
fixed keep/spill value-ranking recipe, not rematerialization or other memory
physics.

**Question.** Can message passing rank exact-scored keep/spill changes on the
binding-memory DAG environment better than separately trained MP-OFF, a
hand-feature MLP and a 250 ms memory-aware hand search? The user explicitly
requested training despite the [parent screen](dag_memory_lifetime_s0_v1.md)
failing its 8% funding and under-half search-hardness bars. This exception
applies to this pilot and does not change the parent's measurement.

The [protocol](../../experiments/dag_memory_value_learning_v1_protocol.json)
fixes 64 training, 16 validation and 16 sealed fresh workflow seeds before
generation. Exact single-bit outcome values are measured at visited hand
retention states. Every learned proposal is replayed and accepted only if it
strictly improves summed job completion. All arms receive the same base
features and affordable fixed 1/2/4-hop graph summaries. The matched GNN
alone propagates learned messages; MP-OFF disables those messages, and a
separate MLP uses the same hand-graph features. Two draws per learned arm
train through `run_experiment.py` with offline W&B. Validation selects by
complete served objective including epoch zero. Full planning time, from
hand initialization through features, model and exact scoring, counts toward
250 ms. A full live HeROsim gate runs regardless of validation direction.

A GNN-specific claim requires a positive paired interval against all matched
learned controls and the timed hand search. The two-draw/16-environment pilot
can diagnose the mechanism but is not a powered definitive comparison.

## Record

- [2026-09-22 — matched training and sealed live gate](#2026-09-22--matched-training-and-sealed-live-gate)

### 2026-09-22 — matched training and sealed live gate

**Verdict: NO GNN-SPECIFIC WIN.** The [corpus](../../simulation_data/gnn_environment_search_v1/dag_memory_value_corpus_v1/METADATA.json)
contains 128 labelled training states from 64 workflows and 32 validation
states from 16 workflows; its 16 test workflows were recorded by identity
only, without computing test plans or costs. Each state has exact full-cost
labels for all valid one-bit keep/spill changes. The train/validation/test
environment identities are disjoint. All six draws were trained through
`run_experiment.py`, logged to offline W&B, and selected by complete served
validation objective including epoch zero. Selected epochs were 2/2 for GNN,
1/1 for MP-OFF and 1/2 for hand-feature MLP. Checkpoints and contracts are
under `models/dag_memory_value_learning_v1/`.

The [sealed gate](../../simulation_data/gnn_environment_search_v1/dag_memory_value_live_v1/read.json)
ran each checkpoint, the timed memory-aware hand search, best simple rule
and a two-start 2048-step feasible reference on 16 fresh 8-job × 8-operation
× 8-host workflows. All 144 plans were replayed in actual HeROsim: 9,216
operation completions and memory-event counts matched the independent cost
evaluator. Every learned and hand decision completed within 250 ms. Memory
bound on all 16 reference schedules. Mean summed job completion times (ms)
were GNN 3230.59, MP-OFF 3164.45, hand-feature MLP 3150.98, timed hand
3130.25, and slow reference 2940.63. The reference's median gain over hand
was 4.64% (bootstrap interval 2.87%–8.05%).

For each physical workflow, the two matched training draws were averaged
before the median and bootstrap over 16 workflows; draws are not treated as
independent replicates. Median GNN gain over MP-OFF was +0.07% (95% interval
−3.38% to +0.61%), versus hand-feature MLP −0.50% (−5.08% to +0.95%), and
versus timed hand −2.75% (−5.62% to 0.00%). Only 6 of 32 GNN draws beat the
timed hand on their paired workflow. GNN scores and strictly accepts exact
improvements, so this failure is in move ranking under the common budget,
not an unsafe deployment regression. The 16-workflow, two-draw pilot is
exploratory rather than a powered proof of equivalence; it supplies no
positive GNN-specific evidence. The parent S0's 8% oracle-headroom and
under-half hand-capture bars remain failed, consistent with this live result.

This closes training on this fixed FIFO keep/spill recipe. A new memory
lineage needs a changed mechanism, such as rematerialization, and fresh
pretraining headroom and hardness gates; repeating these six models or
scaling their corpus is not supported by this result.
