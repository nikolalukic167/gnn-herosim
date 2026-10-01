# raw_plan_v1 — does the learned scorer still beat search without engineered plan context, and does message passing earn its keep there?

**Status:** `REGISTERED` (2026-09-30). Every read below was fixed before any data.

**Question.** [`peak_controls_v1`](peak_controls_v1.md) found the peak-load win is carried by a learned scorer over the
engineered `partial_state_v4` plan-context columns ψ, not by message passing. The MP-OFF twin ties the GNN and beats it
at ×5. ψ hands the scorer what matters: exchange seconds to placed partners, committed service, log backlog and route
co-use. Remove it, and the model must learn the plan's effect from raw facts. That is where a graph model should beat
a pointwise one.

**The raw plan** (`raw_plan_v1`, `src/policy/gnn/plan_raw.py`, `NEAR_RTT_PLAN_RAW` / sidecar `plan_raw`). Two columns
sit on every bipartite edge, in both directions: the committed flag, and the committed count on the platform. They feed
the bipartite conv and the scorer. Same-node platform edges join the conv with a type flag, so co-location is
learnable. There are no seconds, bytes or routes. The encode reruns for every distinct committed set.
`make_partial_state_score_fn` dispatches to it, so training, validation decode and live serving share one closure.
Node and edge features are unchanged and still include light normalisations, for example queue depth over its P90.

**Arms** (`backlog_corpus_v1` cache and split, xs1load's recipe otherwise, 4 seeds each, served with 3 self-refine
passes):
- `rawgnn`: peer conv plus the bipartite edge conv (a_tp zeroed, raw plan live) and typed same-node edges.
  `experiments/raw_plan_v1_rawgnn.yaml`.
- `rawmlp`: the same inputs, every message-passing layer skipped. It sees a partner's placement only when that
  partner lands on one of its own candidates, via the count column. `experiments/raw_plan_v1_rawmlp.yaml`.

**Cells:** the ×2 / ×3 / ×5 grounded cells of [`peak_load_v1`](peak_load_v1.md), phase `rawplan` (1,152 runs; run as `rawmlp` + `rawgnn`, one arm each). Every
other arm is read from `groundedladder`, `peakctl` and `peakmlp`.

**Reads, per rung** (`scripts_cosim/raw_plan_v1_read.py`). Labels follow `peak_load_v1`, with Holm over the three rungs
within each family.
1. **R1** — `rawgnn` vs `rawmlp`. `CONFIRMED` means message passing recovers plan context that the pointwise class
   cannot see.
2. **R2** — each vs CD and vs `cdextr`. A `CONFIRMED` result means a scorer without engineered context still beats
   search.
3. **R3** — `rawgnn` vs `xs1load` and `rawmlp` vs `xs1mpoff`: what ψ was worth to each class.

**Tests:** `tests/test_plan_raw.py` checks the following:
- the block marks both directions and the platform counts;
- a partner's placement moves the GNN's logits;
- the MP-OFF twin is blind to a partner placed elsewhere but reads its own count;
- a block of the wrong width is refused.

**Parents:** [`peak_controls_v1`](peak_controls_v1.md), [`exchange_seconds_v1`](exchange_seconds_v1.md),
[`backlog_corpus_v1`](backlog_corpus_v1.md).

## Record

### 2026-09-30 — registered

Code, configs, the train sbatch (`raw_plan_v1_train.sbatch`, with `SMOKE=1` for a 2-epoch pipeline check), the gate
phase and the reader were committed before any run. Local tests pass: `test_plan_raw` (5), and the serving, contract
and parity suites (129).
