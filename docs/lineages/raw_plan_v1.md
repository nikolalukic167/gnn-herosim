# raw_plan_v1 — does the learned scorer still beat search without engineered plan context, and does message passing earn its keep there?

**Status:** `CLOSED` (2026-10-01) — **`MP-BEATS-TWIN / CD-FASTER`**. Registered 2026-09-30; every read below was
fixed before any data.

**Outcome** (12 topologies × 4 windows × 4 seeds per rung; median over topologies of the per-topology median paired
%; exact Wilcoxon; Holm over the three rungs; read `raw_plan_v1/raw_plan_read.json`):
- **R1 `CONFIRMED` at every rung.** Without engineered plan context, `rawgnn` beats its MP-OFF twin `rawmlp`
  **−13.9 / −24.3 / −11.8 %** at ×2 / ×3 / ×5, 12/12 topologies each, Holm p = .0015. This is the program's first
  live message-passing win over a twin trained identically — but on inputs from which the pointwise class is
  deliberately blind to the partner plan.
- **R2 `REF-FASTER` at every rung.** Both raw arms lose to CD (`rawgnn` +26.1 / +53.0 / +23.9 %, 0/12) and to
  `cdextr` (+31.7 / +72.5 / +30.1 %).
- **R3:** engineered context ψ is worth +40.5 / +83.8 / +32.4 % to the GNN and +63 / +157 / +65 % to the twin.
  The deficit is queue, not exchange (×2: queue 25.3 s vs 13.4 s for `xs1load` and 20.6 s for CD; exchange
  3.60 vs 3.02 / 3.24 s).
- **Do not quote** R1 as "the GNN beats the MLP" without "raw plan, both arms lose to CD". Training was stopped by
  hand at epoch 110–115 of 300 after a plateau; every selected checkpoint (and every `rawmlp` one) lies within the
  first 100 epochs. Successor: [`raw_plan_v2`](raw_plan_v2.md).

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

### 2026-10-01 — gate read: `MP-BEATS-TWIN / CD-FASTER`

**Training.** `rawmlp` (job 820944) ran to patience: last improvement at epochs 93 / 44 / 92 / 30, best val regret
30.8–31.2 s. `rawgnn` (job 820943) was cancelled by hand at epoch 110–115 of 300 (user request, after a plateau:
≤ 0.4 s gain between epochs ~45 and ~110). Its best checkpoints, saved with sidecars on every improvement, are from
epochs 80 / 70 / 95 / 90 (val regret 19.10 / 19.43 / 19.47 / 18.78 s), md5-identical before and after the cancel.
The post-train checks (`joint_burst_v2_sidecheck.py` plus the plan_raw sidecar/weight asserts of
`raw_plan_v1_train.sbatch`) were run by hand and passed on all four. W&B shows the four runs as killed.

**Gate.** Phases `rawmlp` and `rawgnn` (job 821601), 576/576 runs each, driver rc 0, no failures. Read
`scripts_cosim/raw_plan_v1_read.py --raw rawgnn rawmlp` → `raw_plan_v1/raw_plan_read.json`.

| Contrast | ×2 | ×3 | ×5 |
|---|---|---|---|
| rawgnn vs rawmlp | −13.88 % (12/12) ✓ | −24.27 % (12/12) ✓ | −11.75 % (12/12) ✓ |
| rawgnn vs CD | +26.05 % (0/12) | +53.02 % (0/12) | +23.86 % (0/12) |
| rawgnn vs cdextr | +31.70 % (0/12) | +72.54 % (1/12) | +30.08 % (0/12) |
| rawmlp vs CD | +44.17 % | +93.29 % | +39.80 % |
| rawgnn vs xs1load | +40.52 % | +83.79 % | +32.41 % |
| rawmlp vs xs1mpoff | +62.90 % | +156.67 % | +64.78 % |
| rawgnn vs reactive | −12.92 % (n.s.) | −27.88 % (Holm .054) | −7.52 % ✓ |
| rawgnn vs decima | −6.20 % ✓ | −1.77 % | +6.87 % |

Mean latency / queue (s), ×2: rawgnn 29.3 / 25.3, rawmlp 33.2 / 28.6, xs1load 16.8 / 13.4, CD 24.2 / 20.6, cdextr
19.7 / 16.1. The raw GNN's loss is queue: it co-locates nearly as well as CD (exchange 3.60 vs 3.24 s per task)
but does not price committed load. Code read (not shown): the bipartite conv mean-aggregates, and the recipe
inherits `NEAR_RTT_MP_BIPARTITE_EDGE_ATTR_ZERO=1`, so the conv never sees exec time — the hypotheses
`raw_plan_v2` tests.

### 2026-09-30 — registered

Code, configs, the train sbatch (`raw_plan_v1_train.sbatch`, with `SMOKE=1` for a 2-epoch pipeline check), the gate
phase and the reader were committed before any run. Local tests pass: `test_plan_raw` (5), and the serving, contract
and parity suites (129).
