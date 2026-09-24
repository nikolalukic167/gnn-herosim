# pointwise_baseline_v1 — is the pointwise baseline this programme keeps quoting actually the MLP?

**Status:** `PARKED` (2026-09-19) — **`MLP-CANNOT-BE-SERVED-UNDER-PARTIAL-STATE`.** The bars
are signed and the arms are trained; the lineage is parked on a missing serving
capability, found by a one-arm smoke test before any gate arm ran. Every measured result below
stands; resuming needs the serving path built and its parity proved, not just a re-run.

**The tabular MLP has never been served live under a partial-state representation in this
programme.** The MLP serving path scores every edge once per batch, and the partial-state block
is a function of the committed prefix, so one-shot scoring structurally cannot carry it — it
never has, for any contract. The first `mlp_batch` arm completed normally and read
`fallback_decisions: 50000 / 50000`: shortest-queue on every task, wearing the MLP's name.
**This also explains the substitution the lineage was registered to question — `mpoff` was the
only pointwise arm that could be served at all.** Full account, and the four defects fixed on
the way, in the Record.

Every bar is a module constant in `scripts_cosim/pointwise_baseline_v1_read.py`, committed with
its 12 tests before the reader was pointed at a results directory. **Nothing is read and no bar
is relaxed**; J0–J4 stand exactly as signed.

**The question, in one sentence: every graph-vs-pointwise number in this programme compares
the GNN against its own message-passing-disabled twin, not against a pointwise model.**

CLAUDE.md's own framing says:

> Knative is the industry-standard reactive baseline, and the MLP is the pointwise control.
> **The MLP is a control, not a straw man** — the program's repeated finding is that it ties,
> so treat "the MLP cannot match this" as a hypothesis to test, never an assumption to write
> from.

and `best_arm_v1` had to close carrying:

> **`mpoff_516` is the MP-OFF twin of a GNN, not the MLP.** The pointwise baseline this clause
> names has never been the tabular MLP.

`mpoff` shares the GNN's encoder, scorer and decoder with message passing switched off. That
is an **ablation**, not a model class. The tabular MLP has never been served on these cells.

**Why this is worth a lineage and not a footnote.** The chain
`peer_only_v1` → `bipartite_edge_v1` → `bipartite_aggr_v1` → `best_arm_v1` →
`corpus_matched_v1` has produced increasingly sharp statements about "graph versus pointwise",
and **all of them rest on the substitution.** If the twin and the MLP agree, the whole
back-catalogue stands as written and gains a real baseline. If they disagree, every one of
those statements is a statement about an ablation and must be restated as one. Either outcome
is worth more than another rung of any existing ladder.

## What makes the comparison fair

**The serving path is shared — and this is where the lineage blocked.** `MLPBatchScheduler`
inherits graph build, decode and roll-forward from `GNNScheduler` (via
`XGBoostBatchScheduler`) and replaces only the scoring, which is what makes the contrast
readable at all: this programme has been burned by serving-layout confounds twice
(`herosim-inference-layout-confound`, `herosim-live-quality-is-a-training-draw-lottery`).

**But it scores every edge ONCE per batch**, and the partial-state block changes at every
decode step, so that path cannot serve a partial-state arm — see the Record. The route through
is `decode_prefix_conditioned`, which is model-agnostic (it takes
`score_fn(task_idx, committed)`) and **refuses to fall back**; what is missing is an MLP
counterpart to `make_partial_state_score_fn`, plus the offline/live parity proof that any new
serving path in this record has to pass before its numbers are believed.

**The corpus is matched, at both levels.** `corpus_matched_v1` closed
`MODEL-CLASS-EDGE-IS-CORPUS-CONTINGENT`, so a single-corpus answer here would be a single-level
answer. The MLP is trained on **both** the 516 and the 1,670 psv3 caches, against the graph and
twin arms already served on each.

**The representation is matched.** The MLP reads the same `partial_state_v3` block the graph
arms read. That required wiring: `_batch_edge_feature_dims` refused v3 outright, on the stated
grounds that *"the MLP is not an arm of that lineage"* — organisational, not physical, since
`partial_state_columns` has always been contract-aware. The new layout is **`dim47crk`**
(dim25cr 25 + the v3 block 22), a **separate name**, never a widened `dim63crk`: a checkpoint
declares `inference_feature_layout`, so a distinct name is the only thing stopping v2 weights
from being served v3 features. v1/v2 stay byte-identical. Pinned in
`tests/test_dim47crk_layout.py`.

**The learning rate is selected, not inherited.** `literature_reeval_v1` found a published
method's reported tie was caused by the authors' own recipe under-training it. Handing this
programme's baseline the GNN's lr untested would repeat that error **in the direction that
flatters the GNN**. Three learning rates (5e-4, 1e-3, 2e-3) are trained per corpus and one is
selected on held-out **offline** score; the live gate only ever sees the winner.

## Registered reads

All at the three **unsaturated** client rungs used by F and G — 20 (which **is** the R0 server
cell, 6 servers / 20 clients), 40, 80 — at **both** corpora. Bars reused **unchanged**:
**|median| ≥ 5 %, two-sided Wilcoxon p < 0.05, n ≥ 16 checkpoints**, unit = checkpoint
(`collapse_to_seed`), negative = the first-named arm faster.

- **J0** — learning-rate selection, per corpus, on held-out **offline** score at 3 seeds.
  **No live arm runs before this resolves.** Recorded whichever way it goes, including if the
  GNN's own 2e-3 wins.
- **J1** — graph arm vs **MLP**, per rung, per corpus, corpus and representation held fixed.
- **J2** — the composite per corpus, under F2/G2's rule **unchanged**: two wins and no losses.
- **J3** — **`mpoff` vs MLP**, same corpus, per rung. **The load-bearing read.**
- **J4** — MLP vs reactive Knative, descriptive context only, never a bar.

## Consequences, signed in advance

1. **J3 = `MP-OFF-TWIN-IS-NOT-THE-MLP`** (separation at ≥ 1 rung) ⇒ the substitution is not
   sound. **Every earlier "pointwise" claim in this record — including `best_arm_v1`'s F2 and
   CLAUDE.md's standing pointwise clause — is a claim about the GNN's own ablation and is
   restated as one.** This is the outcome that would most change the record.
2. **J3 = `MP-OFF-TWIN-STANDS-IN-FOR-THE-MLP`** ⇒ the substitution was sound **on these
   cells**, the back-catalogue stands as written, and it finally has the baseline its framing
   always claimed. A non-separation is **not** proof of equality and is never a general licence
   to substitute.
3. **J2 = `GRAPH-BEATS-THE-POINTWISE-MODEL-CLASS`** at a corpus ⇒ the programme has, for the
   first time, a graph-beats-pointwise result against a real model class on a matched corpus
   and a shared decoder. It would be **scoped to that corpus** — `corpus_matched_v1` showed a
   matched verdict can itself be corpus-contingent.
4. **J2 = `POINTWISE-MODEL-CLASS-BEATS-GRAPH`** ⇒ CLAUDE.md's pointwise clause strengthens from
   an arm statement into a model-class one, and route 1's `program_verdict_v1` closure ("the
   MLP is the correctly specified model class") gains a live confirmation it never had.
5. **J2 disagrees between the two corpora** ⇒ recorded as a scope limit, exactly as G4 was;
   name the corpus in every quote.

## Registered expectations

Scored after the read; being wrong is recorded, not rewritten.

- **J3: UNCERTAIN, leaning `MP-OFF-TWIN-STANDS-IN-FOR-THE-MLP`.** The record's repeated finding
  is that pointwise arms tie one another (`program_verdict_v1`: the co-sim target is
  pointwise-separable, so both are correctly specified). But the twin keeps the GNN's *decoder
  and scorer*, which `herosim-mp-is-not-what-makes-the-gnn-reliable` credits with the arm's
  reliability — so a twin advantage would not be surprising either.
- **J1 at 40 and 80 clients: `GRAPH-FASTER-THAN-MLP`**, matching G1's shape at 1,670.
- **J1 at 20 clients: `NOT-SEPARATED`** — every arm loses to reactive there by +69 to +96 %.
- **J2 at 1,670: `GRAPH-BEATS-THE-POINTWISE-MODEL-CLASS`; at 516: UNCERTAIN.** This is the
  prediction most likely to be wrong, and it is the one that matters most.
- **J0: UNCERTAIN.** Nothing in the record predicts the MLP's best lr on a v3 representation.

## Method

- Training: `experiments/pointwise_baseline_v1_{516,1670}_mlp_{lr5e4,lr1e3,lr2e3}.yaml` via
  `scripts_cosim/datalab/pointwise_baseline_v1_train.sbatch`. The task table carries all three
  learning rates at full 16-seed width so indices never move once J0 selects; only the needed
  blocks are submitted. The sbatch asserts `num_datasets` and the split totals before training
  — *a corpus is its SPLIT, not its directory*.
- **An MLP checkpoint has no `.contract.json`** (`herosim-mlp-contract-lives-in-the-pt`); a
  blanket sidecar requirement silently blocks every MLP arm. Its contract lives **inside the
  `.pt`** (`inference_feature_layout`, `partial_state_contract`, `peer_mass`,
  `queue_feature_contract`), and those keys are pinned there instead — a real check, not a
  waived one.
- Serving: `--policy mlp_batch --mlp-model <ckpt>` on the same cells, workload and physics as
  every other arm in `results/po_v1` and `results/po_v1_clients`.
- Reading: `scripts_cosim/pointwise_baseline_v1_read.py`, bars committed first.

## Record

*(newest first; appended as the reads land)*

### 2026-09-19 — BLOCKED: the tabular MLP has never been servable under a partial-state representation

**A one-arm smoke test stopped this lineage from producing a clean, completely invalid
result.** It is the reason the lesson *"an instrument is not on until a result file has its
output"* exists, and it paid for itself the first time it was run here.

**What happened.** One `mlp_batch` arm was served on the real R0 cell with the gate's own
workload, physics and env. It completed: 50,000 tasks, 0 failed, 39 s wall,
`averageElapsedTime = 20.91 s` — which at that rung would have been **faster than reactive
Knative (22.22 s)** and far faster than both learned arms (`mpoff_516` +69.34 %, `gnnedge0`
+94.99 %). A headline result, from a well-formed result file, on the right cells.

**It was not the MLP.** `stats.schedulerCounters` reads
**`gnn_pure_decisions: 0`, `fallback_decisions: 50000`** — the scheduler fell back to
shortest-queue on **every single task**. The 20.91 s is a shortest-queue heuristic wearing the
MLP's name. Had the smoke test been skipped and the registered 384-arm gate run instead, this
lineage would have reported "the pointwise model class beats everything" and the number would
have been a queue heuristic's.

**The cause, and why it is structural rather than a bug.**
`[MLP Batch] Feature dim mismatch: 25 != 47`. `MLPBatchScheduler.build_feature_matrix`
assembles `task ⊕ platform ⊕ edge ⊕ candidate-relative` = **dim25cr (25)** and stops. It never
appends the partial-state block, **for any contract** — not `dim47crk`, and not `dim63crk`
either. `partial_state_columns` is consumed only by `src/policy/gnn/partial_state_edges.py`
and the cache builder; **no MLP serving path has ever called it.**

That is not an oversight in one function. The partial-state block is a function of the
**committed prefix**, so it changes at every decode step, and the MLP path scores **all edges
once per batch**. One-shot scoring structurally cannot carry a prefix-conditioned feature. It
is the same gap `src/policy/gnn/prefix_serving.py` was built to close for the GNN.

**The finding, which is worth more than the blocked comparison:**

> **The tabular MLP has never been served live under a partial-state representation in this
> programme.** Every live MLP arm in the record is a `dim22`/`dim25cr` arm; every partial-state
> ("T1") MLP result is an **offline** one.

**This explains the substitution the lineage was registered to question.** `mpoff` was not
chosen as "the pointwise arm" out of carelessness — **it was the only pointwise arm that could
be served at all** under the representation the graph arms use. That is a much better reason
than the record had recorded, and it is still not the same thing as the MLP.

**What was fixed on the way.**

1. **`dim47crk`** — the MLP layout under `partial_state_v3` now exists
   (`dim25cr` 25 + the v3 block 22 = 47), as a **separate layout name**, never a widened
   `dim63crk`. `_batch_edge_feature_dims` had refused v3 outright on the stated grounds that
   *"the MLP is not an arm of that lineage"* — organisational, not physical. v1/v2 stay
   byte-identical; `tests/test_dim47crk_layout.py` (8) pins all of it. **Training works**: the
   first v3 MLP trained to val_edge_acc 0.823 and declares `inference_feature_layout=dim47crk`.
2. **The B2 ablation width was the v1/v2 constant 38.** It zeroes the *last n* columns; on a
   47-column arm that zeroes the 22-column partial-state block **plus 16 columns of dim25cr**,
   and reports the result as the registered B2 quantity — inflated, in the direction that makes
   any arm look more partial-state-dependent than it is. Now contract-driven. The one
   checkpoint trained under the bug was deleted rather than kept.
3. **A fallback bar**, in `scripts_cosim/datalab/pointwise_baseline_v1_smoke.sbatch`. The
   counters were already exported (`stats.schedulerCounters`) — the instrument was on, and
   **nothing asserted it**. The smoke test now refuses any arm with `fallback_decisions != 0`.
   Note the GNN gate arms are safe by construction: they decode via
   `decode_prefix_conditioned`, whose docstring is explicit that *"a failed decode is an error
   here, never a fallback"*. The hazard is specific to the batch-scheduler path.
4. **"1670" is a label, not a count** — the corpus is 1,654 datasets. The training guard
   asserted the name and failed 9 of its own selection tasks. Filed in `gate-tools.md`.

**What remains, and its cost.** `decode_prefix_conditioned` is model-agnostic — it takes a
`score_fn(task_idx, committed) -> logits`. So serving the MLP needs an MLP counterpart to
`make_partial_state_score_fn`: rebuild this task's `[n_candidates, 47]` rows from the committed
prefix in the training extractor's exact column order, forward, return. The decoder, the caps,
the masks and the peer-group batching are all reused unchanged, and the prefix path **refuses
to fall back**, which removes the hazard above entirely.

That is a new serving path, and a serving path is not trustworthy until it is **proved
bit-identical to the offline extractor on held-out datasets** — the discipline
`peer_affinity_v1` stage 3 used (34/34) before any live number from it was believed. Estimate:
the scorer and its parity proof, then the 32 training runs (18 already done) and 384 gate arms.

**Nothing is read, and no bar is relaxed.** J0–J4 stand exactly as signed.
