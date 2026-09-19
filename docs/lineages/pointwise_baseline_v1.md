# pointwise_baseline_v1 — is the pointwise baseline this programme keeps quoting actually the MLP?

**Status:** `REGISTERED` (2026-09-19) — bars signed before any arm is served. Every bar is a
module constant in `scripts_cosim/pointwise_baseline_v1_read.py`, committed with its 12 tests
before the reader was pointed at a results directory.

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

**The serving path is shared.** `MLPBatchScheduler` inherits graph build, decode and
roll-forward from `GNNScheduler` (via `XGBoostBatchScheduler`) and replaces **only the
scoring** — one batched `[N_edges, 22] → [N_edges]` pass through a pointwise MLP. Same
decoder, same peer-group batching, same capacity masks, same physics, same cells, same
workload. This programme has been burned by serving-layout confounds twice
(`herosim-inference-layout-confound`, `herosim-live-quality-is-a-training-draw-lottery`), so
the shared path is the reason this contrast is readable at all.

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
