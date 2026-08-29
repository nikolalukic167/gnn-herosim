# mp_ablation_v1 — REGISTERED

> **Status:** `REGISTERED` · **Index:** [LINEAGES.md](../../LINEAGES.md) · **Registered:** 2026-08-29, BEFORE any run

**Question.** `objective_pivot_v1` Phase 1 established that the GNN's severe-collapse burden
is stochastically smaller than the MLP's. That claim is currently worded as a statement about
**graph-awareness** — but GNN and MLP differ in architecture, capacity, regularisation and
optimiser path, not only in whether they pass messages. **Is message passing the cause?**

This screen is a **control on the program's only established claim**. It is registered before
any run, and its reading rules are fixed here.

## Design — paired, not two independent groups

`GNN_DISABLE_MESSAGE_PASSING=1` skips the GIN call in `TaskPlacementGNN.forward`
(`src/policy/gnn/gnn_model.py:427`); the GIN module is still constructed, so the **parameter
set and its initialisation are identical**. At a fixed seed the two arms therefore start from
bit-identical weights and differ only in the forward path. That makes this a **paired**
design, which is why n=16 is adequate.

| | arm A (MP-ON) | arm B (MP-OFF) |
|---|---|---|
| checkpoints | existing `gnn-draw-s{1..16}` | new `gnn-draw-mpoff-s{1..16}` |
| `GNN_DISABLE_MESSAGE_PASSING` | unset | **`1` at BOTH train and serve** |
| commit | `c08aa7e`, clean | `c08aa7e`, clean (same pinned worktree) |
| corpus / cache | `graphs_cache_full_corpus_siv1_dim14` | identical |
| cells | the same 30 | the same 30 |
| arms in summary | `gnndraws{1..16}` | `gnnmpoff{1..16}` |

**The failure mode this design must avoid** is a train/serve mismatch — serving a
message-passing graph a checkpoint was not trained on cost 12.4× live RTT on 2026-08-16.
The flag is an **environment variable read at both train and serve time**, not a sidecar
property, so it must be set in both sbatch files. It is recorded in gate
`run_provenance.env`, which is what the VOID gate below reads.

## Statistics, fixed now

- **PRIMARY — paired Wilcoxon signed-rank** on the per-seed **mean margin vs same-cell
  Knative** (16 pairs, continuous, no ties). Two-sided, α = 0.05. Asks: does removing message
  passing change placement quality at all?
- **CO-PRIMARY — per-seed severe-collapse count** (cells at `total_rtt` ≥ +50% vs Knative),
  paired, one-sided exact sign test over **non-tied pairs only**, with the tie count reported.
  MP-ON counts are mostly zero, so ties are expected and the test may be underpowered **by
  construction** — that is stated here, in advance, so a null result is not read as evidence
  of equivalence.
- **Descriptive:** +30% and +100% collapse counts; per-block split on backbone vs flat cells,
  because `objective_pivot_v1`'s exploratory section measured the GNN's latency edge to be
  backbone-only (−25.1% vs +2.5%). If message passing matters anywhere, that is where.

## Reading rules — decided before the numbers exist

1. **MP-OFF materially worse** (primary significant, MP-OFF margin higher; or MP-OFF collapse
   count higher on the sign test) ⇒ message passing is load-bearing. The Phase 1 claim may
   keep its "graph-aware" wording.
2. **No detectable difference** ⇒ the reliability edge is **not** attributable to message
   passing on this evidence. The Phase 1 claim must be reworded from "graph-aware" to a
   statement about the model class as a whole, and the paper must report this control. This
   is a *failure to detect*, never a proof of equivalence — an equivalence claim would need a
   registered TOST with a margin, which this screen does not run.
3. **MP-OFF materially better** ⇒ report as-is; the graph channel is actively harmful and the
   architecture work in Phase I of the grant proposal is re-scoped around that.

**VOID** (no verdict computed, fix and re-run the named arms): any MP-OFF arm whose gate
`run_provenance` does not record `GNN_DISABLE_MESSAGE_PASSING=1`, or records a commit other
than `c08aa7e`, a dirty tree, or any registered env axis differing from arm A's pin table.

## Execution

Two-stage gate (seeds 1–8, then 9–16) with summary extraction between stages, because
480 results × ~95 MB ≈ 45 GB against ~50 GB of quota headroom. The 2026-08-28 incident
(exhausted `/home` quota → 0-byte SLURM logs → 191 lost tasks) is the reason this is staged
rather than submitted as one 480-task array.
