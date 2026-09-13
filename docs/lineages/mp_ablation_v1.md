# mp_ablation_v1 — CLOSED

> **Status:** `CLOSED` · **Index:** [LINEAGES.md](../../LINEAGES.md) · **Registered:** 2026-08-29 BEFORE any run · **Closed:** 2026-08-29

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

## Record

### 2026-08-29 — OUTCOME: **NO_DIFFERENCE_DETECTED**, and every line points the other way

Chain: **724732** (train, 16 MP-OFF draws, 16/16 COMPLETED) → **725150** (gate seeds 1–8,
240/240) → **725391** (gate seeds 9–16, 240/240) → **725636** (extract, 1650 results).
Zero failures across all 496 tasks. All 480 MP-OFF results verified to parse **end-to-end**,
not merely at the extractor's prefix. VOID gate passed: all 16 MP-OFF arms recorded
`c08aa7e`, clean tree, and **`GNN_DISABLE_MESSAGE_PASSING=1`**. Lever confirmed effective
before gating: all 16 MP-OFF checkpoints differ from their same-seed MP-ON pair.

**Registered verdict: `NO_DIFFERENCE_DETECTED`.** Primary paired exact Wilcoxon on per-seed
mean margin vs Knative: **p = 0.05066**, against α = 0.05. The bar was missed by 0.00066.

**But the direction is unambiguous, and it is the opposite of the hypothesis.**

| line | reading |
|---|---|
| primary, mean paired difference | **−5.63 pp** — MP-OFF *better*; 13 of 16 seeds negative |
| co-primary, severe collapse (≥ +50%) | MP-OFF **better in 2, worse in 0**, tied in 14 |
| descriptive +30% | MP-ON `[0,8,0,0,10,2,0…]` vs MP-OFF `[0,0,0,0,3,0,0…]` |
| descriptive +100% | both clean, 0/16 |
| backbone cells | MP-ON −25.1% vs MP-OFF **−34.4%** (Δ **−9.3 pp**) |
| flat cells | MP-ON +2.5% vs MP-OFF +4.3% (Δ +1.8 pp) |

Every secondary line agrees with the primary's direction: **message passing over the current
graph is not contributing, and looks actively harmful** — most of all on the backbone cells,
which is exactly where `objective_pivot_v1` measured the GNN's latency edge to live.

**The registration governs, and it says `NO_DIFFERENCE_DETECTED`.** p = 0.05066 is not
p ≤ 0.05, and the reading rules were fixed before the numbers existed. Reclassifying this as
`MP_HARMFUL` post-hoc — on a 0.00066 margin, with a two-sided test, after seeing the
direction — is exactly the move this program's discipline exists to prevent. The consequence
registered for a null stands: **the Phase 1 claim must be reworded away from "graph-aware".**

**What this does and does not license:**
- It does **not** show message passing is useless in general. It shows that *this* message
  passing — GIN over bipartite task↔platform edges plus same-node platform↔platform edges —
  does not carry the reliability edge on these 30 cells.
- It does **not** prove equivalence. A null here is a failure to detect; an equivalence claim
  needs a registered TOST with a margin, which this screen did not run.
- It **does** relocate the credit. With the GIN skipped, the model is a per-entity encoder
  plus an `EdgeScorer` with masked softmax over candidate placements. That is what beat the
  MLP in Phase 1. **The edge is attributable to the scoring/decode architecture, not to
  graph reasoning.**

**Consequence for the planned architecture work.** The prior recommendation was to add
link/route structure so contention on shared bottleneck links becomes representable. That
recommendation is now *ambiguous* rather than wrong, and the ablation cannot resolve it:
either (a) message passing is unhelpful here in general, or (b) it is being run over the
**wrong graph** — bipartite plus same-node edges cannot express shared-link contention, so it
aggregates noise, and a link-aware graph might help where this one hurts. Distinguishing (a)
from (b) requires building the link-aware graph and re-running this same ablation on it. That
is now the decisive experiment, and it must be registered before it is run.

**Tool correction found and fixed during scoring.** The co-primary's direction was inverted:
`worse` was computed as `sum(... for a, b in zip(voff, von) if b > a)`, which tests
`von > voff` — MP-**ON** collapsing more — while being labelled and reported as MP-OFF
collapsing more, and the one-sided p was computed against the wrong hypothesis (reported
p = 0.25 where the correct value is p = 1). The primary, and therefore the verdict, is
unaffected. Fixed, with a regression test that fails against the old code
(`test_co_primary_direction_is_not_inverted`); suite 14 → 15.
