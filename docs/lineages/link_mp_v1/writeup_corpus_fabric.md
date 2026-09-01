# The corpus is the lever: training-fabric/serving-fabric match dominates architecture

*Draft write-up, 2026-09-01. Every number below is from a registered gate; sources:
`docs/lineages/link_mp_v1.md` (primary), `docs/lineages/topology_transfer_v1.md`
(converse), `docs/lineages/program_verdict_v1.md` (mechanism). Frozen numbers — edit
only with a source-node edit.*

## Claim

For learned task placement in a bandwidth-constrained cluster, **matching the training
corpus's network fabric to the serving fabric is worth ~13 percentage points of live
latency vs Knative — more than every architecture intervention measured in this program
combined (≤5 pp each)** — and the converse holds with teeth: a training corpus whose
fabric never binds makes fabric features *label-irrelevant*, so no model trained on it
can learn topology no matter its architecture.

## Evidence 1 — the ~13 pp corpus effect (link_mp_v1, 48 registered arms)

Three model families × 16 paired seeds, all trained on a corpus whose backbone bandwidth
actually binds (1,675 brute-force-labelled datasets; core-link fabrics at 0.5 and
1.5 GB/s), all gated on the same 20 backbone cells against frozen same-cell Knative
baselines:

| arm | graph | message passing | mean vs Knative |
|---|---|---|---|
| lgon | `core_v1` link graph | on | **−38.5%** |
| lgmpoff | — | off (pointwise scorer) | **−38.0%** |
| lgctrl | old task↔platform graph | on | **−33.5%** |
| deployed reference | old graph | on, **fabric-blind corpus** | **−25.1%** |

Every family trained on the binding-fabric corpus beats the deployed fabric-blind model
by 8–13 pp. The spread *between* families — the entire architecture axis — is ≤5 pp.

## Evidence 2 — architecture, measured on the same corpus, moves ≤5 pp

With the corpus held fixed, the registered comparisons (one-sided exact Wilcoxon,
n = 16 paired seeds):

- Old-graph message passing is **harmful**: no-MP beats it by +4.50 pp (p = 0.00107).
- The link-aware graph **repairs exactly that harm**: +4.98 pp over the old graph
  (p = 0.00459).
- Repaired MP then **ties** the pointwise scorer: +0.47 pp (p = 0.372) — the ceiling
  `program_verdict_v1` predicts for a pointwise-separable supervised target.

Architecture choices matter for *not losing* points (the wrong graph costs ~4–5 pp);
they do not buy points the corpus has not already put on the table.

## Evidence 3 — the converse killed a lineage silently (topology_transfer_v1)

The one prior attempt to give the model backbone topology (`gnn_topo`,
`use_network_entities=True`) trained on a corpus with a 1000 MB/s backbone — bandwidth
that never binds at the corpus's transfer sizes. Its link features were therefore
**label-irrelevant**: the brute-force RTT labels contained no fabric term for the
features to explain. The arm failed (pooled win rate 0.449, CI [0.417, 0.481], resolved,
5/5 seeds converged) and *nothing in the training stack could have surfaced why* — loss
curves, accuracy, and gates all behave normally on a corpus whose labels simply lack the
signal. The lineage's pending re-run was retired on this evidence (2026-09-01).

## Evidence 4 — reliability rides the corpus too

The same 48 binding-fabric arms produced the program's **first all-clean collapse
table**: zero severe collapses (≥+50% vs Knative; also checked at +30% and +100%) in all
48 arms × 20 cells — across all three families, including the old-graph control. Prior
records on fabric-blind training showed collapse rates as high as 26/30 swinging on the
training seed alone. Collapse-freedom here tracked the corpus, not the architecture.

## Mechanism, in one sentence

A supervised label can only carry signal for physics that binds during labelling: if the
fabric never queues, the brute-force RTT of every placement is fabric-additive, the
label's fabric term is zero, and a fabric-aware feature is noise to the loss — so the
corpus, not the model class, decides what is learnable.

## Prescription

Before any architecture experiment, run the two-line corpus check:

1. **Does the resource the new inputs describe actually bind in the labels?** (Measure
   its share of label variance, or sweep its capacity and watch the labels move.)
2. **Ablate the feature and refit.** If the label fit is unchanged, the corpus cannot
   support the experiment — fix the corpus first; no training run on it is informative.

This is the same discipline as the program's one-integer and `--spread-plans-only`
controls: interrogate what the labels can express before crediting or blaming a model.

## Scope

Backbone (finite-bandwidth) serving cells only; on flat cells the deployed GNN was
*worse* than Knative (+2.5%) and no fabric-match dividend exists to collect. Corpus
frozen at 93% of its grid (the missing tail is the heaviest-contention combos — the
effect may be understated, not overstated). Margins are vs frozen same-cell Knative
baselines. The result does **not** contradict `program_verdict_v1`: the supervised
target stays pointwise-separable — this is a claim about corpus quality, not a
graph-architecture win.
