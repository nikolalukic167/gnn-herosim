# p5b_draw_study — CLOSED

> **Status:** `CLOSED` &nbsp;·&nbsp; **Index:** [LINEAGES.md](../../LINEAGES.md) &nbsp;·&nbsp; **Record spans:** 2026-08-24 → 2026-08-24

**Outcome.** **Q1 = LOTTERY, Q2 = DRAW-DOMINATED.** The MLP trainer never seeded torch, so every MLP checkpoint before 2026-08-24 is an unreproducible draw. Collapse counts swing 0→26 on the seed alone. **Retires "the MLP collapses 7/30"** and every inference built on it.

**Entry points:** `scripts_cosim/datalab/p5b_draw_study_{train,gate}.sbatch`, `important/score_p5b_draw_study.py`; `torch.manual_seed` fix in `train_mlp_dim22_from_batch.py`

**Datasets:** none — 16 checkpoints over the two existing caches

**Related:** [p5b_candidate_relative](p5b_candidate_relative.md) · [trainer_determinism_v1](trainer_determinism_v1.md) · [gnn_draw_study_v1](gnn_draw_study_v1.md)

## Standing (from the index table)

**Closed 2026-08-24. Q1 = LOTTERY, Q2 = DRAW-DOMINATED, stable at +30/+50/+100%.** Found first that the MLP trainer **never seeded torch** — `--random-state` pinned the split, not the weights — so every MLP checkpoint here before today is an unreproducible draw (the GNN trainer always seeded; the asymmetry went unnoticed). Then measured the full `{dim14,tempfix} × {dim22,dim25cr} × seeds{1..4}` grid, 480 gate runs: collapse counts swing **0→10, 0→11, 0→21, 0→26** on the seed alone, and the candrel effect flips sign *within* both caches. **Retires "the MLP collapses 7/30" (that config gives 0, 0, 21, 16), the "same count, different set ⇒ architectural" inference, and P5b's split — all noise.** The GNN's 0-collapse record survives on these cells (0/30 both arms, −18.9%/−27.1%) but at 2–3 draws vs a measured MLP draw distribution it is p ≈ 0.125 — **unfalsified, not established.** Any future reliability gate must compare draw *distributions*. Outcome below.

> **Split note.** This node was carved out of a single 4,995-line `LINEAGES.md` on
> 2026-08-27; the section bodies below are byte-for-byte as written. An *above* / *below*
> pointer inside one may refer to a section that now lives in another node — the
> **Related** links say which.

## Record

Newest first; the sections themselves are in chronological order below.

- [p5b_draw_study — OUTCOME: **Q1 = LOTTERY, Q2 = DRAW-DOMINATED** (2026-08-24)](#p5b-draw-study-outcome-q1-lottery-q2-draw-dominated-2026-08-24)
- [p5b_draw_study — PRE-REGISTRATION (written 2026-08-24, before any run)](#p5b-draw-study-pre-registration-written-2026-08-24-before-any-run)

---

### p5b_draw_study — PRE-REGISTRATION (written 2026-08-24, before any run)

**🔴 First, a defect that reframes every MLP result in this repo.** Nothing ever called
`torch.manual_seed`. `--random-state` seeded the parent split
(`split_by_parent_three_way`) and the batch order (`random.Random`) — **the model's weight
init came from OS entropy**. Verified by construction: two identical invocations of the
trainer produce different first-layer weights; with an explicit seed they are bit-identical.

Consequences, stated plainly:
- **Every MLP checkpoint produced before 2026-08-24 is an unreproducible draw.** Re-running
  its exact command cannot recover it. `random_state: 42` in those `.meta.json` files
  describes the split, not the model.
- The 2026-08-23 subsection above describes the `mlp` vs `mlptempfix` comparison as
  "an A/B on training data alone" and concludes the collapse is architectural because
  "only the checkpoint and the sweep dir differ". **That description is wrong**: the two
  checkpoints differ by cache *and* by an uncontrolled weight init. The observation
  (7/30 each, different victims) stands; the attribution to the cache does not.
- P5b's own confound is therefore not merely "cache vs seed" but "cache vs an
  uncontrolled draw", which is worse and cannot be resolved by re-reading anything.

`torch.manual_seed(args.random_state)` + `np.random.seed` are now wired in, and
checkpoints record `torch_seeded: true` so a seeded checkpoint can be told from a drawn
one. Verified: same seed → bit-identical weights.

**The study.** A full grid, so variance can finally be attributed instead of assumed:
`{dim14, dim14_tempfix} × {dim22, dim25cr} × seeds {1,2,3,4}` = 16 checkpoints × the same
30 backbone cells = 480 gate runs (`p5b_draw_study_{train,gate}.sbatch`). The `dim22` arms
are **not** padding: they measure what a *fixed* cache and layout does across draws, which
is the quantity every reliability claim in this program has silently assumed to be small.
Existing s42 checkpoints are retained as a 5th, unseeded draw and are never mixed into the
seeded statistics.

**Criterion, registered.** Collapse = `total_rtt` ≥ **+50%** vs the same-cell Knative arm.
Chosen over the P5b detector because `chosen_queue_vs_min` p95 is *measured* invalid for
candidate-relative arms (fires on 5/30 healthy `mlpcandrel` cells, one a −2.5% win).
Sensitivity at +30% and +100%; a verdict that does not hold at all three is INDETERMINATE.

**Q1 — is pointwise reliability a draw lottery?** Per condition, the range of collapse
counts across its 4 seeds.
- **LOTTERY** iff the largest within-condition range ≥ **5**/30.
- **STABLE** iff every condition's range ≤ **2**/30.
- **PARTIAL** otherwise.
Rationale for 5: P5b's headline effect was 7→2 and 7→17. If a *fixed* cache and layout
swings ≥ 5 cells on the seed alone, the feature effect was never distinguishable from a
draw, and neither was the 7/30-vs-7/30 result the architectural reading rests on.

**Q2 — does the candrel effect have a cache-determined sign?** Per (cache, seed),
`delta = collapses(dim25cr) − collapses(dim22)` at the same cache and seed: 8 deltas.
- **CACHE-DETERMINED** iff all 4 deltas of one cache are > 0 and all 4 of the other < 0
  (perfect sign separation; coin-flip null p = 2 × 2⁻⁸ = 0.0078).
- **DRAW-DOMINATED** iff the sign is mixed within either cache.
- Ties (delta = 0) count against separation.

**Q3 — descriptive, no threshold.** Mean and range of collapse count per layout, pooled
over caches and seeds (8 checkpoints each). Reported whatever Q1 says, but **not**
interpreted as a feature effect if Q1 returns LOTTERY.

**Validity gate 2 still applies** to every `dim25cr` checkpoint (CR-ablation argmax change
≥ 5%), asserted in the training sbatch, which refuses to finish otherwise.

**What closes the story.** Q1 = LOTTERY would mean the honest headline is *"pointwise
reliability on this benchmark is a property of the draw; the GNN's 0/120 is the only
claim that survives, and it needs its own multi-seed check"*. Q1 = STABLE with Q2 =
CACHE-DETERMINED would restore a real, attributable feature/corpus effect. Either way the
`p5b_candidate_relative` INDETERMINATE resolves into a statement that can be written down.

### p5b_draw_study — OUTCOME: **Q1 = LOTTERY, Q2 = DRAW-DOMINATED** (2026-08-24)

Trains job `711758` (16/16; every `dim25cr` arm passed validity gate 2, 0.179–0.291); gate
array `711774`, **480/480 COMPLETED**, 96 sweep dirs, zero failures. Artifact:
`simulation_data/p5b_draw_study_verdict.json`.

**Collapse counts, `total_rtt` ≥ +50% vs the same-cell Knative arm:**

| condition | s1 | s2 | s3 | s4 | range |
|---|---|---|---|---|---|
| `dim14 / dim22` | 0/30 | 0/30 | 8/30 | 10/30 | **10** |
| `dim14 / dim25cr` | 5/30 | 3/30 | 0/30 | 11/30 | **11** |
| `tempfix / dim22` | 0/30 | 0/30 | 21/30 | 16/30 | **21** |
| `tempfix / dim25cr` | 26/30 | 0/30 | 0/30 | 7/30 | **26** |

**Q1 = LOTTERY** (worst range 26/30 against a threshold of 5). **Q2 = DRAW-DOMINATED** —
paired deltas are `dim14: +5, +3, −8, +1` and `tempfix: +26, 0, −21, −9`; the sign is mixed
*within* both caches, so cache separation is arithmetically impossible. **Both verdicts
hold at +30%, +50% and +100%**, as the registered rule requires. Q3 (descriptive): pooled
over caches and seeds the two layouts have the **same median, 4.0/30** — the
candidate-relative feature has no average effect whatsoever.

**This retires three claims that were load-bearing in this file.**
1. **"The MLP collapses 7/30" was one draw.** The same configuration
   (`tempfix / dim22` — the corrected-cache baseline) gives **0, 0, 21, 16** across seeds.
   Two of four draws never collapse on any of the 30 cells.
2. **"Exactly 7 of 30 under each checkpoint — same count, different set" is a
   coincidence**, not the signature of an architectural failure. It was two samples from a
   distribution whose range is 21.
3. **P5b's 7→2 / 7→17 split was noise.** It sits well inside the baseline's own draw
   spread, and the feature's pooled median effect is zero.

**The seeding defect is MLP-specific.** `src/notebooks/train_near_rtt.py:104-107` seeds
`random`/`numpy`/`torch`/`torch.cuda`; the MLP trainer seeded none of them until today. The
GNN's checkpoints were always reproducible and its arms are genuine distinct draws — the
asymmetry went unnoticed because nobody compared the two trainers' seeding.

**What survives, and how strong it actually is.** On these 30 cells both GNN arms are 0/30
collapses with mean margins −18.9% (`deployed`) and −27.1% (`tempfix`), against MLP arms at
6/30, 7/30, 12/30, 2/30 and margins +25.9%, +48.0%, +203.8%, −6.9%. Real — **but it is 2–3
GNN draws against a now-measured MLP draw distribution.** Under the MLP's own draw-level
rate (4 of 8 seeded draws collapse at least once), three clean GNN draws is
p ≈ 0.5³ = **0.125 — not significant**. *The GNN's reliability advantage is not
established; it is unfalsified.* Establishing it requires exactly what was just done to the
MLP: ≥ 8 seeded GNN draws × 30 cells, pre-registered.

**Consequences for the program.**
- **Do not write "the GNN never collapses and the MLP does."** Write, if anything: *across
  the draws tested, every GNN draw was collapse-free while MLP draws collapsed in 4 of 8 —
  a difference this evidence cannot separate from a 1-in-8 outcome.*
- **P5a is superseded, not merely unrunnable.** Any reliability gate on this benchmark must
  compare *draw distributions*; one checkpoint per arm measures nothing. This is the most
  important design change to come out of `program_verdict_v1`.
- The terminal negative from `program_verdict_v1` (single-batch co-sim targets are
  pointwise-separable) is **untouched** — it never rested on any checkpoint.
- Every past per-checkpoint live-gate margin in this file inherits the caveat. The numbers
  are correct; what they measure is one draw.

**Status: CLOSED.** `p5b_candidate_relative`'s INDETERMINATE is resolved: the feature did
nothing, the cache did nothing, and the variable that moved was the one nobody was
controlling.

---
