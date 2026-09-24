# trainer_determinism_v1 — ACTIVE

> **Status:** `ACTIVE` &nbsp;·&nbsp; **Index:** [LINEAGES.md](../../LINEAGES.md) &nbsp;·&nbsp; **Record spans:** 2026-08-25 → 2026-08-25

**Outcome.** The seed fix reached **1 of 4 trainers**. Three defect classes now — newest: `prepare_graphs_cache` seeded 42 at module import and clobbered every GNN draw's seed. `tests/test_trainer_determinism.py` covers every trainer; run it before training anything you intend to gate.

**Related:** [p5b_draw_study](p5b_draw_study.md) · [gnn_draw_study_v1](gnn_draw_study_v1.md)

> **Split note.** This node was carved out of a single 4,995-line `LINEAGES.md` on
> 2026-08-27; the section bodies below are byte-for-byte as written. An *above* / *below*
> pointer inside one may refer to a section that now lives in another node — the
> **Related** links say which.

## Record

Newest first; the sections themselves are in chronological order below.

- [trainer_determinism_v1 — the seed fix reached 1 of 4 trainers (2026-08-25)](#trainer-determinism-v1-the-seed-fix-reached-1-of-4-trainers-2026-08-25)

---

### trainer_determinism_v1 — the seed fix reached 1 of 4 trainers (2026-08-25)

`p5b_draw_study` fixed `train_mlp_dim22_from_batch.py` and stopped there.
`train_mlp.py`, `train_mlp_ce_reduced.py` and `train_mlp_dim22_from_seq.py` still had the
identical defect — split and batch order seeded, weight init from OS entropy — so
"every MLP checkpoint before 2026-08-24 is an unreproducible draw" was still *true going
forward* for three of the four trainers. All three now seed torch and stamp `torch_seeded`.

`torch.use_deterministic_algorithms(True, warn_only=True)` added to all four notebook GNN
trainers (`train_near_rtt`, `train`, `train_ram`, `train_seq`), default-on, escape hatch
`NEAR_RTT_NONDETERMINISTIC=1`. `gnn_necessity_ablation.py` has had this since 2026-08-19;
the trainers that produce deployable checkpoints did not. **This changes training numerics**
— a different algorithm is selected — so checkpoints trained from here are not bit-comparable
to earlier ones. Deliberate, and it lands *before* the draw study trains.

**`tests/test_trainer_determinism.py`** (10 tests, ~12 s, no GPU). Two runs at one seed must
give bit-identical weights. Both dynamic arms are verified to have teeth: remove
`torch.manual_seed` and the MLP arm fails 2 ways, the GNN arm diverges 28/31 tensors.

**What the test does not cover, measured rather than assumed.** It cannot catch removal of
`use_deterministic_algorithms`. The GIN nondeterminism recorded on 2026-08-19 **did not
reproduce on this box at any size tried** — 12 to 200 graphs, 2 to 5 epochs, node edges on
and off, flag on and off, all bit-identical. So the dynamic test cannot discriminate, and
the guard for that half is a *static* assertion that the line is present in every trainer.
Absence on one box is not evidence the op is gone; it is build- and hardware-dependent.

**Status: ACTIVE.** No CI exists in this repo — the command is documented in `CLAUDE.md`
and is manual.

---
