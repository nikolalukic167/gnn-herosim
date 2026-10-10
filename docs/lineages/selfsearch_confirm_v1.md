# selfsearch_confirm_v1 — does a pure GNN policy with search on its own score beat CD on fresh topologies?

**Status:** `REGISTERED` (2026-10-11). Every bar below was signed before any data.

**Why.** On the 12 development topologies, the no-split GNN choosing the slate plan with the highest score of its own
(gnn_selfsearch: exact up to 625 plans, coordinate ascent above; no S, no CD, no hand term) beat CD by −10.67 / −9.35 %
(11/12 at both rungs). It tied cd_exactS and cdxapply (`accel_nosplit_v1`, 2026-10-11). The registered no-split gate
had heavy at only −1.65 % (DIRECTION-ONLY). This tests the self-search arm where it has never been seen.

## Design (fixed now)

- **Environment:** the accel gate's, at both rungs (moderate m 11.6139, heavy m 27.622665025860204 exact),
  fastest_compatible, 4 WF1 windows (byte-identical to the accel gate's).
- **Topologies:** 24 fresh ids from the next free pool after a disjointness scan of datalab and every rp/* branch. They
  must not touch 16369–16399 (cost_to_go_v1 sealed plus spares) or any prior pool. No admission screen. A topology is
  replaced only on a build failure, by the next id.
- **Code:** one commit for every arm, rp/gnn-selfsearch a81137ab or a harness-only descendant. Identity is shown on 2
  cells before the full run.
- **Arms:**
  - **Primary:** gnn_eng and gnn_eng_physmp, no-split plus self-search (`GNN_SELF_SEARCH=1`), seeds 1 and 2;
  - CD, the bar, seed 0.
  - **Secondary, does the search add:** the same two checkpoints, no-split without self-search.
  - **Secondary, MP:** self-search on a separately trained MP-OFF twin's score (the g3 recipe with message passing
    disabled in training and serving, seeds 1 and 2). It runs only if the twin is trained and contract-checked before
    the single read (amended 2026-10-11 11:30). Otherwise the arm is recorded as not run, and no MP claim is possible.
  - **Descriptive (no family):** cd_exactS and cdxapply. These are the fair-bar hand searches holding the model's
    information.
- **Statistic:** paired % per topology (median over windows and seeds), then the median over 24, with an exact two-sided
  Wilcoxon.
- **Primary family:** {gnn_eng, physmp} self-search vs CD × {moderate, heavy}, Holm over 4.
- **WIN:** an arm at median ≤ −5 % with Holm p < .05 at **both** rungs, and no rung with CD Holm-confirmed faster.
- **Per-rung labels:** CONFIRMED, DIRECTION-ONLY, NOT-SEPARATED or CD-FASTER.
- **Secondary families, Holm within each:**
  - self-search vs no-split, × 2 checkpoints × 2 rungs;
  - GNN self-search vs twin self-search, × 2 rungs, if the twin runs.
- **Sealed:** SEALED=1, with the reader committed before the data. Stuck cells are killed by progress rate, never
  rerun, excluded and counted. Decision time is reported and not scored.
- **Scope of any claim:**
  - a WIN reads "a learned scorer with search on its own score beats CD". It is a learned-policy result, not a hand
    hybrid.
  - It is a message-passing result only if GNN self-search beats twin self-search.
  - It is never "beats the best hand rule" unless it also beats cd_exactS, which here is descriptive.
- **Prediction (coordinator):** both rungs CONFIRMED vs CD, since the development margins are about 2× the bar. It
  ties cd_exactS.

## Record (newest first)

- 2026-10-11 11:30 — **Amendment before any gate data (coordinator):** the MP twin arm may be submitted as a separate
  job after the main gate starts. Conditions:
  - it runs on the same pin plus only the twin kind, merged with identity shown;
  - it runs before the single read. The reader (fdd6f798) reads every arm once, after all cells land.
  - The primary family is untouched.
  - Gate setup: topologies 16490–16513 (spares 16514–16523, selection md5 15a36704); inputs 856266, 24/24 built; harness
    rp/selfsearch-confirm 79ea002c; reader fdd6f798, committed first.
- 2026-10-11 10:00 — **Registered** (coordinator; the trigger in `accel_nosplit_v1`'s gnn_selfsearch entry fired).
  - S4: id scan, inputs, harness, reader.
  - S7: MP-OFF twin training at the g3 recipe, in parallel.
