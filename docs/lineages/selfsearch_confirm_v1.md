# selfsearch_confirm_v1 — does a pure GNN policy with search on its own score beat CD on fresh topologies?

**Status:** `CLOSED` (2026-10-11) — **DIRECTION-ONLY** (moderate CONFIRMED, heavy DIRECTION-ONLY; no WIN). Registered
2026-10-11; every bar below was signed before its data.

**Outcome.** A pure learned policy, the no-split GNN choosing the slate plan with the best score of its own (no S, no
CD), beats CD on 24 fresh topologies. The margin is confirmed at moderate; at heavy it is significant but under the −5 %
bar.
- **Primary, vs CD, median paired % (Holm over 4), moderate / heavy:**
  - gnn_eng self-search: −8.46 % (23/24, Holm < .0001) / −3.09 % (19/24, Holm .0017);
  - physmp self-search: −10.09 % (23/24, < .0001) / −3.76 % (20/24, .0017).
  - No rung has CD faster.
- **Search adds over the plain no-split decode**, significant at both rungs and both checkpoints: −1.0 / −1.6 % and
  −2.6 / −3.0 %.
- **Not a message-passing result:** gnn_eng self-search vs its separately trained MP-OFF twin's self-search is
  +0.07 / −0.24 % (NOT-SEPARATED).
- **The hand searches holding the same information are faster at heavy:** vs cd_exactS +3.66 / +3.40 % (2/24),
  vs cdxapply +2.21 / +1.78 %. At moderate they tie.
- The development heavy margin (−9.4 % on 12 topologies) shrank to −3.1 / −3.8 % on fresh topologies. Moderate held.
- **Quote as:** "a learned scorer with search on its own score beats CD, confirmed at moderate". Never as a
  message-passing win, and never as beating the best hand search.

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

- 2026-10-11 21:00 — **CLOSED DIRECTION-ONLY** (S4; gate 856313 at 79ea002c plus twin 856438 at 2c84ab4e; read once as
  856584 with reader fdd6f798, both gate dirs; 2,688 summaries, 0 failed, 24 topologies 16490–16513).
  - Search ran on every batch. gnn_eng exact / ascent batches: 2.14 M / 0.42 M at moderate, 1.97 M / 0.51 M at heavy.
    The plan changed vs the decode in 37 / 41 % of batches.
  - Decision time per task, median 16–25 ms.
  - **Primary:** as in the head.
  - **Secondary, does the search add (Holm over 4, all significant):** gnn_eng −0.97 / −1.57 %; physmp −2.59 / −3.00 %.
    The reader's per-rung labels inherit the −5 % bar, so read the numbers.
  - **MP family (Holm over 2):** +0.07 % (11/24, p .81) / −0.24 % (13/24, Holm .39). NOT-SEPARATED.
  - **Descriptive:**
    - self-search vs cd_exactS: −0.16 / +3.66 % (gnn_eng), −1.96 / +3.40 % (physmp);
    - vs cdxapply: −1.87 / +2.21 %, −3.12 / +1.78 %;
    - no-split vs CD: −7.44 / −1.67 %, −7.96 / −0.55 %.
  - **Prediction:** both rungs CONFIRMED. It held at moderate and missed at heavy.
- 2026-10-11 17:45 — **MP-OFF twin trained and checked** (S7; rp/accel-twin 2d398731, the gnn_eng **g4** recipe, which
  is the served checkpoint, with message passing disabled in training and serving; job 856255, 100-epoch cap, W&B
  gnn-accel-replica-v1).
  - **Corpus:** the accel_replica_v1 cache (2,731 wf1_train tables plus 438 wf1_heldout as val, split sha f5bc4f51),
    the same as the original.
  - **Checkpoints, md5:** seed 1 bc44f9c2…, seed 2 7798e128…. Sidechecks pass.
  - **Serving smoke 856430:** self-search ran on all batches.
  - **Offline best-val regret:** twin 0.645 / 0.655 s against gnn_eng g4 0.781 / 0.692 s. That ranks epochs, not
    arms, but it is disclosed as a prior that the twin may tie or beat the GNN (cf. `gnn_seeded_cd_mp_ablation_v1`).
  - **Gate harness for the twin arm:** 2c84ab4e = 79ea002c plus one line.
- 2026-10-11 15:00 — **Gate launched** (S4).
  - Identity at 79ea002c, decision-identical:
    - cd_exactS and physmp no-split: 0 differences;
    - cdxapply and self-search: identical latency; only counters absent at the old pin and a wall-clock timer differ.
  - **Failed launch 856305:** all 8 tasks exited in 1–4 s because the selection file lacked `"verdict": "DESIGN-READY"`
    (`fresh_topo_burst_v1_gate.py:1086`). No cell ran.
    - The key was added; the topologies are unchanged and the md5 is now 63d4eae2.
    - The cost_to_go_v1 selection file has the same gap and must be fixed before its gate.
  - **Relaunched as 856313:** 2,304 cells, sealed, ETA about 3–4 h. The twin arm follows separately.
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
