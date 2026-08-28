# objective_pivot_v1 — ACTIVE

> **Status:** `ACTIVE` &nbsp;·&nbsp; **Index:** [LINEAGES.md](../../LINEAGES.md) &nbsp;·&nbsp; **Record spans:** 2026-08-28 →

**Outcome.** Current work. Program pivot registered 2026-08-28 (user decision): **stop
engineering the environment, change the training objective.** Phase 1 = a properly
powered **draw-distribution** reliability gate (the P5b control already ran and is CLOSED
— feature null, collapse draw-dominated; `gnn_draw_study_v1`'s own arithmetic says the
+50% line needs ≥ 12 draws). Phase 2 = the P3 horizon pilot as registered in
`program_verdict_v1`. Phase 3 = P1 closed-loop training (DAgger/policy-gradient against
the live simulator), the one path `program_verdict_v1` left open to the latency claim.
No phase's numbers exist yet.

**Related:** [program_verdict_v1](program_verdict_v1.md) · [cosim_deepdive_v1](cosim_deepdive_v1.md) · [gnn_draw_study_v1](gnn_draw_study_v1.md) · [p5b_draw_study](p5b_draw_study.md) · [route_b_env_pivot_v1](route_b_env_pivot_v1.md) · [route_b_v1](route_b_v1.md)

**Attachment:** [PHASE 1 REGISTRATION](objective_pivot_v1/phase1-registration-draft.md) — **SIGNED OFF 2026-08-28**, conditional on its §Venue & code identity: every new arm (train + gate) runs at the frozen arms' recorded commit `c08aa7e` in a clean tree, asserted per-arm by the scorer; a provenance mismatch VOIDs the arm. Power tables computed before sign-off; rank-sum primary at +50%, +100% sensitivity, +30% descriptive, n=16, MLP group frozen.

## Record

Newest first; the sections themselves are in chronological order below.

- [objective_pivot_v1 — Phase 1 LAUNCHED: pinned chain submitted (2026-08-28)](#objective-pivot-v1-phase-1-launched-pinned-chain-submitted-2026-08-28)
- [objective_pivot_v1 — PROGRAM REGISTRATION (2026-08-28)](#objective-pivot-v1-program-registration-2026-08-28)

---

### objective_pivot_v1 — PROGRAM REGISTRATION (2026-08-28)

**Why this lineage exists.** Every failed strategy in this repo's record is a strategy of
one kind: make a GNN beat the MLP through a single-batch supervised co-sim target. That
path is closed by measurement, not by fatigue — five physics mechanisms
(`graph_structure_physics` → `link_contention_v1`), the live state distribution
(`cosim_deepdive_v1`: 4,400 live states incl. all 14 MLP collapse trajectories, median
additive R² 0.99999), the P7 warmth-stratum controls (spread-plans R² = 1.00000 exactly),
route A (coupling without contention: pointwise-optimal), and route B stage 2 (with
contention: the GNN loses to pointwise-plus-prefix at memorization). `program_verdict_v1`
is the terminal synthesis: "no corpus design can reopen this; only a change of objective
can."

Two facts the failures obscured, and this lineage is built on:

1. **The GNN holds a real, direction-unambiguous live reliability edge — but it is a
   distribution claim, not an invariant, and it is not yet established at α=0.05.**
   The exploratory 30/30-beats-Knative / 0-collapse record was retired by the draw
   studies: with seeds varied, **2 of 8 GNN draws collapse** (worst draw 3/30) against
   an MLP whose draws collapse 4-of-8-at-least-once with a worst draw of **26/30** and
   range 26 (`gnn_draw_study_v1`, `p5b_draw_study` — both CLOSED, both verdicts
   INDETERMINATE on the registered sensitivity ladder; Fisher p=0.156 at +50%, rank-sum
   p=0.0274 at +50% but 0.1041 at +30%). "The GNN never collapses" **must not be
   written.** The honest claim in reach is *"GNN reliability is a tight distribution
   with a tail; the MLP's is a lottery"* — and per `gnn_draw_study_v1`'s own closing
   guidance the way to establish it is **more draws (≥ 12 at the +50% line), not a
   different rule.** That is Phase 1.
2. **The live edge is closed-loop** (dispersal/reliability), so the only objective that
   can train *toward* it is the live metric itself. External literature agrees: Decima
   (SIGCOMM 2019) trained its graph scheduler with RL against the running environment
   (+21% avg JCT), and the GNN-scheduling survey literature finds RL-trained schedulers
   beat per-step-imitation-trained ones. The simulator's role changes from **label
   factory** to **training environment**; co-sim's remaining roles are the M4
   separability diagnostic and P3 horizon labels.

**Fork context.** `route_b_env_pivot_v1` PARKED 2026-08-28 (its closing entry has the
full accounting); `route_c_link_transfer_v1` stays REGISTERED but superseded in priority.

**The phases, in order (each later phase gated by its own sign-off, not by this
registration):**

- **Phase 1 — establish the reliability distribution claim, properly powered.**
  `program_verdict_v1`'s P5b/P5a wording (2026-08-24) is superseded by the two draw
  studies that ran after it:
  - **P5b is DONE and CLOSED** (`p5b_draw_study`): the candidate-relative feature has
    zero pooled effect (both layouts median 4.0/30) and MLP collapse is a draw lottery
    (range 26 on the seed alone). Do not re-run it. Its lasting design law: **any
    reliability gate compares draw distributions; one checkpoint per arm measures
    nothing.** Its collapse rule (`total_rtt ≥ +50%` vs same-cell Knative, sensitivity
    at +30/+100) replaces the `chosen_queue_vs_min` detector, which is measured invalid
    for candidate-relative arms.
  - **The gate is therefore a draw-distribution extension of `gnn_draw_study_v1`**, per
    that node's own closing guidance ("more draws — ≥ 12 at the +50% line — not a
    different rule"): register, then train GNN seeds 9–16 (existing seeded draws 1–8
    stay valid, nothing re-run), gate on the same 30 cells with the same frozen Knative
    and MLP comparison arms, same three thresholds, same sensitivity clause. Statistic
    choices (dichotomy vs rank, final n, whether the MLP group also grows, margin
    co-primary vs Knative) are fixed in the Phase 1 registration **before** any new
    draw is trained — drafted for user sign-off, not decided unilaterally.
  - Either outcome is bankable: established → the paper's registered reliability claim,
    stated as a distribution ("tight with a tail vs a lottery"), never "never
    collapses"; not established at the registered power → the P6 frame (terminal
    separability negative + the diagnostic as the reusable artifact) with the draw
    studies reported as-is.
- **Phase 2 — P3 horizon pilot,** run exactly as pre-registered 2026-08-24 in
  `program_verdict_v1` (co-primaries, n ≥ 300, in-harness h calibration, h ≥ 5 s for
  axis-level terminality). Doubled value here: if it fires, its horizon labels are
  Phase 3's DAgger targets; if it nulls, Phase 3 goes straight to policy gradient. Then
  the ~0.5-day residency-hold scaling pilot with BOTH controls pre-registered
  (one-integer count column AND `--spread-plans-only`) before any Phase 3 spend.
- **Phase 3 — P1 closed-loop training.** Objective = the live metric itself. Policy =
  the served GNN architecture, warm-started from the `tempfix` checkpoint, MP bipartite
  (hard-stop on same-node edges without new physics), gated `mp_gate` residual
  initialized so the starting policy is the Phase 1 behaviour — the floor is what we
  already have. Loop: episodes via `executesimulation.py` (measured 755–1002 s per full
  trace; the 30k-event trace for the inner loop), DAgger stage only if P3 fired, then
  REINFORCE/PPO with a **paired-seed common-random-numbers baseline** (possible since the
  2026-08-18 workload-seed fix). **Fair arms, registered before training:** closed-loop
  GNN vs closed-loop MLP (same loop) vs Knative vs both frozen supervised checkpoints.
  **Kill criterion, registered before spend:** if at the pilot budget the paired-seed
  improvement over the frozen init is ≤ 0 at the registered n, P1 freezes as
  measured-negative — which closes the last open path and is itself an answer.
  Budget anchor from `program_verdict_v1`: ~1–2 week build, ~500 CPU-h pilot,
  ~5K CPU-h gate scale.

**Standing discipline carried into every phase:** wandb on every training run;
`tests/test_trainer_determinism.py` green (and extended to any new trainer) before
anything gated; no per-experiment trainer forks; provenance from `run_provenance`, never
banners; PARITY.md chain before any cross-venue comparison; no threshold edits after
data exists.

**Explicitly stopped by this registration:** physics levers / grids / corpus designs
aimed at making supervised targets non-additive; checkpoint promotion on offline
regret/accuracy (hard stop: −54% offline regret bought +9.9% worse live); unregistered
comparisons of any number that could become a claim.

**Literature anchors:** Decima — Mao et al., *Learning Scheduling Algorithms for Data
Processing Clusters*, SIGCOMM 2019 (<https://web.mit.edu/decima/content/sigcomm-2019.pdf>);
*Graph Neural Networks for Job Shop Scheduling Problems: A Survey*
(<https://arxiv.org/html/2406.14096v1>); *A Review of Deep RL in Serverless Computing*
(<https://arxiv.org/pdf/2311.12839>).

---

### objective_pivot_v1 — Phase 1 LAUNCHED: pinned chain submitted (2026-08-28)

Phase 1 executes as registered. Tooling at commit `900fec3`:

- **Pinned worktree** `~/gnn-herosim-pin-c08aa7e` created on datalab at
  `c08aa7ee140fd…` (detached, CODE_PATHS clean, verified), with symlinks into the main
  checkout for exactly the four data paths the runners touch (`models`, `logs`,
  `simulation_data/normal_sim_sweeps`, `simulation_data/graphs_cache_full_corpus_siv1_dim14`,
  `data/nofs-ids/traces`) — code from 2026-08-25, data shared with the frozen arms.
- **Job chain, each step gated on the previous** (`--dependency=afterok`):
  **719808** `objp1-determinism` (trainer determinism green *in the pinned tree*, the
  registration's precondition) → **719809** `objp1-train` (seeds 9–16, array 0-7,
  GPU-a40/l40s; refuses to overwrite an existing draw; same corpus/arch guards as the
  seed 1–8 array) → **719810** `objp1-gate` (240 cells, array 0-239%60, CPU-amd; block
  mapping byte-identical to job 712389's).
- Every sbatch asserts the pin (exact commit + clean CODE_PATHS) before spending
  compute; the gate results' `run_provenance.code` will therefore record `c08aa7e`,
  clean — the identity the scorer's VOID gate requires.
- `extract_gate_stats_summary.py` extended to arms `gnndraws1–16` and now carries
  `code_commit`/`code_dirty` + the registered env axes into the summary.
- **Scorer** `scripts_cosim/important/score_objective_pivot_phase1.py` (10 tests,
  `scripts_cosim/test_score_objective_pivot_phase1.py`, all green): VOID-gates
  provenance before computing anything (verified to refuse on the current summary,
  where the new arms are absent); rank-sum primary at +50% (fixed-seed 200k-permutation,
  midranks), must-hold at +100%, +30% and the dichotomy descriptive, sign-test
  secondary (≥ 13/16 negative mean margins vs Knative). Constants are the registration;
  the script takes no tuning arguments.
- Sync hygiene: cluster fast-forward was blocked by its own untracked copy of
  `route_b_pivot_h3_ctrl_additivity.json` (now git-tracked); md5-verified byte-identical
  to the tracked blob before removing it (pitfall #3 discipline), then ff'd to `900fec3`.

**Next session:** when 719810 completes — re-run
`extract_gate_stats_summary.py --out simulation_data/gate_stats_summary.json` on the
cluster, pull the summary back, run the scorer, and write the outcome here and in the
index row. The verdict is whatever the scorer prints; no reading of partial results
before the chain finishes.
