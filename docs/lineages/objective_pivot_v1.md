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
  `objp1-determinism` (trainer determinism green *in the pinned tree*, the
  registration's precondition) → `objp1-train` (seeds 9–16, array 0-7,
  GPU-a40/l40s; refuses to overwrite an existing draw; same corpus/arch guards as the
  seed 1–8 array) → `objp1-gate` (240 cells, array 0-239%60, CPU-amd; block
  mapping byte-identical to job 712389's).
- **First submission (719808→719809→719810) died at the precondition**: the cluster
  `gnn` env had no `pytest`, so the determinism job failed after its pin check and the
  dependents went `DependencyNeverSatisfied`. Fix per the datalab-pitfalls #10
  discipline: `pip install pytest` (pure-Python; numpy/scipy/sklearn/torch verified
  untouched), then the mandatory numerical-inertness proof —
  `verify_venue_parity.py --mode logits --assert`: **max |Δ| = 0.0 exactly, 0/256
  argmax flips**. Chain resubmitted: **719817** (determinism) → **719818** (train) →
  **719819** (gate).
- **Determinism precondition satisfied in full, not just formally.** 719817 passed
  8/10 with the two *bit-identical* trainer tests skipped — their small regime_b test
  caches existed only locally. Caches rsynced over (5.7 MB) and linked into the pinned
  worktree; verification job **719838** then ran the suite complete in the pinned tree:
  **10/10 PASSED**, including `test_gnn_training_is_bit_identical_at_a_fixed_seed` —
  the tree the draws train from is proven bit-reproducible, while seeds 9–12 were
  already training on L40S nodes (their per-checkpoint `deterministic_algorithms`
  contract assertions in the sbatch remain the per-draw guarantee).
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

### 2026-08-29 — Phase 1 OUTCOME: **PASS**. The reliability claim is established.

Chain as run: **719818** (train, seeds 9–16, 8/8 COMPLETED, ~16 min/draw on L40S) →
**719819** (gate, 49/240 cells before the incident below) → **724238** (gate re-run,
191/191 COMPLETED) → **724714** (summary extract, 1170 results over 6 blocks).
All 240 new-arm results verified to parse **end-to-end** as JSON, not merely at the
prefix the extractor reads. Scorer VOID gate passed: all 8 new arms recorded
`c08aa7e`, clean tree, registered env axes.

Verdict from `score_objective_pivot_phase1.py` (constants ARE the registration;
`simulation_data/objective_pivot_phase1_verdict.json`):

| threshold | role | GNN collapse counts (s1..s16) | MLP (frozen) | rank-sum p |
|---|---|---|---|---|
| +50% | **PRIMARY** | `0,1,0,0,3,0,0,0,0,0,0,0,0,0,0,0` | `0,0,8,10,5,3,0,11,0,0,21,16,26,0,0,7` | **0.00143** |
| +100% | SENSITIVITY (must hold) | all zero | `0,0,6,8,3,2,0,8,0,0,18,12,23,0,0,6` | **0.00045** |
| +30% | descriptive only | `0,8,0,0,10,2,0,0,0,0,0,0,0,0,0,0` | `0,1,8,10,6,5,0,12,0,0,22,17,26,0,0,7` | 0.00538 |

**The registered claim, now writable:** across seeded draws, the GNN's severe-collapse
burden (cells at `total_rtt` ≥ +50% vs same-cell Knative) is stochastically smaller
than the MLP's. **Mandatory limitation (registered): the claim is scoped to severe
collapse only.** At +30% the GNN is still clearly better but two draws (s2=8, s5=10)
carry real burden — this is "a tight distribution with a tail versus a lottery", never
"the GNN does not collapse", which `gnn_draw_study_v1` already falsified. At +100% the
GNN is clean in 16/16 draws, which is where the separation is starkest.

**SECONDARY (its own claim) also PASSES:** per-draw mean margin vs same-cell Knative is
negative in **16/16** draws (bar ≥ 13/16), range −1.2% (s5) to −27.2% (s16),
sign-test p = 1.5e−05.

Dichotomy figures are descriptive as registered (clean 14/16 vs 7/16 at +50%,
Fisher p = 0.0117) — its joint power at n=16 was computed as ~0.135 *before* sign-off,
which is why it never carried the verdict.

**Incident — the `/home` quota, and why it matters beyond tonight.** Gate 719819 lost
191 of 240 tasks at 20:26 UTC; a blind resubmit (720349) lost all 191 again. Root cause
was **not** cluster instability but an exhausted per-account `/home` quota (~250G):
compute nodes could still allocate inodes but not write bytes (`Disk quota exceeded`),
so SLURM created **0-byte** `.out`/`.err` files and every data-writing task died — some
mid-simulation, later ones in under a second. `df` was actively misleading (6.3T free on
the filesystem; the limit is per-user), and `sacct -a` returns only this account's jobs,
so "no other users were affected" was an artifact, not evidence. What finally isolated
it: a **single task, run alone**, failed instantly — ruling out both scale and code — and
a direct compute-node write test returned the quota error. Space freed by deleting the
three PARKED `route_b` dataset dirs (588 datasets, 55G;
`gnn_datasets_dag4_route_b_pilot_v1_8task`, `..._route_b_pivot_h3_ctrl`,
`..._route_b_pivot_h3` — none were in `REGISTRY.json`), 250G → 196G. Re-run then
completed 191/191 with zero failures.

**Two traps this incident exposes for any future gate re-run** (both now also in
`docs/gates/gate-tools.md`):
1. The runner's skip guard (`file exists` + nonzero `total_rtt`) treats a result written
   by a task that later *failed* as complete. Six such files existed here; four were
   92–97 MB with plausible RTTs and would have been silently adopted on resubmit. Delete
   any result whose SLURM task did not reach `COMPLETED` before re-running.
2. A quota/IO failure mid-write can truncate a result whose *prefix* still parses — and
   both the skip guard and `extract_gate_stats_summary.py` read only the prefix. Verify
   full-document JSON parse, not prefix validity. (Checked here: 240/240 clean.)

**Phase 1 is CLOSED.** Next: Phase 2, the P3 horizon pilot pre-registered in
`docs/lineages/program_verdict_v1.md` (co-primaries, n ≥ 300, in-harness h-calibration at
h ∈ {2,5,10} first, h registered before the array). Its horizon labels double as Phase 3
DAgger targets.

### 2026-08-29 — EXPLORATORY (post-hoc, NOT part of the registered claim)

Everything below was computed after the verdict, was not pre-registered, and may not be
cited as a claim. It is recorded because it changes what Phases 2/3 should ask.

**Pipeline, stated exactly** (the question "was this co-sim → train → big-workload test?"):

| stage | what it actually was |
|---|---|
| data gen | co-simulation, **2,651 datasets**, each a **4-task** batch with every placement brute-forced (`4tasks_contention_v2/v3/v4_pilot`, `4tasks_1060_warmth_v2`, `4tasks_sparse_warmth_v2`, `4tasks_highq_safe_20260606`); 20 client / 20 server nodes, sparse, `p ∈ [0.15, 0.5]`, `node_disk_v2`. **`link_topology` is absent from these datasets.** |
| train | `graphs_cache_full_corpus_siv1_dim14` → `near-rtt-v2-full-corpus-siv1-dim14-ce-only` config, 16 seeded draws, `dim22` / `scale_invariant_v1` |
| test | live simulation, **301,352 events** (`workload-150-100`) and **351,767 events** (`workload-175-100`) — ~300–350k tasks, not 500k — over 30 cells |

So yes, it is the co-sim → train → live-gate pipeline, but with a **large deliberate
generalisation gap**: fitted on 4-task brute-forced batches carrying no link model, then
judged on ~300k-event workloads, two thirds of whose cells carry a network backbone the
training corpus never contained.

**The 30 cells are not one environment.** 20 carry an explicit backbone
(`n_core` 4 or 8, access link 20 ms, core link 4 ms, finite bandwidth 0.5 or 1.5 Mbps,
44–48 links, 20 routes); 10 are flat (`link_topology: None`, i.e. the pre-network shape).

**Mean margin vs same-cell Knative, split on that axis:**

| block | kind | GNN mean | GNN worst | MLP mean | MLP worst |
|---|---|---|---|---|---|
| `bbrob/bb_core4_bw0p5` | backbone | **−26.3%** | +16.7% | +21.1% | +502.2% |
| `bbrob/bb_core8_bw1p5` | backbone | **−22.8%** | +10.1% | +25.2% | +285.5% |
| `drawgate/backbone` | backbone | **−24.8%** | +21.0% | +9.4% | +230.5% |
| `promo175/backbone` | backbone | **−26.3%** | +13.7% | +9.3% | +349.7% |
| `drawgate/nobackbone` | FLAT | **+2.6%** | +60.8% | +120.9% | +1686.1% |
| `promo175/nobackbone` | FLAT | **+2.4%** | +63.2% | +96.9% | +1611.5% |
| **backbone (20 cells)** | | **−25.1%** | | +16.3% | |
| **FLAT (10 cells)** | | **+2.5%** | | +108.9% | |

**Read this carefully, because it splits the two results apart:**

1. **The GNN's *latency* advantage is a network-contention phenomenon.** On backbone
   cells it averages −25.1% vs Knative; on flat cells it is **+2.5%, i.e. slightly
   WORSE than Knative**. The headline per-draw figures (best draw −27.2%) are carried
   almost entirely by the 20 backbone cells. Remove finite link bandwidth and the
   latency edge disappears. This is the first direct evidence that the
   `feat/network-contention-v1` environment change is what the graph model exploits —
   consistent with route A's finding that coupling alone is insufficient and contention
   is required.
2. **The *reliability* result is NOT the same phenomenon and does not need the backbone.**
   The MLP is at its most catastrophic on the FLAT cells (+108.9% mean, worst
   **+1686%**), where the GNN sits near parity. So the registered severe-collapse claim
   holds across both halves, while the latency margin does not.

**Distribution shape** (16 draws each, mean margin over all 30 cells): GNN best −27.2%,
median −18.1%, worst −1.2%, **16/16 below Knative**. MLP best −29.2%, median +19.2%,
worst **+328.8%**, only **7/16** below Knative. The MLP's *best* draws edge out the GNN's
best — its ceiling is not the problem, its floor is. "A lottery versus a tight
distribution" is the accurate summary, not "the graph model is faster".

**Consequences for Phases 2/3, and one new control:**
- The latency question should be asked **on backbone cells**, where the effect lives;
  a flat-cell latency comparison is measuring a regime with no edge to find.
- **Untested confound, now the biggest threat to the headline:** GNN and MLP differ in
  architecture, capacity and regularisation, not only in graph-awareness. No
  message-passing ablation exists in the record (`gnn_necessity_ablation.py` burned a job
  and never live-gated; `topology_transfer_v1` FAILED all arms). `GNN_DISABLE_MESSAGE_PASSING`
  exists and the Phase 1 pin table already requires it unset, so the harness is ready.
  Recommend running it against these same 30 cells, pre-registered, **before** the Phase 3
  spend — if MP-off is equally reliable, the claim must be reworded away from "graph-aware".

### 2026-08-29 — Phase 1 claim REWORDED (required by mp_ablation_v1's registered null)

`mp_ablation_v1` returned `NO_DIFFERENCE_DETECTED` (primary p = 0.05066), and its
registration fixed the consequence in advance: **the Phase 1 claim is reworded away from
"graph-aware".** The statistic, the data and the PASS are untouched — only the causal
attribution changes.

- **Was (attribution unsupported):** "the *graph-aware* model's severe-collapse burden is
  stochastically smaller than the pointwise MLP's."
- **Now:** "across seeded draws, **the GNN model class** — a per-entity encoder plus a
  masked-softmax `EdgeScorer` over candidate placements — has a severe-collapse burden
  stochastically smaller than the pointwise MLP's (rank-sum p = 0.00143 at +50%; clean
  16/16 at +100%). **The message-passing channel is not what produces this**: disabling it
  leaves the edge intact and, directionally, improves it. The credit belongs to the scoring
  and decode architecture."

Any write-up of Phase 1 must report `mp_ablation_v1` alongside it. Reporting the reliability
result without the control would assert exactly the attribution the control failed to find.

### Training-venue audit — the pin escape changed nothing, proven bitwise (2026-08-31)

Found while building `link_mp_v1`: the shared training wrapper's `PROJECT_ROOT` default
`cd`'s to the main checkout, so the Phase 1 seeds 9-16 draws (job 719818) actually
trained at main HEAD (~`2c49fc4`), not at the bannered pin `c08aa7e` — the sbatch's
`[PIN]` assertion did not bind the venue (gate-tools 2026-08-30 row; gates were never
affected, their runner self-anchors and their provenance always recorded the true pin).

Settled empirically rather than by code reading: venuecheck job **728341** retrained
seed 9 in the c08aa7e worktree with `PROJECT_ROOT` actually bound, on the same node
class as the original (l40s), deterministic algorithms on. Result: **bitwise identical
across all 31 tensors** vs the shipped `gnn-draw-s9.pt`. The ~5,000 route_b lines
between c08aa7e and the training-day HEAD are flag-gated and inert at default flags —
now measured, not assumed. **Phase 1's draws, its PASS, and the seeds 1-8/9-16
exchangeability claim stand exactly as constructed.** The audit checkpoint
(`gnn-draw-s9-venuecheck.pt`) is deleted after comparison; this entry is its record.

### 2026-09-01 — Phase 2 (P3 horizon pilot): build complete, implementation decisions fixed BEFORE any pilot data

The registration (program_verdict_v1, thresholds fixed 2026-08-24) left the mechanics of
"continue trace arrivals for a ~h-second horizon" unpinned. Fixed now, before the
calibration or array produce a single number:

- **Label = the horizon return**: total RTT over batch AND horizon tasks
  (`rtt_from_stats` unmodified over the extended workload). A batch placement that
  ignites queue runaway pays for it in the arrivals it damages. M4 unmodified, per
  registration.
- **Follow-on policy = the determined scheduler's `(-1,-1)` auto-resolve** (least-loaded
  valid replica) — the same fixed shortest-queue-style rule in every combo, reacting to
  the state the batch placement created. No ML policy inside the label.
- **Replica set = frozen at capture state for the horizon** (no autoscaling inside the
  mini co-sim). Defensible at h ≤ 10 s; stated as a scope limit of any conclusion.
- **Horizon window = (t_snap, t_snap + h], timestamps shifted** so arrivals land 10 ms
  after the batch at t=0 (`slice_horizon_events`; boundary semantics under test).
- **Capture extension**: snapshots now record `replicas_by_type` — the full per-type
  replica state — in BOTH capture copies (`live_audit.py` and the knative_network_batch
  scheduler's own). Measured necessity, not taste: with batch-candidates-only seeding,
  horizon arrivals from other client nodes found no valid replica and the mini co-sim
  HUNG (retry loop, no crash). A reachability pre-flight in the sweep now refuses such
  snapshots loudly. Pre-P3 bbrob snapshots preserved as `snapshots_pre_p3/` (they remain
  the valid t=0 / WS3 record).
- Guards: `scripts_cosim/test_p3_horizon_oracle.py` (15 tests: window boundaries, id
  mapping, h=0 bit-identity, pre-flight); full suite 390 green. End-to-end smoke on a
  fresh 30k-trace capture: 16/16 combos at h=2 s, 2,943 horizon arrivals, label spread
  3.8% across batch placements.

**In flight:** job 732878 re-captures the 10 bbrob backbone cells (WL150-100,
stride 31, cap 2000) with the new field; job 732888 (dependent) runs the registered
calibration — 3 snapshots × h ∈ {2, 5, 10}, top-k 4 = 256 combos, worst-case fabric
bb_core4_bw0p5. h is then fixed by the registered rule (largest affordable, ≥ 5 s for
axis terminality) and signed before the array is queued.

### 2026-09-01 — Phase 2: calibration COMPLETE, h REGISTERED at 10 s, pilot array queued

Registered calibration (jobs 732888 + 732918; 732888's h=10 stage OOM'd at 48 G with 16
workers and was rerun at 256 G — peak 71 GB; h=2/h=5 completed in the first job):

| h | snapshot t | combos | horizon arrivals | wall ms/combo (16 workers) |
|---|---|---|---|---|
| 2 | 9.1 / 62.1 / 114.1 s | 256 each | 2,508 / 6,091 / 2,005 | 150 / 368 / 141 |
| 5 | 9.1 / 62.1 / 114.1 s | 256 each | 7,266 / 15,373 / 2,849 | 453 / 884 / 168 |
| 10 | 9.1 / 62.1 / 114.1 s | 256 each | 19,370 / 30,474 / 2,849 | 1,011 / 1,760 / 179 |

- **(a) In-process: CONFIRMED.** Worst combo 1.76 s wall ≈ 28 CPU-s — far below the
  5.1 s-per-combo startup scenario that would have forced a re-scope.
- **(b) h = 10 s, by the registered rule.** ~16–28 CPU-s/combo ⇒ n=300 × 256 combos ≈
  **340–600 CPU-h**, inside the 2026-08-24 anchor (~300 CPU-h × 1.5–2 fabric overhead);
  the h ≥ 5 axis-terminality bound is satisfied with margin.
- **Truncated windows near trace end are KEPT.** The t=114.1 s snapshot's h=5 and h=10
  windows are identical (2,849 arrivals — trace ends at 118.7 s). The registration
  contains no exclusion, and dropping snapshots post-hoc is exactly the sampling-bias
  pattern the sweep docstring forbids; `n_horizon_events` per meta records the
  truncation for the analysis to see.

**Array queued:** `scripts_cosim/datalab/p3_horizon_pilot.sbatch` — 10 tasks
(2 backbone fabrics × 5 bbrob cells) × 30 snapshots = **300 snapshots, 76,800 mini
co-sims**, out to `simulation_data/snapshot_sweeps_p3_h10/`. Readout after completion:
`separability_diagnostic.py` (M4 unmodified) + the registered co-primaries with the
node-count and link repair controls.

### 2026-09-01 — Phase 2 RESULT: the registered P3 co-primary FIRED — but read the chaos caveat below before acting on it

Pilot job 732927: 10/10 tasks COMPLETED, **300/300 snapshots, 76,800 mini co-sims,
zero failed combos**. Readout `simulation_data/p3_h10_readout.json` (M4 unmodified).

**Registered verdict, by the 2026-08-24 rule (co-primaries are an OR):**

| endpoint | measured | bar | result |
|---|---|---|---|
| (a) median `additive_choice_regret_rel` | **0.00842** | > 0.02 | not met |
| (b) fraction of snapshots with regret > 2% | **21.67% (65/300)** | ≥ 15% | **MET** |
| node-count repair, affected stratum | median 0.000, mean 0.009 | < 0.5 | passes |
| link repair (best of k1/k2/excess), affected stratum | median 0.000, mean 0.016 | < 0.5 | passes |

Binomial: P(X ≥ 65 | p = 0.033, the t=0 base rate) = **1.5e-33**; against the 15% bar
itself, p = 0.0013; Wilson 95% CI on the rate [17.4%, 26.7%]. Regret distribution:
mean 1.45%, p90 3.15%, p99 7.77%, max 37.9%. **Both repair controls are essentially
zero** — whatever this is, it is neither node-occupancy-count nor link-load shaped,
the first mechanism in this program to escape both.

Per-cell, the firing mass is concentrated: the two `p50` cells and `core4/cell02_p35`
carry 14/30, 13/30, 14/30 above the bar, while three cells fire 0/30.

**Additive R² collapses to 0.0487 (median 0.0463)** — against ~0.988 for the t=0
`contention_v2` target. Read literally, ~95% of the placement-driven variance in the
horizon return is not pointwise-expressible.

**⚠ EXPLORATORY CHAOS CONTROL (post-hoc, not registered, and it may overturn the
reading above).** A deterministic discrete-event simulator can amplify a tiny placement
difference into large, arbitrary RTT differences 10 s later — deterministic but
encoding nothing about placement quality. That would present exactly as "regret" and as
a collapsed R², while being unlearnable. The test is horizon rank-stability: a genuinely
better placement should be better at every horizon. On the three calibration snapshots
(h ∈ {2,5,10}, 256 shared plans each) the Spearman rank correlations are
**+0.10/+0.03/−0.02, −0.02/−0.11/+0.06, +0.18/+0.09/+0.18** — i.e. near zero, and the
h=10 optimum ranks as poorly as **206/256** at h=2. That is the chaos signature, not the
structure signature.

**But that control is not yet decisive**: all three calibration snapshots sit in
`core4/cell01`, the *lowest*-signal cell in the pilot (0/30 above the bar), where every
difference is tiny and near-zero correlation is expected regardless. Job 733075 re-runs
the control at h = 2 and 5 on the `p50` cells that carry the firing mass. **Until it
returns, the horizon labels must not be used as Phase 3 DAgger targets** — training on
chaos would teach noise with a plausible-looking loss curve.

### 2026-09-01 — Phase 2 FINAL: P3 fired on the letter, but the signal is CHAOS — horizon labels are dead as supervised targets

The chaos control (job 733075) re-ran h = 2 and h = 5 on `cell04_p50` in both fabrics —
the cells carrying the firing mass — and compared each snapshot's full 256-plan **ranking**
against h = 10. Logic: a genuinely better placement is better at every horizon, so
structure ⇒ rank correlation near 1; chaotic amplification ⇒ rank correlation near 0.

| stratum | n | median ρ(h2,h10) | median ρ(h5,h10) | median rank of the h10-optimum at h5 |
|---|---|---|---|---|
| all compared | 60 | **−0.027** | **+0.004** | **120 / 256** |
| above the 2% bar | 27 | **−0.032** | **−0.009** | **140 / 256** |

Random ranking would give ρ = 0 and rank 128/256. **The measurement is
indistinguishable from random on both counts, and it is *worse* than random on the
snapshots that fired.** The top-regret snapshot (37.9%) has ρ = −0.036 / −0.108.

**Verdict: the registered co-primary (b) fired, and what it fired on is not learnable.**
The horizon return is dominated by deterministic chaos — a placement difference reshuffles
queue orderings seconds later and moves total RTT arbitrarily, encoding nothing about
placement quality. This also explains the two results that looked most exciting:
the collapsed additive R² (0.0487) is ~95% *unexplainable* rather than ~95% joint
structure, and both repair controls read ~0 because noise has no node-count or link-load
shape either. The pilot is sound; the label is not.

**Decisive form of the argument (state it this way, not as "the horizons disagree"):** if
the ranking at h = 10 does not survive a change of horizon length, the label is not a
stable property of the (state, action) pair at all — so it cannot generalise to a
different trace, seed, or horizon, and no model can fit it in a way that transfers.

**Consequences, in order:**
1. **Horizon labels must NOT be used as DAgger targets.** Training on them would fit
   noise with healthy-looking curves — the failure mode this program has been burned by
   before (offline regret −54% → live +9.9% *worse*).
2. **Phase 3 goes straight to policy gradient**, exactly as the registration's null branch
   specified. P3's upside is spent; no re-runs with tweaks.
3. **This is not a blocker for Phase 3 — it is a calibration of it.** Policy gradient
   estimates an *expected* return over many episodes; the chaos measured here is the
   per-episode variance it must average through. The registered **paired-seed
   common-random-numbers baseline is now load-bearing rather than a nicety**, and the
   pilot budget must be sized against a per-episode noise level of ~1.5% mean / 3.2% p90
   in total RTT. Recorded here so Phase 3 sizes its n from a measurement instead of a guess.
4. The residency-hold scaling pilot (~0.5 d, both controls pre-registered) remains the
   registered next step before P1 spend.

Artifacts: `simulation_data/p3_h10_readout.json` (300 snapshots),
`simulation_data/snapshot_sweeps_p3_h10/`, `simulation_data/snapshot_sweeps_p3_chaos/`.

### 2026-09-01 — Phase 2b: the residency-hold pilot is RETIRED on measurement (its mechanism cannot occur in co-sim)

`program_verdict_v1`'s sequence put a ~0.5-day residency-hold scaling pilot before any P1
spend: hold a node's compute slot across the whole residency (cold start + exec) instead
of exec alone, on the argument that "cold start at 38 s vs 0.024 s exec is a ~1,500×
longer hold, so mechanism #1's `nodeContentionTime ≡ 0.0` says nothing about it." The
paper step ran first, as registered, and a feasibility check killed it before any build:

**Cold start is exactly zero in the co-sim regime, everywhere.** Sampled across **all 16
collections, 328 datasets: 8 with any cold start**, all inside one 8-task `route_c`
collection, max total **0.206 s** over 8 tasks. `contention_v2/v3/v4/v5`, all three
`link_mp_v1` fabrics, `topo_transfer_v1`, all three `*_warmth_v2` corpora — **0.000 s**.
Including `regime_b_cold_burst_v1`, a corpus named for cold bursts.

**Mechanism (why it is structural, not a grid accident):** `system_state.replicas` is
keyed by task type, warmup runs each replica with its own type, and
`sandbox_is_warm` (`warmth.py:197-202`) returns True exactly when the platform's previous
task type equals this task's type. A dnn2 task can only be placed on a dnn2 replica, which
was warmed by dnn2 ⇒ warm by construction ⇒ `cold_start_duration = 0` on every enumerated
placement of every sweep.

**So the 38 s figure is a table entry, not an event:** it is `cnn`'s `coldStartDuration`
on `xavierDla` in `task-types-cnn.json`, never incurred in any co-sim dataset. Residency
therefore equals exec (~0.001–3.1 s, typically 0.024 s), which means **the proposed
residency hold IS the exec hold that already measured `nodeContentionTime ≡ 0.0`** — the
lever is a no-op on this regime, and its 1,500× premise does not hold.

**Making it fire would require a new corpus regime** in which a placement can land on a
sandbox warmed for a *different* type — i.e. shared/overlapping replica pools — which is
a physics-lever corpus design aimed at breaking additivity: the activity
`objective_pivot_v1` explicitly stopped at registration. Retired rather than rebuilt.
Cost: ~15 minutes of checking against the registered ~0.5 day.

## Phase 3 (P1 closed-loop) — REGISTRATION, signed 2026-09-01 before any arm exists

Branch `feat/closed-loop-p1`. Registered under the Phase 3 entry above; the three
deviations from that text are amendments, each with its measured reason.

**AMENDMENT A — warm start is `models/gnn-linkmp-lgon-s8.pt`, not `tempfix`.** The
Phase 3 text predates `link_mp_v1`. Its stated intent is "the floor is what we already
have"; what we now have is −38.0% vs Knative with zero severe collapses in 48 arms,
against the `tempfix`-era −25.1%. Warm-starting from the weaker checkpoint would build
the whole phase on a floor we have already beaten.

**AMENDMENT B — message passing runs over `core_v1`, not bipartite.** The Phase 3 text
says "MP bipartite (hard-stop on same-node edges without new physics)". `link_mp_v1`
measured old-graph MP as *harmful* (−4.50 pp, p = 0.00107) and `core_v1` as the repair
(+4.98 pp, p = 0.00459). The same-node hard-stop is untouched; this changes which graph
MP runs over, in the direction the evidence points. `lgon-s8` is already a `core_v1`
checkpoint, so this and Amendment A are the same decision.

**AMENDMENT C — the pilot is gated behind a sampling-feasibility probe (below) rather
than proceeding straight to the ~500 CPU-h budget.** Reason: the supervised evidence is
now uniformly negative (five mechanisms + P3 chaos), so RL's prior is lower than when
Phase 3 was written, and the cheapest way to find that out is a probe that cannot be
confused with a result.

### Registered arms (fixed now, before training)

| arm | what it is |
|---|---|
| **CL-GNN** | `lgon-s8` warm start, trained closed-loop on the live metric |
| **CL-MLP** | the MLP baseline in the identical loop (same objective, budget, seeds) |
| **Frozen-GNN** | `lgon-s8` served as-is — the floor Amendment A defines |
| **Frozen-MLP** | the promoted MLP served as-is |
| **Knative** | the reactive baseline, unchanged |

Every arm is paired on workload seed (**common random numbers**); the primary statistic
is the paired difference CL-GNN − Frozen-GNN, not either arm's absolute RTT.

### Registered kill criterion (unchanged from the Phase 3 text)

If, at the pilot budget, the paired-seed improvement over the frozen init is **≤ 0** at
the registered n, P1 freezes as **measured-negative** — which closes the last open path
and is itself an answer. No re-runs with tweaked hyperparameters.

### Increment 1 — the sampling-feasibility probe (registered read, before it is run)

A policy-gradient loop needs a *stochastic* policy; the served decode is pure argmax
(`seq_decode.decode_sequential_placement`), so sampling is new machinery and its cost is
unmeasured. The probe adds a temperature-sampled decode and measures, on the 30k-event
inner-loop trace over ≥ 3 backbone cells:

1. **Exploration cost** — sampled total RTT vs the same checkpoint's argmax RTT, per
   temperature τ ∈ {0.1, 0.3, 1.0}.
2. **Per-episode noise under CRN** — the paired standard deviation of sampled episodes at
   fixed τ and fixed trace, which is the quantity the episode budget must be sized from
   (the P3 chaos measurement, ~1.5% mean / 3.2% p90, is a *snapshot-horizon* number and
   does not transfer to full episodes; it is the reason this must be measured, not assumed).

**Registered reading, fixed before the probe runs.** Let `d(τ)` be sampled-vs-argmax
relative RTT and `sd` the paired episode standard deviation at the best τ:
- **GO** if some τ gives `d(τ) ≤ 0.10` **and** `sd ≤ 0.05` — exploration is affordable and
  the signal is separable from noise at a feasible n; size the pilot as
  `n ≥ (2 · sd · 2.8 / MDE)²` for a minimum detectable effect MDE = 0.03, and report that n.
- **NO-GO** if every τ gives `d(τ) > 0.25`, or `sd > 0.15`. Then the loop would spend its
  entire budget paying for exploration or averaging out its own noise, and P1 freezes as
  **measured-infeasible** — reported as such, not as a negative result about graph models.
- Anything between is **INDETERMINATE** and returns to the user with the numbers; it is
  not resolved by picking a τ after seeing the outcome.

### 2026-09-01 — Phase 3 Increment 1 RESULT: **GO**, with one registration defect to fix before the pilot

Job 733169 (branch `feat/closed-loop-p1`), 48 full episodes, 3 backbone cells ×
(argmax + 3 temperatures × 5 seeds), 30k-event inner-loop trace, warm start
`gnn-linkmp-lgon-s8`. Artifacts: `simulation_data/p3_sampling_probe/`.

| T | exploration cost d (mean) | d (median) | within-cell paired sd | bars |
|---|---|---|---|---|
| 0.1 | **+0.05%** | +0.25% | **0.0029** | pass (≤10%, ≤5%) |
| 0.3 | **+1.01%** | +1.14% | **0.0039** | pass |
| 1.0 | **+6.24%** | +5.79% | **0.0053** | pass |

**Verdict: GO.** All three temperatures clear the exploration bar with room, and the
paired noise is an order of magnitude below the 5% bar — the common-random-numbers
pairing works far better on full episodes than the P3 snapshot chaos suggested it might.

**Two things the raw verdict does not say, both recorded before acting on them:**

1. **A cheap temperature could have been cheap because it does nothing.** The registered
   read has no exploration term, so `d ≈ 0` is consistent with a policy that simply
   reproduces argmax and would teach a policy gradient nothing. Measured directly
   (new `explore_rate` on the episode trajectory): at **T = 0.1, 17.3% of the 29,998
   decisions differ from argmax** while costing +0.05% RTT. So the cheapest temperature
   is genuinely exploring, and the verdict survives the objection. Recorded because the
   objection was real and the answer was not obvious — not because it changed the result.
2. **On `cell02_p35`, sampling at T = 0.1 BEATS argmax by 0.75%** (all 5 seeds negative,
   −0.28% to −1.08%). A random perturbation of the served policy improving on it is
   direct evidence that the frozen argmax policy is not at a local optimum — the
   headroom P1 exists to capture.

**⚠ REGISTRATION DEFECT — the pilot size must not be taken from this run as computed.**
The registered formula `n ≥ (2·sd·2.8/MDE)²` with the measured sd returns **n ≥ 1**,
which is arithmetically correct and operationally absurd. Two flaws, both mine, both in
the *reading* rather than the measurement:
   - it has **no floor**, so an unusually small sd collapses the pilot to a single
     episode pair with no protection against a bad draw;
   - the sd it consumes is the spread of the *frozen* policy's sampled episodes, which
     is **not** the variance of the quantity the pilot actually tests (the paired
     difference between a *trained* arm and the frozen one, whose variance grows as the
     arms diverge).
   The measurement stands; the sizing rule does not. **Fixing it is an amendment to be
   signed before the pilot runs, not a number to pick now** — writing it down here, with
   the flaw named, so the next step starts from an honest record rather than from
   `n ≥ 1`.

**Next:** sign the sizing amendment (with a floor and a divergence-aware variance), then
build the two-pass REINFORCE loop (pass 1 samples under `no_grad`, pass 2 replays a
uniform subsample of decisions with grad, rescaled by T/k — unbiased) against the
registered arms and kill criterion.
