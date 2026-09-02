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
   (new `explore_rate` on the episode trajectory, one episode per T on `cell01`,
   29,998 decisions each):

   | T | explore_rate | mean log-prob | d |
   |---|---|---|---|
   | 0.1 | **0.173** | -0.368 | +0.05% |
   | 0.3 | **0.240** | -0.539 | +1.01% |
   | 1.0 | **0.356** | -0.864 | +6.24% |

   All three explore substantially. Exploration rises smoothly with T while cost rises
   ~120x faster across the same range, so **T = 0.1 buys 17.3% exploration at
   essentially no RTT cost** and the cheapest temperature is not a degenerate argmax
   clone. The verdict survives the objection. Recorded because the
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

### 2026-09-02 — Phase 3 Increment 2: the closed-loop trainer is BUILT (no result yet)

Branch `feat/closed-loop-p1`. This entry records the machinery and what building it
measured. **No arm has been trained to a result and the pilot budget is still unsigned**
— see the sizing defect above, which this increment does not resolve.

**Files.** `scripts_cosim/closed_loop/{episode,adapters,train_closed_loop}.py`,
`scripts_cosim/datalab/p3_closed_loop_pilot.sbatch`, and the guards in
`scripts_cosim/test_closed_loop_{gradient,episode}.py`. Suite: 437 passing.

**Estimator.** REINFORCE with a **self-critical baseline**: `A = (RTT_greedy − RTT_sampled)
/ RTT_greedy`, where `RTT_greedy` is the current policy's own argmax episode on the same
cell and trace. Argmax is deterministic given (weights, cell, trace), so one baseline
episode serves all S sampled ones — the loop costs `1+S` episodes per cell per step, not
`2S`. Advantages are **not** mean-centred, deliberately: the self-critical baseline
already puts a meaningful zero at "sampling matched greedy", and subtracting the step
mean would make a step in which *every* episode beat greedy teach nothing.

**Two passes, because an episode is ~25k decode batches and the autograd graph does not
fit.** Pass 1 runs the episode as a subprocess of `executesimulation.py` under `no_grad`
and reservoir-samples k decode batches (inputs + chosen indices) to disk; pass 2 replays
those k with grad and forms `(N/k)·Σ log π(a|s)`. Algorithm R gives every batch inclusion
probability exactly `k/N`, which is what makes the rescale unbiased — measured over 3,000
trials, not asserted.

**Both arms share one loop.** `MLPBatchScheduler → XGBoostBatchScheduler → GNNScheduler`,
so the sampled decode, the episode trajectory and the replay hook were already inherited;
only the replay *payload* differs (a PyG `Data` for the GNN, the serving matrix plus its
row spans for the MLP). CL-GNN and CL-MLP therefore run identical code, which the
registration's "same loop, same objective, same budget" requires.

**Four things this increment measured, all of which could have gone the other way:**

1. **Pass 2 reproduces pass 1 exactly.** Max per-decision log-prob replay error on the
   real `lgon-s8` checkpoint: **1.1e-16** — float64 machine epsilon, i.e. bit-exact.
   Gradient reaches **40/40** parameters, all finite. The trainer re-checks this every
   step and aborts on drift, because a stored payload that stops reconstructing its
   decode would silently point the gradient at a distribution the simulator never
   sampled from, and nothing downstream would notice.
2. **Episodes are reproducible and concurrency-safe.** The same `--seed` gives
   bit-identical episode returns across runs, and `--episode-workers 4` reproduces the
   sequential numbers exactly. This is what the registered CRN pairing rests on.
3. **The headroom the probe found is real and reproducible.** On `cell02_p35`, sampled
   episodes beat the frozen greedy policy on **both** seeds (+0.27%, +1.02%) — the same
   direction Increment 1 saw on that cell. `cell01_p25` goes the other way (−0.42%,
   −0.72%). The policy is not at a local optimum, and the variation is across cells, not
   noise within one.
4. **Cost.** ~72 s per episode on the 30k-event inner-loop trace. 3 cells × (1+4) = 15
   episodes/step ⇒ ~50 min wall for 20 steps at 8 workers. The pilot is cheap; the
   binding constraint is the unsigned budget, not compute.

**Two defects found and fixed while building, recorded because both would have produced a
plausible-looking training curve rather than an error:**

- The advantage clip was specified in raw relative-RTT units while standardisation
  rescaled the advantages into z-units, so at small S *every* episode pinned to the clip
  boundary and the step carried only the sign of the advantage, not its size.
  Standardisation is now off by default and the trainer **raises** if all advantages clip.
- `load_policy` demanded a `.contract.json` sidecar for both arms. **No MLP checkpoint in
  the tree has one** — the MLP's contract lives inside the `.pt` (`input_dim`,
  `inference_feature_layout`, `queue_feature_contract`), which is what
  `MLPBatchScheduler.set_models` reads. The sidecar rule as written would have blocked
  CL-MLP and Frozen-MLP outright. `require_contract` now enforces the *right* declaration
  per arm; the principle is unchanged — a checkpoint that cannot state its own contract is
  refused, never defaulted.

**One deliberate non-reuse.** `run_sampling_probe.py` is the registered Increment-1
instrument and stays frozen at the code that produced job 733169's GO verdict, so the
trainer's episodes do not import from it. `scripts_cosim/test_closed_loop_episode.py`
holds the two environments together instead: if either drifts, the GO verdict stops
covering the pilot and the test fails rather than nobody noticing.

**Next:** the sizing amendment (unchanged, still needs signing), then the shakedown run to
confirm the loop moves over 20 steps, then the pilot at the signed n and a live gate of
the resulting checkpoint against Frozen-GNN / Frozen-MLP / Knative under CRN.

### 2026-09-02 — AMENDMENT D to the Phase 3 registration (pilot sizing + tuning budget)

**Signed by the user 2026-09-02**, in response to a recommendation stated in full *before*
any arm was trained and before any pilot number existed. Authorising message: "do all 3
please and use recommended values", following an explicit enumeration of the variance
fix, the floor, and the tuning allowance. Nothing below was chosen after seeing a
closed-loop result — no arm has produced one.

This amendment changes **only** the three things named here. The kill criterion, the five
registered arms, the 3% MDE and the paired primary statistic are untouched; an amendment
is a tempting moment to soften them and it is deliberately not taken.

**D1 — the sizing rule was sizing on the wrong random variable.** The registered rule
`n >= (2·sd·2.8/MDE)²` took `sd` from Increment 1: the paired standard deviation of the
**frozen** policy replaying one trace at different sampling seeds. That is *evaluation*
noise, it is genuinely tiny (0.0029–0.0053), and it is not the variance that decides
whether a result replicates. The replication unit for a training claim is the **training
run** — same settings, different seed, different final policy — and that variance is, for
this loop, **unmeasured**. Substituting evaluation noise for training-run noise is what
produced `n >= 1`. The rule is therefore restated as: **`n` counts paired training seeds,
and `sd` must be the across-training-run standard deviation of the gate statistic.**

**D2 — floor of 16 paired training seeds per trained arm.** Rationale, in order of
weight: (a) this program's own precedent — the Phase 1 draw study, `mp_ablation_v1` and
`link_mp_v1` all ran 16 paired seeds, and the CL arms are compared against Frozen
distributions characterised at that n, so a smaller CL arm would be an unpaired comparison
wearing a paired test's clothes; (b) the deep-RL replication literature (Henderson et al.
2018; Agarwal et al. 2021) puts 3–5 runs squarely in the range where identical algorithms
split into apparently-significant groups; (c) it costs ~190 CPU-h against the registered
~500 CPU-h anchor, so the floor is affordable and compute is not the binding constraint.
The floor binds regardless of what D1's formula returns once the variance is measured; if
the formula returns more than 16, the formula wins.

**D3 — a pre-registered tuning budget, on cells disjoint from the gate.** The kill
criterion is deliberately unforgiving: paired improvement ≤ 0 freezes P1 as
measured-negative with no re-runs. Pointed at a loop whose learning rate is currently an
unexamined guess, that converts a tuning failure into "closed-loop RL does not work here"
and forbids the re-run that would catch it — a false negative closing the program's last
open path. The method therefore gets one fair shot first, fixed now:

| | cells | used for |
|---|---|---|
| **train** | `bbrob_bb_core4_bw0p5` cell01, cell02, cell04 | the loop's episodes |
| **dev** | `bbrob_bb_core4_bw0p5` cell03, cell05 | hyperparameter selection ONLY — never reported as a gate |
| **gate** | `bbrob_bb_core8_bw1p5`, all 5 cells | the registered verdict; unseen fabric and unseen cells |

Tuning grid, fixed before it runs: `lr ∈ {1e-6, 1e-5, 1e-4} × T ∈ {0.1, 0.3}`, one seed
each, 20 steps, `reservoir_k = 64`, `episodes_per_cell = 4`. Selection statistic: greedy
total RTT on the two dev cells, lower is better. **The selected configuration is frozen
before the pilot begins and is not revisited**; a second tuning pass after seeing pilot or
gate numbers is exactly the fishing the kill criterion exists to prevent.

**Unchanged and restated so the amendment cannot be read as loosening them:** the verdict
comes from the held-out live gate, never the training curve (this program has already been
caught once by a result that fired every registered bar and turned out to be chaos); the
primary statistic remains the paired CL-minus-Frozen difference under common random
numbers, tested by exact Wilcoxon signed-rank; and `≤ 0 at the registered n` still freezes
P1 as measured-negative.

### 2026-09-02 — Phase 3 shakedown: **the loop moves** (non-inferential, by design)

Job 733519, 59m26s, 20 steps × 3 train cells × (1 greedy + 4 sampled) = 300 episodes,
`lgon-s8` warm start, T = 0.1, lr = 1e-5, `reservoir_k` = 64, seed 1.

**This produces no claim and cannot.** It ran on the training cells with an untuned
learning rate at n = 1 seed, which is below the Amendment D2 floor by fifteen. Its only
question was whether the machinery learns at all, because if it does not, every sizing
question above is moot.

| | step 1 | step 10 | step 20 |
|---|---|---|---|
| greedy total RTT (mean over train cells) | 12,340,038 | 12,166,685 | 12,193,201 |
| vs step 1 | — | **+1.41%** | **+1.19%** |

**It learns, and then it wobbles.** The curve rises monotonically for ten steps, peaks at
+1.41%, then oscillates between +0.6% and +1.2% for the remaining ten. That is ordinary
REINFORCE variance, and it is also the first evidence that lr = 1e-5 may be past the
useful point for this problem — which is exactly what the Amendment D3 grid exists to
settle, and is why the grid brackets it on both sides (1e-6, 1e-4).

**The correspondence check held for every step.** Max per-decision log-prob replay error
never exceeded **4.4e-16** across all 240 sampled episodes — pass 2 differentiates the
distribution pass 1 sampled from, continuously, not just in the unit tests.

**`frac_sampled_beat_greedy` falls from 0.33 to ~0.08 as training proceeds**, and mean
advantage drifts more negative. This is the expected signature of a sharpening policy —
as greedy improves, a temperature-perturbed version of it beats it less often — but it is
also the signature of a loop that has stopped discovering and is only reducing entropy.
The two are not distinguishable from this run, and the dev-cell selection in the tuning
sweep is what separates them: a policy that merely sharpened on the training cells does
not transfer, and a policy that learned something does.

**Note for anyone reading the training curve as a result: do not.** The registered verdict
is the paired CL-minus-Frozen difference on the held-out `bb_core8_bw1p5` gate over 16
training seeds. A +1.19% training-cell curve at one seed is consistent with a real effect
and equally consistent with one lucky draw — this program has already been caught once by
a measurement that fired every registered bar and turned out to be chaos.

### 2026-09-02 — Phase 3 tuning (Amendment D3), GNN arm: **lr = 1e-4, T = 0.1 selected**

Array 733566, six configurations, ~1h05–1h15 each, all COMPLETED. Trained on
cell01/02/04 of `bbrob_bb_core4_bw0p5`; selected on the **dev** cells cell03/05, which no
arm trained on and which the gate never sees.

| lr | T | train-cell change (step 20) | **dev-cell vs frozen** |
|---|---|---|---|
| 1e-4 | **0.1** | +10.63% | **+8.50%** |
| 1e-4 | 0.3 | +5.40% | +5.55% |
| 1e-5 | 0.3 | +1.43% | +0.74% |
| 1e-5 | 0.1 | +0.63% | +0.14% |
| 1e-6 | 0.3 | +0.36% | +0.07% |
| 1e-6 | 0.1 | +0.12% | −0.01% |

**Selected: `lr = 1e-4, T = 0.1`. Frozen; not revisited.**

**The selection is not a memorisation artefact, and that was the thing worth checking.**
Dev tracks train closely at every rung (+10.6/+8.5, +5.4/+5.6, +1.4/+0.7), so the ordering
on held-out cells is the ordering on training cells. The shakedown's open question —
whether the loop was learning or merely sharpening its own argmax — resolves as *learning*
at this configuration: sharpening does not transfer to cells the policy never trained on.

**Two limitations, recorded now rather than after they become convenient:**

1. **The optimum sits on the grid boundary.** lr = 1e-4 is the largest value tested and
   won by an order of magnitude over the next rung. Standard practice would extend the
   grid; Amendment D3 fixed it, and extending it now would be exactly the "second tuning
   pass" the amendment forbids. Proceeding at 1e-4 therefore likely **understates** what
   the loop can do, and a wider grid is a separate registered question, not a rescue.
2. **The winner had not converged at step 20** (+8.6% at step 18, +10.4% at 19, +10.6% at
   20 — still rising). The pilot nevertheless runs at 20 steps, because 20 steps is what
   the dev evaluation validated; running longer would put a configuration into the pilot
   that no held-out measurement has seen. The step budget, not the learning rate, is now
   the binding constraint on this loop.

**Why this stage was worth its hour.** At the shakedown's lr = 1e-5 the loop returns
+0.14% on dev — indistinguishable from nothing. Under the pre-Amendment-D plan the pilot
would have run there, returned a null, and the kill criterion would have frozen P1 as
measured-negative with no re-runs permitted. The registered verdict would have been a
statement about a learning rate, recorded as a statement about closed-loop RL.

**MLP arm tuning submitted (job 733631), same grid, same budget, same cells**, warm start
`batch_edge_mlp_full_corpus_siv1_dim22_batchcache_tempfix.pt`. Tuning only the GNN and
handing CL-MLP an inherited learning rate would bias the one comparison the program exists
to make; the Phase 3 registration's "identical loop, same objective, budget and seeds"
requires both arms get the same tuning treatment. (Caveat carried forward: that checkpoint
predates 2026-08-24, so like every MLP checkpoint in the tree it is an unreproducible
training draw — a property of the Frozen-MLP arm, not of this stage.)

### 2026-09-02 — Phase 3 tuning, MLP arm: **the loop gives CL-MLP nothing**

Array 733631, same grid, same budget, same train and dev cells as the GNN arm, warm start
`batch_edge_mlp_full_corpus_siv1_dim22_batchcache_tempfix.pt`.

| lr | T | dev vs frozen |
|---|---|---|
| 1e-6 | **0.1** | **+0.025%** |
| 1e-6 | 0.3 | +0.010% |
| 1e-5 | 0.1 | −0.023% |
| 1e-5 | 0.3 | −0.130% |
| 1e-4 | 0.1 | −0.203% |
| 1e-4 | 0.3 | −0.346% |

**Selected: `lr = 1e-6, T = 0.1`** — by the registered statistic (lowest dev RTT), which
for this arm is the rung that barely moves the weights at all. Every configuration that
actually trained the MLP made it **worse**, monotonically in the learning rate.

**Against the GNN arm on the identical protocol: +8.50% vs +0.025%.** The same loop, the
same objective, the same cells, the same episode budget, the same grid, each arm at its
own dev optimum.

**Read this as a lead, not a result.** It is n = 1 seed per configuration at the selection
stage. The registered verdict is the 16-seed paired gate on the unseen `bb_core8_bw1p5`
fabric, and this program has been caught once already by a measurement that fired every
bar and turned out to be chaos.

**The alternative explanation, checked rather than waved away.** A grid tuned for a
52,801-parameter GNN need not reach a 22→64→1 MLP's useful range. But the MLP's response
is *monotone decreasing in lr* and its optimum sits at the smallest rung, so the trend
points toward lr → 0 — "do not train" — not toward a larger value the grid missed.
Extending upward would make it worse; extending downward asymptotes to the frozen
checkpoint. The grid range is not the constraint here. What cannot be excluded from n = 1
is that this particular MLP draw is unrepresentative, which is exactly what the 16-seed
pilot measures.

**Each arm runs the pilot at its own dev optimum, not a shared learning rate.** The
registration asks for the same loop, objective, budget and seeds — not the same
hyperparameters — and handing CL-MLP a value selected for the GNN would make the headline
comparison a statement about tuning transfer. Both arms get their best shot; that is what
makes the difference between them attributable to the model class.

**Pilot launched: job 733648**, 32 runs (16 paired seeds × 2 arms), `%8` concurrency,
~4 h. Hyperparameters frozen at the values above and not revisited.

### 2026-09-02 — Phase 3 GATE: **NOT ESTABLISHED** on the primary; the kill criterion does not fire

Job 734064, 13m21s. Held-out `bbrob_bb_core8_bw1p5`, all 5 cells — a different network
fabric from the one the pilot trained on and different cells from the dev pair that
selected the hyperparameters. 35 arms × 5 cells = 175 greedy episodes; every arm argmax,
the configuration the live gates serve.

**Absolute standings on the unseen fabric** (mean total RTT over the 5 cells):

| arm | mean total RTT | vs Knative |
|---|---|---|
| Frozen-GNN (`lgon-s8`) | **4,945,399** | **−44.8%** |
| Frozen-MLP (`fc_siv1_dim22_tempfix`) | 6,340,642 | −29.2% |
| Knative | 8,953,094 | — |

**PRIMARY — CL-GNN minus Frozen-GNN, paired over 16 training seeds:**

| | |
|---|---|
| mean | **+0.819%** |
| median | **+0.274%** |
| seeds better than frozen | **8/16** |
| exact Wilcoxon one-sided | **p = 0.3718** (α = 0.05) |
| per-seed spread | −11.50% … +10.84%, **sd = 5.84 pp** |

**Verdict: NOT ESTABLISHED.** The registered kill criterion fires on `improvement ≤ 0`;
the median is **+0.27%**, so **it does not fire and P1 does not freeze.** Neither does
anything get claimed. The registration's binary did not anticipate "positive but not
significant", and this is that: an indeterminate outcome, recorded as one.

**SECONDARY — CL-MLP minus Frozen-MLP:** mean **−0.053%**, median −0.059%, **1/16** better,
p ≈ 1.0, **sd = 0.0325 pp**. Negative for that arm. It is *not* the primary statistic and
no kill criterion attaches to it (see the tool correction below).

**The one thing here that IS established, and it is not the thing we set out to test.**
The same loop, objective, budget, grid and seeds moves the two model classes by amounts
differing by a factor of **180 in standard deviation** — 5.84 pp for the GNN against
0.033 pp for the MLP. The GNN result is under-powered; the *asymmetry* is not, because the
MLP's variance is small enough to resolve at n = 16 with room to spare. **Closed-loop
training can move the graph model and cannot move the pointwise one.** That is a
well-powered finding about trainability, and it is emphatically **not** a latency claim:
"can be moved" is not "is improved", and at lr = 1e-4 the GNN moves the wrong way in 8
cases out of 16.

**Achieved power, now that the variance Amendment D1 called unmeasured is measured.**
Across-run sd = 5.84 pp, so detecting the registered 3% MDE needs **n ≥ 119 paired seeds**
— roughly 7.4× the floor D2 set and ~700 CPU-h against the ~500 CPU-h anchor. The 16-seed
gate could not have resolved a 3% effect at this instability even if one were there. D2's
floor was right to exist and was still too small; sizing it required the measurement only
the pilot could produce, which is the circularity D1 named and could not escape.

**Two defects in our own instruments, both found by this run:**

1. **`analyze_gate.py` printed the program-closing kill language for a secondary arm.**
   The CL-MLP readout said "P1 freezes as measured-negative" because the script applied the
   criterion to whatever arm it was pointed at. The registration defines the kill on the
   paired CL-GNN difference *and nothing else*. Fixed: the kill now requires an explicit
   `--primary`, and a negative secondary arm is reported as exactly that. Left unfixed,
   a script's default would have entered the record as a verdict nobody signed.
2. **Amendment D3's tuning stage used one seed per configuration** — which, at the sd this
   run measured, cannot distinguish a configuration from a draw. `lr = 1e-4` won the grid
   on seed 1's +10.6%, the third-best of the sixteen draws that seed later turned out to
   sit among. The pilot then ran at a learning rate selected by noise. This is a defect in
   an amendment **I drafted and recommended**, and it is named here rather than discovered
   later: a tuning stage must be powered like the comparison it feeds.

**Status: the last open path to the latency claim is neither closed nor confirmed.**
Resolving it needs either n ≈ 119 at this instability (over the registered anchor), or a
configuration stable enough to shrink the spread — and the second requires a re-tune,
which is a new registration, not a re-run. **That choice is not taken here.**
