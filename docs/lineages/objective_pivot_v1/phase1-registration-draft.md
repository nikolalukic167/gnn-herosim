# objective_pivot_v1 — PHASE 1 REGISTRATION — SIGNED OFF 2026-08-28

> Drafted 2026-08-28; §Venue & code identity added the same day at the user's condition
> ("make sure they run on the same infra as before"); signed off with that section
> binding. The sign-off line at the bottom is the authority.

## The question

Is the GNN's live reliability edge over the pointwise MLP — direction-unambiguous in
`gnn_draw_study_v1` at every threshold, established at none — real at α = 0.05 when
measured the only way `p5b_draw_study` licenses: as a **draw-distribution** comparison?

**The claim under test, worded now so it cannot drift:** *"Across seeded training draws
on the 30 backbone gate cells, the GNN's per-draw severe-collapse burden (cells at
`total_rtt` ≥ +50% vs same-cell Knative) is stochastically smaller than the MLP's."*
Severe-collapse reliability. **Not** "the GNN never collapses" (FALSIFIED,
`gnn_draw_study_v1`), and **not** a claim at the +30% line (see §Scope).

## Design

- **+8 seeded GNN draws**: seeds 9–16 via `NEAR_RTT_TRAIN_SEED`, deployed config
  (`graphs_cache_full_corpus_siv1_dim14`), split fixed at `random_state=42`,
  `NEAR_RTT_MP_RESIDUAL` / `GNN_MP_NODE_EDGES` asserted unset — byte-for-byte the
  `gnn_draw_study_v1` recipe. Existing seeded draws 1–8 stay valid and are included;
  nothing is re-run.
- **Gate**: same 30 backbone cells, same frozen same-cell Knative arms, same collapse
  rule. 8 × 30 = 240 new gate tasks.
- **MLP comparison group: FROZEN** at the 16 seeded `p5b_draw_study` draws already in
  `gate_stats_summary.json`, read at scoring time, never typed in, never re-selected.
- **Entry points**: extend `datalab/gnn_draw_study_{train,gate}.sbatch` for seeds 9–16;
  new scorer `scripts_cosim/important/score_objective_pivot_phase1.py` (a sibling of
  `score_gnn_draw_study.py`, not an edit — its thresholds are frozen history).
- **Preconditions**: `tests/test_trainer_determinism.py` green on the GNN trainer before
  seed 9 is queued; PARITY chain before any cross-venue read; §Venue & code identity
  below satisfied on every new arm.

## Venue & code identity — the "same infra as before" guarantee (binding)

**What the frozen arms recorded** (verified 2026-08-28 by reading
`run_provenance` from the cluster result JSONs, e.g.
`drawgate_backbone_gnndraws1/results/cell01_p25_s9001_s0_gnn.json`, SLURM array
712389/712390): full code provenance existed for every arm in the comparison group —
the instrumentation landed 2026-08-21 (`374accc`), four days before the draw-study jobs.
The recorded identity is:

| field | recorded value |
|---|---|
| `code.commit` | `c08aa7ee140fd51e3d384f97df3f31b126df96ab` (`c08aa7e`) |
| `code.branch` / `code.dirty` | `feat/network-contention-v1` / **false** (empty-diff sha `e3b0c44…`) |
| `env.INFERENCE_FEATURE_LAYOUT` | `dim22` |
| `queue_feature_contract` | `scale_invariant_v1` |
| `warmth_physics` | `node_disk_v2` |
| `env.GNN_DECODE_MODE` / batch | `argmax` / `GNN_BATCH_SIZE=4`, `GNN_BATCH_TIMEOUT=0.002` |
| `env.HEROSIM_GNN_DEVICE` | `cpu` |
| `env.TOPOLOGY_FEATURE_CONTRACT` | `src_index_v0` |
| `env.GNN_MP_NODE_EDGES` / residual | unset |

**Why pinning is required, not optional:** `git diff --stat c08aa7e..HEAD -- src/` is
~5,000 changed lines (route_b work in `infrastructure.py`, `orchestrator.py`,
`simulation.py`, `seq_decode.py`, `gnn_model.py`, `feature_builder.py`,
`train_near_rtt.py`). Most of it is opt-in, but "opt-in and therefore a no-op on this
path" is an assumption of exactly the class this repo's record rejects (job 708549:
an uncommitted diff was 23.3% of total_rtt; lesson of 2026-08-18: *build the baseline
arm from the same tree as the treatment arm* — here the baseline arms exist at
`c08aa7e`, so the treatment draws must come from `c08aa7e`).

**The rule:**
1. **Every new training run (seeds 9–16) and every new gate run executes at commit
   `c08aa7e` in a clean tree** — a dedicated pinned checkout on datalab (e.g.
   `git worktree add`/clone at `c08aa7e`, never a `checkout` that moves the main repo's
   branch), micromamba `gnn`, `HEROSIM_PY=python3`, `datalab-pitfalls` preflight.
2. **The scorer asserts, per new arm, before computing anything:**
   `code.commit == c08aa7e…`, `code.dirty == false`, and every env row of the table
   above equal to the recorded value. **Any mismatch ⇒ that arm is VOID** (re-run it;
   never scored, never dropped silently).
3. `env_fingerprint` / `python_env` are recorded and diffed but a mismatch there is a
   flagged warning, not a VOID — library versions are MEASURED to contribute exactly
   0.0 to GNN logits (PARITY.md); code identity is the axis that moves numbers.
4. The **scorer itself** (`score_objective_pivot_phase1.py`) runs at HEAD — it is
   arithmetic on recorded numbers, not simulation — and lives in the main tree.
5. Post-Phase-1 work (P3, P1) returns to HEAD; the pin is scoped to this gate's
   exchangeability requirement, not adopted as a program-wide freeze.

## Statistics (fixed at sign-off)

- **Primary**: one-sided Mann-Whitney rank-sum (midranks, exact/permutation) on the
  per-draw collapse-**count** vectors at the **+50%** threshold, GNN n=16 vs MLP n=16,
  α = 0.05.
- **Sensitivity (must also clear for the verdict)**: same statistic at **+100%**.
- **Secondary (its own claim, reported separately)**: sign test on the 16 GNN draws'
  mean `total_rtt` margin vs same-cell Knative; bar: ≥ 13/16 draws negative
  (p = 0.0106 under the null).
- **Descriptive only, no verdict role**: the +30% row (both statistics), the clean/
  unclean dichotomy at all thresholds, per-draw worst-cell magnitudes, Q1-style ranges.

## Why these instruments — power, computed before registration (2026-08-28)

Exact Fisher requirements at n=16 vs the frozen MLP group (validated against the three
p-values recorded in `gnn_draw_study_v1`): clearing needs ≥ 12/16 clean at +30% and
≥ 13/16 at +50% — i.e. at most one unclean draw among the 8 new ones at each line.
Monte-Carlo over the measured per-draw outcome triples:

| instrument | P(verdict clears) |
|---|---|
| dichotomy, all-three-thresholds clause (the `gnn_draw_study_v1` rule) at n=16 | **≈ 0.135** |
| rank-sum at +50% (bootstrap from measured count vectors) | **≈ 1.00** |
| rank-sum at +30% (conservative MLP floor: +50% counts) | **≈ 0.27** |

The dichotomy discards magnitude — a 26-cell draw scores like a 1-cell draw — and the
measured edge *is* magnitude (GNN worst draw 3/30 at +50% vs MLP worst 26/30). Re-running
it at any affordable n produces INDETERMINATE with high probability; registering it again
would spend 8 GPU trainings to learn nothing. The rank statistic is the correctly shaped
instrument for the distribution claim and is essentially fully powered at the +50% line.

## Scope — why +30% is out of the verdict, stated before the data

At +30% the GNN's own tail is real (measured counts 0, 8, 0, 0, 10, 2, 0, 0), and no
affordable n powers that line (≈ 0.27 at n=16 even against a conservative MLP floor).
The +30% row is therefore **pre-committed as a reported limitation**, in this file and in
any paper text: *the established claim, if the gate passes, covers severe collapse only;
at +30% the direction is consistent but unestablished.* Narrowing scope after a failed
broad verdict would be threshold-shopping; narrowing it **before** the new data, with the
old INDETERMINATE reported alongside, is a scoped claim. The distinction is this
paragraph existing now.

## Named selection hazards (the §6-style disclosure)

1. **The primary statistic was chosen after seeing the n=8 data.** The
   `gnn_draw_study_v1` addendum's post-hoc rank check (p = 0.0274 at +50%) is part of why
   rank-sum is proposed. Mitigations: the 8 new draws are untouched by that selection;
   the MLP group is frozen; the claim wording and scope are fixed in this file before any
   new training; the old registered dichotomy remains reported (descriptive) and its
   INDETERMINATE stands as the answer to the *original* broad question.
2. **The +50% scoping follows the same hazard** and carries the same mitigations; the
   +30% limitation sentence is mandatory in any publication of the result.
3. If the sign-off wants the fully conservative alternative instead — verdict computed on
   the 8 **new** draws alone (n=8 vs 16, rank-sum at +50%) — that option is priced here:
   bootstrap power ≈ high but below the n=16 read; it removes hazard 1 entirely at the
   cost of discarding half the evidence. Decide at sign-off, not after.

## Outcomes

- **PASS (primary + sensitivity clear):** the paper's reliability claim is written as
  registered above — distribution-shaped, severe-collapse-scoped, with the +30%
  limitation sentence. Phase 2 (P3 pilot) proceeds regardless.
- **FAIL:** the reliability claim is dropped; the program's publishable frame is P6
  (terminal separability negative + the reusable diagnostic) and the draw studies
  reported as-is. Phase 2/3 proceed regardless — they answer a different question.
- **VOID** (any arm trained with residual/node-edge flags set, determinism test not
  green, missing draws, scorer deviating from this file, or any §Venue & code identity
  mismatch): fix and re-run the affected arm; a VOID is not a FAIL.

## Cost

8 GPU trainings + 240 CPU gate tasks — the same shape as jobs 712381/712389, which
completed 240/240 with zero failures.

---

**Sign-off:** **SIGNED OFF 2026-08-28 (user, Nikola Lukic)**, conditional on §Venue &
code identity as written: the user's stated condition was *"make sure they run on the
same infra as before"*, and the section was verified against the cluster's recorded
provenance (commit `c08aa7e`, clean tree, job array 712389) and added **before** this
sign-off. Any new arm whose provenance mismatches that section is VOID. Statistic
choices adopted as drafted: rank-sum primary at +50%, +100% sensitivity, +30%
descriptive limitation, n=16 with the existing 8 draws included, MLP group frozen.
