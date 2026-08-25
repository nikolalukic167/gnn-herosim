# route_b_v1 — STAGE 2 PRE-REGISTRATION

**DRAFT FOR SIGN-OFF — written 2026-08-25, before any DAG cache, masked decoder, T1 feature
layout, trained model, fresh corpus, or gate script exists.** Nothing below is executed until
this document is signed off. On sign-off it is committed and a
`### route_b_v1 — STAGE 2 PRE-REGISTRATION` entry in `LINEAGES.md` records the registration
by reference to this file at its commit SHA. Until then, building the cache, the decoder, the
feature layout, or training anything is invalid work.

Stage 1 record: `LINEAGES.md:3752–4051` (PASS, commits 2c3ebbc → 4bf714a). Stage 1
established that node-memory contention stacked on route A's pairwise transfer term makes the
best pointwise surrogate wrong by >5% on 17.2% of Arm S datasets (Wilson CI [0.126, 0.229]),
that neither count-based repair closes it (median repair fraction 0.000), and that
competition without coupling (Arm B0) produces only decoder-shaped error. Stage 1 proved the
structure **exists**. Nothing yet shows a model can **learn** it.

---

## 1. The hypothesis

A GNN trained on route B corpora learns the joint (contention + coupling) structure well
enough that, decoding through the **same** constraint-aware masked decoder, it achieves lower
held-out regret than the strongest constraint-aware pointwise MLP — and the win is
attributable to **architecture** (what message passing computes), not to features the MLP was
denied, not to a decoder the MLP was denied, and not to a regime (B0) where pointwise scoring
is already sufficient.

The null this must beat is not "the MLP" — it is *pointwise scoring plus decoder state*.
Stage 1's k-integer caveat already showed a pointwise surrogate with the constraint's count
statistic pulls the firing fraction from 0.172 to 0.118: the strongest pointwise baseline
absorbs roughly a third of the effect before any training happens. The gate is powered
against what remains.

---

## 2. The feature-parity contract — the centre of this registration

The decoder is sequential and masked: at step *t* the partial assignment already encodes
where previously-placed tasks (including parents) went — which is precisely the stage-1
coupling. A pointwise model with partial-assignment features can therefore see the coupling
without message passing. What each arm is *allowed to see at each decode step* is the
experiment; it is fixed here, before the DAG cache is designed, because the cache decides it.

**Three information tiers:**

- **T0** — static per-candidate features only. No partial-assignment state.
  Concretely: the existing `dim25cr` layout (`src/policy/tabular/reduced_features.py`,
  `DIM25CR_FEATURE_DIM = 25`), unchanged.
- **T1** — T0 + partial-assignment state expressible pointwise, i.e. any function of
  *(this candidate, the partial assignment so far)*:
  per-node×type occupancy of the partial assignment, remaining node capacity, and the
  already-committed placements of THIS task's parents (parent node id / hop distance to the
  candidate).
- **T2** — the full graph: DAG structure + network graph via message passing, including
  information about **not-yet-placed** tasks and their candidate sets.

**The rule, registered verbatim:**

> Any feature available to the GNN that can be expressed as a function of (this candidate,
> the partial assignment) MUST be given to the T1 MLP arm. The GNN's advantage must come only
> from what message passing computes that no such function can. A DAG cache built before this
> contract is registered is invalid work.

**The registered T1 layout — `dim36crk` = dim25cr + 11 partial-state columns**, computed by a
single-source function (`partial_state_columns(...)` in
`src/policy/tabular/reduced_features.py`, same one-definition rule as
`candidate_relative_queue_columns`; the training extractor and
`MLPBatchScheduler`/offline-eval path must both import it — a second copy of the formula is
forbidden). Per (task, candidate) edge, given the partial assignment:

| # | column | why |
|---|---|---|
| 25–28 | per-type occupancy count on the candidate's node (4 types: dnn1/dnn2/rf/cnn) | the k-integer statistic, localized — the constraint's own sufficient statistic |
| 29 | total committed memory demand on the candidate's node / `cap_node(α)` | contention pressure |
| 30 | remaining capacity on the candidate's node **after** hypothetically placing this task, / `cap_node(α)` | the marginal the mask enforces, as a score-side signal |
| 31 | would-violate indicator (1 if this placement is infeasible under the mask) | redundant with the mask, but lets the score anticipate it |
| 32 | number of this task's parents already committed / number of parents | decode-order awareness |
| 33–34 | min and max hop distance from committed parents' nodes to the candidate's node (0 if no parent committed) | route A's distance term, pointwise-expressible given committed parents |
| 35 | Σ over committed parents of `n_hops × payload / bottleneck_bandwidth` on the parent→candidate route, normalized by the per-dataset max of that quantity | the pairwise transfer cost itself, made pointwise via decoder state |

Uncommitted parents contribute zeros (col 32 marks how much is missing). Under the
scarcity-pressure decode order parents are **not** guaranteed to precede children; this is
the same information the GNN's sequential decode has at that step, so parity holds by
construction.

**Stated plainly: T1 is not a pointwise baseline, and stage 1's result does not transfer to
it.** Columns 33–35 hand the MLP parent hop distance and the parent→candidate transfer cost —
route A's coupling term itself — and because those columns depend on other tasks' placements,
MLP(T1)'s plan-level score is **not separable**. Stage 2 therefore tests GNN vs an
**autoregressive-pointwise** baseline. Stage 1's `R_exact` measured a *separable* surrogate,
and its "median repair fraction 0.000" was measured with count columns only; nothing in the
stage-1 record says a T1-expressible surrogate cannot close the effect. That question is
answered *before* any build work by the §9a pre-probe zero, whose reading is registered there
before the number is looked at.

**Features that would make a GNN win a FEATURE result, named now** (each is
pointwise-expressible from (candidate, partial assignment) and therefore in T1 — if the
cache gives the GNN any of these and the MLP lacks the corresponding column, the comparison
is void): parent→candidate hop distance and payload-transfer estimates (cols 33–35);
node occupancy / remaining capacity (cols 25–31). **What is legitimately T2-only:** anything
involving not-yet-placed tasks — sibling/child candidate sets, aggregate demand still to be
placed, multi-hop network aggregates around candidates of other unplaced tasks. That is the
only room the architecture claim has, and it is exactly the "who must yield" computation
stage 1 showed matters.

**Registered outcome meanings (fixed now so no result can be reframed later):**

- GNN(T2) > MLP(T1): the architecture claim — the thing being tested.
- MLP(T1) ≈ GNN(T2), both > MLP(T0): pointwise-plus-decoder-state suffices; the coupling is
  real but reachable without a graph. **This is a GOOD, publishable result** (it says the
  field's pointwise schedulers need decoder-state features, not GNNs) and is registered as
  such — it cannot be read as an embarrassment.
- GNN(T2) > MLP(T1) only because the GNN saw a pointwise-expressible feature the MLP lacked:
  a FEATURE result, excluded by construction via the rule above; discovering a violation
  after the fact VOIDs the comparison.

---

## 3. Arms

All trained arms train on the same corpus, same labels, same α; all decode through the ONE
shared masked decoder (§4). Grouped argmax is not an arm (registered stage 1).

| arm | model | tier | draws | role |
|---|---|---|---|---|
| A1 | GNN (`train_near_rtt.py`, CE-only) | T2 | 8 | the claim arm |
| A2 | MLP `dim36crk` (`train_mlp_dim22_from_batch.py` + new flag) | T1 | 8 | the registered strongest pointwise baseline |
| A3 | MLP `dim25cr` | T0 | 8 | isolates the value of decoder-state features |
| A4 | exact-assignment decode of A2's scores | T1 | (reuses A2's 8 draws) | separates score-wrong from decoder-myopic — the B0 lesson |
| F1 | constrained sweep optimum | — | — | floor 0 by construction |
| F2 | min-marginal greedy (`greedy_masked_plan` on true marginals) | oracle | — | stage 1's `R_greedy`, context row |
| F3 | random-feasible (uniform over feasible sweep rows, seeded) | — | — | sanity floor |

**A4 definition** (the stage-1 registration said "exact-assignment decode arm on the MLP's
scores" without saying how; fixed here): for each feasible sweep row (plan), the plan-level
score is Σ over tasks of A2's edge score for (task, its placement in that plan), with each
edge's T1 columns computed from the partial assignment induced by the shared decoder's task
order applied to that plan; A4 picks the feasible-set argmin (argmax of score). Exhaustive —
mean 647 feasible rows at α=2.0 — no ILP/Hungarian; implemented against
`marginal_surrogate_regret`'s row-iteration machinery in
`scripts_cosim/score_route_b_contention.py`.

**Sensitivity rows, not arms, not the gate:** GNN with `GNN_DISABLE_MESSAGE_PASSING=1` at
eval (an eval-ablation of A1's checkpoints — honestly labelled as such, NOT a trained
T1-GNN); near-RTT ranking loss variant of A1; makespan objective; α=3.0.

**Loss:** CE-only primary (`regret-loss-weight: 0, ce-loss-weight: 1`) against tied-optimal
label sets (§5). Training goes through `experiments/route_b_stage2_*.yaml` +
`run_experiment.py` — no new `train_*.py`, ever. Seeds 1..8 per arm; wandb per repo rule.

**What a "draw" varies — registered, because today the two arms' seeds vary different
things:** MLP `--random-state` seeds initialisation AND the train/val split
(`split_by_parent_three_way`), while the GNN's `NEAR_RTT_TRAIN_SEED` seeds torch only — so
as shipped, A2's draw distribution would include split variance that A1's does not, and the
paired comparison would be confounded. Therefore: **one canonical-parent train/val split,
computed once (from seed 42), written to a split artifact consumed by every arm and every
draw** (build item B6). A draw varies **initialisation and batch order only**, in both
trainers. Per-arm randomness sources are stated in each checkpoint's sidecar/meta
(`split_artifact` hash included), and the determinism test covers the pinned-split path.

---

## 4. The shared masked decoder

One new mode in `src/policy/gnn/seq_decode.py::KNOWN_DECODE_MODES` (working name
`masked_scarcity`), inherited by the MLP automatically because `MLPBatchScheduler` uses
`GNNScheduler._decode_placements` verbatim — a single implementation both models plug scores
into, per the stage-1 registration.

**Semantics, pinned to the stage-1 reference implementation** so the acceptance check is
exact: the mode replicates `greedy_masked_plan`
(`scripts_cosim/score_route_b_contention.py:228`) — tasks committed in ascending
(best available score, task_id) order ("scarcity-pressure order" is hereby pinned to this,
the order stage 1 actually used); mask = no replica reuse AND no placement pushing a node
over remaining capacity under `cap_node(α)`; ties by placement id.

**Registered prohibitions:**
- `uniq_platform`'s silent relax-to-argmax path is **forbidden** in this mode. An infeasible
  completion is a **counted failure** (`no_feasible_rows`-style loud accounting in
  `GnnDecodeRunStats`), never a fallback to unmasked argmax.
- **Acceptance check (build item, VOID if it ever fails):** fed the true min-marginals, the
  new mode must reproduce `greedy_masked_plan` to 1e-9 on **every** stage-1 corpus dataset
  (204 Arm S + 204 Arm B0, α ∈ {2.0, 3.0}), verified through an extension of
  `verify_route_b_scorer_agreement.py` (which imports nothing from the scorer). The 13
  positive-control tests in `tests/test_route_b_positive_controls.py` must still pass after
  any decoder-adjacent edit.
- `src/policy/gnn_hetero/seq_decode.py` is a second decoder copy with no
  `KNOWN_DECODE_MODES` registry: **hetero is out of scope for stage 2**, and a static test
  asserts the new mode's name does not appear in `src/policy/gnn_hetero/` (drift guard).

---

## 5. Corpus, split, and labels

**Grid frozen:** `ROUTE_B_PILOT_V1_GRID` exactly as stage 1 registered it
(`scripts_cosim/generate_gnn_datasets_fast.py:281–321`) — 2 connection probabilities × 2
replica configs × 3 queue distributions = 12 datasets per seed. α ladder frozen at
{∞, 3.0, 2.0}, **α = 2.0 (tight) primary**, rtt primary objective. No `task-types.json`
edits; no episode-physics changes beyond route A's landed term.

**Seed blocks** (existing ranges 101–148, 201–214, 701–750, 801–850, 901–917 are avoided):

| block | seeds | datasets | role |
|---|---|---|---|
| TRAIN (Arm S) | 1001–1075 | 900 | training corpus; internal val split by canonical parent |
| HOLDOUT-P (Arm S) | 2001–2025 | 300 | **PRIMARY gate set — statistically unseen** |
| HOLDOUT-B0 (Arm B0) | 2001–2025 | 300 | specificity control (S-trained models evaluated on it) |
| ladder reserve (Arm S) | 2026–2042, 2043–2067 | +204, +300 | generated only if the power ladder escalates |

**Why the primary holdout is fresh:** stage 1's 204 Arm S datasets have known statistics, and
their firing stratum (the 35) was *identified by* `R_exact` measured on those same datasets —
gating on that member list would condition on a noisy outcome measured on the evaluation set
itself (guaranteed regression to the mean). Therefore:

- The **firing stratum is a definition, never a member list**: on any evaluation set it means
  "datasets with `R_exact > 5%` computed on that set at α=2.0, rtt" (stage 1's registered
  statistic, unchanged).
- The stage-1 204 become a **SECONDARY replication set**, always reported alongside the
  primary and always labelled *statistics-known*. If primary and secondary disagree, the
  disagreement is **reported, not reconciled**.
- All-datasets and firing-stratum views are reported on **both** sets, always both.

**Labels:** any-of-K tied-optimal sets — for each training dataset, the set of feasible plans
(α=2.0) within `rtt_eps = 1e-9` of the constrained optimum. CE loss treats any member as
correct (sum of probabilities over the tied set). `audit_label_provenance`
(`scripts_cosim/gnn_necessity_ablation.py:276`) gains a tie-tolerant mode; a deliberately
corrupted label must still make it fail (fail-loud check, part of the controls). The graph
cache carries the feasibility mask and the capacity map for the full α ladder in its
`metadata.json`/contract, and every checkpoint's sidecar (`.contract.json` for the GNN, the
inline-keys + `.pt.meta.json` pair for the MLP) declares the layout, the queue contract, and
the α it was trained under — one contract, sidecar rule; a sidecar-less checkpoint is not
evidence (stage-1 rule, unchanged).

---

## 6. Statistics and the power computation (shown, per D3b)

**Primary statistic.** Per (arm, draw, dataset): decode regret
`100 × (cost(decoded plan) − cost(constrained optimum)) / cost(constrained optimum)` at
α=2.0, rtt, decoding through the shared masked decoder over the enumerated feasible set.
Per dataset: **median over the arm's 8 draws** → `regret_i(arm)` (p5b rule: distributions
over draws, never a single draw; per-draw distributions are also reported in full).
Paired per-dataset differences, on the same datasets:

- `Δ_i = regret_i(A2) − regret_i(A1)` (strongest MLP minus GNN; positive = GNN better)
- `Δ4_i = regret_i(A4) − regret_i(A1)` (exact-decoded MLP minus GNN)
- `Δ3_i = regret_i(A3) − regret_i(A1)` (static MLP minus GNN)
- `ΔB0_i = regret_i(A2) − regret_i(A1)` computed on HOLDOUT-B0

Test: one-sided paired t (direction registered: GNN better), α=0.05, plus a 10,000-resample
paired bootstrap CI reported alongside (the t is the gate; the bootstrap is a check — if they
disagree materially that is reported).

**Power — a calibrate-then-freeze two-step (the α-ladder pattern), because both inputs are
currently estimates.** Script: `scripts_cosim/route_b_stage2_power.py`, committed with this
doc; **every number in this section comes out of that script**, including the effect table,
the compound-condition power, and the f=0 null false-pass row. Assumptions, stated:

- **Effect floor:** the MLP(T1) per-dataset floor is the **§9a T1-repaired residual
  distribution** (`r_exact_repaired_t1_pct`), measured before any corpus is generated. The
  kint distribution (mean 1.918, frac>5% 0.118) was the planning-draft floor and is obsolete
  the moment §9a runs — it is the floor of a count-augmented *separable* surrogate, not of
  T1. The GNN captures a fraction *f* of the T1-closable gap.
- **Noise:** the planning-draft σ = 2.5% was borrowed from B0's max surrogate-fitting
  residual under separable physics — an unrelated quantity. The quantity needed is the
  seed-to-seed variance of trained-model decode regret per dataset, which is unmeasured
  because nothing has been trained. **σ is therefore estimated from the §9 pre-probe**
  (expanded to 4 draws per arm for this purpose): per-dataset paired-difference σ across
  pre-probe draws, pooled over the smoke datasets.
- Holdout composition i.i.d. from the stage-1 empirical distribution, zeros included (power
  must come from the all-datasets view because the stratum is applied post hoc).

**The freeze:** n (holdout size) and M (materiality floor) are frozen from the recomputed
power table — §9a floor + §9-measured σ — **before HOLDOUT-P is generated**, and recorded in
the LINEAGES registration entry as an amendment in the registered direction (calibration,
not threshold revision; the PASS logic and all §8 rules are untouched by the freeze).
**Registered now:** if measured σ exceeds 2.5% by more than 1.5× (σ > 3.75%), n = 300 is
insufficient and the ladder **starts higher** (504 or 804 per the recomputed table) rather
than escalating after the fact. Planning-draft table (kint floor, σ = 2.5%, shown for scale
only — superseded by the freeze): n=204/300/504 give power 0.86/0.95/0.99 at f=0.3; compound
condition-1 false-pass under f=0 at n=300, floor 0.25%: 0.043.

**Provisional pending the freeze:** primary holdout n = 300 (25 seeds); materiality floor
**M = 0.25%** on the mean paired difference. **Power ladder, both axes:** datasets
300 → 504 → 804 (reserve seed blocks above), then draws 8 → 12 per arm; each escalation is
taken only on the straddle rule in §8, and exhausting the ladder is VOID-UNDERPOWERED,
never FAIL.

**Sensitivity rows only, never the gate:** makespan objective; α=3.0; firing-stratum views
(both sets); the MP-disabled eval-ablation; near-RTT loss variant; per-draw (unaggregated)
distributions; win/loss counts per dataset.

---

## 7. Positive controls, frozen

All frozen before any corpus scoring; **any control failure makes stage 2 runs VOID, not
FAIL**, and controls re-run after any scorer/decoder/feature edit.

1. **Decoder-identity control:** the §4 acceptance check (new mode ≡ `greedy_masked_plan` to
   1e-9 on all 408 stage-1 datasets × both binding α rungs, via the independent verifier).
2. **Separation control (rigged corpus):** a synthetic mini-corpus built from Control 2's
   3×3 matching-shaped cost structure (`tests/test_route_b_positive_controls.py`), where the
   optimum provably requires joint information. MLP(T1) trained to convergence on it must
   retain regret ≥ the closed-form pointwise bound (150% shape), and a GNN — or any joint
   scorer — must be able to reach 0. This proves the harness can detect the separation it
   claims to measure, in both directions.
3. **Label-audit control:** tie-tolerant `audit_label_provenance` passes on the smoke cache;
   a deliberately corrupted label makes it fail loudly.
4. **Trainer-determinism control:** both trainers bit-identical at a fixed seed
   (`tests/test_trainer_determinism.py`, extended to the new flag/layout).
5. **Independent gate recomputation:** the stage-2 gate statistics are recomputed from the
   per-dataset regret table by an extension of `verify_route_b_scorer_agreement.py` (no
   imports from the gate script); disagreement > 1e-9 ⇒ VOID.

---

## 8. THE GATE

Gate script: `scripts_cosim/score_route_b_stage2_gate.py`, **no threshold arguments** —
constants in-file, same discipline as `score_route_b_gate.py`. Everything below is evaluated
on HOLDOUT-P (n=300, α=2.0, rtt, 8-draw medians) unless stated.

**Conditions:**

1. **Architecture margin:** mean `Δ` > 0 with one-sided 95% CI excluding 0, AND point
   estimate ≥ 0.25%.
2. **Not decoder-shaped:** mean `Δ4` > 0 with one-sided 95% CI excluding 0 (the GNN beats the
   strongest MLP even under the MLP's exact decode).
3. **Specificity (B0):** NOT (mean `ΔB0` ≥ 0.25% with one-sided 95% CI excluding 0) — no
   material, significant GNN edge where pointwise scoring is sufficient.
4. **Learnability:** mean `Δ3` > 0 with one-sided 95% CI excluding 0 (the GNN beats the
   static MLP — something graph- or state-shaped was learned at all).
5. **Tail:** the GNN's **worst-draw** mean regret (per draw: mean over HOLDOUT-P; worst of
   the 8 draws) does not exceed MLP(T1)'s worst-draw mean regret by more than **1.0%**
   (absolute). Rationale, registered: the per-dataset median over 8 draws scores a model
   that collapses on 2 of 8 draws identically to one that never does — the exact failure
   `gnn_draw_study_v1` falsified ("the GNN never collapses": 2/8 draws collapsed while the
   rest read clean), and p5b's rule is compare *distributions*. 1.0% = 4× the provisional
   materiality floor, below the 2.49% B0 noise ceiling; it is recomputed proportionally if
   the §6 freeze moves M, in the registered direction only.

**Verdict mapping — first matching rule wins, walked top to bottom (every cell lands in
exactly one of VOID / VOID-UNDERPOWERED / PASS / FAIL; FAIL always carries its registered
named sub-verdict):**

| # | rule | verdict |
|---|---|---|
| V1 | any §7 control fails; verifier disagreement; feature-parity violation discovered (a pointwise-expressible GNN feature absent from `dim36crk`); any used dataset with `sweep_complete: false`; any arm's train-set regret above F3 (random-feasible) at convergence — training instrumentation broke | **VOID** (named trigger) |
| V2 | condition 3 violated | **FAIL-ATTRIBUTION** (the win, if any, is not the mechanism) |
| V3 | conditions 1, 2, 4, 5 all hold | **PASS** |
| V3b | conditions 1, 2, 4 hold, condition 5 fails | **FAIL-TAIL** — the mean wins, the tail does not; "a tight distribution with a tail" is the honest description and it is not a PASS |
| V4 | condition 1's CI straddles 0 AND point estimate ≥ 0.25% | escalate the ladder (300→504→804 datasets, then 8→12 draws); ladder exhausted → **VOID-UNDERPOWERED** |
| V5 | \|mean Δ\| < 0.25% AND mean `Δ3` > 0 with CI excluding 0 AND the same for `regret_i(A3) − regret_i(A2)` (i.e. A2 also beats A3) | **FAIL-ARCHITECTURE / POINTWISE-SUFFICIENT** — the registered GOOD result: decoder-state features suffice, coupling confirmed learnable, no graph needed |
| V6 | mean `Δ` CI (two-sided 95%) entirely below 0 | **FAIL-REVERSED** (the MLP wins) |
| V7 | condition 1 holds but condition 2 fails | **FAIL-DECODER-SHAPED** (the MLP's scores were fine; its greedy decode was myopic) |
| V8 | condition 4 fails | **FAIL-NOT-LEARNABLE** |
| V9 | anything remaining (e.g. significant but immaterial: CI > 0, point < 0.25%, V5 preconditions unmet) | **FAIL-IMMATERIAL** |

**The thresholds above may not be revised after data exists. A near-miss is a FAIL or a VOID
per the rules; there is no third option.**

Secondary replication (stage-1 204) and all sensitivity rows are reported in the same outcome
entry, labelled, and cannot change the verdict.

---

## 9a. Pre-probe ZERO: the T1-surrogate repair (registered abort — runs before anything in §10)

Offline, on the stage-1 corpus, for the price of a scorer run: recompute `R_exact` on the
stage-1 204 (Arm S, α=2.0, rtt) with the surrogate augmented by the **T1 plan-level column
set** — every §2 column computable from an enumerated sweep row (25–31 directly; 32–35 from
the plan's own placements; col 30 dropped as collinear with 29 + intercept, col 32 dropped
as constant at a full plan). Concretely, added to the registered
`y ~ a + b·Σm + counts` machinery in `score_route_b_contention.py` as one column set:
the k-integer counts; the four per-type quadratic co-residency sums Σ_t occ_{node(t)}[k];
Σ_t load/cap and the over-cap count; per-task min/max parent-hop sums; and the three
coupling-term columns Σ_edges hops/bottleneck, Σ_edges latency, and the same-node-parent
count — computed from the dataset's own `link_topology.routes`/`network_maps`, i.e. exactly
the quantities `_dependency_transfer_time` charges (payload is uniform at 800 MB, so the LS
coefficient absorbs it and the column set **spans the charged coupling term exactly**).
Median repair fraction and firing fraction reported exactly as stage-1 condition 2 did, and
the repair independently recomputed through `verify_route_b_scorer_agreement.py
--check-repairs`, never the scorer alone.

**The reading, registered now, before the number exists:**

- **Median repair fraction ≥ 0.5** (over the stage-1 firing datasets): the architecture claim
  is **pre-falsified** — a T1-expressible surrogate closes the structure, so nothing is left
  for message passing that decoder-state features cannot reach. Stage 2 does not run as
  registered. Recorded as **NO-GO-PREPROBE-T1**; if anything continues it is the reduced
  question "do decoder-state features suffice" (the V5 outcome), which needs no GNN.
- **Median repair fraction < 0.5**: stage 2 proceeds, and the T1-repaired per-dataset
  residual distribution (`r_exact_repaired_t1_pct`) **replaces** the kint distribution as
  MLP(T1)'s floor in the §6 power computation — the kint floor (mean 1.918) is the floor of a
  *count-augmented separable* surrogate and is registered as obsolete for powering the
  moment this runs.

### §9a OUTCOME — run 2026-08-25, reading applied as registered: **NO-GO-PREPROBE-T1**

Arm S, α=2.0, rtt, all 204 stage-1 datasets. Base statistic reproduced exactly (35/204 =
0.172 firing). **Median T1 repair fraction over the 35 firing datasets: 1.000** (mean 0.730;
26/35 closed ≥ 0.5; kint for comparison: median 0.000, mean 0.357 — matching the stage-1
scrutiny to the digit). Residual after the T1 repair: `frac(R_exact>5%)` falls 0.172 →
**0.054** (11/204, max 22.2%). Column attribution (ablation over the firing 35): the
parent-coupling block alone (kint + cols 33–35 analogues) closes at median 1.000; the
occupancy block alone at median 0.892 — the two sub-blocks are largely redundant routes to
the same closure. Independent verification: `verify_route_b_scorer_agreement.py
--check-repairs` agrees on **all 204 cells and all 612 repair values** (1int, kint, t1),
zero tie-acceptances needed. Two verifier defects were found and fixed to get there, both
recorded in LINEAGES: the pure-Python solver (standardized normal equations) did not reach
the true LS optimum on the wider t1 matrix (replaced with hand-rolled MGS QR), and the
verifier's 1int column was `max` excess where the registration says `sum` — a real bug the
old solver had masked, which also dissolves the stage-1 scrutiny's "ds_00008 genuine FP
tie" interpretation (with the correct column and solver there is no tie).

**Consequence, per the registered reading:** the architecture claim is pre-falsified on this
grid — a T1-expressible (pointwise + partial-assignment-state) surrogate closes the median
firing dataset completely, so stage 2 as registered (§3–§8) does not run. What survives is
the reduced V5-shaped question (do decoder-state features suffice in a *trained,
cross-dataset* model — the surrogate here is a per-dataset LS fit, an expressiveness bound,
not a trained model), and an 11-dataset residual stratum (5.4%, below stage 1's 10% bar)
where even the T1 surrogate stays >5% wrong. Neither justifies the registered build queue
without a new registration.

## 9. Pre-probe: the overfit kill condition (registered abort)

Before the full corpus spend, on the existing 12-dataset smoke corpus
(`gnn_datasets_dag4_route_b_smoke_s`, after the §10 ssc repair + smoke cache): train **4
draws** each of GNN(T2) and MLP(T1) to convergence and evaluate **on the training set** at
α=2.0. (4 draws, not 2, because this run doubles as the §6 σ calibration: per-dataset
paired-difference σ across draws, pooled over the smoke datasets, feeds the
calibrate-then-freeze step.)

**Abort condition:** if GNN(T2) mean train-set regret (over the 12 datasets, median of 4
draws) is ≥ MLP(T1)'s, stage 2 stops there — a model that cannot beat pointwise-plus-state
when both are allowed to memorise will not generalise past it. Recorded in `LINEAGES.md` as
**NO-GO-PREPROBE**, for the price of a smoke corpus. Expected direction, registered: GNN
strictly lower train regret; the smoke's ~2 firing datasets are where the gap should appear.
(This replaces any "beats random-feasible" bar, which could not abort anything.)

---

## 10. Build items — gated steps with acceptance checks (in order; none starts before sign-off)

| # | item | acceptance check |
|---|---|---|
| PP0 | **§9a pre-probe zero** — T1-surrogate repair on the stage-1 204 (scorer extension + verifier `--check-repairs` extension), result reported before anything below starts | verifier agrees with the scorer on `r_exact_repaired_t1_pct` per its registered tolerance on every firing dataset; the §9a reading applied as registered — a ≥0.5 median ends stage 2 as NO-GO-PREPROBE-T1 |
| B0 | ssc repair (`--repair`/`--rewrite-ssc`) + `METADATA.json` + `VALIDATION_REPORT.json` + `REGISTRY.json` registration for all route_b collections (4 existing + new blocks) | `system_state_captured_unique.json` present in every ds; `validate_dataset_collection.py` passes; collections appear in `REGISTRY.json` (this is the documented fourth-time-bitten prerequisite, promoted to a gated step) |
| B1 | masked decoder mode + verifier extension + hetero static guard (§4) | 1e-9 agreement on all 408 stage-1 datasets × α∈{2.0,3.0}; 13 positive-control tests pass; relax-path forbidden and infeasible completions counted in decode stats |
| B2 | `dim36crk` layout: `partial_state_columns` in `reduced_features.py`, `feature_builder` wiring, trainer flag | single-source import on both train and serve paths; cr-style ablation gate (zeroing the 11 columns moves ≥5% of held-out argmaxes, else the arm is VOID as not-actually-T1); determinism test extended and passing |
| B3 | DAG-aware graph cache for route_b (extend `prepare_graphs_cache.py`): parent edges, T2 features per §2, feasibility mask + capacity map + tied-optimal label sets in the cache contract | feature-parity audit: every pointwise-expressible cache feature has its `dim36crk` counterpart (checked against the §2 table); tie-tolerant label audit passes; corrupted-label control fails loudly |
| B4 | fresh corpora (TRAIN 1001–1075, HOLDOUT-P and HOLDOUT-B0 2001–2025) | `sweep_complete: true` and `placements.jsonl` present in every ds; B0 applied to all new collections |
| B5 | smoke cache + pre-probe (§9) | abort rule applied as registered |
| B6 | `experiments/route_b_stage2_{gnn,mlp_t1,mlp_t0}_seed{1..8}.yaml` (or one templated config per arm with per-seed env), wandb, lineage row; **pinned-split artifact** (one canonical-parent split from seed 42, consumed by every arm and draw; draws vary init + batch order only) | `tests/test_run_experiment.py` passes; every checkpoint has its sidecar, including the `split_artifact` hash; determinism test covers the pinned-split path |
| B7 | exact-assignment arm (A4) | fed true min-marginals it reproduces `R_exact` to 1e-9 on the stage-1 corpus |
| B8 | `score_route_b_stage2_gate.py` (no threshold args) + independent recomputation | §7 controls all pass before any HOLDOUT-P scoring; outcome row written to `LINEAGES.md` |

**Also registered with this doc (change 6, free corroboration):** Arm S at α=∞ fires 0/204 —
route A's physics condition (pairwise transfer on, no scarcity) corroborated at n=204 on
fixed instrumentation with the verifier agreeing to 1e-9. Per the stage-1 record
(LINEAGES.md:4040–4045) this is **corroborating evidence, not a literal re-run of route A's
grid** (route B uses 6 servers / per_client=0; route A used a different server count and
replica config) — the literal re-verification is route A's own 6-dataset retro-check,
already recorded. The 2×2 framing (coupling × competition, internally controlled) is valid
within route B's grid and is stated as such, not as a route-A replication.

---

## 11. Deviations from prior registered text, recorded honestly

- Stage 1 registered the strongest-MLP arm as "dim25cr + the k-integer features." T1 as fixed
  here **adds** parent-placement/hop/transfer columns (33–35) beyond the k-integer counts.
  The consequence is not merely "a stronger baseline": those columns make MLP(T1)'s
  plan-level score **non-separable**, so stage 2's comparison is GNN vs
  autoregressive-pointwise, a baseline against which stage 1's separable-surrogate result
  (and its median-0.000 repair) says nothing. Whether a T1-expressible surrogate already
  closes the effect is measured *first*, by the §9a pre-probe zero, under a reading
  registered before the number exists.
- "Exact-assignment decode arm on the MLP's scores" is here **defined** (it never was):
  exhaustive feasible-sweep argmin of the plan-level sequential score (§3, A4). No solver.
- "Scarcity-pressure order" is here **pinned** to `greedy_masked_plan`'s ascending
  (best score, task_id) order — the order stage 1 actually used — so the decoder acceptance
  check is exact rather than aspirational.
- The primary holdout is a **fresh** corpus; the stage-1 204 are demoted to a secondary,
  statistics-known replication set. Reason recorded in §5 (selection-on-own-outcome).
  The stage-1 registration named no holdout, so this is a specification, not a revision.
- The pre-probe is an overfit test (§9), replacing the too-weak "beats random-feasible" idea
  from the planning draft; registered before any smoke training exists.

## 12. Scope exclusions

No `task-types.json` edits. No episode-physics changes beyond route A's landed term. No live
serving in stage 2 — that is stage 3, which requires its own pre-registration and inherits
condition-style constraints from this one (same decoder discipline, draw distributions,
sealed cells). No datalab until the local smoke passes end-to-end (and then only under
`PARITY.md` rules, `HEROSIM_PY` guard, provenance-stamped). No new `train_*.py`. Grouped
argmax is not an arm. Thresholds immutable per §8.

## 13. Status

**PRE-PROBE ZERO RAN 2026-08-25: NO-GO-PREPROBE-T1 (§9a outcome above).** Stage 2 as
registered in §3–§8 is pre-falsified and does not run; the build queue B0–B8 is not
executed. This document stands as the registration under which that reading was fixed
before the number existed. The open decision — whether to pursue the reduced V5-shaped
question (decoder-state features in a trained model, no GNN needed) or to close route B's
GNN argument here — belongs to the user and requires its own registration either way.
