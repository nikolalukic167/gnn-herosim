# route_b_v1 — STAGE 2 PRE-REGISTRATION

**DRAFT FOR SIGN-OFF — written 2026-08-25, before any DAG cache, masked decoder, T1 feature
layout, trained model, fresh corpus, or gate script exists.** Nothing below is executed until
this document is signed off. On sign-off it is committed and a
`### route_b_v1 — STAGE 2 PRE-REGISTRATION` entry in `LINEAGES.md` records the registration
by reference to this file at its commit SHA. Until then, building the cache, the decoder, the
feature layout, or training anything is invalid work.

Stage 1 record: the `### route_b_v1` entries in `LINEAGES.md`, from PRE-REGISTRATION through
the stage-1 PASS and its post-PASS scrutiny (commits 2c3ebbc → 4bf714a). *(Line-number
references were used here originally and went stale the moment the stage-1 entry was amended
in place; anchors are the section headings.)* Stage 1
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

**A second registration defect, found the same way as the tie rule (recorded 2026-08-25):**
§4 pinned "scarcity-pressure order" to `greedy_masked_plan`'s ascending `(best available
marginal, task_id)`. On this corpus that order does not exist: in **all 204 datasets the four
min-marginal minima are exactly tied** — every task's best placement lies in the globally best
plan, so `min_p m_t(p)` equals the global minimum RTT for every task — and the tie-break
collapses the order to `task_id`, which is the DAG's topological order. Measured consequences:
**0 of 816 DAG edges decode child-before-parent** (§2's hedge that "parents are not guaranteed
to precede children" never fires) and **0 of 816 steps** have a task's best choice already
taken by an earlier task; only capacity ever blocks the top choice, on 167/816 = 20.5% of
steps. The registered order therefore carries no scarcity information whatsoever. This is the
same class of hole as the unspecified tie rule: a registration naming a discriminator that is
constant on its own corpus. **A corrected stage 2 must fix both**, or it repeats the error
under a new name.

Also measured: **T1 ≡ T0 at decode step 0** (all eleven partial-state columns are zero when
nothing is placed), and the prefix-oracle curve (7.78 → 9.84 → 1.98 → 0.31) puts essentially
all decoder myopia in the first two of four steps. The framing this supports is that four
tasks is too small a joint decision to test the architecture question, not that the question
is answered — the four-task limit is doing the work here, and no sentence built on it should
read as "the architecture claim is false."

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
the same closure. *(§9b amendment: both numbers reproduce, but "largely redundant routes"
is wrong — **both blocks contain kint**, which alone closes 0.000; it is kint combined with
either the quadratics (0.843) or the parent columns (1.000) that closes the effect, and the
parent block stripped of kint reaches only 0.392. This ablation had no code until §9b.)*
Independent verification: `verify_route_b_scorer_agreement.py
--check-repairs` agrees on **all 204 cells and all 612 repair values** (1int, kint, t1),
zero tie-acceptances needed. Two verifier defects were found and fixed to get there, both
recorded in LINEAGES: the pure-Python solver (standardized normal equations) did not reach
the true LS optimum on the wider t1 matrix (replaced with hand-rolled MGS QR), and the
verifier's 1int column was `max` excess where the registration says `sum` — a real bug the
old solver had masked, which also **retracts** the stage-1 scrutiny's "ds_00008 genuine FP
tie" interpretation (with the correct column and solver there is no tie there; the stage-1
entry in `LINEAGES.md` is amended in place, not merely superseded here). Prediction ties
are nonetheless real elsewhere in this machinery — §9b measures them and finds 22/35 firing
datasets tie at the full-T1 argmin without the median moving at all.

**Consequence, per the registered reading:** the architecture claim is pre-falsified on this
grid — a T1-expressible (pointwise + partial-assignment-state) surrogate closes the median
firing dataset completely, so stage 2 as registered (§3–§8) does not run. What survives is
the reduced V5-shaped question (do decoder-state features suffice in a *trained,
cross-dataset* model — the surrogate here is a per-dataset LS fit, an expressiveness bound,
not a trained model), and an 11-dataset residual stratum (5.4%, below stage 1's 10% bar)
where even the T1 surrogate stays >5% wrong. Neither justifies the registered build queue
without a new registration.

## 9b. Coefficient transfer: does the §9a bound survive ONE coefficient set? (registered 2026-08-25, before the number exists)

§9a's repair fits **fresh coefficients on every dataset's own sweep**. A trained
cross-dataset model gets **one** coefficient set. So §9a bounds what a T1-expressible
surrogate can do *per dataset*, which is strictly more than what a single model can do, and
NO-GO-PREPROBE-T1 is only as strong as that gap is small. This measures the gap. It spends
one scorer pass on data already on disk and trains nothing.

**The obstruction, stated before the design.** The kint block cannot carry a shared
coefficient: its columns are one per `(node, task_type)` pair *in that dataset's own
demand*, so the vocabulary and its width differ across datasets (K ∈ 8…13 over the stage-1
corpus, X widths 21–26). There is no cross-dataset coefficient vector to fit. Dropping it
moves the pooled surrogate **closer** to the registered T1 layout, not further: §2 cols
25–28 are *candidate-relative* per-type occupancy, whose plan-level rendering is the `quad`
block; node-identified kint is strictly more than the registered feature layout ever gives a
model. Three cells are therefore reported, so that the cost of dropping kint is never
confused with the cost of pooling:

| cell | fit | isolates |
|---|---|---|
| A | per-dataset, full T1 | reproduces §9a (median 1.000) or the harness is wrong |
| B | per-dataset, T1 − kint | the cost of dropping the un-poolable block |
| C | **pooled, T1 − kint** | **the registered statistic** |

**Cell C specification, fixed now.** One design matrix over all 35 firing datasets' full
sweeps (fit on the full sweep, decode on the feasible subset — §9a's choice unchanged);
columns = one intercept indicator per dataset, one shared coefficient `b` on `Σ_t m_t(p_t)`,
and the 9 shared T1−kint columns (`quad`×4, `load_over_cap`, `overcap_tasks`, `min_hop_sum`,
`max_hop_sum`, `transfer`, `latency_sum`, `same_node_edges`). Per-dataset intercepts are
free: an additive constant cannot change a within-dataset argmin. Fit target is **raw y in
seconds**, which keeps the coefficients in physical units. Decode, tie-break and the
`min(base, repaired)` clamp are the registered §9a ones, reused, not re-typed.

**The reading, registered before the number exists:**

- **Cell C median repair fraction ≥ 0.5** — the bound transfers to a single coefficient set.
  A trained pointwise-plus-state model would realize it; the reduced V5 question is answered
  on paper and no build queue is justified to confirm it. Route B's GNN argument closes.
- **Cell C median < 0.5** — the per-dataset fit was exploiting freedom one model does not
  have. NO-GO-PREPROBE-T1 is weaker than it reads, V5 becomes a real empirical question, and
  it gets its own registration before anything is built.
- **Cell B median already < 0.5** — cell C cannot be read as a *pooling* test at all, because
  the drop is attributable to dropping kint. Recorded as **VOID-KINT-CONFOUNDED**, with the
  named follow-up (a node-agnostic canonicalization of kint) rather than as evidence either
  way.
- **Sensitivity, declared non-decisive now:** cell C refit on per-dataset-normalized y. If it
  straddles 0.5 against the primary, that is reported as a straddle and resolved by neither.

**Two physical predictions, also registered before the fit.** Payload is uniform at
800,000,000 bytes and `Platform._payload_transfer_time` charges
`n_hops × payload / (bottleneck_mbps × 1024²)` whenever a fabric is configured (the
`ROUTE_B_PILOT_V1_GRID` corpus sets `server_mesh=True`), while `_dependency_transfer_time`
adds latency in raw seconds. So on a correctly specified fit the pooled coefficients should
land near **transfer = 800e6/1048576 = 762.939453125** and **latency_sum = 1.0**. These are
predictions about the *physics*, not gate conditions: episode RTT is an aggregate over a
critical path, so deviation is informative rather than disqualifying.

**Evidence ranking, fixed now so it cannot be chosen afterwards.** The cell-C repair
fraction is decisive; per-dataset coefficient dispersion is **descriptive**. Dispersion is
reported for every column (mean, sd, median, IQR, sd/mean) alongside each dataset's condition
number and the columns the QR drops as dependent — a coefficient on a near-dependent column
is not identified, and its spread says nothing about the physical term. `same_node_edges` is
exactly `4 − remote_edge_count` on `diamond4`, so it is collinear with the hop block by
construction. sd/mean is not reported where |mean| is within one sd of zero.

**Independent recomputation is mandatory, not optional**, and for the same reason §9a's was:
`verify_route_b_scorer_agreement.py` rebuilds the pooled design from each dataset's raw files
and solves it with its own pure-Python QR, agreeing to 1e-9 on every coefficient and every
repair fraction. That verifier is the artifact that hid two real bugs behind three rounds of
checking; as of this commit it is backstopped by closed-form fixtures
(`tests/test_route_b_repair_fixtures.py`), and it still gets to disagree.

### §9b OUTCOME — run 2026-08-25, reading applied as registered: **VOID-KINT-CONFOUNDED**

Cell A (per-dataset, full T1) reproduces §9a exactly: median **1.0000**, mean 0.7302, 26/35,
11 residual. Cell B (per-dataset, T1 − kint) median **0.3922** — already below 0.5, which is
the registered VOID trigger, so cell C (pooled, median 0.0000) cannot be read as a test of
pooling: the drop is attributable to dropping `kint`, not to sharing coefficients. **§9b
neither weakens nor strengthens NO-GO-PREPROBE-T1; the reduced V5 question stays open and
stays empirical.** The sensitivity (equal dataset weight) agrees at 0.0000, no straddle.

Two things it did settle:

- **§9a is tie-robust.** Cell A's median is 1.0000 under the registered tie-break, under
  optimistic tie resolution, and under pessimistic — despite 22/35 firing datasets tying at
  the argmin (max group 8). Cell B is genuinely indeterminate by contrast: band
  [0.392, 1.000], ties up to 16 plans wide, i.e. the node-agnostic columns cannot separate
  those plans at all. The independent verifier surfaced this first, as three cell-B
  disagreements (0.000 vs 1.000) that are legitimate alternative argmins of a tied group.
- **The physical predictions are not testable on these fits.** Pooled `transfer` = 330.96
  against the predicted 762.939453125, `latency_sum` = −38.4 against 1.0, with per-dataset
  dispersion in the thousands. The cell-B/C designs are mis-specified by construction
  (they omit the block doing the work), 9/35 are rank-deficient, and `same_node_edges` is
  collinear with the hop block on `diamond4`. Reported because registered; decisive of
  nothing, exactly as the evidence ranking said in advance.

Verification: `--check-blocks`, 315/315 (dataset, arm) fractions to 1e-9 across 9 arms.
Artifact: `simulation_data/route_b_coefficient_transfer.json`.

## 9c. Is `kint` a T1 feature at all? (registered 2026-08-25, before the numbers exist)

**The objection, which is against §9a itself.** §9a's T1 column set includes `kint`: one free
coefficient per `(node, task_type)` pair, i.e. a per-dataset lookup table over node
**identities**. **No column of the §2 `dim36crk` table is identity-indexed.** Cols 25–28 are
per-type occupancy on *the candidate's own node* — anonymous, fixed width, four columns. So
§9a's kill test may have been run with a surrogate strictly more expressive than the T1 arm
it stands in for, which would make it no bound on that arm at all. §2's own verbatim rule
cuts both ways: features the GNN has must be given to the MLP, and features the MLP cannot
have must not be credited to it.

**One of the two measurements is already in hand, and this is the load-bearing observation.**
The scorer's `quad` block is *exactly* the plan-level rendering of cols 25–28:
`quad[k] = Σ_n tot[n]·occ[n][k] = Σ_t occ_{node(t)}[k]`. Likewise `load_over_cap` = col 29,
`overcap_tasks` = col 31, `min/max_hop_sum` = cols 33–34, `transfer` = col 35. Therefore
**T1 − kint is precisely the `dim36crk`-expressible plan-level set**, and §9b's cell B *is*
the anonymous closure measurement. Its value is **0.392** under the registered tie-break —
below the 0.5 bar. No new column set needs inventing; what needs settling is (a) whether the
identity block is nonetheless reproducible from node features, and (b) how cell B's tie band
is to be read, since that band ([0.392, 1.000]) straddles the threshold.

**(a) Identity or features?** Take the fitted per-`(node, task_type)` coefficients from the
35 firing datasets and regress them on that node's own identity-free features — capacity at
α, max single demand, replica count on the node (total and of that type), queue depth on the
node (total and of that type), mean/min hop distance to the other server nodes, link degree,
mean bottleneck bandwidth — plus a type indicator. Pooled across datasets, **held out by
dataset** (fit on 34, predict the 35th, cycled), scored as pooled out-of-sample R².

- **Held-out R² ≥ 0.5** — the block is a function of node features; a cross-dataset model
  with node features can reproduce it and identity indexing was parameterization
  convenience. **§9a's NO-GO stands as written.**
- **Held-out R² < 0.5** — the surrogate memorized per-dataset node identity. No pointwise
  model with the registered feature set can carry it, and **§9a does not bound the T1 arm.**

**(b) How to read a tie band, registered now.** A tie group is a set of feasible plans the
surrogate scores identically; a real masked decoder must pick one and cannot pick the best by
oracle. So:

- **`optimistic` (best plan in the tie group) is NOT a valid decoder reading** and is
  reported only as an upper bound. Crediting a surrogate with plans it cannot distinguish is
  precisely the error §9a exists to avoid.
- **`mean_tied` — the expected regret under an arbitrary tie-break — is the registered fair
  reading**, because that is what a decoder with a fixed but uninformative tie rule achieves
  in expectation. `registered` (sorted plan key) and `pessimistic` are reported alongside.
- The **anonymous closure verdict is read off `mean_tied`**, with `registered` and
  `pessimistic` required to agree in direction. If they disagree, the outcome is
  **VOID-TIE-INDETERMINATE** and neither branch below is taken.

**Verdict for (b):**

- **Anonymous median ≥ 0.5** — NO-GO-PREPROBE-T1 stands, correctly measured against the
  registered feature set.
- **Anonymous median < 0.5** — **VOID-T1-MISSPECIFIED.** §9a's kill test used a component
  that is not a T1 feature; stage 2's architecture question is **reopened**, because the
  structure is not reachable by a pointwise model with the registered feature set — which is
  exactly what stage 2 was built to test.

**A sub-0.5 anonymous median is NOT a licence to start the build queue.** It is reported, and
stage 2 is re-registered with the corrected T1 definition before anything is built. §9a's
purpose was to kill cheaply; a corrected §9a that fails to kill changes the registration, not
the discipline.

**Exploratory arm, labelled non-registered:** `krank` — occupancy indexed by node *rank*
under a canonical identity-free ordering (capacity, then mean hop distance) rather than by
node name. Anonymous and fixed-width, but preserving the cross-node distribution `quad` sums
away. It is not in `dim36crk` and no verdict is read from it; it exists to separate "needs
identity" from "needs per-node resolution keyed by something", which is the question a
corrected stage 2 would have to answer.

### §9c OUTCOME — run 2026-08-25, readings applied as registered

**(a) Identity, not features. Held-out-by-dataset R² = 0.0138** (in-sample 0.0974) over 381
gauge-centered coefficients across 35 datasets, against capacity, max demand, replica counts
(total and per type), queue depth (total and per type), mean/min hop distance, link degree
and mean bottleneck, plus a type indicator. The in-sample figure is the telling one: node
features barely explain these coefficients *even without* a generalization gap, so this is
not a small-sample effect. **Registered reading: R² < 0.5 ⇒ the surrogate memorized
per-dataset node identity, and §9a does not bound the T1 arm.**

**(b) VOID-TIE-INDETERMINATE.** The dim36crk-expressible set (cell B) gives median
**0.6483** under the registered fair reading `mean_tied`, but **0.3922** under both
`registered` (sorted plan key) and `pessimistic`. The three readings disagree in direction
across the 0.5 bar, so per §9c neither branch is taken. Note *why* they disagree: with tie
groups up to 16 plans wide, the sorted-plan-key rule happens to land **worse than an average
tie-break** on these datasets. **This is a genuine specification gap in stage 2, not a
numerical nuisance — §4's decoder never specified what to do with tied scores, and the
anonymous closure verdict flips on that choice.** Any corrected registration must pin it.

**Consequence: NO-GO-PREPROBE-T1 is retracted as measured.** It was computed with `kint`, a
block that (i) has no column in the §2 `dim36crk` table, (ii) is identity-indexed and so
carries no cross-dataset coefficient vector at all (§9b), and (iii) is not recoverable from
node features (§9c(a)). Stage 2's architecture question is **reopened**. Per §9c this is
**not** a licence to start the build queue: stage 2 is re-registered with a corrected T1
definition — including a tie rule — before anything is built.

**Exploratory, no verdict read from it, and it is why the reopening may be short.** Replacing
`kint` with `krank` — occupancy indexed by identity-free node *rank* (ascending capacity,
then mean hop), padded to a common width — closes **1.000** per dataset, identical to the
identity-indexed block. So the closure never needed node *identity*; it needed per-node
occupancy *resolution*, which `dim36crk`'s four candidate-local columns do not provide. And
unlike `kint`, `krank` pools: **one coefficient set across all 35 datasets closes a median
of 0.790** (mean_tied 0.824; 20/35 ≥ 0.5). That is the follow-up the §9b VOID named, and it
suggests the pointwise-plus-decoder-state conclusion survives correction — reached by a
richer but still pointwise, still anonymous, still single-coefficient-set feature layout,
with no message passing. It is unregistered and independently unverified; it is a
hypothesis for the corrected registration, not a result.

**Verification.** Cells and every ablation arm: `--check-blocks`, 315/315 to 1e-9. **The
§9c(a) regression and both `krank` arms are NOT independently verified** — they are single
implementations, stated as such.

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
fixed instrumentation with the verifier agreeing to 1e-9. Per the stage-1 record (the
"Route A cross-reference" paragraph of the stage-1 OUTCOME entry in `LINEAGES.md`) this is
**corroborating evidence, not a literal re-run of route A's
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
- **§9b was added AFTER §9a's outcome was known (2026-08-25), which §8's immutability clause
  forbids for gate thresholds.** Recorded as a deviation rather than quietly inserted. Two
  things make it a defensible one, and a reader is free to weigh them: (a) §9b does not touch
  any §8 gate condition or threshold — stage 2 is already pre-falsified and cannot be revived
  by it; it tests the *scope* of the §9a reading, which §9a itself flagged in its own outcome
  text ("the surrogate here is a per-dataset LS fit … not a trained model"). (b) Its own
  reading, its three cells, its VOID condition and its evidence ranking were all written into
  this file **before the pooled fit was run even once** (the harness that computes it did not
  exist yet; §9b's commit is the one that adds it). What it can do is weaken
  NO-GO-PREPROBE-T1; it cannot strengthen it, since the per-dataset bound is an upper bound on
  the pooled one by construction.

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

**§9c RAN 2026-08-25: NO-GO-PREPROBE-T1 IS RETRACTED AS MEASURED.** `kint` is not a T1
feature under this document's own §2 rule — no `dim36crk` column is identity-indexed — and
§9c(a) measured its coefficients to be unrecoverable from node features (held-out R² 0.014).
The dim36crk-expressible closure is **indeterminate** (0.392 vs 0.648 depending on a tie rule
§4 never specified). **Stage 2's architecture question is REOPENED**, and must be
re-registered with a corrected T1 definition and a tie rule before anything is built.
Exploratory and unverified, but pointing the same way as §9a did: an identity-free
*rank-indexed* occupancy block closes 1.000 per dataset and **0.790 under one pooled
coefficient set**, so the correction may well restore the conclusion without a graph.

**§9b RAN 2026-08-25: VOID-KINT-CONFOUNDED (§9b outcome above).** The attempt to settle
that open decision on paper **failed on its own terms**, and the registered VOID branch is
why that is visible rather than dressed up as an answer. NO-GO-PREPROBE-T1 is unmoved, and
is now additionally known to be tie-robust. What §9b removes is a specific shortcut: "the
per-dataset bound obviously transfers to one coefficient set" is not available, because the
block carrying the closure (`kint`, per-(node, task_type) counts) has no cross-dataset
coefficient vector at all. Whether a *trained* model reaches that structure through node
features is a real empirical question, and the only way to answer it is to train something.

Two stale defaults, noted rather than fixed, since they bind only if V5 reopens:
`route_b_stage2_power.py:37,40` still default to the stage-1 report (which carries no `t1`
column) and to the `r_exact_repaired_kint_pct` floor that §9a declared obsolete; and
`score_route_b_gate.py:90` iterates only `("1int","kint")`, so the headline `t1` statistic
never passed through the registered gate machinery — it was computed ad hoc from the frozen
JSON. Anyone reopening V5 should fix both before quoting a power number.
