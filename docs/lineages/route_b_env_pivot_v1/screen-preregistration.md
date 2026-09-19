# route_b_env_pivot_v1 — SCREEN PRE-REGISTRATION

**Written 2026-08-27, before any pivot corpus exists. Nothing in §5 executes until this
document is signed off and a LINEAGES registration entry records it at its commit SHA.**

Fork context: route_b_v1 stage 2 closed **NO-GO-PREPROBE** (LINEAGES 2026-08-26): on the
current environment a GNN cannot beat pointwise-plus-prefix even at memorization. The user
chose CLAUDE.md option 2 — change the environment so exploitable joint structure exists.
This document registers the **screen** that decides whether a candidate environment
actually contains such structure, for the price of a scorer run and with no training. A
corpus that passes earns a v3 training registration (its own sign-off); it does not earn
training by passing.

## 1. The hypothesis, and why the current environment fails it

Under the fixed topological masked decode, a pointwise scorer with prefix features sees
the entire past; the only information legitimately exclusive to a graph model is the
**future** — the not-yet-committed tasks and their candidate sets. The current
environment gives that future nothing to do: no contested indivisible resource exists
(node memory is inert, α is label-side, contention is additive/linear), replicas are
disjoint per task type, and demands are uniform per type — so count statistics close the
structure (§9a/§9c/§9d; per-dataset closure 0.988 even at 8 tasks).

The target hardness shape is positive control 2's matching rig
(`tests/test_route_b_positive_controls.py:124-165`): non-additive costs, a binding
constraint that removes the low-marginal pairing, and coupling concentrated in the
forbidden region so occupancy columns are degenerate on the feasible set — provably
≥150% regret for ANY pointwise scorer. The pivot tries to produce that shape
**organically** via two levers: **overlapping scarce eligibility** (the no-replica-reuse
mask makes each (node, platform) an indivisible resource; scarcity + overlap makes it
contested across types) and **per-instance demand heterogeneity** (packing: whether a
cheap feasible completion exists depends on future tasks' sizes — a lookahead no fixed
column set computes).

**Registered decisions inherited from the user (2026-08-27):** scarcity is
**label-side only** (same claim shape as stage 2 — offline decode regret; physics
enforcement is a stage-3 matter); the **8-task rung is in scope** (the v2-era "8-task has
no gate role" decision is superseded for the pivot, recorded here); ladder exhaustion is
**terminal** — FAIL-BY-EXHAUSTION closes the static pivot and the fork (dynamic
environment vs closing the GNN argument) returns to the user as its own decision.

## 2. The extended pointwise competitor (fairness floor of this screen)

The screen is honest only if the pointwise side gets every feature it could legally have.
Beyond the frozen stage-2 `t1` set (kint/quad/cap/hop/coupling) and `linkrank`, two new
**opt-in** blocks exist (built and verified 2026-08-27, before this registration):

- **`hetdem`** (8 cols): per-instance-demand sufficient statistics — demand-weighted
  per-type co-residency, real-demand load/cap and over-cap, max single-node total demand
  /cap, demand-weighted excess sharing.
- **`futureint`** (4 cols): future-interaction columns — candidate-node quantities
  weighted by the aggregate demand of not-yet-committed tasks whose eligibility includes
  that node, under the fixed topological order. (Plain aggregate future demand is
  sweep-constant and intercept-absorbed — measured and fixture-pinned; only the
  interaction renderings are non-vacuous.) These columns use future tasks' static
  demands/eligibility only — never where they will go.

**The screen's kill arm is `t1x` = t1 + hetdem + futureint + linkrank.** The pooled arm
is krank + demand-weighted krank + hetdem + futureint + linkrank
(`route_b_coefficient_transfer.py --add-linkrank --extended-blocks`). Every extended
column has an independent 1e-9 recomputation in `verify_route_b_scorer_agreement.py`
(own solver, no scorer imports) and closed-form fixtures with verified teeth, including:
a heterogeneous rig hetdem provably closes, a packing rig it provably cannot (hand-computed
floor), and a uniform-demand redundancy rig.

**Claim-to-beat context, measured on the stage-1 pilot BEFORE this registration (stated
as context, never a bar):** on the pilot's firing 35, `t1x` closes 27/35 at median 1.0,
and the extended pooled closure is **0.892** (up from the frozen 0.648). The current
environment is pointwise-closed against the extended competitor too; the pivot must beat
*this* floor, not the weaker frozen one.

## 3. The ladder (order fixed; no post-hoc rungs without a new registration)

All rungs keep the 204-dataset shape (2 conn-probs × 2 replica-configs × 3 queue-dists ×
17 seeds), Arm-S physics (`HEROSIM_DATA_LOCALITY=1`, `HEROSIM_OUTPUT_SIZE_BYTES=8e8`),
`node_disk_v2`. Fresh seed blocks per rung — H0: 3001–3017, H1: 3101–3117, H2: 3201–3217,
H3: 3301–3317 — none previously used. Each rung has a **paired separable control** (same
grid and seeds, `HEROSIM_DATA_LOCALITY`/`HEROSIM_OUTPUT_SIZE_BYTES` unset) whose R_exact
must be ≈ 0.

| rung | preset | levers added | primary α |
|---|---|---|---|
| H0 | `route_b_pivot_h0` | config-only scarcity squeeze (4 servers, per_server 1, low replica-server-pct) | 2.0 (`alpha_max`) |
| H1 | `route_b_pivot_h1` | + per-instance `demand_spread` U[0.5, 2.0], `cap_mode: alpha_mean` | 2.0 (`alpha_mean`) |
| H2 | `route_b_pivot_h2` | + `replica_overlap: true` | 2.0 (`alpha_mean`) |
| H3 | `route_b_pivot_h3` | + `dag_instances: 2` (8 tasks); `MAX_PLACEMENT_COMBINATIONS_SKIP` derived at 16,777,216 | 4.0 (doubling correspondence, same no-task-count-term argument as §9d) |

α response ladder per rung ({1.5, 2.0, 3.0} for H0–H2; {3.0, 4.0, 6.0} for H3) plus the
always-appended unconstrained anchor — sensitivity rows, never the verdict α.

**Registered feasibility fallback (no post-hoc knob-turning):** if a rung's primary α
shows dirty counters (`no_feasible_rows > 0` or `greedy_stuck > 0`), the rung is read at
the **tightest α on its registered ladder with clean counters**; if none is clean, the
rung is VOID-INFEASIBLE and the ladder continues. H0 is expected to FAIL — it calibrates
the screen and confirms the NO-GO negative on squeezed configs.

**Generation integrity:** this sandbox has shown non-deterministic worker truncation
during sweep enumeration. Generation is seed-deterministic, so failed datasets are
re-generated until 204/204 SUCCESS with `sweep_complete: true` and zero skip-counter
hits — a pure retry, not selection. A rung that cannot reach 204/204 clean is
VOID-GENERATION.

## 4. Bars (fixed now; a near-miss is a FAIL, there is no third option)

Every statistic is read at the rung's primary α on the **firing stratum by definition**
(`R_exact > 5%` computed on that rung), tie bands `[pessimistic, mean_tied, optimistic]`
mandatory, `mean_tied` the fair reading, with `registered` and `pessimistic` required to
agree in direction across each bar — disagreement is **VOID-TIE-INDETERMINATE** for that
rung, never a pass.

- **S0 — controls (VOID gates):** positive-control suite incl. the new fixtures passes;
  paired separable control `r_exact.frac_gt_1pct ≤ 0.02`; counters clean per §3;
  `saturated` repairs are refusals, never zeros; independent verifier agreement
  (`--check-repairs`, `--check-blocks`, `--check-krank` with `--extended-blocks`) to
  1e-9 on every (dataset, arm). Any S0 failure ⇒ the rung is VOID (named trigger).
- **S1 — structure exists:** `r_exact.frac_gt_5pct ≥ 0.25` (pilot baseline 0.172).
- **S2 — the kill bar:** per-dataset **`t1x`** repair fraction, **median over the firing
  stratum < 0.5 on `mean_tied`**. This is the direct negation of what closed stage 2.
- **S3 — pooled:** extended pooled closure (krank + demand-krank + hetdem + futureint +
  linkrank, one coefficient set) `median_mean_tied ≤ 0.5` on the firing stratum.
- **S4 — attribution guard (the §9d lesson):** the parent-coupling block alone
  (hop+coupling) must close **< 0.8 median** on the firing stratum, and the full block
  ablation table (kint / quad / occupancy / parent-coupling / hetdem / futureint arms) is
  reported. A rung that fires loudly but whose closure is carried by one pointwise block
  is pointwise-reachable regardless of R_exact — screening on R_exact alone would have
  passed the 8-task corpus.

**Build item B1 (before any rung is scored, VOID if skipped):** the `ablation()` arms in
`route_b_coefficient_transfer.py` extended with `hetdem` and `futureint` single-block
arms so S4's table exists, with the same independent verification.

## 5. Readings (registered before any number exists)

- **First rung passing S1–S4 with S0 clean ⇒ PIVOT-CANDIDATE.** The ladder stops. The
  next step is a v3 stage-2-style training registration on that corpus (fresh holdout
  blocks, draw distributions, the §8-style gate) — a separate document with its own
  sign-off. Passing the screen licenses drafting that registration, nothing more.
- **All rungs FAIL ⇒ FAIL-BY-EXHAUSTION** — terminal for the static environment pivot.
  The fork (dynamic/closed-loop environment vs closing route B's GNN argument for the
  paper) returns to the user; each direction needs its own registration.
- **A VOID rung** (tie-indeterminate, infeasible, generation) is neither pass nor fail;
  the ladder continues past it. An exhausted ladder containing VOIDs is recorded as
  exhausted-with-voids, stated plainly.
- Wherever the 4-task limit is doing the work, the registered sentence remains "the
  corpus is too small to test the architecture claim", never "the architecture claim is
  false" (§4 of the stage-2 registration, carried forward).

## 6. Scope exclusions

No physics changes (label-side only, per the user decision). No training, no caches, no
checkpoints under this registration. No `task-types.json` demand-table edits (per-instance
scaling only). No new decode modes. Thresholds immutable once signed off; every deviation
is logged in the LINEAGES outcome entry with its evidence.
