# route_b_env_pivot_v1 — ACTIVE

> **Status:** `ACTIVE` &nbsp;·&nbsp; **Index:** [LINEAGES.md](../../LINEAGES.md)

**Outcome.** Current work. Screen registered 2026-08-27. `greedy_stuck` was measured as **decoder myopia on every rung** (backtracking rescues 458/458); **AMENDMENT 2** replaced the decoder and **H0/H1 counters went clean**; both paired controls were then generated under AMENDMENT 1 and **both PASS S0**. The bars were then read for the first time: **H0 is VOID-TIE-INDETERMINATE on S1** (`registered` 0.1667 fails, `pessimistic` 0.2549 passes) and **H1 FAILS S1** (all four band members below the 0.25 bar). **S2, the kill bar, is uncomputable on this grid** — its `t1x` competitor needs ≥82 sweep rows and the pivot's scarcity squeeze produces 16 and 64, so it is refused on 204/204 datasets in both arms. S3/S4 looked blocked the same way but were not: the transfer tool aborted the whole run on one arm's refusal, and with that **tool defect fixed** (refusals recorded per arm, pooled block de-nested; 0 pre-existing values moved, 12 tests with verified teeth) **S4 PASSES on both rungs** (`hop+coupling` closes 0.0000 on the full firing stratum) and **S3 lands on the bar** — H0 `mean_tied` 0.500000, H1 0.500835 against `≤ 0.5`. H2 stays VOID-GENERATION and H3 was never generated, so the ladder is **not** exhausted and no pivot-level verdict is licensed. **The S2 problem was then measured to be a grid problem, not a competitor problem** — a 204/204 wide-arm probe makes `t1x` fit on 41/41 firing datasets with **zero saturation in both arms**, while contention binds *harder* (`cw_infeas` 0.91) and the squeeze is untouched (2 hosting nodes on every dataset of every rung), and **H3 was measured to generate 0/204 as registered** (uniqueness exhaustion: 8 tasks over a pool of 2 or 4). Both are the subject of **AMENDMENT 3, drafted and awaiting sign-off**. The pair it proposes (`per_server` 4 and 5) was then probed at 4 tasks, 204/204, and **passes all four bars** — S1 0.2843/0.2941/**0.3137 pess** vs `≥ 0.25`, S2 `t1x` **58/58 fitted, 0 saturated**, S3 and S4 at 0.0000 — so signing the amendment is very likely to produce the ladder's first PIVOT-CANDIDATE. **That is a probe, not a rung reading**: S0's paired control is ungenerated and the amended rung is registered on a fresh seed block precisely so its bars are read on unseen data. §6 of the amendment names the selection hazard in full. Of the other two open questions, the S0 reading-rule one is **moot** (the sub-gates are reachable now and they pass) and S3's knife-edge is **left alone** — moving a bar after watching it land on the line is the post-hoc bar-moving the registration forbids.

**Related:** [route_b_v1](route_b_v1.md) · [route_a_v1](route_a_v1.md)

**Attachment:** [SCREEN PRE-REGISTRATION](route_b_env_pivot_v1/screen-preregistration.md)

**Attachment:** [AMENDMENT 1 — the S0 separable-control definition](route_b_env_pivot_v1/screen-amendment-1.md)

**Attachment:** [ladder feasibility findings (2026-08-27)](route_b_env_pivot_v1/ladder-findings.md) — **§9 supersedes §3 and §4.1**

**Attachment:** [AMENDMENT 2 — the decoder behind `greedy_stuck`](route_b_env_pivot_v1/screen-amendment-2.md) — **signed off 2026-08-27**

**Attachment:** [AMENDMENT 3 — the H2/H3 grid, so S2 becomes computable](route_b_env_pivot_v1/screen-amendment-3.md) — **DRAFT, not signed off**

> **Split note.** This node was carved out of a single 4,995-line `LINEAGES.md` on
> 2026-08-27; the section bodies below are byte-for-byte as written. An *above* / *below*
> pointer inside one may refer to a section that now lives in another node — the
> **Related** links say which.

## Record

Newest first; the sections themselves are in chronological order below.

- [route_b_env_pivot_v1 — the pair AMENDMENT 3 proposes passes all four bars in probe (2026-08-28)](#route-b-env-pivot-v1-the-pair-amendment-3-proposes-passes-all-four-bars-in-probe-2026-08-28)
- [route_b_env_pivot_v1 — S2 is a grid problem, not a competitor problem; and H3 does not generate as registered (2026-08-27)](#route-b-env-pivot-v1-s2-is-a-grid-problem-not-a-competitor-problem-and-h3-does-not-generate-as-registered-2026-08-27)
- [route_b_env_pivot_v1 — the transfer tool's abort-on-refusal fixed; S3 and S4 read on both rungs (2026-08-27)](#route-b-env-pivot-v1-the-transfer-tools-abort-on-refusal-fixed-s3-and-s4-read-on-both-rungs-2026-08-27)
- [route_b_env_pivot_v1 — S1–S4 read on H0 and H1: H0 VOID-TIE-INDETERMINATE, H1 FAILS S1, and S2 is uncomputable on this grid (2026-08-27)](#route-b-env-pivot-v1-s1s4-read-on-h0-and-h1-h0-void-tie-indeterminate-h1-fails-s1-and-s2-is-uncomputable-on-this-grid-2026-08-27)
- [route_b_env_pivot_v1 — H0 and H1 controls generated under AMENDMENT 1; both PASS S0 (2026-08-27)](#route-b-env-pivot-v1-h0-and-h1-controls-generated-under-amendment-1-both-pass-s0-2026-08-27)
- [route_b_env_pivot_v1 — SCREEN PRE-REGISTRATION (signed off 2026-08-27)](#route-b-env-pivot-v1-screen-pre-registration-signed-off-2026-08-27)

---

### route_b_env_pivot_v1 — SCREEN PRE-REGISTRATION (signed off 2026-08-27)

Stage 2's **NO-GO-PREPROBE** (`route_b_v1`, 2026-08-26) resolved to **CLAUDE.md option 2** —
user decision 2026-08-27: change the environment so exploitable joint structure exists,
rather than continue arguing the architecture claim on an environment where count
statistics already close the structure. `docs/lineages/route_b_env_pivot_v1/screen-preregistration.md` at commit
`019bdcb` is the registration: a **training-free screen** that decides whether a
candidate environment actually contains pointwise-irreducible structure, for the price
of a scorer run. Passing earns a v3 training registration (its own sign-off); it does
not earn training by passing.

- **Extended pointwise competitor, built and independently verified BEFORE
  registration:** `t1x` = `t1` (kint/quad/cap/hop/coupling) + `hetdem` (8 cols,
  demand-weighted sufficient statistics) + `futureint` (4 cols, candidate-node x
  not-yet-committed-task-demand interaction under the fixed topological order) +
  `linkrank`. Every new column has an independent 1e-9 recomputation in
  `verify_route_b_scorer_agreement.py` (own MGS-QR solver, no scorer imports) and
  closed-form fixtures with verified teeth (a heterogeneous rig `hetdem` provably
  closes; a packing rig it provably cannot; a uniform-demand redundancy rig). **Measured
  on the OLD pilot corpus before this registration, stated as context, never a bar:**
  `t1x` closes 27/35 of the pilot's firing datasets at median 1.0, and the extended
  pooled closure (krank + demand-krank + hetdem + futureint + linkrank, one coefficient
  set) is **0.892** (up from the frozen 0.648) — the environment is pointwise-closed
  against the extended competitor too, so the pivot must beat *this* floor, not the
  weaker frozen one.
- **The ladder** (order fixed, no post-hoc rungs): H0 config-only scarcity squeeze → H1
  + per-instance `demand_spread` U[0.5, 2.0] (`cap_mode: alpha_mean`) → H2 +
  `replica_overlap: true` → H3 + `dag_instances: 2` (8-task) at the doubled-α
  correspondence. Fresh seed blocks per rung (H0 3001–3017, H1 3101–3117, H2 3201–3217,
  H3 3301–3317), none previously used. Each rung carries a paired separable control
  (same grid/seeds, `HEROSIM_DATA_LOCALITY`/`HEROSIM_OUTPUT_SIZE_BYTES` unset).
- **Bars, fixed now:** S1 firing stratum `r_exact.frac_gt_5pct ≥ 0.25` (pilot baseline
  0.172); S2 (kill bar) `t1x` per-dataset repair fraction median on the firing stratum
  **< 0.5 on `mean_tied`**; S3 (pooled) extended pooled closure `median_mean_tied ≤ 0.5`
  on the firing stratum; S4 (attribution guard, the §9d lesson) the parent-coupling
  block alone (hop+coupling) must close **< 0.8 median**, full block-ablation table
  (kint/quad/occupancy/parent-coupling/hetdem/futureint) reported — a rung that fires
  loudly but whose closure is carried by one pointwise block is pointwise-reachable
  regardless of R_exact, exactly the failure mode that would have passed the 8-task
  corpus on R_exact alone. Tie-band direction agreement (`registered` vs `pessimistic`)
  is binding across every bar; disagreement is VOID-TIE-INDETERMINATE, never a pass.
- **Readings:** first rung passing S1–S4 with S0 clean ⇒ **PIVOT-CANDIDATE** — licenses
  drafting a v3 stage-2-style training registration on that corpus, nothing more.
  All rungs FAIL ⇒ **FAIL-BY-EXHAUSTION** — terminal for the static environment pivot;
  the fork (dynamic/closed-loop environment vs closing route B's GNN argument for the
  paper) returns to the user, each direction needing its own registration. A VOID rung
  (tie-indeterminate / infeasible / generation) is neither pass nor fail; the ladder
  continues past it.
- **Registered supersessions:** the v2-era "8-task has no gate role" decision is
  superseded for this pivot — H3 is in scope. Scarcity stays **label-side only** (same
  claim shape as stage 2 — offline decode regret; physics enforcement is a stage-3
  matter), per the user's 2026-08-27 decision.
- **Build items landed before registration** (commits `ac2b162`, `f7aaae2`, `dcb8e12`,
  `96abe6a`, all before `019bdcb`): `hetdem`/`futureint` T1 blocks + `t1hd`/`t1x` scorer
  arms + independent verifier extension + fixtures; `demand_spread` generator grid key +
  `cap_mode` scorer option (`alpha_max` default unchanged / `alpha_mean` /
  `{"absolute": x}`); `replica_overlap` generator grid key, discovered live-path
  crashes fixed (`precreate_replicas`'s own disjoint-platform dedup silently dropped
  every task type but the first sharing a platform; a double `platform.initialized`
  SimPy-Event trigger; `DeterminedOrchestrator`'s `average_contention` seeding skipped
  for a second type sharing an already-claimed platform) — all three unrelated to the
  co-sim brute-force sweep, which already handled overlap correctly; ladder presets
  `route_b_pivot_h0`..`h3` (204-shape each, strictly nested). Every addition is opt-in;
  every frozen artifact reproduces byte-identically without the new flags/keys.

**Status: REGISTERED AND SIGNED OFF at `019bdcb`. Phase B (the ladder) may start.**

---

#### route_b_env_pivot_v1 — LADDER PROGRESS (2026-08-27, IN FLIGHT — no rung readable yet)

**Full detail: `docs/lineages/route_b_env_pivot_v1/ladder-findings.md`.** Summary and status only here. **No bar,
threshold, grid, α ladder or reading rule has been changed by any of this work.**

**Rung status.** H0 **VOID-INFEASIBLE** (recorded in
`simulation_data/route_b_pivot_h0_reading.json`, amended in place). H1 **VOID-INFEASIBLE**
— same shape: no α on the registered ladder has clean counters. H2 **VOID-GENERATION** —
102/204 SUCCESS, below §3's 204/204 requirement. H3 not attempted. S1–S4 have not been read
on any rung, so **nothing is yet known about whether the pivot environment contains
pointwise-irreducible structure.**

**Why the rungs keep voiding — three distinct causes, all measured:**

1. **No α is both clean and binding.** Fine sweep on H1: at α=3.9 the constraint binds on
   204/204 with 80 stuck; at α=4.0 stuck is 0 and it binds on **0/204**, byte-identical to
   the unconstrained anchor. Relaxing α is therefore not a route to a readable rung and
   must not be attempted as one.
2. **On H0/H1 `greedy_stuck` is a configuration artifact.** It is fully explained by
   single-node confinement — no dataset with 0 confined tasks is ever stuck, and the 64-row
   arm *always* has exactly 2, in both rungs. Root cause is FCFS allocation in
   `generate_infrastructure.py:625-660`: early task types consume the platforms, and the
   confined set is always exactly `('rf','cnn')`, the last two in iteration order.
   **Raising `replica_server_percentage` does NOT fix this** — two full 204-dataset probes
   give an identical confinement histogram `{0:102, 2:102}` at 2, 3 and 4 hosting nodes.
   The knob changes how many nodes are *eligible*, not how the allocator spreads across
   them.
3. **On H2 `greedy_stuck` is decoder myopia instead.** `replica_overlap` (H2's own
   registered lever) drives confinement to **zero** and produces the contested-slot shape
   the pivot wanted. What remains is provably a decoder limitation: over identical
   candidate sets, **backtracking rescues 66/66 (α=2.0) and 27/27 (α=2.4)** — 100% — and a
   feasible plan exists in the enumerated sweep in every stuck case.

**Why H2 cannot be read anyway:** `replica_overlap` and the `per_server=1` arm are
incompatible under the no-replica-reuse mask. Four tasks need four distinct platforms;
overlap collapses that arm onto two, so **zero** unique assignments exist (proven by
enumeration: skipped datasets have 4 tasks / 2 distinct platforms; surviving ones have 4
platforms and exactly 4! = 24 plans).

**Gate-tool defects found and fixed en route** (details in GATE TOOLS): the `decode_regret`
tie-break artifact and the `r_exact` greedy-censored denominator (both `2fa4b50`, both
additive/auditable, byte-identity verified on the frozen pilot); the verifier's missing
`demand_scale` (`90a3c1b`), which would have voided H1's S0 gate for a spurious reason and
was caught only because Phase 4 was run before Phase 6. One defect is recorded **unfixed**:
skip reasons are mislabelled `too_many_combinations` under `replica_overlap`. *(Fixed later
the same day — see the SECOND PASS entry below and GATE TOOLS.)*

**AMENDMENT 1** (`docs/lineages/route_b_env_pivot_v1/screen-amendment-1.md`, commit `3719aad`) is signed off
and landed: the S0 control definition gains `HEROSIM_STORAGE_NEUTRAL=1`, because the
registered ablation never produced separable physics — an always-on storage-tier
parent-locality branch charges ~0.0156 s per task based on where a task's *parents* ran.
The `node_disk_v2` cold-pull hypothesis is **FALSIFIED** (contingency `cold>=2 & genuine =
0/31`; `averagePullTime = averageColdStartTime = 0.0`). H0 is not regenerated under it —
that rung is void on feasibility regardless.

**Open decision, with the user.** Five options are laid out in
`docs/lineages/route_b_env_pivot_v1/ladder-findings.md` §6, each needing its own amendment; the two that
measurement has already **ruled out** are relaxing α and raising
`replica_server_percentage`. The best-evidenced remaining option is a backtracking decoder
(100% rescue), which would make `greedy_stuck` measure the environment rather than the
decoder.


---

#### route_b_env_pivot_v1 — SECOND PASS (2026-08-27, later session): three findings were arm-scoped

**Full detail: `docs/lineages/route_b_env_pivot_v1/ladder-findings.md` §9.** **No bar,
threshold, grid, α ladder or reading rule has been changed by any of this work either.**
S1–S4 remain unread on every rung; no rung's status changes.

**Preflight, all green.** Full suite 326 passing at `8acfac1` (338 after this session's
tests). Independent verifier agrees to **1e-9 on every cell**: H0 612 cells / 1,326 repair
values, H1 542 / 1,046 — re-run after every edit below and still green. H2 was **not**
verified, deliberately: it is VOID-GENERATION and its half-corpus must not be read.

**The session's rule, applied to this document's own findings:** every number states the
arm it was measured on.

1. **`greedy_stuck` is decoder myopia on EVERY rung — supersedes §3 and §4.1.** §4.1's
   100% backtracking rescue was measured on H2 only, and within H2 on the 102 datasets that
   generated — one arm. Extended over identical candidate sets, demands, caps and option
   ordering: **H0 95/95 and 87/87, H1 100/100 and 83/83** — 365/365, on top of H2's 93/93,
   for **458/458 across the ladder**. Every stranded dataset had a feasible plan in its own
   enumerated sweep. So `greedy_stuck` has never measured the environment on any rung, and
   it is **logically redundant** with `no_feasible_rows`.
2. **§3's causal claim holds at α=3.0 and fails at H1's registered primary α=2.0.** "No
   dataset with zero confined tasks is ever stuck" is true on H0 at both α and on H1 at
   α=3.0, and false at H1 α=2.0, where **71 of 102 zero-confinement datasets are stuck** —
   H1's `alpha_mean` caps stop tracking the sweep max, so the 16-row arm's caps bind with no
   confinement at all. Same defect class on the α axis instead of the arm axis.
3. **`no_feasible_rows` is confounded with the arm and had no breakdown reporting it** —
   the stricter censor, one counter over from the greedy denominator fixed earlier the same
   day. At H1 α=2.0 all **70/70** censored datasets are in the 64-row arm and **0** in the
   16-row arm, so `n_exact_scored = 134` is 102-of-102 on one arm and 32-of-102 on the
   other, and `n_greedy_scored = 34` is `{16: 31, 64: 3}`. Now reported as `censoring_by_arm`
   in every rung's artifact (additive; **0 pre-existing values moved**, byte-identity
   verified against the frozen `route_b_pivot_h0_rtt.json`; commit `f407f91`).
4. **H1's paired separable control was never generated.** `gnn_datasets_route_b_pivot_h1_ctrl`
   is an **empty directory**. Moot while H1 is VOID-INFEASIBLE, but SCREEN §3 requires it and
   S0 reads a bar on it, so it must exist before H1 is ever read. Nothing in the record said
   it was missing.

**Gate-tool defect fixed** (details in GATE TOOLS): the skip-reason mislabelling recorded
*unfixed* above is now fixed — `classify_empty_combinations` compares the pre-uniqueness
product against the threshold before attributing, names `uniqueness_exhausted` separately,
refuses to guess (`unknown`, logged loudly), and carries diagnostics so the census is
legible (commit `f407f91`). 9 tests, led by the `replica_overlap` arm the original
assumption never saw. The
102 H2 `skip_reason.json` files already on disk are left as the old engine wrote them.

**Recommendation → AMENDMENT 2, signed off the same day.** See the next entry.


---

#### route_b_env_pivot_v1 — AMENDMENT 2 SIGNED OFF AND LANDED (2026-08-27); H0/H1 counters re-read

**Registration:** `docs/lineages/route_b_env_pivot_v1/screen-amendment-2.md`, signed off by
the user 2026-08-27, amending `screen-preregistration.md` @ `019bdcb` as already amended by
`screen-amendment-1.md` @ `3719aad`. **Scope: the scorer's offline decoder only.** No bar,
threshold, grid, α ladder, seed block, corpus or reading rule changed. §3's registered
fallback keeps its exact wording — the `greedy_stuck > 0` disjunct simply stops firing,
because the condition it tests stopped being true.

**What changed.** `greedy_masked_plan`'s single forward pass is replaced by
`complete_masked_plan`: same task order, same option ordering, same replica-reuse mask,
same capacity test, same tie-breaks — a dead end backtracks instead of returning `None`.
Justified by the 458/458 rescue measured before sign-off.

**§5's obligations are discharged, not promised** (full table in the amendment §8):

- `legacy_forward_only` reproduces the pre-amendment counters **exactly** on every cell of
  H0, H0-control and H1 — the both-numbers rule, from the same run.
- **Deviation logged:** `greedy_stuck` → 0 on every cell. H0 α=2.0 95→0 (n_greedy_scored
  109→204), H0 α=3.0 87→0 (117→204), H1 α=2.0 100→0 (34→134), H1 α=3.0 83→0 (121→204).
- **Byte-identity** enforced in `score_dataset` on every dataset, not as a one-off: moving
  a plan the forward-only decode already found raises. Three corpora re-scored, no raise;
  explicit diff shows **0 non-greedy keys moved** — `r_exact`, every repair, every LS
  statistic and every band untouched.
- **Independent verifier**, own complete search, no scorer import: **1,766 (dataset, α)
  cells agree to 1e-9** (H0 612, H0-ctrl 612, H1 542), 3,398 repair values, 0 ties accepted.
- **B1 acceptance re-passes.** The production *serving* decoder is forward-only and out of
  scope, so `--check-decoder` compares it against `legacy_forward_only` (408 cells on H0,
  182 stuck cells matched). Pre-amendment reports are read as forward-only at the top
  level, announced once per run — the **frozen stage-1 pilot artifacts stay checkable
  without being re-scored**.

**Rung status — counters only. NO S-BAR HAS BEEN READ ON ANY RUNG.**

| rung | reads at | `no_feasible_rows` | `greedy_stuck` | binds on | status |
|---|---|---|---|---|---|
| H0 | **α=2.0** (its registered primary) | 0 | 0 | 204/204 | counters CLEAN — no longer VOID-INFEASIBLE |
| H1 | **α=3.0** (tightest clean α on its ladder; α=2.0 has nofeas 70) | 0 | 0 | 204/204 | counters CLEAN — no longer VOID-INFEASIBLE |
| H2 | — | — | — | — | **still VOID-GENERATION** (102/204; untouched by this amendment) |

**What still blocks a reading.** S0 is a VOID gate and includes the paired separable
control. H0's control exists (204/204); **H1's does not** — `gnn_datasets_route_b_pivot_h1_ctrl`
is an empty directory, so H1 cannot pass S0 until it is generated under AMENDMENT 1's
`HEROSIM_STORAGE_NEUTRAL=1` definition. H0's control was generated under the **old**
definition, which AMENDMENT 1 established never produced separable physics. Reading either
rung's S1–S4 is a separate decision and has not been taken.


---

#### route_b_env_pivot_v1 — H0 and H1 controls generated under AMENDMENT 1; both PASS S0 (2026-08-27)

**Decision.** The user asked for all control groups to be generated, 2026-08-27. This
**supersedes AMENDMENT 1 §5's** "H0 is not re-generated under the amended definition" —
§5's reasoning was that H0 was VOID on feasibility regardless, and AMENDMENT 2 cleared
H0's counters the same day, so the premise is gone. §5 is not standing guidance any more.

Scope: **controls only.** No main (Arm S) corpus was regenerated, no threshold, bar, grid,
α ladder or reading rule moved, and **S1–S4 were not read on any rung.**

**Generated** (grid + seeds identical to each rung's main corpus, which is what "paired"
means — H0 3001–3017, H1 3101–3117):

| corpus | result |
|---|---|
| `gnn_datasets_route_b_pivot_h0_ctrl` | regenerated, 204/204 SUCCESS, 0 skips, 0 truncated sweeps |
| `gnn_datasets_route_b_pivot_h1_ctrl` | generated for the first time, 204/204 SUCCESS, 0 skips, 0 truncated sweeps |

Both carry `num_placements` `{16: 102, 64: 102}` — cell-for-cell paired with their mains.
The old H0 control is retained as `gnn_datasets_route_b_pivot_h0_ctrl_OLDDEF`.

H2's control was **not** generated: `replica_overlap` and the `per_server=1` arm are
structurally incompatible (4 tasks need 4 distinct platforms, overlap leaves 2), which comes
from the topology rather than the physics env vars, so an H2 control would be crippled
102/204 identically. That needs `ladder-findings.md` §6 options 3/4 and its own amendment.
H3 has never been generated at all.

**S0 result — the bar is `r_exact.frac_gt_1pct ≤ 0.02`, whole band reported.** Denominators
are `{16: 102, 64: 102}` at both reading α with zero censoring, so neither statistic is
concentrated in one arm.

| rung | α | `registered` | `mean_tied` | `optimistic` | `pessimistic` | S0 |
|---|---|---|---|---|---|---|
| H0 control | **2.0** (registered primary) | 0.0098 | 0.0000 | 0.0000 | 0.0098 | **PASS** |
| H1 control | **3.0** (its reading α) | 0.0000 | 0.0000 | 0.0000 | 0.0000 | **PASS** |

`registered` and `pessimistic` agree in direction on both, so neither is
VOID-TIE-INDETERMINATE.

**AMENDMENT 1 §3's falsifiable prediction is confirmed, not falsified.** Against the same
H0 control grid under the *old* definition (`registered` 0.0784 / `mean_tied` 0.0392 /
`optimistic` 0.0196 / `pessimistic` 0.0833), the storage-neutral lever cuts `registered`
**8×** and takes `mean_tied` to exactly **0**. The lever does what §4 says it does.

**Verification, all four steps.** Generation integrity above; independent verifier
(`--check-repairs`, own solver, no scorer imports) **612 (dataset, α) cells agree to 1e-9**
on the H0 control and **542** on H1's, 1,326 and 1,046 repair values, 0 machine-precision
ties accepted; per-arm histogram read before quoting, above. `legacy_forward_only`
reproduces the pre-AMENDMENT-2 decoder from the same run — 102 stuck (H0 ctrl α=2.0) and
101 (H1 ctrl α=3.0), **all in the 64-row arm**, `rescued_by_completion` 102 and 101 — so
the 458/458 rescue reproduces on the controls too. `legacy_greedy_censored` degenerates to
the `r_exact` block, as expected post-AMENDMENT-2.

**Two VOID-GENERATION corpora were produced and discarded first, and the cause is a
generation-recipe trap worth knowing** (tool row in GATE TOOLS). The first pass omitted
`HEROSIM_COSIM_KEEP_ALIVE=1000000 HEROSIM_RETAIN_TASK_TIMES=1` — half the corpus recipe
(`route_b_v1` Artifacts) which AMENDMENT 1 §3 carries forward implicitly as "everything
else about the control is unchanged". At the default 30 s keep-alive the autoscaler evicts
a forced replica mid-episode and the episode dies with `Invalid forced placement … not in
replicas`. Both rungs still reported **204/204 SUCCESS**: the damage was 13 (H0) and 14 (H1)
truncated sweeps plus one hard failure, visible only in `placement_metadata.json`
(`sweep_complete: false`), and it scattered `num_placements` across 12–14 distinct values so
the controls were no longer arm-comparable with their mains. **Attribution was measured, not
assumed** — regenerating the whole H0 control grid at HEAD with *no* lever reproduced both
symptoms (9 truncated, same scatter), and two no-keep-alive runs of the identical command
disagreed with each other (9 vs 13 truncated, different histograms), which is the
nondeterministic set-order tie-break `executecosimulation.py:89-99` documents.
`HEROSIM_STORAGE_NEUTRAL` accounts for neither symptom; its only effect is the cost
arithmetic (ds_00067 best RTT 2.385686 → 2.355926).

**Frozen reports:** `simulation_data/route_b_pivot_h0_ctrl_amd1_rtt.json` and
`route_b_pivot_h1_ctrl_amd1_rtt.json` (`--include-per-dataset`, cap modes `alpha_max` and
`alpha_mean` respectively). The older `route_b_pivot_h0_ctrl_rtt.json` is left untouched as
the OLDDEF reading.

**What this does and does not unblock.** S0 is now clean on H0 and H1, so both rungs are
readable for the first time. **Whether to read S1–S4 is a separate decision and has not been
taken.** H2 remains VOID-GENERATION.


---

#### route_b_env_pivot_v1 — S1–S4 read on H0 and H1: H0 VOID-TIE-INDETERMINATE, H1 FAILS S1, and S2 is uncomputable on this grid (2026-08-27)

**Decision.** The user authorised reading the bars, 2026-08-27, immediately after the
controls landed. Read at each rung's reading α on the firing stratum, whole tie band,
per-arm denominators checked first. **No threshold, bar, grid, α ladder or reading rule
moved.** Frozen artifacts: `simulation_data/route_b_pivot_{h0,h1}_bars_rtt.json`.

**S1 — `r_exact.frac_gt_5pct ≥ 0.25`.** Denominators `{16: 102, 64: 102}`, zero censoring
on both rungs, so neither statistic is arm-confounded.

| rung | α | `registered` | `mean_tied` | `optimistic` | `pessimistic` | reading |
|---|---|---|---|---|---|---|
| H0 | 2.0 | 0.1667 FAIL | 0.2402 FAIL | 0.0735 FAIL | **0.2549 PASS** | **VOID-TIE-INDETERMINATE** |
| H1 | 3.0 | 0.1716 FAIL | 0.1961 FAIL | 0.1569 FAIL | 0.2059 FAIL | **FAIL** |

H0's `registered` and `pessimistic` land on opposite sides of the bar, which §4 makes
VOID-TIE-INDETERMINATE, never a pass. H1's four members agree, so its FAIL is clean.
`mean_tied` 0.2402 against a 0.25 bar is a near-miss, and §4 is explicit that a near-miss is
a FAIL. S1's statistic is **unchanged by AMENDMENT 2** — the pre-amendment frozen
`route_b_pivot_h0_rtt.json` carries the same 0.1667, confirming "0 non-greedy keys moved".

**S2 — the kill bar — cannot be computed on this grid at all.** S2 reads the per-dataset
`t1x` repair fraction, and `t1x` is **refused by the saturation guard on 204/204 datasets in
both rungs, in both arms**. The guard needs `n_rows ≥ 2 × n_params`; measured on this grid:

| competitor | `n_params` | needs | 16-row arm | 64-row arm |
|---|---|---|---|---|
| `t1` (registered blocks) | 21 | ≥ 42 rows | refused | OK |
| **`t1x` (the S2 competitor)** | **41** | **≥ 82 rows** | **refused** | **refused** |

**Root cause: the pivot's own scarcity squeeze shrank the sweeps below the competitor's
fittability threshold.** The bars were calibrated on the stage-1 pilot corpus, whose sweeps
are **256–1248 rows** (`{256:9, 320:9, 432:8, 504:7, 576:28, 672:28, 756:34, 840:29,
1024:15, 1152:25, 1248:12}`) — every one of them fits `t1x`. H0 drops
`server_node_counts` 6→4 and `replica_configs` from 2–3/server to 1–2/server, which is what
creates the contention the screen exists to test, and it takes the sweeps to **16 and 64**.
Nothing could have caught this before a rung was generated: the grid and the bars were both
registered, and their incompatibility only exists in the product.

**S3 / S4 — initially reported as blocked for the same reason; that was only half right,
and the other half was a tool defect, now fixed (see the next entry).**
`route_b_coefficient_transfer.py` raised on the first firing dataset in the 16-row arm
(`repair over blocks ['kint','quad','cap','hop','coupling'] hit the saturation guard —
refusing to report an interpolated zero`), so no transfer report existed on either rung.
But S4's own bar block `hop+coupling` is 7 parameters and fits in **both** arms, and S3's
statistic is a **pooled** fit that cannot interpolate at all — neither was blocked by the
grid; both were blocked by an unrelated arm's refusal aborting the run. **S3 and S4 are
read in the next entry.** What survives the guard on the firing stratum, by arm:

| arm | H0 α=2.0 (firing 34) | H1 α=3.0 (firing 35) |
|---|---|---|
| `1int` | 34/34 `{16:24, 64:10}` | 35/35 `{16:18, 64:17}` |
| `kint` / `t1` / `t1hd` / `lnk` / `t1lnk` | 10/34 `{64:10}` | 17/35 `{64:17}` |
| **`t1x`** | **0/34** | **0/35** |

Any S3/S4 number from this corpus would be over the 64-row arm only, on 10 and 17 datasets.

**Arm-scoped diagnostic, explicitly not a bar reading** (`registered` point values, 64-row
arm, no tie band): on **H0** every computable competitor closes **0.000** of the regret on
the median firing dataset — the structure survives all of them. On **H1** `t1`/`t1hd`/`lnk`/
`t1lnk` close **1.000** at the median (mean 0.649) while `kint` alone closes ~0, so the
closure is carried by the `quad`/`cap`/`hop`/`coupling` blocks. That is the S4 attribution
worry in miniature, and it points the same way as H1's S1 FAIL. On n=10 and n=17 in one arm
it is a hint, not a result.

**S0 is clean on everything reachable and cannot be fully discharged.** Control bar PASS on
both rungs (previous entry); counters clean; positive-control suite **48 passed**;
independent verifier `--check-repairs` agrees to 1e-9 on all four corpora (H0 612, H0-ctrl
612, H1 542, H1-ctrl 542). But S0 also names `--check-blocks` and `--check-krank` with
`--extended-blocks`, and **both take the transfer report that cannot be produced** — so two
of S0's three verifier passes are unreachable on this grid.

**Reading.** H0 is **VOID-TIE-INDETERMINATE** on S1. H1 **FAILS** S1, which is terminal for
the rung on its own, since the bars are a conjunction. **The ladder is not exhausted** —
H2 is VOID-GENERATION and H3 has never been generated — so this is **not**
FAIL-BY-EXHAUSTION, and no verdict on the pivot as a whole is licensed yet.

**One open question that is the user's, not this entry's.** §4 says any S0 failure makes a
rung VOID. Whether *unreachable* verifier sub-gates count as an S0 failure is a reading-rule
question the registration does not answer: on the strict reading both rungs are VOID and
H1's FAIL is not readable either; on the lenient reading H1's FAIL stands. **Deciding it
here would be moving a reading rule, so it is not decided here** — it needs an amendment.
The same amendment is the natural home for the S2 problem, whose options are visible but
untaken: re-register the kill bar against a competitor that fits a 64-row sweep, widen the
grid's `replica_configs` so sweeps clear 82 rows (which loosens the very squeeze H0 exists
to create), or pool across datasets instead of fitting per-dataset.

> **Superseded the same day, by measurement — see the 2026-08-27 entry "S2 is a grid
> problem, not a competitor problem" below.** Two of the three sentences above are wrong.
> The S0 reading-rule question is **moot**: the transfer-tool fix made every verifier
> sub-gate reachable and they pass, so S0 is fully discharged on both rungs and no
> amendment is needed for it. And widening `replica_configs` does **not** loosen the
> squeeze — the squeeze is `server_node_counts=[4] × replica_server_percentage=0.5`,
> i.e. exactly 2 hosting nodes, which `per_server` does not touch; measured `{2: 204}` on
> every rung including the widened probe. The remaining live option is the grid one, and
> it is **AMENDMENT 3**.


---

#### route_b_env_pivot_v1 — the transfer tool's abort-on-refusal fixed; S3 and S4 read on both rungs (2026-08-27)

**What was actually wrong.** The saturation guard is correct and stays: a fit with fewer
than `2 x n_params` rows could interpolate the sweep, so its regret is refused rather than
reported as a zero. What was wrong is what `route_b_coefficient_transfer.py` did with a
refusal — **one refused arm raised and aborted the entire run.** The tool already had the
right contract, but only on its `t1x` arm (try/except, record `saturated`, exclude from the
median); every other per-dataset call site aborted.

That cost the screen two bars it could have read all along:

| bar | its column set | params | needs | 16-row arm | 64-row arm |
|---|---|---|---|---|---|
| **S4** | `hop+coupling` alone | 7 | 14 rows | **fits** | **fits** |
| **S3** | pooled, one coefficient set | 82–83 | 1024–1376 rows available | **n/a — pooled** | **n/a — pooled** |
| S2 | `t1x` per dataset | 41 | 82 rows | refused | refused |

Only **S2** is genuinely blocked by the grid. S4's bar block fits in both arms, and S3's
statistic is fit across the whole corpus at once (per-dataset intercepts, one shared
coefficient set), so no single dataset's sweep width can make it interpolate — measured
**1024 rows against 82 parameters** on H0 and **1376 against 83** on H1, both clearing the
same 2x criterion comfortably. S3 was withheld only because it was nested inside a
per-dataset "did every arm fit?" condition that does not apply to it.

**The fix** (`scripts_cosim/route_b_coefficient_transfer.py`): a dedicated
`SaturationRefusal(RuntimeError)` raised by `Cell.repair` / `Cell.repair_band`, with
`try_repair` / `try_repair_band` tolerating **exactly** that and nothing else — `except
RuntimeError` at a call site would also have swallowed "no feasible rows", "non-positive
optimum" and the firing-set disagreement, all of which must stay fatal. Every per-dataset
arm now reports `n_fitted` / `n_saturated` and a `by_arm` breakdown keyed on the
unconstrained sweep size, and takes its median over the fitted subset; the pooled block is
de-nested and reports its own rows-vs-parameters headroom. A cell with nothing fitted
reports `median: null` and a named `VOID-CELL-B-UNFITTABLE`, never a zero.

**No registered semantics moved.** Byte-identity verified where it claims to be inert: on
the stage-1 pilot corpus (256–1248-row sweeps, nothing refuses) the post-fix artifact is
diffed against a pre-fix baseline from the same commit — **0 pre-existing values moved**,
every added key being one of the refusal-accounting fields. 12 new tests in
`tests/test_route_b_transfer_saturation_refusal.py`, **all 12 verified to fail against the
pre-fix code**, on a rig that is the same firing cell at two sweep widths (54 rows vs 36,
identical `r_base` 7.2848) so "fits vs refused" cannot be confounded with a physics
difference.

**S4 — attribution guard. Bar: `hop+coupling` alone closes < 0.8 median on the firing
stratum.** Read on the **full** stratum, no refusals, so no arm confounding:

| rung | α | `hop+coupling` median | n | S4 |
|---|---|---|---|---|
| H0 | 2.0 | **0.0000** | 34/34 | **PASS** |
| H1 | 3.0 | **0.0000** | 35/35 | **PASS** |

The parent-coupling block closes *nothing* on either rung — the failure mode S4 exists to
catch is absent. `pessimistic` is by construction ≤ `registered` and clipped at 0, so both
band members sit far below 0.8 and agree in direction. (The `ablation()` arms report a
point value, not a tie band — a pre-existing gap, harmless at 0.0 vs 0.8, but it means S4's
band-agreement cannot be checked *in general* from this artifact.)

Full table, with the denominators the fix now exposes. The arms that include `kint` or
`hetdem` are **64-row-arm only** (10/34 and 17/35) and must be read as such:

| arm | H0 median (n) | H1 median (n) |
|---|---|---|
| `quad` | 0.0000 (34) | 0.0172 (35) |
| **`parent-coupling (hop+coupling)`** | **0.0000 (34)** | **0.0000 (35)** |
| `futureint` | 0.4393 (34) | 0.5702 (35) |
| `kint` | 0.0000 (10, 64-row only) | 0.0000 (17, 64-row only) |
| `occupancy (kint+quad+cap)` | 0.0000 (10, 64-row only) | 1.0000 (17, 64-row only) |
| `hetdem` | 0.0000 (10, 64-row only) | 1.0000 (17, 64-row only) |
| `full T1` | 0.0000 (10, 64-row only) | 1.0000 (17, 64-row only) |

**S3 — pooled closure. Bar: `median_mean_tied ≤ 0.5` on the firing stratum.**

| rung | α | `registered` | `mean_tied` (fair reading) | `pessimistic` | S3 |
|---|---|---|---|---|---|
| H0 | 2.0 | 0.2322 | **0.500000** (0.49999999999999944) | 0.2322 | **PASS by equality** |
| H1 | 3.0 | 0.1261 | **0.500835** | 0.1261 | **FAIL** |

**Both land on the bar, and H0's pass is decided at the sixteenth decimal.** This is not a
floating-point artifact of a degenerate statistic — checked: the surrogate fully separates
on 26/34 (H0) and 23/35 (H1) datasets, where `mean_tied == registered` exactly, and only
1/34 has the surrogate separating nothing. The median sits at 0.5 because the handful of
*tied* datasets (`n_tied` 2–16) carry `mean_tied` values of exactly 0.50000 and land on the
median position. So the reading is real, and it is genuinely a coin-toss at the threshold.
§4's "a near-miss is a FAIL, there is no third option" plainly covers H1; H0 satisfies
`≤ 0.5` by equality. **Flagged rather than adjudicated** — a bar this marginal is the
user's call, and moving it is not mine.

**S2 is unchanged and still uncomputable** — `t1x` needs 82 rows per dataset and the grid
yields 64 at most. No tool fix reaches it; it needs the amendment described in the previous
entry.

**Rung verdicts do not change.** The bars are a conjunction and S1 already decided both
rungs: **H0 VOID-TIE-INDETERMINATE**, **H1 FAIL**. What changed is that the screen is now
*readable* instead of *broken* — and what it says where it can now speak (S4 clean on both,
S3 on the knife-edge) is recorded above rather than lost to an abort.

**Naming mismatch, flagged not edited:** the artifact block carrying S3's statistic is still
called `krank_pooled_exploratory` and its own `note` says "Exploratory", wording from the
route_b_v1 era. The pivot registration promoted this quantity to a registered bar. Which
block S3 names is a registration matter, so the string is left alone.

**Frozen artifacts:** `simulation_data/route_b_pivot_{h0,h1}_transfer.json`.


---

#### route_b_env_pivot_v1 — S2 is a grid problem, not a competitor problem; and H3 does not generate as registered (2026-08-27)

**What was asked.** The previous entry left S2 uncomputable and named three options, two of
which turn out to be wrong. This entry measures the third and adds a rung-level defect
nobody had looked for. **No bar was moved and no corpus was registered** — the work here is
a probe and a drafted amendment.

**S2 becomes computable, and the fix is the grid.** Preset
`route_b_pivot_h2_widearm_probe` (H2's shape with `replica_configs` `per_server` 1→3 and
2→4) generated **204/204 SUCCESS**, `sweep_complete: true` on all 204, `num_placements`
histogram exactly **`{360: 102, 1680: 102}`**, zero skips. Scored at the registered
`--cap-mode alpha_mean`; the independent verifier agrees on **612 (dataset, α) cells to
1e-9** across 2,448 repair values, 2 machine-precision ties accepted. `n_exact_scored` reads
`{360: 102, 1680: 102}` — **zero censoring, so nothing below is arm-confounded.**

| bar | registered | reading at α=2.0 | |
|---|---|---|---|
| S1 `r_exact.frac_gt_5pct` | ≥ 0.25 | 0.2010 reg / 0.2059 mean_tied / 0.2108 pess | **FAIL**, band agrees |
| **S2 `t1x` per dataset** | **< 0.5** | **41/41 fitted, 0 saturated**, median 1.55e-15 | **PASS** |
| S3 extended pooled | ≤ 0.5 | median 1.55e-15; 49,080 rows vs 89 params | **PASS** |
| S4 `hop+coupling` | < 0.8 | median 0.0000, 41/41 fitted | **PASS** |

S2's `by_arm` is `{360: {fitted 15, saturated 0}, 1680: {fitted 26, saturated 0}}`, against
**refused 204/204** on the registered grid. Counters stay clean at the registered primary α
(`greedy_stuck` 0, `no_feasible_rows` 0, `saturated_fit_frac` 0.00), the α cliff survives
(α=1.5 is 204/204 infeasible), and contention binds **harder** than on the registered grid:
`componentwise_infeasible_frac` **0.91**, 143.5 feasible rows of 1020.

**The "widening loosens the squeeze" claim is FALSIFIED.** It was an assumption, written
into this node and `ladder-findings.md` and never measured. The squeeze is
`server_node_counts=[4] × replica_server_percentage=0.5` ⇒ exactly **2 hosting nodes**;
`per_server` sets how many platform slots sit on those two nodes, not how many nodes exist.
Hosting-node histogram is `{2: 204}` on H0, H1, H2 **and** the widened probe, while the
candidate pool moves 8/12 · 2/4 · 6/8. The earlier claim is withdrawn.

**H3 generates 0/204 as registered — a rung-level defect never previously measured.** Under
`replica_overlap` all task types draw from one pool of `per_server × n_hosting_nodes` slots,
and the combination generator requires **globally distinct** replicas across tasks. H3 is
`dag_instances=2`, so 8 tasks draw on that one pool and it must hold ≥ 8:

| grid | arm | pool | pre-uniqueness | result |
|---|---|---|---|---|
| H3 as registered | `per_server=1` | 2 | 256 | **0/12 `uniqueness_exhausted`** |
| | `per_server=2` | 4 | 65,536 | **0/12 `uniqueness_exhausted`** |
| candidate | `per_server=4` | 8 | 16,777,216 | 40,320 rows = 8P8 |
| | `per_server=5` | **9** | 43,046,721 | 362,880 rows = 9P8 |

Measured by `route_b_pivot_h3_genprobe_{registered,wide}` with
`MAX_PLACEMENT_COMBINATIONS_SKIP` raised to 5e9 so a skip cannot be the threshold. The
`per_server=5` pool is **9, not the 10 the arithmetic predicts** — one hosting node carries
fewer suitable platforms. The skip reasons also re-confirm the `f407f91` attribution fix on
a grid it had not been exercised on: `uniqueness_exhausted`, never `too_many_combinations`.
**Without an amendment H3 is a structural VOID-GENERATION and the ladder cannot be
exhausted.** §3's registered `MAX_PLACEMENT_COMBINATIONS_SKIP = 16,777,216` also needs
re-deriving: the comparison is a strict `>`, so it admits `per_server=4` with zero headroom
and skips `per_server=5` outright.

**AMENDMENT 3 is drafted and NOT signed off.** It proposes `replica_configs`
`[(0,4,0.7,0.5), (0,5,0.7,0.5)]` for H2 and H3 with a fresh seed block for H2, and moves
nothing else — S2's threshold and its `t1x` competitor stay exactly as registered; only the
corpus the bar is evaluated on changes. It states its own costs: H2/H3 stop being
cell-for-cell comparable with H0/H1, and it **predicts no pass** — the probed neighbour grid
failed S1 at 0.2010, and the amendment says so in advance so no later reading of it can be
post-hoc. The selection hazard is named there too: the per-arm split (0.1471 at
`per_server=3` vs 0.2549 at `per_server=4`) was visible before the pair was chosen, which is
why H2 gets fresh seeds 3401–3417 and why the choice is argued from H3's generability and
H2↔H3 comparability rather than from a firing rate.

**The other two open questions are closed without action.** The S0 reading-rule question
("do unreachable sub-gates make a rung VOID?") is **moot** — the transfer-tool fix made every
sub-gate reachable and they pass. S3's knife-edge is **left alone**: both rungs are already
decided by S1, so it changes nothing, and moving a threshold after watching it land on the
line is exactly what §6's "thresholds immutable once signed off" forbids. Worth *noting* for
H2/H3 that a bar both rungs land within 0.001 of may be low-information — noting, not moving.

**Frozen artifacts:** `simulation_data/route_b_pivot_h2_widearm_probe_{rtt,transfer}.json`.
Both are **probe** artifacts on an unregistered grid, not rung readings.


---

#### route_b_env_pivot_v1 — the pair AMENDMENT 3 proposes passes all four bars in probe (2026-08-28)

**Why this was measured.** AMENDMENT 3's §3 named `replica_configs` `[(0,4), (0,5)]` for
H2/H3 with only its H3 row measured; the H2 row was arithmetic. The 4-task shape cost 50
minutes locally against a cluster job for H3, so it was measured rather than predicted.

**Preset `route_b_pivot_h2_proposed_probe`**, deliberately on H2's **currently registered**
seeds 3201–3217 — already burned by the wide-arm probe — so that the fresh block AMENDMENT 3
reserves for the amended rung stays unseen. **204/204 SUCCESS** in 50.1 min,
`sweep_complete: true` on all 204, `num_placements` exactly **`{1680: 102, 3024: 102}`**
(the predicted 8P4 and 9P4, confirmed), hosting nodes `{2: 204}`. Independent verifier:
**612 (dataset, α) cells to 1e-9** over 2,448 repair values, **0** machine-precision ties.
Denominators `{1680: 102, 3024: 102}`, zero censoring.

| bar | registered | reading at α=2.0 (`alpha_mean`) | |
|---|---|---|---|
| S1 `r_exact.frac_gt_5pct` | ≥ 0.25 | 0.2843 reg / 0.2941 mean_tied / **0.3137 pess** / 0.2500 opt | **PASS**, band agrees |
| S2 `t1x` per dataset | < 0.5 | median 0.0000; **58/58 fitted, 0 saturated** | **PASS** |
| S3 extended pooled | ≤ 0.5 | median 0.0000; 143,136 rows vs 106 params | **PASS** |
| S4 `hop+coupling` | < 0.8 | median 0.0000; 58/58 fitted | **PASS** |

Counters clean (`greedy_stuck` 0, `no_feasible_rows` 0, `saturated_fit_frac` 0.00), α=1.5
still 204/204 infeasible so the cliff survives, and `componentwise_infeasible_frac` reads
**0.93** — the tightest contention of any corpus in this lineage. S2's `by_arm` is
`{1680: {fitted 24, saturated 0}, 3024: {fitted 34, saturated 0}}`.

**Per arm, because a pooled number one arm carries is this lineage's recurring defect:**

| arm | `per_server` | mean feasible rows | `cw_infeas` | S1 reg / mean_tied / pess |
|---|---|---|---|---|
| 1,680 | 4 | 232.9 | 0.90 | 0.2353 / 0.2549 / 0.2647 |
| 3,024 | 5 | 381.2 | 0.90 | 0.3333 / 0.3333 / 0.3627 |

Both arms fire and both clear the bar on `pessimistic`; the `per_server=4` arm's
`registered` member sits just under it at 0.2353, so **the pooled pass is carried more by
the `per_server=5` arm.** Recorded so the pooled number is not read as uniform.

**This is a probe, not a rung reading, and the distinction is load-bearing.** S0's paired
separable control for this grid **has not been generated**, and S0 is a VOID gate that can
still fail the rung outright. The corpus sits on seeds the amended rung will not use. The
transfer tool's own top-level `verdict` here is `VOID-KINT-CONFOUNDED`, inherited from the
route_b_v1 §9b/§9c machinery — not one of S1–S4, reported rather than omitted.

**What it means for sign-off.** AMENDMENT 3 stopped being a procedural fix for an
uncomputable bar. On the pair it proposes, all four bars pass, so signing it is likely to
produce the ladder's **first PIVOT-CANDIDATE** — which under §5 stops the ladder and
licenses drafting a v3 training registration. The amendment's §6 was rewritten to say this
in advance, and to name the selection hazard at full strength: a grid was adjusted after a
FAIL and the bar then passed. What bounds it — the H3-generability argument that selects
`per_server ≥ 4` with every S1 number unseen, the fresh seed block 3401–3417, the ungenerated
S0 control — is recorded there as bounds, not as absolution. **The decision is the user's.**

**Frozen artifacts:** `simulation_data/route_b_pivot_h2_proposed_probe_{rtt,transfer}.json`,
both probe artifacts on an unregistered grid.
