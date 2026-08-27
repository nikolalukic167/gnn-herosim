# route_b_env_pivot_v1 — ACTIVE

> **Status:** `ACTIVE` &nbsp;·&nbsp; **Index:** [LINEAGES.md](../../LINEAGES.md)

**Outcome.** Current work. Screen registered 2026-08-27; **no ladder rung is readable yet** — H0/H1 VOID-INFEASIBLE, H2 VOID-GENERATION, defeated by configuration and decoder artifacts before their bars are consulted.

**Related:** [route_b_v1](route_b_v1.md) · [route_a_v1](route_a_v1.md)

**Attachment:** [SCREEN PRE-REGISTRATION](route_b_env_pivot_v1/screen-preregistration.md)

**Attachment:** [AMENDMENT 1 — the S0 separable-control definition](route_b_env_pivot_v1/screen-amendment-1.md)

**Attachment:** [ladder feasibility findings (2026-08-27)](route_b_env_pivot_v1/ladder-findings.md)

> **Split note.** This node was carved out of a single 4,995-line `LINEAGES.md` on
> 2026-08-27; the section bodies below are byte-for-byte as written. An *above* / *below*
> pointer inside one may refer to a section that now lives in another node — the
> **Related** links say which.

## Record

Newest first; the sections themselves are in chronological order below.

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
skip reasons are mislabelled `too_many_combinations` under `replica_overlap`.

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
