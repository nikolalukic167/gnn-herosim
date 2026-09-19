# route_b_env_pivot_v1 — AMENDMENT 4: S0 measures additivity directly

> **DRAFT — NOT SIGNED OFF.** Drafted 2026-08-28 at `d79ea54`. Amends
> `screen-preregistration.md` @ `019bdcb`, as already amended by `screen-amendment-1.md`
> @ `3719aad`, `screen-amendment-2.md` (signed off 2026-08-27) and `screen-amendment-3.md`
> (signed off 2026-08-28). **Nothing here is live.** No rung may be read under it, and the
> amended H2 stays VOID until this is either signed off or rejected.

**Scope: how S0 is measured. Nothing else.** Not S0's threshold — the proposed statistic
needs its own, and §4 sets one from measured calibration. Not S1–S4, not a grid, not a seed
block, not an α ladder, not the physics, not the control's env recipe (AMENDMENT 1's
definition is untouched). **H0 and H1 are not regenerated; §5 shows their S0 verdicts
survive the change.**

⚠ **This amendment is proposed after S0 failed, which is exactly the shape of a bad
amendment.** §7 states the case that it is not one, and states what would make it one. Read
§7 before §3.

## 1. The problem this exists to solve

S0 is the screen's separability gate. It asks: **does an additive min-marginal surrogate
recover the constrained optimum on the paired separable control?** The bar is
`r_exact.frac_gt_1pct ≤ 0.02`, read at the rung's registered primary α.

The amended H2's control **failed it by 24×** (`optimistic` 0.4853, `mean_tied` 0.7696, max
regret 59.5%). The rung is VOID and its four bars — which the AMENDMENT 3 probe had passed —
are unreadable.

**The failure was uninterpretable for a full session**, and that is the defect this document
is about. S0 reads additivity *through* a decoder and a capacity cap, so a failure has at
least three candidate causes and the statistic cannot distinguish them:

1. the control arm's cost is genuinely not additive;
2. a bug in the decode path;
3. something in the cap path.

Eliminating (3) took an argument about `node_caps` being plan-independent. Eliminating (2)
was not possible from the statistic at all — it required building a separate measurement.
The independent verifier could not help: it agrees with the scorer to 1e-9 on all 612 cells
**using the same cap definition and the same additive-surrogate definition**, so it checks
the arithmetic and is silent on whether the cost is additive. A VOID gate whose failures
cannot be attributed is a gate that stops the screen without telling anyone what to fix.

## 2. The measurement

The co-sim sweep is exhaustive, so `placements/placements.jsonl` holds **every** plan with
its true `rtt`. Additivity can therefore be tested directly:

> Regress `rtt` on one-hot `(task, platform)` indicators over the **full** sweep. If
> `cost(plan) = Σ_t c_t(p_t)` exactly, the fit is exact: R² = 1, residual 0.

No decoder, no cap, no surrogate, no tie-break is in that path. Implemented as
`scripts_cosim/measure_route_b_additivity.py`, importing nothing from the scorer (the same
independence discipline as `verify_route_b_scorer_agreement.py`), with 8 tests on synthetic
sweeps whose additivity is known by construction.

### 2.1 Calibration — the controls that PASS S0 fit almost exactly

Whole corpora, every dataset, split on the `replica_configs` arm (`n_rows`):

| control corpus | S0 | arm | R² median | R² min | residual, median % of mean rtt |
|---|---|---|---|---|---|
| H0 ctrl | **PASS** | 16 / 64 | 0.999991 / 0.999757 | 0.4577 / 0.9032 | **0.159% / 0.721%** |
| H1 ctrl | **PASS** | 16 / 64 | 0.999975 / 0.999856 | 0.7204 / 0.8423 | **0.193% / 0.999%** |
| **H2 ctrl** | **FAIL** | 1680 / 3024 | **0.7826 / 0.8161** | 0.5331 / 0.5310 | **12.822% / 13.814%** |
| **H3 ctrl** | — | 40320 / 362880 | **0.3568 / 0.5258** | 0.1087 / 0.3063 | **9.538% / 10.498%** |

H3's control was measured after this document was first drafted (datalab job 719077) and is
included because it is a second, independent instance of the same failure — not because the
thresholds need it. §4's values are set from the *passing* rungs alone and would be
identical had H3 never been measured. Note H2 and H3 disagree in direction on the two axes
(H3 has the lower R² but the smaller residual, because `rtt` varies less across its plans),
which is why §3 requires **both** bars rather than either one.

This is the fact the amendment rests on: **the proposed statistic already separates the
rungs that pass S0 from the one that fails, by a factor of ~13 in residual, on the existing
frozen corpora, with no threshold chosen yet.** It is not a statistic that needs H2 to fail
in order to be interesting.

The R² *minimum* is low on a handful of datasets in every corpus, H0's control included
(0.4577). That is why §4's threshold is on the **median**, not the minimum — a few
degenerate datasets are expected and are not what S0 is asking about.

### 2.2 The residual is co-residency-shaped — which is the diagnosis S0 could not give

Pooled by the maximum number of tasks sharing one node, as % of each dataset's mean `rtt`:

| max tasks/node | H0 ctrl mean | H1 ctrl mean | **H2 ctrl mean** | H2 ctrl RMS |
|---|---|---|---|---|
| 2 | −0.056% | −0.029% | +2.736% | 12.416% |
| 3 | +0.028% | +0.033% | −0.968% | 13.619% |
| **4** | **+0.006%** | **−0.037%** | **−21.706%** | **24.691%** |

At full co-location the additive model **over-predicts** H2's true RTT by ~22%; H0/H1 show
no co-residency structure at all. A same-node **pair** term lifts H2's median R² only to
0.897 / 0.903, so roughly half the coupling is higher-order. (A same-`(node, platform)` pair
term contributes exactly zero — vacuous, since the sweep requires globally distinct
replicas.)

**This is what a control gate should produce on failure:** not "the surrogate mis-ranked",
but "the cost is non-additive, the coupling is concentrated at co-location, and it is not
purely pairwise." Every one of those is actionable; none of them is available from
`frac_gt_1pct`.

### 2.3 ⚠ The stated limit of S0-c: it is degenerate when the pool is barely larger than the task count

The co-residency breakdown has power only if co-residency actually *varies* across the
sweep. It does not on H3: 8 tasks in a pool of 8 or 9 over 2 hosting nodes means almost
every plan has the same occupancy profile, only two levels occur (4 and 5 on the fullest
node), and **both report mean residual 0.000%** — a near-constant indicator is absorbed
into the fit. So on H3 the measurement yields the *magnitude* of non-additivity and says
nothing about its *shape*, and **H2's co-location story is not established there.**

This is a limit on S0-c, the reporting clause, not on S0-a/S0-b, the bars — those are
unaffected and H3 fails both on both arms. **S0-c must be reported as `DEGENERATE` rather
than as a zero whenever fewer than three co-residency levels occur**, so an uninformative
breakdown is never mistaken for a measured absence of co-residency structure. That
distinction is exactly the kind this lineage has been bitten by before.

## 3. What is proposed

**S0 becomes a direct additivity test on the paired separable control**, replacing the
decoder-mediated one. Per rung, over the **full** sweep of **every** dataset in the control
corpus, split on the `replica_configs` arm:

- **S0-a (the bar):** the per-dataset **median** additive R², reported per arm, must be
  **≥ 0.99** on **every** arm.
- **S0-b (the bar):** the per-dataset **median** residual RMS, as % of that dataset's mean
  `rtt`, must be **≤ 2.0%** on **every** arm.
- **S0-c (reporting, not a bar):** the co-residency breakdown of §2.2 and the pairwise-model
  R² are reported on every rung, pass or fail, so a failure arrives with its diagnosis
  attached. **Reported as `DEGENERATE`, never as a zero, when fewer than three co-residency
  levels occur in the sweep** — see §2.3, which is why.

Both bars must pass for S0 to pass. Failure is a **VOID** for the rung, exactly as now — the
consequence of S0 is unchanged, only the measurement is.

**Unchanged and explicitly carried forward:** AMENDMENT 1's control definition and its
five-variable env recipe; the requirement that S0 is read *before* any S1–S4 number; the
generation-integrity obligations of AMENDMENT 3 §7; the independent-verifier 1e-9 sub-gate
on the scorer's own statistics.

## 4. Where the thresholds come from

Not from H2, and not chosen to make anything pass. From the two corpora that **already
pass** S0 as registered:

- worst median R² across all four passing arms: **0.999757** (H0's 64-row arm);
- worst median residual across all four passing arms: **0.999%** (H1's 64-row arm).

`R² ≥ 0.99` and `residual ≤ 2.0%` sit roughly a factor of two outside the worst passing arm
on each axis — loose enough that a genuinely separable rung is not failed by numerical
noise, tight enough that H2 fails by ~13× on residual and ~0.2 in R². **No value in the
range 0.95–0.999 for R², or 1.5%–10% for residual, changes any verdict on any existing
corpus.** The thresholds are therefore not knife-edges, and §7's "the bar was tuned to the
answer" objection has no purchase: any defensible choice gives the same three verdicts.

## 5. What happens to H0 and H1 — nothing

Under the proposed S0, H0's and H1's controls **pass on both bars on both arms** (§2.1:
worst median R² 0.999757 ≥ 0.99; worst median residual 0.999% ≤ 2.0%). Their S0 verdicts are
unchanged, so **their S1–S4 readings stand exactly as measured** and neither rung is
regenerated or re-read. The amendment cannot rescue or damage any existing verdict.

H2 fails the proposed S0 as it failed the registered one, and **H3's control fails it too**
(§2.1) — on both bars on both arms, and it would fail the registered S0 as well. **This
amendment does not unVOID H2 or H3.** Anyone reading it as a route to reading either rung's
bars has misread it — see §6.

Consequence, stated plainly because it is the real state of the screen: H2 and H3 are the
ladder's **only** `replica_overlap` rungs and both controls fail separability, so under
either the registered or the proposed S0 the ladder stands at **H0
VOID-TIE-INDETERMINATE / H1 FAIL / H2 VOID / H3 VOID — no PIVOT-CANDIDATE.** ⚠ That is
"the screen could not measure it", **not** "there is no exploitable joint structure in the
amended environment" — Arm S's bars have never been read on either overlap rung, and the
same grid shape passed all four bars in probe. Signing this amendment does not change that
and does not license the second claim.

## 6. What this amendment does NOT propose

- **It does not unVOID H2 or H3.** Both still fail S0 under §3, and the amended H2's S1–S4
  remain unread and unreadable.
- **It does not change the control's physics or env recipe.** AMENDMENT 1 stands. The
  finding that `HEROSIM_STORAGE_NEUTRAL` + locality-off is not separable at H2's
  co-location densities is *reported* here, not *fixed* here. Making the control genuinely
  separable is a different amendment, and §8 is where it lives.
- **It does not move S1, S2, S3 or S4**, their thresholds, their competitors, or their
  reading rules.
- **It does not touch any grid, seed block, α ladder or the squeeze.**
- **It does not license reading a bar on a VOID rung.**

## 7. Why this is not "moving the bar after it failed"

The objection is the right one to raise and it has to be answered on the merits, not waved
past. Four things distinguish this from bar-shopping, and the fourth is the one that matters:

1. **The consequence is unchanged.** S0 remains a VOID gate with the same power over the
   same rungs. Nothing becomes readable that was not readable before.
2. **The verdicts are unchanged.** H0 pass, H1 pass, **H2 fail** — under both the old and
   the new statistic (§2.1, §5). An amendment that moved a bar to rescue a rung would change
   at least one verdict. This one changes none.
3. **The thresholds are calibrated off the passing rungs, not the failing one** (§4), and no
   defensible value changes any verdict.
4. **It makes the gate strictly harder to satisfy by accident, not easier.** The registered
   S0 can pass on a non-additive corpus whenever the surrogate's argmin happens to land on
   the true optimum — which is precisely how the tie artifact (`gate-tools.md`, 2026-08-27)
   produced a misleading reading once already. The proposed S0 cannot: it tests the property
   S0 is *named* for, so a corpus that passes it is separable in fact and not by luck.

**What would make this a bad amendment**, stated so it can be checked rather than argued:
if it were signed off and H2 then *passed*. It does not (§5). If a future revision of this
document moves §4's thresholds such that H2 passes, that revision is bar-shopping regardless
of its justification, and this paragraph is the pre-registration that says so.

## 8. The alternative, and why it is not proposed here

**Fix the control's physics so it is genuinely separable — find the term that makes
co-location cheap and switch it off, then regenerate both H2 and H3 controls.** This is the
`separable-control-was-never-separable` lineage repeating: AMENDMENT 1 found one such term
(the storage-tier parent-locality branch) and neutralised it, and the measurement in §2.2
says at least one more survives at H2's densities.

It is the more *complete* fix and it should probably happen. It is not proposed here for
three reasons:

1. **The term has not been identified.** §2.2 characterises the residual's shape; it does
   not name the code path. Proposing to switch off an unidentified term is not a
   specification.
2. **It costs two 204-dataset control corpora and does not stand alone.** Even after
   regeneration, something has to *verify* the result is separable — and the only honest
   verifier is §2's direct test. The two are complementary, and this one is the prerequisite.
3. **It may not be achievable.** If the coupling is intrinsic to running four tasks on two
   hosting nodes, no env var switches it off, and the screen's answer is that S0 is not
   satisfiable at these densities. §2's statistic is what would establish that; the
   registered S0 could not distinguish it from a decoder bug.

A third option — **declare S0 unreadable on overlap rungs and drop it** — is rejected: S0 is
what makes every other bar mean anything, and a screen that reads S1–S4 without a working
separability control is measuring nothing.

## 9. Reporting obligations if signed off

- **Per-arm everything**, on the `n_rows` key. A median without its arm split is not a
  reading (route-b-preflight step 3).
- **S0-c reported on every rung, pass or fail** — the co-residency table and the pairwise R².
  The diagnosis is the point of the change; omitting it on a pass forfeits the benefit.
- **H0 and H1 re-reported once under the new statistic** (§2.1's numbers, from the frozen
  corpora, no regeneration) so the ladder is read cell-for-cell on one definition.
- **The frozen artifact** `simulation_data/route_b_pivot_additivity_controls.json` is the
  calibration of record; a re-run must reproduce it.
- **Move no other threshold.** S1–S4 read on their registered values, unchanged.

## 10. Sign-off

Amends `screen-preregistration.md` @ `019bdcb`, as already amended by
`screen-amendment-1.md` @ `3719aad`, `screen-amendment-2.md`, and `screen-amendment-3.md`.

**Signed off by the user: ☐ — NOT SIGNED OFF.**

**What sign-off would license, and only this:** recording S0 under §3's statistic and §4's
thresholds on the four control corpora already measured (H0, H1, H2, H3 — all four numbers
exist and are frozen in `route_b_pivot_additivity_controls.json` and
`route_b_pivot_h3_ctrl_additivity.json`). It would **not** license unVOIDing H2 or H3,
reading S1–S4 on any VOID rung, regenerating anything, or moving any other threshold.

**What rejecting it costs:** nothing already measured. The registered S0 gives the same four
verdicts. The difference is only whether future failures arrive attributable or not.
