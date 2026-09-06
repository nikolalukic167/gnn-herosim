# route_b_v1 — PARKED

> **Status:** `PARKED` &nbsp;·&nbsp; **Index:** [LINEAGES.md](../../LINEAGES.md) &nbsp;·&nbsp; **Record spans:** 2026-08-25 → 2026-08-28

**Outcome.** **Stage 1 PASS** — contention + coupling produces the non-pointwise structure five mechanisms and route A could not. **Stage 2 NO-GO-PREPROBE** (2026-08-26): a GNN cannot beat pointwise-plus-prefix on this environment even at memorization. Forked to `route_b_env_pivot_v1` — **that fork was PARKED 2026-08-28** (see its closing entry), so this lineage is PARKED with it: both measured results stand, and the program's effort moved to [objective_pivot_v1](objective_pivot_v1.md) (change the training objective, not the environment).

**Related:** [route_b_env_pivot_v1](route_b_env_pivot_v1.md) · [route_a_v1](route_a_v1.md) · [route_c_link_transfer_v1](route_c_link_transfer_v1.md)

**Attachment:** [STAGE 2 PRE-REGISTRATION (v2, corrected)](route_b_v1/stage2-preregistration.md)

> **Split note.** This node was carved out of a single 4,995-line `LINEAGES.md` on
> 2026-08-27; the section bodies below are byte-for-byte as written. An *above* / *below*
> pointer inside one may refer to a section that now lives in another node — the
> **Related** links say which.

## Record

Newest first; the sections themselves are in chronological order below.

- [route_b_v1 — stage 2 §9 pre-probe: **NO-GO-PREPROBE** (2026-08-26)](#route-b-v1-stage-2-9-pre-probe-no-go-preprobe-2026-08-26)
- [route_b_v1 — stage 2 build queue: A1 is genuinely T2, B6 closed, pilot cache built (2026-08-26)](#route-b-v1-stage-2-build-queue-a1-is-genuinely-t2-b6-closed-pilot-cache-built-2026-08-26)
- [route_b_v1 — STAGE 2 CORRECTED PRE-REGISTRATION (v2, signed off 2026-08-26)](#route-b-v1-stage-2-corrected-pre-registration-v2-signed-off-2026-08-26)
- [route_b_v1 — §9d: 8-task probe — pooled `krank` closure SURVIVES the doubling, attenuated (2026-08-26)](#route-b-v1-9d-8-task-probe-pooled-krank-closure-survives-the-doubling-attenuated-2026-08-26)
- [route_b_v1 — §9c: `kint` is not a T1 feature — **NO-GO-PREPROBE-T1 RETRACTED AS MEASURED** (2026-08-25)](#route-b-v1-9c-kint-is-not-a-t1-feature-no-go-preprobe-t1-retracted-as-measured-2026-08-25)
- [route_b_v1 — §9b coefficient transfer: **VOID-KINT-CONFOUNDED**, and the §9a bound is confirmed tie-robust (2026-08-25)](#route-b-v1-9b-coefficient-transfer-void-kint-confounded-and-the-9a-bound-is-confirmed-tie-robust-2026-08-25)
- [route_b_v1 — STAGE 2 PRE-REGISTRATION (recorded retroactively 2026-08-25)](#route-b-v1-stage-2-pre-registration-recorded-retroactively-2026-08-25)
- [route_b_v1 — stage 2 pre-probe zero: **NO-GO-PREPROBE-T1** (2026-08-25)](#route-b-v1-stage-2-pre-probe-zero-no-go-preprobe-t1-2026-08-25)
- [route_b_v1 — OUTCOME: **PASS (stage 1).** Contention + coupling produces the non-pointwise structure five mechanisms and route A could not (2026-08-25)](#route-b-v1-outcome-pass-stage-1-contention-coupling-produces-the-non-pointwise-structure-five-mechanisms-and-route-a-could-not-2026-08-25)
- [route_b_v1 — calibration freeze + three registration amendments (2026-08-25, before any gated corpus exists)](#route-b-v1-calibration-freeze-three-registration-amendments-2026-08-25-before-any-gated-corpus-exists)
- [route_b_v1 — PRE-REGISTRATION (written 2026-08-25, before any route B corpus exists)](#route-b-v1-pre-registration-written-2026-08-25-before-any-route-b-corpus-exists)

---

### route_b_v1 — PRE-REGISTRATION (written 2026-08-25, before any route B corpus exists)

**The hypothesis.** Route A violated the composition theorem's *separability* hypothesis
and its conclusion still held: coupling without competition leaves every task free to take
its individual favourite. This lineage attacks the other hypothesis, **free choice**:
contention for a scarce resource, so that some task must yield. Mechanism: **node-memory
knapsack** — a plan is feasible iff every node's co-resident demand
`Σ memReq[task_type][platform_type] ≤ cap_node(α)`, with
`cap_node(α) = α × max single candidate demand on that node`. Demands are the welded
`task-types.json` values (type-asymmetric on GPU: dnn1/dnn2 0.9, rf 1.5, cnn 1.3);
**the file is not edited** — the scarcity knob is per-node capacity. Memory occupancy does
not change episode physics, so the constraint is applied to the full enumerated sweep **at
scoring time**; one corpus serves the whole tightness ladder, and stage-1 zero-diff is
structural. Stacked design, two arms differing in exactly one flag: **Arm S** (primary) =
diamond4 DAG, distinct types, server mesh + backbone, `HEROSIM_DATA_LOCALITY=1`, payload
800 MB (the point where route A measured the pairwise term at 10–30% of episode cost);
**Arm B0** = identical, locality OFF. Rationale: under competition with *separable* costs
any regret is decoder myopia and a perfect decoder erases it — only competition **plus**
coupling forces the score itself to be joint. B0's predicted-zero is the built-in
instrumentation control.

**Statistics** (`scripts_cosim/score_route_b_contention.py`, per dataset / α / objective):
`R_greedy` = feasibility-masked sequential greedy over min-marginals (deployable pointwise
scheduler) vs constrained sweep optimum. **`R_exact` (primary)** = feasible-set exhaustive
argmin of the min-marginal-sum surrogate `Σ_t m_t(p_t)` — on separable physics
`m_t(p) = c_t(p) + const`, so `R_exact ≡ 0` under ANY feasibility restriction; nonzero
constrained `R_exact` can be neither a decoder nor an LS-fitting artifact. Repairs =
`y ~ a + b·Σm + counts` (one-integer: node-occupancy excess sharing, the program's
established collision column; k-integer: per-node×type counts, the constraint's own
sufficient statistic), fit on the full sweep, **refused as `saturated` when rows < 2×params**
— never silently reported. Views: memory-feasible (primary), memory-feasible ∩ spread
(secondary). A full indicator-LS surrogate is a sensitivity row only: measured on the m3
pilot it fires ~12% even *unconstrained* (collision channel + argmin tie noise) where the
registered statistic measures the established 0.000% — it is not the gate.

**Deviation from the phase-1 plan text, recorded honestly and made before any route B
corpus existed:** the plan named the LS surrogate as `R_exact`'s fit; measurement on the
m3 pilot showed that statistic broken as a gate (12% unconstrained false-fire), and it was
replaced by the min-marginal-sum form, which is *stronger for the pointwise side* (exactly
optimal wherever physics is separable). Control 1's expectation was also re-derived from
395.45% to 450% when the rig arithmetic was corrected to min-over-totals marginals.

**Positive controls, frozen (`tests/test_route_b_positive_controls.py`, 12 tests, all
passing before this entry):** Control 1 (separable, hot-node cap, wrong-task yield):
`R_greedy = 450.000000%` exactly, `R_exact = 0`; cap removed → both 0. Control 2 (pairwise
matching-shaped costs, 3×3): `R_exact = R_greedy = 150.000000%` exactly, 1int repair
**cannot** clean it (150% in every LS branch), kint repair **refused as saturated** at rig
scale; cap removed → 0. The guard exists because the 4-row rig caught the scorer's first
version reporting interpolated repairs as 0.0 — kept as a regression test. **Any control
failure makes route B runs VOID, not NO-GO**, and controls re-run after any scorer edit.
Still owed before corpus scoring: the end-to-end rigged dataset through the real
generation→sweep→scorer path (predicted `R_greedy ≥ 50%`, cross-checked by an independent
reader of the produced placements.jsonl).

**Pre-probe — run 2026-08-25, route B SURVIVES.** Registered kill condition (at the
tightest non-degenerate α on the existing m3 pilot n=200: `R_greedy>1%` on <5% of datasets
AND max < 17.25%): does **not** fire — measured 7% firing with max **92.10%** (rtt) /
**157.81%** (makespan) at α=1.0; `R_exact` stays ≈0 (≤1.46% max, 1% firing), as the
theorem predicts on separable physics. The M3 matching hint amplifies 5×. Calibration
finding: α=1.0 leaves the free-choice plan infeasible in only 10% of datasets — the corpus
grid needs scarcer candidates (`per_client=0`, fewer server hosts) to reach the 30–70%
band. Frozen: `simulation_data/route_b_preprobe_{rtt,makespan}.json`.

**THE GATE — Arm S, registered tightness, n=200, both objectives scored, rtt primary:**
- **PASS** iff ALL of: (1) fraction of datasets with `R_exact > 5%` is ≥ 10% with the 95%
  binomial CI excluding 10% from below; (2) repair fraction < 0.5 for BOTH count repairs
  (medians over firing datasets, saturated repairs excluded and counted); (3) the firing
  fraction rises monotonically along ∞ → loose → tight; (4) the spread-view firing
  fraction is nonzero.
- **FAIL** iff the CI excludes 10% from above, or condition (2) fails (count-shaped ⇒
  sixth confirmation of the empirical rule; route B closed as a GNN argument).
- **VOID** iff any positive control fails, or Arm B0 fires materially
  (`R_exact > 1%` on > 2% of datasets — theorem says ~0, so that is instrumentation), or
  the CI straddles 10% → escalate n = 200 → 400 → 800; ladder exhausted →
  **VOID-UNDERPOWERED**, never FAIL. Arm-vs-arm comparisons go through
  `gate_statistics.paired_regret_comparison` / `pooled_phase4_verdict`.
- **Tightness two-step:** three α values chosen from the smoke corpus so "tight" makes the
  free-choice plan infeasible in 30–70% of datasets, then **frozen here before the n=200
  corpus is scored**. Zero-feasible datasets are counted (`no_feasible_rows`), never
  dropped.

**The thresholds above may not be revised after data exists. A near-miss is a FAIL or a
VOID per the rules; there is no third option.**

**Stage 2 (conditional on PASS), binding constraints registered now:** any "GNN beats MLP
under constraints" claim requires (a) ONE shared constraint-aware sequential
feasibility-masked decoder used by both models (scarcity-pressure order, single
implementation both models plug scores into); (b) the MLP arm at its strongest (dim25cr +
the k-integer features); (c) an exact-assignment decode arm on the MLP's scores. Labels
become any-of-K tied-optimal sets; `audit_label_provenance` gains a tie-tolerant mode; the
cache carries the feasibility mask + capacity map (one contract, sidecar rule). Grouped
argmax is not an arm. Scope exclusions: no edits to `task-types.json`, no episode-physics
changes beyond route A's landed term, no training/checkpoints/live gates/datalab in stage
1, no new `train_*.py` ever.

**Status: PRE-REGISTERED.** Outcome row to follow.

---

### route_b_v1 — calibration freeze + three registration amendments (2026-08-25, before any gated corpus exists)

**What the smoke (12 matched datasets per arm, designated calibration data in the
registered two-step) established, and two harness defects it caught first:**

1. **Mid-episode replica scale-down corrupted DAG sweeps and their substrate.**
   `KEEP_ALIVE = 30 s` evicts idle replicas; under Arm S physics parents run past 30 s,
   so children's *forced* replicas were scaled down before dispatch — 66–72/240 rows
   lost per dataset, nondeterministically (the unstable scale-down victim sort), with
   `sweep_complete: false` recorded and nothing reading it. Worse: the same eviction ran
   during the *warmup capture*, so the enumerator's candidate substrate itself varied
   with physics speed — Arm B0 and Arm S got different enumerations (270 vs 576 plans on
   the same seed) until fixed. Fixes: `cosim_keep_alive()` env override
   (`HEROSIM_COSIM_KEEP_ALIVE`, unset = bit-identical); workers now append tracebacks to
   `placement_errors.log` (preserved next to `placement_metadata.json` — a truncated
   sweep without its error log is undebuggable); the route B scorer **refuses** truncated
   or metadata-less sweeps. Both corpus arms generate with the override set; enumeration
   counts verified identical across arms on all 12 smoke seeds.
2. **The smoke result itself, matched arms:** Arm B0 `R_exact` max 2.07% (the known
   collision/link residue; `R_greedy` up to 3578% — greedy myopia under scarcity is
   catastrophic but decoder-shaped). Arm S `R_exact` max **42.0%**, firing 25–33% at the
   >5% level, count repairs closing **nothing** — the joint signature the lineage
   predicts, 20× the B0 residue. Makespan channel fires too (max 19.6%).

**Amendments, each disclosed with what had been seen when it was made.** No gated
(n=200) data exists; the smoke's 12 datasets/arm had been scored. The PASS fraction
(≥10% at `R_exact > 5%`), the magnitude bar, the repair threshold (<0.5), α ladder
values, and the power ladder are all UNTOUCHED from the blind registration.

- **(A1) Arm B0 VOID trigger, was: `R_exact > 1%` on > 2% of datasets.** Premise error,
  visible in the registration's own text ("separable costs ⇒ surrogate = truth"): a
  backbone corpus is NOT separable — the collision channel and the link channel are
  real, known, count-shaped-or-thin couplings that produce exactly the 1–2% B0 residue
  measured. As registered, VOID would trip on real physics, not instrumentation.
  **Now: VOID iff B0 shows `R_exact > 5%` (the material bar) on > 2% of datasets.**
  Direction: loosens a validity check, does not touch the claim gate.
- **(A2) Tightness calibration, was: "tight" = free-choice plan infeasible in 30–70% of
  datasets.** Unsatisfiable: the α response is cliff-shaped (0.92 → 0.00 between α 3.2
  and 3.4 on the smoke) because CPU demands are near-equal; no α lands in the band.
  The band was a proxy for "binding but not degenerate" — replaced by the direct
  criteria: `no_feasible_rows = 0`, `greedy_stuck = 0`, mean feasible rows ≥ 50, and
  cw-infeasible ≥ 30%. **Frozen ladder: α ∈ {∞, 3.0 (loose), 2.0 (tight)}, tight = 2.0
  primary.** On the smoke: cw-infeasible 0.75–1.00, feasible rows 388–584, zero stuck,
  zero empty at both binding rungs.
- **(A3) PASS condition 3, was: firing fraction rises monotonically ∞ → loose → tight.**
  Two defects: (i) transplanted from route A, where the lever scaled a physics term —
  here the lever restricts a feasible set and the within-binding-regime gradient is
  flat/noisy (smoke: 0.33 at loose vs 0.25 at tight — one dataset's difference at
  n=12); (ii) as registered, "conditions 1,2,4 hold but 3 fails" lands in NONE of
  PASS/FAIL/VOID — an undefined outcome cell. **Now: (3′) the unconstrained rung fires
  `R_exact > 5%` on < 2% of datasets AND each binding rung fires above the unconstrained
  rung** — the free-choice attribution the condition was always meant to capture.
  Disclosed plainly: this amendment was made after seeing the 12-dataset smoke values;
  a reader may discount condition 3′ accordingly. Conditions 1, 2, 4 stand as
  registered blind. Outcome-cell closure, fixed before any corpus scoring: if the CI
  clears the PASS bar but condition 3′ or 4 fails, the verdict is **FAIL** with the
  failed condition named — the effect exists but is not attributable as registered;
  there is no fourth outcome. `score_route_b_gate.py` implements exactly this mapping
  and takes no threshold arguments.

**End-to-end control, form finalized:** the registered "rigged dataset" is superseded by
something stronger — `verify_route_b_scorer_agreement.py`, an independent from-scratch
recomputation of `R_greedy` and `R_exact` from `placements.jsonl` (no imports from the
scorer), which must agree within 1e-9 on **every** corpus dataset; plus the measured fact
that the real generation → sweep → scorer path fires at rig-scale magnitudes on the smoke
(`R_exact` 42%, `R_greedy` 151% on Arm S) — the "predicted ≥ 50% end-to-end fire" is
satisfied by measurement. Any verifier disagreement ⇒ VOID.

**Status: CALIBRATION FROZEN.** Next: zero-diff proof, corpus generation (2 arms ×
n≈200, same seeds, env-matched keep-alive), verifier, gate.

---

### route_b_v1 — OUTCOME: **PASS (stage 1).** Contention + coupling produces the non-pointwise structure five mechanisms and route A could not (2026-08-25)

**The registered row** (`score_route_b_gate.py`, no threshold arguments; Arm S, tight
α=2.0, rtt, n=204, zero truncated sweeps, enumerations bit-matched across arms):

> **35/204 = 17.2%** of datasets with `R_exact > 5%`, Wilson 95% CI **[0.126, 0.229]**
> — excludes 0.10 from below (condition 1 ✓). Median repair fraction **0.000** for BOTH
> the one-integer excess-sharing column and the k-integer per-node×type count vector,
> over all 35 firing datasets, none saturated (condition 2 ✓). Attribution:
> **0/204 fire unconstrained**; both binding rungs fire at 0.172 (condition 3′ ✓).
> Spread view: 14/109 firing — not collision-channel-only (condition 4 ✓). Arm B0
> validity: **0/204** above the material bar (max 2.49%) ✓. Independent verifier:
> 612 + 612 (dataset, α) cells agree to 1e-9 ✓. Positive controls 13/13 ✓.
> **VERDICT: PASS.**

**Full cell table** (`frac(R_exact > 5%)` / max `R_exact`):

| arm | objective | α=2.0 (tight) | α=3.0 (loose) | ∞ |
|---|---|---|---|---|
| **S** (coupling+competition) | rtt | **0.172** / 53.5% | 0.172 / 48.7% | 0.000 / 0 |
| **S** | makespan | 0.162 / 32.1% | 0.118 / 27.4% | 0.000 / 0 |
| **B0** (competition only) | rtt | 0.000 / 2.5% | 0.000 / 2.0% | 0.000 / 0 |
| **B0** | makespan | 0.000 / 1.6% | 0.000 / 1.7% | 0.000 / 0 |

**What this establishes.**
1. **The composition theorem's free-choice hypothesis is the load-bearing one, and
   violating BOTH hypotheses at once is what creates structure.** Under the memory
   knapsack + the 800 MB pairwise transfer, the best additive surrogate *with a perfect
   decoder* is suboptimal by up to 53% on 17% of datasets — a target no pointwise
   scorer can express regardless of decode.
2. **The effect is not LINEARLY count-shaped.** *(Amended in place 2026-08-25 by the §9b
   block ablation — see the correction paragraph below. The original text read "The
   effect is not count-shaped … does NOT extend here", which is wrong as written.)* The
   empirical rule that killed five co-location mechanisms ("every escape collapses to an
   occupancy integer") does not extend here **in its linear form**: the constraint's own
   sufficient statistic (per-node×type counts), entered linearly, repairs a median of
   exactly nothing. (Honest detail: the k-integer repair does pull ~1/3 of firing
   datasets under the 5% bar — 0.172 → 0.118 — but the median closure is 0.000 and the
   registered condition is decisive.)

   **The correction.** kint is *linear* in the counts. Adding the per-type quadratic
   co-residency sums Σ_t occ_{node(t)}[k] — the SAME statistic, entered nonlinearly, with
   no parent-placement or network columns at all — takes the median closure from 0.000 to
   **0.843**, and adding load/cap and the over-cap count takes it to **0.892**. So the
   occupancy rule DOES extend; a linear repair simply could not see it, and reporting
   "not count-shaped" on the strength of a linear fit was an overreach. What the stage-1
   PASS actually established is narrower and still stands: *the linear* count repair
   closes nothing, which is what registered condition 2 tested and what the 17.2% firing
   rate is measured against. Neither the PASS nor any of its four gate conditions moves —
   condition 2 was registered with the 1int/kint linear repairs and both still close a
   median of 0.000. Reproduced by `route_b_coefficient_transfer.py` (arms `kint`,
   `kint+quad`, `occupancy`), independently recomputed by
   `verify_route_b_scorer_agreement.py --check-blocks`, 315/315 arm-values to 1e-9.
3. **Competition alone is not sufficient either — the stacking argument was right.**
   Arm B0's score-side structure never crosses 2.5%, while its *greedy* regret reaches
   3578%: scarcity without coupling produces only decoder-shaped error, which a better
   decoder erases. Coupling decides *who should yield*; that is the graph question.
4. Both objectives fire; makespan is slightly weaker (0.162/0.118) but the same shape.

**What is NOT established, stated before anyone asks.** No model has been trained;
nothing here says a GNN can *learn* this structure, and nothing compares GNN to MLP —
that is stage 2, valid only with the registered decoder discipline (one shared
constraint-aware sequential masked decoder, dim25cr+k-integer MLP arm, exact-assignment
decode arm). Nothing about live serving. One topology family (6 servers, per_client=0,
diamond4 over dnn1/dnn2/rf/cnn), one demand table (the welded task-types.json), one
frozen α ladder. The 17.2% firing fraction is a property of this grid, not a universal
rate.

**Falsified along the way:** the B0-as-separable premise in the original registration
(a backbone corpus carries the collision + link channels; amendment A1); the 30–70%
tightness band (cliff-shaped α response; A2); the monotone-in-α firing condition (A3);
and the first scorer's LS-surrogate `R_exact` (12% false-fire unconstrained) plus its
unguarded repair fits (Control 2 caught interpolation at rig scale).

**Artifacts.** Corpora (local, gitignored): `gnn_datasets_dag4_route_b_pilot_v1_arm_{s,b0}`
(204 each; regenerable from `ROUTE_B_PILOT_V1_GRID` seeds 901–917 with
`HEROSIM_COSIM_KEEP_ALIVE=1000000 HEROSIM_RETAIN_TASK_TIMES=1`, Arm S adding
`HEROSIM_DATA_LOCALITY=1 HEROSIM_OUTPUT_SIZE_BYTES=800000000`). Frozen reports:
`simulation_data/route_b_pilot_v1_arm_{s,b0}_{rtt,makespan}.json`,
`route_b_preprobe_{rtt,makespan}.json`. Tools: `score_route_b_contention.py`,
`score_route_b_gate.py`, `verify_route_b_scorer_agreement.py`,
`tests/test_route_b_positive_controls.py` (13 tests).

**Post-PASS scrutiny (2026-08-25, same day, before anyone else read the result): a
clean confirmation gets the same suspicion a clean zero got all session.** Four checks,
run against the standing worry that a result matching the hypothesis this precisely is
exactly the one nobody feels the urge to re-check.

1. **Gate condition 2 (repairs close nothing) independently reconfirmed, with an
   honest caveat found along the way.** Extended `verify_route_b_scorer_agreement.py`
   with `--check-repairs`: a from-scratch LS fit (pure Python, no numpy — a hand-rolled
   Gaussian-elimination normal-equations solve) recomputing both repairs directly from
   each dataset's files. First run disagreed with the scorer on one dataset (10.6% vs
   42.0%) — traced to normal equations squaring the design matrix's condition number
   across columns of wildly different scale (intercept, RTT-magnitude sums, 0/1 counts);
   fixed by standardizing columns before solving (confirmed against numpy/SVD:
   coefficients now agree to 1e-13). One dataset (`ds_00008`, Arm S, α=2.0) still
   disagrees even after the fix — traced *at the time* to a "genuine near-tie": 4+
   feasible plans with materially different true costs (78.1s, 60.8s, 58.2s) predicted
   equal to ~13 significant figures by the fitted surrogate.

   > **RETRACTED 2026-08-25 (same day, later session).** This was **not** a genuine tie
   > and was not real: it was an artifact of *two* verifier bugs compounding — the
   > standardized-normal-equations solver still not reaching the true LS optimum on the
   > wider t1 matrix, and the verifier's 1int column computing `max` node-occupancy
   > excess where the registration says `sum`. Each masked the other. With the MGS-QR
   > solver and the correct column, scorer and verifier agree outright on `ds_00008` and
   > **all 612 repair values agree with zero tie-acceptances**. The original sentence
   > below — "This is real, not an artifact" — was wrong, and is struck rather than
   > silently deleted, because it was recorded here as an established finding and read
   > that way. Detail in the stage-2 pre-probe entry's defect 2. (Prediction ties DO
   > occur in this machinery and are real where they occur — see §9b's tie bands, where
   > 22/35 firing datasets tie at the full-T1 argmin — but `ds_00008` under the registered
   > columns was not one of them.)

   ~~**This is real, not an artifact — reported, not hidden.**~~
   Independently recomputing repair_fraction for **all 35 firing datasets from scratch**
   (ignoring the scorer's numbers entirely): **median 0.000 for 1int, median 0.000 for
   kint** (kint mean 0.357, max 1.0 — a few datasets ARE fully repaired by kint, but the
   *median*, the registered statistic, is exactly what the scorer reported). **Gate
   condition 2 holds under independent, from-scratch recomputation.**
2. **The amendment to condition 3 is disclosed with its own counterfactual, not just
   its rationale.** Route A's own precedent for "rising" (`score_route_a_scaling_probe.py`
   `rising = means[-1] > means[0] + 1e-9`, endpoints only) would read the observed
   sequence 0.000 (∞) → 0.172 (loose) → 0.172 (tight) as rising (0.172 > 0.000) — **the
   original wording, read consistently with the one existing precedent in this
   codebase, would ALSO pass.** Read literally step-wise (every adjacent pair strictly
   increasing), the flat loose→tight step (0.172 = 0.172) would **fail** it. Both
   readings are stated because the wording is genuinely ambiguous and the amendment was
   made after seeing this exact number — a reader is free to prefer either. What is not
   ambiguous: no reading of the original condition, applied honestly, changes the
   PASS verdict, because it was never the swing condition — condition 1 (the CI) and
   condition 2 (repairs) carry the result.
3. **Firing rate reported above the B0 noise floor, not just against the 5% bar.**
   B0's own residue tops out at 2.49% (α=2.0) — a floor under every Arm S number. Arm S
   firing fraction at `R_exact >` 5.0% / 7.5% / 10.0%: **0.172 / 0.157 / 0.123.** The
   effect does not thin out approaching a threshold four times the B0 floor; at >10% it
   still clears the registered 10% PASS bar on its own. Not floor-sensitive.
4. **Alpha provenance.** The frozen ladder (α ∈ {∞, 3.0, 2.0}) was calibrated on route
   B's own smoke corpus (`gnn_datasets_dag4_route_b_smoke_{s,b0}`, 12 datasets/arm on
   `ROUTE_B_PILOT_V1_GRID`, matching the gated corpus's topology and physics exactly),
   not on the unrelated m3 pilot — see amendment A2 above. The freeze commit
   (`2c3ebbc`… through the calibration-freeze entry) predates the n=204 generation run.
   Realised componentwise-plan-infeasible fraction at the frozen tight rung: 0.44–0.50
   across the corpus (table above) — comfortably binding, not degenerate.

**Route A cross-reference.** Arm S's unconstrained cell (α=∞: locality on, 800 MB
payload, DAG dispatch, on the keep-alive-fixed harness) is physics-adjacent to route
A's own condition but **not a re-run of route A's grid** (route B's grid uses 6 servers /
`per_client=0`; route A's used a different server count and replica config) — it should
be read as corroborating evidence at n=204, not as route A's own probe repeated. The
literal re-verification of route A's condition is the 6-dataset retro-check recorded in
`route_a_v1` above, which is the one that actually reused route A's grid.

**Status: PASS — stage 1 CLOSED, and re-checked. Stage 2 (can a GNN learn it and beat the
constraint-aware pointwise baseline?) requires its own pre-registration before any
training run.**

### route_b_v1 — stage 2 pre-probe zero: **NO-GO-PREPROBE-T1** (2026-08-25)

Stage 2's pre-registration was drafted (`docs/lineages/route_b_v1/stage2-preregistration.md`, this commit)
and, on review, the user identified its load-bearing hole before sign-off: the registered
strongest-MLP arm (T1 = dim25cr + k-integer + partial-assignment state, including
parent-placement/hop/transfer columns) is **not a pointwise baseline** — its plan-level
score is non-separable, and stage 1's `R_exact` (a *separable* surrogate) and its
median-0.000 count-repair result say nothing about it. The doc's §9a registered an offline
kill test with the reading fixed **before** the number existed: recompute `R_exact` on the
stage-1 204 (Arm S, α=2.0, rtt) with the surrogate augmented by the full T1 plan-level
column set — kint + per-type quadratic co-residency + load/cap + over-cap count + min/max
parent-hop sums + `Σ_edges hops/bottleneck` + `Σ_edges latency` + same-node-parent count,
the last three computed from each dataset's own `link_topology.routes`, i.e. exactly what
`_dependency_transfer_time` charges (uniform 800 MB payload absorbed by the LS
coefficient, so the columns **span the charged coupling term exactly**). Registered
reading: median repair fraction ≥ 0.5 over the stage-1 firing datasets ⇒ the architecture
claim is pre-falsified and stage 2 does not run as registered.

**Result: median T1 repair fraction 1.000** (mean 0.730; 26/35 firing datasets closed
≥ 0.5; kint comparison: median 0.000, mean 0.357, matching the stage-1 scrutiny to the
digit). `frac(R_exact > 5%)` falls 0.172 → **0.054** (11/204 residual, max 22.2%).
Attribution ablation over the firing 35: the parent-coupling block alone closes at median
1.000, the occupancy block alone at median 0.892 — two largely redundant routes to the
same closure.

> **AMENDED 2026-08-25 by §9b, which put this ablation in code for the first time.** Both
> numbers reproduce exactly (1.000 and 0.892), but "two largely redundant routes" is
> wrong: **both blocks contained `kint`**, and `kint` is the shared ingredient. Membership,
> now unambiguous — occupancy = `kint+quad+cap`, parent-coupling *as originally run* =
> `kint+hop+coupling` (PREREG:406's parenthetical "kint + cols 33–35 analogues" was
> literal). Stripped of `kint`, the parent block alone closes only **0.392**, and `quad`
> alone closes **0.000**. The honest decomposition is that neither block is a route on its
> own: `kint` alone closes 0.000, and it is `kint` *combined with* either the quadratics
> (0.843) or the parent columns (1.000) that closes the effect. When this prose was
> written, no committed code computed it — the fitted coefficients were discarded at every
> solver call site — so it could not be checked. It can now:
> `route_b_coefficient_transfer.py`, verified 315/315 arm-values to 1e-9.

Scorer: `score_route_b_contention.py` (`t1` repair, constrained rungs only);
report frozen at `simulation_data/route_b_stage2_preprobe_t1_rtt.json`. Verification:
`verify_route_b_scorer_agreement.py --check-repairs` agrees on **all 204 cells and all
612 repair values (1int, kint, t1), zero tie-acceptances**.

**Two verifier defects found and fixed en route — both matter to the stage-1 record:**

1. **The pure-Python solver did not reach the true LS optimum on the wider t1 matrix.**
   Standardized normal equations (the stage-1 scrutiny's own fix) produced fitted values
   diverging from the unique LS projection on ds_00008 (fitted values on fit rows are
   solver-independent, so this is a numerical failure, not a tie). Replaced with
   hand-rolled MGS QR (one re-orthogonalization pass, dependent-column dropping) — still
   no numpy, still zero scorer imports; verified against numpy to 1.6e-13.
2. **The verifier's 1int column was `max` node-occupancy excess where the registration
   (and `separability_diagnostic._excess_sharing`) says `sum`.** A real bug, masked by
   defect 1: the imprecise fits happened to argmin onto the same plans on every dataset
   previously checked. The QR solver exposed it (ds_00019: scorer 1.036 vs
   verifier-with-max 9.633). With both fixes, **ds_00008's recorded "genuine
   floating-point-level tie" dissolves** — scorer and verifier now agree outright there;
   that scrutiny interpretation is superseded (the disagreement was the verifier's wrong
   column plus solver imprecision, not an inherent tie). Stage 1's gate verdict is
   untouched (the gate consumed the scorer's statistics, which were correct and are now
   re-verified under the fixed verifier), but the stage-1 claim "1int independently
   confirmed" was, until today, confirmed against a different column definition. It is
   now actually confirmed: 612/612 repair values agree.

**What this establishes:** on this grid, a pointwise score given partial-assignment state
(exactly what a sequential masked decoder exposes for free) is sufficient at the
surrogate-expressiveness level to close the median stage-1 firing dataset completely. The
"GNN beats strongest-MLP under constraints" claim is pre-falsified before any cache,
decoder, model, or corpus was built — for the price of one scorer run. **What is NOT
established:** the T1 repair is a per-dataset LS fit on the dataset's own sweep — an
expressiveness upper bound, not a trained cross-dataset model; whether a *trained*
pointwise-plus-state model realizes this bound is the reduced V5-shaped question
("decoder-state features suffice — no graph needed"), which needs no GNN and its own
registration if pursued. The 11-dataset residual stratum (5.4%, below stage 1's 10% bar,
max 22.2%) is real but does not clear the program's own materiality standard. The α=∞
rung's 0/204 remains corroborating evidence for route A's conclusion at n=204 (not a
literal grid re-run — see the stage-1 outcome entry's caveat).

**Status: PROVISIONAL — NO-GO-PREPROBE-T1 was RETRACTED AS MEASURED on 2026-08-25 by §9c
(entry below). Do not read this entry as settled.** The T1 column set used here includes
`kint`, one free coefficient per `(node, task_type)` — an identity-indexed per-dataset
lookup table with **no corresponding column in the registration's own §2 `dim36crk`
table**. So the kill test was run with a surrogate strictly more expressive than the T1 arm
it stands in for. Stripped of that block, the closure of the actually-registered feature set
is 0.392–0.648 depending on a tie rule §4 never specified, and §9c(a) measured the block's
coefficients to be unrecoverable from node features (held-out R² 0.014). **Stage 2's
architecture question is reopened and requires re-registration with a corrected T1
definition before anything is built.** The one thing this entry establishes unconditionally
is the reverse-direction result: whatever closes the effect, it is *not* message passing
that is needed — see §9c's exploratory pooled `krank` (0.790 under one coefficient set).

---

### route_b_v1 — STAGE 2 PRE-REGISTRATION (recorded retroactively 2026-08-25)

`docs/lineages/route_b_v1/stage2-preregistration.md` at commit `df9971e` is the registration under which
the §9a pre-probe's reading was fixed before its number existed. Its own header (line 6)
required this row and it was never written — recorded now, late, rather than left absent.
§9b was added to that file 2026-08-25 (this commit) and is disclosed as a post-outcome
deviation in its §11.

---

### route_b_v1 — §9b coefficient transfer: **VOID-KINT-CONFOUNDED**, and the §9a bound is confirmed tie-robust (2026-08-25)

**The question.** §9a's T1 repair fits fresh coefficients on *every dataset's own sweep*;
a trained cross-dataset model gets **one** set. So NO-GO-PREPROBE-T1 rests on a bound that
may not transfer. Registered in `docs/lineages/route_b_v1/stage2-preregistration.md` §9b before the number
existed, with three cells so that "cost of dropping kint" could never be confused with
"cost of pooling", and with the VOID condition written in advance.

**The obstruction, found while designing and stated before measuring:** `kint` **cannot be
pooled at all.** Its columns are one per `(node, task_type)` pair *in that dataset's own
demand*, so vocabulary and width both vary (K ∈ 8…13, X widths 21–26 over this corpus).
There is no cross-dataset coefficient vector to fit.

| cell | fit | median | tie-band | ≥0.5 |
|---|---|---|---|---|
| A | per-dataset, full T1 | **1.0000** | **[1.0000, 1.0000]** | 26/35 |
| B | per-dataset, T1 − kint | 0.3922 | [0.3922, 1.0000] | 17/35 |
| C | **pooled**, T1 − kint | 0.0000 | [0.0000, 1.0000] | 16/35 |
| C′ | pooled, equal dataset weight (sensitivity) | 0.0000 | [0.0000, 1.0000] | 15/35 |

**VERDICT: VOID-KINT-CONFOUNDED**, the registered branch — cell B is already below 0.5, so
cell C cannot be read as a test of *pooling*: the drop is attributable to dropping `kint`,
which no single coefficient set can carry anyway. **§9b does not weaken NO-GO-PREPROBE-T1
and does not strengthen it. The V5 question stays open and stays empirical.**

**What §9b did establish, and it is the more useful half:**

1. **The §9a statistic is tie-robust, which nobody had checked.** Cell A's median is 1.0000
   whether prediction ties at the argmin are resolved optimistically, pessimistically, or
   by the registered plan-key tie-break — even though **22/35 firing datasets do tie**
   (max group 8). A NO-GO resting on a tie-break would have been worth exactly as much as
   `ds_00008`'s retracted "genuine tie". It does not.
2. **Cell B, by contrast, is genuinely indeterminate** — band [0.392, 1.000] straddles the
   0.5 threshold, with ties up to 16 plans wide. This is not float noise: stripped of
   `kint`, the 9 node-agnostic columns **cannot separate up to 16 feasible plans at all**.
   That is the finding, not a nuisance. The independent verifier surfaced it first, as
   three "TIE (accepted)" lines on cell B where scorer and verifier picked different plans
   from the same tied group (0.000 vs 1.000) — a disagreement that is real and that the
   band now reports as first-class output rather than a footnote.
3. **The block attribution now exists in code** and both prose numbers reproduce — but
   their interpretation was wrong; see the amendment in the §9a entry above and the
   in-place correction of stage-1 finding #2.

| arm | blocks | median | ≥0.5 | residual >5% |
|---|---|---|---|---|
| kint | linear counts | 0.0000 | 13/35 | 24 |
| quad | quadratic counts only | 0.0000 | 7/35 | 32 |
| **kint+quad** | **counts, nonlinearly** | **0.8429** | 23/35 | 15 |
| occupancy | kint+quad+cap | 0.8924 | 24/35 | 14 |
| parent-coupling | hop+coupling | 0.3922 | 17/35 | 22 |
| parent-coupling incl kint | kint+hop+coupling | 1.0000 | 27/35 | 10 |
| full T1 | all five | 1.0000 | 26/35 | 11 |

**Coefficients (descriptive, as registered — the repair fraction is the decisive statistic
and this is not).** The pooled `transfer` coefficient is 330.96 against the registered
physical prediction of 762.939453125 (`800e6 / 1024²`), and pooled `latency_sum` is −38.4
against a predicted 1.0. Per-dataset dispersion is enormous (transfer mean −277, sd 8106).
**None of this is evidence about the physics**, and the registration said so in advance:
the cell-B/C fits are mis-specified by construction (they omit the block that does the
work), 9/35 per-dataset designs are rank-deficient, and `same_node_edges = 4 −
remote_edge_count` is collinear with the hop block on `diamond4`. Recorded because it was
registered, and because a *correctly specified* pooled fit would be the place to test the
762.94 prediction properly.

**Re-derived 2026-08-25 for the 8-task probe.** The "4" here is `diamond4`'s total edge
count (4 parent-child pairs), not a hardcoded constant in the scorer — `score_route_b_contention.py`'s
`same_node_edges`/`transfer` loop (`fn`, around line 596) sums over `parents_of` for
whatever edges the plan's DAG actually has, so no code change is needed. For two diamond4
instances co-decided in one episode (8 tasks, 8 edges total, 4 per instance), the identity
becomes `same_node_edges = 8 − remote_edge_count` and the collinearity with the hop block
persists at the new constant — the mechanism (total edge count is fixed per dataset, so
same-node and remote edge counts are complementary) is unchanged by doubling, only the
number is.

**Verification.** `verify_route_b_scorer_agreement.py --check-blocks` — an independent
pure-Python/QR recomputation from each dataset's raw files — agrees on **315/315 (dataset,
arm) repair fractions to 1e-9** across 35 datasets and 9 arms, with the three cell-B tie
acceptances described above. Cell A reproduces §9a exactly (median 1.0000, mean 0.7302,
26/35, 11 residual). The verifier itself is, as of this commit, backstopped by
`tests/test_route_b_repair_fixtures.py`: 16 closed-form fixtures (29 with the positive
controls) covering `solve_least_squares` against textbook OLS, the `sum`-vs-`max` 1int
distinction that survived three rounds of checking, the t1 columns hand-computed on a
4-node toy, the scorer/verifier cap-convention divergence on an uncapped node, and the
saturation guard. Refactor safety: `t1_cols`'s new block registry is proven **byte-identical**
on the frozen §9a report (204 datasets × 3 α).

**Residual stratum (11 datasets, DESCRIPTIVE — 5.4% is below the program's 10% materiality
bar and this is not a claim).** No separating structure found. Medians, residual vs closed:
feasible fraction 0.643/0.643, distinct nodes in the optimum 2/2, same-node edges 2/2,
max load/cap 0.921/0.921, kint width 11/11. The only gaps are small and in the direction
you would expect from their larger regret: R_exact 16.8 vs 14.7, transfer in the optimum
0.0140 vs 0.0100, hop sum 11 vs 10, RTT spread CV 0.197 vs 0.175. **There is no "a graph is
needed when X" sentence here** — at n=11 with no separating feature, the residual reads as
the tail of the same distribution, not a distinct stratum. The edge closes cleanly.

**Artifacts:** `simulation_data/route_b_coefficient_transfer.json`;
`scripts_cosim/route_b_coefficient_transfer.py`; `--check-blocks` in the verifier;
`tests/test_route_b_repair_fixtures.py`.

**Status: §9b returned VOID on its own question — whether the bound survives one coefficient
set is NOT answered by this method, because the block carrying the closure is not poolable.
Superseded in part by §9c below, which asked why that block was in the column set at all.**

---

### route_b_v1 — §9c: `kint` is not a T1 feature — **NO-GO-PREPROBE-T1 RETRACTED AS MEASURED** (2026-08-25)

**The objection, against §9a itself.** §9a's T1 set includes `kint`: one free coefficient per
`(node, task_type)`, a per-dataset lookup table over node **identities**. **No column of the
registration's own §2 `dim36crk` table is identity-indexed** — cols 25–28 are per-type
occupancy on *the candidate's own node*: anonymous, fixed width, four columns. §2's verbatim
rule cuts both ways, and a feature the MLP cannot have must not be credited to it. So §9a's
kill test may have been run with a surrogate strictly more expressive than the arm it stands
in for. Registered in `docs/lineages/route_b_v1/stage2-preregistration.md` §9c before either number existed.

**The load-bearing observation, and it needed no new code.** The scorer's `quad` block is
*exactly* the plan-level rendering of cols 25–28:
`quad[k] = Σ_n tot[n]·occ[n][k] = Σ_t occ_{node(t)}[k]`. Likewise `load_over_cap` = col 29,
`overcap_tasks` = col 31, `min/max_hop_sum` = 33–34, `transfer` = 35. **T1 − kint is
precisely the dim36crk-expressible set — so §9b's cell B already WAS the anonymous closure
measurement**, at 0.392. `kint` is the only T1 block with no §2 column.

| measurement | result | registered reading |
|---|---|---|
| **(a)** kint coefficients regressed on node features, held out by dataset | **R² = 0.0138** (in-sample 0.0974) | < 0.5 ⇒ **identity-memorized; §9a does not bound the T1 arm** |
| **(b)** anonymous (dim36crk) closure | `mean_tied` **0.648** vs `registered`/`pessimistic` **0.392** | directions disagree ⇒ **VOID-TIE-INDETERMINATE** |

**(a) is decisive and (b) is a specification gap.** On (a), the in-sample R² is the telling
figure: node features barely explain these coefficients even without a generalization gap,
so it is not a small-sample artifact — the block is genuinely a per-dataset identity lookup.
On (b), the readings disagree because tie groups run up to **16 plans wide** and the
sorted-plan-key rule lands *worse than an average tie-break* on this corpus. **§4's decoder
never specified what to do with tied scores, and the anonymous verdict flips on that choice.**
That is a real hole in the registration, not a numerical nuisance.

**A second registration defect, found the same way (recorded 2026-08-25):** §4 pinned
"scarcity-pressure order" to `greedy_masked_plan`'s ascending `(best available marginal,
task_id)`. On this corpus that order does not exist: in **all 204 datasets the four
min-marginal minima are exactly tied** — every task's best placement lies in the globally
best plan, so `min_p m_t(p)` equals the global minimum RTT for every task — and the tie-break
collapses the order to `task_id`, which is the DAG's topological order. Measured
consequences: **0 of 816 DAG edges decode child-before-parent** (§2's hedge that "parents are
not guaranteed to precede children" never fires) and **0 of 816 steps** have a task's best
choice already taken by an earlier task; only capacity ever blocks the top choice, on
167/816 = 20.5% of steps. The registered order therefore carries no scarcity information
whatsoever. This is the same class of hole as the unspecified tie rule: a registration naming
a discriminator that is constant on its own corpus. **A corrected stage 2 must fix both**, or
it repeats the error under a new name.

Also measured: **T1 ≡ T0 at decode step 0** (all eleven partial-state columns are zero when
nothing is placed), and the prefix-oracle curve (7.78 → 9.84 → 1.98 → 0.31) puts essentially
all decoder myopia in the first two of four steps. **Write-up rule, binding: wherever the
four-task limit is doing the work, the sentence is "the corpus is too small to test the
architecture claim", never "the architecture claim is false."** These two measurements
support the first reading, not the second.

**Consequence: NO-GO-PREPROBE-T1 is retracted as measured, and stage 2's architecture
question is reopened.** Per §9c this is explicitly **not** a licence to start the build queue:
the corrected T1 definition — including a tie rule — gets re-registered first. §9a's purpose
was to kill cheaply; a corrected §9a that fails to kill changes the registration, not the
discipline.

**Exploratory (NOT registered, NOT independently verified, no verdict read from it) — and
it is why the reopening may be short.** Replacing `kint` with `krank`, occupancy indexed by
identity-free node **rank** (ascending capacity, then mean hop, padded to a common width):

| arm | median | ≥0.5 |
|---|---|---|
| krank + dim36crk, per dataset | **1.000** | 26/35 |
| **krank + dim36crk, ONE pooled coefficient set** | **0.790** (mean_tied 0.824) | 20/35 |

So the closure never needed node *identity* — it needed per-node occupancy **resolution**,
which dim36crk's four candidate-local columns do not supply. And unlike `kint`, `krank`
pools: a single coefficient set over identity-free columns closes the median firing dataset
at 0.79, with no message passing. **That is the follow-up the §9b VOID named, and it points
where §9a did: the structure looks reachable by a pointwise scorer, just not by the one
stage 2 registered.** A hypothesis for the corrected registration, not a result.

**Verification.** Cells and all ablation arms: `--check-blocks`, 315/315 to 1e-9. **The §9c(a)
regression and both `krank` arms are single implementations and are NOT independently
verified** — stated rather than implied. Gauge note: within a dataset the `kint` columns for
a given type sum to exactly 1 (each type has one task in `diamond4`), so the fit is
rank-deficient by one dimension per type and the coefficients are defined only up to a
per-type shift; `lstsq` returns the minimum-norm representative, a convention. (a) therefore
scores coefficients **centered within (dataset, task_type)**, the gauge-invariant content.

**Status: NO-GO-PREPROBE-T1 retracted as measured. Stage 2 REOPENED, pending re-registration
with (i) a T1 definition that either justifies identity-indexed columns or replaces them with
an anonymous per-node-resolution block, and (ii) a decoder tie rule. Nothing is built until
that registration exists. The 8-task probe of the exploratory pooled result is §9d below.**

### route_b_v1 — §9d: 8-task probe — pooled `krank` closure SURVIVES the doubling, attenuated (2026-08-26)

**The question.** §9c's exploratory pooled result — ONE identity-free coefficient set
(`krank` + dim36crk) closing the median firing dataset at 0.790 — was measured on 4-task
episodes, where the joint decision is small enough that a lookup-table-shaped fit is cheap.
Does it survive doubling the joint decision to 8 tasks (2 `diamond4` DAG instances per
episode, independently drawn client nodes, byte-identical infrastructure per index)?

**Corpus.** `gnn_datasets_dag4_route_b_pilot_v1_8task`, 204 datasets, generated on datalab
(job 713673): 204/204 complete, 0 silent skips, 0 truncated sweeps, 0 worker failures; sweep
sizes 27,648–516,096 (~41M sims, ~23 GB). Venue measured not to be a variable twice over:
the 4-task identity gate (job 713654, 16/16 artifact hashes match the frozen local corpus)
plus an 8-task spot check (cluster `ds_00002` vs the validated local smoke run —
`best.json`/`workload.json` byte-identical, 161,280-row `placements.jsonl` identical as a
set, `infrastructure.json` differing only in `metadata.{generation_time,config_file}`).
Generation recipe is the full Arm S env block — job 713615 failed for lack of
`HEROSIM_COSIM_KEEP_ALIVE`/`HEROSIM_RETAIN_TASK_TIMES`, and the skip threshold had to be
raised to 2,000,000 against the *pre*-uniqueness `total_possible` (hard bound 1,248² =
1,557,504) after 1,000,000 silently dropped `ds_00026`; both are documented in
`route_b_8task_probe.sbatch`.

**Alpha correspondence — registered a priori, not searched.** `cap_node = alpha ·
max_single_demand` has no task-count term, so 8 tasks against the same cap is ~2× tighter;
the equal-tightness match to `TIGHT_ALPHA = 2.0` is its double, **4.0** (ladder
3.0/4.0/5.0/6.0 covers both doubled rungs plus the response curve). At the primary 4.0 the
silent-bias counters are clean: `greedy_stuck = 0`, `no_feasible_rows = 0` (at 3.0: 76
stuck, 2 no-feasible — the tight end is real, and it is not the primary).

**Firing: 33/204 = 16.2%** at `r_exact_pct > 5.0`, vs the 4-task 35/204 = 17.2% — the
a-priori doubling landed on matched power (pooled statistic rests on 19 closed cells vs 20),
so the two closures are directly comparable. Firing `r_exact_pct` spans 5.3–63.7%, median
14.3%.

| arm (exploratory, NOT registered, no verdict read from it) | 4-task (§9c) | 8-task |
|---|---|---|
| krank + dim36crk, per dataset | 1.000 (26/35) | **0.988** (20/33) |
| **krank + dim36crk, ONE pooled coefficient set** | 0.790 (mean_tied 0.824, 20/35) | **0.617** (mean_tied 0.617, 19/33) |

**Reading: the layout hypothesis survives the doubling, attenuated.** A single pooled,
identity-free coefficient set still closes the median firing dataset above half
(0.790 → 0.617); per-dataset closure is essentially unchanged (1.000 → 0.988). Per §9c's
own framing this remains **evidence about the corrected-registration hypothesis (per-node
occupancy *resolution*, no identity, no message passing), not a gate** — it feeds the
stage-2 re-registration and changes no verdict.

**Registered readings of §9c(a)/(b), applied to this corpus — both land opposite their
4-task values:**

- **(a)** `kint` coefficients regressed on node features, held out by dataset:
  **R² = 0.607** (in-sample 0.644; 390 coefficients, 33 datasets) — ≥ 0.5 reads
  *feature-representable*, where the 4-task corpus measured 0.0138 (*identity-memorized*).
  With 2 tasks of each type per dataset the per-type gauge degeneracy §9c's verification
  note describes is also broken, so the fit is better-posed here, not just luckier.
- **(b)** anonymous (dim36crk-expressible) closure: **all three tie readings agree at
  0.988** (optimistic upper bound 0.997) — the 4-task VOID-TIE-INDETERMINATE does not recur
  at 8 tasks; the script's registered rule prints `NO-GO-PREPROBE-T1-STANDS`, and the
  transfer's top-level verdict is `BOUND-TRANSFERS`.
- Decomposition against the registered physical predictions: the closure is carried by the
  **parent-coupling block** (hop+coupling pooled median 0.997, 23/33); the occupancy blocks
  (`kint`/`quad`/cap) pool to median **0.000**. At 8 tasks the pooled structure is
  parent-coupling-shaped, not occupancy-shaped.

**Provenance.** Corpus job 713673, scorer job 713793, transfer job 713794 (41 min), all
CPU-amd, repo at `72d75e7` both venues. Artifacts (both venues):
`simulation_data/route_b_8task_rtt.json` (frozen report, `--include-per-dataset`),
`simulation_data/route_b_8task_coefficient_transfer.json`. Harnesses:
`scripts_cosim/datalab/route_b_8task_{probe,score,transfer}.sbatch`,
`route_b_venue_identity_gate.sbatch`.

**Status: route_b_v1 item 4 CLOSED — probe complete, outcome recorded. The anonymous
per-node-resolution layout hypothesis survives the 4→8 task doubling at matched firing
power (0.790 → 0.617 pooled, per-dataset ~1.0 both). Exploratory throughout; the stage-2
re-registration required by §9c remains the gating step.**

### route_b_v1 — STAGE 2 CORRECTED PRE-REGISTRATION (v2, signed off 2026-08-26)

The §9c-mandated re-registration exists and is signed off:
`docs/lineages/route_b_v1/stage2-preregistration.md` **at commit `b7553cf`** is the registration under
which stage 2 now runs (in-place v2 rewrite; the retracted 2026-08-25 text remains
readable at `df9971e`/`597e7ab`, and the file's §11 logs every replacement with its
evidence). What it fixes, in one line each:

- **T1 layout `dim63crk`** (was `dim36crk`): + 24-col `krank` one-hot — the anonymous
  per-node-resolution block §9c named and §9d validated, pinned to `krank_cols`
  (`route_b_coefficient_transfer.py`), whose per-edge sum reproduces the pooled surrogate
  exactly; + 4 `linkrank` edge cols (registered **no-op** expectation per the route_c
  FAIL-BY-EXHAUSTION, included for feature parity; user decision 2026-08-26); − old col 32
  (constant ≡ 1 under the corrected order).
- **Decode order = DAG topological, ties by `task_id`** (mode `masked_topo`) — registers
  what §9c measured the retracted "scarcity-pressure order" to already collapse to; the
  frozen stage-1 greedy plans stay the 1e-9 acceptance target.
- **Tie rule registered**: [pessimistic, mean_tied, optimistic] band mandatory on every
  gate statistic, `mean_tied` the fair reading, direction disagreement =
  VOID-TIE-INDETERMINATE (§8 V1 trigger) — the §9c(b) hole, closed.
- **Power refloored**: §9a t1 residual floor (the frozen conservative substitute — the
  artifacts carry no per-dataset pooled-krank residual); provisional HOLDOUT-P
  **n = 504** (seeds 2001–2042), ladder 504 → 804. The two stale defaults the old §13
  recorded (`route_b_stage2_power.py`, `score_route_b_gate.py`) are fixed in `b7553cf`;
  the gate script's cond-2 kill set is unchanged and its verdict on the frozen reports is
  unchanged (29 tests pass).
- **8-task corpus: no gate role** (user decision 2026-08-26) — §9d's pooled-krank
  0.790/0.617 numbers stand in §2 as the honest claim-to-beat, context only.
- **PP0′**: the single-implementation krank arms get independent verification
  (`verify_route_b_scorer_agreement.py` extension, 1e-9) before any training — a
  verification VOID-gate, explicitly not a re-registered kill test for a number §9d
  already saw. **MLP-first training order** registered (the cheap falsifier runs first).

**Status: REGISTERED AND SIGNED OFF. The §9c gating condition is discharged; the §10
build queue (B0–B8, PP0′ first) may start. Nothing was built before this row existed.**

### route_b_v1 — stage 2 build queue: A1 is genuinely T2, B6 closed, pilot cache built (2026-08-26)

Build progress under the v2 registration. **No GNN-vs-MLP performance was measured** —
the registered comparison is now *runnable*, not run.

- **A1 implementation** (commit `28fbe35`). Before this, the "GNN" arm was structurally
  identical to A3: message passing saw only the bipartite task↔platform graph, and
  `masked_topo` decode read one static logit vector computed before any placement was
  committed. Now, behind default-off flags (`NEAR_RTT_MP_DAG_EDGES` /
  `NEAR_RTT_TASK_TYPE_ONEHOT` / `NEAR_RTT_PARTIAL_STATE_EDGES`): undirected workload-DAG
  edges plus the mandatory 4-way task-type one-hot (a fairness repair — the T1 MLP
  already sees task type via krank) enter message passing; the 38 prefix columns enter
  at the EdgeScorer only, in a separate `partial_state_edge_attr` (`edge_attr` stays
  5-wide, load-bearing for the A2/A3 extractors); the decoder re-scores each task
  against the committed prefix via `score_fn`. Serving refuses a
  `partial_state_edge_features` checkpoint (live prefix construction is stage 3) — a
  refusal that initially did NOT fire because `checkpoint_mp_config`'s key whitelist
  silently dropped the sidecar field (memory
  `herosim-sidecar-keys-need-serving-whitelist`). Verified: frozen decoder acceptance
  unchanged at 408 cells; T1/T2 column parity bit-identical keyed on `logit_idx`.
- **B6 closed** (commit `0ac184c`): one shared split artifact,
  `experiments/route_b_stage2_split_v1.json` — 142/31/31 over the 204 pilot parents,
  seed 42, sha256 `0171ef14…`. The GNN loads it via `NEAR_RTT_SPLIT_ARTIFACT`, the MLP
  via `--split-artifact`; both fail loud on any artifact/corpus parent-set mismatch, the
  GNN additionally on the `train_all` / <10-graph bypasses (the sidecar would otherwise
  claim a split the run did not use), and the MLP refuses `--val-size`/`--test-size`
  alongside an artifact. A "draw" therefore varies initialisation and batch order ONLY —
  §3's definition. Sidecar/meta stamp `{path, sha256}`, verified byte-identical across
  an A1 2-epoch smoke and an MLP 1-epoch smoke on the real cache. 18 tests in
  `tests/test_split_artifact.py`, including the cross-trainer parity of the two
  duplicated parent-derivation implementations the scheme keys on.
- **Real DAG cache built**: `graphs_cache_route_b_pilot_s_dag` — 204 graphs from
  `gnn_datasets_dag4_route_b_pilot_v1_arm_s`, built locally in 7.65 s (no datalab job;
  the 8-hour recache precedent was a 2,816-dataset merged corpus). All 204 datasets
  passed every alpha-ladder feasibility gate (`inf`/`3.0`/`2.0`).
  `--platform-feature-dim 14` was passed explicitly — the CLI default is 16 and the
  trainer only *warns* on a mismatch — plus `--queue-feature-contract legacy_v0`. The
  12-graph smoke cache stays: two tests hardcode its path.
- `experiments/route_b_stage2_a1.yaml` repointed to the real cache and pinned to the
  artifact; its B6 do-not-run warning removed as discharged.

**Next, per registered order: write `experiments/route_b_stage2_a{2,3}.yaml` (they do
not exist) and run the MLP arms first, then A1 — multi-seed, all arms on the same split
artifact.** The honest risk stands (`HANDOVER_route_b_stage2_a1.md` §6): the measured
contention ceiling (<10%, `herosim-link-contention-charges-input-ingress`) may be too
low for a GNN win; that outcome would point at the environment (CLAUDE.md option 2),
not at more model work.

### route_b_v1 — stage 2 §9 pre-probe: **NO-GO-PREPROBE** (2026-08-26)

- **Registered deviations, per user decision 2026-08-26:** the §9 pre-probe ran on the
  **pilot-204 corpus** (`gnn_datasets_dag4_route_b_pilot_v1_arm_s` — the stage-1 corpus
  carrying the 35 firing datasets) instead of the registered 12-graph smoke corpus;
  "train-set eval" = the shared split artifact's 142-parent train split (full-204 view
  reported alongside, labelled); 4 draws/arm (seeds 1–4), all arms on split artifact
  sha256 `0171ef14…` — a draw varies init + batch order only (§3). MLP arms trained and
  their aggregate was written to disk **before** any A1 training (§3 registered order;
  `route_b_stage2_preprobe_mlp_aggregate.json`).
- **Build items landed to make the arms runnable:** `run_experiment.py` bare-flag args +
  `--seed` (per-seed env/argv, B6's templated-config option); A3 tied-label dim25cr
  extraction — a label-parity repair mandated by §3's "same labels, same α" (the plain
  dim25cr path labels from `graph.y`, the unconstrained sweep argmin, not the α=2.0
  tied-optimal set); `scripts_cosim/eval_route_b_stage2_arm.py` computing the §6
  registered statistic (decode regret vs the α=2.0 constrained-feasible optimum),
  self-check reproducing the frozen greedy plans to 1e-9 on all 408 stage-1 cells. Note
  recorded: the trainer's internal `regret_masked_topo` divides by the **unconstrained**
  sweep minimum and is a checkpoint-selection convenience, not the registered statistic —
  the two disagree by construction (e.g. `ds_00000`: 19.4% vs 57.3% for the same decoded
  plan).
- **VOID first A1 sweep, cause found and fixed:** the four initial A1 "draws" were one
  draw repeated — `src/notebooks/prepare_graphs_cache.py` seeded random/numpy/torch with
  a hardcoded 42 at **module import**, and `train_near_rtt.py`'s
  `from ...prepare_graphs_cache import DAG_TASK_TYPE_VOCAB` executed that after the
  trainer's own `NEAR_RTT_TRAIN_SEED` seeding, so every draw's weight init and batch
  order came from 42 (sidecars stamped the right `train_seed`; weights bitwise
  identical; wandb curves identical to full precision). **Third seeding-defect class in
  this repo** (see `herosim-pythonhashseed-tiebreak-nondeterminism` and the MLP
  `torch.manual_seed` gap for the first two). Fix: seeding moved into that script's
  `main()`; regression tests go through the real `run_experiment --seed` subprocess
  path (same seed ⇒ bit-identical, different seeds ⇒ diverge — the existing in-process
  determinism tests could not see the import chain). MLP draws were never affected.
- **Results** (mean decode regret vs α=2.0 constrained optimum; train-split = median
  over 4 draws, per-draw in parentheses): A1 GNN(T2) **28.45%** (24.43/32.46/23.67/36.55);
  A2 dim63crk(T1) **19.34%** (19.93/18.74/21.08/13.06); A3 dim25cr(T0) **17.81%**
  (15.76/18.88/16.97/18.65). Full-204 view alongside (labelled): A1 29.38%, A2 20.10%,
  A3 17.29%. Floors: F3 uniform-feasible exact expectation **89.73%** train mean — no
  arm above it, so §8 V1's instrumentation check does not fire; F2 greedy-on-true-
  marginals 8.86% mean / 0.00% median. 0 infeasible completions in all 12 valid draws;
  every tie band collapsed to a point.
- **Reading, applied as registered (§9): A1 train-set regret ≥ A2's ⇒ NO-GO-PREPROBE.**
  A model that cannot beat pointwise-plus-state when both are allowed to memorize will
  not generalize past it. Stage 2 stops here: the B4 fresh-corpus generation and the §8
  gate do not run under this registration. Facts recorded without advocacy: A1 trained
  under its config's `epochs: 40` (convergence not separately verified); the four-task-
  limit sentence of §4 stands ("the corpus is too small to test the architecture claim",
  never "the architecture claim is false").
- **σ calibration (§6), recorded for any future re-registration:** pooled per-dataset
  paired-difference σ (A2−A1) = 18.36%; per-arm seed-to-seed σ A1 13.57% / A2 6.77% /
  A3 7.60% — far above the registered 3.75% trigger, so the provisional n=504 ladder
  start was insufficient regardless.
- **Context observations, no verdict read:** A3(T0) ≤ A2(T1) at 4 draws — the corrected
  T1 layout did not help a trained model here; the cheap-falsifier scenario (T1 decodes
  near-optimally ⇒ V5 on paper) did NOT occur (T1 sits ~19% train regret). Consistent
  with the handover §6 honest risk: the open decision — environment pivot (CLAUDE.md
  option 2) vs closing route B's GNN argument — belongs to the user and needs its own
  registration either way.
- **Artifacts:** `simulation_data/route_b_stage2_{a1,a2,a3}_eval_seed{1..4}.json`,
  per-arm aggregates, `route_b_stage2_preprobe_mlp_aggregate.json`,
  `route_b_stage2_preprobe_readings.json`; checkpoints + sidecars in `models/`
  (gitignored); wandb project `gnn-route-b-stage2`, 16 online runs (8 MLP + 4 void A1 +
  4 valid A1).

## 2026-09-03 — Exploration probe: stage 2's abort statistic inverts once both arms are trained to their fit ceiling

**Not a re-registration.** One seed per arm, no threshold, no verdict. It re-measures the
quantity the stage-2 §9 abort condition was read on, and that quantity moves by an order
of magnitude, so the abort's *premise* is the finding.

**What the abort said.** §9's NO-GO-PREPROBE fired on `A1_median >= A2_median` in **train**
decode regret — deliberately a memorization comparison: if the GNN cannot out-fit the
pointwise competitor on data it has seen, generalization is not worth testing. Measured
then: A1 (T2 GNN) **28.45%**, A2 (T1 MLP + prefix) **19.34%**, A3 (T0 MLP) 17.81%, median
of 4 draws. A1 ran at `epochs: 40`, and the node recorded convergence as unverified.

**It was not at its fit ceiling.** Same arm, same cache, same split artifact, same
`masked_topo` decoder, same α=2.0 — only the epoch budget changes
(`experiments/route_b_fit_a1_e300*.yaml`, `NEAR_RTT_SAVE_FINAL=1`):

| arm | epochs | train regret mean | train median | train zero-regret | test regret mean |
|---|---|---:|---:|---:|---:|
| A1 GNN (val-selected) | 300 | **2.91%** | 0.00% | 87% | 22.56% |
| A1 GNN (last epoch) | 300 | **2.54%** | 0.00% | 94% | 26.15% |
| A1 GNN lr 2e-3 (val-selected) | 300 | 3.55% | 0.00% | 92% | **18.61%** |
| A1 GNN lr 2e-3 (last epoch) | 300 | **1.99%** | 0.00% | 98% | 23.08% |
| A1 MP-OFF (last epoch, flag matched) | 300 | 9.23% | 0.00% | 59% | 11.71% |
| A1 MP-OFF (val-selected, flag matched) | 300 | 12.67% | 0.00% | 53% | **11.14%** |
| A2 MLP + prefix | 600 (patience 600) | 10.41% | 0.00% | 71% | 15.38% |
| — stage 2 reference (median of 4 draws) | 40 / 100 | A1 28.45 · A2 19.34 · A3 17.81 | | | |
| — floors | | F2 greedy-on-true-marginals 8.86 · F3 random-feasible 89.73 | | | |

**Reading, in three parts.**

1. **The abort statistic inverts.** A1 goes 28.45% → 2.0–2.9%, from *worse* than the
   pointwise competitor to **4–5× better**, and below the F2 greedy-on-true-marginals
   floor of 8.86%. The gap is far outside this arm's own recorded draw spread (§6 pooled
   per-dataset σ 13.57 pp; the four draws' means spanned 23.7–36.6%). So "a GNN cannot
   beat pointwise-plus-prefix on this environment even at memorization" measured an
   arm 40 epochs short of memorization. ⚠ One seed — this licenses re-registering the
   pre-probe with a converged budget, not overturning its verdict.
2. **Generalization still goes the other way, and that is the real result.** On held-out
   parents the ordering is MP-OFF 11.1% < MLP-plus-prefix 15.4% < GNN 18.6–26.2%. The
   GNN converts its extra capacity into overfitting on 142 training parents. The honest
   sentence is **fit ceiling favours the GNN, held-out favours the pointwise arms** —
   which is what stage 2 would have concluded from a *generalization* statistic, and it
   never got to run one because the memorization gate aborted first.
3. **Message passing is load-bearing on THIS graph, unlike every earlier corpus.**
   MP-OFF fits 9.2–12.7% against MP-ON's 2.0–2.9% — a ~4× capacity gap from the GIN over
   task↔task DAG edges. `mp_ablation_v1` and `link_mp_v1` both measured MP as neutral or
   harmful, but neither had DAG edges in the graph. It buys capacity here and, at this
   corpus size, spends it on overfitting.

**Instrument defect found and fixed (see GATE TOOLS 2026-09-03).** The MP-OFF row was
first scored at **72.23%** train regret because `GNN_DISABLE_MESSAGE_PASSING` is
weight-invisible, was absent from the contract sidecar, and the offline evaluator neither
set nor checked it — so MP-OFF weights were served *with* message passing. A 5.7× error
that reads as a decisive ablation. The live-gate scorers were protected by a
`run_provenance` assertion; the offline evaluator was not. Sidecars now record
`disable_message_passing` and `eval_route_b_stage2_arm.py` refuses a mismatch.

**Artifacts:** `simulation_data/explore_fit/eval_*.json`, checkpoints
`models/route-b-fit-a1-e300{,-mpoff,-lr2e3}-seed1{,-final}.pt` (+ sidecars),
`models/tabular/route_b_fit_a2_long_seed1.pt`, logs `logs/explore/`.

## 2026-09-06 — Phase 0: instrument audit of the fit-ceiling probe before adding seeds

Decision (user, 2026-09-06): pursue the fit-ceiling split as the live thread, but audit
the apparatus first. Six checks; two changed the reading, one found a bug elsewhere.

1. **Live-loader MP-OFF hole (bug, fixed).** The 2026-09-03 sidecar fix covered the
   offline evaluator only; `executesimulation.load_gnn_model` had the identical hole and
   was never audited. GATE TOOLS 2026-09-06 has the record; it does not touch any number
   in this node.
2. **The regret statistic, not the model, makes the held-out table.** Per-dataset regret
   is heavy-tailed by construction: in every parent sampled (hard and easy alike) only
   **0.1–2.1%** of the 320–1,248 enumerated placements lie within 5% of the optimum and the
   median placement is 2–3× the optimum, so a one-step miss costs 50–200%. On the 31 test
   parents, three datasets carry **52–60%** of every arm's summed regret; in absolute
   seconds the arms' *medians* are 0.03 s (MP-OFF), 0.9 s (MLP+prefix), 2.7–2.9 s (GNN)
   against means of 6–12 s. **`ds_00019` and `ds_00078` are among the worst four for all
   four arms** — a property of those parents, not of any model. The ordering survives a
   robust read: per-dataset paired wins over the 31 test parents are MP-OFF 10.8, MLP 9.0,
   GNN-lr2e-3 6.0, GNN-default 5.2 (21 multi-way ties, mostly at zero). So the 2026-09-03
   direction stands, but its mean-based magnitudes do not; Phase 1 reads paired
   per-seed medians and win counts as primary — the same correction GATE TOOLS 2026-08-19
   made for `gnn_necessity_ablation.py`.
3. **Val selection froze at epoch 140/300** (lr 2e-3 arm): the last "new best val" is at
   epoch 140 and train CE keeps falling for the remaining 160 epochs. Independent
   corroboration of "capacity spent on overfitting"; the last-epoch checkpoint's worse
   test regret (23.08% vs 18.61%) is that overfitting, not noise.
4. **Trainer-side val metric is a faithful proxy (suspected bug, cleared).** The trainer
   validates against the capped near-RTT sidecar and charges an unmapped decode
   `max(worst regret in the cap, 1.0 s)`; the cap's worst regret is **96.7%** of the true
   full-sweep worst regret on the 31 val parents (min 77.6%), so selection is not
   systematically lenient. The 32–45% "unmapped" rate in the training log and the 0
   infeasible decodes at evaluation are the same decodes seen through two lookup tables.
5. **Evaluator is deterministic** — two runs on the same checkpoint are bit-identical.
6. **`partial-state: null` in the A2 yaml means the flag is ON** (`run_experiment.py:181`
   emits the bare flag) — confirmed by `input_dim=63 layout=dim63crk` in the training log,
   so the MLP+prefix arm was what its label says.

7. **The pointwise floor is the decoder, not capacity (measured).** MLP+prefix at
   hidden 256 (`route-b-fit-a2-wide-h256`, same seed 42, 600 epochs): train regret
   **11.95%** vs 10.41% at hidden 64, test 14.31% vs 15.38%, zero-regret share 65% vs
   70% — 4× the width moves nothing past noise, and both sit on the F2
   greedy-on-true-marginals floor (8.86%). So the fit-ceiling table's "MP fits 4–5×
   better" is MP versus a *decoder-bounded* pointwise arm: without message passing the
   scorer cannot beat greedy-on-marginals even when it has memorised the marginals.
   Report `simulation_data/explore_fit/eval_a2_wide_h256.json`.

**Phase 1 launched 2026-09-06, datalab job 740198** (22 tasks): GNN lr 2e-3 seeds 2–8,
**MP-OFF at lr 2e-3** seeds 1–8 (`experiments/route_b_fit_p1_mpoff_lr2e3.yaml` — the
probe's MP-OFF ran at 5e-4 against MP-ON at 2e-3, a learning-rate confound this removes),
MLP+prefix seeds 2–8. Reader: `scripts_cosim/analyze_route_b_fit_p1.py`. Exploration, no
threshold, no verdict.
