# route_b_env_pivot_v1 — AMENDMENT 3: the H2/H3 grid, so S2 becomes computable

> **SIGNED OFF 2026-08-28 by the user.** Drafted 2026-08-27 at `f9384db`, committed at
> `44df151`; §2.4 and §6 revised 2026-08-28 at `71a3b42` after the proposed pair was probed
> and **passed all four bars** — the user signed off with §6's selection hazard stated (read
> §6 first). Amends
> `screen-preregistration.md` @ `019bdcb` as already amended by `screen-amendment-1.md`
> @ `3719aad` and `screen-amendment-2.md` (signed off 2026-08-27). **§3 is now live:** the
> amended H2 and H3 grids may be generated, under §7's reporting obligations. The LINEAGES
> registration entry records this document at its commit SHA.

**Scope: the `replica_configs` grid key and seed block on H2 and H3, plus H3's derived
`MAX_PLACEMENT_COMBINATIONS_SKIP`. Nothing else.** No bar value, no competitor, no α
ladder, no reading rule, no physics, no corpus shape. **H0 and H1 are not re-read and not
regenerated; their verdicts stand as measured.**

## 1. The problem this exists to solve

S2 — the kill bar, the one bar that is the direct negation of what closed stage 2 — **cannot
be computed on the registered grid at all.** Its `t1x` competitor carries 41 parameters, so
the saturation guard needs ≥ 82 sweep rows per dataset; the pivot's scarcity squeeze
produces 16 and 64. It is **refused on 204/204 datasets in both arms** of H0 and H1.

This is not a tool defect. The 2026-08-27 transfer-tool fix recovered S3 and S4, which had
looked equally dead; S2 is the residue that no code change reaches. Nor is it fixable from
the bar side:

- **Do not weaken the competitor.** A 16-row sweep admits ≤ 8 parameters. A competitor that
  small failing to close proves nothing — it is too weak to close anything.
- **Do not make it pooled.** That is what S3 already is. S2 is deliberately the *stronger*
  per-dataset test: if per-dataset fitting cannot close it, no single model can.

The remaining lever is the grid, and it is the honest one, because **the bar stays exactly
as registered** — threshold 0.5, competitor `t1x`, reading `median over the firing stratum
on mean_tied`. Only the corpus it is evaluated on changes.

## 2. The measurement

### 2.1 The wide-arm probe (H2's shape, `per_server` 1→3 and 2→4)

Preset `route_b_pivot_h2_widearm_probe`, generated 204/204 SUCCESS in 23.5 min,
`sweep_complete: true` on all 204, `num_placements` histogram **exactly
`{360: 102, 1680: 102}`**, zero skips. Scored at the rung's registered `--cap-mode
alpha_mean`; independent verifier agrees on **612 (dataset, α) cells to 1e-9** over 2,448
repair values (2 machine-precision ties accepted, the pre-existing escape class).
Denominators `n_exact_scored` = `{360: 102, 1680: 102}`, **zero censoring** — nothing below
is arm-confounded.

**Every bar becomes computable, and S2's saturation goes to zero on both arms:**

| bar | registered value | reading at α=2.0 | |
|---|---|---|---|
| S1 `r_exact.frac_gt_5pct` | ≥ 0.25 | 0.2010 reg / 0.2059 mean_tied / 0.2108 pess | **FAIL** (band agrees) |
| **S2 `t1x` per-dataset** | **< 0.5** | **41/41 fitted, 0 saturated**, median 1.55e-15 | **PASS** |
| S3 extended pooled | ≤ 0.5 | median 1.55e-15; 49,080 rows vs 89 params | **PASS** |
| S4 `hop+coupling` | < 0.8 | median 0.0000, 41/41 fitted | **PASS** |

S2's `by_arm`: `{360: {fitted 15, saturated 0}, 1680: {fitted 26, saturated 0}}`. Against
the registered grid's **refused 204/204**, this is the whole point of the amendment.

**Counters, contention and the α cliff all survive:**

| α | mean feasible rows | `cw_infeas` | `greedy_stuck` | `no_feasible_rows` | `saturated_fit_frac` |
|---|---|---|---|---|---|
| 1.5 | 0.0 | — | 0 | **204** | 0.00 |
| **2.0** | 143.5 | **0.91** | **0** | **0** | **0.00** |
| 3.0 | 758.2 | 0.48 | 0 | 0 | 0.00 |

α=2.0 is clean on both counters, so the registered feasibility fallback does not fire and
the rung reads at its registered primary. Contention binds **harder** than the registered
grid, not softer (0.91 componentwise-infeasible).

### 2.2 The squeeze is untouched — measured, not assumed

`ladder-findings.md` and the node previously stated that widening `replica_configs` would
loosen the squeeze H0 exists to create. **That was an assumption and it is false.** The
squeeze is `server_node_counts=[4] × replica_server_percentage=0.5`, which places replicas
on exactly **2 hosting nodes**; `per_server` sets how many platform slots sit on those two
nodes, not how many nodes there are. Hosting-node histogram, every dataset of every corpus:

| corpus | hosting nodes | candidate pool |
|---|---|---|
| H0 | `{2: 204}` | 8 / 12 |
| H1 | `{2: 204}` | 8 / 12 |
| H2 (registered) | `{2: 204}` | 2 / 4 |
| H2 wide-arm probe | `{2: 204}` | 6 / 8 |

**This correction is recorded here and in the node; the earlier claim is withdrawn.**

### 2.3 H3 does not generate at all as registered

Under `replica_overlap` every task type draws from **one** pool of
`per_server × n_hosting_nodes` slots, and `generate_brute_force_placement_combinations`
requires **globally distinct** replicas across tasks. H3 is `dag_instances=2` — 8 tasks (two
instances of the same 4 task types) drawing on that one pool — so it needs a pool of ≥ 8.

Measured by `route_b_pivot_h3_genprobe_registered` (2 seeds × both arms × 2 conn-probs × 3
queue-dists = 24 datasets, `MAX_PLACEMENT_COMBINATIONS_SKIP` raised to 5e9 so a skip cannot
be the threshold):

| arm | pool | pre-uniqueness product | result |
|---|---|---|---|
| `per_server=1` | 2 | 256 | **0/12 — `uniqueness_exhausted`** |
| `per_server=2` | 4 | 65,536 | **0/12 — `uniqueness_exhausted`** |

**H3 as registered generates 0/204.** This has never been measured before; H3 has never been
generated. Without this amendment the rung is a structural VOID-GENERATION and the ladder
cannot be exhausted. (The skip reasons also re-confirm the `f407f91` attribution fix on a
grid it had not been exercised on: `uniqueness_exhausted`, not `too_many_combinations`.)

Measured for the candidate arms by `route_b_pivot_h3_genprobe_wide`:

| arm | pool | pre-uniqueness | **unique rows / dataset** |
|---|---|---|---|
| `per_server=4` | 8 | 16,777,216 | **40,320** = 8P8 |
| `per_server=5` | **9** | 43,046,721 | **362,880** = 9P8 |

The `per_server=5` pool is **9, not the 10 the arithmetic predicts** — one hosting node
carries fewer platforms of a suitable type. Measured on all 12 datasets of that arm.

### 2.4 The proposed pair, probed at 4 tasks — it passes all four bars

`route_b_pivot_h2_proposed_probe` is H2's shape with **exactly the `replica_configs` §3
proposes**, generated on H2's **currently registered** seeds 3201–3217 (already burned by
§2.1's probe) precisely so the fresh block stays unseen. 204/204 SUCCESS in 50.1 min,
`sweep_complete: true` on all 204, `num_placements` histogram exactly
**`{1680: 102, 3024: 102}`** — the predicted 8P4 and 9P4, confirmed. Independent verifier:
**612 (dataset, α) cells agree to 1e-9** over 2,448 repair values, **0** machine-precision
ties. Denominators `{1680: 102, 3024: 102}`, zero censoring.

| bar | registered | reading at α=2.0 | |
|---|---|---|---|
| S1 `r_exact.frac_gt_5pct` | ≥ 0.25 | 0.2843 reg / 0.2941 mean_tied / **0.3137 pess** / 0.2500 opt | **PASS**, band agrees |
| S2 `t1x` per-dataset | < 0.5 | median 0.0000; **58/58 fitted, 0 saturated** | **PASS** |
| S3 extended pooled | ≤ 0.5 | median 0.0000; 143,136 rows vs 106 params | **PASS** |
| S4 `hop+coupling` | < 0.8 | median 0.0000; 58/58 fitted | **PASS** |

Counters clean at the registered primary α (`greedy_stuck` 0, `no_feasible_rows` 0,
`saturated_fit_frac` 0.00); α=1.5 remains 204/204 infeasible, so the cliff survives;
`componentwise_infeasible_frac` **0.93**, the tightest of any corpus in this lineage.
S2's `by_arm` is `{1680: {fitted 24, saturated 0}, 3024: {fitted 34, saturated 0}}`.

**Per arm**, because a pooled number that one arm carries is the defect this lineage keeps
producing:

| arm | `per_server` | mean feasible rows | `cw_infeas` | S1 reg / mean_tied / pess |
|---|---|---|---|---|
| 1,680 | 4 | 232.9 | 0.90 | 0.2353 / 0.2549 / 0.2647 |
| 3,024 | 5 | 381.2 | 0.90 | 0.3333 / 0.3333 / 0.3627 |

Both arms fire and both clear the bar on `pessimistic`; the `per_server=4` arm's
`registered` member sits just under it (0.2353) and the pooled pass is carried more by the
`per_server=5` arm. Stated so the pooled number is not read as uniform.

**Three things this probe is NOT.** It is not a rung reading — S0's **paired separable
control has not been generated**, and S0 is a VOID gate that can still fail. It is not on
the seeds the amended rung would use. And the transfer tool's own top-level `verdict` for
this corpus is `VOID-KINT-CONFOUNDED`, inherited from the route_b_v1 §9b/§9c machinery —
that is not one of S1–S4 and does not bear on them, but it is reported here rather than
omitted.

## 3. What is proposed

**One change, applied to H2 and H3 alike:**

```python
"replica_configs": [
    (0, 4, 0.7, 0.5),
    (0, 5, 0.7, 0.5),
],
```

with, per §3's fresh-seed-block-per-rung discipline:

| rung | seeds | pool | rows / dataset | `MAX_PLACEMENT_COMBINATIONS_SKIP` |
|---|---|---|---|---|
| H2 (amended) | **3401–3417** (fresh) | 8 / 9 | **1,680 / 3,024 (measured, §2.4)** | default suffices (products 4,096 / 6,561) |
| H3 (amended) | 3301–3317 (unchanged, never generated) | 8 / 9 | 40,320 / 362,880 (measured) | **≥ 25,600,000,000** |

**H3's skip threshold, re-derived in §3's own conservative style.** The registered
16,777,216 came from `max(per_server)=2 × server_node_counts=4 = 8` candidates per type,
`8^8`. At `max(per_server)=5` the same conservative bound is `5 × 4 = 20`, `20^8 =
25,600,000,000`; register **30,000,000,000** for headroom. The *measured* products are
smaller (16,777,216 and 43,046,721) because the true pool is `per_server × 2 hosting nodes`
capped by platform-type suitability — but the registered value must be the derived bound,
not the observed one, since the threshold tests the **pre-uniqueness** product and sizing it
from observed sweeps is exactly the defect
`herosim-cosim-skip-threshold-is-pre-uniqueness` records. Note the comparison is strict
`>`, so the old 16,777,216 would admit the `per_server=4` arm with **zero** headroom and
skip the `per_server=5` arm outright.

**Why one pair for both rungs, and why `per_server ≥ 4`.** H3 cannot generate below a pool
of 8, which forces `per_server ≥ 4` there regardless of anything else. Using the same pair
on H2 keeps **H2 and H3 cell-for-cell comparable to each other**, which is the comparability
that the ladder's last two rungs actually need, and it is the choice that requires the
fewest registered values to move.

**Fresh seeds for H2 are not optional.** The wide-arm probe inherited H2's registered seeds
3201–3217, and §2.1's S1 reading was taken on that corpus. Registering H2 on those seeds
would mean reading a bar on a corpus whose bar has already been seen. 3401–3417 is a fresh
block, distinct from 3001–3017, 3101–3117, 3201–3217 and 3301–3317.

## 4. What this amendment does NOT propose

- **No bar value moves.** S1 stays ≥ 0.25, S2 stays `t1x` median < 0.5 on `mean_tied`, S3
  stays ≤ 0.5, S4 stays < 0.8. S0's gates are unchanged.
- **No competitor changes.** `t1x` remains t1 + hetdem + futureint + linkrank at 41
  parameters. The saturation guard (`n_rows ≥ 2 × n_params`) is unchanged.
- **No α ladder, cap mode, reading rule, physics or corpus shape changes.** All rungs keep
  the 204-dataset 2 × 2 × 3 × 17 shape; this changes the *values* in one grid key, not the
  shape.
- **H0 and H1 are untouched.** Not re-read, not regenerated, not re-scored. Their verdicts —
  H0 VOID-TIE-INDETERMINATE on S1, H1 FAIL on S1 — stand exactly as measured. Re-running
  them under a changed grid would be re-reading a rung after seeing its result.
- **No verdict on any rung.** §2.4 measures a *probe* on an unregistered grid and unregistered-for-this-purpose seeds, with S0's control not yet generated. It is evidence about what the amended grid does, not a rung reading, and it is filed as such. See §6.

## 5. The cost, stated plainly

**H2 and H3 stop being cell-for-cell comparable with H0 and H1.** The registered ladder was
built so a rung's numbers sit beside the frozen pilot and beside each other at matched
shape; after this, the last two rungs share a shape with each other but not with the first
two.

This is judged acceptable because §5's reading rule is "**first rung passing S1–S4 against
absolute bars**", not a monotone rung-to-rung comparison — no bar is defined relative to a
previous rung's value. But it is a real cost, it is not recoverable, and it should be
weighed at sign-off rather than discovered later.

A second, smaller cost: H3's `per_server=5` arm is 362,880 full simulations per dataset.
At the wide-arm probe's measured throughput this rung is a **datalab job**, not a local run
— which §3 already anticipated for H3 on other grounds.

## 6. Consequence stated in advance, so no reading of it can be post-hoc

**On the pair this amendment proposes, all four bars pass.** §2.4 measured it. Signing this
amendment is therefore not a procedural tidy-up that unblocks a stalled bar — it is very
likely to produce the ladder's **first rung passing S1–S4**, which under §5 of the
registration means **PIVOT-CANDIDATE**: the ladder stops, and drafting a v3 training
registration becomes licensed. **Sign it knowing that, or do not sign it.**

**The selection hazard, at its maximum, named rather than hidden.** This is the part that
needs the user's judgement, not mine. The sequence was: S2 was uncomputable → a wide-arm
probe fixed that but read S1 at 0.2010 (FAIL) → its per-arm split showed `per_server=3` at
0.1471 and `per_server=4` at 0.2549 → the proposed pair moved *up* to `per_server` 4 and 5
→ that pair passes S1 at 0.2843. **A grid was adjusted and the bar then passed.** That is
the shape the registration exists to prevent, and no amount of good faith in the reasoning
changes the shape.

What bounds it, stated as bounds and not as absolution:

1. **The choice has a structural justification that does not reference S1, and that
   justification came first.** H3 cannot generate below a pool of 8, which forces
   `per_server ≥ 4` there on pain of 0/204; using one pair for both rungs is what keeps H2
   and H3 comparable to each other. That argument selects `per_server ≥ 4` with every S1
   number unseen. It is recorded in §3 on its own terms.
2. **The registered corpus is not the probed corpus.** H2's fresh block 3401–3417 means the
   S1 that decides the amended rung is read on 204 datasets nobody has scored. Given §2.4,
   **this clause is now load-bearing rather than hygienic** — it is the only thing standing
   between a probed pass and a registered one, and it must not be relaxed for the
   convenience of reusing an existing corpus.
3. **S0 can still fail.** The paired separable control for the amended grid does not exist
   yet. S0 is a VOID gate; if the control does not reach `r_exact.frac_gt_1pct ≤ 0.02` the
   rung is VOID regardless of §2.4.
4. **A probe is not a verdict, and this document does not record one.** §2.4's numbers are
   filed as probe artifacts on an unregistered grid, and the node says so.

**If the user judges the hazard too large, the honest alternatives are:** register a
different fresh pair chosen without reference to §2.4 (any pool ≥ 8 pair generates); or
accept §8's option and drop S2 ladder-wide. Both are worse on the measurements and better
on the optics, and that trade is the user's to make, not this document's.

## 7. Reporting obligations if signed off

Same treatment every registered change in this lineage has received:

- **Per-arm everything.** Every statistic quoted from an amended rung carries its
  `censoring_by_arm` breakdown and its `n_exact_scored` / `n_greedy_scored`, on the
  `n_rows` key. A denominator concentrated in one arm makes the statistic "over that arm
  only" whatever its name — five of this lineage's six defects were that failure.
- **Generation integrity before scoring.** 204/204 `sweep_complete: true` and the
  `num_placements` histogram asserted (`{1680: 102, 3024: 102}` for H2) before any bar is
  read. `SUCCESS` counts and zero skip files are not evidence of a usable sweep.
- **The full five-variable env recipe** — `HEROSIM_DATA_LOCALITY`,
  `HEROSIM_OUTPUT_SIZE_BYTES`, `HEROSIM_COSIM_KEEP_ALIVE`, `HEROSIM_RETAIN_TASK_TIMES`, and
  `HEROSIM_STORAGE_NEUTRAL` unset for Arm S / set for the paired control.
- **Paired separable control per amended rung**, under AMENDMENT 1's definition, passing
  S0's `r_exact.frac_gt_1pct ≤ 0.02` before any S1–S4 number is read.
- **Independent verifier to 1e-9 on every (dataset, α) cell**, refusals included;
  disagreement is an S0 VOID.
- **Move no threshold.** S1–S4 read on their registered values, unchanged.

## 8. The alternative, and why it was not taken

**Declare S2 `VOID-UNCOMPUTABLE` for the whole ladder and read only S1/S3/S4.** It is free,
it moves no grid, and it preserves cell-for-cell comparability across all four rungs.

It was rejected because S2 is the bar most tied to the research question. It is the
per-dataset kill test — the direct negation of what closed stage 2 — and it is the only bar
that asks whether a *per-dataset*-fitted pointwise competitor, which is strictly stronger
than any single trained model, fails to close the structure. A screen that cannot read it
can conclude "structure fires and a pooled model does not close it", which is materially
weaker than what the pivot was registered to decide. Given that the alternative costs a grid
key and the bar itself stays fixed, dropping S2 is the more expensive option in the currency
that matters.

A third option — re-registering S2 against a competitor that fits a 64-row sweep — is
covered in §1 and rejected there. The node and the LINEAGES index row previously framed the
open question that way; that framing is superseded by this document.

## 9. Sign-off

Amends `screen-preregistration.md` @ `019bdcb`, as already amended by
`screen-amendment-1.md` @ `3719aad` and `screen-amendment-2.md`.

**Signed off by the user: ☑ 2026-08-28.** Given after §2.4's all-pass probe result and
§6's selection hazard were put to the user explicitly, not before them.

**What the sign-off licenses, and only this:** generating the amended H2 (fresh seeds
3401–3417) and H3 (3301–3317) under §3's `replica_configs`, each with its paired separable
control, and reading S0–S4 on them under §7's obligations. It does **not** license
re-reading H0 or H1, moving any bar, or treating §2.4's probe as a rung verdict.

**S0 is still the first gate.** The paired separable control for the amended grid does not
exist yet; if it fails `r_exact.frac_gt_1pct ≤ 0.02` the rung is VOID no matter what §2.4
measured. Generate the control before reading any S1–S4 number on the registered corpus.
