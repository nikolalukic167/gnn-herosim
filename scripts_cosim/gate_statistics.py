"""Gate statistics for the topology_transfer_v1 Phase 4 gate.

WHY THIS MODULE EXISTS
----------------------
The pre-registered Phase 4 gate reads the *slope of regret against topology
size*. Measured on `shallow_v1` (200 datasets, sweeps 16..17248 plans), that
slope is not safe to read off `regret_mean` / `p90` / `max`:

Holding the decision rule FIXED at every size -- the additive-fit argmin, whose
expressive class is identical at 16 plans and at 17k -- and binning datasets by
sweep size, the aggregate still drifts:

Every row is the GAP between two fixed rules, normalised the same way
(range / mean |value|). An honest gate statistic would read 0.00.

    statistic                 drift
    regret_mean                2.58
    trimmed mean (10%)         3.68   <- trimming makes it WORSE, and flips sign
    regret_p90                 2.83
    regret_max                 1.41
    log-mean (log1p)           2.64   <- log does NOT fix it
    headroom-normalised        2.92   <- normalising by the landscape is worse
    opt_recovered_frac         2.12   <- and it flips SIGN in the top bin
    median per-dataset ratio   0.00   <- degenerate, no resolution
    mean per-dataset ratio     0.27
    win rate                   0.36

So the fix is NOT a more robust average over datasets -- the binding problem was
never outlier-robustness, it is aggregation ORDER. Between-dataset regret
heterogeneity is what tracks sweep size, and averaging raw regrets across
datasets lets it leak into the gate. Comparing the two models **within each
dataset first** and only then aggregating a bounded per-dataset comparison
removes most of it.

`regret_median` and the median headroom ratio are degenerate on this corpus
(61.5% of datasets are solved exactly by the additive rule, so the median is
0.0 / 1.0 and carries no resolution). They are reported, never gated on.

WHAT IS GATED
-------------
Primary   `win_rate`            -- chance is exactly 0.5 at EVERY topology size,
                                   which is the size-invariant null the raw
                                   regret slope does not have.
Co-primary `regret_ratio_mean`  -- keeps magnitude, which win_rate discards.
Reported   regret mean/p90/max  -- comparability with earlier LINEAGES rows;
                                   demoted from gate criteria, not removed.

POWER
-----
Paired bootstrap on the same corpus: SE of the mean-regret gap is 0.037 at
n=30 held-out datasets, i.e. a minimum detectable gap of ~0.149 -- while the
GNN-vs-pointwise gaps actually observed on shallow_v1 are 0.003-0.02. At n=30
the gate is under-powered for its own effect by 7-50x, which is why seed 44
reversed its verdict between two identical commands. Pairing recovers little
(per-dataset regret correlation between two rules is only 0.349, so paired SE
0.0737 vs unpaired 0.0838). `power_note()` makes this loud at report time
instead of leaving it to be discovered after a multi-seed run.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, Optional

import numpy as np

TIE_ATOL = 1e-9


def _binom_two_sided_p(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial sign test. No scipy dependency."""
    if n <= 0:
        return float("nan")

    def pmf(i: int) -> float:
        return math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))

    obs = pmf(k)
    # sum every outcome no more likely than the observed one
    total = sum(pmf(i) for i in range(n + 1) if pmf(i) <= obs * (1 + 1e-12))
    return float(min(1.0, total))


def paired_regret_comparison(
    model_per_ds: Dict[str, float],
    ref_per_ds: Dict[str, float],
    n_boot: int = 4000,
    seed: int = 0,
) -> dict:
    """Compare a model against a reference on the datasets they BOTH decided.

    Every statistic here is computed per dataset and only then aggregated --
    see the module docstring for why aggregating raw regrets instead lets
    landscape heterogeneity masquerade as a size trend.
    """
    shared = sorted(set(model_per_ds) & set(ref_per_ds))
    if not shared:
        return {"n_paired": 0}
    m = np.array([model_per_ds[d] for d in shared], float)
    r = np.array([ref_per_ds[d] for d in shared], float)

    wins = int(np.sum(m < r - TIE_ATOL))
    losses = int(np.sum(m > r + TIE_ATOL))
    ties = len(shared) - wins - losses
    win_rate = (wins + 0.5 * ties) / len(shared)

    # magnitude, per dataset, before any averaging
    ratio = (1.0 + m) / (1.0 + r)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(shared), (n_boot, len(shared)))
    wr_boot = np.array(
        [
            (np.sum(m[i] < r[i] - TIE_ATOL) + 0.5 * np.sum(np.abs(m[i] - r[i]) <= TIE_ATOL))
            / len(i)
            for i in idx
        ]
    )
    gap_boot = np.array([m[i].mean() - r[i].mean() for i in idx])

    return {
        "n_paired": len(shared),
        # --- primary
        "win_rate": float(win_rate),
        "win_rate_ci95": [float(np.percentile(wr_boot, 2.5)),
                          float(np.percentile(wr_boot, 97.5))],
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "sign_test_p": _binom_two_sided_p(wins, wins + losses) if (wins + losses) else float("nan"),
        # --- co-primary (magnitude)
        "regret_ratio_mean": float(ratio.mean()),
        "regret_ratio_median": float(np.median(ratio)),
        # --- reported, not gated
        "regret_gap_mean": float(m.mean() - r.mean()),
        "regret_gap_se": float(gap_boot.std()),
        "min_detectable_gap": float(2.0 * gap_boot.std()),
    }


def power_note(cmp: dict, observed_gap: Optional[float] = None) -> str:
    """One line saying whether this comparison could have resolved its own effect."""
    if not cmp.get("n_paired"):
        return "no paired datasets"
    mdg = cmp["min_detectable_gap"]
    gap = abs(observed_gap if observed_gap is not None else cmp["regret_gap_mean"])
    if gap >= mdg:
        return f"resolved (|gap| {gap:.4f} >= MDG {mdg:.4f})"
    ratio = mdg / gap if gap > 0 else float("inf")
    return (
        f"!! UNDERPOWERED: |gap| {gap:.4f} is {ratio:.0f}x below the minimum "
        f"detectable gap {mdg:.4f} at n={cmp['n_paired']}. The regret gap is "
        f"noise at this test-split size; read win_rate instead."
    )


# --- Pre-registered Phase 4 power ladder -------------------------------------
# FIXED 2026-08-19, BEFORE any topo_transfer_v1 corpus exists. This is the whole
# point: v1's gate was falsified because its PASS condition was readable off
# landscape drift, and that was only discovered because the criterion had been
# written down in advance. Choosing a power threshold AFTER seeing a borderline
# number is the same failure with a different mechanism.
#
# The ladder is entered at tier 0.02 as a cheap first pass, NOT as a standalone
# decision. The arithmetic that forces the escalation rule: the win_rate
# deviations from 0.5 actually observed on shallow_v1 are 0.017-0.033 (seeds
# 42/43/44), while tier 0.02 at 360 held-out datasets/size buys MDG ~= 0.021 --
# INSIDE that range, not below it. So a straddling CI at tier 0.02 is the
# EXPECTED outcome against an effect this size, and reporting it as "topology
# transfer failed" would be a false negative manufactured by under-powering.
#
# UNITS -- read before comparing these numbers to the LINEAGES MDG table. The tier
# NAMES come from that table, which is in REGRET-GAP units (2*SE of the mean-regret
# gap). The primary statistic is `win_rate`, so `required_n_for_effect` works in
# win_rate units instead: the CI half-width of a proportion, ~1.96*sqrt(0.25/n).
# The two are not interchangeable -- the tier names are labels, not the quantity
# solved for. In win_rate units the observed shallow_v1 effects need, per held-out
# size:
#
#     effect (|win_rate - 0.5|)   datasets/size needed
#     0.033 (seed 42)                    ~880    <- tier 0.01 covers this
#     0.017 (seeds 43/44)              ~3,400    <- NO registered tier covers this
#
# That asymmetry is why `phase4_verdict` can return INCONCLUSIVE_LADDER_EXHAUSTED.
# If the true effect sits at the BOTTOM of the observed range, even tier 0.01 will
# not resolve it, and the ladder has to say so rather than decay into a FAIL.
# tier_launch ADDED 2026-08-19, after the arithmetic above showed tier_0.02 cannot
# resolve either observed effect and tier_0.01 (1600/size, ~16h wall-clock) is not
# worth pre-committing to for the STRONGER of the two effects alone. ~900/size covers
# the 0.033 effect (needs ~880) at roughly the cost already budgeted for tier_0.02,
# and is the tier Phase 4 actually launches at if tier_0.02 escalates. The weaker
# effect (0.017, ~3,400/size) stays off the ladder -- see the module docstring; that
# is a separate, larger, not-yet-approved allocation, not silently folded in here.
# `target_win_rate_effect` is the effect each tier resolves in the PRIMARY statistic's
# own units (win_rate CI half-width at that n) -- unlike `target_mdg` (regret-gap
# units, kept only for cross-reference to the old LINEAGES table), this is the number
# that is actually load-bearing in `required_n_for_effect` / `phase4_verdict`.
PHASE4_TIERS = (
    {"name": "tier_0.02", "target_mdg": 0.02, "target_win_rate_effect": 0.052,
     "datasets_per_size": 360},
    {"name": "tier_launch", "target_win_rate_effect": 0.033, "datasets_per_size": 900},
    {"name": "tier_0.01", "target_mdg": 0.01, "target_win_rate_effect": 0.024,
     "datasets_per_size": 1600},
)

VERDICT_PASS = "PASS"
VERDICT_FAIL = "FAIL"
VERDICT_ESCALATE = "ESCALATE"
VERDICT_EXHAUSTED = "INCONCLUSIVE_LADDER_EXHAUSTED"


def required_n_for_effect(cmp: dict, target_effect: Optional[float] = None) -> Optional[int]:
    """Held-out datasets/size needed to resolve an effect of the observed size.

    The `win_rate` CI half-width shrinks as 1/sqrt(n), so scaling the CURRENT
    half-width down to the observed |win_rate - 0.5| gives the n at which this
    effect would clear its own noise floor. `None` when the observed effect is
    exactly 0 (no finite n resolves it).
    """
    if not cmp.get("n_paired"):
        return None
    lo, hi = cmp["win_rate_ci95"]
    half_width = (hi - lo) / 2.0
    effect = target_effect if target_effect is not None else abs(cmp["win_rate"] - 0.5)
    if effect <= 0 or half_width <= 0:
        return None
    return int(math.ceil(cmp["n_paired"] * (half_width / effect) ** 2))


def phase4_verdict(cmp: dict, tier_name: str = "tier_0.02", tiers=PHASE4_TIERS) -> dict:
    """Pre-registered verdict for one paired comparison at one power tier.

    A straddling CI is NOT a null result -- it is an under-powered one, and the
    two are reported differently on purpose:

      PASS      CI excludes 0.5 on the model's side.
      FAIL      CI excludes 0.5 on the REFERENCE's side. This is the only
                outcome that licenses "the model does not transfer".
      ESCALATE  CI straddles 0.5 AND a higher tier in the ladder carries enough
                datasets/size to resolve the observed effect. Auto-escalates;
                it is not a verdict, it is a request for more corpus.
      INCONCLUSIVE_LADDER_EXHAUSTED
                CI straddles 0.5 and no tier in the ladder can resolve the
                effect. Report the effect size and the n it would need; still
                never report it as a failure to transfer.
    """
    if not cmp.get("n_paired"):
        return {"verdict": VERDICT_ESCALATE, "reason": "no paired datasets", "tier": tier_name}

    lo, hi = cmp["win_rate_ci95"]
    effect = abs(cmp["win_rate"] - 0.5)
    need = required_n_for_effect(cmp)

    if lo > 0.5:
        return {"verdict": VERDICT_PASS, "tier": tier_name, "win_rate": cmp["win_rate"],
                "ci95": [lo, hi], "effect": effect, "required_n": need,
                "reason": f"CI [{lo:.3f},{hi:.3f}] excludes 0.5 above"}
    if hi < 0.5:
        return {"verdict": VERDICT_FAIL, "tier": tier_name, "win_rate": cmp["win_rate"],
                "ci95": [lo, hi], "effect": effect, "required_n": need,
                "reason": f"CI [{lo:.3f},{hi:.3f}] excludes 0.5 below -- the reference wins"}

    # Smallest tier that both (a) resolves the observed effect and (b) actually buys
    # more datasets than this run had. Condition (b) matters because a seed-level run
    # (n=30) can be resolved by a tier it simply has not been run at yet, and that is
    # an escalation to that tier -- not to the one above it.
    nxt = None
    if need is not None:
        for t in tiers:
            if t["datasets_per_size"] >= need and t["datasets_per_size"] > cmp["n_paired"]:
                nxt = t
                break

    if nxt is not None:
        return {"verdict": VERDICT_ESCALATE, "tier": tier_name, "escalate_to": nxt["name"],
                "escalate_datasets_per_size": nxt["datasets_per_size"],
                "win_rate": cmp["win_rate"], "ci95": [lo, hi], "effect": effect,
                "required_n": need,
                "reason": (f"CI [{lo:.3f},{hi:.3f}] straddles 0.5 at n={cmp['n_paired']}; "
                           f"an effect of {effect:.3f} needs n>={need}")}

    return {"verdict": VERDICT_EXHAUSTED, "tier": tier_name, "win_rate": cmp["win_rate"],
            "ci95": [lo, hi], "effect": effect, "required_n": need,
            "reason": (f"CI [{lo:.3f},{hi:.3f}] straddles 0.5 and no pre-registered tier "
                       f"reaches the n>={need} an effect of {effect:.3f} would need")}


def escalation_note(verdict: dict) -> str:
    """One line for the report. Loud on the distinction that matters."""
    v = verdict.get("verdict")
    if v == VERDICT_PASS:
        return f"GATE {v} @ {verdict['tier']} -- {verdict['reason']}"
    if v == VERDICT_FAIL:
        return f"GATE {v} @ {verdict['tier']} -- {verdict['reason']}"
    if v == VERDICT_ESCALATE:
        # A run smaller than its own tier (a seed-level run, say) escalates TO that
        # tier; printing "@ tier_0.02 -> tier_0.02" would just read as a bug.
        at = "" if verdict["escalate_to"] == verdict["tier"] else f" @ {verdict['tier']}"
        return (
            f">> GATE {v}{at} -> {verdict['escalate_to']} "
            f"({verdict['escalate_datasets_per_size']}/held-out size). {verdict['reason']}. "
            f"NOT a null result: this is the expected outcome at this power against an "
            f"effect this size. Pre-registered 2026-08-19; do not report as a failure to "
            f"transfer."
        )
    return (
        f">> GATE {v} @ {verdict['tier']}. {verdict['reason']}. Report the effect size and "
        f"the n it needs -- still NOT a failure to transfer."
    )


# --- Multi-seed pooling ------------------------------------------------------
# FIXED 2026-08-19. The v2 gate's FAIL condition ("the two co-primary statistics
# disagree in sign", LINEAGES topology_transfer_v1) is only safe to evaluate on
# AGGREGATED, multi-seed statistics. Evaluated per seed it is not robust: seed 44
# on shallow_v1 has win_rate 0.533 (GNN ahead) and regret_ratio_mean 1.999 (GNN
# far behind on magnitude) simultaneously, from a single cliff-shaped dataset
# (ds_00157) blowing up one seed's split -- a real bimodal distribution (ratio
# ~0.99 in calm seeds, ~2.0 in blow-up seeds), not gate noise. A per-seed FAIL
# rule lets exactly one unlucky seed veto the whole lineage.
#
# The fix: pool `win_rate` across seeds as a mean with a seed-level CI (already
# how the frozen 5-seed calibration reports it), and pool `regret_ratio_mean`
# across seeds as a MEDIAN, not a mean -- a median is robust to exactly one seed
# landing in the blow-up mode where a mean is not. Compare the sign of those two
# pooled statistics ONCE, at the end, not seed-by-seed.
def pool_seed_comparisons(seed_comparisons: Iterable[dict]) -> dict:
    """Pool per-seed `paired_regret_comparison` dicts into one multi-seed summary.

    `win_rate` pools as mean-of-seeds with a CI from the seed-to-seed sd (95%,
    normal approx: mean +/- 1.96*sd/sqrt(n_seeds)) -- the seed calibration showed
    this is the LARGER source of variance, not within-seed bootstrap noise.
    `regret_ratio_mean` pools as the MEDIAN of the per-seed values, for the
    robustness reason in the section docstring above.
    """
    usable = [c for c in seed_comparisons if c.get("n_paired")]
    n_seeds = len(usable)
    if n_seeds == 0:
        return {"n_seeds": 0}

    win_rates = np.array([c["win_rate"] for c in usable], float)
    ratios = np.array([c["regret_ratio_mean"] for c in usable], float)
    mean_wr = float(win_rates.mean())
    sd_wr = float(win_rates.std(ddof=1)) if n_seeds > 1 else 0.0
    half_width = 1.96 * sd_wr / math.sqrt(n_seeds) if n_seeds > 1 else float("inf")

    return {
        "n_seeds": n_seeds,
        # datasets/held-out-size, for the escalation ladder -- the SMALLEST seed's
        # n_paired, so a partial/uneven run never claims more power than it has.
        "n_paired": int(min(c["n_paired"] for c in usable)),
        "win_rate": mean_wr,
        "win_rate_sd": sd_wr,
        "win_rate_ci95": [mean_wr - half_width, mean_wr + half_width],
        "regret_ratio_median": float(np.median(ratios)),
        "regret_ratio_per_seed": ratios.tolist(),
    }


def co_primary_sign_agree(pooled: dict, atol: float = TIE_ATOL) -> Optional[bool]:
    """Do pooled `win_rate` and pooled `regret_ratio_median` point the same way?

    `None` when either statistic sits exactly at its null (0.5 / 1.0) -- a tie
    has no sign to disagree with. `regret_ratio` < 1 favors the model, same
    direction as `win_rate` > 0.5, so the comparison is `(win_rate - 0.5) > 0`
    against `(1 - ratio) > 0`.
    """
    if not pooled.get("n_seeds"):
        return None
    wr = pooled["win_rate"] - 0.5
    rr = 1.0 - pooled["regret_ratio_median"]
    if abs(wr) <= atol or abs(rr) <= atol:
        return None
    return (wr > 0) == (rr > 0)


def pooled_phase4_verdict(
    seed_comparisons: Iterable[dict], tier_name: str = "tier_0.02", tiers=PHASE4_TIERS
) -> dict:
    """`phase4_verdict`, pooled across seeds, with the co-primary sign check.

    The escalation ladder (PASS/FAIL-on-CI/ESCALATE/EXHAUSTED) runs on the
    pooled `win_rate` CI exactly as `phase4_verdict` does on a single-seed one.
    On top of that: a result that would otherwise PASS (CI excludes 0.5 above)
    is downgraded to FAIL if the pooled median `regret_ratio` disagrees in sign
    -- win_rate says the model wins more often, but the co-primary magnitude
    statistic says it loses on balance. One outlier seed cannot trigger this by
    itself: `regret_ratio` is pooled as a median, not a mean (see section
    docstring), so a single blow-up seed does not move it.
    """
    pooled = pool_seed_comparisons(seed_comparisons)
    if not pooled.get("n_seeds"):
        return {"verdict": VERDICT_ESCALATE, "reason": "no seeds with paired datasets",
                 "tier": tier_name}

    verdict = phase4_verdict(pooled, tier_name=tier_name, tiers=tiers)
    verdict["n_seeds"] = pooled["n_seeds"]
    verdict["regret_ratio_median"] = pooled["regret_ratio_median"]
    agree = co_primary_sign_agree(pooled)
    verdict["co_primary_sign_agree"] = agree

    if verdict["verdict"] == VERDICT_PASS and agree is False:
        lo, hi = verdict["ci95"]
        verdict["verdict"] = VERDICT_FAIL
        verdict["reason"] = (
            f"win_rate CI [{lo:.3f},{hi:.3f}] excludes 0.5 above, but the pooled "
            f"median regret_ratio ({pooled['regret_ratio_median']:.3f}) disagrees in "
            f"sign across {pooled['n_seeds']} seeds -- co-primaries do not agree"
        )

    return verdict


def size_trend(per_size: Dict[object, dict], key: str = "win_rate") -> dict:
    """Trend of a gate statistic across topology sizes, with monotonicity.

    `per_size` maps size -> the dict returned by `paired_regret_comparison`.
    Monotonicity is reported, never asserted: the Phase 4 PASS condition is a
    trend that exceeds what a CONSTANT-QUALITY rule pair drifts by at the same
    sizes (see `drift_anchor_note`), not a bare monotone sequence.
    """
    sizes = sorted(per_size)
    vals = [per_size[s].get(key) for s in sizes]
    ok = [v for v in vals if v is not None]
    if len(ok) < 2:
        return {"sizes": sizes, "values": vals, "monotonic": None}
    diffs = np.diff(ok)
    return {
        "sizes": sizes,
        "values": vals,
        "monotonic_down": bool(np.all(diffs <= 0)),
        "monotonic_up": bool(np.all(diffs >= 0)),
        "total_change": float(ok[-1] - ok[0]),
    }


DRIFT_ANCHOR_NOTE = """\
A size trend is only evidence of transfer if it exceeds the trend of a rule pair
whose expressive class is IDENTICAL at every size. Run the additive-fit argmin
and the additive+one-integer rule (separability_diagnostic.variance_decomposition,
no training required) on the same held-out datasets at every size and subtract
their trend. On shallow_v1 that constant-quality pair drifts by 2.58 in
regret_mean and reverses the sign of the opt_recovered gap between bins -- large
enough to satisfy the original "the gap widens monotonically" PASS condition on
its own."""


def drift_anchor_note() -> str:
    return DRIFT_ANCHOR_NOTE


def format_comparison_table(
    comparisons: Dict[str, dict], reference: str, order: Optional[Iterable[str]] = None
) -> str:
    names = list(order) if order is not None else list(comparisons)
    out = [
        f"--- PAIRED vs {reference} (primary gate statistics) ---",
        f"{'model':<12}{'n':>5}{'win_rate':>10}{'95% CI':>18}{'sign p':>9}"
        f"{'ratio_mean':>12}{'gap':>10}{'MDG':>9}",
    ]
    for n in names:
        c = comparisons.get(n)
        if not c or not c.get("n_paired"):
            continue
        lo, hi = c["win_rate_ci95"]
        out.append(
            f"{n:<12}{c['n_paired']:>5}{c['win_rate']:>10.3f}"
            f"{f'[{lo:.3f},{hi:.3f}]':>18}{c['sign_test_p']:>9.3f}"
            f"{c['regret_ratio_mean']:>12.4f}{c['regret_gap_mean']:>10.4f}"
            f"{c['min_detectable_gap']:>9.4f}"
        )
    return "\n".join(out)
