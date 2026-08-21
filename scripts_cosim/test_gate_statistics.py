"""Contract tests for the Phase 4 gate statistics.

The point of these is that a gate which lies is worse than no gate. Each test
pins one property the topology_transfer_v1 gate depends on.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gate_statistics import (  # noqa: E402
    PHASE4_TIERS,
    VERDICT_ESCALATE,
    VERDICT_EXHAUSTED,
    VERDICT_FAIL,
    VERDICT_PASS,
    _binom_two_sided_p,
    co_primary_sign_agree,
    escalation_note,
    format_comparison_table,
    paired_regret_comparison,
    phase4_verdict,
    pool_seed_comparisons,
    pooled_phase4_verdict,
    power_note,
    required_n_for_effect,
    size_trend,
)


def _pd(values):
    return {f"ds_{i:05d}": v for i, v in enumerate(values)}


# ---------------------------------------------------------------- win_rate basics
def test_win_rate_counts_wins_losses_ties():
    model = _pd([0.0, 0.5, 0.2])
    ref = _pd([0.1, 0.2, 0.2])
    c = paired_regret_comparison(model, ref, n_boot=200)
    assert (c["wins"], c["losses"], c["ties"]) == (1, 1, 1)
    assert c["win_rate"] == pytest.approx(0.5)


def test_win_rate_is_one_when_model_always_better():
    c = paired_regret_comparison(_pd([0.0, 0.0, 0.0]), _pd([0.1, 0.2, 0.3]), n_boot=200)
    assert c["win_rate"] == 1.0
    assert c["losses"] == 0


def test_only_shared_datasets_are_compared():
    model = {"a": 0.0, "b": 0.1, "c": 0.9}
    ref = {"a": 0.5, "b": 0.5}
    c = paired_regret_comparison(model, ref, n_boot=200)
    assert c["n_paired"] == 2  # 'c' has no reference decision and is dropped


def test_no_shared_datasets_reports_zero_not_crash():
    assert paired_regret_comparison({"a": 0.1}, {"b": 0.2})["n_paired"] == 0


# ------------------------------------------- the property the gate actually needs
def test_win_rate_is_invariant_to_per_dataset_regret_rescaling():
    """THE size-comparability property.

    Larger topologies have wider sweeps, so per-dataset regret scales change.
    win_rate must not move when each dataset's regrets are rescaled, because
    that rescaling is landscape, not model quality. regret_gap_mean does move,
    which is exactly why it was demoted from the gate.
    """
    rng = np.random.default_rng(0)
    m = rng.random(60) * 0.1
    r = rng.random(60) * 0.1
    scale = np.exp(rng.normal(0, 2.0, 60))  # wildly heterogeneous landscapes
    base = paired_regret_comparison(_pd(m), _pd(r), n_boot=200)
    scaled = paired_regret_comparison(_pd(m * scale), _pd(r * scale), n_boot=200)
    assert scaled["win_rate"] == pytest.approx(base["win_rate"])
    # and the demoted statistic is NOT invariant -- the reason for this module
    assert scaled["regret_gap_mean"] != pytest.approx(base["regret_gap_mean"])


def test_win_rate_chance_level_is_half_regardless_of_regret_scale():
    """Chance is 0.5 at every topology size -- the size-invariant null."""
    rng = np.random.default_rng(7)
    for scale in (1e-3, 1.0, 1e3):
        draws = [
            paired_regret_comparison(
                _pd(rng.random(200) * scale), _pd(rng.random(200) * scale), n_boot=50
            )["win_rate"]
            for _ in range(10)
        ]
        assert abs(float(np.mean(draws)) - 0.5) < 0.06


def test_ratio_mean_keeps_magnitude_that_win_rate_discards():
    """win_rate alone cannot see a blowup, which is why ratio_mean is co-primary."""
    narrow = paired_regret_comparison(_pd([0.09, 0.09, 0.09]), _pd([0.1, 0.1, 0.1]), n_boot=100)
    wide = paired_regret_comparison(_pd([0.001, 0.001, 0.001]), _pd([0.1, 0.1, 0.1]), n_boot=100)
    assert narrow["win_rate"] == wide["win_rate"] == 1.0
    assert wide["regret_ratio_mean"] < narrow["regret_ratio_mean"]


# ---------------------------------------------------------------------- sign test
def test_sign_test_ignores_ties_and_is_symmetric():
    c = paired_regret_comparison(_pd([0.0, 0.0, 0.2]), _pd([0.1, 0.1, 0.2]), n_boot=100)
    assert c["ties"] == 1
    assert c["sign_test_p"] == pytest.approx(_binom_two_sided_p(2, 2))


def test_binom_p_matches_hand_computed_values():
    assert _binom_two_sided_p(5, 10) == pytest.approx(1.0)
    assert _binom_two_sided_p(10, 10) == pytest.approx(2 * 0.5 ** 10)
    assert _binom_two_sided_p(0, 10) == pytest.approx(2 * 0.5 ** 10)


# --------------------------------------------------------------------- power note
def test_power_note_fires_when_gap_is_below_the_noise_floor():
    rng = np.random.default_rng(1)
    # identical quality plus heavy tails: a tiny gap on a noisy corpus
    m = rng.random(30) * 0.05
    r = m + rng.normal(0, 0.5, 30)
    c = paired_regret_comparison(_pd(m), _pd(np.abs(r)), n_boot=500)
    note = power_note(c, observed_gap=0.003)
    assert note.startswith("!! UNDERPOWERED")
    assert "win_rate" in note


def test_power_note_reports_resolved_for_a_large_gap():
    c = paired_regret_comparison(_pd([0.0] * 40), _pd([1.0] * 40), n_boot=500)
    assert power_note(c).startswith("resolved")


def test_min_detectable_gap_shrinks_with_more_datasets():
    rng = np.random.default_rng(3)
    mdg = []
    for n in (30, 240):
        m, r = rng.random(n) * 0.1, rng.random(n) * 0.1
        mdg.append(paired_regret_comparison(_pd(m), _pd(r), n_boot=800)["min_detectable_gap"])
    assert mdg[1] < mdg[0] / 2  # ~1/sqrt(n)


# -------------------------------------------------------------------- size trend
def test_size_trend_detects_monotone_decline():
    per_size = {
        20: {"win_rate": 0.8},
        40: {"win_rate": 0.7},
        80: {"win_rate": 0.55},
    }
    t = size_trend(per_size)
    assert t["sizes"] == [20, 40, 80]
    assert t["monotonic_down"] is True and t["monotonic_up"] is False
    assert t["total_change"] == pytest.approx(-0.25)


def test_size_trend_needs_two_points():
    assert size_trend({20: {"win_rate": 0.8}})["monotonic"] is None


# ------------------------------------------------------- pre-registered escalation
def _cmp_with_ci(win_rate, lo, hi, n=360):
    """A comparison dict with a hand-set CI, so verdict logic is tested directly."""
    return {"n_paired": n, "win_rate": win_rate, "win_rate_ci95": [lo, hi],
            "regret_gap_mean": 0.0, "min_detectable_gap": 0.02}


def test_verdict_passes_only_when_ci_excludes_half_from_above():
    v = phase4_verdict(_cmp_with_ci(0.58, 0.52, 0.64))
    assert v["verdict"] == VERDICT_PASS


def test_verdict_fails_only_when_the_reference_wins_significantly():
    v = phase4_verdict(_cmp_with_ci(0.42, 0.36, 0.48))
    assert v["verdict"] == VERDICT_FAIL
    assert "reference wins" in v["reason"]


def test_straddling_ci_escalates_and_is_never_reported_as_a_failure():
    """The whole point of the rule: under-powered != null.

    These are the real shallow_v1 seed-42 numbers at n=30. An effect of 0.033
    against a half-width of 0.083 needs n>=192, so the escalation target is the
    first tier that clears it -- tier 0.02, i.e. "go generate the corpus".
    """
    v = phase4_verdict(_cmp_with_ci(0.533, 0.450, 0.617, n=30))
    assert v["verdict"] == VERDICT_ESCALATE
    assert v["escalate_to"] == "tier_0.02"
    note = escalation_note(v)
    assert "NOT a null result" in note
    assert "do not report as a failure to transfer" in note


def test_escalation_target_skips_tiers_too_small_for_the_effect():
    """Same win_rate measured at tier 0.02 escalates to 0.01, not back to 0.02.

    This CI's implied half-width needs ~1534/size, which tier_launch (900) does
    not clear, so this still lands on tier_0.01 -- see the next test for a
    scenario tier_launch DOES resolve.
    """
    v = phase4_verdict(_cmp_with_ci(0.533, 0.5 - 0.0516, 0.5 + 0.0516 + 0.033, n=360))
    assert v["verdict"] == VERDICT_ESCALATE
    assert v["escalate_to"] == "tier_0.01"


def test_escalation_target_lands_on_the_launch_tier_when_it_resolves_the_effect():
    """tier_launch (900/size) exists to catch effects tier_0.02 can't but don't need 0.01's cost."""
    # n=360 => half-width ~0.0516; effect 0.033 => needs ~880, inside tier_launch (900)
    v = phase4_verdict(_cmp_with_ci(0.533, 0.5 - 0.0516, 0.5 + 0.0516, n=360))
    assert v["verdict"] == VERDICT_ESCALATE
    assert v["escalate_to"] == "tier_launch"


def test_escalation_from_the_top_tier_is_exhausted_not_failed():
    v = phase4_verdict(_cmp_with_ci(0.505, 0.44, 0.57), tier_name="tier_0.01")
    assert v["verdict"] == VERDICT_EXHAUSTED
    assert "NOT a failure to transfer" in escalation_note(v)


def test_required_n_scales_as_inverse_square_of_the_effect():
    half = _cmp_with_ci(0.55, 0.50 - 0.05, 0.50 + 0.05 + 0.05)  # half-width 0.05
    n_big = required_n_for_effect(half, target_effect=0.05)
    n_small = required_n_for_effect(half, target_effect=0.025)
    assert n_small == pytest.approx(4 * n_big, rel=0.02)


def test_required_n_is_none_for_a_zero_effect():
    assert required_n_for_effect(_cmp_with_ci(0.5, 0.45, 0.55)) is None
    # and that case must still not read as a failure
    assert phase4_verdict(_cmp_with_ci(0.5, 0.45, 0.55))["verdict"] == VERDICT_EXHAUSTED


def test_tier_ladder_is_ordered_and_matches_the_registered_targets():
    names = [t["name"] for t in PHASE4_TIERS]
    assert names == ["tier_0.02", "tier_launch", "tier_0.01"]
    ns = [t["datasets_per_size"] for t in PHASE4_TIERS]
    assert ns == sorted(ns)  # a tier only ever buys MORE power
    assert ns == [360, 900, 1600]


def test_tier_0_02_cannot_resolve_the_effect_it_will_be_run_against():
    """Documents the arithmetic that forces the escalation rule to exist.

    Observed shallow_v1 effects are 0.017-0.033 from 0.5. Run at tier 0.02
    (360 datasets/held-out size) neither is resolved -- every CI still contains
    0.5 -- and the two ends of that range escalate DIFFERENTLY: 0.033 is within
    reach of tier 0.01, 0.017 is not within reach of any registered tier. Both
    outcomes are non-failures; only their remedies differ.
    """
    outcomes = {}
    for wr in (0.517, 0.533):
        wins = int(round(360 * wr))
        c = paired_regret_comparison(
            _pd([0.0] * wins + [1.0] * (360 - wins)),
            _pd([1.0] * wins + [0.0] * (360 - wins)),
            n_boot=400,
        )
        lo, hi = c["win_rate_ci95"]
        assert lo < 0.5 < hi  # tier 0.02 cannot resolve either effect
        v = phase4_verdict(c)
        assert v["verdict"] in (VERDICT_ESCALATE, VERDICT_EXHAUSTED)
        assert "NOT a" in escalation_note(v)  # explicitly disclaims failure
        outcomes[wr] = v["verdict"]
    assert outcomes[0.533] == VERDICT_ESCALATE
    assert outcomes[0.517] == VERDICT_EXHAUSTED  # ~3,400/size, off the ladder


def test_no_paired_datasets_escalates_rather_than_failing():
    assert phase4_verdict({"n_paired": 0})["verdict"] == VERDICT_ESCALATE


# --------------------------------------------------- multi-seed pooling / sign check
def _seed_cmp(win_rate, ratio, n=30):
    return {"n_paired": n, "win_rate": win_rate, "regret_ratio_mean": ratio}


def test_pool_seed_comparisons_uses_seed_to_seed_sd_for_the_ci():
    """Reproduces the frozen 5-seed shallow_v1 numbers (LINEAGES topology_transfer_v1)."""
    seeds = [
        _seed_cmp(0.500, 0.987), _seed_cmp(0.450, 1.006), _seed_cmp(0.533, 1.999),
        _seed_cmp(0.550, 0.991), _seed_cmp(0.583, 0.975),
    ]
    pooled = pool_seed_comparisons(seeds)
    assert pooled["n_seeds"] == 5
    assert pooled["win_rate"] == pytest.approx(0.5232, abs=1e-3)
    lo, hi = pooled["win_rate_ci95"]
    assert lo == pytest.approx(0.479, abs=5e-3)
    assert hi == pytest.approx(0.568, abs=5e-3)


def test_pool_seed_comparisons_ratio_median_resists_one_blowup_seed():
    """The whole point of pooling by median: seed 44's 1.999 must not move it."""
    calm = [_seed_cmp(0.55, 0.99) for _ in range(4)]
    blowup = [_seed_cmp(0.55, 2.0)]
    pooled = pool_seed_comparisons(calm + blowup)
    assert pooled["regret_ratio_median"] == pytest.approx(0.99)
    # a mean, by contrast, WOULD move a lot -- this is the property being fixed
    naive_mean = np.mean([0.99, 0.99, 0.99, 0.99, 2.0])
    assert pooled["regret_ratio_median"] < naive_mean - 0.1


def test_pool_seed_comparisons_ignores_seeds_with_no_paired_datasets():
    pooled = pool_seed_comparisons([_seed_cmp(0.6, 0.9), {"n_paired": 0}])
    assert pooled["n_seeds"] == 1


def test_pool_seed_comparisons_empty_reports_zero_not_crash():
    assert pool_seed_comparisons([])["n_seeds"] == 0


def test_co_primary_sign_agree_true_when_both_favor_the_model():
    pooled = pool_seed_comparisons([_seed_cmp(0.6, 0.9), _seed_cmp(0.58, 0.85)])
    assert co_primary_sign_agree(pooled) is True


def test_co_primary_sign_agree_false_when_they_point_opposite_ways():
    pooled = pool_seed_comparisons([_seed_cmp(0.6, 1.2), _seed_cmp(0.58, 1.3)])
    assert co_primary_sign_agree(pooled) is False


def test_co_primary_sign_agree_none_at_the_null():
    pooled = pool_seed_comparisons([_seed_cmp(0.5, 1.5), _seed_cmp(0.5, 0.7)])
    assert co_primary_sign_agree(pooled) is None
    pooled2 = pool_seed_comparisons([_seed_cmp(0.6, 1.0), _seed_cmp(0.58, 1.0)])
    assert co_primary_sign_agree(pooled2) is None


def test_pooled_verdict_downgrades_pass_to_fail_on_sign_disagreement():
    """A decisive win_rate PASS with a decisive opposite-sign ratio is not a PASS."""
    seeds = [_seed_cmp(0.75, 1.4, n=900) for _ in range(5)]
    v = pooled_phase4_verdict(seeds, tier_name="tier_launch")
    assert v["verdict"] == VERDICT_FAIL
    assert v["co_primary_sign_agree"] is False
    assert "disagrees in sign" in v["reason"]


def test_pooled_verdict_one_blowup_seed_does_not_veto_an_otherwise_clean_pass():
    """The exact bug being fixed: seed 44's ratio must not flip this to FAIL."""
    seeds = [_seed_cmp(0.75, 0.98, n=900) for _ in range(4)] + [_seed_cmp(0.75, 2.0, n=900)]
    v = pooled_phase4_verdict(seeds, tier_name="tier_launch")
    assert v["verdict"] == VERDICT_PASS
    assert v["co_primary_sign_agree"] is True


def test_pooled_verdict_straddling_ci_still_escalates_regardless_of_ratio():
    seeds = [_seed_cmp(0.50, 0.99), _seed_cmp(0.45, 1.01), _seed_cmp(0.53, 2.0),
              _seed_cmp(0.55, 0.99), _seed_cmp(0.58, 0.98)]
    v = pooled_phase4_verdict(seeds)
    assert v["verdict"] in (VERDICT_ESCALATE, VERDICT_EXHAUSTED)


def test_pooled_verdict_no_seeds_escalates_rather_than_failing():
    assert pooled_phase4_verdict([])["verdict"] == VERDICT_ESCALATE


def test_format_table_skips_models_with_no_paired_datasets():
    table = format_comparison_table(
        {"gnn_base": paired_regret_comparison(_pd([0.0]), _pd([0.1]), n_boot=50),
         "gnn_node": {"n_paired": 0}},
        "pointwise",
        ["gnn_base", "gnn_node"],
    )
    assert "gnn_base" in table and "gnn_node" not in table
