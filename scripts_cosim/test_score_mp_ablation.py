"""Tests for score_mp_ablation.py — the registration's arithmetic and its VOID gate."""
import json
import sys
from math import comb, isclose
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "important"))
import score_mp_ablation as S  # noqa: E402


def _prov(**over):
    p = dict(S.REQUIRED_PROVENANCE)
    p.update(over)
    return p


def _summary(mp_off_prov=None, on_rtt=90.0, off_rtt=95.0, kn=100.0, n_cells=30):
    """6 blocks x 5 cells = 30, knative plus both arms of all 16 seeds."""
    blocks = sorted(S.BACKBONE_BLOCKS | S.FLAT_BLOCKS)
    out = {}
    made = 0
    for blk in blocks:
        out[blk] = {}
        for c in range(5):
            if made >= n_cells:
                break
            cell = f"cell{c:02d}"
            entry = {"knative": {"total_rtt": kn}}
            for s in S.SEEDS:
                entry[S.MP_ON[s]] = {"total_rtt": on_rtt, "provenance": _prov()}
                entry[S.MP_OFF[s]] = {"total_rtt": off_rtt,
                                      "provenance": mp_off_prov or _prov()}
            out[blk][cell] = entry
            made += 1
    return out


# ---- VOID gate ---------------------------------------------------------------------
def test_clean_summary_passes_the_pin():
    assert S.assert_mp_off_provenance(_summary()) == []


def test_void_when_the_ablation_lever_is_not_set():
    """The whole screen is invalid if MP-OFF arms were served WITH message passing."""
    void = S.assert_mp_off_provenance(_summary(mp_off_prov=_prov(GNN_DISABLE_MESSAGE_PASSING=None)))
    assert len(void) == len(S.SEEDS)
    assert any("GNN_DISABLE_MESSAGE_PASSING" in why for _, why in void)


def test_void_on_wrong_commit():
    void = S.assert_mp_off_provenance(_summary(mp_off_prov=_prov(code_commit="deadbeef")))
    assert len(void) == len(S.SEEDS)


def test_void_on_dirty_tree():
    void = S.assert_mp_off_provenance(_summary(mp_off_prov=_prov(code_dirty=True)))
    assert len(void) == len(S.SEEDS)


def test_void_on_wrong_cell_count():
    void = S.assert_mp_off_provenance(_summary(n_cells=29))
    assert void and all("cells" in why for _, why in void)


def test_void_when_arms_absent():
    s = _summary()
    for blk in s:
        for cell in s[blk]:
            for seed in S.SEEDS:
                s[blk][cell].pop(S.MP_OFF[seed])
    void = S.assert_mp_off_provenance(s)
    assert len(void) == len(S.SEEDS)


# ---- Wilcoxon ----------------------------------------------------------------------
def test_wilcoxon_all_positive_is_the_minimum_possible_p():
    """Every pair in one direction -> p = 2/2^n (both extreme tails)."""
    d = [1.0, 2.0, 3.0, 4.0, 5.0]
    p, w, n = S.wilcoxon_exact_two_sided(d)
    assert n == 5 and isclose(w, 15.0)
    assert isclose(p, 2 / 2 ** 5, rel_tol=1e-12)


def test_wilcoxon_symmetric_differences_give_p_one():
    p, _, _ = S.wilcoxon_exact_two_sided([1.0, -1.0, 2.0, -2.0])
    assert isclose(p, 1.0, rel_tol=1e-12)


def test_wilcoxon_ignores_exact_zero_differences():
    p_a, _, n_a = S.wilcoxon_exact_two_sided([1.0, 2.0, 3.0])
    p_b, _, n_b = S.wilcoxon_exact_two_sided([1.0, 2.0, 3.0, 0.0, 0.0])
    assert n_a == n_b == 3 and isclose(p_a, p_b)


def test_wilcoxon_is_deterministic():
    d = [0.4, -1.2, 3.3, 0.1, -0.7, 2.2, 1.9, -0.3]
    assert S.wilcoxon_exact_two_sided(d) == S.wilcoxon_exact_two_sided(d)


# ---- Sign test ---------------------------------------------------------------------
def test_sign_test_matches_binomial_tail():
    assert isclose(S.sign_test_exact(13, 16),
                   sum(comb(16, i) for i in range(13, 17)) / 2 ** 16)


def test_sign_test_with_no_untied_pairs_is_p_one():
    """Ties are expected by construction; an all-tied co-primary must not claim anything."""
    assert S.sign_test_exact(0, 0) == 1.0


# ---- Reading rules -----------------------------------------------------------------
def test_no_difference_when_arms_are_identical(capsys, tmp_path):
    s = _summary(on_rtt=90.0, off_rtt=90.0)
    p = tmp_path / "s.json"
    p.write_text(json.dumps(s))
    sys.argv = ["score_mp_ablation.py", "--summary", str(p)]
    assert S.main() == 0
    assert "NO_DIFFERENCE_DETECTED" in capsys.readouterr().out


def test_void_short_circuits_before_any_statistic(capsys, tmp_path):
    s = _summary(mp_off_prov=_prov(GNN_DISABLE_MESSAGE_PASSING=None))
    p = tmp_path / "s.json"
    p.write_text(json.dumps(s))
    sys.argv = ["score_mp_ablation.py", "--summary", str(p)]
    assert S.main() == 2
    out = capsys.readouterr().out
    assert "VOID" in out and "PRIMARY" not in out


# ---- Regression: the co-primary's direction was inverted once. ----------------------
def test_co_primary_direction_is_not_inverted(capsys, tmp_path):
    """MP-OFF strictly BETTER on collapse must report better=N, worse=0 -- not the reverse.

    The first implementation zipped (voff, von) and tested `von > voff` while calling the
    result `worse`, so an arm that collapsed LESS was reported as collapsing MORE, and the
    one-sided p was computed against the wrong hypothesis.
    """
    s = _summary()
    blk = sorted(S.BACKBONE_BLOCKS)[0]
    cell = sorted(s[blk])[0]
    # Make MP-ON collapse in this cell (>= +50% vs knative=100) and MP-OFF not.
    s[blk][cell][S.MP_ON[1]]["total_rtt"] = 400.0
    s[blk][cell][S.MP_OFF[1]]["total_rtt"] = 90.0
    p = tmp_path / "s.json"
    p.write_text(json.dumps(s))
    sys.argv = ["score_mp_ablation.py", "--summary", str(p)]
    assert S.main() == 0
    out = capsys.readouterr().out
    assert "MP-OFF worse in 0, better in 1" in out, out
