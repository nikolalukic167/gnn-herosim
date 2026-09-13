"""Tests for score_link_mp_v1.py — the registration's arithmetic and its VOID gate."""
import sys
from math import isclose
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "important"))
import score_link_mp_v1 as S  # noqa: E402

FAKE_PIN = "a" * 40


@pytest.fixture(autouse=True)
def _pin(monkeypatch):
    """Tests run against an injected pin; the shipped None must refuse on its own."""
    monkeypatch.setattr(S, "PIN_COMMIT", FAKE_PIN)


def _prov(family, **over):
    p = dict(S.FAMILY_PROVENANCE[family])
    p["code_commit"] = FAKE_PIN
    p.update(over)
    return p


def _summary(rtt=None, prov_over=None, n_cells=20):
    """4 backbone blocks x 5 cells = 20; knative plus all 48 arms.

    `rtt` maps family -> per-arm total_rtt (default lgon 70, lgctrl 80, lgmpoff 75,
    i.e. lgon best). `prov_over` maps family -> provenance overrides.
    """
    rtt = {"lgon": 70.0, "lgctrl": 80.0, "lgmpoff": 75.0, **(rtt or {})}
    prov_over = prov_over or {}
    out, made = {}, 0
    for blk in S.BACKBONE_BLOCKS:
        out[blk] = {}
        for c in range(5):
            if made >= n_cells:
                break
            cell = f"cell{c:02d}"
            entry = {"knative": {"total_rtt": 100.0}}
            for fam, arms in (("lgon", S.LGON), ("lgctrl", S.LGCTRL),
                              ("lgmpoff", S.LGMPOFF)):
                for s in S.SEEDS:
                    entry[arms[s]] = {
                        "total_rtt": rtt[fam],
                        "provenance": _prov(fam, **prov_over.get(fam, {})),
                    }
            out[blk][cell] = entry
            made += 1
    return out


# ---- VOID gate ---------------------------------------------------------------------
def test_clean_summary_passes_the_pin():
    assert S.assert_provenance(_summary()) == []


def test_unpinned_registration_refuses_to_score(monkeypatch):
    """Before the amendment writes the arms' commit, scoring must refuse outright."""
    monkeypatch.setattr(S, "PIN_COMMIT", None)
    void = S.assert_provenance(_summary())
    assert void and "pin" in void[0][1]


def test_void_when_lgon_served_without_the_link_graph():
    """The whole point: an lgon arm served the OLD graph is not an lgon arm."""
    void = S.assert_provenance(
        _summary(prov_over={"lgon": {"NETWORK_GRAPH_CONTRACT": None}}))
    assert len(void) == len(S.SEEDS)
    assert all(arm.startswith("lgon") for arm, _ in void)
    assert any("NETWORK_GRAPH_CONTRACT" in why for _, why in void)


def test_void_when_a_control_arm_gets_the_link_graph():
    """lgctrl served core_v1 would smuggle the treatment into the control."""
    void = S.assert_provenance(
        _summary(prov_over={"lgctrl": {"NETWORK_GRAPH_CONTRACT": "core_v1"}}))
    assert len(void) == len(S.SEEDS)
    assert all(arm.startswith("lgctrl") for arm, _ in void)


def test_void_when_the_mp_off_lever_is_not_set():
    void = S.assert_provenance(
        _summary(prov_over={"lgmpoff": {"GNN_DISABLE_MESSAGE_PASSING": None}}))
    assert len(void) == len(S.SEEDS)
    assert all(arm.startswith("lgmpoff") for arm, _ in void)


def test_void_when_mp_off_leaks_into_lgon():
    void = S.assert_provenance(
        _summary(prov_over={"lgon": {"GNN_DISABLE_MESSAGE_PASSING": "1"}}))
    assert len(void) == len(S.SEEDS)


def test_void_on_wrong_commit_and_dirty_tree():
    assert len(S.assert_provenance(
        _summary(prov_over={"lgon": {"code_commit": "deadbeef"}}))) == len(S.SEEDS)
    assert len(S.assert_provenance(
        _summary(prov_over={"lgctrl": {"code_dirty": True}}))) == len(S.SEEDS)


def test_void_on_wrong_cell_count():
    void = S.assert_provenance(_summary(n_cells=19))
    assert void and all("cells" in why for _, why in void)


# ---- One-sided Wilcoxon -------------------------------------------------------------
def test_one_sided_all_positive_is_the_minimum_possible_p():
    d = [1.0, 2.0, 3.0, 4.0, 5.0]
    p, w, n = S.wilcoxon_exact_one_sided_greater(d)
    assert n == 5 and isclose(w, 15.0)
    assert isclose(p, 1 / 2 ** 5, rel_tol=1e-12)


def test_one_sided_all_negative_is_p_one():
    p, _, _ = S.wilcoxon_exact_one_sided_greater([-1.0, -2.0, -3.0])
    assert isclose(p, 1.0, rel_tol=1e-12)


def test_one_sided_tails_cover_the_distribution():
    """P(W >= w) + P(W <= w) = 1 + P(W == w): both tails computed from the same null."""
    d = [1.5, -0.5, 2.0, 3.0, -2.5, 0.75]
    p_h1, w, n = S.wilcoxon_exact_one_sided_greater(d)
    p_opp, _, _ = S.wilcoxon_exact_one_sided_greater([-x for x in d])
    assert p_h1 + p_opp >= 1.0  # equality iff P(W == w) = 0; ties make it strict
    assert 0.0 < p_h1 < 1.0 and 0.0 < p_opp <= 1.0


def test_primary_direction_is_not_inverted():
    """lgon better (lower RTT) must yield a SMALL H1 p — the mp_ablation regression."""
    good = _summary(rtt={"lgon": 60.0, "lgmpoff": 90.0})
    p = S.paired_wilcoxon(good, S.LGON, S.LGMPOFF, "test")
    assert p["p_one_sided_h1"] == 1 / 2 ** 16
    assert p["mean_diff_pp"] > 0

    bad = _summary(rtt={"lgon": 90.0, "lgmpoff": 60.0})
    q = S.paired_wilcoxon(bad, S.LGON, S.LGMPOFF, "test")
    assert q["p_one_sided_h1"] == 1.0
    assert q["p_one_sided_opposite"] == 1 / 2 ** 16


# ---- Sign test ----------------------------------------------------------------------
def test_sign_test_matches_binomial():
    assert isclose(S.sign_test_exact(5, 5), 1 / 32)
    assert isclose(S.sign_test_exact(0, 0), 1.0)
    assert isclose(S.sign_test_exact(3, 4), (4 + 1) / 16)


# ---- Verdict mapping (end-to-end through main) --------------------------------------
def _run_main(summary, tmp_path, monkeypatch, capsys):
    import json
    p = tmp_path / "summary.json"
    p.write_text(json.dumps(summary))
    out = tmp_path / "verdict.json"
    monkeypatch.setattr(sys, "argv",
                        ["score", "--summary", str(p), "--json-out", str(out)])
    rc = S.main()
    capsys.readouterr()
    return rc, (json.loads(out.read_text()) if out.exists() else None)


def test_verdict_wins_attributed(tmp_path, monkeypatch, capsys):
    s = _summary(rtt={"lgon": 60.0, "lgctrl": 85.0, "lgmpoff": 80.0})
    rc, rep = _run_main(s, tmp_path, monkeypatch, capsys)
    assert rc == 0 and rep["verdict"] == "LINK_MP_WINS_ATTRIBUTED"


def test_verdict_wins_unattributed(tmp_path, monkeypatch, capsys):
    # lgon beats lgmpoff but exactly ties lgctrl -> attribution fails.
    s = _summary(rtt={"lgon": 60.0, "lgctrl": 60.0, "lgmpoff": 80.0})
    rc, rep = _run_main(s, tmp_path, monkeypatch, capsys)
    assert rc == 0 and rep["verdict"] == "LINK_MP_WINS_UNATTRIBUTED"


def test_verdict_opposite_direction(tmp_path, monkeypatch, capsys):
    s = _summary(rtt={"lgon": 95.0, "lgctrl": 70.0, "lgmpoff": 70.0})
    rc, rep = _run_main(s, tmp_path, monkeypatch, capsys)
    assert rc == 0 and rep["verdict"] == "OPPOSITE_DIRECTION"


def test_verdict_no_difference(tmp_path, monkeypatch, capsys):
    s = _summary(rtt={"lgon": 75.0, "lgctrl": 75.0, "lgmpoff": 75.0})
    rc, rep = _run_main(s, tmp_path, monkeypatch, capsys)
    assert rc == 0 and rep["verdict"] == "NO_DIFFERENCE_DETECTED"


def test_void_short_circuits_before_any_statistic(tmp_path, monkeypatch, capsys):
    s = _summary(prov_over={"lgon": {"NETWORK_GRAPH_CONTRACT": None}})
    rc, rep = _run_main(s, tmp_path, monkeypatch, capsys)
    assert rc == 2 and rep is None
