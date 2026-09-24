"""pointwise_baseline_v1: the registered bars behave as signed, before any arm is served.

This lineage can invalidate the framing of every earlier graph-vs-pointwise result in the
record, so the reader is exercised on every pattern that could do so — and on the ones that
must NOT, in particular a partial ladder and a non-separation being mistaken for equality.

Run: pipenv run python3 -m pytest scripts_cosim/test_pointwise_baseline_v1_read.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts_cosim.pointwise_baseline_v1_read import (  # noqa: E402
    J_ALPHA, J_CLIENTS, J_CORPORA, J_MIN_SEEDS, J_SEPARATE_PCT,
    V_GRAPH_BEATS_MLP, V_GRAPH_FASTER, V_GRAPH_VS_MLP_NOT_ESTABLISHED,
    V_MLP_BEATS_GRAPH, V_MLP_FASTER, V_NOT_SEP,
    V_TWIN_IS_FAIR, V_TWIN_IS_NOT_THE_MLP,
    read_j1, read_j2, read_j3, read_j3_composite, saturated,
)
from scripts_cosim.best_arm_v1_read import F_ALPHA, F_MIN_SEEDS, F_SEPARATE_PCT  # noqa: E402
from scripts_cosim.peer_only_v1_read import V_UNREADABLE  # noqa: E402

N = J_MIN_SEEDS


def _by(base, step=0.1):
    return {s: base + step * s for s in range(N)}


def test_the_bars_are_the_chains_bars_unchanged():
    """J must be readable against F and G rung by rung, which it is not if the bar moved."""
    assert (J_SEPARATE_PCT, J_ALPHA, J_MIN_SEEDS) == (F_SEPARATE_PCT, F_ALPHA, F_MIN_SEEDS)
    assert (J_SEPARATE_PCT, J_ALPHA, J_MIN_SEEDS) == (5.0, 0.05, 16)
    assert J_CLIENTS == (20, 40, 80)
    assert J_CORPORA == ("1670", "516")


# --- J1 -----------------------------------------------------------------------------------

def test_j1_is_oriented_so_a_faster_graph_arm_reads_graph_faster():
    assert read_j1(_by(80.0), _by(100.0))["verdict"] == V_GRAPH_FASTER
    assert read_j1(_by(120.0), _by(100.0))["verdict"] == V_MLP_FASTER
    assert read_j1(_by(101.0), _by(100.0))["verdict"] == V_NOT_SEP
    a, b = _by(80.0), _by(100.0)
    assert read_j1(b, a)["verdict"] == V_MLP_FASTER          # swapping must FLIP it


def test_j1_refuses_a_short_read_rather_than_relaxing_the_bar():
    short_a, short_b = {s: 80.0 for s in range(5)}, {s: 100.0 for s in range(5)}
    assert read_j1(short_a, short_b)["verdict"] == V_UNREADABLE
    long_a, long_b = {s: 80.0 for s in range(10)}, {s: 100.0 for s in range(10)}
    assert read_j1(long_a, long_b, min_seeds=10)["verdict"] == V_GRAPH_FASTER


# --- J2 -----------------------------------------------------------------------------------

def _rungs(*verdicts):
    return {c: {"verdict": v, "median": -1.0 * (i + 1)}
            for i, (c, v) in enumerate(zip(J_CLIENTS, verdicts))}


def test_j2_uses_the_chains_rule_unchanged_two_wins_and_no_losses():
    assert read_j2(_rungs(V_NOT_SEP, V_GRAPH_FASTER, V_GRAPH_FASTER))["verdict"] \
        == V_GRAPH_BEATS_MLP
    # two wins AND a loss -- best_arm_v1's actual ladder shape -- must not qualify
    assert read_j2(_rungs(V_MLP_FASTER, V_GRAPH_FASTER, V_GRAPH_FASTER))["verdict"] \
        == V_GRAPH_VS_MLP_NOT_ESTABLISHED


def test_j2_reads_the_mlp_side_when_the_mlp_wins_twice():
    r = read_j2(_rungs(V_MLP_FASTER, V_MLP_FASTER, V_NOT_SEP))
    assert r["verdict"] == V_MLP_BEATS_GRAPH and r["mlp_wins"] == 2


def test_j2_does_not_call_a_split_ladder_a_failure():
    """The false sentence best_arm_v1's F2 had to be fixed for, pinned here in advance."""
    r = read_j2(_rungs(V_MLP_FASTER, V_NOT_SEP, V_GRAPH_FASTER))
    assert r["verdict"] == V_GRAPH_VS_MLP_NOT_ESTABLISHED
    assert "SPLIT" in r["why"] and "faster at no rung" not in r["why"]
    r0 = read_j2(_rungs(V_MLP_FASTER, V_NOT_SEP, V_NOT_SEP))
    assert "faster at no rung" in r0["why"]


def test_j2_will_not_decide_on_a_partial_ladder():
    rungs = _rungs(V_GRAPH_FASTER, V_GRAPH_FASTER, V_NOT_SEP)
    rungs[80] = {"verdict": V_UNREADABLE}
    r = read_j2(rungs)
    assert r["verdict"] == V_UNREADABLE and r["missing"] == [80]
    del rungs[20]
    assert read_j2(rungs)["verdict"] == V_UNREADABLE


# --- J3: the load-bearing read -------------------------------------------------------------

def test_j3_is_oriented_so_a_faster_twin_reads_twin_faster():
    assert read_j3(_by(80.0), _by(100.0))["verdict"] == "TWIN-FASTER"
    assert read_j3(_by(120.0), _by(100.0))["verdict"] == "MLP-FASTER"
    assert read_j3(_by(101.0), _by(100.0))["verdict"] == V_NOT_SEP


def test_j3_composite_invalidates_the_substitution_on_a_single_separated_rung():
    r = read_j3_composite({20: {"verdict": V_NOT_SEP}, 40: {"verdict": "MLP-FASTER"},
                           80: {"verdict": V_NOT_SEP}})
    assert r["verdict"] == V_TWIN_IS_NOT_THE_MLP and sorted(r["separated_at"]) == [40]
    assert "must be restated" in r["why"]


def test_j3_composite_never_calls_a_non_separation_equality():
    r = read_j3_composite({c: {"verdict": V_NOT_SEP} for c in J_CLIENTS})
    assert r["verdict"] == V_TWIN_IS_FAIR
    assert "not proof of" in r["why"] and "ON THESE CELLS" in r["why"]


def test_j3_composite_is_unreadable_with_nothing_readable_and_reports_gaps():
    assert read_j3_composite({})["verdict"] == V_UNREADABLE
    assert read_j3_composite({c: {"verdict": V_UNREADABLE} for c in J_CLIENTS})["verdict"] \
        == V_UNREADABLE
    r = read_j3_composite({20: {"verdict": V_NOT_SEP}})
    assert r["verdict"] == V_TWIN_IS_FAIR and r["unreadable"] == [40, 80]


def test_unknown_saturation_is_not_a_pass():
    assert saturated(None) is None
    assert saturated(0.73) is False
    assert saturated(0.95) is True
