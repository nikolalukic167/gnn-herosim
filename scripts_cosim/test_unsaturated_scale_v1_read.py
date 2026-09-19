"""unsaturated_scale_v1: the registered bars behave as signed, before any arm is served.

Run: pipenv run python3 -m pytest scripts_cosim/test_unsaturated_scale_v1_read.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts_cosim.unsaturated_scale_v1_read import (  # noqa: E402
    K_ALPHA, K_ARMS, K_CELLS, K_FACTORS, K_GNN, K_GRAPH, K_MIN_SEEDS, K_SATURATION_SHARE,
    K_SEPARATE_PCT, K_TARGET_SHARE, K_TWIN,
    V_ARM_BEATS_REACTIVE, V_GRAPH_FASTER, V_LEARNED_BEATS_REACTIVE, V_NOT_SEP,
    V_NO_LEARNED_BEATS_REACTIVE, V_NO_UNSATURATED_RUNG, V_POINTWISE_FASTER,
    V_REACTIVE_FASTER, V_SUM_COSTS, V_SUM_HELPS, V_SUM_NOT_SEP,
    V_UNSATURATED_AT_THE_EDGE, V_UNSATURATED_RUNG,
    read_k1, read_k2, read_k3, read_k4, read_s1, saturated, select_factor,
)
from scripts_cosim.corpus_matched_v1_read import G_ALPHA, G_MIN_SEEDS, G_SEPARATE_PCT  # noqa: E402
from scripts_cosim.peer_only_v1_read import B7_SATURATED_QUEUE_SHARE, V_UNREADABLE  # noqa: E402

N = K_MIN_SEEDS


def _by(base, step=0.1):
    return {s: base + step * s for s in range(N)}


def _ladder(shares):
    """shares: per factor, one number applied to all four cells (or a dict per cell)."""
    out = {}
    for f, v in shares.items():
        out[f] = v if isinstance(v, dict) else {c: v for c in K_CELLS}
    return out


def test_the_bars_are_the_chains_bars_unchanged():
    assert (K_SEPARATE_PCT, K_ALPHA, K_MIN_SEEDS) == (G_SEPARATE_PCT, G_ALPHA, G_MIN_SEEDS)
    assert K_SATURATION_SHARE == B7_SATURATED_QUEUE_SHARE == 0.90
    assert K_TARGET_SHARE < K_SATURATION_SHARE            # inside the band, not at the edge
    assert 300 in K_FACTORS and K_FACTORS == tuple(sorted(K_FACTORS))
    assert len(K_CELLS) == 4 and "cs80s9004" not in K_CELLS  # 9004 hangs on every policy
    assert K_GRAPH in K_ARMS and K_TWIN in K_ARMS and K_GNN in K_ARMS


# --- S1 -----------------------------------------------------------------------------------

def test_s1_selects_the_smallest_factor_inside_the_band_not_the_closest():
    s1 = read_s1(_ladder({300: 0.99, 500: 0.95, 700: 0.88, 1000: 0.79, 2000: 0.70,
                          4000: 0.62, 8000: 0.55}))
    sel = select_factor(s1)
    assert sel["verdict"] == V_UNSATURATED_RUNG and sel["factor"] == 1000 and sel["primary"]
    assert s1["monotone"] is True
    # 700 is under the bar but outside the band, and must NOT be preferred over 1000
    assert s1["factors"][700]["saturated"] is False and not s1["factors"][700]["in_band"]


def test_s1_falls_to_the_edge_only_when_nothing_is_in_band():
    s1 = read_s1(_ladder({f: 0.87 for f in K_FACTORS}))
    sel = select_factor(s1)
    assert sel["verdict"] == V_UNSATURATED_AT_THE_EDGE and sel["factor"] == 300
    assert "every quote must say so" in sel["why"]


def test_s1_no_unsaturated_rung_is_a_finding_and_marks_s2_secondary():
    s1 = read_s1(_ladder({f: 0.99 - 0.001 * i for i, f in enumerate(K_FACTORS)}))
    sel = select_factor(s1)
    assert sel["verdict"] == V_NO_UNSATURATED_RUNG and sel["primary"] is False
    assert sel["factor"] == 8000            # the least-saturated, for the secondary read
    assert "baseline" in sel["why"]


def test_s1_a_factor_missing_a_cell_cannot_be_selected():
    ladder = _ladder({300: 0.99, 1000: 0.70, 2000: 0.60})
    del ladder[1000]["cs80s9003"]
    s1 = read_s1(ladder)
    assert s1["factors"][1000]["verdict"] == V_UNREADABLE
    assert select_factor(s1)["factor"] == 2000
    assert select_factor(read_s1({}))["verdict"] == V_UNREADABLE


def test_unknown_saturation_is_not_a_pass():
    assert saturated(None) is None
    assert saturated(0.79) is False
    assert saturated(0.90) is True


# --- K1 / K2 / K3: orientation is fixed -------------------------------------------------

def test_k1_is_oriented_so_a_faster_arm_reads_arm_beats_reactive():
    assert read_k1(_by(80.0), _by(100.0))["verdict"] == V_ARM_BEATS_REACTIVE
    assert read_k1(_by(120.0), _by(100.0))["verdict"] == V_REACTIVE_FASTER
    assert read_k1(_by(101.0), _by(100.0))["verdict"] == V_NOT_SEP
    a, b = _by(80.0), _by(100.0)
    assert read_k1(b, a)["verdict"] == V_REACTIVE_FASTER      # swapping must FLIP it


def test_k2_and_k3_orientation():
    assert read_k2(_by(80.0), _by(100.0))["verdict"] == V_GRAPH_FASTER
    assert read_k2(_by(120.0), _by(100.0))["verdict"] == V_POINTWISE_FASTER
    # gnn slower than gnnedge0 = POSITIVE median = sum costs
    assert read_k3(_by(120.0), _by(100.0))["verdict"] == V_SUM_COSTS
    assert read_k3(_by(80.0), _by(100.0))["verdict"] == V_SUM_HELPS
    assert read_k3(_by(101.0), _by(100.0))["verdict"] == V_SUM_NOT_SEP


def test_reads_refuse_a_short_ladder_rather_than_relaxing_the_bar():
    short_a, short_b = {s: 80.0 for s in range(5)}, {s: 100.0 for s in range(5)}
    assert read_k1(short_a, short_b)["verdict"] == V_UNREADABLE
    assert read_k2(short_a, short_b)["verdict"] == V_UNREADABLE


# --- K4 -----------------------------------------------------------------------------------

def _k1s(**verdicts):
    return {a: {"verdict": v} for a, v in verdicts.items()}


def test_k4_fires_on_any_single_registered_winner_and_names_it():
    r = read_k4({**{a: {"verdict": V_REACTIVE_FASTER} for a in K_ARMS},
                 K_GRAPH: {"verdict": V_ARM_BEATS_REACTIVE}})
    assert r["verdict"] == V_LEARNED_BEATS_REACTIVE and r["winners"] == [K_GRAPH]
    assert not r["secondary"]


def test_k4_an_unreadable_arm_is_missing_never_a_loss():
    k1 = {a: {"verdict": V_REACTIVE_FASTER} for a in K_ARMS}
    k1[K_GNN] = {"verdict": V_UNREADABLE}
    r = read_k4(k1)
    assert r["verdict"] == V_NO_LEARNED_BEATS_REACTIVE
    assert r["missing"] == [K_GNN] and K_GNN not in r["losers"]
    assert "not a verdict about them" in r["why"]
    assert read_k4({a: {"verdict": V_UNREADABLE} for a in K_ARMS})["verdict"] == V_UNREADABLE


def test_k4_a_not_separated_ladder_is_not_a_win_and_secondary_is_carried():
    r = read_k4({a: {"verdict": V_NOT_SEP} for a in K_ARMS}, primary=False)
    assert r["verdict"] == V_NO_LEARNED_BEATS_REACTIVE and r["secondary"] is True
    assert r["losers"] == []


def test_k4_ignores_an_arm_that_was_never_registered():
    k1 = {a: {"verdict": V_REACTIVE_FASTER} for a in K_ARMS}
    k1["posthoc_arm"] = {"verdict": V_ARM_BEATS_REACTIVE}
    assert read_k4(k1)["verdict"] == V_NO_LEARNED_BEATS_REACTIVE
