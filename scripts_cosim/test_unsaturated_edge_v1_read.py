"""unsaturated_edge_v1: the registered bars behave as signed, before any learned arm is served.

Run: pipenv run python3 -m pytest scripts_cosim/test_unsaturated_edge_v1_read.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts_cosim.unsaturated_edge_v1_read import (  # noqa: E402
    E_ALPHA, E_ARMS_BY_RUNG, E_BASELINES, E_DESIGN_SD_BAR, E_GNN, E_GRAPH, E_MIN_CHECKPOINTS,
    E_PRIMARY_RUNG, E_RANDOM, E_REACTIVE, E_RUNGS, E_SATURATION_SHARE, E_SEPARATE_PCT,
    E_STUDY_TOPOLOGIES, E_TARGET_SHARE, E_TOPOLOGY_CANDIDATES, E_TWIN, E_WINDOWS,
    V_ARM_BEATS_RANDOM, V_ARM_BEATS_REACTIVE, V_DESIGN_READY, V_DESIGN_SHORT, V_GRAPH_FASTER,
    V_HEADLINE_BELOW_BAR, V_HEADLINE_BOTH, V_HEADLINE_PRIMARY, V_NOT_SEP, V_POINTWISE_FASTER,
    V_POWER_DELIVERED, V_POWER_NOT_DELIVERED, V_RANDOM_NOT_SEP, V_REACTIVE_FASTER, V_SUM_COSTS,
    V_SUM_NOT_SEP, V_WINDOW_CONSISTENT, V_WINDOW_FLIPS,
    checkpoint_stats, environments, pair_checkpoint_stats, read_e1, read_e2, read_e3, read_e4,
    read_e5, read_e6, read_e7, read_m0, select_topologies,
)
from scripts_cosim.unsaturated_scale_v2_read import (  # noqa: E402
    M_ALPHA, M_DESIGN_SD_BAR, M_MIN_CHECKPOINTS, M_SATURATION_SHARE, M_SEPARATE_PCT,
    M_TARGET_SHARE, M_TOPOLOGY_CANDIDATES, M_WINDOWS,
)
from scripts_cosim.peer_only_v1_read import V_UNREADABLE  # noqa: E402

N = E_MIN_CHECKPOINTS
ENVS40 = environments(40, [9001, 9002, 9003, 9005])


def _stats(base, step=0.1):
    return {s: base + step * s for s in range(N)}


# --- the chain ----------------------------------------------------------------------------

def test_every_bar_is_v2s_so_the_two_rungs_are_one_measurement():
    assert (E_SEPARATE_PCT, E_ALPHA, E_MIN_CHECKPOINTS) == (M_SEPARATE_PCT, M_ALPHA, M_MIN_CHECKPOINTS)
    assert (E_TARGET_SHARE, E_SATURATION_SHARE, E_DESIGN_SD_BAR) == (M_TARGET_SHARE, M_SATURATION_SHARE, M_DESIGN_SD_BAR)
    assert E_TOPOLOGY_CANDIDATES == M_TOPOLOGY_CANDIDATES and E_WINDOWS == M_WINDOWS
    assert E_SEPARATE_PCT == 5.0 and E_ALPHA == 0.05 and E_MIN_CHECKPOINTS == 16


def test_the_design_is_sixteen_environments_per_rung_and_the_graph_arm_runs_on_both():
    assert E_RUNGS == (40, 80) and E_PRIMARY_RUNG == 40
    assert E_STUDY_TOPOLOGIES * len(E_WINDOWS) == 16
    assert 9004 not in E_TOPOLOGY_CANDIDATES          # hangs on every policy
    for nc in E_RUNGS:
        assert E_GRAPH in E_ARMS_BY_RUNG[nc]
    assert E_TWIN in E_ARMS_BY_RUNG[40] and E_GNN in E_ARMS_BY_RUNG[40]
    assert E_REACTIVE in E_BASELINES and E_RANDOM in E_BASELINES
    for nc in E_RUNGS:                                  # a baseline is never a learned arm
        assert not set(E_BASELINES) & set(E_ARMS_BY_RUNG[nc])


# --- M0 -----------------------------------------------------------------------------------

def _shares(default=0.70, overrides=None):
    out = {(nc, t, w): default for nc in E_RUNGS for t in E_TOPOLOGY_CANDIDATES for w in E_WINDOWS}
    out.update(overrides or {})
    return out


def test_m0_is_per_rung_and_unknown_is_not_a_pass():
    m0 = read_m0(_shares(overrides={(40, 9001, "w0"): 0.85, (80, 9001, "w0"): None}))
    rows = m0["environments"]
    assert rows[(40, 9001, "w0")]["admissible"] is False and rows[(40, 9001, "w0")]["saturated"] is False
    assert rows[(80, 9001, "w0")]["admissible"] is False and rows[(80, 9001, "w0")]["saturated"] is None
    assert rows[(80, 9001, "w1")]["admissible"] is True
    assert m0["n_screened"] == 2 * 12 * 4 and m0["n_admissible"] == 96 - 2


def test_selection_is_lowest_seed_ids_admissible_on_all_four_windows_per_rung():
    # 9001 fails one window at C40 only; 9002 fails at C80 only.
    m0 = read_m0(_shares(overrides={(40, 9001, "w2"): 0.95, (80, 9002, "w0"): 0.81}))
    s40, s80 = select_topologies(m0, 40), select_topologies(m0, 80)
    assert s40["verdict"] == V_DESIGN_READY and s40["topologies"] == [9002, 9003, 9005, 9101]
    assert s80["verdict"] == V_DESIGN_READY and s80["topologies"] == [9001, 9003, 9005, 9101]
    assert len(s40["environments"]) == 16 and s40["environments"][0] == (40, 9002, "w0")


def test_selection_refuses_a_ragged_design():
    over = {(40, t, "w0"): 0.9 for t in E_TOPOLOGY_CANDIDATES[:9]}
    s = select_topologies(read_m0(_shares(overrides=over)), 40)
    assert s["verdict"] == V_DESIGN_SHORT and len(s["qualified"]) == 3
    assert select_topologies(read_m0(_shares(overrides=over)), 80)["verdict"] == V_DESIGN_READY


# --- the statistic ------------------------------------------------------------------------

def _arm(envs, seeds=range(1, N + 1), factor=0.8):
    return {(e, s): 20.0 * factor * (1 + 0.01 * s) for e in envs for s in seeds}


def test_checkpoint_stats_pairs_on_the_environment_and_fails_loud_when_ragged():
    react = {e: 20.0 * (1 + 0.5 * i) for i, e in enumerate(ENVS40)}
    arm = {(e, s): react[e] * 0.8 for e in ENVS40 for s in range(1, N + 1)}
    st = checkpoint_stats(arm, react, ENVS40)
    assert len(st) == N and all(abs(v + 20.0) < 1e-9 for v in st.values())
    del arm[(ENVS40[3], 5)]
    with pytest.raises(ValueError, match="missing environments"):
        checkpoint_stats(arm, react, ENVS40)


def test_e1_e2_e3_e4_fire_in_the_registered_directions():
    assert read_e1(_stats(-8.0))["verdict"] == V_ARM_BEATS_REACTIVE
    assert read_e1(_stats(+8.0))["verdict"] == V_REACTIVE_FASTER
    assert read_e1(_stats(-3.0))["verdict"] == V_NOT_SEP           # real but under the bar
    assert read_e2(_stats(-8.0))["verdict"] == V_GRAPH_FASTER
    assert read_e2(_stats(+8.0))["verdict"] == V_POINTWISE_FASTER
    assert read_e3(_stats(+8.0))["verdict"] == V_SUM_COSTS
    assert read_e3(_stats(+1.0))["verdict"] == V_SUM_NOT_SEP
    assert read_e4(_stats(-8.0))["verdict"] == V_ARM_BEATS_RANDOM
    assert read_e4(_stats(-1.0))["verdict"] == V_RANDOM_NOT_SEP


def test_a_significant_effect_under_five_percent_is_not_a_win():
    r = read_e1(_stats(-1.6, step=0.01))
    assert r["p"] < 0.05 and r["verdict"] == V_NOT_SEP and r["ahead"] == N


def test_fewer_than_sixteen_checkpoints_is_unreadable():
    r = read_e1({s: -20.0 for s in range(15)})
    assert r["verdict"] == V_UNREADABLE and r["n"] == 15


def test_pair_stats_pairs_on_both_environment_and_checkpoint():
    a = {(e, s): 10.0 * (1 + s) for e in ENVS40 for s in range(1, N + 1)}
    b = {(e, s): 12.5 * (1 + s) for e in ENVS40 for s in range(1, N + 1)}
    st = pair_checkpoint_stats(a, b, ENVS40)
    assert all(abs(v + 20.0) < 1e-9 for v in st.values())


# --- E5 / E6 / E7 -------------------------------------------------------------------------

def test_e5_reads_the_worst_registered_arm_at_that_rung():
    good = {a: _stats(-5.0, step=0.5) for a in E_ARMS_BY_RUNG[40]}
    assert read_e5(good, 40)["verdict"] == V_POWER_DELIVERED
    bad = dict(good); bad[E_GNN] = {s: (-40.0 if s % 2 else 40.0) for s in range(N)}
    r = read_e5(bad, 40)
    assert r["verdict"] == V_POWER_NOT_DELIVERED and r["worst_sd"] > E_DESIGN_SD_BAR
    assert read_e5({E_GRAPH: _stats(-5.0)}, 80)["verdict"] == V_POWER_DELIVERED


def test_e6_names_the_arms_that_flip_sign_between_windows():
    per = {a: {"w0": -20.0, "w1": -2.0, "w2": -1.0, "w3": -0.5} for a in E_ARMS_BY_RUNG[40]}
    per[E_TWIN] = {"w0": -3.0, "w1": +1.0, "w2": +2.0, "w3": +1.5}
    r = read_e6(per, 40)
    assert r["flips"] == [E_TWIN]
    assert r["arms"][E_GRAPH]["verdict"] == V_WINDOW_CONSISTENT
    assert r["arms"][E_GRAPH]["spread_pp"] == pytest.approx(19.5)
    assert read_e6({}, 80)["arms"][E_GRAPH]["verdict"] == V_UNREADABLE


def _e1(verdict):
    return {"verdict": verdict, "n": N, "median": -8.0 if verdict == V_ARM_BEATS_REACTIVE else -2.0, "p": 0.01}


def test_e7_composes_the_graph_arm_across_the_two_rungs():
    ok = {40: {"verdict": V_POWER_DELIVERED}, 80: {"verdict": V_POWER_DELIVERED}}
    both = {40: {E_GRAPH: _e1(V_ARM_BEATS_REACTIVE)}, 80: {E_GRAPH: _e1(V_ARM_BEATS_REACTIVE)}}
    assert read_e7(both, ok)["verdict"] == V_HEADLINE_BOTH
    prim = {40: {E_GRAPH: _e1(V_ARM_BEATS_REACTIVE)}, 80: {E_GRAPH: _e1(V_NOT_SEP)}}
    assert read_e7(prim, ok)["verdict"] == V_HEADLINE_PRIMARY
    # a win at 80 alone is NOT the headline holding: the headline lives at 40
    sec = {40: {E_GRAPH: _e1(V_NOT_SEP)}, 80: {E_GRAPH: _e1(V_ARM_BEATS_REACTIVE)}}
    assert read_e7(sec, ok)["verdict"] == V_HEADLINE_BELOW_BAR


def test_e7_refuses_to_call_an_underpowered_negative_a_tie():
    neg = {40: {E_GRAPH: _e1(V_NOT_SEP)}, 80: {E_GRAPH: _e1(V_NOT_SEP)}}
    r = read_e7(neg, {40: {"verdict": V_POWER_NOT_DELIVERED}, 80: {"verdict": V_POWER_DELIVERED}})
    assert r["verdict"] == V_HEADLINE_BELOW_BAR and r["interpretable"] is False
    r = read_e7(neg, {40: {"verdict": V_POWER_DELIVERED}, 80: {"verdict": V_POWER_DELIVERED}})
    assert r["interpretable"] is True
    assert read_e7({40: {}, 80: {}}, {})["verdict"] == V_UNREADABLE
