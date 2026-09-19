"""batch_window_edge_v1: the registered bars behave as signed, before any arm at a new window.

Run: pipenv run python3 -m pytest scripts_cosim/test_batch_window_edge_v1_read.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts_cosim.batch_window_edge_v1_read import (  # noqa: E402
    V_GRAPH_FASTER, V_LONGER_FASTER, V_NOT_SEP, V_POINTWISE_FASTER, V_RANDOM_NOT_SEP,
    V_REACTIVE_FASTER, V_SHORTER_BEATS_RANDOM, V_SHORTER_BEATS_REACTIVE, V_SHORTER_FASTER,
    V_WAIT_DID_NOT_FALL, V_WAIT_FELL, V_WINDOW_NOT_THE_LEVER, V_WINDOW_SELECTED, W_ALPHA,
    W_BASE_WINDOW_S, W_GRAPH, W_MIN_CHECKPOINTS, W_RUNG, W_SEPARATE_PCT, W_TWIN, W_WINDOWS_S,
    read_w1, read_w2, read_w3, read_w4, read_w5, read_w6, select_window, split_environments,
)
from scripts_cosim.unsaturated_edge_v1_read import E_GRAPH, E_PRIMARY_RUNG, E_TWIN  # noqa: E402
from scripts_cosim.peer_only_v1_read import V_UNREADABLE  # noqa: E402

N = W_MIN_CHECKPOINTS


def _stats(base, step=0.1):
    return {s: base + step * s for s in range(N)}


def test_bars_are_the_chains_and_the_rung_is_the_headline_rung():
    assert (W_SEPARATE_PCT, W_ALPHA, W_MIN_CHECKPOINTS) == (5.0, 0.05, 16)
    assert W_RUNG == E_PRIMARY_RUNG == 40
    assert W_GRAPH == E_GRAPH and W_TWIN == E_TWIN
    assert W_WINDOWS_S == (2.0, 4.0, 8.0) and W_BASE_WINDOW_S == 16.0
    assert all(w < W_BASE_WINDOW_S for w in W_WINDOWS_S)     # shorter windows only


def test_split_is_lowest_seed_screens_and_cannot_be_steered():
    s = split_environments([9103, 9001, 9101, 9003])
    assert s["screen_topologies"] == [9001] and s["held_out_topologies"] == [9003, 9101, 9103]
    assert len(s["screen"]) == 4 and len(s["held_out"]) == 12
    assert s["screen"][0] == (40, 9001, "w0")
    with pytest.raises(ValueError, match="study topologies"):
        split_environments([9001, 9002])


def test_selection_picks_the_lowest_screen_median_and_16s_winning_closes_the_lineage():
    st = {16.0: _stats(-6.0), 8.0: _stats(-9.0), 4.0: _stats(-12.0), 2.0: _stats(-4.0)}
    r = select_window(st)
    assert r["verdict"] == V_WINDOW_SELECTED and r["window_s"] == 4.0
    st[16.0] = _stats(-20.0)
    assert select_window(st)["verdict"] == V_WINDOW_NOT_THE_LEVER


def test_selection_ignores_a_window_without_a_full_checkpoint_set():
    """A window that hung on one checkpoint is not the winner however good its 15 look."""
    st = {16.0: _stats(-6.0), 4.0: {s: -30.0 for s in range(15)}, 8.0: _stats(-7.0)}
    r = select_window(st)
    assert r["window_s"] == 8.0 and 4.0 not in r["medians"]
    assert select_window({4.0: _stats(-30.0)})["verdict"] == V_UNREADABLE


def test_w1_to_w4_fire_in_the_registered_directions():
    assert read_w1(_stats(-8.0))["verdict"] == V_SHORTER_BEATS_REACTIVE
    assert read_w1(_stats(+8.0))["verdict"] == V_REACTIVE_FASTER
    assert read_w1(_stats(-3.0))["verdict"] == V_NOT_SEP
    assert read_w2(_stats(-8.0))["verdict"] == V_SHORTER_FASTER
    assert read_w2(_stats(+8.0))["verdict"] == V_LONGER_FASTER
    assert read_w3(_stats(-8.0))["verdict"] == V_GRAPH_FASTER
    assert read_w3(_stats(+8.0))["verdict"] == V_POINTWISE_FASTER
    assert read_w4(_stats(-8.0))["verdict"] == V_SHORTER_BEATS_RANDOM
    assert read_w4(_stats(-1.0))["verdict"] == V_RANDOM_NOT_SEP
    assert read_w1({s: -20.0 for s in range(15)})["verdict"] == V_UNREADABLE


def test_w5_and_w6():
    r = read_w5({"w0": -20.0, "w1": -6.0, "w2": -5.0, "w3": -4.0})
    assert r["verdict"] == "SIGN-CONSISTENT-ACROSS-WINDOWS" and r["spread_pp"] == pytest.approx(16.0)
    assert read_w5({"w0": -20.0})["verdict"] == V_UNREADABLE
    keys = [((40, 9003, "w0"), s) for s in range(1, N + 1)]
    assert read_w6({k: 2.1 for k in keys}, {k: 6.8 for k in keys})["verdict"] == V_WAIT_FELL
    assert read_w6({k: 6.9 for k in keys}, {k: 6.8 for k in keys})["verdict"] == V_WAIT_DID_NOT_FALL
    assert read_w6({}, {k: 6.8 for k in keys})["verdict"] == V_UNREADABLE
