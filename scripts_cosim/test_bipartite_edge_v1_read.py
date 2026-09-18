"""bipartite_edge_v1: the registered bars behave as signed, before any arm exists.

The point of running these now is that a bar which has never been exercised is a bar nobody
has read. Each test below corresponds to a way peer_only_v1's readers actually failed:

* a verdict that inverts when the same reader is called with its arguments swapped (C4);
* a reader that relaxes its own minimum when an arm is lost (never allowed — the registered
  read says UNREADABLE and a disclosed read prints beside it);
* a compound verdict (CONV-IS-THE-LEVER, BIPARTITE-REPAIRED) that reads as if it fired when
  the bar it depends on did not.

Run: pipenv run python3 -m pytest scripts_cosim/test_bipartite_edge_v1_read.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts_cosim.bipartite_edge_v1_read import (  # noqa: E402
    D_ALPHA, D_MIN_SEEDS, D_SEPARATE_PCT, D_RUNGS, D_CLIENTS,
    V_ATTRS_ARE_LEVER, V_ATTRS_COST, V_CLOSES_NEGATIVE, V_CONV_IS_LEVER,
    V_BIPARTITE_REPAIRED, V_D2_NOT_SEP, V_EDGE_COSTS, V_EDGE_HELPS, V_EDGE_NOT_SEP,
    V_PENALTY_SURVIVES,
    read_d1, read_d2, read_d3, read_d4, read_lineage, rung_cells,
)
from scripts_cosim.peer_only_v1_read import V_UNREADABLE, V_PAIR_TIE  # noqa: E402

N = D_MIN_SEEDS


def _by_seed(value, n=N, step=0.0):
    return {s: value + step * s for s in range(n)}


# --- the bars are what the node says they are -----------------------------------------

def test_the_registered_constants_match_the_signed_node():
    assert (D_SEPARATE_PCT, D_ALPHA, D_MIN_SEEDS) == (5.0, 0.05, 16)
    assert D_RUNGS == ("R3",) and D_CLIENTS == (40,)


# --- D1 --------------------------------------------------------------------------------

def test_d1_fires_helps_only_when_the_new_arm_is_faster_by_the_bar():
    faster = read_d1(_by_seed(70.0, step=0.1), _by_seed(100.0, step=0.1))
    assert faster["verdict"] == V_EDGE_HELPS and faster["median"] < -D_SEPARATE_PCT

    slower = read_d1(_by_seed(130.0, step=0.1), _by_seed(100.0, step=0.1))
    assert slower["verdict"] == V_EDGE_COSTS

    tie = read_d1(_by_seed(101.0, step=0.1), _by_seed(100.0, step=0.1))
    assert tie["verdict"] == V_EDGE_NOT_SEP


def test_d1_cannot_invert_when_its_arguments_are_swapped():
    """The C4 defect, pinned. `read_c1(gnnres, peeronly)` labelled "13 % SLOWER" as the
    bipartite stage HELPING: the number was right and the word was backwards. Swapping the
    arguments here must flip the verdict, not preserve it."""
    a, b = _by_seed(70.0, step=0.1), _by_seed(100.0, step=0.1)
    assert read_d1(a, b)["verdict"] == V_EDGE_HELPS
    assert read_d1(b, a)["verdict"] == V_EDGE_COSTS


def test_d1_refuses_a_short_read_rather_than_relaxing_its_own_bar():
    short = {s: 70.0 for s in range(N - 3)}
    assert read_d1(short, {s: 100.0 for s in range(N - 3)})["verdict"] == V_UNREADABLE
    # A DISCLOSED read is possible, but only because the caller asked for it by name.
    disclosed = read_d1(short, {s: 100.0 for s in range(N - 3)}, min_seeds=N - 3)
    assert disclosed["verdict"] == V_EDGE_HELPS


# --- D2, the load-bearing bar ----------------------------------------------------------

def test_d2_separates_the_attributes_from_the_conv():
    attrs = read_d2(_by_seed(70.0, step=0.1), _by_seed(100.0, step=0.1), d1_verdict=V_EDGE_HELPS)
    assert attrs["verdict"] == V_ATTRS_ARE_LEVER

    cost = read_d2(_by_seed(130.0, step=0.1), _by_seed(100.0, step=0.1), d1_verdict=V_EDGE_HELPS)
    assert cost["verdict"] == V_ATTRS_COST


def test_d2_calls_a_gain_the_conv_when_the_zeroed_control_matches_it():
    """The whole reason gnnedge0 is trained. If the treatment ties its own zeroed control
    while D1 gained, the gain is mean-aggregation or MLP shape and must be reported that
    way -- not as edge-conditioning."""
    r = read_d2(_by_seed(100.5, step=0.1), _by_seed(100.0, step=0.1), d1_verdict=V_EDGE_HELPS)
    assert r["verdict"] == V_CONV_IS_LEVER


def test_d2_does_not_claim_the_conv_is_the_lever_when_d1_found_nothing():
    """A tie here plus a tie there is two ties, not a mechanism."""
    r = read_d2(_by_seed(100.5, step=0.1), _by_seed(100.0, step=0.1), d1_verdict=V_EDGE_NOT_SEP)
    assert r["verdict"] == V_D2_NOT_SEP
    assert read_d2(_by_seed(100.5, step=0.1), _by_seed(100.0, step=0.1))["verdict"] == V_D2_NOT_SEP


# --- D3 --------------------------------------------------------------------------------

def test_d3_reports_the_penalty_surviving_when_peeronly_is_still_ahead():
    r = read_d3(_by_seed(120.0, step=0.1), _by_seed(100.0, step=0.1), d1_verdict=V_EDGE_HELPS)
    assert r["verdict"] == V_PENALTY_SURVIVES


def test_d3_only_calls_the_bipartite_stage_repaired_if_d1_actually_gained():
    tied = (_by_seed(100.5, step=0.1), _by_seed(100.0, step=0.1))
    assert read_d3(*tied, d1_verdict=V_EDGE_HELPS)["verdict"] == V_BIPARTITE_REPAIRED
    # Tying peeronly because BOTH arms are bad is not a repair.
    assert read_d3(*tied, d1_verdict=V_EDGE_NOT_SEP)["verdict"] == V_PAIR_TIE


# --- D4 --------------------------------------------------------------------------------

def test_d4_carries_the_saturation_classification_with_the_number():
    unsat = read_d4(_by_seed(70.0, step=0.1), _by_seed(100.0, step=0.1), queue_share=0.73)
    assert unsat["saturated"] is False and unsat["queue_share"] == 0.73
    assert read_d4(_by_seed(70.0, step=0.1), _by_seed(100.0, step=0.1),
                   queue_share=0.95)["saturated"] is True
    # Unknown is not a pass: with no queue share the classification stays None, never False.
    assert read_d4(_by_seed(70.0, step=0.1), _by_seed(100.0, step=0.1))["saturated"] is None


# --- the kill condition ----------------------------------------------------------------

def test_the_kill_condition_fires_only_on_two_registered_ties():
    d1_tie = {"verdict": V_EDGE_NOT_SEP}
    d2_tie = {"verdict": V_D2_NOT_SEP}
    assert read_lineage(d1_tie, d2_tie)["verdict"] == V_CLOSES_NEGATIVE
    assert read_lineage({"verdict": V_EDGE_HELPS}, d2_tie)["verdict"] == "OPEN"
    # An arm lost to a resource kill must not be read as a negative result.
    assert read_lineage({"verdict": V_UNREADABLE}, d2_tie)["verdict"] == V_UNREADABLE


def test_rung_cells_refuses_a_bare_string():
    with pytest.raises(ValueError, match="not a string"):
        rung_cells("R3")
    assert rung_cells(["cs80s9001", "cs80s9002"]) == ("cs80s9001", "cs80s9002")
