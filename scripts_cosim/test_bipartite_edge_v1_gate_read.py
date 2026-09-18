"""bipartite_edge_v1: the gate reader turns summaries into the registered verdicts correctly.

Each test here is a defect this programme has already paid for once:

* the reactive baseline vanishing because it is keyed at seed 0 and the reader intersected
  seed sets -- it printed "no reactive at R3" on a table plainly containing one (C4);
* a strict collapse taken on an arm that lost a cell to an OOM, which crashed a read of 125
  healthy arms instead of reporting one unreadable bar (C2/C4);
* a compound verdict reading as if the bar it depends on had fired;
* "unknown is not a pass" -- an unknown saturation must never render as unsaturated.

Run: pipenv run python3 -m pytest scripts_cosim/test_bipartite_edge_v1_gate_read.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts_cosim.bipartite_edge_v1_gate_read import (  # noqa: E402
    EDGE, EDGE0, GNN, PEERONLY, REACTIVE, _read_one, _reactive_like, report,
)
from scripts_cosim.bipartite_edge_v1_read import (  # noqa: E402
    V_CONV_IS_LEVER, V_EDGE_HELPS, V_EDGE_NOT_SEP, V_PENALTY_SURVIVES,
)
from scripts_cosim.peer_only_v1_read import V_UNREADABLE  # noqa: E402

CELLS = ["cs80s9001", "cs80s9002", "cs80s9003", "cs80s9005"]
SEEDS = list(range(1, 17))


def _arm(base, jitter=0.1):
    return {(c, s): base + jitter * s + 0.01 * i for i, c in enumerate(CELLS) for s in SEEDS}


def _table(edge=70.0, edge0=100.0, gnn=100.0, peeronly=60.0, reactive=200.0):
    t = {EDGE: _arm(edge), EDGE0: _arm(edge0), GNN: _arm(gnn), PEERONLY: _arm(peeronly)}
    # Reactive is DETERMINISTIC and keyed at seed 0 -- that is the whole hazard.
    t[REACTIVE] = {(c, 0): reactive + 0.01 * i for i, c in enumerate(CELLS)}
    return t


def test_the_reactive_baseline_does_not_vanish_when_it_is_keyed_at_seed_zero():
    t = _table()
    rep = _reactive_like(t, CELLS, SEEDS)
    assert rep is not None and set(rep) == set(SEEDS)
    assert len(set(rep.values())) == 1, "reactive is deterministic; replication must not jitter it"

    r = _read_one(t, "R3", queue_share=0.95)["registered"]
    assert r["D4"]["verdict"] != V_UNREADABLE, "the C4 defect: reactive present but read as absent"
    assert r["D4"]["median"] < 0, "gnnedge at 70 s vs reactive at 200 s must read as faster"


def test_a_missing_cell_makes_the_registered_read_unreadable_and_the_disclosed_one_readable():
    t = _table()
    del t[EDGE][(CELLS[2], 13)]          # one arm lost to an OOM, as three were in C2/C4
    out = _read_one(t, "R3", queue_share=0.95)
    assert out["registered"]["D1"]["verdict"] == V_UNREADABLE
    # The bar is NOT relaxed -- the disclosed read is a separate, named block.
    assert out["disclosed"]["n_checkpoints"] == 15
    assert out["disclosed"]["D1"]["verdict"] == V_EDGE_HELPS
    # And the other bars still report rather than taking the whole read down.
    assert out["disclosed"]["D3"]["verdict"] == V_PENALTY_SURVIVES


def test_the_control_turns_a_d1_gain_into_conv_is_the_lever():
    """gnnedge beats gnn, and ties its own zeroed control: the gain is the conv, not the
    attributes. Signed in advance precisely so it cannot be reported as edge-conditioning."""
    r = _read_one(_table(edge=70.0, edge0=70.2, gnn=100.0), "R3", queue_share=0.95)["registered"]
    assert r["D1"]["verdict"] == V_EDGE_HELPS
    assert r["D2"]["verdict"] == V_CONV_IS_LEVER


def test_the_kill_condition_fires_on_two_registered_ties():
    r = _read_one(_table(edge=100.3, edge0=100.0, gnn=100.1), "R3", queue_share=0.95)["registered"]
    assert r["D1"]["verdict"] == V_EDGE_NOT_SEP
    assert r["lineage"]["verdict"] == "EDGE-CONDITIONING-DOES-NOT-TRANSFER"


def test_an_unknown_saturation_never_renders_as_unsaturated():
    r = _read_one(_table(), "R3", queue_share=None)["registered"]
    assert r["D4"]["saturated"] is None
    assert "UNKNOWN -- not a pass" in report({"bar": {"separate_pct": 5.0, "alpha": 0.05,
                                                      "min_seeds": 16},
                                              "rungs": [{"rung": "R3", "cells": CELLS,
                                                         "registered": r, "disclosed": r}]})


def test_a_rung_with_no_shared_cells_is_unreadable_rather_than_empty():
    t = _table()
    t[EDGE] = {}
    assert _read_one(t, "R3", queue_share=0.9)["verdict"] == V_UNREADABLE


def test_the_report_prints_a_real_ahead_count_and_refuses_a_missing_one():
    """`paired_tie` names it `v3_ahead`; reading `ahead` printed "None/16" beside a -20.37 %
    headline. A missing count must never render as if it were a reported one."""
    import pytest
    from scripts_cosim.bipartite_edge_v1_gate_read import _fmt
    r = _read_one(_table(), "R3", queue_share=0.95)["registered"]
    assert r["D1"]["v3_ahead"] == 16, "all 16 checkpoints should be ahead on this synthetic table"
    assert "16/16" in _fmt(r["D1"]) and "None" not in _fmt(r["D1"])
    with pytest.raises(KeyError, match="no ahead-count"):
        _fmt({"verdict": "X", "median": -1.0, "p": 0.01, "n": 16})
