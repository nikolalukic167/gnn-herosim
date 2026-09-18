#!/usr/bin/env python3
"""The client-axis loader: arm labelling and the collision guard. No cluster, no torch."""
from __future__ import annotations

import pytest

from scripts_cosim.peer_only_v1_client_read import tables, _reactive_like


def _row(clients, cell, kind, seed, corpus, elapsed, queue=1.0):
    return {"arm": f"{cell}__C{clients}__{corpus}_{kind}_s{seed}", "clients": clients,
            "cell": cell, "arm_kind": kind, "checkpoint_seed": seed, "corpus": corpus,
            "averageElapsedTime": elapsed, "averageQueueTime": queue}


def test_v3ext_is_relabelled_to_the_arm_it_actually_extends():
    """v3ext IS partial_state_v3's mpoff re-served; A0 proves it bit-identical."""
    t = tables([_row(40, "cc40s9001", "mpoff", 1, "v3ext", 30.0)])
    assert "516_mpoff" in t[40]["elapsed"] and "v3ext_mpoff" not in t[40]["elapsed"]


def test_the_two_mpoff_corpora_stay_separate_arms():
    t = tables([_row(40, "cc40s9001", "mpoff", 1, "v3ext", 30.0),
                _row(40, "cc40s9001", "mpoff", 1, "1670", 36.0)])
    assert t[40]["elapsed"]["516_mpoff"][("cc40s9001", 1)] == 30.0
    assert t[40]["elapsed"]["1670_mpoff"][("cc40s9001", 1)] == 36.0


def test_a_colliding_arm_name_fails_loud_rather_than_letting_one_win():
    with pytest.raises(ValueError, match="colliding"):
        tables([_row(40, "cc40s9001", "mpoff", 1, "1670", 30.0),
                _row(40, "cc40s9001", "mpoff", 1, "1670", 31.0)])


def test_reactive_carries_no_checkpoint_seed():
    rows = [{"arm": "cc40s9001__C40__reactive", "clients": 40, "cell": "cc40s9001",
             "arm_kind": "reactive", "checkpoint_seed": 0, "corpus": "none",
             "averageElapsedTime": 30.5, "averageQueueTime": 22.0}]
    assert list(tables(rows)[40]["elapsed"]["reactive"]) == [("cc40s9001", 0)]


def test_reactive_is_replicated_across_the_learned_arms_seed_keys():
    """Reactive is deterministic; every checkpoint is paired against the SAME baseline."""
    reactive = {("a", 0): 10.0, ("b", 0): 20.0}
    out = _reactive_like(reactive, ["a", "b"], [1, 2, 3])
    assert out == {1: 15.0, 2: 15.0, 3: 15.0}


def test_a_summary_without_a_client_count_is_not_a_client_axis_summary():
    from scripts_cosim.peer_only_v1_client_read import load
    with pytest.raises(ValueError):
        tables([{"arm": "x", "cell": "c"}])
