#!/usr/bin/env python3
"""scheduler_residence_v1 R3 -- cell minting and selection.

A minted cell that differs from its base in anything but the topology seed is a different
experiment wearing the same name; and a selection made on the DEPENDENT variable is choosing
the answer. Both are pinned here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts_cosim.scheduler_residence_v1_r3_cells import (  # noqa: E402
    SELECT_KEY,
    TIEBREAK_KEY,
    CellMintError,
    parse_seeds,
    select_spanning,
    with_seed,
)

BASE = {"nodes": {"client_nodes": {"count": 20}, "server_nodes": {"count": 6}},
        "network": {"topology": {"seed": 9001, "type": "sparse"}, "average": 100},
        "scheduler": {"batch_size": 10, "batch_timeout": 16.0}}


def test_parse_seeds_ranges_and_lists():
    assert parse_seeds("1-3") == [1, 2, 3]
    assert parse_seeds("1,5,7") == [1, 5, 7]
    assert parse_seeds("1-2,9") == [1, 2, 9]


def test_parse_seeds_rejects_empty():
    with pytest.raises(CellMintError):
        parse_seeds("  ")


def test_with_seed_changes_exactly_one_field():
    got = with_seed(BASE, 9999)
    assert got["network"]["topology"]["seed"] == 9999
    # everything else byte-identical
    a, b = json.loads(json.dumps(got)), json.loads(json.dumps(BASE))
    a["network"]["topology"]["seed"] = b["network"]["topology"]["seed"]
    assert a == b


def test_with_seed_does_not_mutate_the_base():
    with_seed(BASE, 4242)
    assert BASE["network"]["topology"]["seed"] == 9001


def test_minting_refuses_a_base_that_would_change_more_than_the_seed(monkeypatch):
    """The guard must actually fire, not just exist."""
    import scripts_cosim.scheduler_residence_v1_r3_cells as m

    def sneaky(base, seed):
        cfg = json.loads(json.dumps(base))
        cfg["network"]["topology"]["seed"] = seed
        cfg["scheduler"]["batch_timeout"] = 8.0        # the treatment, smuggled in
        m._assert_only_seed_differs(base, cfg, seed)
        return cfg

    with pytest.raises(CellMintError) as exc:
        sneaky(BASE, 7777)
    assert "batch_timeout" in str(exc.value)


def _row(seed, min_reach, spread=0.0):
    return {"seed": seed,
            "structure": {SELECT_KEY: float(min_reach), TIEBREAK_KEY: float(spread)}}


def test_selection_spans_the_range_and_keeps_both_extremes():
    rows = [_row(9100 + i, i % 5) for i in range(30)]
    picked = select_spanning(rows, 6)
    vals = [p["structure"][SELECT_KEY] for p in picked]
    assert min(vals) == 0.0 and max(vals) == 4.0
    assert len(picked) == 6
    assert len({p["seed"] for p in picked}) == 6      # no cell selected twice


def test_selection_is_deterministic():
    rows = [_row(9100 + i, i % 5) for i in range(30)]
    assert select_spanning(rows, 6) == select_spanning(rows, 6)


def test_selection_refuses_when_too_few_are_measurable():
    rows = [{"seed": 1, "structure": {SELECT_KEY: None}}] * 3
    with pytest.raises(CellMintError):
        select_spanning(rows, 5)


def test_selection_never_reads_a_latency_field():
    """Selection is on the independent variable. If a latency key is present it is ignored."""
    clean = [_row(9100 + i, i % 5) for i in range(30)]
    tempted = [_row(9100 + i, i % 5) for i in range(30)]
    for r in tempted:
        # A dependent variable, ordered to invert the selection if it were ever consulted.
        r["structure"]["queue_excess_s"] = -float(r["seed"])
    assert ([p["seed"] for p in select_spanning(tempted, 6)]
            == [p["seed"] for p in select_spanning(clean, 6)])


def test_selection_of_one_returns_the_lowest():
    rows = [_row(9100 + i, i % 5) for i in range(30)]
    assert select_spanning(rows, 1)[0]["structure"][SELECT_KEY] == 0.0
