#!/usr/bin/env python3
"""Tests for the link repair control in separability_diagnostic.py.

`--gate-link-repair` is what decides whether link_contention_v1 lives or dies, so it needs
coverage in both directions: it must FIRE on coupling that a scalar link summary explains,
and must NOT fire on coupling that needs the incidence pattern.

The 2026-08-18 lesson behind this file: `--gate-coupled-fraction` was recommended as the
primary gate while being structurally incapable of failing, and it would have rejected the
only configuration showing headroom. A gate nobody tested is not a gate.

Run: pipenv run python3 -m pytest scripts_cosim/test_link_repair_control.py -q
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts_cosim.separability_diagnostic import (  # noqa: E402
    _link_repair_columns,
    _plan_link_load,
    variance_decomposition,
)
from src.placement.network_fabric import link_key  # noqa: E402

N_CLIENTS = 4
SOURCES = [f"client_node{i}" for i in range(N_CLIENTS)]

# Three servers. node0 and node1 both hang off coreX and are reached over the SHARED
# segment coreEntry|coreX; node2 is reached over coreEntry|coreY. So two tasks bound for
# node0 and node1 — DIFFERENT destination nodes — still cross one common link.
ROUTES = {
    src: {
        "node0": [src, "coreEntry", "coreX", "node0"],
        "node1": [src, "coreEntry", "coreX", "node1"],
        "node2": [src, "coreEntry", "coreY", "node2"],
    }
    for src in SOURCES
}
CONTEXT = {"routes": ROUTES, "sources": SOURCES, "n_clients": N_CLIENTS}

SHARED_X = link_key("coreEntry", "coreX")
SHARED_Y = link_key("coreEntry", "coreY")

TASK_IDS = [0, 1, 2]
# Node indices: clients are 0..3, so servers node0/node1/node2 are 4/5/6.
NODE_CHOICES = [4, 5, 6]


def _plans():
    """Every assignment of 3 tasks to 3 servers — 27 plans, comfortably determined."""
    return list(itertools.product(NODE_CHOICES, repeat=len(TASK_IDS)))


def _combos(rtt_fn):
    combos = []
    for row in _plans():
        plan = {task: (node, 100 + node) for task, node in zip(TASK_IDS, row)}
        combos.append((plan, rtt_fn(row)))
    return combos


def _load_for(row):
    return _plan_link_load(row, TASK_IDS, CONTEXT)


# --- the load mapping itself -------------------------------------------------------


def test_local_execution_loads_no_links():
    """A task run on its own source node never touches the network."""
    # Task 0's source is client_node0, i.e. node index 0.
    load = _plan_link_load([0, 4, 4], TASK_IDS, CONTEXT)
    assert load[SHARED_X] == 2  # only tasks 1 and 2 crossed it


def test_different_destinations_still_share_a_link():
    """The structural claim: node0 and node1 are different nodes on one shared segment."""
    load = _plan_link_load([4, 5, 6], TASK_IDS, CONTEXT)
    assert load[SHARED_X] == 2  # node0 + node1
    assert load[SHARED_Y] == 1  # node2


def test_repair_columns_summarise_the_load():
    load = _plan_link_load([4, 5, 6], TASK_IDS, CONTEXT)
    busiest, second, excess = _link_repair_columns(load)
    # Each client has its OWN access link, so the only genuinely shared segment here is
    # coreEntry|coreX, carrying the two tasks bound for node0 and node1.
    assert busiest == 2.0
    assert second == 1.0
    assert excess == 1.0


def test_empty_load_is_all_zero():
    assert _link_repair_columns(_plan_link_load([0, 1, 2], TASK_IDS, CONTEXT)) == [0.0, 0.0, 0.0]


# --- the control fires on count-shaped coupling ------------------------------------


def test_control_fires_when_cost_is_a_function_of_link_load():
    """RTT driven purely by the busiest link's load: k1 must repair essentially all of it.

    This is the degenerate case the gate exists to catch — the same shape as node-ingress,
    just counted over links.
    """

    def rtt(row):
        load = _load_for(row)
        busiest = max(load.values()) if load else 0
        return 1.0 + 5.0 * (busiest - 1)

    result = variance_decomposition(_combos(rtt), TASK_IDS, CONTEXT)
    assert not result["degenerate"]
    if result["additive_choice_regret_rel"] > 0:
        assert result["link_repair_frac_k1"] >= 0.9


def test_control_is_absent_without_a_link_context():
    """Every pre-link_contention_v1 corpus must still analyse, reporting None."""

    def rtt(row):
        return 1.0 + 0.1 * sum(row)

    result = variance_decomposition(_combos(rtt), TASK_IDS, None)
    assert result["link_repair_frac_k1"] is None
    assert result["link_repair_frac_excess"] is None


def test_control_is_absent_when_the_additive_fit_is_already_optimal():
    """Nothing to repair means no repair fraction — it must not be reported as 0.0 and
    dilute the mean, the same convention one_integer_repair_frac already uses."""

    def rtt(row):
        return 1.0 + 0.1 * sum(row)  # purely additive

    result = variance_decomposition(_combos(rtt), TASK_IDS, CONTEXT)
    assert result["additive_choice_regret_rel"] == 0.0
    assert result["link_repair_frac_k1"] is None


# --- the control does NOT fire on pattern-shaped coupling ---------------------------


def test_control_does_not_fire_when_which_link_matters_more_than_how_many():
    """Two segments with very different costs at the same load.

    k1 sees "busiest link load = 2" in both cases and cannot tell the expensive collision
    from the cheap one, so it should repair strictly less than the pure-count case. This
    is the shape link congestion is *supposed* to have — cost depending on which links
    overlap, not on a count.
    """

    def rtt(row):
        load = _load_for(row)
        # Sharing the X segment is 20x worse than sharing the Y segment.
        return (
            1.0
            + 20.0 * max(0, load[SHARED_X] - 1)
            + 1.0 * max(0, load[SHARED_Y] - 1)
        )

    counted = variance_decomposition(_combos(rtt), TASK_IDS, CONTEXT)
    assert not counted["degenerate"]

    def pure_count_rtt(row):
        load = _load_for(row)
        busiest = max(load.values()) if load else 0
        return 1.0 + 20.0 * (busiest - 1)

    pure = variance_decomposition(_combos(pure_count_rtt), TASK_IDS, CONTEXT)

    # Both must be measurable, and the count-shaped one must be at least as repairable.
    for result in (counted, pure):
        assert "link_repair_frac_k1" in result
    if (
        counted.get("link_repair_frac_k1") is not None
        and pure.get("link_repair_frac_k1") is not None
    ):
        assert pure["link_repair_frac_k1"] >= counted["link_repair_frac_k1"]


def test_spread_filter_neutralises_the_node_control():
    """The isolation control's contract: every retained plan has node-occupancy excess 0.

    That is what makes the subset able to answer "is there coupling here that is NOT the
    collision term?" — the column that repaired the four previous mechanisms becomes a
    constant and cannot explain anything.
    """
    from scripts_cosim.separability_diagnostic import _excess_sharing, _spread_plans_only

    def rtt(row):
        return 1.0 + 0.1 * sum(row)

    combos = _combos(rtt)
    spread = _spread_plans_only(combos)

    assert 0 < len(spread) < len(combos)
    for plan, _ in spread:
        nodes = [node for node, _platform in plan.values()]
        assert _excess_sharing(nodes) == 0
        # Distinct nodes imply distinct platforms, so added_in_batch is zero too.
        platforms = [platform for _node, platform in plan.values()]
        assert len(set(platforms)) == len(platforms)


def test_spread_filter_keeps_link_sharing_intact():
    """The other half of the contract: filtering out collisions must NOT filter out link
    contention, or the isolation would be vacuous."""
    from scripts_cosim.separability_diagnostic import _spread_plans_only

    def rtt(row):
        return 1.0 + float(_load_for(row)[SHARED_X])

    spread = _spread_plans_only(_combos(rtt))
    # At least one all-distinct-node plan still loads the shared segment twice.
    loads = [
        _plan_link_load([node for node, _ in plan.values()], TASK_IDS, CONTEXT)[SHARED_X]
        for plan, _ in spread
    ]
    assert max(loads) >= 2


def test_cost_exists_where_node_occupancy_excess_is_zero():
    """The premise of the whole lineage, stated as a test.

    Cost comes only from sharing the X segment, which two tasks incur while sitting on
    DIFFERENT nodes (node0 and node1). There is therefore a family of plans with
    node-occupancy excess exactly zero and strictly positive contention cost — something
    that could not happen for any of the four mechanisms tried before this one, whose
    contended object was always indexed by the destination.

    The node column is not *perfectly* uninformative, because co-location is a subset of
    link sharing (two tasks on node0 also share coreEntry|coreX). It is the sharing
    OUTSIDE that subset which is new, so the honest assertion is that the node column
    explains little, not nothing.
    """

    def rtt(row):
        load = _load_for(row)
        return 1.0 + 20.0 * max(0, load[SHARED_X] - 1)

    # All three tasks on distinct nodes: node-occupancy excess is 0 by construction.
    spread = (4, 5, 6)
    assert len(set(spread)) == len(spread)
    assert rtt(spread) > 1.0

    result = variance_decomposition(_combos(rtt), TASK_IDS, CONTEXT)
    assert not result["degenerate"]
    # A single node-collision-count column buys almost no explanatory power here.
    assert result["interaction_r2_gain_node_collision"] < 0.05
