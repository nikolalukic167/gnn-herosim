"""env_lever_v1: the shared bars behave as signed, before any levered arm is served.

Run: pipenv run python3 -m pytest scripts_cosim/test_env_lever_v1_read.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts_cosim.env_lever_v1_read import (  # noqa: E402
    L_ARMS, L_BATCHED, L_GRAPH, L_IMMEDIATE, L_LEVERS, L_LINEAGE_OF, L_PAYLOAD_SCALES, L_RUNG,
    L_TWIN, L_WAIT_COLLAPSED_S, V_ARM_BEATS_REACTIVE, V_DESIGN_READY, V_DESIGN_SHORT,
    V_GRAPH_MATCHES_RULE, V_NO_REGIME, V_NOT_SEP, V_REACTIVE_FASTER, V_REGIME_FROM,
    V_RULE_BEATS_REACTIVE, V_RULE_FASTER_THAN_GRAPH, V_WAIT_COLLAPSED, V_WAIT_DID_NOT_COLLAPSE,
    broadcast_rule, checkpoint_stats, environments, pair_checkpoint_stats, read_l0, read_l1,
    read_l2, read_l3, read_l4, read_l6,
)
from scripts_cosim.unsaturated_edge_v1_read import E_TOPOLOGY_CANDIDATES, E_WINDOWS  # noqa: E402
from scripts_cosim.peer_only_v1_read import V_UNREADABLE  # noqa: E402

ENVS = environments(40, [9001, 9003, 9101, 9104])
SEEDS = list(range(1, 17))


def test_levers_map_to_three_lineages_and_only_burst_carries_the_twin():
    assert set(L_LINEAGE_OF) == set(L_LEVERS)
    assert set(L_LINEAGE_OF.values()) == {"burst_groups_v1", "payload_scale_v1", "backbone_sparsity_v1"}
    assert L_TWIN in L_ARMS["burst"] and all(L_TWIN not in L_ARMS[l] for l in L_LEVERS if l != "burst")
    assert all({L_GRAPH, L_IMMEDIATE, L_BATCHED} <= set(L_ARMS[l]) for l in L_LEVERS)
    assert L_RUNG == 40 and L_PAYLOAD_SCALES == (0.1, 1.0, 3.0, 10.0) and L_WAIT_COLLAPSED_S == 1.0


def test_l0_screens_the_lever_itself_and_unknown_is_not_a_pass():
    ok = {(40, t, w): 0.5 for t in E_TOPOLOGY_CANDIDATES for w in E_WINDOWS}
    r = read_l0(ok)
    assert r["verdict"] == V_DESIGN_READY and r["selection"]["topologies"] == [9001, 9002, 9003, 9005]
    # burst raises the instantaneous load: three topologies unknown on w0, one saturated
    short = dict(ok)
    for t in (9001, 9002, 9003):
        short.pop((40, t, "w0"))
    short[(40, 9005, "w0")] = 0.91
    r = read_l0(short)
    assert r["verdict"] == V_DESIGN_READY and r["selection"]["topologies"] == [9101, 9102, 9103, 9104]
    for t in E_TOPOLOGY_CANDIDATES[:9]:
        short[(40, t, "w1")] = None
    assert read_l0(short)["verdict"] == V_DESIGN_SHORT


def test_l1_and_l2_fire_each_way_with_their_units():
    arm = {(e, s): 20.0 * (1 - 0.10 - 0.001 * s) for e in ENVS for s in SEEDS}
    react = {e: 20.0 for e in ENVS}
    st = checkpoint_stats(arm, react, ENVS)
    assert len(st) == 16 and read_l1(st)["verdict"] == V_ARM_BEATS_REACTIVE
    slow = {(e, s): 20.0 * (1 + 0.10 + 0.001 * s) for e in ENVS for s in SEEDS}
    assert read_l1(checkpoint_stats(slow, react, ENVS))["verdict"] == V_REACTIVE_FASTER
    assert read_l2({e: -9.0 - 0.1 * i for i, e in enumerate(ENVS)})["verdict"] == V_RULE_BEATS_REACTIVE
    assert read_l2({e: -1.0 + 0.1 * i for i, e in enumerate(ENVS)})["verdict"] == V_NOT_SEP
    assert read_l2({e: -9.0 for e in ENVS[:15]})["verdict"] == V_UNREADABLE


def test_l3_pairs_the_graph_arm_against_the_broadcast_rule():
    rule = {e: 20.0 for e in ENVS}
    graph = {(e, s): 22.0 + 0.01 * s for e in ENVS for s in SEEDS}
    st = pair_checkpoint_stats(graph, broadcast_rule(rule, SEEDS), ENVS)
    assert read_l3(st)["verdict"] == V_RULE_FASTER_THAN_GRAPH and read_l3(st)["median"] > 5
    tie = {(e, s): 20.0 * (1 + 0.01 * ((s % 2) * 2 - 1)) for e in ENVS for s in SEEDS}
    assert read_l3(pair_checkpoint_stats(tie, broadcast_rule(rule, SEEDS), ENVS))["verdict"] == V_GRAPH_MATCHES_RULE


def test_l4_is_a_bar_on_the_lever_not_on_the_arm():
    assert read_l4({(e, s): 0.3 for e in ENVS for s in SEEDS})["verdict"] == V_WAIT_COLLAPSED
    assert read_l4({(e, s): 7.1 for e in ENVS for s in SEEDS})["verdict"] == V_WAIT_DID_NOT_COLLAPSE
    assert read_l4({})["verdict"] == V_UNREADABLE


def test_l6_names_the_smallest_monotone_scale():
    r = read_l6({0.1: V_NOT_SEP, 1.0: V_RULE_BEATS_REACTIVE, 3.0: V_RULE_BEATS_REACTIVE}, beats=V_RULE_BEATS_REACTIVE)
    assert r["verdict"] == f"{V_REGIME_FROM}-x1" and r["from_scale"] == 1.0
    r = read_l6({0.1: V_NOT_SEP, 1.0: V_NOT_SEP, 3.0: V_RULE_BEATS_REACTIVE}, beats=V_RULE_BEATS_REACTIVE)
    assert r["verdict"] == f"{V_REGIME_FROM}-x3"
    assert read_l6({0.1: V_NOT_SEP, 1.0: V_NOT_SEP, 3.0: V_NOT_SEP}, beats=V_RULE_BEATS_REACTIVE)["verdict"] == V_NO_REGIME
    # fires at 1 but not at 3: not a regime
    assert read_l6({0.1: V_NOT_SEP, 1.0: V_RULE_BEATS_REACTIVE, 3.0: V_NOT_SEP}, beats=V_RULE_BEATS_REACTIVE)["verdict"] == V_NO_REGIME
    # a scale with no design (UNREADABLE) is left out of the composition, not counted as a failure
    r = read_l6({0.1: V_NOT_SEP, 1.0: V_RULE_BEATS_REACTIVE, 3.0: V_RULE_BEATS_REACTIVE, 10.0: V_UNREADABLE}, beats=V_RULE_BEATS_REACTIVE)
    assert r["verdict"] == f"{V_REGIME_FROM}-x1" and r["scales"][10.0] == V_UNREADABLE
