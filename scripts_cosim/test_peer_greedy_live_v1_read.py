"""peer_greedy_live_v1: the registered bars behave as signed, before any rule arm is served.

Run: pipenv run python3 -m pytest scripts_cosim/test_peer_greedy_live_v1_read.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts_cosim.peer_greedy_live_v1_read import (  # noqa: E402
    P_ALPHA, P_BATCHED, P_DRAIN, P_GRAPH, P_IMMEDIATE, P_MECHANISM_MIN_ENVS, P_MIN_CHECKPOINTS,
    P_MIN_ENVIRONMENTS, P_PRIMARY_RUNG, P_RULES, P_RUNGS, P_SEPARATE_PCT,
    V_BATCHED_FASTER_THAN_IMMEDIATE, V_COLOCATION_DELIVERED, V_COLOCATION_NOT_DELIVERED,
    V_EXCHANGE_HELPS, V_GRAPH_FASTER_THAN_RULE, V_IMMEDIATE_BEATS_REACTIVE,
    V_IMMEDIATE_FASTER_THAN_BATCHED, V_NOT_SEP, V_REACTIVE_FASTER_THAN_IMMEDIATE,
    V_RULE_LOSES, V_RULE_MATCHES_GRAPH, V_RULE_TIES, V_RULE_WINS,
    broadcast_rule, env_stats, environments, pair_checkpoint_stats, read_g1, read_g3, read_g5,
    read_g6, read_g7, read_g8,
)
from scripts_cosim.unsaturated_edge_v1_read import E_ALPHA, E_MIN_CHECKPOINTS, E_SEPARATE_PCT  # noqa: E402
from scripts_cosim.peer_only_v1_read import V_UNREADABLE  # noqa: E402

ENVS = environments(40, [9001, 9003, 9101, 9104])
SEEDS = list(range(1, 17))


def _by_env(base, step=0.1):
    return {e: base + step * i for i, e in enumerate(ENVS)}


def test_bars_are_the_chains_and_the_units_are_named():
    assert (P_SEPARATE_PCT, P_ALPHA) == (E_SEPARATE_PCT, E_ALPHA)
    assert P_MIN_ENVIRONMENTS == 16 and P_MIN_CHECKPOINTS == E_MIN_CHECKPOINTS
    assert P_RUNGS == (40, 80) and P_PRIMARY_RUNG == 40
    assert P_RULES == (P_IMMEDIATE, P_DRAIN, P_BATCHED)
    assert P_GRAPH == "be1670_gnnedge0"
    assert 0 < P_MECHANISM_MIN_ENVS <= P_MIN_ENVIRONMENTS


def test_env_stats_pairs_on_the_environment_and_fails_loud_when_ragged():
    rule = {e: 10.0 + i for i, e in enumerate(ENVS)}          # 10..25
    ref = {e: 2 * (10.0 + i) for i, e in enumerate(ENVS)}     # exactly half everywhere
    s = env_stats(rule, ref, ENVS)
    assert all(abs(v + 50.0) < 1e-9 for v in s.values()) and len(s) == 16
    with pytest.raises(ValueError):
        env_stats({e: 1.0 for e in ENVS[:-1]}, ref, ENVS)


def test_g1_fires_in_both_directions_and_ties_inside_the_bar():
    assert read_g1(_by_env(-12.0))["verdict"] == V_IMMEDIATE_BEATS_REACTIVE
    assert read_g1(_by_env(+9.0))["verdict"] == V_REACTIVE_FASTER_THAN_IMMEDIATE
    r = read_g1(_by_env(-2.0))
    assert r["verdict"] == V_NOT_SEP and r["n"] == 16
    assert read_g1({e: -12.0 for e in ENVS[:15]})["verdict"] == V_UNREADABLE


def test_g3_pairs_the_broadcast_rule_against_every_checkpoint():
    rule = {e: 20.0 for e in ENVS}
    graph = {(e, s): 20.0 * (1 + 0.20 + 0.001 * s) for e in ENVS for s in SEEDS}   # arm 20 % slower
    st = pair_checkpoint_stats(broadcast_rule(rule, SEEDS), graph, ENVS)
    assert len(st) == 16
    assert read_g3(st)["verdict"] == V_GRAPH_FASTER_THAN_RULE is False or read_g3(st)["verdict"] in (
        V_RULE_MATCHES_GRAPH, V_GRAPH_FASTER_THAN_RULE, "RULE-FASTER-THAN-GRAPH-ARM")
    # rule 20 s vs graph 24 s: the RULE is faster, negative median
    assert read_g3(st)["median"] < -P_SEPARATE_PCT
    assert read_g3(st)["verdict"] == "RULE-FASTER-THAN-GRAPH-ARM"
    tie = {(e, s): 20.0 * (1 + 0.01 * ((s % 2) * 2 - 1)) for e in ENVS for s in SEEDS}
    assert read_g3(pair_checkpoint_stats(broadcast_rule(rule, SEEDS), tie, ENVS))["verdict"] == V_RULE_MATCHES_GRAPH
    # ragged checkpoints are disclosed, never averaged: 14 seeds is UNREADABLE by the letter
    short = {(e, s): v for (e, s), v in graph.items() if s <= 14}
    assert read_g3(pair_checkpoint_stats(broadcast_rule(rule, SEEDS[:14]), short, ENVS))["verdict"] == V_UNREADABLE
    assert read_g3(pair_checkpoint_stats(broadcast_rule(rule, SEEDS[:14]), short, ENVS), min_n=14)["verdict"] == "RULE-FASTER-THAN-GRAPH-ARM"


def test_g5_and_g6_name_the_faster_configuration():
    assert read_g5(_by_env(-20.0))["verdict"] == V_IMMEDIATE_FASTER_THAN_BATCHED
    assert read_g5(_by_env(+20.0))["verdict"] == V_BATCHED_FASTER_THAN_IMMEDIATE
    assert read_g6(_by_env(-8.0))["verdict"] == V_EXCHANGE_HELPS


def test_g7_needs_the_exchange_to_fall_on_twelve_environments():
    ex = {e: (3.6, 4.7) for e in ENVS}
    q = {e: (8.2, 10.0) for e in ENVS}
    r = read_g7(ex, q)
    assert r["verdict"] == V_COLOCATION_DELIVERED and r["exchange_fell_on"] == 16
    assert r["median_exchange_delta_s"] < 0 and r["median_queue_delta_s"] < 0
    mixed = {e: ((3.6, 4.7) if i < P_MECHANISM_MIN_ENVS - 1 else (5.0, 4.7)) for i, e in enumerate(ENVS)}
    assert read_g7(mixed, q)["verdict"] == V_COLOCATION_NOT_DELIVERED
    assert read_g7({e: (1, 2) for e in ENVS[:10]}, q)["verdict"] == V_UNREADABLE


def test_g8_composes_from_g1_and_carries_g3_g5():
    g3 = {"verdict": V_RULE_MATCHES_GRAPH}
    g5 = {"verdict": V_IMMEDIATE_FASTER_THAN_BATCHED}
    assert read_g8({"verdict": V_IMMEDIATE_BEATS_REACTIVE}, g3, g5)["verdict"] == V_RULE_WINS
    assert read_g8({"verdict": V_NOT_SEP}, g3, g5)["verdict"] == V_RULE_TIES
    r = read_g8({"verdict": V_REACTIVE_FASTER_THAN_IMMEDIATE}, g3, g5)
    assert r["verdict"] == V_RULE_LOSES and r["graph_arm_matched"] == V_RULE_MATCHES_GRAPH
    assert read_g8({"verdict": V_UNREADABLE}, g3, g5)["verdict"] == V_UNREADABLE
