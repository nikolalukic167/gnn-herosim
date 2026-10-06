"""Checks that the lookahead screen measures information loss and real costs."""
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

from scripts_cosim.peer_lookahead_screen import (
    audit_live_block, coordinate_descent, cycle_edges, factors, greedy, messages, plan_cost, sweep,
)


def test_tree_messages_recover_exact_root_completion_costs():
    cost = np.array([[0., 1.], [1., 0.], [0., 2.], [8., 0.]])
    pairs = {(i, i+1): np.array([[0., 3.], [3., 0.]]) for i in range(3)}
    plans = np.array(list(product(range(2), repeat=4)))
    values = sweep(cost, pairs, plans)
    exact = np.array([values[plans[:, 0] == c].min() for c in range(2)])
    belief = messages(cost, pairs, 3)[0]
    assert np.allclose(exact - exact.min(), belief - belief.min())
    # Changing only the distant task is invisible for two rounds, visible at three.
    other = cost.copy()
    other[3] = [0., 8.]
    assert np.allclose(messages(cost, pairs, 2)[0], messages(other, pairs, 2)[0])
    assert messages(cost, pairs, 3)[0].argmin() != messages(other, pairs, 3)[0].argmin()


def test_matched_cycles_preserve_degree_and_two_hop_root_information():
    a = cycle_edges([0, 1, 2, 3, 4, 5, 6, 7])
    b = cycle_edges([0, 1, 2, 5, 4, 3, 6, 7])
    for edges in (a, b):
        assert [sum(i in e for e in edges) for i in range(8)] == [2]*8
    assert {e for e in a if 0 in e or 1 in e or 7 in e} == {e for e in b if 0 in e or 1 in e or 7 in e}
    cost = np.random.default_rng(42).uniform(0, 10, (8, 3))
    penalty = np.ones((3, 3)) - np.eye(3)
    assert np.allclose(messages(cost, dict.fromkeys(a, penalty), 2)[0],
                       messages(cost, dict.fromkeys(b, penalty), 2)[0])


def test_exchange_factor_matches_sum_of_runtime_endpoint_charges(monkeypatch):
    from src.placement.infrastructure import Platform
    from src.placement.orchestrator import build_peer_exchange_table
    monkeypatch.setenv('HEROSIM_PEER_EXCHANGE', '1')
    payload, bandwidth, latency = 200e6, 1000., .01
    nodes = [SimpleNamespace(node_name=f'node{i}', network={'bandwidth': bandwidth},
                             network_map={f'node{1-i}': latency}, fabric=None) for i in range(2)]
    tasks = {i: SimpleNamespace(id=i, platform=SimpleNamespace(node=nodes[i])) for i in range(2)}
    orch = SimpleNamespace(peer_exchange=build_peer_exchange_table([[0, 1, payload]]), task_by_id=tasks)
    actual = 0.
    for i in range(2):
        nodes[i].orchestrator_ref = orch
        platform = Platform.__new__(Platform)
        platform.node = nodes[i]
        actual += platform._peer_exchange_time(tasks[i])
    paper = SimpleNamespace(k=2, n_cand=2, node=np.array([[0, 1], [0, 1]]),
                            plat=np.array([[0, 1], [0, 1]]), q=np.zeros(2),
                            src=SimpleNamespace(per_byte=(np.ones((2, 2))-np.eye(2))/(bandwidth*1024**2),
                                                latency=(np.ones((2, 2))-np.eye(2))*latency))
    peer, _ = factors(paper, [(0, 1)], payload)
    assert peer[0, 1][0, 1] == pytest.approx(actual)
    assert peer[0, 1][0, 0] == 0
    assert factors(paper, [(0, 1)], 0)[0] == {}


def test_vector_sweep_matches_scalar_objective_and_cd_never_increases_it():
    cost = np.array([[0., 1.], [1., 0.], [2., 0.]])
    pair = {(0, 1): np.array([[0., 3.], [3., 0.]]),
            (1, 2): np.array([[1., 0.], [0., 2.]])}
    plans = np.array(list(product(range(2), repeat=3)))
    assert np.allclose(sweep(cost, pair, plans), [plan_cost(cost, pair, p) for p in plans])
    for plan in plans:
        changed = coordinate_descent(cost, pair, plan, 3)
        assert plan_cost(cost, pair, changed) <= plan_cost(cost, pair, plan) + 1e-10


def test_real_prefix_builder_drops_unarrived_peer_edges():
    audit = audit_live_block()
    assert audit['diagnostics']['peers_outside_batch'] == 1
    assert audit['peer_edge_count'] == audit['prefix_peer_pair_count'] == 0
    assert audit['future_peer_graph_visible_to_current_decoder'] is False


def test_peer_mass_greedy_reprices_after_a_commitment():
    cost = np.array([[0., 1.], [1., 0.]])
    pair = {(0, 1): np.array([[0., 10.], [10., 0.]])}
    # Independent argmins split the pair; conditioning on the first commitment
    # makes the second task pay one unit to save ten units of exchange.
    assert greedy(cost, pair, lookahead=pair) == [0, 0]
