"""peer_affinity_v1 Phase 0 -- the paper probe must be exactly what it claims.

(a) x == 0 => the plan cost is the additive base plus the count-shaped sharing term, and on
    collision-free plans it is exactly additive (R^2 == 1, argmin == optimum);
(b) co-located peers exchange for free; (c) the exchange is symmetric in (i, j);
(d) the per-byte transfer equals Platform._payload_transfer_time on a real NetworkFabric;
(e) the enumeration is the full n_cand^k product and plan_index inverts it;
(f) the greedy with an exact prefix is feasible and never beats the enumerated optimum.

Runs on the first arm_b0 dataset when present; skips (never fails) otherwise.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts_cosim"))

import peer_affinity_probe as PA  # noqa: E402

SRC = PA.CORPUS_DEFAULT / "ds_00000"
pytestmark = pytest.mark.skipif(not (SRC / "placements/placements.jsonl").exists(),
                                reason="arm_b0 ds_00000 not on this machine")


@pytest.fixture(scope="module")
def paper():
    src = PA.Source(SRC)
    p = PA.Paper(src, k=6, n_cand=3, partners=2, seed=7001)
    p.enumerate()
    return p


def test_enumeration_is_the_full_product(paper):
    assert paper.N == 3 ** 6
    assert len({tuple(r) for r in paper.P.tolist()}) == paper.N
    for r in (0, 17, paper.N - 1):
        assert paper.plan_index(paper.P[r].tolist()) == r


def test_x_zero_is_base_plus_sharing_and_additive_when_collision_free(paper):
    y0 = paper.total(0.0)
    assert np.allclose(y0, paper.BASE + paper.SHARE)
    cf = np.nonzero(paper.collision_free)[0]
    assert len(cf) > 50
    assert np.allclose(y0[cf], paper.BASE[cf])
    pos, r2, _n = PA.fit_argmin(paper, y0, cf, cf, ("ind",))
    assert r2 > 1 - 1e-9
    assert y0[cf[pos]] == pytest.approx(y0[cf].min())


def test_colocated_peers_are_free_and_exchange_is_symmetric(paper):
    src = paper.src
    n = src.per_byte.shape[0]
    assert np.all(np.diag(src.per_byte) == 0) and np.all(np.diag(src.latency) == 0)
    assert np.allclose(src.per_byte, src.per_byte.T) and np.allclose(src.latency, src.latency.T)
    assert src.max_asymmetry < 1e-6
    # a plan with every task on one node carries zero exchange
    for node in range(n):
        rows = np.nonzero((paper.NODE == node).all(1))[0]
        if len(rows):
            assert np.all(paper.exchange(1e9)[rows] == 0.0)
            break
    else:
        pytest.skip("no all-on-one-node plan at this candidate draw")


def test_per_byte_matches_the_simulator_transfer(paper):
    from src.placement.infrastructure import Platform
    from src.placement.network_fabric import NetworkFabric
    import simpy

    src = paper.src
    infra = __import__("json").load(open(SRC / "infrastructure.json"))
    fabric = NetworkFabric(simpy.Environment(), infra["link_topology"])

    class FakeNode:
        def __init__(self, name):
            self.node_name = name
            self.fabric = fabric
            self.network = {"bandwidth": 1000.0}

    payload = 123_456_789.0
    for a in range(len(src.nodes)):
        for b in range(len(src.nodes)):
            if a == b:
                continue
            plat = Platform.__new__(Platform)
            plat.node = FakeNode(src.nodes[b])
            sim = Platform._payload_transfer_time(plat, src.nodes[a], payload)
            assert sim == pytest.approx(src.per_byte[a, b] * payload, rel=1e-9)


def test_greedy_is_feasible_and_never_beats_the_optimum(paper):
    x = 200e6
    for alpha in (1.5, 2.0):
        cap = paper.caps(alpha)
        y = paper.total(x)
        feas = np.nonzero((paper.load <= cap[None, :] + PA.EPS).all(1))[0]
        if len(feas) == 0:
            continue  # the screen reports this as no_feasible_rows; nothing to compare here
        opt = y[feas].min()
        for look in (False, True):
            plan = PA.greedy(paper, x, cap, list(range(paper.k)), look)
            if plan is None:
                continue
            idx = paper.plan_index(plan)
            assert idx in set(feas.tolist())
            assert y[idx] >= opt - 1e-9
