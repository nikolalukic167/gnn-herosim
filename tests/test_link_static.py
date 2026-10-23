"""partial_state_v8: the static link block S1-S3 (link_features.static_edge_columns) and its conv attribute (gnn/link_static.py)."""
import math
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from test_link_features import (  # noqa: E402
    NAMES, REPLICAS, SPECS, MIB, base_ctx_kwargs, fabric_env, node_exchange, psc_of, start_flows, stored_block, synthetic_graph, topology,
)

from src.placement import link_features as LF  # noqa: E402
from src.policy.gnn import link_static as LS  # noqa: E402
from src.policy.tabular import reduced_features as RF  # noqa: E402

V7, V8 = RF.PARTIAL_STATE_CONTRACT_V7, RF.PARTIAL_STATE_CONTRACT_V8


def tables(fabric_env):
    env, fabric = fabric_env
    start_flows(env, fabric, SPECS, until=0.0)
    counts = LF.counts_of(LF.flows_live(fabric.fairshare, env.now))
    return counts, LF.routes_between(fabric.link_topology, NAMES), node_exchange(fabric.link_topology, NAMES)


def test_v8_is_33_wide_and_extends_v7_in_every_family():
    assert RF.partial_state_feature_dim(V8) == RF.partial_state_feature_dim(V7) + LF.LINK_STATIC_DIM == 33
    for fam in (RF.VALID_PARTIAL_STATE_CONTRACTS, RF.PEER_BLOCK_CONTRACTS, RF.LOAD_SECONDS_CONTRACTS, RF.BOTH_DIRECTION_CONTRACTS, RF.FOUR_TYPE_CONTRACTS,
                RF.INFLIGHT_CONTRACTS, RF.LINK_CONTRACTS):
        assert V8 in fam and V7 in fam
    assert RF.STATIC_LINK_CONTRACTS == frozenset({V8})


def test_static_columns_against_hand_computed_values(fabric_env):
    counts, routes, nx = tables(fabric_env)
    b = 8 * MIB
    # no peers: zeros
    assert LF.static_edge_columns(counts, routes, nx, [], 1) == (0.0, 0.0, 0.0)
    # a partner whose only candidate node is the candidate itself moves nothing
    assert LF.static_edge_columns(counts, routes, nx, [([1], b)], 1) == (0.0, 0.0, 0.0)
    # a partner with an empty candidate list is skipped
    assert LF.static_edge_columns(counts, routes, nx, [([], b)], 1) == (0.0, 0.0, 0.0)
    # one partner, candidates (0, 1) for candidate node 1: node 1 contributes 0 but counts in the mean; s0>core_0 and core_0>s1 carry 2 flows each
    s1, s2, s3 = LF.static_edge_columns(counts, routes, nx, [([0, 1], b)], 1)
    rate = min(4.0 * MIB / 3, 10.0 * MIB / 3)
    shared = b / rate + nx[(0, 1)][1]
    solo = b * nx[(0, 1)][0] + nx[(0, 1)][1]
    assert s1 == math.log1p(shared / 2) and s2 == math.log(shared / solo) and s3 == math.log1p(2.0 / 2)
    # duplicate candidate nodes weigh by multiplicity: (0, 0, 1) is 2/3 of the s0 price
    d1, d2, d3 = LF.static_edge_columns(counts, routes, nx, [([0, 0, 1], b)], 1)
    assert d1 == math.log1p(2 * shared / 3) and d2 == pytest.approx(s2, abs=1e-12) and d3 == math.log1p(2 * 2.0 / 3)
    # two partners: sums add, busiest is the max of the per-partner means, sums in the order given
    t1, t2, t3 = LF.static_edge_columns(counts, routes, nx, [([0], b), ([2], b)], 1)
    sh2 = b / min(10.0 * MIB / 3, 20.0 * MIB / 2) + nx[(2, 1)][1]  # s2>core_0 is loaded by 1 flow, core_0>s1 by 2
    assert t1 == math.log1p(shared + sh2)
    assert t3 == math.log1p(max(2.0, 2.0))
    # an empty fabric prices solo: S2 is 0 and S3 is 0
    e1, e2, e3 = LF.static_edge_columns({}, routes, nx, [([0], b)], 1)
    assert e1 == math.log1p(b * nx[(0, 1)][0] + nx[(0, 1)][1]) and e2 == 0.0 and e3 == 0.0


def test_partial_state_columns_carry_s1_to_s3_after_e1_to_e3_and_are_prefix_free(fabric_env):
    counts, routes, nx = tables(fabric_env)
    kw = base_ctx_kwargs(3, REPLICAS, counts, routes, nx)
    v7 = RF.PartialStateContext(**kw)
    v8 = RF.PartialStateContext(**{**kw, "contract": V8, "node_rank": RF.krank_node_order(kw["node_caps"], {n: float(n) for n in range(3)}, contract=V8)})
    commits = [{}, {0: (0, 0)}, {0: (0, 0), 1: (2, 20)}, {1: (1, 10), 2: (2, 21)}]
    blocks = [RF.partial_state_columns(v8, 0 if 0 not in c else 2, REPLICAS, c) for c in commits]
    ref = RF.partial_state_columns(v8, 2, REPLICAS, {})
    for c in commits:
        t = 2 if 2 not in c else 0
        got = RF.partial_state_columns(v8, t, REPLICAS, c)
        assert got.shape == (len(REPLICAS), 33)
        for i, cand in enumerate(REPLICAS):
            partners = LS.static_partners(v8, t)
            assert got[i, 30:33].tolist() == np.float32(LF.static_edge_columns(counts, routes, nx, partners, cand[0])).tolist()
        # the v7 columns of the same context are untouched
        want7 = RF.partial_state_columns(v7, t, REPLICAS, c)
        assert np.array_equal(got[:, :30], want7)
    # prefix-free: for one task S1-S3 are identical whatever is committed
    outs = [RF.partial_state_columns(v8, 2, REPLICAS, c)[:, 30:33] for c in ({}, {0: (0, 0)}, {0: (0, 0), 1: (2, 20)}, {1: (1, 10)})]
    assert all(np.array_equal(outs[0], o) for o in outs[1:]) and np.array_equal(outs[0], ref[:, 30:33])
    assert len(blocks) == len(commits)
    # the block moves with the flows, not a constant: a quiet fabric gives other values
    quiet = RF.PartialStateContext(**{**kw, "contract": V8, "link_counts": {}, "node_rank": v8.node_rank})
    assert not np.array_equal(RF.partial_state_columns(quiet, 2, REPLICAS, {})[:, 30:33], ref[:, 30:33])


def test_link_static_attr_rows_equal_the_columns_forward_and_reverse_and_live_equals_cache(fabric_env, monkeypatch):
    monkeypatch.setenv("PARTIAL_STATE_CONTRACT", V8)
    env, fabric = fabric_env
    start_flows(env, fabric, SPECS, until=0.7)
    nx = node_exchange(fabric.link_topology, NAMES)
    node_names = [NAMES[r[0]] for r in REPLICAS]
    live_flows = LF.flows_live(fabric.fairshare, env.now)
    live_psc = psc_of(LF.counts_of(live_flows), LF.routes_between(fabric.link_topology, NAMES), nx)
    stored_flows = LF.flows_stored(stored_block(fabric.fairshare, SPECS))
    cache_psc = psc_of(LF.counts_of(stored_flows), LF.routes_between({"routes": topology()["routes"], "links": topology()["links"]}, NAMES), nx)
    plat = np.asarray([LF.platform_columns(LF.node_raw(live_flows, n)) for n in node_names])
    live_g, cache_g = synthetic_graph(plat, live_psc), synthetic_graph(plat, cache_psc)
    LS.attach_link_static(live_g)
    LS.attach_link_static(cache_g)
    a, b = live_g.link_static_attr, cache_g.link_static_attr
    assert a.shape == (live_g.edge_index.size(1), 3) and a.dtype == torch.float32
    assert a.numpy().tobytes() == b.numpy().tobytes()  # live vs cache, bitwise
    ctx = RF.build_partial_state_context_from_graph(live_g)
    from src.policy.gnn.partial_state_edges import candidate_edge_rows
    from src.policy.gnn.plan_raw import _edge_platforms

    rows = candidate_edge_rows(live_g)
    _, reverse = _edge_platforms(live_g)
    for t in range(live_g.n_tasks):
        cols = RF.partial_state_columns(ctx, t, [tuple(c) for c in live_g.task_logit_to_placement[t]], {})
        for k in range(len(REPLICAS)):
            assert a[rows[t][k]].tolist() == np.float32(cols[k, 30:33]).tolist()
            assert torch.equal(a[reverse[rows[t][k]]], a[rows[t][k]])
    # every row of the bipartite edge_index is a candidate row or its reverse: no row is left at zero by accident of alignment
    assert int(((a != 0).any(dim=1)).sum()) > 0


def test_a_v8_graph_without_link_tables_fails_loud(fabric_env, monkeypatch):
    monkeypatch.setenv("PARTIAL_STATE_CONTRACT", V8)
    counts, routes, nx = tables(fabric_env)
    psc = psc_of(counts, routes, nx)
    psc["link_counts"] = None
    g = synthetic_graph(np.zeros((len(REPLICAS), 3)), psc)
    with pytest.raises(ValueError):
        LS.link_static_attr(g)
