"""partial_state_v9: the outbound link block R1-R3, U1 (link_features.out_edge_columns), in the sequential columns and in the batched teacher-forced loss."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import test_batched_tf_link as T  # noqa: E402
from test_batched_tf_link import N_TASKS, scenario  # noqa: E402
from test_link_features import (  # noqa: E402
    NAMES, REPLICAS, SPECS, MIB, base_ctx_kwargs, fabric_env, node_exchange, start_flows,
)

from src.placement import link_features as LF  # noqa: E402
from src.policy.gnn import batched_tf as B  # noqa: E402
from src.policy.tabular import reduced_features as RF  # noqa: E402

V7, V8, V9 = RF.PARTIAL_STATE_CONTRACT_V7, RF.PARTIAL_STATE_CONTRACT_V8, RF.PARTIAL_STATE_CONTRACT_V9
OUT0 = 33  # first R column: after the 27 v5 columns, E1-E3 (27-29), S1-S3 (30-32)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("PARTIAL_STATE_CONTRACT", V9)
    monkeypatch.setenv("PARTIAL_STATE_LOAD_SECONDS", "1")
    monkeypatch.setenv("PARTIAL_STATE_EXCHANGE_SECONDS", "1")
    monkeypatch.setenv("PARTIAL_STATE_PEER_MASS", "1")


def tables(fabric_env):
    env, fabric = fabric_env
    start_flows(env, fabric, SPECS, until=0.0)
    counts = LF.counts_of(LF.flows_live(fabric.fairshare, env.now))
    return counts, LF.routes_between(fabric.link_topology, NAMES), node_exchange(fabric.link_topology, NAMES)


def test_v9_is_37_wide_and_extends_v8_in_every_family():
    assert RF.partial_state_feature_dim(V9) == RF.partial_state_feature_dim(V8) + LF.LINK_OUT_DIM == 37
    for fam in (RF.VALID_PARTIAL_STATE_CONTRACTS, RF.PEER_BLOCK_CONTRACTS, RF.LOAD_SECONDS_CONTRACTS, RF.BOTH_DIRECTION_CONTRACTS, RF.FOUR_TYPE_CONTRACTS,
                RF.INFLIGHT_CONTRACTS, RF.LINK_CONTRACTS, RF.STATIC_LINK_CONTRACTS):
        assert V9 in fam and V8 in fam
    assert RF.OUT_LINK_CONTRACTS == frozenset({V9})


def test_out_columns_against_hand_computed_values(fabric_env):
    counts, routes, nx = tables(fabric_env)
    b = 8 * MIB
    assert LF.out_edge_columns(counts, routes, nx, [], 1) == (0.0, 0.0, 0.0, 0.0)
    assert LF.out_edge_columns(counts, routes, nx, [(1, b)], 1) == (0.0, 0.0, 0.0, 0.0)  # co-located: nothing moves
    # one partner on node 0: the flow runs node 1 -> node 0 over the candidate's uplink first; the S columns of the v8 fixture put 2 flows on core_0>s1, 1 on s2>core_0
    hops = routes[(1, 0)]
    act = [counts.get(k, 0) for k, _ in hops]
    r1, r2, r3, u1 = LF.out_edge_columns(counts, routes, nx, [(0, b)], 1)
    rate = min(bw * MIB / (n + 1) for (_k, bw), n in zip(hops, act))
    lat = nx[(1, 0)][1]
    shared = b / rate + lat if any(act) else b * nx[(1, 0)][0] + lat
    solo = b * nx[(1, 0)][0] + lat
    assert r1 == math.log1p(shared)
    assert r2 == (math.log(shared / solo) if shared > solo else 0.0)
    assert r3 == math.log1p(float(max(act)))
    assert u1 == math.log1p(b * (act[0] + 1) / (hops[0][1] * MIB))
    # two remote partners: R sums, and U1 prices the two flows as sharing the uplink with one another
    q = LF.out_edge_columns(counts, routes, nx, [(0, b), (2, 2 * b)], 1)
    assert q[3] == math.log1p(3 * b * (counts.get(routes[(1, 0)][0][0], 0) + 2) / (routes[(1, 0)][0][1] * MIB))
    # an empty fabric prices solo: R2 and R3 are 0
    e = LF.out_edge_columns({}, routes, nx, [(0, b)], 1)
    assert e[0] == math.log1p(b * nx[(1, 0)][0] + nx[(1, 0)][1]) and e[1] == 0.0 and e[2] == 0.0


def test_outbound_is_the_inbound_price_of_the_same_flow_seen_from_the_other_end(fabric_env):
    """R1-R3 of a candidate are E1-E3 of its partner for the same flow: one pricing rule, applied to the direction E1-E3 leave out."""
    counts, routes, nx = tables(fabric_env)
    b = 8 * MIB
    for a, c in ((0, 1), (1, 0), (1, 2), (2, 1)):
        inbound = LF.edge_columns(counts, routes, nx, [(c, b)], a)
        outbound = LF.out_edge_columns(counts, routes, nx, [(c, b)], a)
        mirrored = LF.edge_columns(counts, routes, nx, [(a, b)], c)  # the same flow seen from the other end, priced by E1-E3
        assert outbound[:3] == mirrored


def test_v8_columns_are_untouched_and_r_is_prefix_dependent(monkeypatch):
    for seed in range(3):
        rnd, ctx, g, place = scenario(seed, monkeypatch)
        ctx.contract = V9
        ctx8 = scenario(seed, monkeypatch)[1]
        ctx8.contract = V8
        for t in range(N_TASKS):
            committed = {j: tuple(place[j][0]) for j in range(t)}
            w9 = RF.partial_state_columns(ctx, t, [tuple(c) for c in place[t]], committed)
            w8 = RF.partial_state_columns(ctx8, t, [tuple(c) for c in place[t]], committed)
            assert w9.shape[1] == 37 and np.array_equal(w9[:, :OUT0], w8)
            if not committed:
                assert (w9[:, OUT0:] == 0).all()


def test_batched_out_block_equals_the_definition_on_random_plans_masks_and_prefixes(monkeypatch):
    for seed in range(5):
        rnd, ctx, g, place = scenario(seed, monkeypatch)
        ctx.contract = V9
        S = B._Static(g, ctx)
        assert S.template.shape[1] == 37
        nonzero = 0
        for _ in range(6):
            plan = [rnd.randrange(len(place[t])) for t in range(N_TASKS)]
            for M in (np.tril(np.ones((N_TASKS, N_TASKS)), -1), 1.0 - np.eye(N_TASKS), np.zeros((N_TASKS, N_TASKS))):
                out = B._blocks(S, ctx, g, plan, M)
                assert out.shape == (S.R, 37)
                for t in range(N_TASKS):
                    committed = {j: tuple(place[j][int(plan[j])]) for j in range(N_TASKS) if M[t, j] > 0}
                    want = RF.partial_state_columns(ctx, t, [tuple(c) for c in place[t]], committed).astype(np.float32)
                    got = out[S.start[t]:S.start[t] + S.C[t]]
                    assert np.array_equal(got[:, OUT0:], want[:, OUT0:])
                    assert np.abs(got[:, :OUT0] - want[:, :OUT0]).max() <= 1e-5
                    nonzero += int((got[:, OUT0:] != 0).any(1).sum())
        assert nonzero > 0  # not vacuous: the outbound block is live in the loss input


@pytest.mark.parametrize("trained,serving", [(V8, V9), (V9, V8), (V7, V9), (V9, V7)])
def test_a_checkpoint_of_the_other_contract_is_refused(trained, serving):
    with pytest.raises(RF.PartialStateContractMismatchError):
        RF.require_matching_partial_state_contract(trained, serving, model_label="ckpt")
    RF.require_matching_partial_state_contract(trained, trained, model_label="ckpt")


def test_a_sidecarless_checkpoint_is_refused_for_v9():
    with pytest.raises(RF.PartialStateContractMismatchError):
        RF.require_matching_partial_state_contract(None, V9, model_label="ckpt")
