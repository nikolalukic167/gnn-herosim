"""partial_state_v3: the size-free krank block (docs/lineages/partial_state_v3.md).

v3 keeps v2's columns 0-9 and the 4 linkrank columns byte-for-byte and replaces the
24-column rank one-hot with 8 scalars (rank_frac, inv_n in the task's own type slot). The
v2 rank must be exactly recoverable at N <= 6, and a cluster with more than KRANK_WIDTH
candidate-hosting nodes must serve under v3 where v2 raises.
"""
import math

import numpy as np
import pytest

from src.policy.tabular import reduced_features as rf
from src.policy.tabular.reduced_features import (
    KRANK_FEATURE_DIM, KRANK_TYPES, KRANK_V3_FEATURE_DIM, KRANK_V3_SCALARS, KRANK_WIDTH,
    LINKRANK_FEATURE_DIM, PARTIAL_STATE_BASE_DIM, PARTIAL_STATE_CONTRACT_V1,
    PARTIAL_STATE_CONTRACT_V2, PARTIAL_STATE_CONTRACT_V3, PARTIAL_STATE_FEATURE_DIM,
    PARTIAL_STATE_V3_FEATURE_DIM, PartialStateContext, krank_feature_dim, krank_node_order,
    partial_state_columns, partial_state_feature_dim, validate_partial_state_contract,
)


def _ctx(contract, n_nodes, task_types=(0, 1, 2, 3)):
    """A peer corpus (peer_norm > 0, no DAG parents) with one candidate per node for every
    task, uniform caps, ingress routes over one shared link so linkrank is exercised."""
    nodes = [f"n{i}" for i in range(n_nodes)]
    caps = {n: 100.0 for n in nodes}
    hop = {n: float(i) for i, n in enumerate(nodes)}          # distinct -> distinct ranks
    node_rank = krank_node_order(caps, hop, contract=contract)
    tasks = list(range(len(task_types)))
    cands = {t: [(t, n) for n in nodes] for t in tasks}
    demand = {(t, c): 10.0 for t in tasks for c in cands[t]}
    node_of = {c: c[1] for t in tasks for c in cands[t]}
    route = {(a, b): ((0.0, math.inf) if a == b else (2.0, 1000.0)) for a in nodes for b in nodes}
    ingress = {(t, n): (f"link_{n}",) for t in tasks for n in nodes}
    exchange = {(a, b): ((0.0, 0.0) if a == b else (1.0, 0.5)) for a in nodes for b in nodes}
    ctx = PartialStateContext(
        node_caps=caps, demand=demand, node_of=node_of,
        task_type_index={t: k for t, k in zip(tasks, task_types)},
        parents={t: [] for t in tasks}, route_hops_bneck=route, payload_bytes=1.0,
        transfer_norm=1.0, node_rank=node_rank, ingress_links=ingress,
        core_links=frozenset({"link_n0"}),
        # every task has a committed-side partner (0 or 1) and an uncommitted one, so
        # both the committed-exchange and the peer-mass columns are exercised
        peer_pairs={(0, 1): 1.0, (1, 0): 1.0, (2, 0): 1.0, (3, 1): 1.0, (2, 3): 1.0, (3, 2): 1.0},
        node_exchange=exchange, peer_norm=1.0, cand_nodes={t: nodes for t in tasks},
        contract=contract,
    )
    return ctx, cands


def test_v3_is_a_valid_contract_and_the_dims_are_what_the_node_says():
    assert validate_partial_state_contract("partial_state_v3") == PARTIAL_STATE_CONTRACT_V3
    assert KRANK_V3_SCALARS == 2 and KRANK_V3_FEATURE_DIM == KRANK_TYPES * 2 == 8
    assert PARTIAL_STATE_V3_FEATURE_DIM == PARTIAL_STATE_BASE_DIM + 8 + LINKRANK_FEATURE_DIM == 22
    assert partial_state_feature_dim(PARTIAL_STATE_CONTRACT_V1) == PARTIAL_STATE_FEATURE_DIM == 38
    assert partial_state_feature_dim(PARTIAL_STATE_CONTRACT_V2) == 38
    assert partial_state_feature_dim(PARTIAL_STATE_CONTRACT_V3) == 22
    assert krank_feature_dim(PARTIAL_STATE_CONTRACT_V2) == KRANK_FEATURE_DIM == 24
    assert krank_feature_dim(PARTIAL_STATE_CONTRACT_V3) == 8


def test_the_pad_is_a_property_of_the_one_hot_contracts_only():
    caps = {f"n{i}": 1.0 for i in range(KRANK_WIDTH + 1)}
    hop = {n: float(i) for i, n in enumerate(caps)}
    with pytest.raises(ValueError, match="exceed the registered pad"):
        krank_node_order(caps, hop, contract=PARTIAL_STATE_CONTRACT_V2)
    order = krank_node_order(caps, hop, contract=PARTIAL_STATE_CONTRACT_V3)
    assert sorted(order.values()) == list(range(KRANK_WIDTH + 1))
    # the ORDER itself is contract-free
    small = {n: caps[n] for n in list(caps)[:3]}
    assert krank_node_order(small, {n: hop[n] for n in small}, contract=PARTIAL_STATE_CONTRACT_V2) == \
        krank_node_order(small, {n: hop[n] for n in small}, contract=PARTIAL_STATE_CONTRACT_V3)


def test_v3_emits_22_columns_with_the_scalars_in_the_own_type_slot():
    ctx, cands = _ctx(PARTIAL_STATE_CONTRACT_V3, n_nodes=6)
    block = partial_state_columns(ctx, 2, cands[2], {})          # task 2 is type 2
    assert block.shape == (6, 22)
    k = 2
    for i, cand in enumerate(cands[2]):
        r = ctx.node_rank[cand[1]]
        slot = PARTIAL_STATE_BASE_DIM + k * KRANK_V3_SCALARS
        assert block[i, slot] == pytest.approx(r / 5)
        assert block[i, slot + 1] == pytest.approx(1 / 6)
        others = [PARTIAL_STATE_BASE_DIM + j for j in range(8) if j not in (2 * k, 2 * k + 1)]
        assert np.all(block[i, others] == 0.0)


def test_v3_base_and_linkrank_columns_are_byte_identical_to_v2():
    """The P0 property, on a synthetic dataset: only the krank block differs."""
    v2, c2 = _ctx(PARTIAL_STATE_CONTRACT_V2, n_nodes=5)
    v3, c3 = _ctx(PARTIAL_STATE_CONTRACT_V3, n_nodes=5)
    committed = {0: c2[0][1], 1: c2[1][3]}                        # a prefix on two nodes
    for t in (2, 3):
        b2 = partial_state_columns(v2, t, c2[t], committed)
        b3 = partial_state_columns(v3, t, c3[t], committed)
        assert b2.shape == (5, 38) and b3.shape == (5, 22)
        np.testing.assert_array_equal(b2[:, :PARTIAL_STATE_BASE_DIM], b3[:, :PARTIAL_STATE_BASE_DIM])
        np.testing.assert_array_equal(b2[:, -LINKRANK_FEATURE_DIM:], b3[:, -LINKRANK_FEATURE_DIM:])
        assert np.any(b2[:, -LINKRANK_FEATURE_DIM:] != 0.0)       # linkrank actually exercised
        assert np.any(b2[:, 7:9] != 0.0)                          # the peer block too


def test_v2_rank_is_exactly_recoverable_from_v3_at_every_small_n():
    for n in range(1, KRANK_WIDTH + 1):
        v2, c2 = _ctx(PARTIAL_STATE_CONTRACT_V2, n_nodes=n)
        v3, c3 = _ctx(PARTIAL_STATE_CONTRACT_V3, n_nodes=n)
        for t in range(4):
            b2 = partial_state_columns(v2, t, c2[t], {})
            b3 = partial_state_columns(v3, t, c3[t], {})
            k = v2.task_type_index[t]
            for i in range(n):
                hot = np.flatnonzero(b2[i, PARTIAL_STATE_BASE_DIM:PARTIAL_STATE_BASE_DIM + 24] > 0.5)
                r2, k2 = divmod(int(hot[0]), KRANK_TYPES)
                assert k2 == k
                rank_frac, inv_n = b3[i, PARTIAL_STATE_BASE_DIM + 2 * k], b3[i, PARTIAL_STATE_BASE_DIM + 2 * k + 1]
                n_rec = int(round(1.0 / inv_n))
                assert n_rec == n
                assert (0 if n == 1 else int(round(rank_frac * (n - 1)))) == r2


def test_v3_serves_a_cluster_the_one_hot_cannot():
    ctx, cands = _ctx(PARTIAL_STATE_CONTRACT_V3, n_nodes=80)
    block = partial_state_columns(ctx, 1, cands[1], {})
    assert block.shape == (80, 22)
    slot = PARTIAL_STATE_BASE_DIM + 1 * KRANK_V3_SCALARS
    assert block[:, slot].min() == 0.0 and block[:, slot].max() == pytest.approx(1.0)
    assert np.all(block[:, slot + 1] == pytest.approx(1 / 80))
    with pytest.raises(ValueError, match="exceed the registered pad"):
        _ctx(PARTIAL_STATE_CONTRACT_V2, n_nodes=80)


def test_the_mlp_layout_refuses_v3(monkeypatch):
    monkeypatch.setenv(rf.PARTIAL_STATE_CONTRACT_ENV, PARTIAL_STATE_CONTRACT_V3)
    with pytest.raises(ValueError, match="no MLP layout"):
        rf._batch_edge_feature_dims(14, candidate_relative=True, partial_state=True)
    monkeypatch.setenv(rf.PARTIAL_STATE_CONTRACT_ENV, PARTIAL_STATE_CONTRACT_V2)
    dim, _, layout = rf._batch_edge_feature_dims(14, candidate_relative=True, partial_state=True)
    assert (dim, layout) == (rf.DIM63CRK_FEATURE_DIM, "dim63crk")


def test_v3_needs_a_peer_corpus_like_v2():
    with pytest.raises(ValueError, match="requires a peer_exchange corpus"):
        PartialStateContext(
            node_caps={}, demand={}, node_of={}, task_type_index={}, parents={},
            route_hops_bneck={}, payload_bytes=1.0, transfer_norm=1.0, node_rank={},
            ingress_links={}, core_links=frozenset(), peer_norm=0.0,
            contract=PARTIAL_STATE_CONTRACT_V3,
        )
