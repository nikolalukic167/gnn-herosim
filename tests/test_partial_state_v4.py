"""partial_state_v4: v3 plus the CD greedy's load terms in seconds (docs/lineages/load_repr_v1.md).

Columns 0-21 are v3's byte for byte; 22-24 carry log1p(backlog), log1p(service committed by
batch-mates to the SAME replica) and log1p(their sum). $PARTIAL_STATE_LOAD_SECONDS=0 zeroes the
three (the disabled twin). The backlog is the clock the co-sim replays
(live_snapshot_seed.seeded_backlog_seconds), refactored out of _seed_platform_state.
"""
import math

import numpy as np
import pytest

from src.placement.live_snapshot_seed import _approx_comm, seeded_backlog_seconds
from src.policy.tabular.reduced_features import (
    LOAD_SECONDS_DIM, PARTIAL_STATE_CONTRACT_V3, PARTIAL_STATE_CONTRACT_V4, PartialStateContext,
    krank_node_order, partial_state_columns, partial_state_feature_dim, partial_state_mlp_layout,
    validate_partial_state_contract,
)

NODES = ("n0", "n1", "n2")
# two replicas per node; every task may use every replica, so batch-mates share keys
REPLICAS = [(n, p) for n in NODES for p in (0, 1)]
BACKLOG = {r: float(i) * 2.0 for i, r in enumerate(REPLICAS)}  # (n0,0) idle


def _ctx(contract, n_tasks=3, **kw):
    caps = {n: math.inf for n in NODES}
    hop = {n: float(i) for i, n in enumerate(NODES)}
    tasks = list(range(n_tasks))
    extra = {}
    if contract == PARTIAL_STATE_CONTRACT_V4:
        extra = dict(backlog_s=BACKLOG,
                     service_s={(t, r): 1.0 + t + 0.5 * r[1] for t in tasks for r in REPLICAS})
    extra.update(kw)
    ctx = PartialStateContext(
        node_caps=caps, demand={(t, r): 1.0 for t in tasks for r in REPLICAS},
        node_of={r: r[0] for r in REPLICAS}, task_type_index={t: t % 4 for t in tasks},
        parents={t: [] for t in tasks},
        route_hops_bneck={(a, b): ((0.0, math.inf) if a == b else (1.0, 100.0)) for a in NODES for b in NODES},
        payload_bytes=0.0, transfer_norm=0.0, node_rank=krank_node_order(caps, hop, contract=contract),
        ingress_links={(t, n): () for t in tasks for n in NODES}, core_links=frozenset(),
        peer_pairs={(0, 1): 1.0, (1, 0): 1.0}, peer_norm=extra.pop("peer_norm", 1.0),
        node_exchange={(a, b): ((0.0, 0.0) if a == b else (1.0, 0.1)) for a in NODES for b in NODES},
        cand_nodes={t: list(NODES) for t in tasks}, contract=contract, **extra,
    )
    return ctx


def test_v4_is_valid_and_25_wide_and_has_no_mlp_layout():
    assert validate_partial_state_contract("partial_state_v4") == PARTIAL_STATE_CONTRACT_V4
    assert LOAD_SECONDS_DIM == 3
    assert partial_state_feature_dim(PARTIAL_STATE_CONTRACT_V3) == 22
    assert partial_state_feature_dim(PARTIAL_STATE_CONTRACT_V4) == 25
    with pytest.raises(ValueError, match="no MLP layout"):
        partial_state_mlp_layout(PARTIAL_STATE_CONTRACT_V4)


def test_v4_keeps_v3_columns_and_prices_committed_service_per_replica(monkeypatch):
    monkeypatch.delenv("PARTIAL_STATE_LOAD_SECONDS", raising=False)
    committed = {0: ("n1", 1), 1: ("n1", 1)}
    v3 = partial_state_columns(_ctx(PARTIAL_STATE_CONTRACT_V3), 2, REPLICAS, committed)
    v4 = partial_state_columns(_ctx(PARTIAL_STATE_CONTRACT_V4), 2, REPLICAS, committed)
    assert v4.shape == (len(REPLICAS), 25)
    np.testing.assert_array_equal(v4[:, :22], v3)
    for i, r in enumerate(REPLICAS):
        # tasks 0 and 1 committed to (n1, 1): 1.5 + 2.5 seconds; nothing elsewhere,
        # including the sibling replica (n1, 0) on the same node
        c_s = 4.0 if r == ("n1", 1) else 0.0
        assert v4[i, 22] == pytest.approx(math.log1p(BACKLOG[r]), rel=1e-6)
        assert v4[i, 23] == pytest.approx(math.log1p(c_s), rel=1e-6)
        assert v4[i, 24] == pytest.approx(math.log1p(BACKLOG[r] + c_s), rel=1e-6)


def test_committed_service_charges_the_peer_transfer_across_nodes(monkeypatch):
    monkeypatch.delenv("PARTIAL_STATE_LOAD_SECONDS", raising=False)
    # partners 0 and 1 on different nodes: each pays 1.0 B x 1.0 s/B + 0.1 s latency
    committed = {0: ("n0", 0), 1: ("n1", 1)}
    v4 = partial_state_columns(_ctx(PARTIAL_STATE_CONTRACT_V4), 2, REPLICAS, committed)
    want = {("n0", 0): 1.0 + 1.1, ("n1", 1): 2.5 + 1.1}
    for i, r in enumerate(REPLICAS):
        assert v4[i, 23] == pytest.approx(math.log1p(want.get(r, 0.0)), rel=1e-6)


def test_the_twin_zeroes_exactly_the_load_columns(monkeypatch):
    committed = {0: ("n1", 1)}
    monkeypatch.setenv("PARTIAL_STATE_LOAD_SECONDS", "1")
    on = partial_state_columns(_ctx(PARTIAL_STATE_CONTRACT_V4), 2, REPLICAS, committed)
    monkeypatch.setenv("PARTIAL_STATE_LOAD_SECONDS", "0")
    off = partial_state_columns(_ctx(PARTIAL_STATE_CONTRACT_V4), 2, REPLICAS, committed)
    np.testing.assert_array_equal(on[:, :22], off[:, :22])
    assert not off[:, 22:].any() and on[:, 22:].any()
    monkeypatch.setenv("PARTIAL_STATE_LOAD_SECONDS", "2")
    with pytest.raises(ValueError, match="expected 0 or 1"):
        _ctx(PARTIAL_STATE_CONTRACT_V4)


def test_v4_refuses_to_serve_without_its_ingredients():
    with pytest.raises(ValueError, match="needs backlog_s and service_s"):
        _ctx(PARTIAL_STATE_CONTRACT_V4, backlog_s=None, service_s=None)
    ctx = _ctx(PARTIAL_STATE_CONTRACT_V4, backlog_s={("n0", 0): 0.0})
    with pytest.raises(ValueError, match="no backlog"):
        partial_state_columns(ctx, 0, REPLICAS, {})


TT = {"executionTime": {"xavierGpu": 0.4}, "stateSize": {"app": {"input": 2e6, "output": 1e6}}}


def _old_seed_clock(spec, tt, ptype):
    """_seed_platform_state's formula before the refactor, verbatim."""
    q = int(spec.get("queue_length", 0) or 0)
    cur = float(spec.get("current_task_remaining", 0) or 0)
    com = float(spec.get("comm_remaining", 0) or 0)
    if q <= 0 and cur <= 0.0 and com <= 0.0:
        return None
    if tt is None:
        return None
    total = cur + com + q * (float(tt["executionTime"].get(ptype, 0.0)) + _approx_comm(tt))
    drain = float(spec.get("queue_drain_seconds", 0.0) or 0.0)
    if drain > 0.0:
        total = cur + com + drain
    return total


@pytest.mark.parametrize("spec", [
    {},
    {"queue_length": 3},
    {"queue_length": 3, "queue_drain_seconds": 17.5},
    {"current_task_remaining": 0.2, "comm_remaining": 0.03},
    {"queue_length": 2, "current_task_remaining": 0.2, "comm_remaining": 0.03, "queue_drain_seconds": 9.0},
])
@pytest.mark.parametrize("tt", [TT, None])
def test_seeded_backlog_is_the_old_seed_clock(spec, tt):
    assert seeded_backlog_seconds(spec, tt, "xavierGpu") == _old_seed_clock(spec, tt, "xavierGpu")


def test_exchange_seconds_is_log1p_seconds_and_free_of_peer_norm(monkeypatch):
    """exchange_seconds_v1: columns 7-8 in log1p seconds, independent of the per-batch peer_norm;
    every other column unchanged; off by default."""
    monkeypatch.delenv("PARTIAL_STATE_EXCHANGE_SECONDS", raising=False)
    committed = {1: ("n1", 0)}
    off = partial_state_columns(_ctx(PARTIAL_STATE_CONTRACT_V4, peer_norm=4.0), 0, REPLICAS, committed)
    monkeypatch.setenv("PARTIAL_STATE_EXCHANGE_SECONDS", "1")
    on = partial_state_columns(_ctx(PARTIAL_STATE_CONTRACT_V4, peer_norm=4.0), 0, REPLICAS, committed)
    on_other_norm = partial_state_columns(_ctx(PARTIAL_STATE_CONTRACT_V4, peer_norm=0.5), 0, REPLICAS, committed)
    np.testing.assert_array_equal(on, on_other_norm)
    rest = [c for c in range(on.shape[1]) if c not in (7, 8)]
    np.testing.assert_array_equal(on[:, rest], off[:, rest])
    # task 0 on n0 vs partner on n1: 1 byte * 1.0 s/B + 0.1 s latency; same node costs nothing
    x = {n: (0.0 if n == "n1" else 1.1) for n in NODES}
    for i, r in enumerate(REPLICAS):
        assert on[i, 7] == pytest.approx(math.log1p(x[r[0]]), rel=1e-6)
        assert off[i, 7] == pytest.approx(x[r[0]] / 4.0, rel=1e-6)
    # peer mass: task 1 is committed, so nothing is left to look ahead to for task 0
    assert np.all(on[:, 8] == 0.0)
    lone = partial_state_columns(_ctx(PARTIAL_STATE_CONTRACT_V4), 0, REPLICAS, {})
    x_mass = {n: sum((0.0 if n == m else 1.1) for m in NODES) / len(NODES) for n in NODES}
    for i, r in enumerate(REPLICAS):
        assert lone[i, 8] == pytest.approx(math.log1p(x_mass[r[0]]), rel=1e-6)


def test_exchange_seconds_refuses_pre_v4_contracts(monkeypatch):
    monkeypatch.setenv("PARTIAL_STATE_EXCHANGE_SECONDS", "1")
    with pytest.raises(ValueError, match="only under partial_state_v4"):
        _ctx(PARTIAL_STATE_CONTRACT_V3)
    monkeypatch.setenv("PARTIAL_STATE_EXCHANGE_SECONDS", "yes")
    with pytest.raises(ValueError, match="expected 0 or 1"):
        _ctx(PARTIAL_STATE_CONTRACT_V4)
