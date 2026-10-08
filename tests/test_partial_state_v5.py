"""partial_state_v5: v4 plus the exchange priced in the direction the simulator charges it, and four-type blocks.

Columns 0-24 are v4's, except 7-8 and 23 which price the PULL peer->candidate (what Platform._peer_exchange_time charges
the task at its own input stage); 25-26 carry the PUSH candidate->peer (what the partner pays). On uniform links pull ==
push and v5 reduces to v4 with exchange in seconds. The W3 case (cellular 4 MB/s out, 75 MB/s in) is the reason.
"""
import math
import os
from pathlib import Path

import numpy as np
import pytest

from src.placement.dag_workload import route_hops_and_bottleneck
from src.placement.four_type_features import (
    PLATFORM_EXTRA_DIM, TASK_EXTRA_DIM, platform_extra_columns, task_extra_columns, widen_graph,
)
from src.placement.network_fabric import transmission_hops
from src.policy.tabular.reduced_features import (
    BOTH_DIRECTION_CONTRACTS, EXCHANGE_PUSH_DIM, FOUR_TYPE_CONTRACTS, LOAD_SECONDS_CONTRACTS,
    PARTIAL_STATE_CONTRACT_V3, PARTIAL_STATE_CONTRACT_V4, PARTIAL_STATE_CONTRACT_V5, PartialStateContext,
    krank_node_order, partial_state_columns, partial_state_feature_dim, partial_state_mlp_layout,
    validate_partial_state_contract,
)

NODES = ("n0", "n1", "n2")
REPLICAS = [(n, p) for n in NODES for p in (0, 1)]
BACKLOG = {r: float(i) * 2.0 for i, r in enumerate(REPLICAS)}
MB = 1024 * 1024


def _exchange(asym: bool):
    """(seconds-per-byte, latency) keyed by traversal direction; asym: leaving n0 is 10x slower than entering it."""
    out = {}
    for a in NODES:
        for b in NODES:
            if a == b:
                out[(a, b)] = (0.0, 0.0)
            else:
                spb = 1.0 * (10.0 if (asym and a == "n0") else 1.0)
                out[(a, b)] = (spb, 0.1)
    return out


def _ctx(contract, n_tasks=3, exchange=None, **kw):
    caps = {n: math.inf for n in NODES}
    hop = {n: float(i) for i, n in enumerate(NODES)}
    tasks = list(range(n_tasks))
    extra = {}
    if contract in LOAD_SECONDS_CONTRACTS:
        extra = dict(backlog_s=BACKLOG,
                     service_s={(t, r): 1.0 + t + 0.5 * r[1] for t in tasks for r in REPLICAS})
    extra.update(kw)
    return PartialStateContext(
        node_caps=caps, demand={(t, r): 1.0 for t in tasks for r in REPLICAS},
        node_of={r: r[0] for r in REPLICAS}, task_type_index={t: t % 4 for t in tasks},
        parents={t: [] for t in tasks},
        route_hops_bneck={(a, b): ((0.0, math.inf) if a == b else (1.0, 100.0)) for a in NODES for b in NODES},
        payload_bytes=0.0, transfer_norm=0.0, node_rank=krank_node_order(caps, hop, contract=contract),
        ingress_links={(t, n): () for t in tasks for n in NODES}, core_links=frozenset(),
        peer_pairs={(0, 2): 2.0, (2, 0): 2.0, (1, 2): 3.0, (2, 1): 3.0}, peer_norm=1.0,
        node_exchange=exchange if exchange is not None else _exchange(False),
        cand_nodes={t: list(NODES) for t in tasks}, contract=contract, **extra,
    )


def test_v5_is_valid_27_wide_and_has_no_mlp_layout():
    assert validate_partial_state_contract("partial_state_v5") == PARTIAL_STATE_CONTRACT_V5
    assert partial_state_feature_dim(PARTIAL_STATE_CONTRACT_V3) == 22
    assert partial_state_feature_dim(PARTIAL_STATE_CONTRACT_V4) == 25
    assert partial_state_feature_dim(PARTIAL_STATE_CONTRACT_V5) == 25 + EXCHANGE_PUSH_DIM == 27
    assert PARTIAL_STATE_CONTRACT_V5 in BOTH_DIRECTION_CONTRACTS | FOUR_TYPE_CONTRACTS | LOAD_SECONDS_CONTRACTS
    assert PARTIAL_STATE_CONTRACT_V4 not in BOTH_DIRECTION_CONTRACTS | FOUR_TYPE_CONTRACTS
    with pytest.raises(ValueError, match="no MLP layout"):
        partial_state_mlp_layout(PARTIAL_STATE_CONTRACT_V5)


def test_v5_on_uniform_links_is_v4_in_seconds_plus_a_repeat(monkeypatch):
    monkeypatch.delenv("PARTIAL_STATE_LOAD_SECONDS", raising=False)
    committed = {0: ("n0", 0)}  # task 1, also a partner of task 2, is still unplaced: the peer-mass column is live
    monkeypatch.setenv("PARTIAL_STATE_EXCHANGE_SECONDS", "1")
    v4 = partial_state_columns(_ctx(PARTIAL_STATE_CONTRACT_V4), 2, REPLICAS, committed)
    monkeypatch.delenv("PARTIAL_STATE_EXCHANGE_SECONDS")
    v5 = partial_state_columns(_ctx(PARTIAL_STATE_CONTRACT_V5), 2, REPLICAS, committed)
    assert v5.shape == (len(REPLICAS), 27)
    np.testing.assert_allclose(v5[:, :25], v4, rtol=1e-6)
    np.testing.assert_allclose(v5[:, 25], v5[:, 7], rtol=1e-6)
    np.testing.assert_allclose(v5[:, 26], v5[:, 8], rtol=1e-6)
    assert v5[:, 25].any() and v5[:, 26].any()


def test_pull_and_push_are_different_columns_on_an_asymmetric_pair(monkeypatch):
    monkeypatch.delenv("PARTIAL_STATE_LOAD_SECONDS", raising=False)
    ex = _exchange(asym=True)  # n0 -> anywhere costs 10 s/byte; anywhere -> n0 costs 1 s/byte
    # task 2's committed partner (task 0, payload 2 B) sits on n0; the candidate is on n1
    committed = {0: ("n0", 0)}
    v5 = partial_state_columns(_ctx(PARTIAL_STATE_CONTRACT_V5, exchange=ex), 2, REPLICAS, committed)
    i = REPLICAS.index(("n1", 0))
    pull = 2.0 * ex[("n0", "n1")][0] + 0.1  # the task pulls from its partner: n0 -> n1
    push = 2.0 * ex[("n1", "n0")][0] + 0.1  # the partner pulls from the task: n1 -> n0
    assert pull != push
    assert v5[i, 7] == pytest.approx(math.log1p(pull), rel=1e-6)
    assert v5[i, 25] == pytest.approx(math.log1p(push), rel=1e-6)
    v4 = partial_state_columns(_ctx(PARTIAL_STATE_CONTRACT_V4, exchange=ex), 2, REPLICAS, committed)
    # v4 read candidate -> peer into column 7 (normalised by peer_norm): the push direction under another name
    assert v4[i, 7] == pytest.approx(push / 1.0, rel=1e-6)


def test_committed_service_column_prices_the_pull(monkeypatch):
    monkeypatch.delenv("PARTIAL_STATE_LOAD_SECONDS", raising=False)
    ex = _exchange(asym=True)
    committed = {0: ("n0", 0), 2: ("n1", 1)}  # task 0 on n0 pulls task 2's data from n1; task 2 pulls from n0
    v5 = partial_state_columns(_ctx(PARTIAL_STATE_CONTRACT_V5, exchange=ex), 1, REPLICAS, committed)
    service = {t: 1.0 + t + 0.5 * p for t, p in ((0, 0), (2, 1))}
    want0 = service[0] + (2.0 * ex[("n1", "n0")][0] + 0.1)  # task 0 (n0) pulls from task 2 (n1): n1 -> n0
    want2 = service[2] + (2.0 * ex[("n0", "n1")][0] + 0.1)  # task 2 (n1) pulls from task 0 (n0): n0 -> n1
    for i, r in enumerate(REPLICAS):
        want = {("n0", 0): want0, ("n1", 1): want2}.get(r, 0.0)
        assert v5[i, 23] == pytest.approx(math.log1p(want), rel=1e-6)


def test_v5_refuses_to_serve_without_its_ingredients():
    with pytest.raises(ValueError, match="needs backlog_s and service_s"):
        _ctx(PARTIAL_STATE_CONTRACT_V5, backlog_s=None, service_s=None)


# ---- the W3 case: one cellular server (4 MB/s out, 75 MB/s in) and one wired server --------------------------------

def _w3_topology():
    links = {
        "cell|core": {"latency": 0.01, "bandwidth_mbps": 4.0, "access_node": "cell",
                      "bandwidth_out_mbps": 4.0, "bandwidth_in_mbps": 75.0},
        "core|wired": {"latency": 0.01, "bandwidth_mbps": 117.0},
    }
    routes = {"cell": {"wired": ["cell", "core", "wired"]}, "wired": {"cell": ["wired", "core", "cell"]}}
    return routes, links


def _node_exchange_like_the_cache(routes, links, names):
    """prepare_graphs_cache.attach_dag_partial_state_block's formula, on the same helper."""
    out = {}
    for a in names:
        for b in names:
            if a == b:
                out[(a, b)] = (0.0, 0.0)
                continue
            h, bneck = route_hops_and_bottleneck(routes, links, a, b)
            out[(a, b)] = (transmission_hops(float(h)) / (float(bneck) * MB), 0.02)
    return out


def test_cellular_pull_reads_75_mb_per_s_and_push_reads_4():
    routes, links = _w3_topology()
    assert route_hops_and_bottleneck(routes, links, "wired", "cell")[1] == 75.0  # into the cellular node
    assert route_hops_and_bottleneck(routes, links, "cell", "wired")[1] == 4.0   # out of it
    names = ("cell", "wired")
    ex = _node_exchange_like_the_cache(routes, links, names)
    reps = [("cell", 0), ("wired", 0)]
    tasks = [0, 1]
    caps = {n: math.inf for n in names}
    payload = 40e6

    def ctx(contract):
        kw = dict(backlog_s={r: 0.0 for r in reps}, service_s={(t, r): 1.0 for t in tasks for r in reps})
        return PartialStateContext(
            node_caps=caps, demand={(t, r): 1.0 for t in tasks for r in reps}, node_of={r: r[0] for r in reps},
            task_type_index={0: 0, 1: 1}, parents={0: [], 1: []},
            route_hops_bneck={(a, b): ((0.0, math.inf) if a == b else route_hops_and_bottleneck(routes, links, a, b))
                              for a in names for b in names},
            payload_bytes=0.0, transfer_norm=0.0,
            node_rank=krank_node_order(caps, {n: 1.0 for n in names}, contract=contract),
            ingress_links={(t, n): () for t in tasks for n in names}, core_links=frozenset(),
            peer_pairs={(0, 1): payload, (1, 0): payload}, peer_norm=1.0, node_exchange=ex,
            cand_nodes={t: list(names) for t in tasks}, contract=contract, **kw)

    committed = {0: ("wired", 0)}                     # the partner is on the wired server
    v5 = partial_state_columns(ctx(PARTIAL_STATE_CONTRACT_V5), 1, reps, committed)
    i = reps.index(("cell", 0))                       # the candidate is the cellular server
    pull_s = payload * transmission_hops(2.0) / (75.0 * MB) + 0.02
    push_s = payload * transmission_hops(2.0) / (4.0 * MB) + 0.02
    assert v5[i, 7] == pytest.approx(math.log1p(pull_s), rel=1e-5)   # the task pulls INTO the cellular node: 75 MB/s
    assert v5[i, 25] == pytest.approx(math.log1p(push_s), rel=1e-5)  # the partner pulls OUT of it: 4 MB/s
    assert push_s / pull_s > 10.0
    # v4 read the wrong one for the pull: its column 7 carries the 4 MB/s direction
    v4 = partial_state_columns(ctx(PARTIAL_STATE_CONTRACT_V4), 1, reps, committed)
    assert v4[i, 7] == pytest.approx(push_s / 1.0, rel=1e-5)


CAL_CFG = Path("/home/nikola.lukic/gnn-herosim/simulation_data/workload_fix_v1/inputs/cfg_cal/cc40s9601.json")


@pytest.mark.skipif(not CAL_CFG.exists(), reason="calibration config only on datalab")
def test_topology_9601_with_w3_reads_75_on_the_pull_side():
    import contextlib
    import copy
    import io
    import json

    from src.executesimulation import prepare_infrastructure_for_real_simulation
    from src.placement.network_fabric import DEFAULT_ACCESS_MIX

    cfg = json.loads(CAL_CFG.read_text())
    cfg["network"]["backbone"]["access_classes"] = {"mix": dict(DEFAULT_ACCESS_MIX)}
    with contextlib.redirect_stdout(io.StringIO()):
        inf = json.loads(json.dumps(prepare_infrastructure_for_real_simulation(
            copy.deepcopy(cfg), None, Path(__file__).resolve().parents[1] / "data" / "nofs-ids"), default=str))
    lt = inf["link_topology"]
    classes = {n: s["class"] for n, s in lt["access_classes"].items()}
    cell = next(n for n in classes if not n.startswith("client_node") and classes[n] == "cellular")
    wired = next(n for n in classes if not n.startswith("client_node") and classes[n] == "wired")
    pull = route_hops_and_bottleneck(lt["routes"], lt["links"], wired, cell)[1]
    push = route_hops_and_bottleneck(lt["routes"], lt["links"], cell, wired)[1]
    assert (pull, push) == (75.0, 4.0)


# ---- four-type blocks -----------------------------------------------------------------------------------------------

def test_four_type_extra_columns_are_appended_not_inserted():
    te = task_extra_columns(["dnn1", "dnn2", "rf", "cnn"])
    assert te.shape == (4, TASK_EXTRA_DIM)
    np.testing.assert_array_equal(te, [[0, 0], [0, 0], [1, 0], [0, 1]])
    pe = platform_extra_columns([{"dnn1"}, {"rf", "dnn2"}, {"cnn", "rf"}, set()])
    assert pe.shape == (4, PLATFORM_EXTRA_DIM)
    np.testing.assert_array_equal(pe, [[0, 0], [1, 0], [1, 1], [0, 0]])


def test_widen_graph_appends_and_removes_the_held_apart_columns():
    import torch
    from torch_geometric.data import Data

    g = Data(task_features=torch.ones(4, 3), platform_features=torch.ones(5, 14))
    g.four_type_task_extra = torch.zeros(4, 2)
    g.four_type_platform_extra = torch.zeros(5, 2)
    widen_graph(g)
    assert g.task_features.shape == (4, 5) and g.platform_features.shape == (5, 16)
    assert not hasattr(g, "four_type_task_extra")
    assert torch.equal(g.task_features[:, :3], torch.ones(4, 3))
    with pytest.raises(ValueError, match="no four_type extras"):
        widen_graph(Data(task_features=torch.ones(1, 3), platform_features=torch.ones(1, 14)))
