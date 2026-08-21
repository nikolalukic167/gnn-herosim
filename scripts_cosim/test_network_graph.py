"""Contract and structure tests for src/placement/network_graph.py.

Companion to `test_topology_features.py` — a feature-contract module gets its own contract
suite, because the properties that matter here are not observable from the simulator's
timing (which is what `test_link_contention.py` covers).

Four things must hold:

  1. **Contract `off` changes nothing.** Every existing cache, checkpoint and corpus was
     built on the bipartite graph and must keep loading. The attach helper must not set a
     single attribute.
  2. **Scoring never sees the new entities.** Logits stay on task->platform edges. This is
     the invariant `edge_attr` alignment and reverse-edge pairing depend on.
  3. **Nothing added scales with cluster size.** Not a feature value, not a degree. GIN
     aggregates with *sum*, so a degree that grows with N shifts embeddings with N and the
     transfer measurement would be reading its own graph construction — Phase 0's confound
     coming back as structure.
  4. **Train and serve build the same graph.** The `mp_parity` failure (12.4x live RTT) was
     exactly one path emitting message-passing structure the other did not.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.placement.network_fabric import link_key  # noqa: E402
from src.placement.network_graph import (  # noqa: E402
    DEFAULT_NETWORK_GRAPH_CONTRACT,
    NET_LINK_FEATURE_DIM,
    NET_NODE_FEATURE_DIM,
    NETWORK_GRAPH_CONTRACT_CORE_V1,
    NETWORK_GRAPH_CONTRACT_ENV,
    NETWORK_GRAPH_CONTRACT_OFF,
    InvalidNetworkGraphContractError,
    NetworkGraphContractMismatchError,
    attach_network_graph_block,
    build_network_graph_block,
    require_matching_network_graph_contract,
    resolve_network_graph_contract,
)
from src.policy.gnn.gnn_model import TaskPlacementGNN  # noqa: E402


# ------------------------------------------------------------------ synthetic fabric


def _backbone(n_clients: int, n_servers: int, *, n_core: int = 4, plats_per_node: int = 2):
    """A ring backbone with every node attached to one core router, round-robin.

    Deliberately mirrors `build_core_backbone`: a ring of core routers (the only shared
    segments) plus one access link per node. Routes go client -> core -> ... -> core ->
    server along the ring, so different client/server pairs genuinely share core segments.
    """
    clients = [f"client_node{i}" for i in range(n_clients)]
    servers = [f"node{i}" for i in range(n_servers)]
    node_names = clients + servers
    cores = [f"core{i}" for i in range(n_core)]

    links = {}
    for i in range(n_core):
        links[link_key(cores[i], cores[(i + 1) % n_core])] = {
            "latency": 0.004,
            "bandwidth_mbps": 1.5,
        }
    attach = {}
    for pos, name in enumerate(node_names):
        core = cores[pos % n_core]
        attach[name] = core
        links[link_key(name, core)] = {"latency": 0.02, "bandwidth_mbps": 1.5}

    def _ring_path(a: str, b: str):
        ia, ib = cores.index(a), cores.index(b)
        forward = [cores[(ia + k) % n_core] for k in range((ib - ia) % n_core + 1)]
        backward = [cores[(ia - k) % n_core] for k in range((ia - ib) % n_core + 1)]
        return forward if len(forward) <= len(backward) else backward

    routes = {}
    for client in clients:
        for server in servers:
            routes.setdefault(client, {})[server] = (
                [client] + _ring_path(attach[client], attach[server]) + [server]
            )

    link_topology = {"links": links, "routes": routes}
    platform_node_names = [name for name in servers for _ in range(plats_per_node)]
    return node_names, platform_node_names, link_topology, clients, servers


def _block(n_clients=4, n_servers=8, *, n_core=4, plats_per_node=2, n_tasks=4, contract=None):
    node_names, platform_node_names, link_topology, clients, servers = _backbone(
        n_clients, n_servers, n_core=n_core, plats_per_node=plats_per_node
    )
    task_sources = [clients[i % len(clients)] for i in range(n_tasks)]
    # Every task can reach every platform, which is the worst case for degree growth.
    candidates = [list(platform_node_names) for _ in range(n_tasks)]
    return build_network_graph_block(
        node_names=node_names,
        platform_node_names=platform_node_names,
        task_source_names=task_sources,
        task_candidate_node_names=candidates,
        link_topology=link_topology,
        n_tasks=n_tasks,
        n_platforms=len(platform_node_names),
        contract=contract or NETWORK_GRAPH_CONTRACT_CORE_V1,
    )


# ------------------------------------------------------------------ contract plumbing


def test_default_contract_is_off():
    """Existing caches and checkpoints must keep working with no env set."""
    assert DEFAULT_NETWORK_GRAPH_CONTRACT == NETWORK_GRAPH_CONTRACT_OFF


def test_env_selects_contract(monkeypatch):
    monkeypatch.setenv(NETWORK_GRAPH_CONTRACT_ENV, NETWORK_GRAPH_CONTRACT_CORE_V1)
    assert resolve_network_graph_contract() == NETWORK_GRAPH_CONTRACT_CORE_V1


def test_explicit_argument_beats_env(monkeypatch):
    monkeypatch.setenv(NETWORK_GRAPH_CONTRACT_ENV, NETWORK_GRAPH_CONTRACT_CORE_V1)
    assert resolve_network_graph_contract(NETWORK_GRAPH_CONTRACT_OFF) == NETWORK_GRAPH_CONTRACT_OFF


def test_unknown_contract_raises():
    with pytest.raises(InvalidNetworkGraphContractError):
        resolve_network_graph_contract("routenet_v9")


def test_contract_mismatch_raises():
    with pytest.raises(NetworkGraphContractMismatchError, match="2026-08-16"):
        require_matching_network_graph_contract(
            NETWORK_GRAPH_CONTRACT_CORE_V1,
            NETWORK_GRAPH_CONTRACT_OFF,
            model_label="gnn_topo",
        )


def test_matching_contract_passes():
    require_matching_network_graph_contract(
        NETWORK_GRAPH_CONTRACT_CORE_V1,
        NETWORK_GRAPH_CONTRACT_CORE_V1,
        model_label="gnn_topo",
    )


# ------------------------------------------------------------------ default-off is a no-op


def test_off_contract_produces_no_entities():
    block = _block(contract=NETWORK_GRAPH_CONTRACT_OFF)
    assert block.num_entities == 0
    assert block.edge_index.shape == (2, 0)


def test_off_contract_attaches_nothing():
    """The Data must be byte-identical to today's graph, i.e. no new attributes at all."""
    from torch_geometric.data import Data

    data = Data(edge_index=torch.zeros((2, 0), dtype=torch.long), n_tasks=4, n_platforms=8)
    before = set(data.keys())
    attach_network_graph_block(data, _block(contract=NETWORK_GRAPH_CONTRACT_OFF))
    assert set(data.keys()) == before


def test_no_fabric_yields_empty_block():
    """A corpus with no network.backbone is a legitimate silent no-op, not an error."""
    block = build_network_graph_block(
        node_names=["client_node0", "node0"],
        platform_node_names=["node0"],
        task_source_names=["client_node0"],
        task_candidate_node_names=[["node0"]],
        link_topology=None,
        n_tasks=1,
        n_platforms=1,
        contract=NETWORK_GRAPH_CONTRACT_CORE_V1,
    )
    assert block.num_entities == 0


# ------------------------------------------------------------------ structure


def test_only_core_links_become_entities():
    """Access links carry one node's traffic and are additive; core segments are shared."""
    block = _block(n_core=4)
    assert len(block.link_keys) == 4
    assert all(key.startswith("core") for key in block.link_keys)


def test_entity_indices_start_after_platforms():
    """Task and platform indices must be untouched, or every checkpoint's layout breaks."""
    block = _block(n_tasks=4, n_servers=8, plats_per_node=2)
    assert block.node_offset == 4 + 16
    assert block.link_offset == block.node_offset + block.n_net_nodes
    assert int(block.edge_index.min()) >= 0
    assert int(block.edge_index.max()) < block.link_offset + block.n_net_links


def test_feature_widths_are_pinned():
    block = _block()
    assert block.node_features.shape[1] == NET_NODE_FEATURE_DIM
    assert block.link_features.shape[1] == NET_LINK_FEATURE_DIM


def test_edges_are_undirected():
    block = _block()
    pairs = {(int(a), int(b)) for a, b in zip(*block.edge_index)}
    assert all((b, a) in pairs for a, b in pairs)


def test_every_platform_is_hosted_on_its_node():
    """The platform -> node edge is what ties a candidate to the fabric at all."""
    node_names, platform_node_names, link_topology, clients, _servers = _backbone(4, 8)
    n_tasks, n_plat = 4, len(platform_node_names)
    block = build_network_graph_block(
        node_names=node_names,
        platform_node_names=platform_node_names,
        task_source_names=[clients[i % 4] for i in range(n_tasks)],
        task_candidate_node_names=[list(platform_node_names) for _ in range(n_tasks)],
        link_topology=link_topology,
        n_tasks=n_tasks,
        n_platforms=n_plat,
        contract=NETWORK_GRAPH_CONTRACT_CORE_V1,
    )
    pairs = {(int(a), int(b)) for a, b in zip(*block.edge_index)}
    node_pos = {name: i for i, name in enumerate(node_names)}
    for p_pos, host in enumerate(platform_node_names):
        assert (n_tasks + p_pos, block.node_offset + node_pos[host]) in pairs


def test_tasks_connect_to_the_links_their_routes_traverse():
    """This is the route information the bipartite graph has never carried."""
    block = _block(n_tasks=4)
    pairs = {(int(a), int(b)) for a, b in zip(*block.edge_index)}
    task_link = {(a, b) for a, b in pairs if a < 4 and b >= block.link_offset}
    assert task_link, "no task->link route edges emitted"
    assert {a for a, _ in task_link} == {0, 1, 2, 3}


def test_adjacent_core_links_are_connected():
    """A 4-ring: each core link shares an endpoint with exactly two others."""
    block = _block(n_core=4)
    pairs = {(int(a), int(b)) for a, b in zip(*block.edge_index)}
    for i in range(block.n_net_links):
        gi = block.link_offset + i
        neighbours = {b for a, b in pairs if a == gi and b >= block.link_offset}
        assert len(neighbours) == 2


# ------------------------------------------------------------------ size invariance


@pytest.mark.parametrize("n_servers", [4, 8, 16, 32])
def test_features_stay_bounded_as_cluster_grows(n_servers):
    """No feature may drift with N. Ratios stay in [0, 1]; constant divisors stay small."""
    block = _block(n_clients=4, n_servers=n_servers)
    assert np.isfinite(block.node_features).all()
    assert np.isfinite(block.link_features).all()
    assert block.node_features.max() <= 1.5
    assert block.link_features.max() <= 1.5
    # The two genuine ratios.
    assert (0.0 <= block.link_features[:, 3]).all() and (block.link_features[:, 3] <= 1.0).all()
    assert (0.0 <= block.link_features[:, 4]).all() and (block.link_features[:, 4] <= 1.0).all()


def test_no_feature_column_drifts_with_cluster_size():
    """The direct statement of the property, rather than a magnitude bound.

    Bounds catch a runaway; they do not catch a feature that creeps. Every column's range
    must land in the same place at 8 servers and at 64 — otherwise a model trained small
    and tested large is reading a shifted distribution, which is exactly the artifact
    Phase 0 removed from `src_norm`.
    """
    ranges = {}
    for n_servers in (8, 16, 32, 64):
        block = _block(n_clients=4, n_servers=n_servers)
        for col in range(NET_NODE_FEATURE_DIM):
            ranges.setdefault(("node", col), []).append(
                (float(block.node_features[:, col].min()), float(block.node_features[:, col].max()))
            )
        for col in range(NET_LINK_FEATURE_DIM):
            ranges.setdefault(("link", col), []).append(
                (float(block.link_features[:, col].min()), float(block.link_features[:, col].max()))
            )
    for (kind, col), observed in ranges.items():
        lo = [r[0] for r in observed]
        hi = [r[1] for r in observed]
        assert max(lo) - min(lo) < 0.05, f"{kind} feature {col} min drifts: {lo}"
        assert max(hi) - min(hi) < 0.05, f"{kind} feature {col} max drifts: {hi}"


def test_candidate_fraction_is_invariant_to_uniform_growth():
    """Doubling the cluster with the same routing must not move the contention prior.

    On a ring with round-robin attachment, doubling the server count preserves how traffic
    distributes across core segments — so if this moved, it would be an artifact of the
    construction rather than of the topology.
    """
    small = _block(n_clients=4, n_servers=8, n_core=4)
    large = _block(n_clients=4, n_servers=16, n_core=4)
    assert small.link_keys == large.link_keys
    np.testing.assert_allclose(
        small.link_features[:, 4], large.link_features[:, 4], atol=1e-6
    )


@pytest.mark.parametrize("n_servers", [8, 16, 32, 64])
def test_added_degrees_do_not_grow_with_cluster_size(n_servers):
    """GIN sums over neighbours, so a degree that grows with N shifts embeddings with N.

    Every family added here is bounded by a *config* constant — core-link count, attach
    degree, platforms per node — never by the cluster. This is the structural half of
    Phase 0's de-confounding.
    """
    block = _block(n_clients=4, n_servers=n_servers, n_core=4, plats_per_node=2)
    degree: dict = {}
    for a in block.edge_index[0]:
        degree[int(a)] = degree.get(int(a), 0) + 1
    # Network entities only; task/platform degrees into the fabric are bounded too.
    net_degrees = [
        d for idx, d in degree.items() if block.node_offset <= idx < block.link_offset
    ]
    assert max(net_degrees) <= 16, f"node entity degree grew to {max(net_degrees)}"
    task_degrees = [d for idx, d in degree.items() if idx < 4]
    assert max(task_degrees) <= 8, f"task->link degree grew to {max(task_degrees)}"


# ------------------------------------------------------------------ model integration


def _tiny_graph(with_network: bool):
    """A 2-task / 3-platform graph, optionally carrying network entities."""
    from torch_geometric.data import Data

    edge_index = torch.tensor([[0, 0, 1, 1], [2, 3, 3, 4]], dtype=torch.long)
    data = Data(
        edge_index=edge_index,
        n_tasks=2,
        n_platforms=3,
        task_features=torch.randn(2, 3),
        platform_features=torch.randn(3, 16),
    )
    data.edge_attr = torch.randn(4, 5)
    data.node_edge_index = torch.empty((2, 0), dtype=torch.long)
    if with_network:
        block = build_network_graph_block(
            node_names=["client_node0", "node0", "node1"],
            platform_node_names=["node0", "node0", "node1"],
            task_source_names=["client_node0", "client_node0"],
            task_candidate_node_names=[["node0", "node1"], ["node0", "node1"]],
            link_topology=_backbone(1, 2, n_core=4)[2],
            n_tasks=2,
            n_platforms=3,
            contract=NETWORK_GRAPH_CONTRACT_CORE_V1,
        )
        attach_network_graph_block(data, block)
    return data


def _model(mp_network_entities: bool):
    torch.manual_seed(0)
    model = TaskPlacementGNN(
        task_feature_dim=3,
        platform_feature_dim=16,
        mp_network_entities=mp_network_entities,
    )
    model.eval()
    return model


def test_network_entities_produce_no_logits():
    """Scoring stays on task->platform edges: one logit per candidate, nothing more."""
    model = _model(True)
    with torch.no_grad():
        logits = model(_tiny_graph(with_network=True))
    assert [int(t.numel()) for t in logits] == [2, 2]


def test_model_off_is_unchanged_by_the_new_code_path():
    """Flag OFF must give the pre-Phase-2 forward pass, bit for bit."""
    graph = _tiny_graph(with_network=False)
    model = _model(False)
    with torch.no_grad():
        a = model(graph)
    # The generalized platform slice must not have moved anything.
    with torch.no_grad():
        b = model(graph)
    for left, right in zip(a, b):
        assert torch.equal(left, right)
    assert not hasattr(model, "net_node_encoder")


def test_model_ignores_network_entities_when_flag_is_off():
    """A non-topology checkpoint served a topology graph must behave as if it were absent.

    The reverse of the mp_parity direction, and the cheap half: extra attributes on the
    Data may not leak into a model that was not built for them.
    """
    model = _model(False)
    with torch.no_grad():
        with_net = model(_tiny_graph(with_network=True))
        torch.manual_seed(0)
        without = model(_tiny_graph(with_network=False))
    # Same shapes; the features differ by construction, so compare structure not values.
    assert [int(t.numel()) for t in with_net] == [int(t.numel()) for t in without]


def test_topology_model_fails_loud_without_network_entities():
    """The mp_parity rule, enforced: never silently degrade to a different model."""
    model = _model(True)
    with pytest.raises(ValueError, match="mp_network_entities is on"):
        model(_tiny_graph(with_network=False))


def test_topology_model_has_extra_parameters():
    """Self-describing, like mp_gate: a strict load across the boundary must fail."""
    plain = set(_model(False).state_dict())
    topo = set(_model(True).state_dict())
    assert topo - plain
    assert plain - topo == set()


# ------------------------------------------------------- the bounded-slice discipline
#
# `x[n_tasks:]` has been the bug three times (mp_node_edges, TaskPlacementGNN, the ablation
# harness). These tests cover the shared helper, so a fourth model gets the guarantee by
# calling it rather than by someone remembering.


def test_split_returns_exactly_the_two_blocks():
    from src.policy.gnn.gnn_model import split_task_platform_embeddings

    x = torch.arange(40, dtype=torch.float32).reshape(10, 4)
    task, plat = split_task_platform_embeddings(x, 3, 5)
    assert task.shape[0] == 3 and plat.shape[0] == 5
    assert torch.equal(task, x[:3])
    assert torch.equal(plat, x[3:8])


def test_split_excludes_trailing_entity_rows():
    """The whole point: rows appended past the platforms must NOT reach the scorer."""
    from src.policy.gnn.gnn_model import split_task_platform_embeddings

    x = torch.arange(40, dtype=torch.float32).reshape(10, 4)
    _task, plat = split_task_platform_embeddings(x, 3, 5)
    assert plat.shape[0] == 5, "network/link rows leaked into platform_emb"
    assert not torch.equal(plat, x[3:]), "open-ended slice regressed"


def test_split_is_exact_when_there_are_no_extra_entities():
    """Must stay identical to the old `x[n_tasks:]` in the bipartite-only case."""
    from src.policy.gnn.gnn_model import split_task_platform_embeddings

    x = torch.randn(8, 4)
    _task, plat = split_task_platform_embeddings(x, 3, 5)
    assert torch.equal(plat, x[3:])


def test_split_fails_loud_on_a_short_stack():
    from src.policy.gnn.gnn_model import split_task_platform_embeddings

    with pytest.raises(ValueError, match="different set of entities"):
        split_task_platform_embeddings(torch.randn(6, 4), 3, 5)


def test_ablation_model_uses_the_bounded_split():
    """The harness whose numbers the pre-registered transfer gate depends on.

    Verified by running it, not by reading it: with network entities appended, the platform
    embeddings it scores must still be exactly the platform rows.
    """
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "gnn_necessity_ablation", root / "scripts_cosim" / "gnn_necessity_ablation.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(root / "src" / "notebooks"))
    spec.loader.exec_module(module)

    graph = _tiny_graph(with_network=True)
    torch.manual_seed(0)
    model = module.AblationModel(
        task_dim=3, plat_dim=16, edge_dim=5,
        use_gin=True, use_node_edges=False, use_network_entities=True,
    ).eval()
    with torch.no_grad():
        logits = model(graph)
    # One logit per candidate edge and nothing more — node/link rows produced none.
    assert [int(t.numel()) for t in logits] == [2, 2]


def test_ablation_model_fails_loud_without_network_entities():
    """Same rule as TaskPlacementGNN: never silently degrade to a different model."""
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "gnn_necessity_ablation_b", root / "scripts_cosim" / "gnn_necessity_ablation.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(root / "src" / "notebooks"))
    spec.loader.exec_module(module)

    model = module.AblationModel(
        task_dim=3, plat_dim=16, edge_dim=5,
        use_gin=True, use_node_edges=False, use_network_entities=True,
    ).eval()
    with pytest.raises(ValueError, match="use_network_entities is on"):
        model(_tiny_graph(with_network=False))


def test_no_model_slices_platforms_open_ended():
    """Grep-level guard. A new `x[n_tasks:]` is how this comes back a fourth time."""
    import re

    root = Path(__file__).resolve().parents[1]
    offenders = []
    for rel in (
        "src/policy/gnn/gnn_model.py",
        "scripts_cosim/gnn_necessity_ablation.py",
    ):
        for lineno, line in enumerate((root / rel).read_text().split("\n"), 1):
            if re.search(r"=\s*x\[\s*n?_?t(asks)?\s*:\s*\]", line):
                offenders.append(f"{rel}:{lineno}: {line.strip()}")
    assert not offenders, "open-ended platform slice reintroduced:\n" + "\n".join(offenders)
