"""Contract tests for src/placement/topology_features.py.

Two things must hold, and they pull in opposite directions:

  1. `src_index_v0` reproduces the pre-2026-08-18 formula **bit-exactly**, so every
     existing cache, checkpoint and corpus keeps reading what it was built with.
  2. `size_invariant_v1` is genuinely invariant to cluster size — that is the whole
     reason the module exists, and the property the topology-transfer study rests on.

The `--gate-coupled-fraction` episode is the reason these exist: a gate nobody exercised.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.placement.topology_features import (  # noqa: E402
    DEFAULT_TOPOLOGY_FEATURE_CONTRACT,
    TOPOLOGY_FEATURE_CONTRACT_ENV,
    TOPOLOGY_FEATURE_CONTRACT_SIZE_INVARIANT_V1,
    TOPOLOGY_FEATURE_CONTRACT_SRC_INDEX_V0,
    InvalidTopologyFeatureContractError,
    TopologyFeatureContractMismatchError,
    build_source_feature_context,
    require_matching_topology_feature_contract,
    resolve_topology_feature_contract,
)


def _cluster(n_clients: int, n_servers: int, *, reachable_frac: float = 1.0):
    """A synthetic cluster. Each client reaches the first `reachable_frac` of servers."""
    client_names = [f"client_node_{i}" for i in range(n_clients)]
    server_names = [f"server_node_{i}" for i in range(n_servers)]
    node_names = client_names + server_names

    n_reachable = int(round(n_servers * reachable_frac))
    network_maps = {}
    for pos, server in enumerate(server_names):
        # network_maps[server] = {source_name: latency}; a server is reachable from every
        # client only if it is among the first n_reachable.
        network_maps[server] = (
            {c: {"latency": 0.01} for c in client_names} if pos < n_reachable else {}
        )
    return node_names, network_maps, client_names, server_names


# ---------------------------------------------------------------- contract plumbing


def test_default_contract_is_legacy():
    """Existing caches and checkpoints must keep working with no env set."""
    assert DEFAULT_TOPOLOGY_FEATURE_CONTRACT == TOPOLOGY_FEATURE_CONTRACT_SRC_INDEX_V0


def test_resolve_precedence(monkeypatch):
    monkeypatch.delenv(TOPOLOGY_FEATURE_CONTRACT_ENV, raising=False)
    assert resolve_topology_feature_contract() == TOPOLOGY_FEATURE_CONTRACT_SRC_INDEX_V0

    monkeypatch.setenv(TOPOLOGY_FEATURE_CONTRACT_ENV, "size_invariant_v1")
    assert (
        resolve_topology_feature_contract()
        == TOPOLOGY_FEATURE_CONTRACT_SIZE_INVARIANT_V1
    )
    # Explicit argument beats the environment.
    assert (
        resolve_topology_feature_contract("src_index_v0")
        == TOPOLOGY_FEATURE_CONTRACT_SRC_INDEX_V0
    )


def test_unknown_contract_fails_loudly():
    with pytest.raises(InvalidTopologyFeatureContractError):
        resolve_topology_feature_contract("reachability_v2")


def test_mismatch_raises():
    with pytest.raises(TopologyFeatureContractMismatchError):
        require_matching_topology_feature_contract(
            "src_index_v0", "size_invariant_v1", model_label="gnn"
        )
    # A checkpoint that records no contract is not second-guessed.
    require_matching_topology_feature_contract(
        None, "size_invariant_v1", model_label="gnn"
    )


# ---------------------------------------------------------------- v0 bit-exactness


@pytest.mark.parametrize("n_clients,n_servers", [(4, 12), (10, 40), (1, 3)])
def test_v0_reproduces_legacy_formula_exactly(n_clients, n_servers):
    """The historical formula, transcribed from the pre-refactor call sites."""
    node_names, network_maps, _, _ = _cluster(n_clients, n_servers)
    ctx = build_source_feature_context(
        node_names, network_maps, contract="src_index_v0"
    )

    first_idx = {}
    for i, name in enumerate(node_names):
        first_idx.setdefault(name, i)

    for name in node_names:
        expected = float(first_idx[name]) / max(len(node_names), 1)
        assert ctx.feature(name) == expected


def test_v0_unknown_source_falls_back_to_zero():
    """Legacy used `.get(n, 0)`; an unseen source must not raise."""
    node_names, network_maps, _, _ = _cluster(2, 4)
    ctx = build_source_feature_context(
        node_names, network_maps, contract="src_index_v0"
    )
    assert ctx.feature("nonexistent_node") == 0.0


def test_v0_honours_supplied_index_map_and_count():
    """The cache builders pass their own map; it must be used verbatim, not re-derived."""
    node_names, network_maps, _, _ = _cluster(2, 4)
    # A deliberately non-positional map, as gnn_hetero's node.id map can be.
    supplied = {name: 100 + i for i, name in enumerate(node_names)}
    ctx = build_source_feature_context(
        node_names,
        network_maps,
        contract="src_index_v0",
        first_idx_by_name=supplied,
        node_count=7,
    )
    assert ctx.feature(node_names[0]) == 100.0 / 7.0
    assert ctx.feature(node_names[3]) == 103.0 / 7.0


# ---------------------------------------------------------------- v1 size invariance


def test_v1_is_invariant_to_cluster_size():
    """The property the whole transfer study rests on.

    Same reachable *fraction*, wildly different cluster sizes — the feature must not move.
    Under v0 the same comparison shifts, which is the confound being removed.
    """
    values = []
    legacy_values = []
    for n_servers in (10, 20, 40, 80, 160):
        node_names, network_maps, clients, _ = _cluster(
            4, n_servers, reachable_frac=0.5
        )
        ctx_v1 = build_source_feature_context(
            node_names, network_maps, contract="size_invariant_v1"
        )
        ctx_v0 = build_source_feature_context(
            node_names, network_maps, contract="src_index_v0"
        )
        values.append(ctx_v1.feature(clients[1]))
        legacy_values.append(ctx_v0.feature(clients[1]))

    assert len(set(values)) == 1, f"v1 varied with cluster size: {values}"
    assert values[0] == pytest.approx(0.5)
    # The control: v0 does move, which is exactly why it could not be used here.
    assert len(set(legacy_values)) > 1, (
        "v0 was expected to vary with cluster size; if it no longer does, this test's "
        "premise — and the reason size_invariant_v1 exists — needs rechecking."
    )


def test_v1_is_bounded_and_tracks_reachability():
    node_names, network_maps, clients, _ = _cluster(3, 20, reachable_frac=0.25)
    ctx = build_source_feature_context(
        node_names, network_maps, contract="size_invariant_v1"
    )
    val = ctx.feature(clients[0])
    assert 0.0 <= val <= 1.0
    assert val == pytest.approx(0.25)


def test_v1_server_source_reaches_itself():
    """A task sourced on a server can always run there; the network map omits that."""
    node_names, network_maps, _, servers = _cluster(2, 8, reachable_frac=0.0)
    # reachable_frac=0 => no server lists any client, so every map is empty.
    ctx = build_source_feature_context(
        node_names, network_maps, contract="size_invariant_v1"
    )
    # A client reaches nothing...
    assert ctx.feature("client_node_0") == 0.0
    # ...but a server still reaches itself: 1 of 8.
    assert ctx.feature(servers[0]) == pytest.approx(1.0 / 8.0)


def test_v1_unknown_source_is_zero_not_error():
    node_names, network_maps, _, _ = _cluster(2, 6, reachable_frac=1.0)
    ctx = build_source_feature_context(
        node_names, network_maps, contract="size_invariant_v1"
    )
    assert ctx.feature("nonexistent_node") == 0.0


def test_v1_empty_cluster_does_not_divide_by_zero():
    ctx = build_source_feature_context([], {}, contract="size_invariant_v1")
    assert ctx.feature("anything") == 0.0


def test_v1_tolerates_missing_network_map_entry():
    """A node with no network_map key at all must read as unreachable, not crash."""
    node_names = ["client_node_0", "server_node_0", "server_node_1"]
    ctx = build_source_feature_context(
        node_names, {"server_node_0": None}, contract="size_invariant_v1"
    )
    assert ctx.feature("client_node_0") == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
