"""route_a: the parent->child transfer term, and the server mesh it needs.

Why this term exists. Every distance the simulator prices today is indexed by
(source client -> execution node): `network_latency`, the ingress pipe and the link fabric
all charge for getting the REQUEST to a node. None of them can see where a *sibling task*
went. With per-task costs separable and placements freely chosen, the componentwise
minimiser is optimal under any monotone aggregation — so no objective, target or scoring
rule can create structure, and five physics mechanisms have already died proving it. A
parent->child transfer is a pairwise term over two jointly-decided placements, which is the
one shape that breaks separability rather than amplifying it.

The guards here are the ones that decide whether the term is worth anything:

1. **Off by default and inert without dependencies.** Every existing corpus is single-task
   applications; they must be bit-identical whether the flag is set or not.
2. **It actually depends on the parent's node.** A term that charged the same wherever the
   parent ran would be another additive column, which is exactly the failure mode of the
   five previous mechanisms.
3. **Fan-in reads every parent**, not `dependencies[-1]` — the storage branch's long-
   standing FIXME, which silently drops all but one parent.
4. **Missing reachability fails loud**, never charges 0.0. A free bad placement is worse
   than a crash: it is the signal, inverted.
5. **The server mesh is opt-in**, so no existing infrastructure changes.

Run: pipenv run python3 -m pytest tests/test_data_locality_cost.py -q
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Dict, List

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.generate_infrastructure import build_server_mesh  # noqa: E402
from src.placement.infrastructure import Platform  # noqa: E402

APP = "nofs-dag4"
OUTPUT_BYTES = 8_000_000
BANDWIDTH_MBPS = 100.0


class FakeNode:
    def __init__(self, name: str, network_map: Dict[str, float]):
        self.node_name = name
        self.network_map = network_map
        self.network = {"bandwidth": BANDWIDTH_MBPS}


class FakePlatform:
    def __init__(self, node: FakeNode):
        self.node = node


class FakeApplication:
    type = {"name": APP}


class FakeTask:
    def __init__(self, name: str, node: FakeNode | None, dependencies: List["FakeTask"] | None = None):
        self.type = {"name": name, "stateSize": {APP: {"input": 153_600, "output": OUTPUT_BYTES}}}
        self.platform = FakePlatform(node) if node is not None else None
        self.dependencies = dependencies or []
        self.application = FakeApplication()

    def __repr__(self):
        return f"FakeTask({self.type['name']})"


def transfer_time(child_node: FakeNode, task: FakeTask) -> float:
    """Call the real method with a Platform whose only live attribute is `node`."""
    platform = Platform.__new__(Platform)
    platform.node = child_node
    return Platform._dependency_transfer_time(platform, task)


@pytest.fixture
def data_locality_on(monkeypatch):
    monkeypatch.setenv("HEROSIM_DATA_LOCALITY", "1")


def expected(latency: float) -> float:
    return OUTPUT_BYTES / (BANDWIDTH_MBPS * 1024 * 1024) + latency


# --------------------------------------------------------------------------------------
# 1. default-off and inert
# --------------------------------------------------------------------------------------


def test_disabled_by_default_costs_nothing(monkeypatch):
    monkeypatch.delenv("HEROSIM_DATA_LOCALITY", raising=False)
    node_a, node_b = FakeNode("node1", {"node2": 0.05}), FakeNode("node2", {"node1": 0.05})
    parent = FakeTask("dnn1", node_a)
    child = FakeTask("dnn2", node_b, [parent])
    assert transfer_time(node_b, child) == 0.0


def test_a_task_with_no_dependencies_costs_nothing_even_when_enabled(data_locality_on):
    node = FakeNode("node1", {})
    assert transfer_time(node, FakeTask("dnn1", node)) == 0.0


# --------------------------------------------------------------------------------------
# 2. the term is a function of WHERE THE PARENT WENT — the whole point
# --------------------------------------------------------------------------------------


def test_a_co_located_parent_is_free(data_locality_on):
    node = FakeNode("node1", {"node2": 0.05})
    parent = FakeTask("dnn1", node)
    child = FakeTask("dnn2", node, [parent])
    assert transfer_time(node, child) == 0.0, "same-node parent needs no network transfer"


def test_cost_varies_with_the_parents_node(data_locality_on):
    """Two placements of the CHILD held fixed; only the parent moves. If this were flat,
    the term would be one more additive column and route A would be pointless."""
    child_node = FakeNode("node1", {"node2": 0.010, "node3": 0.200})
    near_parent = FakeTask("dnn1", FakeNode("node2", {}))
    far_parent = FakeTask("dnn1", FakeNode("node3", {}))

    near = transfer_time(child_node, FakeTask("dnn2", child_node, [near_parent]))
    far = transfer_time(child_node, FakeTask("dnn2", child_node, [far_parent]))

    assert near == pytest.approx(expected(0.010))
    assert far == pytest.approx(expected(0.200))
    assert far > near, "the term must depend on the parent's placement"


def test_cost_scales_with_state_size(data_locality_on):
    """The lever: the coupled term scales with stateSize while queue work does not."""
    child_node = FakeNode("node1", {"node2": 0.010})
    parent = FakeTask("dnn1", FakeNode("node2", {}))
    child = FakeTask("dnn2", child_node, [parent])
    base = transfer_time(child_node, child)

    child.dependencies[0].type["stateSize"][APP]["output"] = OUTPUT_BYTES * 10
    scaled = transfer_time(child_node, child)

    transfer_only_base = base - 0.010
    transfer_only_scaled = scaled - 0.010
    assert transfer_only_scaled == pytest.approx(transfer_only_base * 10)


# --------------------------------------------------------------------------------------
# 3. fan-in reads every parent
# --------------------------------------------------------------------------------------


def test_fan_in_charges_every_parent_not_just_the_last(data_locality_on):
    child_node = FakeNode("node1", {"node2": 0.010, "node3": 0.020})
    parents = [FakeTask("dnn1", FakeNode("node2", {})), FakeTask("rf", FakeNode("node3", {}))]
    child = FakeTask("cnn", child_node, parents)

    got = transfer_time(child_node, child)
    assert got == pytest.approx(expected(0.010) + expected(0.020)), (
        "a fan-in child must pay for both parents; dependencies[-1] would charge one"
    )


def test_fan_in_skips_only_the_co_located_parent(data_locality_on):
    child_node = FakeNode("node1", {"node3": 0.020})
    parents = [FakeTask("dnn1", child_node), FakeTask("rf", FakeNode("node3", {}))]
    child = FakeTask("cnn", child_node, parents)
    assert transfer_time(child_node, child) == pytest.approx(expected(0.020))


# --------------------------------------------------------------------------------------
# 4. failures are loud
# --------------------------------------------------------------------------------------


def test_unreachable_parent_raises_rather_than_charging_zero(data_locality_on):
    child_node = FakeNode("node1", {})  # no entry for node2
    parent = FakeTask("dnn1", FakeNode("node2", {}))
    with pytest.raises(RuntimeError, match="no network_map entry"):
        transfer_time(child_node, FakeTask("dnn2", child_node, [parent]))


def test_unplaced_parent_raises(data_locality_on):
    child_node = FakeNode("node1", {"node2": 0.01})
    parent = FakeTask("dnn1", None)
    with pytest.raises(RuntimeError, match="no platform"):
        transfer_time(child_node, FakeTask("dnn2", child_node, [parent]))


def test_non_positive_bandwidth_raises(data_locality_on):
    child_node = FakeNode("node1", {"node2": 0.01})
    child_node.network = {"bandwidth": 0.0}
    parent = FakeTask("dnn1", FakeNode("node2", {}))
    with pytest.raises(RuntimeError, match="non-positive"):
        transfer_time(child_node, FakeTask("dnn2", child_node, [parent]))


# --------------------------------------------------------------------------------------
# 5. the server mesh
# --------------------------------------------------------------------------------------


def _nodes():
    return [
        {"node_name": "client_node0", "type": "rpi"},
        {"node_name": "node1", "type": "xavier"},
        {"node_name": "node2", "type": "xavier"},
        {"node_name": "node3", "type": "nuc"},
    ]


def _maps(nodes):
    return {n["node_name"]: {} for n in nodes}


def test_server_mesh_is_opt_in():
    nodes, maps = _nodes(), None
    maps = _maps(nodes)
    added = build_server_mesh(maps, nodes, {"network": {}}, random.Random(0))
    assert added == 0
    assert all(peers == {} for peers in maps.values()), "no config key must mean no change"


def test_server_mesh_connects_every_server_pair_and_no_clients():
    nodes = _nodes()
    maps = _maps(nodes)
    added = build_server_mesh(maps, nodes, {"network": {"server_mesh": True}}, random.Random(0))

    servers = [n["node_name"] for n in nodes if not n["node_name"].startswith("client_node")]
    assert added == 3, "3 servers => 3 undirected pairs"
    for i, left in enumerate(servers):
        for right in servers[i + 1:]:
            assert right in maps[left] and left in maps[right], f"{left}<->{right} missing"
            assert maps[left][right] == maps[right][left], "latency must be symmetric"

    # Client adjacency is what candidate filtering reads; the mesh must not touch it.
    assert maps["client_node0"] == {}
    for server in servers:
        assert "client_node0" not in maps[server]


def test_server_mesh_preserves_existing_edges():
    nodes = _nodes()
    maps = _maps(nodes)
    maps["node1"]["node2"] = 0.999
    maps["node2"]["node1"] = 0.999
    build_server_mesh(maps, nodes, {"network": {"server_mesh": True}}, random.Random(0))
    assert maps["node1"]["node2"] == 0.999, "an existing edge must not be overwritten"


def test_server_mesh_is_deterministic_for_a_seed():
    results = []
    for _ in range(2):
        nodes, maps = _nodes(), None
        maps = _maps(nodes)
        build_server_mesh(maps, nodes, {"network": {"server_mesh": True}}, random.Random(1234))
        results.append({k: dict(v) for k, v in maps.items()})
    assert results[0] == results[1]
