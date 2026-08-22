"""Network entities for the placement graph — single source of truth for cache and live.

Why this exists
---------------
`build_pyg_inference_graph` builds `n_tasks + n_platforms` nodes joined by bipartite
task<->candidate-platform edges. There are **zero node<->node link edges**: the network
enters only as a static scalar (`latency`) in the 5-dim `edge_attr`. A model whose graph
never contains the network cannot be claimed to transfer across topologies — RouteNet
transfers because its graph *is* the network. So this module adds the missing entities:

    [tasks] --candidate--> [platforms] --hosted_on--> [nodes]
                                          [nodes] --routes_via--> [links]
                                          [tasks] --routes_via--> [links]
                                          [links] --adjacent-----> [links]

Scoring never moves. Logits stay on task->platform edges only; these entities exist to be
message-passed over, which is why they are returned as a *separate* edge index that the
model concatenates for the GIN pass and nothing else ever sees. `edge_attr` alignment and
the `to_undirected` reverse-edge pairing are untouched.

Two design choices carry the whole lineage
------------------------------------------
**Only core links become entities** (`link_scope="core"`, the default). Per
`network_fabric.py`, access links carry one node's traffic and are perfectly additive; the
core segments are the only shared, contended objects. Including access links would add one
entity per node, and — worse — attaching core *routers* as node entities would give each
router a degree that grows with the cluster (every node attaches to one). `GIN` aggregates
with **sum**, so a degree that grows with N shifts embedding magnitudes with N. That is
Phase 0's confound reappearing as structure: a size-transfer measurement would then be
reading its own graph construction. Under `core`, every added degree is bounded by
`n_core` / `attach_degree` / platforms-per-node — all config constants, all independent of
cluster size.

**Every feature is a ratio or has a constant divisor.** No node counts, no raw indices, no
`/ len(nodes)`. See `topology_features.py` for the feature this rule was written for.

Contracts, as in `queue_features.py` and `topology_features.py`:

  off       no network entities; the graph is byte-identical to the pre-2026-08-18 one.
            The default, so every existing cache, checkpoint and corpus is unaffected.
  core_v1   core-link entities plus the four edge families above.

A checkpoint trained under one must never be served the other — that mismatch is the
`mp_parity` failure (12.4x live RTT), and here it would also change tensor shapes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.placement.network_fabric import is_core_link, route_links

NETWORK_GRAPH_CONTRACT_OFF = "off"
NETWORK_GRAPH_CONTRACT_CORE_V1 = "core_v1"
VALID_NETWORK_GRAPH_CONTRACTS = frozenset(
    {NETWORK_GRAPH_CONTRACT_OFF, NETWORK_GRAPH_CONTRACT_CORE_V1}
)
DEFAULT_NETWORK_GRAPH_CONTRACT = NETWORK_GRAPH_CONTRACT_OFF
NETWORK_GRAPH_CONTRACT_ENV = "NETWORK_GRAPH_CONTRACT"

# Feature widths are pinned: the model builds encoders against them, so a silent change
# would load a checkpoint whose first Linear has the wrong fan-in.
NET_NODE_FEATURE_DIM = 6
NET_LINK_FEATURE_DIM = 5

# Constant divisors. Fixed numbers, never derived from the cluster — that is the point.
_PLATFORMS_PER_NODE_SCALE = 10.0
_CORE_LINKS_PER_NODE_SCALE = 8.0
# Access latency runs ~20ms and core ~4ms, so 10x puts both in the same O(0.1) band as
# every other feature here. Nothing depends on the exact value; keeping the encoder's
# inputs comparable in magnitude does.
_LATENCY_SCALE = 10.0
_BANDWIDTH_SCALE = 10.0


class InvalidNetworkGraphContractError(ValueError):
    """Raised when a network graph contract name is not recognized."""


class NetworkGraphContractMismatchError(ValueError):
    """Raised when a checkpoint's training contract differs from the serving contract."""


def validate_network_graph_contract(contract: str) -> str:
    normalized = str(contract).strip().lower()
    if normalized not in VALID_NETWORK_GRAPH_CONTRACTS:
        raise InvalidNetworkGraphContractError(
            f"Unknown network graph contract {contract!r}; "
            f"expected one of {sorted(VALID_NETWORK_GRAPH_CONTRACTS)}"
        )
    return normalized


def resolve_network_graph_contract(explicit: Optional[str] = None) -> str:
    """Explicit argument wins, then $NETWORK_GRAPH_CONTRACT, then off."""
    if explicit is not None and str(explicit).strip():
        return validate_network_graph_contract(explicit)
    from_env = os.environ.get(NETWORK_GRAPH_CONTRACT_ENV, "").strip()
    if from_env:
        return validate_network_graph_contract(from_env)
    return DEFAULT_NETWORK_GRAPH_CONTRACT


def require_matching_network_graph_contract(
    trained_contract: Optional[str], serving_contract: str, *, model_label: str
) -> None:
    """Fail loudly rather than serve a checkpoint a graph it was never trained on."""
    serving = validate_network_graph_contract(serving_contract)
    if trained_contract is None:
        return
    trained = validate_network_graph_contract(trained_contract)
    if trained != serving:
        raise NetworkGraphContractMismatchError(
            f"{model_label} was trained under network graph contract {trained!r} but the "
            f"run resolves to {serving!r}. Set {NETWORK_GRAPH_CONTRACT_ENV}={trained} (or "
            f"load a checkpoint trained under the serving contract); the contracts differ "
            f"in which entities exist in the graph, and serving a checkpoint entities it "
            f"never trained on is the 2026-08-16 message-passing regression exactly."
        )


@dataclass(frozen=True)
class NetworkGraphBlock:
    """Network entities and the edges joining them to the task/platform graph.

    Global index layout, continuing the existing `tasks | platforms` convention::

        [0, n_tasks)                                     tasks
        [n_tasks, n_tasks + n_platforms)                 platforms
        [node_offset, node_offset + n_net_nodes)         physical nodes
        [link_offset, link_offset + n_net_links)         links

    `edge_index` is undirected (both directions emitted) and references those global
    indices. It is deliberately kept out of `Data.edge_index`: scoring, `edge_attr`
    alignment and reverse-edge pairing all assume `edge_index` is bipartite-only.
    """

    contract: str
    node_offset: int
    link_offset: int
    node_features: np.ndarray  # [n_net_nodes, NET_NODE_FEATURE_DIM]
    link_features: np.ndarray  # [n_net_links, NET_LINK_FEATURE_DIM]
    edge_index: np.ndarray  # [2, E] int64
    node_names: Tuple[str, ...]
    link_keys: Tuple[str, ...]

    @property
    def n_net_nodes(self) -> int:
        return int(self.node_features.shape[0])

    @property
    def n_net_links(self) -> int:
        return int(self.link_features.shape[0])

    @property
    def num_entities(self) -> int:
        return self.n_net_nodes + self.n_net_links


def _empty_block(contract: str, node_offset: int) -> NetworkGraphBlock:
    return NetworkGraphBlock(
        contract=contract,
        node_offset=node_offset,
        link_offset=node_offset,
        node_features=np.zeros((0, NET_NODE_FEATURE_DIM), dtype=np.float32),
        link_features=np.zeros((0, NET_LINK_FEATURE_DIM), dtype=np.float32),
        edge_index=np.zeros((2, 0), dtype=np.int64),
        node_names=(),
        link_keys=(),
    )


def _route_or_empty(
    routes: Mapping[str, Mapping[str, Sequence[str]]], src: str, dst: str
) -> List[str]:
    """Link keys from `src` to `dst`, or `[]` for a local placement.

    A task placed on its own source node never crosses the fabric, and
    `route_links` has no entry for it. Every *other* missing route is a real error and
    is left to raise — a silent `[]` there is how a route silently stops being charged.
    """
    if src == dst:
        return []
    return route_links(routes, src, dst)


def build_network_graph_block(
    *,
    node_names: Sequence[str],
    platform_node_names: Sequence[str],
    task_source_names: Sequence[str],
    task_candidate_node_names: Sequence[Sequence[str]],
    link_topology: Optional[Mapping[str, Any]],
    n_tasks: int,
    n_platforms: int,
    contract: Optional[str] = None,
    link_scope: str = "core",
) -> NetworkGraphBlock:
    """Build the network entity block for one inference step / one cached graph.

    Pure and env-free by design: the cache builders work from dataframes and live
    inference works from `Node` objects, and both reduce to these plain sequences rather
    than one adopting the other's data model. That is what keeps train and serve building
    the *same* graph — the property `mp_parity` proved is not optional.

    Args:
        node_names: Every physical node name (clients and servers), in graph order.
        platform_node_names: Host node name per platform *position*, so entry `p` is the
            host of the platform at global index `n_tasks + p`.
        task_source_names: Source node name per task, entry `t` for global index `t`.
        task_candidate_node_names: Per task, the host node name of each of its candidate
            platforms — **one entry per candidate edge, repeats included**, so the
            per-link candidate fractions weight nodes by how many candidates they carry.
        link_topology: The `link_topology` block (`links` + `routes`). `None` for every
            corpus generated without a backbone, which yields an empty block: contract on
            but no fabric is a legitimate, silent no-op, not an error.
        n_tasks: Task count (index offset for platforms).
        n_platforms: Platform count (index offset for network nodes).
        contract: Explicit contract, else $NETWORK_GRAPH_CONTRACT, else `off`.
        link_scope: `"core"` (default) keeps only genuinely shared core segments as
            entities; `"all"` includes access links too. See the module docstring for why
            `core` is the default in a size-transfer study.

    Returns:
        A `NetworkGraphBlock`. Empty (zero entities, zero edges) under contract `off` or
        when there is no fabric.
    """
    resolved = resolve_network_graph_contract(contract)
    node_offset = int(n_tasks) + int(n_platforms)
    if resolved == NETWORK_GRAPH_CONTRACT_OFF:
        return _empty_block(resolved, node_offset)

    if link_scope not in ("core", "all"):
        raise ValueError(f"link_scope must be 'core' or 'all', got {link_scope!r}")

    links: Mapping[str, Mapping[str, Any]] = (link_topology or {}).get("links") or {}
    routes: Mapping[str, Mapping[str, Sequence[str]]] = (link_topology or {}).get("routes") or {}
    if not links:
        return _empty_block(resolved, node_offset)

    names = [str(n) for n in node_names]
    node_pos = {name: i for i, name in enumerate(names)}
    n_net_nodes = len(names)
    link_offset = node_offset + n_net_nodes

    entity_link_keys = [
        key for key in sorted(links) if link_scope == "all" or is_core_link(key)
    ]
    if not entity_link_keys:
        return _empty_block(resolved, node_offset)
    link_pos = {key: i for i, key in enumerate(entity_link_keys)}

    # --- Route walk -------------------------------------------------------------------
    # One pass over every candidate edge collects everything the features need. Routes are
    # cached per (src, dst) because a task's candidates repeat host nodes constantly.
    route_cache: Dict[Tuple[str, str], List[str]] = {}

    def _entity_links(src: str, dst: str) -> List[str]:
        cached = route_cache.get((src, dst))
        if cached is None:
            cached = [k for k in _route_or_empty(routes, src, dst) if k in link_pos]
            route_cache[(src, dst)] = cached
        return cached

    task_link_pairs: set = set()  # (task_idx, link_pos)
    node_link_pairs: set = set()  # (node_pos, link_pos)
    link_candidate_counts = np.zeros(len(entity_link_keys), dtype=np.float64)
    link_task_counts = np.zeros(len(entity_link_keys), dtype=np.float64)
    node_candidate_counts = np.zeros(n_net_nodes, dtype=np.float64)
    total_candidates = 0

    for t_idx, candidates in enumerate(task_candidate_node_names):
        if t_idx >= len(task_source_names):
            raise ValueError(
                f"task_candidate_node_names has {len(task_candidate_node_names)} entries "
                f"but task_source_names has {len(task_source_names)}"
            )
        src = str(task_source_names[t_idx])
        seen_by_task: set = set()
        for dst_raw in candidates:
            dst = str(dst_raw)
            total_candidates += 1
            dst_pos = node_pos.get(dst)
            if dst_pos is not None:
                node_candidate_counts[dst_pos] += 1.0
            for key in _entity_links(src, dst):
                l_pos = link_pos[key]
                link_candidate_counts[l_pos] += 1.0
                task_link_pairs.add((t_idx, l_pos))
                seen_by_task.add(l_pos)
                if dst_pos is not None:
                    node_link_pairs.add((dst_pos, l_pos))
        for l_pos in seen_by_task:
            link_task_counts[l_pos] += 1.0

    # --- Node entity features ---------------------------------------------------------
    platforms_per_node = np.zeros(n_net_nodes, dtype=np.float64)
    for host in platform_node_names:
        pos = node_pos.get(str(host))
        if pos is not None:
            platforms_per_node[pos] += 1.0

    is_source = np.zeros(n_net_nodes, dtype=np.float64)
    for src in task_source_names:
        pos = node_pos.get(str(src))
        if pos is not None:
            is_source[pos] = 1.0

    core_links_per_node = np.zeros(n_net_nodes, dtype=np.float64)
    for node_p, _link_p in node_link_pairs:
        core_links_per_node[node_p] += 1.0

    # Access latency of the node's own attachment links. A property of the node, in
    # seconds, scaled by a constant — not by anything that grows with the cluster.
    access_latency = np.full(n_net_nodes, np.inf, dtype=np.float64)
    for key, attrs in links.items():
        if is_core_link(key):
            continue
        left, _, right = key.partition("|")
        for endpoint in (left, right):
            pos = node_pos.get(endpoint)
            if pos is None:
                continue
            lat = float(attrs.get("latency", 0.0))
            if lat < access_latency[pos]:
                access_latency[pos] = lat
    access_latency[np.isinf(access_latency)] = 0.0

    node_features = np.zeros((n_net_nodes, NET_NODE_FEATURE_DIM), dtype=np.float32)
    node_features[:, 0] = [
        1.0 if name.startswith("client_node") else 0.0 for name in names
    ]
    node_features[:, 1] = platforms_per_node / _PLATFORMS_PER_NODE_SCALE
    node_features[:, 2] = node_candidate_counts / _PLATFORMS_PER_NODE_SCALE
    node_features[:, 3] = core_links_per_node / _CORE_LINKS_PER_NODE_SCALE
    node_features[:, 4] = access_latency * _LATENCY_SCALE
    node_features[:, 5] = is_source

    # --- Link entity features ---------------------------------------------------------
    # `candidate_fraction` is the contention prior: the share of this batch's candidate
    # placements whose route crosses this segment. A ratio in [0, 1], and not something a
    # pointwise model can compute — it needs the route table to know which candidates
    # share a pipe. Two tasks bound for *different* destination nodes contribute to the
    # same entry here, which is the coupling no node-occupancy count can express.
    denom = float(total_candidates) if total_candidates else 1.0
    link_features = np.zeros((len(entity_link_keys), NET_LINK_FEATURE_DIM), dtype=np.float32)
    for i, key in enumerate(entity_link_keys):
        attrs = links[key]
        link_features[i, 0] = 1.0 if is_core_link(key) else 0.0
        link_features[i, 1] = float(attrs.get("bandwidth_mbps", 0.0)) / _BANDWIDTH_SCALE
        link_features[i, 2] = float(attrs.get("latency", 0.0)) * _LATENCY_SCALE
        link_features[i, 3] = link_task_counts[i] / float(max(n_tasks, 1))
        link_features[i, 4] = link_candidate_counts[i] / denom

    # --- Edges ------------------------------------------------------------------------
    src_idx: List[int] = []
    dst_idx: List[int] = []

    def _add_undirected(a: int, b: int) -> None:
        src_idx.extend([a, b])
        dst_idx.extend([b, a])

    # platform --hosted_on--> node. Degree is platforms-per-node, a config constant.
    for p_pos, host in enumerate(platform_node_names):
        host_pos = node_pos.get(str(host))
        if host_pos is None:
            continue
        _add_undirected(n_tasks + p_pos, node_offset + host_pos)

    # node --routes_via--> link, and task --routes_via--> link. Both bounded by the core
    # link count, i.e. by `n_core`, which is a backbone config constant.
    for node_p, link_p in sorted(node_link_pairs):
        _add_undirected(node_offset + node_p, link_offset + link_p)
    for t_idx, link_p in sorted(task_link_pairs):
        _add_undirected(t_idx, link_offset + link_p)

    # link --adjacent--> link, for links sharing an endpoint. This is the backbone's own
    # shape: it is what lets a two-hop route's congestion reach a neighbouring segment.
    endpoints = [tuple(key.split("|")) for key in entity_link_keys]
    for i in range(len(entity_link_keys)):
        for j in range(i + 1, len(entity_link_keys)):
            if set(endpoints[i]) & set(endpoints[j]):
                _add_undirected(link_offset + i, link_offset + j)

    edge_index = (
        np.asarray([src_idx, dst_idx], dtype=np.int64)
        if src_idx
        else np.zeros((2, 0), dtype=np.int64)
    )

    if not np.isfinite(node_features).all():
        raise ValueError("network_graph: non-finite node entity features")
    if not np.isfinite(link_features).all():
        raise ValueError("network_graph: non-finite link entity features")

    return NetworkGraphBlock(
        contract=resolved,
        node_offset=node_offset,
        link_offset=link_offset,
        node_features=node_features,
        link_features=link_features,
        edge_index=edge_index,
        node_names=tuple(names),
        link_keys=tuple(entity_link_keys),
    )


# `net_edge_index` carries 'index' in its name so PyG batches it with the same +num_nodes
# increment as `edge_index` — the same trick `node_edge_index` relies on. That increment is
# only correct if `num_nodes` counts the network entities too, which `attach_network_graph_block`
# sets explicitly.
NET_EDGE_INDEX_ATTR = "net_edge_index"


def attach_network_graph_block(data: Any, block: NetworkGraphBlock) -> Any:
    """Attach a block's tensors to a PyG `Data`, or leave it untouched when empty.

    Both the cache builders and live inference call *this*, not their own attachment code:
    an attribute one path sets and the other does not is precisely the train/serve split
    that cost 12.4x live RTT. Under contract `off` — and when a corpus simply has no
    fabric — nothing is set at all, so the `Data` stays byte-identical to the pre-2026-08-18
    graph and every existing checkpoint keeps loading.

    Imports torch locally to keep this module's core builder numpy-pure, so the simulator
    can import it without pulling in torch.
    """
    if block.contract == NETWORK_GRAPH_CONTRACT_OFF or block.num_entities == 0:
        return data

    import torch

    data.net_node_features = torch.from_numpy(block.node_features).to(torch.float32)
    data.net_link_features = torch.from_numpy(block.link_features).to(torch.float32)
    setattr(
        data,
        NET_EDGE_INDEX_ATTR,
        torch.from_numpy(block.edge_index).to(torch.long),
    )
    data.n_net_nodes = block.n_net_nodes
    data.n_net_links = block.n_net_links
    # Explicit, because PyG otherwise infers num_nodes from the max index it can see and
    # would mis-offset every batched graph whose last network entity has no edge.
    data.num_nodes = block.link_offset + block.n_net_links
    return data
