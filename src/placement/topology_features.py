"""Topology-derived task features — single source of truth for cache and live inference.

Task feature dim 2 (after the 2-d task-type onehot) encodes *where the task came from*.
The formula used to live in six independent copies:

  src/policy/tabular/feature_builder.py      live inference (GNN + MLP)
  src/notebooks/prepare_graphs_cache.py      main training cache
  src/notebooks/prepare_graphs_ram.py        in-RAM cache variant
  src/notebooks/prepare_graphs_cache_seq.py  sequential-decode cache
  src/policy/gnn_hetero/scheduler.py         hetero scheduler
  src/policy/tabular/reduced_features.py     2-d -> 3-d seq-cache enrichment

which is the same shape of duplication that `queue_features.py` was created to end, and
the same one that cost 12.4x live RTT when train and serve disagreed about message
passing. Any change to this feature that misses one copy silently breaks train/serve
parity, so the formula now lives here and every copy calls it.

Two contracts exist so that checkpoints trained under the old feature are never served the
new one:

  src_index_v0        the historical formula, kept bit-exact for every pre-2026-08-18
                      cache and checkpoint:

                          src_norm = index_of(source_node) / len(nodes)

                      This is a node's *arbitrary enumeration index* scaled by the node
                      count. It is not a topological property: node 5 has no relationship
                      to node 6 beyond the order they were generated in. Two problems
                      follow, and only the second one was ever load-bearing.

                      First, it is noise the models must learn to ignore. Second — and the
                      reason this module exists — its granularity *and* its distribution
                      both change with cluster size. At 50 nodes it takes values in
                      multiples of 0.02; at 100 nodes, multiples of 0.01. A model trained
                      at one size and evaluated at another sees a feature whose meaning
                      shifted underneath it, so any measured degradation across topology
                      sizes is partly this artifact rather than a failure of topological
                      reasoning. It makes the topology-transfer question unanswerable.

  size_invariant_v1   the fraction of server nodes reachable from the source node:

                          src_reach = |{s in servers : source reaches s}| / |servers|

                      A ratio in [0, 1], so it is size-invariant by construction, and a
                      real property of the source: how much of the cluster this task can
                      actually be placed on. Nothing is lost by dropping the index — the
                      source->candidate `latency` the index was standing in for is already
                      carried explicitly as edge attribute 1, per-candidate and exact,
                      which the index never was.

`src_index_v0` remains the default so every existing cache, checkpoint and corpus keeps
reading the features it was built with. Topology-transfer work sets
`TOPOLOGY_FEATURE_CONTRACT=size_invariant_v1` and rebuilds its cache.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

TOPOLOGY_FEATURE_CONTRACT_SRC_INDEX_V0 = "src_index_v0"
TOPOLOGY_FEATURE_CONTRACT_SIZE_INVARIANT_V1 = "size_invariant_v1"
VALID_TOPOLOGY_FEATURE_CONTRACTS = frozenset(
    {
        TOPOLOGY_FEATURE_CONTRACT_SRC_INDEX_V0,
        TOPOLOGY_FEATURE_CONTRACT_SIZE_INVARIANT_V1,
    }
)
DEFAULT_TOPOLOGY_FEATURE_CONTRACT = TOPOLOGY_FEATURE_CONTRACT_SRC_INDEX_V0
TOPOLOGY_FEATURE_CONTRACT_ENV = "TOPOLOGY_FEATURE_CONTRACT"

CLIENT_NODE_PREFIX = "client_node"


class InvalidTopologyFeatureContractError(ValueError):
    """Raised when a topology feature contract name is not recognized."""


class TopologyFeatureContractMismatchError(ValueError):
    """Raised when a checkpoint's training contract differs from the serving contract."""


def validate_topology_feature_contract(contract: str) -> str:
    normalized = str(contract).strip().lower()
    if normalized not in VALID_TOPOLOGY_FEATURE_CONTRACTS:
        raise InvalidTopologyFeatureContractError(
            f"Unknown topology feature contract {contract!r}; "
            f"expected one of {sorted(VALID_TOPOLOGY_FEATURE_CONTRACTS)}"
        )
    return normalized


def resolve_topology_feature_contract(explicit: Optional[str] = None) -> str:
    """Explicit argument wins, then $TOPOLOGY_FEATURE_CONTRACT, then src_index_v0."""
    if explicit is not None and str(explicit).strip():
        return validate_topology_feature_contract(explicit)
    from_env = os.environ.get(TOPOLOGY_FEATURE_CONTRACT_ENV, "").strip()
    if from_env:
        return validate_topology_feature_contract(from_env)
    return DEFAULT_TOPOLOGY_FEATURE_CONTRACT


def require_matching_topology_feature_contract(
    trained_contract: Optional[str], serving_contract: str, *, model_label: str
) -> None:
    """Fail loudly rather than serve a checkpoint a feature it was never trained on."""
    serving = validate_topology_feature_contract(serving_contract)
    if trained_contract is None:
        return
    trained = validate_topology_feature_contract(trained_contract)
    if trained != serving:
        raise TopologyFeatureContractMismatchError(
            f"{model_label} was trained under topology feature contract {trained!r} but "
            f"the run resolves to {serving!r}. Set {TOPOLOGY_FEATURE_CONTRACT_ENV}="
            f"{trained} (or load a checkpoint trained under the serving contract); task "
            "feature dim 2 differs between contracts and a mismatch silently corrupts "
            "every placement decision."
        )


def is_server_node(node_name: str) -> bool:
    return not str(node_name).startswith(CLIENT_NODE_PREFIX)


@dataclass(frozen=True)
class SourceFeatureContext:
    """Everything needed to compute task feature dim 2, built once per snapshot.

    Both the cache builders (which work from dataframes) and live inference (which works
    from `Node` objects) construct this from their own data via
    `build_source_feature_context`, then call `feature()` per task. That keeps the two
    paths sharing one formula without either having to adopt the other's data model.
    """

    contract: str
    node_count: int
    first_idx_by_name: Mapping[str, int]
    server_names: Tuple[str, ...]
    network_maps: Mapping[str, Any]

    def feature(self, source_node: str) -> float:
        name = str(source_node)
        if self.contract == TOPOLOGY_FEATURE_CONTRACT_SRC_INDEX_V0:
            idx = float(self.first_idx_by_name.get(name, 0))
            return idx / float(max(self.node_count, 1))

        if not self.server_names:
            return 0.0
        reachable = sum(
            1
            for server in self.server_names
            # A task sourced on a server node can always be placed on that node itself,
            # which is reachability the network map does not record.
            if server == name or name in (self.network_maps.get(server) or {})
        )
        return float(reachable) / float(len(self.server_names))


def build_source_feature_context(
    node_names: Sequence[str],
    network_maps: Mapping[str, Any],
    *,
    contract: Optional[str] = None,
    first_idx_by_name: Optional[Mapping[str, int]] = None,
    node_count: Optional[int] = None,
) -> SourceFeatureContext:
    """Build the per-snapshot context.

    Args:
        node_names: Every node name, **in generation order**. Order is load-bearing under
            `src_index_v0` (it is the index) and irrelevant under `size_invariant_v1`.
        network_maps: node name -> that node's network map (peer name -> latency info).
        contract: Explicit contract, else $TOPOLOGY_FEATURE_CONTRACT, else the default.
        first_idx_by_name: Pre-computed name -> index map. The cache builders already
            derive this via `groupby(...).first()` and pass it verbatim, so `src_index_v0`
            stays bit-exact against every existing cache and checkpoint no matter how the
            frame is indexed. Derived from `node_names` order when omitted.
        node_count: The `src_index_v0` divisor. Defaults to `len(node_names)`; pass it
            explicitly where the caller's node list may contain entries that produce no
            name (only `reduced_features.py`, which reads nodes back out of JSON), since
            v0 divided by the *unfiltered* count.
    """
    resolved = resolve_topology_feature_contract(contract)

    names = [str(n) for n in node_names]
    if first_idx_by_name is None:
        # First occurrence, matching the cache builders' `groupby(...).first()`. Live
        # inference previously used a dict comprehension, which keeps the *last*
        # occurrence — identical whenever node names are unique (they are), but pinned
        # here so it stays that way.
        derived: Dict[str, int] = {}
        for idx, name in enumerate(names):
            derived.setdefault(name, idx)
        first_idx_by_name = derived

    return SourceFeatureContext(
        contract=resolved,
        node_count=len(names) if node_count is None else int(node_count),
        first_idx_by_name=first_idx_by_name,
        server_names=tuple(n for n in names if is_server_node(n)),
        network_maps=network_maps,
    )
