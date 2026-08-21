"""Describe the infrastructure a training corpus actually spans.

A checkpoint's `.contract.json` records which *feature contracts* it was trained under,
but until now nothing recorded which *infrastructure* it was trained on. That gap let the
`contention_v2_873` sealed-holdout gate run 40-client/40-server p50 topologies against a
model fitted on 20/20 p25 corpora without a word of warning.

This module derives that description from the graph cache's own metadata — never from a
hand-written constant — so the record cannot drift from the data. Fields that are
single-valued across the corpus are emitted as scalars; fields that legitimately span a
range (the full siv1 corpus mixes six connection probabilities) are emitted as
`<field>_values` sets, which `executesimulation.check_checkpoint_corpus_compatibility`
tests for membership rather than equality.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.placement.topology_features import CLIENT_NODE_PREFIX  # noqa: F401  (documents the tier split)

SIMULATION_DATA_ROOT = Path("simulation_data")


def _space_config_path(root: Path, dataset_id: str) -> Path:
    return root / dataset_id / "space_with_network.json"


def _collapse(values: List[Any], field: str) -> Dict[str, Any]:
    """A scalar when the corpus is single-valued on this axis, a set otherwise."""
    unique = sorted({v for v in values if v is not None}, key=lambda v: (str(type(v)), v))
    if not unique:
        return {}
    if len(unique) == 1:
        return {field: unique[0]}
    return {f"{field}_values": unique}


def derive_corpus_provenance(
    cache_metadata: Dict[str, Any],
    simulation_data_root: Path = SIMULATION_DATA_ROOT,
    max_datasets: Optional[int] = None,
) -> Dict[str, Any]:
    """Summarize the infrastructure spanned by the datasets a cache was built from.

    Reads each dataset's `space_with_network.json`. Missing space configs are counted and
    reported rather than skipped silently — a corpus whose provenance cannot be read is a
    fact the checkpoint should carry, not one to swallow.
    """
    dataset_ids: List[str] = list(cache_metadata.get("dataset_ids") or [])
    if max_datasets is not None:
        dataset_ids = dataset_ids[:max_datasets]

    clients, servers, topo_types, conn_probs = [], [], [], []
    unreadable = 0
    for dataset_id in dataset_ids:
        path = _space_config_path(simulation_data_root, dataset_id)
        if not path.is_file():
            unreadable += 1
            continue
        with open(path, "r") as handle:
            space_config = json.load(handle)
        nodes = space_config.get("nodes", {})
        topology = space_config.get("network", {}).get("topology", {})
        clients.append(nodes.get("client_nodes", {}).get("count"))
        servers.append(nodes.get("server_nodes", {}).get("count"))
        topo_types.append(topology.get("type"))
        conn_probs.append(topology.get("connection_probability"))

    provenance: Dict[str, Any] = {
        "collections": [
            str(base).split("simulation_data/")[-1]
            for base in (cache_metadata.get("base_dirs") or [])
        ],
        "n_datasets": len(dataset_ids),
    }
    provenance.update(_collapse(clients, "client_node_count"))
    provenance.update(_collapse(servers, "server_node_count"))
    provenance.update(_collapse(topo_types, "topology_type"))
    provenance.update(_collapse(conn_probs, "connection_probability"))
    if unreadable:
        provenance["datasets_without_space_config"] = unreadable

    warmth = _collect_warmth_physics(cache_metadata, simulation_data_root)
    if warmth:
        provenance.update(_collapse(warmth, "warmth_physics"))
    return provenance


def _collect_warmth_physics(
    cache_metadata: Dict[str, Any], simulation_data_root: Path
) -> List[str]:
    """Warmth model per source collection, from each collection's METADATA.json."""
    found: List[str] = []
    for base in cache_metadata.get("base_dirs") or []:
        collection = str(base).split("simulation_data/")[-1]
        metadata_path = simulation_data_root / collection / "METADATA.json"
        if not metadata_path.is_file():
            continue
        with open(metadata_path, "r") as handle:
            payload = json.load(handle)
        model = (payload.get("physics") or {}).get("warmth_model")
        if model:
            found.append(str(model))
    return found
