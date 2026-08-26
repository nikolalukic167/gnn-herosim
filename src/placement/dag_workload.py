"""Production loaders for a co-sim dataset's DAG structure and ingress endpoints.

Single source for the src/ side (cache builder, offline eval, row extraction).
Task ids follow the same static_order assignment the whole pipeline uses: each
workload event's dag is topologically sorted (graphlib.TopologicalSorter) and its
tasks take consecutive ids in that order. The scripts_cosim analysis stack keeps
its own deliberately independent copies (score_route_b_contention.load_dag_edges
et al.) — that independence is a verification property; do not "deduplicate" it
by importing this module there or vice versa.
"""

from __future__ import annotations

import json
import math
from graphlib import TopologicalSorter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


def load_workload_dag(dataset_dir: Path) -> Dict[str, Any]:
    """(task type names, dag edges, ingress sources) for one dataset, ids in the
    static_order assignment.

    Returns dict with:
      task_type_names  task_id -> type name (list)
      dag_edges        list of (parent_task_id, child_task_id)
      task_sources     task_id -> submitting client node name (list; entries may
                       be None on pre-fabric workloads — consumers that need one
                       must fail loudly, not default)
    """
    with open(Path(dataset_dir) / "workload.json") as fh:
        workload = json.load(fh)
    names: List[str] = []
    sources: List[Optional[str]] = []
    edges: List[Tuple[int, int]] = []
    offset = 0
    for event in workload["events"]:
        dag = event["application"]["dag"]
        node_name = event.get("node_name")
        if isinstance(dag, list):
            names.extend(dag)
            sources.extend([node_name] * len(dag))
            offset += len(dag)
            continue
        if not isinstance(dag, dict):
            raise RuntimeError(
                f"{dataset_dir}: unrecognised dag shape {type(dag)!r}"
            )
        order = list(TopologicalSorter(dag).static_order())
        local = {name: offset + i for i, name in enumerate(order)}
        for child, parents in dag.items():
            for parent in parents:
                edges.append((local[parent], local[child]))
        names.extend(order)
        sources.extend([node_name] * len(order))
        offset += len(order)
    return {"task_type_names": names, "dag_edges": edges, "task_sources": sources}


def parents_map(n_tasks: int, dag_edges: Sequence[Tuple[int, int]]) -> Dict[int, List[int]]:
    parents: Dict[int, List[int]] = {t: [] for t in range(n_tasks)}
    for parent, child in dag_edges:
        parents[child].append(parent)
    return parents


def load_link_topology(dataset_dir: Path) -> Tuple[dict, dict]:
    """(routes, links) from infrastructure.json's link_topology; empty dicts when
    the dataset has no fabric."""
    with open(Path(dataset_dir) / "infrastructure.json") as fh:
        infra = json.load(fh)
    lt = infra.get("link_topology") or {}
    return lt.get("routes") or {}, lt.get("links") or {}


def route_hops_and_bottleneck(
    routes: Mapping[str, Any], links: Mapping[str, Any], src: str, dst: str
) -> Tuple[int, float]:
    """(n_hops, bottleneck_mbps) for src -> dst from the dataset's own routes —
    the quantities Platform._payload_transfer_time charges. Same node: (0, inf).
    Missing route or link on a dataset that has a fabric: fail loud."""
    if src == dst:
        return 0, math.inf
    path = (routes.get(src) or {}).get(dst)
    if not path:
        raise RuntimeError(f"no route {src}->{dst} in link_topology")
    bneck = math.inf
    for a, b in zip(path, path[1:]):
        key = f"{a}|{b}" if a <= b else f"{b}|{a}"
        link = links.get(key)
        if link is None:
            raise RuntimeError(f"route {src}->{dst} uses link {key} absent from links")
        bneck = min(bneck, float(link["bandwidth_mbps"]))
    return len(path) - 1, bneck
