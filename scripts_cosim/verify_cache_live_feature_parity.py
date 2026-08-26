#!/usr/bin/env python3
"""
Golden parity test: cache vs live feature builder must produce identical graphs.

Loads a co-sim dataset SSC and asserts that ``prepare_graphs_cache.build_graph``
and ``build_pyg_inference_graph`` produce eps-equal:
  - Node feature layouts (tasks, platforms)
  - Edge index topology (via placement maps)
  - Edge attributes (incl. ``is_warm``), undirected-aligned
  - Queue normalization
  - ``node_edge_index`` same-node edges
  - Layout forks (dim14 / dim22 / dim24)

Fail-loud with detailed diffs. Exit 0 (pass) or 1 (fail).

Usage:
    pipenv run python3 scripts_cosim/verify_cache_live_feature_parity.py
    pipenv run python3 scripts_cosim/verify_cache_live_feature_parity.py \\
        --dataset simulation_data/gnn_datasets_4tasks_regime_b_cold_burst_v1_oracle_split_cosim/ds_00003
    pipenv run python3 scripts_cosim/verify_cache_live_feature_parity.py --layout dim24
    pipenv run python3 scripts_cosim/verify_cache_live_feature_parity.py --queue-norm adaptive_nonzero
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

NOTEBOOKS = PROJECT_ROOT / "src" / "notebooks"
if str(NOTEBOOKS) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS))

from non_unique_lib.training_contract import load_sweep_minimum  # noqa: E402
from prepare_graphs_cache import (  # noqa: E402
    build_graph as build_cache_graph_dim24,
    extract_dataset_to_dataframes,
    load_extended_state_data,
)
from src.placement.model import SystemState  # noqa: E402
from src.placement.queue_features import (  # noqa: E402
    resolve_queue_feature_contract,
)
from src.policy.tabular.feature_builder import (  # noqa: E402
    build_pyg_inference_graph,
    _uses_candidate_relative_layout,
)
from src.policy.tabular.reduced_features import (  # noqa: E402
    FULL_PLATFORM_QUEUE_DIM,
    candidate_relative_queue_columns,
)

EPS = 1e-6
EPS_RELATIVE = 1e-5

DEFAULT_DATASET = (
    PROJECT_ROOT
    / "simulation_data"
    / "gnn_datasets_4tasks_regime_b_cold_burst_v1_oracle_split_cosim"
    / "ds_00003"
)
DEFAULT_PRIORS = PROJECT_ROOT / "data" / "nofs-ids" / "task-types.json"

# Live queue_norm_mode aliases that match cache prepare_graphs_cache modes.
LIVE_QUEUE_NORM = {
    "scheduler_adaptive": "adaptive",
    "adaptive": "adaptive",
    "adaptive_nonzero": "adaptive_nonzero",
    "fixed": "adaptive",  # live has no fixed; still compare adaptive path
}


class _Triggered:
    def __init__(self, triggered: bool) -> None:
        self.triggered = bool(triggered)


class _HashObj:
    """Hashable stand-in for Node/Platform so SystemState.replicas can hold them."""

    __slots__ = (
        "id",
        "node_name",
        "network_map",
        "network",
        "platforms",
        "storage",
        "type",
        "initialized",
        "previous_task",
        # The shared NetworkFabric a real Node carries; the network feature builder reads
        # `link_topology` off it. Absent on platforms, which is why it is optional here.
        "fabric",
        "_key",
    )

    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)
        self._key = kwargs.get("_key")

    def __hash__(self) -> int:
        return hash(self._key)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _HashObj) and self._key == other._key


def _safe_array(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _compare_arrays(
    name: str,
    cache_arr: Any,
    live_arr: Any,
    eps: float = EPS,
    eps_rel: float = EPS_RELATIVE,
) -> Tuple[bool, str]:
    c = _safe_array(cache_arr)
    l = _safe_array(live_arr)
    if c.shape != l.shape:
        return False, f"{name} shape mismatch: cache {c.shape} vs live {l.shape}"
    if c.size == 0 and l.size == 0:
        return True, ""
    # Identity-keyed edge sets are object arrays of (task, node_name, platform_id) — not
    # numeric, so they compare exactly or not at all.
    if c.dtype == object or l.dtype == object:
        if not np.array_equal(c, l):
            diff = [
                (tuple(cr), tuple(lr))
                for cr, lr in zip(c, l)
                if tuple(cr) != tuple(lr)
            ]
            return False, f"{name} differs on {len(diff)} rows, first: {diff[:2]}"
        return True, ""
    if not np.allclose(c, l, atol=eps, rtol=eps_rel):
        abs_diff = np.abs(c.astype(np.float64) - l.astype(np.float64))
        max_diff_idx = np.unravel_index(int(np.argmax(abs_diff)), abs_diff.shape)
        max_diff = float(abs_diff[max_diff_idx])
        cache_val = float(c[max_diff_idx])
        live_val = float(l[max_diff_idx])
        return (
            False,
            f"{name} values differ: max_diff={max_diff:.8g} at {max_diff_idx} "
            f"(cache={cache_val:.8g}, live={live_val:.8g})",
        )
    return True, ""


def _directed_edge_attr(edge_attr: Any) -> np.ndarray:
    arr = _safe_array(edge_attr)
    if arr.size == 0:
        return arr.reshape(0, 5) if arr.ndim < 2 else arr
    if arr.shape[0] % 2 != 0:
        raise ValueError(f"edge_attr length {arr.shape[0]} is not even (undirected dup)")
    return arr[: arr.shape[0] // 2]


def _edge_index_as_sorted_pairs(edge_index: Any) -> np.ndarray:
    ei = _safe_array(edge_index).astype(np.int64)
    if ei.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    pairs = np.stack([ei[0], ei[1]], axis=1)
    # Canonical undirected key then stable sort for set-equality of topology.
    lo = np.minimum(pairs[:, 0], pairs[:, 1])
    hi = np.maximum(pairs[:, 0], pairs[:, 1])
    canon = np.stack([lo, hi], axis=1)
    order = np.lexsort((canon[:, 1], canon[:, 0]))
    return canon[order]


def _node_edge_pairs(node_edge_index: Any) -> np.ndarray:
    return _edge_index_as_sorted_pairs(node_edge_index)


def _identity_edge_set_as_array(edges: set) -> np.ndarray:
    """A sorted, comparable array view of an identity-keyed edge set."""
    if not edges:
        return np.empty((0, 3), dtype=object)
    rows = sorted((t, name, pid) for t, (name, pid) in edges)
    return np.asarray(rows, dtype=object)


def _platform_identity_by_pos(graph: Any) -> List[Tuple[str, int]]:
    """`(node_name, platform_id)` per platform row, dense over `[0, n_platforms)`.

    Platform *position* is not platform *identity*. The cache enumerates platforms from
    `stats.nodeResults` and live from `config.infrastructure.nodes`, and those orders differ
    on essentially every corpus in the repo. Comparing row `p` to row `p` therefore reports
    a difference that no model can observe: nothing in `TaskPlacementGNN` has a per-position
    parameter (the platform encoder is applied row-wise and the scorer indexes
    `platform_emb` by edge target), so a consistent relabelling of platforms is a no-op.
    Measured: with the two graphs' dim 9-11 estimates equalized, per-candidate logits agree
    to 3e-8 across an ordering permutation of 208 platforms.

    So the comparison has to be by identity. What that still catches — and what actually
    matters — is a graph that is internally *inconsistent*: features describing one platform
    while the edges point at another.
    """
    n_platforms = int(graph.n_platforms)
    by_pos: List[Optional[Tuple[str, int]]] = [None] * n_platforms
    for meta in graph.queue_key_to_platform_meta.values():
        by_pos[int(meta["platform_pos"])] = (
            str(meta["node_name"]),
            int(meta["platform_id"]),
        )
    missing = [i for i, ident in enumerate(by_pos) if ident is None]
    if missing:
        raise ValueError(f"platform positions {missing[:5]} have no identity in the meta map")
    return [ident for ident in by_pos if ident is not None]


def _platform_permutation(cache_g: Any, live_g: Any) -> Optional[np.ndarray]:
    """Row indices reordering the cache's platforms onto the live ordering.

    `None` when the two graphs do not even describe the same set of platforms, which *is* a
    real failure and is reported by the caller rather than papered over here.
    """
    cache_ident = _platform_identity_by_pos(cache_g)
    live_ident = _platform_identity_by_pos(live_g)
    if set(cache_ident) != set(live_ident):
        return None
    cache_pos = {ident: i for i, ident in enumerate(cache_ident)}
    return np.asarray([cache_pos[ident] for ident in live_ident], dtype=np.int64)


def _candidate_relative_by_identity(
    graph: Any, placement_map: Optional[Mapping[int, Any]]
) -> Dict[Tuple[int, Tuple[str, int]], np.ndarray]:
    """`{(task_idx, (node_name, platform_id)): [3]}` — the P5b columns, order-independent.

    Keyed by identity rather than by row for the same reason as
    `_platform_identity_by_pos`: the cache and live builders enumerate platforms (and so
    candidates) in different orders, and the feature is a function of the candidate SET.
    """
    ident_by_pos = _platform_identity_by_pos(graph)
    plat_feats = _safe_array(graph.platform_features)
    out: Dict[Tuple[int, Tuple[str, int]], np.ndarray] = {}
    for t_idx, candidates in (placement_map or {}).items():
        positions = [
            int(graph.queue_key_to_platform_meta[qk]["platform_pos"])
            for qk in _queue_keys_for_candidates(graph, int(t_idx), candidates)
        ]
        cols = candidate_relative_queue_columns(
            plat_feats[positions, FULL_PLATFORM_QUEUE_DIM]
        )
        for row, pos in enumerate(positions):
            out[(int(t_idx), ident_by_pos[pos])] = np.asarray(cols[row], dtype=np.float64)
    return out


def _queue_keys_for_candidates(graph: Any, task_idx: int, candidates: Any) -> List[str]:
    """Queue keys for one task's candidates, in the graph's own candidate order."""
    keys_map = getattr(graph, "task_logit_to_queue_key", None) or {}
    keys = keys_map.get(task_idx)
    if keys is not None:
        return [str(k) for k in keys]
    # Seq/cache graphs that predate the queue-key map: fall back to the identity search
    # resolve_platform_pos already implements.
    meta = graph.queue_key_to_platform_meta
    resolved: List[str] = []
    for node_id, plat_id in candidates:
        for qk, entry in meta.items():
            if int(entry["node_id"]) == int(node_id) and int(entry["platform_id"]) == int(plat_id):
                resolved.append(str(qk))
                break
        else:
            raise ValueError(
                f"no queue_key for candidate (node_id={node_id}, platform_id={plat_id})"
            )
    return resolved


def _bipartite_edges_by_identity(graph: Any) -> set:
    """`{(task_idx, (node_name, platform_id))}` — the edge set, order-independent."""
    ident = _platform_identity_by_pos(graph)
    n_tasks = int(graph.n_tasks)
    ei = _safe_array(graph.edge_index).astype(np.int64)
    edges = set()
    for a, b in zip(ei[0], ei[1]):
        a, b = int(a), int(b)
        # Undirected, so take whichever endpoint is the task.
        if a < n_tasks <= b:
            edges.add((a, ident[b - n_tasks]))
        elif b < n_tasks <= a:
            edges.add((b, ident[a - n_tasks]))
    return edges


def _same_node_edges_by_identity(graph: Any) -> set:
    ident = _platform_identity_by_pos(graph)
    n_tasks = int(graph.n_tasks)
    ei = _safe_array(graph.node_edge_index).astype(np.int64)
    if ei.size == 0:
        return set()
    return {
        frozenset((ident[int(a) - n_tasks], ident[int(b) - n_tasks]))
        for a, b in zip(ei[0], ei[1])
    }


def _edge_attr_by_identity(graph: Any) -> Dict[Tuple[int, Tuple[str, int]], np.ndarray]:
    """Directed task->platform edge attrs keyed by identity, so row order cannot matter."""
    ident = _platform_identity_by_pos(graph)
    n_tasks = int(graph.n_tasks)
    ei = _safe_array(graph.edge_index).astype(np.int64)
    ea = _safe_array(graph.edge_attr)
    out: Dict[Tuple[int, Tuple[str, int]], np.ndarray] = {}
    for row, (a, b) in enumerate(zip(ei[0], ei[1])):
        a, b = int(a), int(b)
        if a < n_tasks <= b:
            out[(a, ident[b - n_tasks])] = ea[row]
    return out


def _load_priors(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"task priors missing: {path}")
    return json.loads(path.read_text())


def _build_live_fixtures(
    ds_path: Path,
    dfs: Dict[str, Any],
    extended: Dict[str, Any],
) -> Tuple[List[Any], SystemState, List[Any]]:
    """Rebuild live Node/Platform/Task/SystemState from SSC + optimal_result infra."""
    opt = json.loads((ds_path / "optimal_result.json").read_text())
    infra_nodes = opt.get("config", {}).get("infrastructure", {}).get("nodes")
    if not infra_nodes:
        raise ValueError(f"{ds_path.name}: optimal_result missing config.infrastructure.nodes")

    df_platforms = dfs["platforms"]
    nodes: List[Any] = []
    for node_id, nd in enumerate(infra_nodes):
        plats: List[Any] = []
        rows = df_platforms[df_platforms["node_id"] == node_id]
        for _, row in rows.iterrows():
            qkey = f"{row.node_name}:{int(row.platform_id)}"
            init = bool(extended["initialized_snapshot"].get(qkey, True))
            prev = (extended["temporal_state"].get(qkey) or {}).get(
                "previous_task_type_name"
            )
            prev_task = (
                SimpleNamespace(type={"name": str(prev)}) if prev is not None else None
            )
            plats.append(
                _HashObj(
                    id=int(row.platform_id),
                    type={"shortName": str(row.platform_type)},
                    initialized=_Triggered(init),
                    previous_task=prev_task,
                    _key=("plat", node_id, int(row.platform_id)),
                )
            )
        nodes.append(
            _HashObj(
                id=int(node_id),
                node_name=str(nd.get("node_name", f"node_{node_id}")),
                network_map=nd.get("network_map", {}) or {},
                network=nd.get("network") or {"bandwidth": 100.0},
                platforms=SimpleNamespace(items=plats),
                storage=SimpleNamespace(items=[]),
                _key=("node", int(node_id)),
            )
        )

    # The fabric the live feature builder reads `link_topology` off. A stand-in rather
    # than a real NetworkFabric because that one needs a simpy env and this harness never
    # advances time — only the topology block is read, and it comes from the same
    # optimal_result.json the cache path uses, which is the point of the comparison.
    link_topology = opt.get("config", {}).get("infrastructure", {}).get("link_topology")
    fabric = SimpleNamespace(link_topology=link_topology) if link_topology else None
    for node in nodes:
        node.fabric = fabric

    name_to_node = {n.node_name: n for n in nodes}
    replicas: Dict[str, set] = {}
    for task_type, lst in (extended.get("replicas") or {}).items():
        bucket = set()
        if not isinstance(lst, list):
            continue
        for item in lst:
            if not (isinstance(item, list) and len(item) >= 2):
                continue
            node = name_to_node.get(str(item[0]))
            if node is None:
                continue
            plat_id = int(item[1])
            plat = next((p for p in node.platforms.items if int(p.id) == plat_id), None)
            if plat is not None:
                bucket.add((node, plat))
        replicas[str(task_type)] = bucket

    system_state = SystemState(
        available_resources={},
        replicas=replicas,
        scheduler_state={},
    )

    tasks: List[Any] = []
    for _, row in dfs["tasks"].iterrows():
        tasks.append(
            SimpleNamespace(
                id=int(row.task_id),
                type={"name": str(row.task_type)},
                node_name=str(row.source_node),
            )
        )
    return nodes, system_state, tasks


def _build_cache_graph(
    dfs: Dict[str, Any],
    extended: Dict[str, Any],
    priors: Dict[str, Any],
    *,
    layout: str,
    queue_norm_mode: str,
    queue_norm_factor: float,
    queue_feature_contract: str,
    queue_snapshot: Mapping[str, int],
    temporal_state: Mapping[str, Mapping[str, float]],
) -> Any:
    """Build cache-path graph for the requested layout."""
    common = dict(
        df_nodes=dfs["nodes"],
        df_tasks=dfs["tasks"],
        df_platforms=dfs["platforms"],
        task_priors=priors,
        queue_norm_factor=queue_norm_factor,
        queue_norm_mode=queue_norm_mode,
        queue_snapshot=queue_snapshot,
        temporal_state=temporal_state,
        initialized_snapshot=extended["initialized_snapshot"],
    )
    # Network entities live in the main cache only; the seq cache below is a separate
    # lineage and its build_graph takes no link_topology.
    dim24_only = dict(
        queue_feature_contract=queue_feature_contract,
        link_topology=dfs.get("link_topology"),
    )
    if layout in ("dim24", "24", "pull_obs", "pull_observables"):
        return build_cache_graph_dim24(**common, **dim24_only)

    if layout in ("dim22", "legacy", "22", "dim25cr", "25", "candrel"):
        # dim22 = dim24 platform features without cold_count / pull_remaining.
        # dim25cr shares this graph exactly: its three extra columns are set-relative and
        # so exist per (task, candidate) group, never on the graph — see the CR block in
        # the comparison section below, which is what actually verifies them.
        g = build_cache_graph_dim24(**common, **dim24_only)
        g.platform_features = g.platform_features[:, :14]
        return g

    if layout in ("dim14", "atomic21", "14"):
        # Seq cache uses is_cold + node_disk_hit (legacy 14-d platform layout) and has no
        # contract-dependent dims: dim7 is the raw count, dim13 is node_disk_hit.
        from prepare_graphs_cache_seq import build_graph as build_cache_graph_seq

        return build_cache_graph_seq(**common)

    raise ValueError(f"Unsupported layout for cache builder: {layout!r}")


def _expected_platform_dims(layout: str) -> int:
    if layout in ("dim24", "24", "pull_obs", "pull_observables"):
        return 16
    if layout in (
        "dim22", "legacy", "22", "dim14", "atomic21", "14", "dim25cr", "25", "candrel",
    ):
        return 14
    raise ValueError(layout)


def _queue_state_for_parity(
    recorded_queue: Mapping[str, int],
    recorded_temporal: Mapping[str, Mapping[str, float]],
    depth_scale: float,
) -> Tuple[Dict[str, int], Dict[str, Dict[str, float]]]:
    """Optionally synthesize queue depth so dim7/dim13 are actually exercised.

    Co-sim datasets used as parity fixtures can have every platform idle (the regime B
    default fixture is 8 keys all at 0), which made the golden test blind to the queue
    features entirely — a cache/live divergence in the dim7 divisor or the dim13 usage
    ratio passed unnoticed. Scaling a deterministic spread of depths over the recorded keys
    covers the range that matters, including depths far past the legacy cap of 100.

    Synthetic depth comes with synthetic temporal state on purpose. The cache decides
    whether to estimate dims 9-11 per *snapshot* (`if temporal_state:`) while the live
    builder decides per *platform* (queue>0 and no recorded remainder), so a queued platform
    with a zero remainder would diverge for reasons that have nothing to do with queue
    scaling. Real snapshots pair depth with a running task, so pairing them here keeps the
    fixture faithful and the test focused.
    """
    queue = {str(k): int(v) for k, v in recorded_queue.items()}
    # Temporal entries are mostly numeric remainders, but newer SSC files also carry
    # `previous_task_type_name` (a task type string, used for the is_warm edge attr).
    # Coercing every value to float choked on those; carry non-numerics through unchanged
    # rather than dropping them, since both builders read that key.
    temporal = {
        str(k): {
            kk: (float(vv) if isinstance(vv, (int, float)) else vv)
            for kk, vv in (v or {}).items()
        }
        for k, v in (recorded_temporal or {}).items()
    }
    if depth_scale <= 0:
        return queue, temporal

    keys = sorted(queue)
    if not keys:
        raise ValueError("cannot synthesize queue depths: recorded snapshot has no keys")
    # Idle / shallow / mid / deep, so the p90 divisor and the log compression both bite.
    pattern = (0, 1, 7, 23, 61, 150)
    for i, key in enumerate(keys):
        queue[key] = int(round(pattern[i % len(pattern)] * depth_scale))
        if queue[key] <= 0:
            continue
        entry = temporal.setdefault(key, {})
        if not float(entry.get("current_task_remaining", 0.0)):
            entry["current_task_remaining"] = 0.5
            entry["cold_start_remaining"] = 0.05
            entry["comm_remaining"] = 0.025
    return queue, temporal


def _assert_no_nonfinite(name: str, arr: Any, failures: List[str]) -> None:
    a = _safe_array(arr)
    if a.size and not np.isfinite(a.astype(np.float64)).all():
        failures.append(f"{name}: contains NaN/Inf")


def verify_parity(
    ds_path: Path,
    *,
    layout: str = "dim24",
    queue_norm_mode: str = "scheduler_adaptive",
    queue_norm_factor: float = 50.0,
    priors_path: Path = DEFAULT_PRIORS,
    verbose: bool = True,
    queue_depth_scale: float = 0.0,
) -> bool:
    failures: List[str] = []
    contract = resolve_queue_feature_contract()

    if verbose:
        print("=== Golden Feature Parity Test (Phase 0.1) ===")
        print(f"Dataset:     {ds_path}")
        print(f"Layout:      {layout}")
        print(f"Queue norm:  {queue_norm_mode}")
        print(f"Contract:    {contract}")
        print(f"Queue depth: {'as recorded' if queue_depth_scale <= 0 else f'synthetic x{queue_depth_scale:g}'}")
        print(f"Priors:      {priors_path}")

    ssc_path = ds_path / "system_state_captured_unique.json"
    jsonl_path = ds_path / "placements" / "placements.jsonl"
    opt_path = ds_path / "optimal_result.json"
    for required in (ssc_path, jsonl_path, opt_path):
        if not required.is_file():
            print(f"ERROR: missing required file {required}", file=sys.stderr)
            return False

    try:
        priors = _load_priors(priors_path)
        extended = load_extended_state_data(ds_path)
        plan, opt_rtt, _combo = load_sweep_minimum(jsonl_path)
        dfs = extract_dataset_to_dataframes(
            opt_path,
            placement_plan=plan,
            opt_rtt=opt_rtt,
            replicas_by_task=extended["replicas"],
        )
    except Exception as exc:
        print(f"ERROR extracting SSC/dataset: {exc}", file=sys.stderr)
        return False

    queue_snapshot, temporal_state = _queue_state_for_parity(
        extended["queue_snapshot"], extended["temporal_state"], queue_depth_scale
    )

    if verbose:
        depths = [int(v) for v in queue_snapshot.values()]
        print(
            f"SSC: tasks={len(dfs['tasks'])} platforms={len(dfs['platforms'])} "
            f"queue_keys={len(queue_snapshot)} "
            f"queue_depth_max={max(depths) if depths else 0} "
            f"initialized={len(extended['initialized_snapshot'])}"
        )

    try:
        cache_g = _build_cache_graph(
            dfs,
            extended,
            priors,
            layout=layout,
            queue_norm_mode=queue_norm_mode,
            queue_norm_factor=queue_norm_factor,
            queue_feature_contract=contract,
            queue_snapshot=queue_snapshot,
            temporal_state=temporal_state,
        )
    except Exception as exc:
        print(f"ERROR building cache graph: {exc}", file=sys.stderr)
        return False

    try:
        nodes, system_state, tasks = _build_live_fixtures(ds_path, dfs, extended)
        live_norm = LIVE_QUEUE_NORM.get(queue_norm_mode, queue_norm_mode)
        os.environ["INFERENCE_FEATURE_LAYOUT"] = layout
        live_g, live_placement = build_pyg_inference_graph(
            tasks,
            system_state,
            queue_snapshot,
            nodes=nodes,
            task_types_data=priors,
            queue_norm_mode=live_norm,
            temporal_state=temporal_state,
            queue_feature_contract=contract,
        )
    except Exception as exc:
        print(f"ERROR building live graph: {exc}", file=sys.stderr)
        return False

    if live_g is None:
        print("ERROR: live builder returned None (no feasible edges)", file=sys.stderr)
        return False

    # --- Dimensional contracts ---
    exp_plat = _expected_platform_dims(layout)
    if int(cache_g.task_features.shape[1]) != 3:
        failures.append(f"cache task_features dim={cache_g.task_features.shape[1]} != 3")
    if int(live_g.task_features.shape[1]) != 3:
        failures.append(f"live task_features dim={live_g.task_features.shape[1]} != 3")
    if int(cache_g.platform_features.shape[1]) != exp_plat:
        failures.append(
            f"cache platform_features dim={cache_g.platform_features.shape[1]} != {exp_plat}"
        )
    if int(live_g.platform_features.shape[1]) != exp_plat:
        failures.append(
            f"live platform_features dim={live_g.platform_features.shape[1]} != {exp_plat}"
        )
    if int(cache_g.n_tasks) != int(live_g.n_tasks):
        failures.append(f"n_tasks cache={cache_g.n_tasks} live={live_g.n_tasks}")
    if int(cache_g.n_platforms) != int(live_g.n_platforms):
        failures.append(
            f"n_platforms cache={cache_g.n_platforms} live={live_g.n_platforms}"
        )

    _assert_no_nonfinite("cache.task_features", cache_g.task_features, failures)
    _assert_no_nonfinite("live.task_features", live_g.task_features, failures)
    _assert_no_nonfinite("cache.platform_features", cache_g.platform_features, failures)
    _assert_no_nonfinite("live.platform_features", live_g.platform_features, failures)
    _assert_no_nonfinite("cache.edge_attr", cache_g.edge_attr, failures)
    _assert_no_nonfinite("live.edge_attr", live_g.edge_attr, failures)

    # --- Value equality ---
    # Everything platform-indexed is compared BY IDENTITY, not by row position. The two
    # builders enumerate platforms in different orders on essentially every corpus here
    # (see _platform_identity_by_pos), and that difference is invisible to the model.
    # Comparing by position turns it into a wall of false failures that hides the real ones.
    perm = _platform_permutation(cache_g, live_g)
    if perm is None:
        failures.append(
            "cache and live describe DIFFERENT platform sets (not merely a different order) "
            "— compare queue_key_to_platform_meta"
        )
    platform_order_differs = perm is not None and not np.array_equal(
        perm, np.arange(len(perm))
    )

    checks: List[Tuple[str, Any, Any]] = [
        ("task_features", cache_g.task_features, live_g.task_features),
    ]
    if perm is not None:
        checks.append(
            (
                "platform_features(by identity)",
                _safe_array(cache_g.platform_features)[perm],
                live_g.platform_features,
            )
        )
    checks.append(
        (
            "edge_index_bipartite(by identity)",
            _identity_edge_set_as_array(_bipartite_edges_by_identity(cache_g)),
            _identity_edge_set_as_array(_bipartite_edges_by_identity(live_g)),
        )
    )

    cache_ea = _edge_attr_by_identity(cache_g)
    live_ea = _edge_attr_by_identity(live_g)
    if set(cache_ea) != set(live_ea):
        failures.append(
            f"edge_attr candidate keys differ by identity: "
            f"cache_only={sorted(set(cache_ea) - set(live_ea))[:3]} "
            f"live_only={sorted(set(live_ea) - set(cache_ea))[:3]}"
        )
    else:
        keys = sorted(cache_ea)
        checks.append(
            (
                "edge_attr(by identity)",
                np.asarray([cache_ea[k] for k in keys]),
                np.asarray([live_ea[k] for k in keys]),
            )
        )

    cache_nei = getattr(cache_g, "node_edge_index", None)
    live_nei = getattr(live_g, "node_edge_index", None)
    if cache_nei is None:
        failures.append("cache missing node_edge_index")
    if live_nei is None:
        failures.append("live missing node_edge_index (unwired same-node edges)")
    if cache_nei is not None and live_nei is not None:
        c_same = _same_node_edges_by_identity(cache_g)
        l_same = _same_node_edges_by_identity(live_g)
        if c_same != l_same:
            failures.append(
                f"node_edge_index differs by identity: "
                f"cache_only={len(c_same - l_same)} live_only={len(l_same - c_same)}"
            )

    # Network entities. Under contract `off` neither side sets them and there is nothing
    # to compare; under `core_v1` a side that sets them alone is the mp_parity failure
    # (12.4x live RTT) with a new name, so a one-sided attribute is a hard failure.
    net_attrs = ("net_node_features", "net_link_features", "net_edge_index")
    cache_has = [a for a in net_attrs if getattr(cache_g, a, None) is not None]
    live_has = [a for a in net_attrs if getattr(live_g, a, None) is not None]
    if set(cache_has) != set(live_has):
        failures.append(
            f"network entity attrs differ: cache={sorted(cache_has)} live={sorted(live_has)} "
            f"(one path emits message-passing structure the other does not)"
        )
    elif cache_has:
        checks.append(
            ("net_node_features", cache_g.net_node_features, live_g.net_node_features)
        )
        checks.append(
            ("net_link_features", cache_g.net_link_features, live_g.net_link_features)
        )
        checks.append(
            (
                "net_edge_index_pairs",
                _edge_index_as_sorted_pairs(cache_g.net_edge_index),
                _edge_index_as_sorted_pairs(live_g.net_edge_index),
            )
        )

    # route_b stage-2 DAG block. The semantics here differ from the network-entity case
    # above: live serving CANNOT build these (prefix construction is stage 3), so the
    # right answer is not "compare them" but "refuse to claim parity". Passing by
    # omission — the harness silently ignoring attrs it does not know about — is exactly
    # how a cache/live divergence goes unnoticed, so an offline-only cache is a hard
    # failure with a specific message instead.
    dag_attrs = (
        "dag_edge_index",
        "dag_parents",
        "task_type_onehot4",
        "partial_state_ctx",
        "tied_optimal_logit_plans",
    )
    cache_dag = [a for a in dag_attrs if getattr(cache_g, a, None) is not None]
    if cache_dag:
        failures.append(
            f"cache carries the offline-only route_b stage-2 DAG block "
            f"({sorted(cache_dag)}); live parity is NOT defined for it — live serving "
            "cannot construct a decode prefix (that is stage 3, and "
            "executesimulation refuses such checkpoints). Run the offline stage-2 "
            "harness instead of this cache/live comparison."
        )

    for name, c_arr, l_arr in checks:
        ok, msg = _compare_arrays(name, c_arr, l_arr)
        if not ok:
            failures.append(msg)

    # Per-dim platform report when platform_features diverge. Permuted onto the live
    # ordering first — comparing row-to-row here was reporting every dim as divergent on
    # any corpus whose platform order differs, which buried the one dim that really did.
    cp = _safe_array(cache_g.platform_features)
    lp = _safe_array(live_g.platform_features)
    if perm is not None and cp.shape == lp.shape:
        cp = cp[perm]
    if cp.shape == lp.shape and cp.size and not np.allclose(cp, lp, atol=EPS, rtol=EPS_RELATIVE):
        diff = np.abs(cp.astype(np.float64) - lp.astype(np.float64))
        for dim in range(cp.shape[1]):
            md = float(diff[:, dim].max())
            if md > EPS:
                failures.append(
                    f"platform_features[:,{dim}] max_diff={md:.8g} "
                    f"(layout={layout}; dim8=shared_fate/is_cold, "
                    f"dim12=target_concurrency, dim13=usage_ratio/node_disk_hit, "
                    f"dim14=node_cold_count, dim15=estimated_pull_remaining)"
                )

    # Placement / warmth contract
    cache_place = getattr(cache_g, "task_logit_to_placement", None) or getattr(
        cache_g, "_task_logit_to_placement", None
    )
    # Compared as SETS per task. The list order is the candidate enumeration order, which
    # inherits the platform enumeration order, and decoding is by identity
    # (`task_logit_to_placement[t][logit]`) — so a reordered candidate list decodes
    # correctly. What must match is *which* placements are candidates for each task.
    cache_sets = {t: set(map(tuple, v)) for t, v in (cache_place or {}).items()}
    live_sets = {t: set(map(tuple, v)) for t, v in (live_placement or {}).items()}
    if cache_sets != live_sets:
        failures.append(
            f"task_logit_to_placement candidate SETS differ: "
            f"cache_keys={sorted(cache_sets)} live_keys={sorted(live_sets)}"
        )
        for t_idx in sorted(set(cache_sets) | set(live_sets)):
            c_set, l_set = cache_sets.get(t_idx, set()), live_sets.get(t_idx, set())
            if c_set != l_set:
                failures.append(
                    f"  task {t_idx}: cache_only={sorted(c_set - l_set)} "
                    f"live_only={sorted(l_set - c_set)}"
                )
                break

    # P5b candidate-relative columns (dim25cr). They never appear on the graph — the
    # training extractor appends them per candidate group and the MLP scheduler appends
    # them per task_boundaries group — so this is the only place the two paths can be
    # compared, and it is the guard against the train/serve skew that cost 12x live RTT
    # the last time a served model saw features its weights had not been fitted on.
    if _uses_candidate_relative_layout(layout):
        cr_cache = _candidate_relative_by_identity(cache_g, cache_place)
        cr_live = _candidate_relative_by_identity(live_g, live_placement)
        if set(cr_cache) != set(cr_live):
            failures.append(
                f"candidate-relative keys differ: cache_only={sorted(set(cr_cache) - set(cr_live))[:3]} "
                f"live_only={sorted(set(cr_live) - set(cr_cache))[:3]}"
            )
        else:
            worst = 0.0
            worst_key = None
            for key in cr_cache:
                delta = float(np.max(np.abs(cr_cache[key] - cr_live[key])))
                if delta > worst:
                    worst, worst_key = delta, key
            if worst > EPS:
                failures.append(
                    f"candidate_relative columns differ: max|cache-live|={worst:.3e} "
                    f"at {worst_key} cache={cr_cache[worst_key]} live={cr_live[worst_key]}"
                )
            elif verbose:
                print(
                    f"candidate_relative: {len(cr_cache)} (task, platform) pairs agree "
                    f"to {worst:.3e}"
                )

    # is_warm: edge attr dim 2 must reflect SSC previous_task_type_name
    cea = _directed_edge_attr(cache_g.edge_attr)
    lea = _directed_edge_attr(live_g.edge_attr)
    if cea.shape == lea.shape and cea.size:
        warm_ok, warm_msg = _compare_arrays("edge_is_warm", cea[:, 2], lea[:, 2])
        if not warm_ok:
            failures.append(warm_msg)
        ssc_has_prev = any(
            isinstance(v, dict) and v.get("previous_task_type_name") is not None
            for v in extended["temporal_state"].values()
        )
        if ssc_has_prev and float(np.max(np.abs(cea[:, 2]))) < EPS:
            failures.append(
                "SSC has previous_task_type_name but cache edge is_warm is all-zero"
            )
        if ssc_has_prev and float(np.max(np.abs(lea[:, 2]))) < EPS:
            failures.append(
                "SSC has previous_task_type_name but live edge is_warm is all-zero"
            )

    # Layout fork spot-checks
    if layout in ("dim24", "dim22") and cp.shape[1] >= 9:
        # dim8 must be shared_fate density in [0,1], not raw is_cold for queue-norm layouts
        for label, arr in (("cache", cp), ("live", lp)):
            if arr.shape[0] and (arr[:, 8].min() < -EPS or arr[:, 8].max() > 1.0 + EPS):
                failures.append(
                    f"{label} platform dim8 (shared_fate) out of [0,1]: "
                    f"min={arr[:, 8].min():.6g} max={arr[:, 8].max():.6g}"
                )

    if verbose:
        print("\n--- Comparison ---")
        print(
            f"cache: tasks={cache_g.n_tasks} plats={cache_g.n_platforms} "
            f"tf={tuple(cache_g.task_features.shape)} pf={tuple(cache_g.platform_features.shape)} "
            f"ei={tuple(cache_g.edge_index.shape)} ea={tuple(cache_g.edge_attr.shape)} "
            f"nei={tuple(_safe_array(cache_nei).shape) if cache_nei is not None else None}"
        )
        print(
            f"live:  tasks={live_g.n_tasks} plats={live_g.n_platforms} "
            f"tf={tuple(live_g.task_features.shape)} pf={tuple(live_g.platform_features.shape)} "
            f"ei={tuple(live_g.edge_index.shape)} ea={tuple(live_g.edge_attr.shape)} "
            f"nei={tuple(_safe_array(live_nei).shape) if live_nei is not None else None}"
        )
        if platform_order_differs:
            n_moved = int((perm != np.arange(len(perm))).sum())
            print(
                f"note:  platform ROW ORDER differs ({n_moved}/{len(perm)} rows moved) — "
                f"cache enumerates from stats.nodeResults, live from "
                f"config.infrastructure.nodes. Benign: the model has no per-position "
                f"parameter, so this is a relabelling. Compared by identity."
            )

    if failures:
        print("\nPARITY FAILURES:", file=sys.stderr)
        for msg in failures:
            print(f"  - {msg}", file=sys.stderr)
        return False

    if verbose:
        print("\n✓ cache ↔ live eps-equality across features, edges, warmth, node_edge_index")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify cache vs live feature builder parity (golden test)"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Path to co-sim dataset dir (must contain system_state_captured_unique.json)",
    )
    parser.add_argument(
        "--layout",
        type=str,
        default="dim24",
        choices=["dim14", "dim22", "dim24", "atomic21", "dim25cr"],
        help="Feature layout to test (dim25cr = dim22 + the P5b candidate-relative columns)",
    )
    parser.add_argument(
        "--queue-norm",
        type=str,
        default="scheduler_adaptive",
        choices=["scheduler_adaptive", "adaptive", "adaptive_nonzero", "fixed"],
        help="Queue normalization mode (cache name; live adaptive aliases applied)",
    )
    parser.add_argument(
        "--priors",
        type=Path,
        default=DEFAULT_PRIORS,
        help="Path to task-types.json priors",
    )
    parser.add_argument(
        "--queue-depth-scale",
        type=float,
        default=0.0,
        help=(
            "Replace the recorded queue snapshot with a synthetic depth spread scaled by "
            "this factor (0 = use as recorded). Needed because idle fixtures leave dim7/dim13 "
            "at zero on both sides; use e.g. 1 for training-scale and 400 for live-scale depth."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose progress output",
    )
    args = parser.parse_args()

    ds_path = args.dataset
    if ds_path is None:
        ds_path = DEFAULT_DATASET
        if not ds_path.exists():
            # Fallback: any oracle_split ds_* with SSC
            parent = DEFAULT_DATASET.parent
            candidates = sorted(
                p
                for p in parent.glob("ds_*")
                if (p / "system_state_captured_unique.json").is_file()
            )
            if not candidates:
                print(
                    f"ERROR: No dataset provided and default {DEFAULT_DATASET} not found.",
                    file=sys.stderr,
                )
                print(
                    "Run: pipenv run python3 scripts_cosim/verify_cache_live_feature_parity.py "
                    "--dataset <path>",
                    file=sys.stderr,
                )
                sys.exit(1)
            ds_path = candidates[0]

    ok = verify_parity(
        ds_path.resolve(),
        layout=args.layout,
        queue_norm_mode=args.queue_norm,
        priors_path=args.priors.resolve(),
        verbose=not args.quiet,
        queue_depth_scale=args.queue_depth_scale,
    )
    if ok:
        print("\n=== PARITY TEST PASS ===")
        sys.exit(0)
    print("\n=== PARITY TEST FAIL ===", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()
