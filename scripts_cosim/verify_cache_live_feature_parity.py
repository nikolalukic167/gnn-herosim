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
from src.policy.tabular.feature_builder import build_pyg_inference_graph  # noqa: E402

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
    if layout in ("dim24", "24", "pull_obs", "pull_observables"):
        return build_cache_graph_dim24(
            **common, queue_feature_contract=queue_feature_contract
        )

    if layout in ("dim22", "legacy", "22"):
        # dim22 = dim24 platform features without cold_count / pull_remaining.
        g = build_cache_graph_dim24(**common, queue_feature_contract=queue_feature_contract)
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
    if layout in ("dim22", "legacy", "22", "dim14", "atomic21", "14"):
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
    temporal = {
        str(k): {kk: float(vv) for kk, vv in (v or {}).items()}
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
    checks: List[Tuple[str, Any, Any]] = [
        ("task_features", cache_g.task_features, live_g.task_features),
        ("platform_features", cache_g.platform_features, live_g.platform_features),
        ("edge_attr", cache_g.edge_attr, live_g.edge_attr),
        (
            "edge_attr_directed",
            _directed_edge_attr(cache_g.edge_attr),
            _directed_edge_attr(live_g.edge_attr),
        ),
        (
            "edge_index_undirected_pairs",
            _edge_index_as_sorted_pairs(cache_g.edge_index),
            _edge_index_as_sorted_pairs(live_g.edge_index),
        ),
    ]

    cache_nei = getattr(cache_g, "node_edge_index", None)
    live_nei = getattr(live_g, "node_edge_index", None)
    if cache_nei is None:
        failures.append("cache missing node_edge_index")
    if live_nei is None:
        failures.append("live missing node_edge_index (unwired same-node edges)")
    if cache_nei is not None and live_nei is not None:
        checks.append(
            (
                "node_edge_index_pairs",
                _node_edge_pairs(cache_nei),
                _node_edge_pairs(live_nei),
            )
        )

    for name, c_arr, l_arr in checks:
        ok, msg = _compare_arrays(name, c_arr, l_arr)
        if not ok:
            failures.append(msg)

    # Per-dim platform report when platform_features diverge
    cp = _safe_array(cache_g.platform_features)
    lp = _safe_array(live_g.platform_features)
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
    if cache_place != live_placement:
        failures.append(
            f"task_logit_to_placement mismatch: cache_keys={sorted(cache_place or {})} "
            f"live_keys={sorted(live_placement or {})}"
        )
        # Detailed first mismatch
        for t_idx in sorted(set(cache_place or {}) | set(live_placement or {})):
            if (cache_place or {}).get(t_idx) != (live_placement or {}).get(t_idx):
                failures.append(
                    f"  task {t_idx}: cache={(cache_place or {}).get(t_idx)} "
                    f"live={(live_placement or {}).get(t_idx)}"
                )
                break

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
        choices=["dim14", "dim22", "dim24", "atomic21"],
        help="Feature layout to test",
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
