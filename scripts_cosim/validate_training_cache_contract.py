#!/usr/bin/env python3
"""Validate that a batch graph cache obeys the training contract.

Fail-loud checks (from research-status / data-leakage canvases):
  1. Every graph has parent_dataset_id
  2. Every y label is present (no -1) and matches placements.jsonl sweep minimum
  3. opt_rtt equals the sweep minimum RTT
  4. Edge is_warm is computed from SSC previous_task_type_name (not all-zero when SSC has warmth)
  5. Canonical-parent split has zero parent overlap
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = PROJECT_ROOT / "src" / "notebooks"
sys.path.insert(0, str(NOTEBOOKS))

from non_unique_lib.training_contract import (  # noqa: E402
    assert_zero_parent_overlap,
    canonical_parent_id,
    combo_from_plan,
    load_sweep_minimum,
    split_ids_by_canonical_parent,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument(
        "--simulation-root",
        type=Path,
        default=PROJECT_ROOT / "simulation_data",
        help="Root containing gnn_datasets_* corpora referenced by dataset_id",
    )
    p.add_argument("--max-graphs", type=int, default=0, help="0 = all graphs")
    p.add_argument("--skip-warmth-audit", action="store_true")
    return p.parse_args()


def _dataset_dir(sim_root: Path, parent_id: str) -> Path:
    path = sim_root / parent_id
    if not path.is_dir():
        raise FileNotFoundError(f"Dataset dir missing for {parent_id}: {path}")
    return path


def _edge_is_warm_values(graph: Any) -> np.ndarray:
    ea = getattr(graph, "edge_attr", None)
    if ea is None or ea.numel() == 0:
        return np.zeros(0, dtype=np.float64)
    arr = ea.detach().cpu().numpy()
    if arr.ndim == 1:
        return arr.astype(np.float64)
    # is_warm is edge feature index 2 in the 5-dim edge attr layout
    n = arr.shape[0]
    if arr.shape[1] <= 2:
        return np.zeros(n, dtype=np.float64)
    return arr[:, 2].astype(np.float64)


def main() -> int:
    args = _parse_args()
    cache_dir = args.cache_dir.resolve()
    meta_path = cache_dir / "metadata.json"
    graphs_path = cache_dir / "graphs.pkl"
    ids_path = cache_dir / "dataset_ids.pkl"
    opt_path = cache_dir / "optimal_rtt.pkl"
    for path in (meta_path, graphs_path, ids_path, opt_path):
        if not path.exists():
            raise FileNotFoundError(path)

    with meta_path.open() as f:
        metadata = json.load(f)
    with graphs_path.open("rb") as f:
        graphs = pickle.load(f)
    with ids_path.open("rb") as f:
        dataset_ids = pickle.load(f)
    with opt_path.open("rb") as f:
        optimal_rtt = pickle.load(f)

    if len(graphs) != len(dataset_ids):
        raise RuntimeError(f"graphs ({len(graphs)}) != dataset_ids ({len(dataset_ids)})")

    n = len(graphs) if args.max_graphs <= 0 else min(args.max_graphs, len(graphs))
    graphs = graphs[:n]
    dataset_ids = dataset_ids[:n]

    failures: List[str] = []
    warmth_checked = 0
    warmth_nonzero_graphs = 0
    label_checked = 0

    for graph, graph_id in zip(graphs, dataset_ids):
        parent = getattr(graph, "parent_dataset_id", None)
        if not parent:
            failures.append(f"{graph_id}: missing parent_dataset_id")
            continue
        parent = canonical_parent_id(parent)
        if canonical_parent_id(graph_id) != parent and not str(graph_id).startswith(parent):
            failures.append(
                f"{graph_id}: parent_dataset_id={parent} inconsistent with graph id"
            )

        try:
            ds_dir = _dataset_dir(args.simulation_root, parent)
        except FileNotFoundError as exc:
            failures.append(str(exc))
            continue

        jsonl = ds_dir / "placements" / "placements.jsonl"
        plan, min_rtt, combo = load_sweep_minimum(jsonl)
        map_rtt = optimal_rtt.get(parent)
        if map_rtt is None:
            map_rtt = optimal_rtt.get(str(graph_id))
        if map_rtt is None:
            failures.append(f"{graph_id}: opt_rtt missing for parent {parent}")
        elif abs(float(map_rtt) - float(min_rtt)) > 1e-9:
            failures.append(
                f"{graph_id}: opt_rtt map {map_rtt} != sweep min {min_rtt}"
            )

        y = graph.y.detach().cpu().numpy()
        if (y < 0).any():
            failures.append(f"{graph_id}: invalid y labels {y.tolist()}")
            continue

        l2p = getattr(graph, "task_logit_to_placement", None) or getattr(
            graph, "_task_logit_to_placement", None
        )
        if l2p is None:
            failures.append(f"{graph_id}: missing task_logit_to_placement")
            continue

        recovered: Dict[str, List[int]] = {}
        ok = True
        for t in range(int(graph.n_tasks)):
            idx = int(y[t])
            mapping = l2p[t] if t < len(l2p) else []
            if idx < 0 or idx >= len(mapping):
                failures.append(f"{graph_id}: y[{t}]={idx} out of range")
                ok = False
                break
            pair = mapping[idx]
            recovered[str(t)] = [int(pair[0]), int(pair[1])]
        if not ok:
            continue

        try:
            recovered_combo = combo_from_plan(recovered)
        except ValueError as exc:
            failures.append(f"{graph_id}: {exc}")
            continue
        if recovered_combo != combo:
            failures.append(
                f"{graph_id}: graph label combo {recovered_combo} != sweep min {combo}"
            )
        else:
            label_checked += 1

        if not args.skip_warmth_audit:
            ssc_path = ds_dir / "system_state_captured_unique.json"
            if ssc_path.is_file():
                with ssc_path.open() as f:
                    ssc = json.load(f)
                tp0 = (ssc.get("task_placements") or [{}])[0]
                temporal = (
                    tp0.get("full_temporal_state_at_scheduling")
                    or ssc.get("full_temporal_state_at_scheduling")
                    or {}
                )
                ssc_has_prev = any(
                    isinstance(v, dict) and v.get("previous_task_type_name") is not None
                    for v in temporal.values()
                )
                warm_vals = _edge_is_warm_values(graph)
                warmth_checked += 1
                if warm_vals.size and float(warm_vals.max()) > 0:
                    warmth_nonzero_graphs += 1
                elif ssc_has_prev and warm_vals.size:
                    failures.append(
                        f"{graph_id}: SSC has previous_task_type_name but all edge is_warm=0"
                    )

    train_g, train_ids, val_g, val_ids, test_g, test_ids = split_ids_by_canonical_parent(
        graphs, dataset_ids
    )
    assert_zero_parent_overlap(train_ids, val_ids, test_ids)

    report = {
        "cache_dir": str(cache_dir),
        "cache_version": metadata.get("version"),
        "training_contract_meta": metadata.get("training_contract"),
        "graphs_audited": n,
        "label_matches_sweep_min": label_checked,
        "warmth_graphs_checked": warmth_checked,
        "warmth_nonzero_graphs": warmth_nonzero_graphs,
        "split_counts": {
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
            "train_parents": len({canonical_parent_id(x) for x in train_ids}),
            "val_parents": len({canonical_parent_id(x) for x in val_ids}),
            "test_parents": len({canonical_parent_id(x) for x in test_ids}),
        },
        "failures": failures[:50],
        "num_failures": len(failures),
    }
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(f"CACHE CONTRACT FAILED: {len(failures)} issues")
    print("CACHE CONTRACT OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
