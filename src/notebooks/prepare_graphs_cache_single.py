#!/usr/bin/env python3
"""
Build 1-task graph cache for Regime B tabular training (marginal oracle co-sim).

One graph per dataset (n_tasks=1, seq_step=0). Compatible with prepare_tabular_dataset.py
--regime single when metadata.single_task=true.

Usage:
  cd /root/projects/my-herosim
  PYTHONPATH=.:src/notebooks pipenv run python3 src/notebooks/prepare_graphs_cache_single.py \\
    --base-dir simulation_data/artifacts/run_queue_big/gnn_datasets_1task \\
    --cache-dir simulation_data/artifacts/run_queue_big/graphs_cache_gnn_datasets_1task
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOKS_ROOT = PROJECT_ROOT / "src" / "notebooks"
for p in (str(PROJECT_ROOT), str(NOTEBOOKS_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from prepare_graphs_cache_seq import (  # noqa: E402
    CACHE_VERSION,
    build_graph,
    load_all_datasets,
)

SINGLE_CACHE_VERSION = f"1.0-single-tabular-{CACHE_VERSION}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 1-task graph cache for Regime B XGBoost.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        required=True,
        help="gnn_datasets_1task root (ds_* subdirs)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Output cache directory",
    )
    parser.add_argument(
        "--priors-path",
        type=Path,
        default=PROJECT_ROOT / "data/nofs-ids/task-types.json",
    )
    parser.add_argument(
        "--queue-norm-mode",
        type=str,
        default="scheduler_adaptive",
        choices=("fixed", "scheduler_adaptive", "adaptive_nonzero"),
    )
    return parser.parse_args()


def build_single_graph_for_dataset(
    dataset_id: str,
    dataset_dict: dict,
    *,
    task_priors: dict,
    queue_norm_factor: float,
    queue_norm_mode: str,
):
    n_tasks = int(dataset_dict.get("num_tasks", len(dataset_dict["tasks"])))
    if n_tasks != 1:
        raise ValueError(f"{dataset_id}: expected 1 task, got {n_tasks}")

    df_nodes = dataset_dict["nodes"]
    df_tasks = dataset_dict["tasks"]
    df_platforms = dataset_dict["platforms"]
    queue_snapshot = dataset_dict.get("queue_snapshot") or {}
    temporal_state = dataset_dict.get("temporal_state") or {}
    initialized_snapshot = dataset_dict.get("initialized_snapshot") or {}

    graph = build_graph(
        df_nodes,
        df_tasks,
        df_platforms,
        task_priors=task_priors,
        queue_norm_factor=queue_norm_factor,
        queue_norm_mode=queue_norm_mode,
        queue_snapshot=queue_snapshot,
        temporal_state=temporal_state,
        initialized_snapshot=initialized_snapshot,
    )

    graph.dataset_id = dataset_id
    graph.parent_dataset_id = dataset_id
    graph.seq_step = 0
    graph.seq_n_tasks = 1
    graph.prefix_augment = False
    return graph


def main() -> None:
    args = parse_args()
    base_dir = args.base_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not base_dir.exists():
        raise FileNotFoundError(f"Base dir not found: {base_dir}")
    if not args.priors_path.exists():
        raise FileNotFoundError(f"Priors not found: {args.priors_path}")

    with open(args.priors_path, "r", encoding="utf-8") as f:
        task_priors = json.load(f)

    all_datasets = load_all_datasets([base_dir], require_queue_data=True)
    if not all_datasets:
        raise RuntimeError(f"No datasets loaded from {base_dir}")

    graphs = []
    dataset_ids = []
    skipped = 0
    queue_norm_factor = 50.0

    for dataset_id, dataset_dict in tqdm(all_datasets.items(), desc="single graphs"):
        try:
            graph = build_single_graph_for_dataset(
                dataset_id,
                dataset_dict,
                task_priors=task_priors,
                queue_norm_factor=queue_norm_factor,
                queue_norm_mode=args.queue_norm_mode,
            )
            y = getattr(graph, "y", None)
            if y is None or int(y[0].item()) < 0:
                skipped += 1
                continue
            graphs.append(graph)
            dataset_ids.append(dataset_id)
        except Exception as exc:
            skipped += 1
            tqdm.write(f"  skip {dataset_id}: {exc}")

    if not graphs:
        raise RuntimeError(f"No graphs built (skipped={skipped})")

    with open(cache_dir / "graphs.pkl", "wb") as f:
        pickle.dump(graphs, f)
    with open(cache_dir / "dataset_ids.pkl", "wb") as f:
        pickle.dump(dataset_ids, f)

    metadata = {
        "version": SINGLE_CACHE_VERSION,
        "single_task": True,
        "sequential_counterfactual": False,
        "num_graphs": len(graphs),
        "base_dir": str(base_dir),
        "queue_norm_mode": args.queue_norm_mode,
        "skipped": skipped,
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(cache_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[+] Wrote {len(graphs)} single-task graphs -> {cache_dir} (skipped={skipped})")


if __name__ == "__main__":
    main()
