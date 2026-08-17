#!/usr/bin/env python3
from __future__ import annotations

"""Prepare a HeteroData graph cache for task-to-platform placement."""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import prepare_graphs_cache

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.policy.gnn_hetero.data import homogeneous_to_hetero


def _arg_value(args: list[str], name: str) -> str | None:
    if name not in args:
        return None
    idx = args.index(name)
    if idx + 1 >= len(args):
        return None
    return args[idx + 1]


def _arg_values(args: list[str], name: str) -> list[str]:
    if name not in args:
        return []
    idx = args.index(name) + 1
    values: list[str] = []
    while idx < len(args) and not args[idx].startswith("--"):
        values.append(args[idx])
        idx += 1
    return values


def resolve_cache_dir(project_root: Path, args: list[str]) -> Path:
    explicit_cache = _arg_value(args, "--cache-dir")
    if explicit_cache:
        return Path(explicit_cache).expanduser().resolve()

    base_dirs = _arg_values(args, "--base-dirs")
    if base_dirs:
        first_base = Path(base_dirs[0]).expanduser()
        if not first_base.is_absolute():
            first_base = project_root / first_base
        if "--merge-datasets" in args:
            return first_base.parent / "graphs_cache_merged_2_3_4_tasks_hetero"
        return first_base.parent / f"graphs_cache_{first_base.name}_hetero"

    base = (
        project_root
        / "simulation_data"
        / "artifacts"
        / "run_queue_big"
        / "gnn_datasets_4tasks_1060"
    )
    return base.parent / f"graphs_cache_{base.name}_scheduler_adaptive_hetero"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare HeteroData graph cache.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--skip-cache-build",
        action="store_true",
        help="Only convert an existing homogeneous-compatible graphs.pkl in the target cache dir.",
    )
    known, _ = parser.parse_known_args(argv)
    return known


def _argv_for_prepare(argv: list[str], cache_dir: Path) -> list[str]:
    out: list[str] = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg == "--skip-cache-build":
            continue
        out.append(arg)
    if "--cache-dir" not in out:
        out.extend(["--cache-dir", str(cache_dir)])
    return out


def convert_cache_to_hetero(cache_dir: Path) -> None:
    graphs_path = cache_dir / "graphs.pkl"
    metadata_path = cache_dir / "metadata.json"
    if not graphs_path.exists():
        raise FileNotFoundError(f"Missing graphs cache: {graphs_path}")

    with open(graphs_path, "rb") as fh:
        graphs = pickle.load(fh)

    hetero_graphs = [homogeneous_to_hetero(graph) for graph in graphs]
    with open(graphs_path, "wb") as fh:
        pickle.dump(hetero_graphs, fh, protocol=pickle.HIGHEST_PROTOCOL)

    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        with open(metadata_path, "r") as fh:
            metadata = json.load(fh)
    metadata["hetero_data"] = True
    metadata["hetero_node_types"] = ["task", "platform"]
    metadata["hetero_edge_types"] = [
        ["task", "can_run_on", "platform"],
        ["platform", "rev_can_run_on", "task"],
    ]
    metadata["hetero_note"] = "graphs.pkl stores PyG HeteroData with compatible placement/logit metadata."
    with open(metadata_path, "w") as fh:
        json.dump(metadata, fh, indent=2)

    total_edges = sum(
        int(graph["task", "can_run_on", "platform"].edge_index.size(1))
        for graph in hetero_graphs
    )
    print(f"[hetero cache] Converted {len(hetero_graphs)} graphs ({total_edges:,} task->platform edges)")


def main() -> None:
    hetero_args = _parse_args(sys.argv[1:])
    project_root = hetero_args.project_root.expanduser().resolve()
    cache_dir = resolve_cache_dir(project_root, sys.argv[1:])

    if not hetero_args.skip_cache_build:
        original_argv = sys.argv
        try:
            sys.argv = [original_argv[0], *_argv_for_prepare(original_argv[1:], cache_dir)]
            prepare_graphs_cache.main()
        finally:
            sys.argv = original_argv

    convert_cache_to_hetero(cache_dir)


if __name__ == "__main__":
    main()
