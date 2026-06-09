#!/usr/bin/env python3
"""
Extract Option B tabular datasets (22-d edge rows) from a sequential graph cache.

Regime batch: all @seq{s} graphs (task_idx == seq_step invariant).
Regime single: only @seq0 graphs (bootstrap for Regime B; see per_arrival plan for co-sim).

Usage (do not run until cache is rebuilt with platform_pos):
  pipenv run python3 src/notebooks/prepare_tabular_dataset.py \\
    --cache-dir simulation_data/artifacts/run_queue_big/graphs_cache_gnn_datasets_4tasks_seq \\
    --output simulation_data/artifacts/tabular/batch_edges.parquet \\
    --regime batch
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.policy.tabular.graph_extraction import (  # noqa: E402
    extract_rows_from_graph,
    rows_to_dataframe,
    should_emit_graph,
    validate_extracted_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Option B tabular edge dataset from sequential PyG cache."
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Directory containing graphs.pkl, dataset_ids.pkl, metadata.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path (.parquet or .csv)",
    )
    parser.add_argument(
        "--regime",
        choices=("batch", "single"),
        default="batch",
        help="batch: all seq steps; single: only seq_step==0 (no @seqneg1)",
    )
    parser.add_argument(
        "--include-prefix-augment",
        action="store_true",
        help="Include @seqneg1 hard-negative graphs (default: exclude)",
    )
    return parser.parse_args()


def load_sequential_cache(cache_dir: Path):
    meta_path = cache_dir / "metadata.json"
    graphs_path = cache_dir / "graphs.pkl"
    ids_path = cache_dir / "dataset_ids.pkl"

    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {cache_dir}")
    if not graphs_path.exists():
        raise FileNotFoundError(f"graphs.pkl not found in {cache_dir}")
    if not ids_path.exists():
        raise FileNotFoundError(f"dataset_ids.pkl not found in {cache_dir}")

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    if not metadata.get("sequential_counterfactual"):
        raise ValueError(
            f"Cache at {cache_dir} is not sequential counterfactual "
            "(metadata.sequential_counterfactual != true). "
            "Run prepare_graphs_cache_seq.py first."
        )

    with open(graphs_path, "rb") as f:
        graphs = pickle.load(f)
    with open(ids_path, "rb") as f:
        dataset_ids = pickle.load(f)

    if len(graphs) != len(dataset_ids):
        raise ValueError(
            f"graphs ({len(graphs)}) and dataset_ids ({len(dataset_ids)}) length mismatch"
        )
    return metadata, graphs, dataset_ids


def main() -> None:
    args = parse_args()
    cache_dir = args.cache_dir.resolve()
    metadata, graphs, dataset_ids = load_sequential_cache(cache_dir)

    exclude_neg = not args.include_prefix_augment
    all_rows = []
    skip_reasons: Counter[str] = Counter()
    emitted_graphs = 0

    for graph, graph_id in tqdm(
        zip(graphs, dataset_ids), total=len(graphs), desc="extract"
    ):
        if not should_emit_graph(
            graph, regime=args.regime, exclude_prefix_augment=exclude_neg
        ):
            continue

        rows, skip_reason = extract_rows_from_graph(graph, graph_id)
        if skip_reason:
            skip_reasons[skip_reason] += 1
            continue
        if not rows:
            skip_reasons["empty_rows"] += 1
            continue

        all_rows.extend(rows)
        emitted_graphs += 1

    if not all_rows:
        raise RuntimeError(
            f"No rows extracted (regime={args.regime}). Skip reasons: {dict(skip_reasons)}"
        )

    df = rows_to_dataframe(all_rows)
    stats = validate_extracted_frame(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    suffix = args.output.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(args.output, index=False)
    elif suffix == ".csv":
        df.to_csv(args.output, index=False)
    else:
        raise ValueError(f"Unsupported output format {suffix!r}; use .parquet or .csv")

    sidecar = {
        "cache_dir": str(cache_dir),
        "cache_version": metadata.get("version"),
        "regime": args.regime,
        "exclude_prefix_augment": exclude_neg,
        "emitted_graphs": emitted_graphs,
        "skip_reasons": dict(skip_reasons),
        **stats,
    }
    sidecar_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)

    print(f"[+] Wrote {stats['num_rows']} rows ({stats['num_graphs']} graphs) -> {args.output}")
    print(f"[+] Sidecar metadata -> {sidecar_path}")
    if skip_reasons:
        print(f"[!] Skipped graphs: {dict(skip_reasons)}")


if __name__ == "__main__":
    main()
