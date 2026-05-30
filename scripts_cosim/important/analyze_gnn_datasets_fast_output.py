#!/usr/bin/env python3
"""
Analyze datasets produced by scripts_cosim/generate_gnn_datasets_fast.py.

Outputs:
- CSV with one row per dataset
- JSON summaries with structure and aggregate statistics
- PNG graphs for fast visual inspection

Usage:
    pipenv run python3 scripts_cosim/important/analyze_gnn_datasets_fast_output.py
    pipenv run python3 scripts_cosim/important/analyze_gnn_datasets_fast_output.py \
        --dataset-root simulation_data/gnn_datasets_5tasks \
        --output-dir logs/gnn_data_analysis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def compute_queue_intensity(params: Dict[str, Any]) -> float:
    q_type = params.get("type")
    if q_type == "poisson":
        return safe_float(params.get("lambda", 0.0))
    if q_type == "normal":
        return safe_float(params.get("mean", 0.0))
    if q_type == "uniform":
        low = safe_float(params.get("low", 0.0))
        high = safe_float(params.get("high", 0.0))
        return (low + high) / 2.0
    if q_type == "constant":
        return safe_float(params.get("value", 0.0))
    return float("nan")


def compute_queue_label(params: Dict[str, Any]) -> str:
    q_type = params.get("type", "unknown")
    if q_type == "poisson":
        return f"poisson:{params.get('lambda', 'na')}"
    if q_type == "normal":
        return f"normal:{params.get('mean', 'na')}"
    if q_type == "uniform":
        return f"uniform:{params.get('low', 'na')}-{params.get('high', 'na')}"
    if q_type == "constant":
        return f"constant:{params.get('value', 'na')}"
    return str(q_type)


def extract_workload_mix(workload: Dict[str, Any]) -> Dict[str, int]:
    events = workload.get("events", [])
    dnn1 = 0
    dnn2 = 0
    for event in events:
        app = event.get("application", {})
        name = str(app.get("name", ""))
        if "dnn1" in name:
            dnn1 += 1
        elif "dnn2" in name:
            dnn2 += 1
    return {"dnn1_tasks": dnn1, "dnn2_tasks": dnn2, "num_tasks": len(events)}


def summarize_structure(dataset_dirs: List[Path]) -> Dict[str, Any]:
    required = [
        "best.json",
        "space_with_network.json",
        "workload.json",
        "optimal_result.json",
        "infrastructure.json",
        "placement_metadata.json",
    ]
    optional = [
        "system_state_captured_unique.json",
        "placement_progress.txt",
        "placements/placements.jsonl",
    ]

    structure = {
        "dataset_count": len(dataset_dirs),
        "required_file_coverage": {},
        "optional_file_coverage": {},
        "sample_keys": {},
    }

    for rel in required:
        found = sum((d / rel).exists() for d in dataset_dirs)
        structure["required_file_coverage"][rel] = {
            "count": found,
            "pct": (100.0 * found / len(dataset_dirs)) if dataset_dirs else 0.0,
        }

    for rel in optional:
        found = sum((d / rel).exists() for d in dataset_dirs)
        structure["optional_file_coverage"][rel] = {
            "count": found,
            "pct": (100.0 * found / len(dataset_dirs)) if dataset_dirs else 0.0,
        }

    if dataset_dirs:
        sample_dir = dataset_dirs[0]
        for rel in ["best.json", "space_with_network.json", "workload.json", "optimal_result.json"]:
            p = sample_dir / rel
            if p.exists():
                obj = load_json(p)
                if isinstance(obj, dict):
                    structure["sample_keys"][rel] = sorted(obj.keys())
                else:
                    structure["sample_keys"][rel] = str(type(obj))

    return structure


def analyze_dataset(dataset_dir: Path) -> Dict[str, Any]:
    row: Dict[str, Any] = {"dataset_id": dataset_dir.name}

    best = load_json(dataset_dir / "best.json")
    cfg = load_json(dataset_dir / "space_with_network.json")
    workload = load_json(dataset_dir / "workload.json")
    metadata = load_json(dataset_dir / "placement_metadata.json")
    optimal = load_json(dataset_dir / "optimal_result.json")

    row["rtt"] = safe_float(best.get("rtt"))

    topology = cfg.get("network", {}).get("topology", {})
    row["connection_probability"] = safe_float(topology.get("connection_probability"))
    row["seed"] = topology.get("seed")

    preinit = cfg.get("preinit", {})
    row["preinit_client_pct"] = safe_float(preinit.get("client_percentage"))
    row["preinit_server_pct"] = safe_float(preinit.get("server_percentage"))

    replicas = cfg.get("replicas", {}).get("dnn1", {})
    row["replicas_per_client"] = replicas.get("per_client")
    row["replicas_per_server"] = replicas.get("per_server")

    qparams = (
        cfg.get("prewarm", {})
        .get("dnn1", {})
        .get("queue_distribution_params", {})
    )
    row["queue_type"] = qparams.get("type", "unknown")
    row["queue_label"] = compute_queue_label(qparams)
    row["queue_intensity"] = compute_queue_intensity(qparams)

    mix = extract_workload_mix(workload)
    row.update(mix)
    row["dnn1_ratio"] = (mix["dnn1_tasks"] / mix["num_tasks"]) if mix["num_tasks"] else float("nan")

    row["num_placements"] = int(metadata.get("num_placements", 0))
    row["num_completed_placements"] = int(metadata.get("completed", 0))

    stats = optimal.get("stats", {})
    row["avg_elapsed_time"] = safe_float(stats.get("averageElapsedTime"))
    row["avg_execution_time"] = safe_float(stats.get("averageExecutionTime"))
    row["avg_cold_start_time"] = safe_float(stats.get("averageColdStartTime"))
    row["avg_queue_time"] = safe_float(stats.get("averageQueueTime"))

    return row


def make_plots(df: pd.DataFrame, output_dir: Path) -> None:
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 6))
    sns.histplot(df["rtt"].dropna(), bins=30, kde=True)
    plt.title("RTT Distribution Across Datasets")
    plt.xlabel("RTT (seconds)")
    plt.ylabel("Dataset count")
    plt.tight_layout()
    plt.savefig(output_dir / "rtt_distribution.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    order = sorted(df["connection_probability"].dropna().unique())
    sns.boxplot(data=df, x="connection_probability", y="rtt", order=order)
    plt.title("RTT by Connection Probability")
    plt.xlabel("Connection probability")
    plt.ylabel("RTT (seconds)")
    plt.tight_layout()
    plt.savefig(output_dir / "rtt_by_connection_probability.png", dpi=180)
    plt.close()

    plt.figure(figsize=(11, 6))
    sns.scatterplot(
        data=df,
        x="queue_intensity",
        y="rtt",
        hue="queue_type",
        alpha=0.75,
        s=60,
    )
    plt.title("RTT vs Queue Intensity")
    plt.xlabel("Queue intensity (distribution central value)")
    plt.ylabel("RTT (seconds)")
    plt.tight_layout()
    plt.savefig(output_dir / "rtt_vs_queue_intensity.png", dpi=180)
    plt.close()

    pivot = (
        df.groupby(["connection_probability", "queue_type"])["dataset_id"]
        .count()
        .unstack(fill_value=0)
        .sort_index()
    )
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="Blues")
    plt.title("Coverage: datasets per topology and queue type")
    plt.xlabel("Queue type")
    plt.ylabel("Connection probability")
    plt.tight_layout()
    plt.savefig(output_dir / "coverage_topology_queue_heatmap.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 6))
    mix_counts = (
        df["dnn1_ratio"]
        .round(2)
        .value_counts()
        .sort_index()
    )
    ax = sns.barplot(x=mix_counts.index.astype(str), y=mix_counts.values)
    ax.set_title("Workload mix distribution")
    ax.set_xlabel("dnn1 ratio")
    ax.set_ylabel("Dataset count")
    plt.tight_layout()
    plt.savefig(output_dir / "workload_mix_distribution.png", dpi=180)
    plt.close()


def build_summary(df: pd.DataFrame) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "dataset_count": int(len(df)),
        "rtt": {
            "mean": float(df["rtt"].mean()),
            "median": float(df["rtt"].median()),
            "min": float(df["rtt"].min()),
            "max": float(df["rtt"].max()),
            "p10": float(df["rtt"].quantile(0.10)),
            "p90": float(df["rtt"].quantile(0.90)),
        },
        "placements": {
            "mean": float(df["num_placements"].mean()),
            "median": float(df["num_placements"].median()),
            "min": int(df["num_placements"].min()),
            "max": int(df["num_placements"].max()),
        },
        "connection_probability_counts": {
            str(k): int(v) for k, v in df["connection_probability"].value_counts().sort_index().items()
        },
        "queue_type_counts": {
            str(k): int(v) for k, v in df["queue_type"].value_counts().sort_index().items()
        },
        "workload_mix_counts": {
            str(k): int(v) for k, v in df["dnn1_ratio"].round(2).value_counts().sort_index().items()
        },
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze generated GNN datasets with plots.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=PROJECT_ROOT / "simulation_data" / "gnn_datasets_5tasks",
        help="Path containing ds_XXXXX dataset folders",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "logs" / "gnn_data_analysis",
        help="Directory for CSV, JSON summaries and PNG plots",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_dirs = sorted(
        [p for p in dataset_root.iterdir() if p.is_dir() and p.name.startswith("ds_")]
    )
    if not dataset_dirs:
        raise SystemExit(f"No dataset directories found under: {dataset_root}")

    structure = summarize_structure(dataset_dirs)
    rows: List[Dict[str, Any]] = []
    skipped = 0

    for ds in dataset_dirs:
        required_paths = [
            ds / "best.json",
            ds / "space_with_network.json",
            ds / "workload.json",
            ds / "placement_metadata.json",
            ds / "optimal_result.json",
        ]
        if not all(p.exists() for p in required_paths):
            skipped += 1
            continue
        try:
            rows.append(analyze_dataset(ds))
        except Exception as exc:
            skipped += 1
            print(f"Skipping {ds.name}: {exc}")

    if not rows:
        raise SystemExit("No complete datasets could be analyzed.")

    df = pd.DataFrame(rows).sort_values("dataset_id").reset_index(drop=True)
    df.to_csv(output_dir / "dataset_metrics.csv", index=False)

    summary = build_summary(df)
    summary["skipped_datasets"] = int(skipped)

    with open(output_dir / "structure_summary.json", "w", encoding="utf-8") as f:
        json.dump(structure, f, indent=2)
    with open(output_dir / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    make_plots(df, output_dir)

    print(f"Analyzed datasets: {len(df)}")
    print(f"Skipped datasets: {skipped}")
    print(f"Output directory: {output_dir}")
    print("Generated files:")
    for name in [
        "dataset_metrics.csv",
        "structure_summary.json",
        "analysis_summary.json",
        "rtt_distribution.png",
        "rtt_by_connection_probability.png",
        "rtt_vs_queue_intensity.png",
        "coverage_topology_queue_heatmap.png",
        "workload_mix_distribution.png",
    ]:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
