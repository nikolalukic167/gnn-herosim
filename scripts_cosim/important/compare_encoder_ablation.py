#!/usr/bin/env python3
"""Score the Stage 1 ceiling + encoder/decode ablation ladder.

Joins the ablation sweep against the baseline trio sweep so the GNN control arm
(full GIN + argmax), Knative and the MLP are read from the run that already
produced them rather than re-simulated.

Emits per-cell total_rtt, p50/p90/p99, and the two decode telemetry numbers that
have to move for an RTT win to be believable (`chosen_queue_vs_min`,
`collision_batch_rate`). An RTT improvement with those flat means something other
than placement quality changed.

Usage:
    pipenv run python3 scripts_cosim/important/compare_encoder_ablation.py \
        --sweep-dir simulation_data/normal_sim_sweeps/gnn_encoder_ablation_20260816
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts_cosim.sweep_metrics import load_metrics  # noqa: E402

# Ladder arm -> what it isolates. Order is the reporting order.
ARM_LABELS = {
    "knative": "Knative (reactive baseline)",
    "ect": "ECT (physics-aware greedy ceiling)",
    "ect_pull": "ECT + FilterStore pull ceiling",
    "mlp_dim22": "MLP (pointwise baseline)",
    "gnn": "GNN control: full GIN + argmax",
    "gnn_dropnodeedges": "GNN rung B: drop same-node MP edges (train/serve parity)",
    "gnn_nomp": "GNN rung C: message passing off",
    "gnn_uniq": "GNN rung D: argmax_uniq",
    "gnn_lqb15": "GNN rung E: logit - 1.5*log1p(queue)",
    "gnn_qfilter8": "GNN rung F: queue filter delta=8",
}


def decode_stats_for(result_path: Path) -> Dict[str, Any]:
    sidecar = result_path.with_suffix(".decode_stats.json")
    if not sidecar.is_file():
        return {}
    return json.loads(sidecar.read_text())


def collect(results_dir: Path, cell: str, seed: int, tag: str) -> Optional[Dict[str, Any]]:
    path = results_dir / f"{cell}_s{seed}_{tag}.json"
    if not path.is_file():
        return None
    row: Dict[str, Any] = dict(load_metrics(path))
    ds = decode_stats_for(path)
    if ds:
        row["qvsmin_mean"] = ds.get("chosen_queue_vs_min", {}).get("mean")
        row["qvsmin_median"] = ds.get("chosen_queue_vs_min", {}).get("median")
        row["collision_rate"] = ds.get("intra_batch_platform_collisions", {}).get(
            "collision_batch_rate"
        )
        uniq = ds.get("uniq_platform") or {}
        row["uniq_relaxed_rate"] = uniq.get("relaxed_rate")
    return row


def fmt(value: Any, width: int, spec: str = "") -> str:
    if value is None:
        return "-".rjust(width)
    if spec:
        return format(value, spec).rjust(width)
    return str(value).rjust(width)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-dir", type=Path, required=True)
    parser.add_argument(
        "--baseline-sweep",
        type=Path,
        default=PROJECT_ROOT
        / "simulation_data/normal_sim_sweeps/full_corpus_siv1_coupled_trio_20260815",
        help="Sweep holding the GNN control, Knative and MLP arms.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    results = args.sweep_dir / "results"
    baseline = args.baseline_sweep / "results"
    if not results.is_dir():
        raise SystemExit(f"FAIL LOUD: missing results dir {results}")
    if not baseline.is_dir():
        raise SystemExit(f"FAIL LOUD: missing baseline results dir {baseline}")

    cells = sorted({p.name.split("_s")[0] for p in results.glob("*_s*.json")
                    if not p.name.endswith(".decode_stats.json")})
    if not cells:
        raise SystemExit(f"FAIL LOUD: no result JSONs under {results}")

    report: Dict[str, Any] = {
        "sweep_dir": str(args.sweep_dir),
        "baseline_sweep": str(args.baseline_sweep),
        "seed": args.seed,
        "cells": {},
    }

    for cell in cells:
        rows: Dict[str, Dict[str, Any]] = {}
        for tag in ARM_LABELS:
            # Control/baseline arms live in the baseline sweep; ladder arms locally.
            src = baseline if tag in ("knative", "mlp_dim22", "gnn") else results
            row = collect(src, cell, args.seed, tag)
            if row is not None:
                rows[tag] = row
        if not rows:
            continue

        kn = rows.get("knative", {}).get("total_rtt")
        print(f"\n=== {cell} (seed {args.seed}) ===")
        header = (
            f"{'arm':<52} {'total_rtt':>14} {'x_kn':>7} {'p50':>9} "
            f"{'p90':>9} {'p99':>9} {'qvsmin':>10} {'collide':>8}"
        )
        print(header)
        print("-" * len(header))
        for tag, row in rows.items():
            ratio = (row["total_rtt"] / kn) if kn else None
            print(
                f"{ARM_LABELS[tag]:<52}"
                f"{fmt(row.get('total_rtt'), 15, ',.0f')}"
                f"{fmt(ratio, 8, '.2f')}"
                f"{fmt(row.get('p50'), 10, '.1f')}"
                f"{fmt(row.get('p90'), 10, '.1f')}"
                f"{fmt(row.get('p99'), 10, '.1f')}"
                f"{fmt(row.get('qvsmin_mean'), 11, ',.0f')}"
                f"{fmt(row.get('collision_rate'), 9, '.3f')}"
            )
            relaxed = row.get("uniq_relaxed_rate")
            if relaxed is not None and relaxed > 0.05:
                print(
                    f"{'':<52}  ^ uniqueness relaxed on {relaxed:.1%} of tasks "
                    "— closer to plain argmax than to a uniq decode"
                )
        # Physics must match or the cells are not comparable.
        physics = {row.get("warmth_physics") for row in rows.values()}
        if len(physics) > 1:
            raise SystemExit(f"FAIL LOUD: {cell} mixes warmth_physics {physics}")
        report["cells"][cell] = {
            "warmth_physics": physics.pop() if physics else None,
            "arms": {tag: row for tag, row in rows.items()},
        }

    out = args.report or (args.sweep_dir / "compare.json")
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
