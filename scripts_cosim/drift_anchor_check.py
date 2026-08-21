#!/usr/bin/env python3
"""Drift-anchor control for topology_transfer_v1 (see gate_statistics.DRIFT_ANCHOR_NOTE).

Runs the pre-registered no-learning control that the Phase 4 gate never executed: two
closed-form rules of IDENTICAL expressive class (the additive-fit argmin, and that same
fit handed one extra scalar column -- node-occupancy excess) evaluated at each held-out
topology size via separability_diagnostic.variance_decomposition. Neither rule can learn
anything; if their regret trend across held-out sizes is comparable in magnitude to the
trend attributed to gnn_base/gnn_node in the real gate, "the gap widens with size" is not
evidence of topological transfer -- it is drift any fixed-capacity comparison would show.

Usage:
  pipenv run python3 scripts_cosim/drift_anchor_check.py \
      simulation_data/gnn_datasets_4tasks_topo_transfer_v1 --held-out-sizes 60 80
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from scripts_cosim.separability_diagnostic import analyze_dataset, load_combos
from scripts_cosim.gate_statistics import size_trend, DRIFT_ANCHOR_NOTE


def server_count(ds_dir: Path) -> int | None:
    infra_path = ds_dir / "infrastructure.json"
    if not infra_path.exists():
        return None
    infra = json.loads(infra_path.read_text())
    names = infra.get("network_maps", {})
    return sum(1 for n in names if not str(n).startswith("client_node"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus_dir", type=Path)
    ap.add_argument("--held-out-sizes", type=int, nargs="+", default=[60, 80])
    ap.add_argument("--limit-per-size", type=int, default=None,
                     help="cap datasets analyzed per size (default: all)")
    args = ap.parse_args()

    print(DRIFT_ANCHOR_NOTE)
    print()

    per_size: Dict[int, dict] = {}
    per_size_n: Dict[int, int] = {}

    for size in args.held_out_sizes:
        additive_vals: List[float] = []
        aug_vals: List[float] = []
        n_seen = 0
        n_used = 0
        for ds_dir in sorted(args.corpus_dir.glob("ds_*")):
            sc = server_count(ds_dir)
            if sc != size:
                continue
            n_seen += 1
            if args.limit_per_size and n_used >= args.limit_per_size:
                continue
            combos = load_combos(ds_dir)
            if not combos:
                continue
            result = analyze_dataset(ds_dir)
            if result is None:
                continue
            m4 = result.get("m4") or {}
            if m4.get("degenerate"):
                continue
            a = m4.get("additive_choice_regret_rel")
            b = m4.get("additive_plus_collision_choice_regret_rel")
            if a is None or b is None:
                continue
            additive_vals.append(a)
            aug_vals.append(b)
            n_used += 1

        if not additive_vals:
            print(f"size={size}: no usable datasets (seen {n_seen})")
            continue

        per_size[size] = {
            "additive_regret_mean": float(np.mean(additive_vals)),
            "aug_regret_mean": float(np.mean(aug_vals)),
        }
        per_size_n[size] = n_used
        print(f"size={size}: n_datasets={n_used}/{n_seen}  "
              f"additive_regret_mean={np.mean(additive_vals):.4f}  "
              f"aug_regret_mean={np.mean(aug_vals):.4f}")

    print()
    additive_trend = size_trend(per_size, key="additive_regret_mean")
    aug_trend = size_trend(per_size, key="aug_regret_mean")
    print("additive-fit argmin trend across held-out sizes:", additive_trend)
    print("additive+one-integer trend across held-out sizes:", aug_trend)

    if additive_trend.get("total_change") is not None and aug_trend.get("total_change") is not None:
        anchor_drift = additive_trend["total_change"] - aug_trend["total_change"]
        print()
        print(f"DRIFT ANCHOR (identical-capacity pair, total_change delta): {anchor_drift:.4f}")
        print("Compare this magnitude against the gnn_base/gnn_node observed size trend")
        print("from the Phase 4 gate reports before treating that trend as evidence of transfer.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
