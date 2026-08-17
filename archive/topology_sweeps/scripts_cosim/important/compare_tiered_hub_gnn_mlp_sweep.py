#!/usr/bin/env python3
"""Compare tiered-hub sweep: dim22 vs atomic21 GNN/MLP + Knative."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

K_CORE_VALUES = (2, 4, 6, 8)
HUB_SEEKER_FRACTIONS = (0.3, 0.5, 0.8)

POLICIES = (
    ("gnn_dim22", "gnn_dim22"),
    ("gnn_atomic21", "gnn_atomic21"),
    ("mlp_dim22", "mlp_dim22"),
    ("mlp_atomic21", "mlp_atomic21"),
    ("knative", "knative"),
)


def seek_suffix(fraction: float) -> str:
    return f"seek{int(round(fraction * 100)):02d}"


def config_name(k_core: int, hub_seeker_fraction: float) -> str:
    return f"hub_k{k_core}_{seek_suffix(hub_seeker_fraction)}"


def load_rtt(path: Path) -> Optional[float]:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    rtt = data.get("total_rtt")
    if rtt is None or not isinstance(rtt, (int, float)) or rtt <= 0:
        return None
    return float(rtt)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, required=True)
    args = parser.parse_args()

    res = args.sweep_dir / "results"
    if not res.is_dir():
        raise FileNotFoundError(f"Missing results dir: {res}")

    header = f"{'Config':<18}" + "".join(f" {label:>12}" for _, label in POLICIES) + f" {'Best':>12}"
    print(f"\nTiered hub sweep — {args.sweep_dir}\n")
    print(header)
    print("-" * len(header))

    win_counts = {label: 0 for _, label in POLICIES}
    complete_rows = 0

    for k_core in K_CORE_VALUES:
        for hub_frac in HUB_SEEKER_FRACTIONS:
            name = config_name(k_core, hub_frac)
            rtts: Dict[str, Optional[float]] = {}
            for _key, label in POLICIES:
                rtts[label] = load_rtt(res / f"{name}_{label}.json")

            cells = []
            best_label = "—"
            best_rtt = None
            for _key, label in POLICIES:
                val = rtts[label]
                if val is None:
                    cells.append(f"{'MISSING':>12}")
                else:
                    cells.append(f"{val:,.0f}".rjust(12))
                    if best_rtt is None or val < best_rtt:
                        best_rtt = val
                        best_label = label

            if best_rtt is not None:
                complete_rows += 1
                win_counts[best_label] = win_counts.get(best_label, 0) + 1

            row = f"{name:<18}" + "".join(cells) + f" {best_label:>12}"
            print(row)

    print("-" * len(header))
    print(f"Complete rows: {complete_rows}/{len(K_CORE_VALUES) * len(HUB_SEEKER_FRACTIONS)}")
    print("Best-policy wins:", ", ".join(f"{k}={v}" for k, v in win_counts.items() if v))


if __name__ == "__main__":
    main()
