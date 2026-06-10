#!/usr/bin/env python3
"""Compare paired GNN vs MLP RTT results from atomic-21 skew sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_rtt(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    rtt = data.get("total_rtt")
    if rtt is None or not isinstance(rtt, (int, float)) or rtt <= 0:
        return None
    return float(rtt)


def pct_delta(gnn: float, mlp: float) -> float:
    if mlp <= 0:
        return float("nan")
    return 100.0 * (mlp - gnn) / mlp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        required=True,
        help="Sweep output root (contains results/)",
    )
    args = parser.parse_args()

    res_dir = args.sweep_dir / "results"
    if not res_dir.is_dir():
        raise FileNotFoundError(f"Missing results dir: {res_dir}")

    configs: List[str] = []
    for path in sorted(res_dir.glob("*.json")):
        name = path.stem
        if name.endswith("_mlp_atomic21"):
            continue
        if (res_dir / f"{name}_mlp_atomic21.json").exists() or name in {
            "default_20_20_p50",
            "05_sparse_40_40_p25",
            "default_20_20_degree_skew",
            "05_sparse_40_40_p25_degree_skew",
        }:
            configs.append(name)

    configs = sorted(set(configs))

    rows: List[Tuple[str, Optional[float], Optional[float], Optional[float]]] = []
    for name in configs:
        gnn_path = res_dir / f"{name}.json"
        mlp_path = res_dir / f"{name}_mlp_atomic21.json"
        gnn_rtt = load_rtt(gnn_path)
        mlp_rtt = load_rtt(mlp_path)
        delta = pct_delta(gnn_rtt, mlp_rtt) if gnn_rtt is not None and mlp_rtt is not None else None
        rows.append((name, gnn_rtt, mlp_rtt, delta))

    print(f"\nAtomic-21 GNN vs MLP — {args.sweep_dir}\n")
    print(f"{'Config':<40} {'GNN RTT':>14} {'MLP RTT':>14} {'MLP-GNN %':>12}")
    print("-" * 84)
    for name, gnn_rtt, mlp_rtt, delta in rows:
        gnn_s = f"{gnn_rtt:,.0f}" if gnn_rtt is not None else "MISSING"
        mlp_s = f"{mlp_rtt:,.0f}" if mlp_rtt is not None else "MISSING"
        delta_s = f"{delta:+.1f}%" if delta is not None else "—"
        marker = " <<" if "degree_skew" in name else ""
        print(f"{name:<40} {gnn_s:>14} {mlp_s:>14} {delta_s:>12}{marker}")

    print("\n(positive MLP-GNN % => GNN lower RTT / better)")
    skew_rows = [r for r in rows if "degree_skew" in r[0]]
    if skew_rows:
        print("\nPrimary skew configs:")
        for name, gnn_rtt, mlp_rtt, delta in skew_rows:
            if gnn_rtt is None or mlp_rtt is None:
                print(f"  {name}: incomplete")
            elif delta is not None and delta >= 5.0:
                print(f"  {name}: PASS (GNN {delta:+.1f}% better than MLP)")
            else:
                print(f"  {name}: gap {delta:+.1f}% (target >= 5%)")


if __name__ == "__main__":
    main()
