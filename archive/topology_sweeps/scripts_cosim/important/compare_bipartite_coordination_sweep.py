#!/usr/bin/env python3
"""Compare sweep_bipartite_coordination_v1: GNN vs MLP dim22 vs Knative (Regime B)."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional

BATCH_SIZE = 4
CONFIG_RE = re.compile(r"^hub_k(\d+)_seek(\d+)$")


def load_rtt(path: Path) -> Optional[float]:
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    rtt = data.get("total_rtt")
    if rtt is None or not isinstance(rtt, (int, float)) or rtt <= 0:
        return None
    return float(rtt)


def fmt_m(rtt: Optional[float]) -> str:
    if rtt is None:
        return "MISSING"
    return f"{rtt / 1e6:.2f}M"


def delta_pct(a: Optional[float], b: Optional[float]) -> str:
    if a is None or b is None or b <= 0:
        return "—"
    return f"{(a - b) / b * 100:+.1f}%"


def regime_label(k: int) -> str:
    if k < BATCH_SIZE:
        return "k<b (starved)"
    if k == BATCH_SIZE:
        return "k=b (marginal)"
    return "k>b (coordination)"


def best_policy(values: dict[str, Optional[float]]) -> str:
    present = {k: v for k, v in values.items() if v is not None}
    if not present:
        return "—"
    winner = min(present, key=present.get)
    tied = [k for k, v in present.items() if v <= present[winner] * 1.005]
    if len(tied) > 1:
        return "tie"
    return winner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        default=Path("simulation_data/normal_sim_sweeps/sweep_bipartite_coordination_v1"),
    )
    args = parser.parse_args()

    res = args.sweep_dir / "results"
    if not res.is_dir():
        raise FileNotFoundError(f"Missing results dir: {res}")

    meta_path = args.sweep_dir / "configs" / "sweep_meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        print(f"Sweep: {meta.get('sweep_id')} · b={meta.get('gnn_batch_size')} · "
              f"latency {meta.get('latency_core_ms')}/{meta.get('latency_periphery_ms')}ms")
    else:
        print(f"Sweep: {args.sweep_dir}")
    print("GNN/MLP = Regime A (batch b=4) · Knative = Regime B (per-arrival)\n")

    rows: list[tuple[int, int, str, Optional[float], Optional[float], Optional[float]]] = []
    for cfg_path in sorted((args.sweep_dir / "configs").glob("hub_k*.json")):
        m = CONFIG_RE.match(cfg_path.stem)
        if not m:
            continue
        k, seek = int(m.group(1)), int(m.group(2))
        name = cfg_path.stem
        gnn = load_rtt(res / f"{name}_gnn_dim22.json")
        mlp = load_rtt(res / f"{name}_mlp_dim22.json")
        knative = load_rtt(res / f"{name}_knative.json")
        rows.append((k, seek, name, gnn, mlp, knative))

    header = (
        f"{'Config':<18} {'k':>2} {'seek':>4} {'Regime':<18} "
        f"{'GNN':>8} {'MLP':>8} {'Knative':>8} {'GvsM':>7} {'Best':>7}"
    )
    print(header)
    print("-" * len(header))

    gnn_wins = mlp_wins = knative_wins = ties = 0
    gnn_mlp_paired = gnn_mlp_missing = 0
    for k, seek, name, gnn, mlp, knative in rows:
        policies = {"GNN": gnn, "MLP": mlp, "Knative": knative}
        best = best_policy(policies)
        if best == "GNN":
            gnn_wins += 1
        elif best == "MLP":
            mlp_wins += 1
        elif best == "Knative":
            knative_wins += 1
        elif best == "tie":
            ties += 1

        if gnn is not None and mlp is not None:
            gnn_mlp_paired += 1
        else:
            gnn_mlp_missing += 1

        print(
            f"{name:<18} {k:>2} {seek:>3}% {regime_label(k):<18} "
            f"{fmt_m(gnn):>8} {fmt_m(mlp):>8} {fmt_m(knative):>8} "
            f"{delta_pct(gnn, mlp):>7} {best:>7}"
        )

    print("-" * len(header))
    print(
        f"GNN/MLP paired: {gnn_mlp_paired}/{len(rows)} · missing {gnn_mlp_missing} · "
        f"Knative: {sum(1 for *_, kn in rows if kn is not None)}/{len(rows)}"
    )
    print(
        f"3-way best (where present): GNN {gnn_wins} · MLP {mlp_wins} · "
        f"Knative {knative_wins} · tie {ties}"
    )
    print("GvsM negative = GNN advantage over MLP · b=4 fixed")


if __name__ == "__main__":
    main()
