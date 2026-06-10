#!/usr/bin/env python3
"""Compare sweep_bipartite_coordination_v1: GNN vs MLP dim22 on tiered-hub grid."""

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


def delta_pct(gnn: Optional[float], mlp: Optional[float]) -> str:
    if gnn is None or mlp is None or mlp <= 0:
        return "—"
    return f"{(gnn - mlp) / mlp * 100:+.1f}%"


def regime_label(k: int) -> str:
    if k < BATCH_SIZE:
        return "k<b (starved)"
    if k == BATCH_SIZE:
        return "k=b (marginal)"
    return "k>b (coordination)"


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
              f"latency {meta.get('latency_core_ms')}/{meta.get('latency_periphery_ms')}ms\n")
    else:
        print(f"Sweep: {args.sweep_dir}\n")

    rows: list[tuple[int, int, str, Optional[float], Optional[float]]] = []
    for cfg_path in sorted((args.sweep_dir / "configs").glob("hub_k*.json")):
        m = CONFIG_RE.match(cfg_path.stem)
        if not m:
            continue
        k, seek = int(m.group(1)), int(m.group(2))
        name = cfg_path.stem
        gnn = load_rtt(res / f"{name}_gnn_dim22.json")
        mlp = load_rtt(res / f"{name}_mlp_dim22.json")
        rows.append((k, seek, name, gnn, mlp))

    header = f"{'Config':<18} {'k':>2} {'seek':>4} {'Regime':<18} {'GNN':>8} {'MLP':>8} {'Δ%':>8} {'Best':>6}"
    print(header)
    print("-" * len(header))

    gnn_wins = mlp_wins = ties = missing = 0
    for k, seek, name, gnn, mlp in rows:
        best = "—"
        if gnn is not None and mlp is not None:
            if gnn < mlp * 0.995:
                best = "GNN"
                gnn_wins += 1
            elif mlp < gnn * 0.995:
                best = "MLP"
                mlp_wins += 1
            else:
                best = "tie"
                ties += 1
        else:
            missing += 1

        print(
            f"{name:<18} {k:>2} {seek:>3}% {regime_label(k):<18} "
            f"{fmt_m(gnn):>8} {fmt_m(mlp):>8} {delta_pct(gnn, mlp):>8} {best:>6}"
        )

    print("-" * len(header))
    paired = gnn_wins + mlp_wins + ties
    print(f"Paired complete: {paired}/{len(rows)} · GNN {gnn_wins} · MLP {mlp_wins} · tie {ties} · missing {missing}")
    print(f"Δ% negative = GNN advantage · b={BATCH_SIZE} fixed")


if __name__ == "__main__":
    main()
