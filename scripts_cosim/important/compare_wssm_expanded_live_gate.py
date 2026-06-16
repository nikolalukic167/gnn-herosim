#!/usr/bin/env python3
"""Compare wssm expanded live gate: GNN vs MLP dim22 vs Knative."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_rtt(path: Path) -> float:
    return float(json.loads(path.read_text())["total_rtt"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", type=Path, required=True)
    args = ap.parse_args()
    results = args.sweep_dir / "results"
    if not results.is_dir():
        raise FileNotFoundError(f"No results dir: {results}")

    configs = sorted({p.name.replace("_knative.json", "") for p in results.glob("*_knative.json")})
    rows = []
    for cfg in configs:
        kn = results / f"{cfg}_knative.json"
        gnn = results / f"{cfg}_gnn_wssm.json"
        mlp = results / f"{cfg}_mlp_dim22.json"
        if not all(p.is_file() for p in (kn, gnn, mlp)):
            missing = [str(p.name) for p in (kn, gnn, mlp) if not p.is_file()]
            print(f"SKIP {cfg}: missing {missing}")
            continue
        r_kn, r_gnn, r_mlp = load_rtt(kn), load_rtt(gnn), load_rtt(mlp)
        rows.append((cfg, r_kn, r_gnn, r_mlp))

    if not rows:
        print("No complete config triples found.")
        return 1

    print(f"\n{'config':<18} {'knative':>14} {'gnn':>14} {'mlp':>14}  {'gnn/kn':>8} {'mlp/kn':>8}  winner")
    print("-" * 90)
    gnn_wins = mlp_wins = kn_wins = 0
    sum_kn = sum_gnn = sum_mlp = 0.0
    for cfg, r_kn, r_gnn, r_mlp in rows:
        g_ratio = r_gnn / r_kn if r_kn > 0 else float("inf")
        m_ratio = r_mlp / r_kn if r_kn > 0 else float("inf")
        if r_gnn <= r_mlp and r_gnn <= r_kn:
            winner = "GNN"
            gnn_wins += 1
        elif r_mlp <= r_gnn and r_mlp <= r_kn:
            winner = "MLP"
            mlp_wins += 1
        else:
            winner = "Kn"
            kn_wins += 1
        sum_kn += r_kn
        sum_gnn += r_gnn
        sum_mlp += r_mlp
        print(
            f"{cfg:<18} {r_kn:>14,.0f} {r_gnn:>14,.0f} {r_mlp:>14,.0f}  "
            f"{g_ratio:>7.2f}x {m_ratio:>7.2f}x  {winner}"
        )

    print("-" * 90)
    print(
        f"{'SUM':<18} {sum_kn:>14,.0f} {sum_gnn:>14,.0f} {sum_mlp:>14,.0f}  "
        f"{sum_gnn/sum_kn:>7.2f}x {sum_mlp/sum_kn:>7.2f}x"
    )
    print(f"\nWins: GNN {gnn_wins}/{len(rows)} · MLP {mlp_wins}/{len(rows)} · Knative {kn_wins}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
