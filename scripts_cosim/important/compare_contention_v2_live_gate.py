#!/usr/bin/env python3
"""
Compare contention_v2 live gate sweep: GNN vs MLP dim22 vs Knative.

Reports total_rtt (gate metric) and the p90/p99 tail of per-task elapsed time,
where a pointwise model's collision cliff shows up.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts_cosim.sweep_metrics import load_metrics  # noqa: E402


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
        gnn = results / f"{cfg}_gnn.json"
        mlp = results / f"{cfg}_mlp_dim22.json"
        if not all(p.is_file() for p in (kn, gnn, mlp)):
            missing = [str(p) for p in (kn, gnn, mlp) if not p.is_file()]
            print(f"SKIP {cfg}: missing {missing}")
            continue
        metrics = {name: load_metrics(p) for name, p in (("kn", kn), ("gnn", gnn), ("mlp", mlp))}
        rows.append(
            (
                cfg,
                metrics["kn"]["total_rtt"],
                metrics["gnn"]["total_rtt"],
                metrics["mlp"]["total_rtt"],
                metrics,
            )
        )

    if not rows:
        print("No complete config triples found.")
        return 1

    print(f"\n{'config':<18} {'knative':>14} {'gnn':>14} {'mlp':>14}  {'gnn/kn':>8} {'mlp/kn':>8}  winner")
    print("-" * 90)
    gnn_wins = mlp_wins = kn_wins = 0
    sum_kn = sum_gnn = sum_mlp = 0.0
    for cfg, r_kn, r_gnn, r_mlp, _ in rows:
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

    print(
        f"\n--- tail of per-task elapsed time (s) ---\n"
        f"{'config':<18} {'metric':>6} {'knative':>10} {'gnn':>10} {'mlp':>10}  winner"
    )
    for cfg, _, _, _, metrics in rows:
        for metric in ("p90", "p99"):
            vals = {name: metrics[name][metric] for name in ("kn", "gnn", "mlp")}
            best = min(vals, key=vals.get)
            print(
                f"{cfg:<18} {metric:>6} "
                f"{vals['kn']:>10.1f} {vals['gnn']:>10.1f} {vals['mlp']:>10.1f}  {best}"
            )
    gate = "PASS" if gnn_wins >= mlp_wins and sum_gnn < sum_mlp else "FAIL"
    print(f"Live gate (GNN >= MLP on wins and sum): {gate}")
    return 0 if gate == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
