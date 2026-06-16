#!/usr/bin/env python3
"""Compare merged weighted live gate: Knative vs MLP vs GNN argmax vs GNN argmax_uniq."""
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
        raise FileNotFoundError(results)

    configs = sorted({p.name.replace("_knative.json", "") for p in results.glob("*_knative.json")})
    print(f"\n{'config':<18} {'knative':>12} {'mlp':>12} {'gnn_uq':>12} {'gnn_arg':>12}  best")
    print("-" * 75)
    sums = {"kn": 0.0, "mlp": 0.0, "uq": 0.0, "arg": 0.0}
    wins = {"kn": 0, "mlp": 0, "uq": 0, "arg": 0}
    for cfg in configs:
        kn_p = results / f"{cfg}_knative.json"
        mlp_p = results / f"{cfg}_mlp_dim22.json"
        uq_p = results / f"{cfg}_gnn_uniq.json"
        arg_p = results / f"{cfg}_gnn_argmax.json"
        if not all(p.is_file() for p in (kn_p, mlp_p, uq_p, arg_p)):
            print(f"SKIP {cfg}: missing outputs")
            continue
        kn, mlp, uq, arg = load_rtt(kn_p), load_rtt(mlp_p), load_rtt(uq_p), load_rtt(arg_p)
        sums["kn"] += kn
        sums["mlp"] += mlp
        sums["uq"] += uq
        sums["arg"] += arg
        row = {"kn": kn, "mlp": mlp, "uq": uq, "arg": arg}
        best = min(row, key=row.get)
        wins[best] += 1
        print(f"{cfg:<18} {kn:>12,.0f} {mlp:>12,.0f} {uq:>12,.0f} {arg:>12,.0f}  {best}")
    print("-" * 75)
    print(f"{'SUM':<18} {sums['kn']:>12,.0f} {sums['mlp']:>12,.0f} {sums['uq']:>12,.0f} {sums['arg']:>12,.0f}")
    print(f"Wins: kn={wins['kn']} mlp={wins['mlp']} gnn_uniq={wins['uq']} gnn_argmax={wins['arg']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
