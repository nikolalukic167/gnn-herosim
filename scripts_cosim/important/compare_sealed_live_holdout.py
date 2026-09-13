#!/usr/bin/env python3
"""Compare sealed multi-seed live holdout: GNN vs MLP vs Knative.

Expects result files named: {config}_s{seed}_{knative|gnn|mlp_dim22}.json
Writes a JSON report with per-config paired stats across seeds.

Reports total_rtt (primary) and the p90/p99 tail of per-task elapsed time. The
tail is the metric a collision-robustness advantage would appear in, so it is
reported alongside the primary rather than left in the result JSON unread.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts_cosim.sweep_metrics import load_metrics  # noqa: E402


TAG = {"knative": "knative", "gnn": "gnn", "mlp": "mlp_dim22"}
FILE_RE = re.compile(
    r"^(?P<cfg>.+)_s(?P<seed>\d+)_(?P<tag>knative|gnn|mlp_dim22)\.json$"
)


def mean_std(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    if len(xs) == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(var)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", type=Path, required=True)
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args()

    results = args.sweep_dir / "results"
    if not results.is_dir():
        raise FileNotFoundError(f"No results dir: {results}")

    # cfg -> seed -> policy -> metrics
    table: Dict[str, Dict[int, Dict[str, Dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    physics_seen: Dict[str, int] = defaultdict(int)
    for path in sorted(results.glob("*.json")):
        if path.name.endswith(".decode_stats.json"):
            continue
        m = FILE_RE.match(path.name)
        if not m:
            continue
        cfg = m.group("cfg")
        seed = int(m.group("seed"))
        tag = m.group("tag")
        policy = {v: k for k, v in TAG.items()}[tag]
        metrics = load_metrics(path)
        table[cfg][seed][policy] = metrics
        physics_seen[str(metrics["warmth_physics"])] += 1

    if len(physics_seen) > 1:
        raise ValueError(
            f"sweep mixes warmth_physics regimes {dict(physics_seen)} — not comparable"
        )
    if physics_seen and set(physics_seen) == {"None"}:
        raise ValueError(
            "no result file declares warmth_physics — sweep predates physics stamping "
            "(HEROSIM_REQUIRE_EXPLICIT_PHYSICS=1), so comparability was never verified"
        )

    if not table:
        print("ERROR: no sealed holdout result files matched", file=sys.stderr)
        return 1

    configs = sorted(table)
    report = {
        "sweep_dir": str(args.sweep_dir),
        "configs": {},
        "overall": {},
    }

    print(
        f"\n{'config':<20} {'seeds':>5} "
        f"{'kn_mean':>12} {'gnn_mean':>12} {'mlp_mean':>12} "
        f"{'gnn/kn':>8} {'mlp/kn':>8}  win_counts"
    )
    print("-" * 110)

    all_kn_sums: List[float] = []
    all_gnn_sums: List[float] = []
    all_mlp_sums: List[float] = []
    total_wins = {"gnn": 0, "mlp": 0, "knative": 0}
    paired_cells = 0

    tail_rows: List[Dict[str, Any]] = []
    tail_wins = {m: {"gnn": 0, "mlp": 0, "knative": 0} for m in ("p90", "p99")}

    for cfg in configs:
        seeds = sorted(table[cfg])
        kn_l, gnn_l, mlp_l = [], [], []
        tail_l: Dict[str, Dict[str, List[float]]] = {
            m: {"knative": [], "gnn": [], "mlp": []} for m in ("p90", "p99")
        }
        wins = {"gnn": 0, "mlp": 0, "knative": 0}
        per_seed = []
        incomplete = []
        for seed in seeds:
            cell = table[cfg][seed]
            if not all(p in cell for p in ("knative", "gnn", "mlp")):
                incomplete.append(seed)
                continue
            kn, gnn, mlp = (cell[p]["total_rtt"] for p in ("knative", "gnn", "mlp"))
            for metric in ("p90", "p99"):
                for policy in ("knative", "gnn", "mlp"):
                    tail_l[metric][policy].append(cell[policy][metric])
            kn_l.append(kn)
            gnn_l.append(gnn)
            mlp_l.append(mlp)
            if gnn <= mlp and gnn <= kn:
                w = "gnn"
            elif mlp <= gnn and mlp <= kn:
                w = "mlp"
            else:
                w = "knative"
            wins[w] += 1
            total_wins[w] += 1
            paired_cells += 1
            per_seed.append(
                {
                    "seed": seed,
                    "knative": kn,
                    "gnn": gnn,
                    "mlp": mlp,
                    "gnn_over_kn": gnn / kn,
                    "mlp_over_kn": mlp / kn,
                    "winner": w,
                }
            )

        if incomplete:
            print(f"WARN {cfg}: incomplete seeds {incomplete}", file=sys.stderr)
        if not kn_l:
            print(f"SKIP {cfg}: no complete seed triples")
            continue

        kn_m, kn_s = mean_std(kn_l)
        gnn_m, gnn_s = mean_std(gnn_l)
        mlp_m, mlp_s = mean_std(mlp_l)
        all_kn_sums.append(sum(kn_l))
        all_gnn_sums.append(sum(gnn_l))
        all_mlp_sums.append(sum(mlp_l))

        win_str = f"GNN {wins['gnn']}/{len(kn_l)} MLP {wins['mlp']}/{len(kn_l)} Kn {wins['knative']}/{len(kn_l)}"
        print(
            f"{cfg:<20} {len(kn_l):>5} "
            f"{kn_m:>12,.0f} {gnn_m:>12,.0f} {mlp_m:>12,.0f} "
            f"{gnn_m/kn_m:>7.2f}x {mlp_m/kn_m:>7.2f}x  {win_str}"
        )
        print(
            f"{'':20} {'±':>5} "
            f"{kn_s:>12,.0f} {gnn_s:>12,.0f} {mlp_s:>12,.0f}"
        )

        tail_report: Dict[str, Any] = {}
        for metric in ("p90", "p99"):
            means = {
                policy: mean_std(tail_l[metric][policy])[0]
                for policy in ("knative", "gnn", "mlp")
            }
            winner = min(means, key=means.get)
            tail_wins[metric][winner] += 1
            tail_report[metric] = {"means": means, "winner": winner}
            tail_rows.append({"config": cfg, "metric": metric, **means, "winner": winner})

        report["configs"][cfg] = {
            "n_seeds": len(kn_l),
            "incomplete_seeds": incomplete,
            "tail": tail_report,
            "knative_mean": kn_m,
            "knative_std": kn_s,
            "gnn_mean": gnn_m,
            "gnn_std": gnn_s,
            "mlp_mean": mlp_m,
            "mlp_std": mlp_s,
            "gnn_over_kn_mean": gnn_m / kn_m,
            "mlp_over_kn_mean": mlp_m / kn_m,
            "wins": wins,
            "per_seed": per_seed,
        }

    if not paired_cells:
        print("ERROR: no complete config×seed triples", file=sys.stderr)
        return 1

    sum_kn = sum(all_kn_sums)
    sum_gnn = sum(all_gnn_sums)
    sum_mlp = sum(all_mlp_sums)
    print("-" * 110)
    print(
        f"{'SUM(all cells)':<20} {paired_cells:>5} "
        f"{sum_kn:>12,.0f} {sum_gnn:>12,.0f} {sum_mlp:>12,.0f} "
        f"{sum_gnn/sum_kn:>7.2f}x {sum_mlp/sum_kn:>7.2f}x"
    )
    print(
        f"\nPaired cell wins: GNN {total_wins['gnn']}/{paired_cells} · "
        f"MLP {total_wins['mlp']}/{paired_cells} · Knative {total_wins['knative']}/{paired_cells}"
    )

    print(
        f"\n--- tail of per-task elapsed time (seconds, seed-averaged) ---\n"
        f"{'config':<20} {'metric':>6} {'knative':>10} {'gnn':>10} {'mlp':>10}  winner"
    )
    for row in tail_rows:
        print(
            f"{row['config']:<20} {row['metric']:>6} "
            f"{row['knative']:>10.1f} {row['gnn']:>10.1f} {row['mlp']:>10.1f}  {row['winner']}"
        )
    for metric in ("p90", "p99"):
        w = tail_wins[metric]
        print(
            f"{metric} config wins: GNN {w['gnn']}/{len(configs)} · "
            f"MLP {w['mlp']}/{len(configs)} · Knative {w['knative']}/{len(configs)}"
        )

    # Deployment conclusion gate (descriptive, not auto-pass): GNN beats MLP on sum and wins
    gnn_vs_mlp_sum = sum_gnn < sum_mlp
    gnn_vs_mlp_wins = total_wins["gnn"] >= total_wins["mlp"]
    gnn_vs_kn_sum = sum_gnn < sum_kn
    verdict = {
        "paired_cells": paired_cells,
        "warmth_physics": next(iter(physics_seen)),
        "tail_wins": tail_wins,
        "tail_rows": tail_rows,
        "sum_knative": sum_kn,
        "sum_gnn": sum_gnn,
        "sum_mlp": sum_mlp,
        "gnn_over_kn_sum": sum_gnn / sum_kn,
        "mlp_over_kn_sum": sum_mlp / sum_kn,
        "wins": total_wins,
        "gnn_beats_mlp_on_sum": gnn_vs_mlp_sum,
        "gnn_beats_mlp_on_wins": gnn_vs_mlp_wins,
        "gnn_beats_kn_on_sum": gnn_vs_kn_sum,
        "claim_support": (
            "GNN preferred vs MLP on sealed holdout"
            if gnn_vs_mlp_sum and gnn_vs_mlp_wins
            else "GNN does not dominate MLP on sealed holdout — do not claim uniform live transfer"
        ),
    }
    report["overall"] = verdict
    print(f"Claim support: {verdict['claim_support']}")

    out = args.report or (args.sweep_dir / "compare.json")
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
