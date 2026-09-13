#!/usr/bin/env python3
"""Compare wssm GNN vs Knative on bipartite-3 × queue_length sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_metrics(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    stats = data.get("stats") or {}
    scale_events = stats.get("scaleEvents") or []
    return {
        "total_rtt": data.get("total_rtt"),
        "queue_length": data.get("queue_length"),
        "avg_queue_s": stats.get("averageQueueTime"),
        "cold_start_pct": stats.get("coldStartProportion"),
        "scale_ups": sum(1 for e in scale_events if e.get("action") == "up"),
        "scale_downs": sum(1 for e in scale_events if e.get("action") == "down"),
    }


def fmt_m(x: Optional[float]) -> str:
    return f"{x / 1e6:.3f}M" if x is not None else "—"


def fmt_s(x: Optional[float]) -> str:
    return f"{x:.2f}s" if x is not None else "—"


def parse_name(stem: str) -> Optional[Tuple[str, str, int]]:
    # hub_k6_seek50__gnn_wssm__ql100
    parts = stem.split("__")
    if len(parts) != 3 or not parts[2].startswith("ql"):
        return None
    try:
        ql = int(parts[2][2:])
    except ValueError:
        return None
    return parts[0], parts[1], ql


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", type=Path, required=True)
    args = parser.parse_args()

    res_dir = args.sweep_dir / "results"
    rows: List[Tuple[str, str, int, Dict[str, Any]]] = []
    for path in sorted(res_dir.glob("*.json")):
        if "decode_stats" in path.name:
            continue
        parsed = parse_name(path.stem)
        if not parsed:
            continue
        cfg, pol, ql = parsed
        rows.append((cfg, pol, ql, load_metrics(path)))

    if not rows:
        print(f"No results in {res_dir}")
        return

    configs = sorted({r[0] for r in rows})
    qls = sorted({r[2] for r in rows})

    print(f"\nwssm GNN vs Knative · bipartite-3 × ql — {args.sweep_dir}\n")
    print("Physics: node_disk_v2 · GNN: warmth-sparse-skew-merged dim22 · workload 125-225\n")

    for cfg in configs:
        print(f"=== {cfg} ===")
        print(f"{'policy':<12} {'ql':>4} {'RTT':>10} {'avgQ':>8} {'cold%':>6} {'↑':>5} {'↓':>5} {'GvsKn':>8}")
        print("-" * 62)
        kn_by_ql = {
            q: m["total_rtt"]
            for c, p, q, m in rows
            if c == cfg and p == "knative" and m["total_rtt"]
        }
        for pol in ("gnn_wssm", "knative"):
            for ql in qls:
                match = [m for c, p, q, m in rows if c == cfg and p == pol and q == ql]
                if not match:
                    continue
                m = match[0]
                gvs = ""
                if pol == "gnn_wssm" and ql in kn_by_ql and m["total_rtt"]:
                    gvs = f"{(m['total_rtt'] - kn_by_ql[ql]) / kn_by_ql[ql] * 100:+.1f}%"
                print(
                    f"{pol:<12} {ql:>4} {fmt_m(m['total_rtt']):>10} "
                    f"{fmt_s(m['avg_queue_s']):>8} "
                    f"{(m['cold_start_pct'] or 0):>5.1f}% "
                    f"{m['scale_ups']:>5} {m['scale_downs']:>5} {gvs:>8}"
                )
        print("  Winner per ql:", end="")
        for ql in qls:
            g = next((m["total_rtt"] for c, p, q, m in rows if c == cfg and p == "gnn_wssm" and q == ql), None)
            k = next((m["total_rtt"] for c, p, q, m in rows if c == cfg and p == "knative" and q == ql), None)
            if g is None or k is None:
                continue
            w = "GNN" if g < k else "Kn"
            print(f" ql{ql}={w}", end="")
        print("\n")

    print("=== Ranking stability (GNN vs Kn per config) ===")
    for cfg in configs:
        winners = []
        for ql in qls:
            g = next((m["total_rtt"] for c, p, q, m in rows if c == cfg and p == "gnn_wssm" and q == ql), None)
            k = next((m["total_rtt"] for c, p, q, m in rows if c == cfg and p == "knative" and q == ql), None)
            if g is None or k is None:
                continue
            winners.append("GNN" if g < k else "Kn")
        flip = len(set(winners)) > 1 if winners else False
        print(f"  {cfg}: {winners}  {'FLIP' if flip else 'stable'}")


if __name__ == "__main__":
    main()
