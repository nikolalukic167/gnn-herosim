#!/usr/bin/env python3
"""Summarize autoscale sensitivity sweep: target concurrency vs RTT / queue / scale events."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_metrics(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    stats = data.get("stats") or {}
    scale_events = stats.get("scaleEvents") or []
    scale_ups = sum(1 for e in scale_events if e.get("action") == "up")
    scale_downs = sum(1 for e in scale_events if e.get("action") == "down")
    system_events = stats.get("systemEvents") or []
    max_replicas = 0
    for ev in scale_events:
        max_replicas = max(max_replicas, int(ev.get("count") or 0))
    for ev in system_events:
        max_replicas = max(max_replicas, int(ev.get("count") or 0))

    return {
        "path": str(path),
        "policy": data.get("policy", path.stem),
        "queue_length": data.get("queue_length"),
        "total_rtt": data.get("total_rtt"),
        "num_tasks": data.get("num_tasks"),
        "avg_queue_s": stats.get("averageQueueTime"),
        "cold_start_pct": stats.get("coldStartProportion"),
        "scale_ups": scale_ups,
        "scale_downs": scale_downs,
        "max_replicas_seen": max_replicas,
    }


def fmt_m(x: Optional[float]) -> str:
    if x is None:
        return "—"
    return f"{x / 1e6:.3f}M"


def fmt_s(x: Optional[float]) -> str:
    if x is None:
        return "—"
    return f"{x:.2f}s"


def fmt_pct(x: Optional[float]) -> str:
    if x is None:
        return "—"
    return f"{x:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        required=True,
        help="Sweep root (contains results/)",
    )
    args = parser.parse_args()

    res_dir = args.sweep_dir / "results"
    if not res_dir.is_dir():
        raise FileNotFoundError(f"Missing results dir: {res_dir}")

    rows: List[Tuple[str, str, int, Dict[str, Any]]] = []
    for path in sorted(res_dir.glob("*.json")):
        if path.name.endswith(".decode_stats.json"):
            continue
        name = path.stem
        parts = name.split("__")
        if len(parts) < 3:
            continue
        cfg, policy, ql = parts[0], parts[1], parts[2]
        try:
            ql_int = int(ql.replace("ql", ""))
        except ValueError:
            continue
        rows.append((cfg, policy, ql_int, load_metrics(path)))

    if not rows:
        print(f"No result JSONs in {res_dir} (expected {{cfg}}__{{policy}}__ql{{N}}.json)")
        return

    configs = sorted({r[0] for r in rows})
    policies = sorted({r[1] for r in rows})
    qls = sorted({r[2] for r in rows})

    print(f"\nAutoscale sensitivity — {args.sweep_dir}\n")
    print(
        "Knob: queue_length (= Knative target_concurrency per platform). "
        "Lower → scale up sooner (more replicas); higher → fewer replicas.\n"
    )

    for cfg in configs:
        print(f"=== {cfg} ===")
        header = f"{'policy':<18} {'ql':>4} {'RTT':>10} {'avgQ':>8} {'cold%':>7} {'↑':>5} {'↓':>5} {'maxR':>5} {'GvsKn%':>8}"
        print(header)
        print("-" * len(header))

        cfg_rows = [(p, q, m) for c, p, q, m in rows if c == cfg]
        kn_by_ql = {q: m["total_rtt"] for p, q, m in cfg_rows if p == "knative_network" and m["total_rtt"]}

        for policy in policies:
            for ql in qls:
                match = [m for p, q, m in cfg_rows if p == policy and q == ql]
                if not match:
                    continue
                m = match[0]
                kn = kn_by_ql.get(ql)
                gvs = ""
                if policy == "gnn" and kn and m["total_rtt"]:
                    gvs = f"{(m['total_rtt'] - kn) / kn * 100:+.1f}%"
                print(
                    f"{policy:<18} {ql:>4} {fmt_m(m['total_rtt']):>10} "
                    f"{fmt_s(m['avg_queue_s']):>8} {fmt_pct(m['cold_start_pct']):>7} "
                    f"{m['scale_ups']:>5} {m['scale_downs']:>5} {m['max_replicas_seen']:>5} {gvs:>8}"
                )

        # Ranking flip check per ql
        print("\n  Winner per target_concurrency (lower RTT):")
        for ql in qls:
            candidates = [
                (p, m["total_rtt"])
                for p, q, m in cfg_rows
                if q == ql and m["total_rtt"] is not None
            ]
            if not candidates:
                continue
            best = min(candidates, key=lambda x: x[1])
            print(f"    ql={ql}: {best[0]} ({fmt_m(best[1])})")
        print()

    # Cross-config sensitivity: does GNN−Knative gap sign change?
    print("=== Ranking stability (GNN vs Knative) ===")
    for cfg_name in configs:
        signs: List[str] = []
        for ql in qls:
            gnn_rtt = next(
                (m["total_rtt"] for cfg, p, q, m in rows if cfg == cfg_name and p == "gnn" and q == ql),
                None,
            )
            kn_rtt = next(
                (
                    m["total_rtt"]
                    for cfg, p, q, m in rows
                    if cfg == cfg_name and p == "knative_network" and q == ql
                ),
                None,
            )
            if gnn_rtt is None or kn_rtt is None:
                continue
            signs.append("GNN" if gnn_rtt < kn_rtt else "Kn")
        if signs:
            flip = len(set(signs)) > 1
            print(f"  {cfg_name}: winners @ ql{qls} → {signs}  {'FLIP' if flip else 'stable'}")


if __name__ == "__main__":
    main()
