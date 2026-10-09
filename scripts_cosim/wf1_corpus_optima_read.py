"""r1_attribution_v1: what the corpus' optima look like. Per dataset: the sweep minimum (ties by plan, lexicographically), whether the
batch sits on ONE node, whether its peer-exchange time is zero, plan counts. The pause line is the share of optima that are
single-node with zero exchange."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    rows, no_meta = [], 0
    for base in a.datasets:
        for d in sorted(base.glob("ds_*")):
            meta_p = d / "placement_metadata.json"
            if not meta_p.is_file():
                no_meta += 1
                continue
            meta = json.loads(meta_p.read_text())
            if meta.get("sweep_complete") is not True:
                continue
            best = None
            for line in open(d / "placements" / "placements.jsonl"):
                if not line.strip():
                    continue
                r = json.loads(line)
                combo = tuple((int(v[0]), int(v[1])) for _, v in sorted(r["placement_plan"].items(), key=lambda kv: int(kv[0])))
                key = (float(r["rtt"]), combo)
                if best is None or key < best[0]:
                    best = (key, r)
            (rtt, combo), r = best
            nodes = {n for n, _ in combo}
            fid = json.loads((d / "fidelity_replay.json").read_text()) if (d / "fidelity_replay.json").is_file() else {}
            rows.append({"dataset": d.name, "source": base.name, "tasks": len(combo), "plans": int(meta["num_placements"]),
                         "distinct_nodes": len(nodes), "single_node": len(nodes) == 1,
                         "exchange_s": float(r.get("peer_exchange_total", float("nan"))), "rtt": rtt,
                         "queued": fid.get("queued_tasks")})
    n = len(rows)
    sn = [r for r in rows if r["single_node"]]
    sn0 = [r for r in sn if r["exchange_s"] == 0.0]
    plans = sorted(r["plans"] for r in rows)
    out = {"datasets": n, "no_metadata": no_meta,
           "single_node": len(sn), "single_node_share": len(sn) / n if n else None,
           "single_node_zero_exchange": len(sn0), "single_node_zero_exchange_share": len(sn0) / n if n else None,
           "zero_exchange_any": sum(1 for r in rows if r["exchange_s"] == 0.0),
           "exchange_nan": sum(1 for r in rows if r["exchange_s"] != r["exchange_s"]),
           "plans": {"min": plans[0], "median": statistics.median(plans), "p90": plans[int(0.9 * (n - 1))], "max": plans[-1]} if n else None,
           "distinct_nodes_hist": {str(k): sum(1 for r in rows if r["distinct_nodes"] == k) for k in sorted({r["distinct_nodes"] for r in rows})}}
    print(json.dumps(out, indent=1))
    if a.out:
        a.out.write_text(json.dumps({"summary": out, "rows": rows}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
