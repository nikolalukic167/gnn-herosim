#!/usr/bin/env python3
"""r1_attribution_v1: how many peer groups does a corpus batch hold, against how many a served batch holds?

Serving (GNN_BATCH_BY_PEER_GROUP=1) decides one peer group per batch (the transitive closure of the peer table). The capture arm batches by
window, so a captured batch -- and a sub-batch cut from it by declared pruning -- can hold several groups, whose joint optimum prices contention
between groups that serving never decides together. This counts them, per dataset and per parent (same snapshot time + trigger task).

  wf1_batch_components.py --datasets <gnn_datasets dir> [--datasets <more>] [--out report.json]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def components(ids, pairs) -> int:
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, *_ in pairs:
        if a in parent and b in parent:
            parent[find(a)] = find(b)
    return len({find(i) for i in ids})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", type=Path, action="append", required=True)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    rows, parents = [], {}
    for base in a.datasets:
        for d in sorted(base.glob("ds_*")):
            infra = json.loads((d / "infrastructure.json").read_text())
            snap = ((infra.get("live_snapshot_seed") or {}).get("fidelity_replay") or {}).get("snapshot")
            if not snap or "fidelity" not in snap:
                continue
            f = snap["fidelity"]
            ids = [int(r["gid"]) for r in f["batch"]]
            comps = components(ids, f["pairs"])
            rows.append({"dataset": f"{base.name}/{d.name}", "tasks": len(ids), "components": comps})
            p = parents.setdefault((round(float(snap["time"]), 6), snap["trigger_task_id"]), {"ids": set(), "pairs": []})
            p["ids"].update(ids)
            p["pairs"].extend(f["pairs"])
    multi = [r for r in rows if r["components"] > 1]
    pmulti = sum(1 for p in parents.values() if components(sorted(p["ids"]), p["pairs"]) > 1)
    out = {"datasets": len(rows), "multi_group_datasets": len(multi), "share": len(multi) / len(rows) if rows else None,
           "parents": len(parents), "multi_group_parents": pmulti, "components_histogram": dict(Counter(r["components"] for r in rows)),
           "multi_group": multi}
    print(json.dumps({k: v for k, v in out.items() if k != "multi_group"}))
    if a.out:
        a.out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
