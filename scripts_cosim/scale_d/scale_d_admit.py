#!/usr/bin/env python3
"""scale_d_v1 admission read (Design "Admission" and Rulings item 7). Reads ONLY queue_share and completion of the bar cells.
A topology is admitted iff all four bar cells (g0, g1 x moderate, heavy) are complete and have a known queue_share <= 0.80. Unknown is a fail.
Completion: the summary exists with num_tasks == 50000 and a finite endTime, there is no .failed.json for the cell, and wallclock_s is below the wall limit (no wall kill).
usage: scale_d_admit.py <cells_bar root> <ids,csv> <servers> <clients> [out.json]
"""
import glob
import json
import math
import os
import sys

root, ids, servers, clients = sys.argv[1], sys.argv[2].split(","), int(sys.argv[3]), int(sys.argv[4])
out = sys.argv[5] if len(sys.argv) > 5 else None
ARM = "cd_ib0_inflightfull_k100_s0"
LIMIT = 43200
res = {}
for t in ids:
    reasons = []
    for w in ("g0", "g1"):
        for r in ("moderate", "heavy"):
            name = f"cc40s{t}__{w}{r}__{ARM}"
            hits = glob.glob(os.path.join(root, "*", name + ".summary.json"))
            if not hits:
                reasons.append(f"{w}{r}: no summary")
                continue
            if glob.glob(os.path.join(root, "*", name + ".failed.json")):
                reasons.append(f"{w}{r}: .failed.json")
                continue
            d = json.load(open(hits[0]))
            if (d.get("servers"), d.get("clients")) != (servers, clients):
                reasons.append(f"{w}{r}: size {d.get('servers')}x{d.get('clients')}")
            if d.get("num_tasks") != 50000 or not isinstance(d.get("endTime"), (int, float)) or not math.isfinite(d["endTime"]):
                reasons.append(f"{w}{r}: incomplete")
            if not isinstance(d.get("wallclock_s"), (int, float)) or d["wallclock_s"] >= LIMIT:
                reasons.append(f"{w}{r}: wall")
            q = d.get("queue_share")
            if not isinstance(q, (int, float)) or not math.isfinite(q):
                reasons.append(f"{w}{r}: queue_share unknown")
            elif q > 0.80:
                reasons.append(f"{w}{r}: queue_share {q:.3f} > 0.80")
    res[t] = reasons
admitted = [t for t in ids if not res[t]]
rep = {"root": root, "ids": ids, "admitted": admitted, "failed": {t: r for t, r in res.items() if r}}
s = json.dumps(rep, indent=1)
print(s)
if out:
    open(out, "w").write(s + "\n")
print(f"admitted {len(admitted)} of {len(ids)}")
