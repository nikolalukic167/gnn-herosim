# Exchange-sensitivity admission read: for each variant's reactive cells (xs/<v>/cells_reactive) it reads ONLY the queue_share field of each summary (no other field).
# A topology is admitted iff all 8 of its reactive cells (4 windows x 2 rungs) have queue_share <= 0.80; unknown (missing, failed, no field) is not a pass.
# Prints per-cell queue_share facts and, per variant, the failing topologies; writes xs/<v>/react_queue_share.json. Rule: x3 is dropped (and disclosed, never re-scaled) if more than 4 topologies fail.
import json, os, re, sys
U = "/share/nikola.lukic/uncut_groups/xs"
IDS = [int(x) for x in open(U + "/ids.txt").read().split()]
VARIANTS = sys.argv[1:] or ["x1", "x0", "x0p1", "x0p3", "x3", "tier0", "azblob"]
NAME = re.compile(r"^cc40s(\d+)__(g\d)(moderate|heavy)__reactive_s0\.(summary|failed)\.json$")
out = {}
for v in VARIANTS:
    d = f"{U}/{v}/cells_reactive"
    rows = {}
    if os.path.isdir(d):
        for fn in sorted(os.listdir(d)):
            m = NAME.match(fn)
            if not m: continue
            qs = json.load(open(os.path.join(d, fn))).get("queue_share") if m.group(4) == "summary" else None
            rows[(int(m.group(1)), m.group(2), m.group(3))] = qs
    bad = {}
    for t in IDS:
        cells = [(t, w, r) for w in ("g0", "g1", "g2", "g3") for r in ("moderate", "heavy")]
        b = [(c, rows.get(c)) for c in cells if rows.get(c) is None or rows[c] > 0.80]
        if b: bad[t] = [f"{c[1]}{c[2]}={q}" for c, q in b]
    adm = [t for t in IDS if t not in bad]
    out[v] = {"cells": len(rows), "failing_topologies": sorted(bad), "admitted": adm, "detail": bad, "max_queue_share": max([q for q in rows.values() if q is not None], default=None)}
    drop = " DROP (more than 4 failing; disclose, never re-scale)" if v == "x3" and len(bad) > 4 else ""
    print(f"{v}: reactive cells {len(rows)} (expect 96); failing topologies {len(bad)} {sorted(bad)}; max queue_share {out[v]['max_queue_share']}{drop}")
    json.dump([{"topology": k[0], "window": k[1], "rung": k[2], "queue_share": q} for k, q in sorted(rows.items())], open(f"{U}/{v}/react_queue_share.json", "w"), indent=1)
    json.dump(out[v], open(f"{U}/{v}/admission.json", "w"), indent=1)
