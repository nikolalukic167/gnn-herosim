# live_link_features_v1 confirmation ids: admission read of the 96 reactive cells under D (cells_react). Reads ONLY the queue_share field of each summary.
# A topology is admitted iff all 8 of its reactive cells (4 windows x 2 rungs) have queue_share <= 0.80; unknown (missing, failed, no field) is not a pass.
import json, os, re
X = "/share/nikola.lukic/uncut_groups/llf_conf"
IDS = [int(x) for x in open(X + "/ids.txt").read().split()]
NAME = re.compile(r"^cc40s(\d+)__(g\d)(moderate|heavy)__reactive_s0\.(summary|failed)\.json$")
d = X + "/cells_react"; rows = {}
for fn in sorted(os.listdir(d)):
    m = NAME.match(fn)
    if not m: continue
    rows[(int(m.group(1)), m.group(2), m.group(3))] = json.load(open(os.path.join(d, fn))).get("queue_share") if m.group(4) == "summary" else None
bad = {}
for t in IDS:
    b = [((t, w, r), rows.get((t, w, r))) for w in ("g0", "g1", "g2", "g3") for r in ("moderate", "heavy") if rows.get((t, w, r)) is None or rows[(t, w, r)] > 0.80]
    if b: bad[t] = [f"{c[1]}{c[2]}={q}" for c, q in b]
out = {"cells": len(rows), "failing_topologies": sorted(bad), "admitted": [t for t in IDS if t not in bad], "detail": bad, "max_queue_share": max([q for q in rows.values() if q is not None], default=None)}
json.dump(out, open(X + "/admission.json", "w"), indent=1)
json.dump([{"topology": k[0], "window": k[1], "rung": k[2], "queue_share": q} for k, q in sorted(rows.items())], open(X + "/react_queue_share.json", "w"), indent=1)
print(f"reactive cells {len(rows)} (expect 96); failing topologies {len(bad)} {sorted(bad)}; admitted {len(out['admitted'])} of {len(IDS)}; max queue_share {out['max_queue_share']}")
