#!/usr/bin/env python3
"""Reader of the outbound_link_features_v1 label probe (rows of label_probe.py). Delta = cellular - wired, one task A moved, everything else held;
d_rtt = the change of the batch's summed RTT (sum of doneTime - scheduledTime), d_out = the change of the partners' summed exchange time.

  label_probe_read.py ROWS_DIR [--out read.json]

Rows used: predicted d_out >= 0.5 s. PASS needs at least 20 such rows, d_out > 0 on at least 90 % of them, the median d_out / predicted d_out in [0.5, 2.0] and the
median d_rtt > 0. Anything else, including fewer rows, is FAIL (unknown is not a pass)."""
import glob
import json
import statistics as st
import sys


def main():
    d = sys.argv[1]
    rows = [json.loads(l) for f in sorted(glob.glob(f"{d}/rows_*.jsonl")) for l in open(f) if l.strip()]
    use = [r for r in rows if r["pred_out"] >= 0.5]
    res = {"rows": len(rows), "rows_used": len(use), "topologies": len({r["stem"].split("__")[0] for r in rows})}
    if use:
        ratio = [r["d_out"] / r["pred_out"] for r in use]
        res.update(share_d_out_positive=sum(r["d_out"] > 0 for r in use) / len(use), median_ratio=st.median(ratio), median_d_rtt=st.median(r["d_rtt"] for r in use),
                   median_d_out=st.median(r["d_out"] for r in use), median_pred_out=st.median(r["pred_out"] for r in use))
        res["PASS"] = bool(len(use) >= 20 and res["share_d_out_positive"] >= 0.9 and 0.5 <= res["median_ratio"] <= 2.0 and res["median_d_rtt"] > 0)
    else:
        res["PASS"] = False
    print(json.dumps(res, indent=1))
    if "--out" in sys.argv:
        json.dump(res, open(sys.argv[sys.argv.index("--out") + 1], "w"), indent=1)
    return 0 if res["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
