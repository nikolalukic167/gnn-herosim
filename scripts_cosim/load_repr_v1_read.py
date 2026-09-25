#!/usr/bin/env python3
"""load_repr_v1 -- the live read, L1/L2/L3 as registered (docs/lineages/load_repr_v1.md).

  load_repr_v1_read.py --gate <fresh gate dir> --probe-dir <v4 dir> --selection selected.json [--out read.json]

Statistic and drop rule are cd_gap_v1_read.contrast's (fresh_topo_burst_v1's): per environment the
median over seeds of the paired % (two learned arms: the same seed paired), per topology the median over
windows, exact two-sided Wilcoxon over topologies. Negative = the first-named arm is faster.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cd_gap_v1_read as C  # noqa: E402


def l2_label(res: dict) -> str:
    r = res.get("read")
    if not r:
        return res.get("verdict", "NO-READ")
    v = r["verdict"]
    if v == "V4LOAD-FASTER":
        return "LOAD-HELPS"
    if v.startswith("V4LOAD-FASTER"):
        return "LOAD-HELPS (direction only)"
    if v.startswith("V4TWIN-FASTER"):
        return "LOAD-HURTS"
    return "NOT-SEPARATED"


def l1_label(res: dict, l2: str) -> str:
    r = res.get("read")
    if not r:
        return res.get("verdict", "NO-READ")
    if r["median_pct"] <= 0.0 or r["p"] >= 0.05:
        return "CLOSES-GAP"
    return "NARROWS" if l2.startswith("LOAD-HELPS") else "NO-EFFECT"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", required=True)
    ap.add_argument("--probe-dir", required=True, nargs="+")
    ap.add_argument("--selection", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--old-gate", nargs="*", default=[],
                    help="Amendment 2: dirs holding the learned arms as served BEFORE the rank fix; read as <kind>_asserved")
    a = ap.parse_args(argv)
    topos = json.load(open(a.selection))["topologies"]
    s = C._load([a.gate] + a.probe_dir)
    if a.old_gate:
        old = C._load(a.old_gate)
        for (t, w, k, sd), row in old.items():
            if k in ("gnnedge0", "mpoff", "v4load", "v4twin"):
                s[(t, w, f"{k}_asserved", sd)] = row
    res = {"L2": C.contrast(s, topos, "v4load", "v4twin"),
           "L1": C.contrast(s, topos, "v4load", "cd"),
           "L3_vs_gnnedge0": C.contrast(s, topos, "v4load", "gnnedge0"),
           "L3_vs_selfpredict": C.contrast(s, topos, "v4load", "selfpredict"),
           "disclosed_v4twin_vs_cd": C.contrast(s, topos, "v4twin", "cd"),
           "disclosed_v4twin_vs_gnnedge0": C.contrast(s, topos, "v4twin", "gnnedge0")}
    res["L2_label"] = l2_label(res["L2"])
    res["L1_label"] = l1_label(res["L1"], res["L2_label"])
    keys = ["L2", "L1", "L3_vs_gnnedge0", "L3_vs_selfpredict", "disclosed_v4twin_vs_cd",
            "disclosed_v4twin_vs_gnnedge0"]
    if a.old_gate:
        res["R1_gnnedge0_fixed_vs_asserved"] = C.contrast(s, topos, "gnnedge0", "gnnedge0_asserved")
        res["R2_gnnedge0_fixed_vs_cd"] = C.contrast(s, topos, "gnnedge0", "cd")
        res["R3_gnnedge0_vs_mpoff_fixed"] = C.contrast(s, topos, "gnnedge0", "mpoff")
        res["R4_mpoff_fixed_vs_asserved"] = C.contrast(s, topos, "mpoff", "mpoff_asserved")
        res["R5_gnnedge0_fixed_vs_selfpredict"] = C.contrast(s, topos, "gnnedge0", "selfpredict")
        keys += [k for k in res if k.startswith("R")]
    print(json.dumps(res, indent=1))
    for k in keys:
        r = res[k].get("read") or {}
        print(f"{k:32s} median {r.get('median_pct', float('nan')):+7.2f} %  p={r.get('p')}  "
              f"first faster {r.get('a_faster')}/{r.get('n')}  dropped {sorted(res[k].get('dropped', {}))}",
              file=sys.stderr)
    print(f"L2: {res['L2_label']}   L1: {res['L1_label']}", file=sys.stderr)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
