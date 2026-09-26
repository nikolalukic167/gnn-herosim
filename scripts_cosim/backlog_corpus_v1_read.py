#!/usr/bin/env python3
"""backlog_corpus_v1 -- the live read, L1/L2/L3 and C1 as registered (docs/lineages/backlog_corpus_v1.md,
Amendment 1).

  backlog_corpus_v1_read.py --dirs <bc1 gate> <ref_fresh_1ae90af> <load_repr_v1 fix> --selection selected.json
                            [--out read.json]

Statistic and drop rule are cd_gap_v1_read.contrast's: per environment the median over seeds of the paired %
(two learned arms: the same seed paired), per topology the median over windows, exact two-sided Wilcoxon over
topologies. Negative = the first-named arm is faster.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cd_gap_v1_read as C  # noqa: E402


def l1_label(res: dict) -> str:
    r = res.get("read")
    if not r:
        return res.get("verdict", "NO-READ")
    v = r["verdict"]
    if v == "BC1LOAD-FASTER":
        return "CORPUS-HELPS"
    if v.startswith("BC1LOAD-FASTER"):
        return "CORPUS-HELPS (direction only)"
    if v.startswith("V4LOAD_SE-FASTER"):
        return "CORPUS-HURTS"
    return "NOT-SEPARATED"


def l2_label(res: dict, l1: str) -> str:
    r = res.get("read")
    if not r:
        return res.get("verdict", "NO-READ")
    if r["median_pct"] <= 0.0 or r["p"] >= 0.05:
        return "CLOSES-GAP"
    return "NARROWS" if l1.startswith("CORPUS-HELPS") else "NO-EFFECT"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", required=True, nargs="+")
    ap.add_argument("--selection", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    topos = json.load(open(a.selection))["topologies"]
    s = C._load(a.dirs)
    res = {"L1": C.contrast(s, topos, "bc1load", "v4load_se"),
           "L2": C.contrast(s, topos, "bc1load", "cd"),
           "L3": C.contrast(s, topos, "bc1load", "bc1mpoff"),
           "C1": C.contrast(s, topos, "v4load_se", "v4load"),
           "reported_bc1load_vs_selfpredict": C.contrast(s, topos, "bc1load", "selfpredict"),
           "reported_bc1load_vs_v4load": C.contrast(s, topos, "bc1load", "v4load"),
           "reported_bc1mpoff_vs_cd": C.contrast(s, topos, "bc1mpoff", "cd"),
           "reported_v4load_se_vs_cd": C.contrast(s, topos, "v4load_se", "cd")}
    if any(k[2] == "bc1load_selfref" for k in s):
        # Amendment 3
        res["S1"] = C.contrast(s, topos, "bc1load_selfref", "cd")
        res["S2"] = C.contrast(s, topos, "bc1load_selfref", "bc1load")
        res["reported_bc1load_selfref_vs_selfpredict"] = C.contrast(s, topos, "bc1load_selfref", "selfpredict")
        res["S1_label"] = ("CLOSES-GAP" if res["S1"].get("read") and (res["S1"]["read"]["median_pct"] <= 0.0
                           or res["S1"]["read"]["p"] >= 0.05) else res["S1"].get("verdict") or "CD-FASTER")
    if any(k[2] == "fc1load_selfref" for k in s):
        # fullctx_refine_v1 (docs/lineages/fullctx_refine_v1.md)
        res["F1"] = C.contrast(s, topos, "fc1load_selfref", "cd")
        res["F2"] = C.contrast(s, topos, "fc1load_selfref", "bc1load_selfref")
        res["reported_fc1load_vs_bc1load"] = C.contrast(s, topos, "fc1load", "bc1load")
        res["reported_fc1load_selfref_vs_selfpredict"] = C.contrast(s, topos, "fc1load_selfref", "selfpredict")
        r1 = res["F1"].get("read")
        if not r1:
            res["F1_label"] = res["F1"].get("verdict", "NO-READ")
        elif r1["p"] < 0.05 and r1["median_pct"] < 0.0:
            res["F1_label"] = "BEATS-CD"
        elif r1["median_pct"] <= 0.0 or r1["p"] >= 0.05:
            res["F1_label"] = "CLOSES-GAP"
        else:
            res["F1_label"] = "CD-FASTER"
        r2 = res["F2"].get("read")
        v2 = r2["verdict"] if r2 else res["F2"].get("verdict", "NO-READ")
        res["F2_label"] = ("TRAINING-HELPS" if v2 == "FC1LOAD_SELFREF-FASTER" else
                           "TRAINING-HELPS (direction only)" if v2.startswith("FC1LOAD_SELFREF-FASTER") else
                           "TRAINING-HURTS" if v2.startswith("BC1LOAD_SELFREF-FASTER") else
                           "NOT-SEPARATED" if r2 else v2)
    if any(k[2] == "xs1load_selfref" for k in s):
        # exchange_seconds_v1 (docs/lineages/exchange_seconds_v1.md)
        res["X1"] = C.contrast(s, topos, "xs1load_selfref", "cd")
        res["X2"] = C.contrast(s, topos, "xs1load_selfref", "fc1load_selfref")
        res["reported_xs1load_vs_fc1load"] = C.contrast(s, topos, "xs1load", "fc1load")
        res["reported_xs1load_selfref_vs_selfpredict"] = C.contrast(s, topos, "xs1load_selfref", "selfpredict")
        r1 = res["X1"].get("read")
        if not r1:
            res["X1_label"] = res["X1"].get("verdict", "NO-READ")
        elif r1["p"] < 0.05 and r1["median_pct"] < 0.0:
            res["X1_label"] = "BEATS-CD"
        elif r1["median_pct"] <= 0.0 or r1["p"] >= 0.05:
            res["X1_label"] = "CLOSES-GAP"
        else:
            res["X1_label"] = "CD-FASTER"
        r2 = res["X2"].get("read")
        v2 = r2["verdict"] if r2 else res["X2"].get("verdict", "NO-READ")
        res["X2_label"] = ("SECONDS-HELP" if v2 == "XS1LOAD_SELFREF-FASTER" else
                           "SECONDS-HELP (direction only)" if v2.startswith("XS1LOAD_SELFREF-FASTER") else
                           "SECONDS-HURT" if v2.startswith("FC1LOAD_SELFREF-FASTER") else
                           "NOT-SEPARATED" if r2 else v2)
    if any(k[2] == "xs1load_cdapply" for k in s):
        # seeded_cd_xs1_v1 (docs/lineages/seeded_cd_xs1_v1.md)
        res["G1"] = C.contrast(s, topos, "xs1load_cdapply", "cd")
        res["G2"] = C.contrast(s, topos, "xs1load_cdapply", "xs1load_selfref")
        res["reported_xs1load_cdapply_vs_bc1mpoff_cdapply"] = C.contrast(s, topos, "xs1load_cdapply", "bc1mpoff_cdapply")
        res["reported_bc1mpoff_cdapply_vs_cd"] = C.contrast(s, topos, "bc1mpoff_cdapply", "cd")
        res["reported_xs1load_cdapply_vs_selfpredict"] = C.contrast(s, topos, "xs1load_cdapply", "selfpredict")
        r1 = res["G1"].get("read")
        if not r1:
            res["G1_label"] = res["G1"].get("verdict", "NO-READ")
        elif r1["p"] < 0.05 and r1["median_pct"] <= -5.0:
            res["G1_label"] = "BEATS-CD"
        elif r1["p"] < 0.05 and r1["median_pct"] < 0.0:
            res["G1_label"] = "BEATS-CD (direction only)"
        elif r1["p"] < 0.05 and r1["median_pct"] > 0.0:
            res["G1_label"] = "CD-FASTER"
        else:
            res["G1_label"] = "TIES-CD"
    res["L1_label"] = l1_label(res["L1"])
    res["L2_label"] = l2_label(res["L2"], res["L1_label"])
    print(json.dumps(res, indent=1))
    for k, v in res.items():
        if not isinstance(v, dict):
            continue
        r = v.get("read") or {}
        print(f"{k:34s} median {r.get('median_pct', float('nan')):+7.2f} %  p={r.get('p')}  "
              f"first faster {r.get('a_faster')}/{r.get('n')}  dropped {sorted(v.get('dropped', {}))}  "
              f"{v.get('verdict', '')}", file=sys.stderr)
    print(f"L1: {res['L1_label']}   L2: {res['L2_label']}   S1: {res.get('S1_label')}   "
          f"F1: {res.get('F1_label')}   F2: {res.get('F2_label')}   X1: {res.get('X1_label')}   "
          f"X2: {res.get('X2_label')}   G1: {res.get('G1_label')}", file=sys.stderr)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
