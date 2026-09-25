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
    print(f"L1: {res['L1_label']}   L2: {res['L2_label']}", file=sys.stderr)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
