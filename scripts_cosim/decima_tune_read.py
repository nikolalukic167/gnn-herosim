#!/usr/bin/env python3
"""decima_rule_v1 tuning read: per alpha, the paired % vs CD per tuning cell (median over the 4 windows),
then the median over the tuning topologies. Picks the alpha with the lowest median; ties go to Decima's
reported best (-1). Never reads the study cells.

  decima_tune_read.py --dir <gates/decimatune> [--out tune.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from statistics import median
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fresh_topo_burst_v1_gate as G  # noqa: E402
import fresh_topo_burst_v1_read as F  # noqa: E402


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    s = F._summaries(a.dir)
    res = {}
    for kind, alpha in G.DECIMA_TUNE_ALPHAS.items():
        per_topo, missing = {}, []
        for t in G.DECIMA_TUNE_TOPOS:
            pcts = []
            for w in G.WINDOWS:
                if (t, w, kind, 0) in s and (t, w, "cd", 0) in s:
                    pcts.append(F._pct(F._el(s[(t, w, kind, 0)]), F._el(s[(t, w, "cd", 0)])))
                else:
                    missing.append(f"{t}/{w}")
            if len(pcts) == len(G.WINDOWS):
                per_topo[t] = median(pcts)
        res[kind] = {"alpha": alpha, "per_topology": per_topo, "missing": missing,
                     "median_pct_vs_cd": median(per_topo.values()) if per_topo else None}
    ok = {k: v for k, v in res.items() if v["median_pct_vs_cd"] is not None}
    if not ok:
        raise SystemExit("FAIL LOUD: no alpha has a complete tuning read")
    best = min(ok, key=lambda k: (round(ok[k]["median_pct_vs_cd"], 6), abs(ok[k]["alpha"] - (-1.0))))
    res["chosen"] = {"kind": best, "alpha": ok[best]["alpha"]}
    for k, v in res.items():
        if k != "chosen":
            print(f"{k:12s} alpha={v['alpha']:+.1f}  vs CD median {v['median_pct_vs_cd']}  missing {v['missing']}",
                  file=sys.stderr)
    print(f"chosen alpha: {res['chosen']['alpha']}", file=sys.stderr)
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
