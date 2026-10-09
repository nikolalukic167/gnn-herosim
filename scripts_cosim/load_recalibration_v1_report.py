#!/usr/bin/env python3
"""Read-only finals report: CD effective share against its band, every CD guard, Knative for context, per rung directory.

  load_recalibration_v1_report.py --root <dir with m<tag>/gate> --out report.json [--cell 9602:g0]
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import load_recalibration_v1_bisect as B  # noqa: E402


def arm_block(cells, expected):
    if not cells:
        return {"cells": 0}
    g = B.guards(cells, expected)
    return {"cells": len(cells), "median_effective_share": st.median(c["effective_queue_share"] for c in cells),
            "median_old_queue_share": st.median(c["queue_share"] for c in cells),
            "median_latency_s": st.median(c["latency_s"] for c in cells), "guards": {k: v for k, v in g.items() if k != "values"},
            "values": g["values"], "per_cell_effective_share": [round(c["effective_queue_share"], 4) for c in cells],
            "per_cell": {f"{c['topology']}_{c['window']}": {k: c[k] for k in ("effective_queue_share", "latency_s", "request_failure_pct", "p95_s", "backlog_ratio", "in_system_ratio", "in_system_half", "end_over_last_arrival")} for c in cells}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--topologies", type=int, nargs="+", default=[9601, 9602, 9607, 9608])
    ap.add_argument("--windows", nargs="+", default=["g0", "g1"])
    ap.add_argument("--rungs", nargs="+", default=None, help="TAG=RUNG, e.g. m1p2584=light; unlisted tags have no band")
    a = ap.parse_args()
    names = dict(s.split("=") for s in (a.rungs or []))
    ev = B.Evaluator(Namespace(topologies=a.topologies, windows=a.windows, work=a.root, seed=[], readonly=[]))
    rep = {}
    for d in sorted(a.root.glob("m*p*/gate")):
        tag = d.parent.name
        m = float(tag[1:].replace("p", "."))
        r = ev.collect(m, d)
        rung = names.get(tag)
        lo, hi = B.BANDS.get(rung, (None, None))
        cd = arm_block(r["cells"]["cd"], ev.expected)
        share = cd.get("median_effective_share")
        rep[tag] = {"m": m, "rung": rung, "band": [lo, hi], "in_band": None if lo is None or share is None else lo <= share <= hi,
                    "cd": cd, "reactive": arm_block(r["cells"]["reactive"], ev.expected), "missing": r["missing"],
                    "allowed_for_rung": B.allowed(r["guards"]["cd"], rung) if rung else None}
    a.out.write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
