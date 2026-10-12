#!/usr/bin/env python3
"""long_horizon_s0_v1 reader (docs/lineages/long_horizon_s0_v1.md), committed before any data.

Per contrast (GNN-family arm G vs hand arm H at one rung) and per hourly arrival bin h = 2..24 (bin 1 is warm-up and is excluded):
  g_h = (mean latency of G in bin h - mean latency of H in bin h) / mean latency of H in bin h, in %; g < 0 = the GNN is faster.
  S = OLS slope of g_h on h over h = 2..24, x 24 h (pp).
Contrasts: {ra_gnn_eng_nosplit, ra_gnn_eng_cdxapply} x {cd at heavy; cd and cd_exactS at moderate} = 6.
Reading (signed bar, directional):
  HORIZON-GNN-GAINS   S <= -2 pp on at least one contrast -> register the multi-day gate
  HORIZON-FLAT        |S| <= 2 pp on every contrast -> CLOSE
  HORIZON-HAND-GAINS  S >= +2 pp on some contrast and none <= -2 pp -> CLOSE (mixed signs are reported, and close unless a
                      GNN-favourable contrast is <= -2 pp, which is HORIZON-GNN-GAINS)
  INCOMPLETE          a contrast is missing (a failed or absent cell): unknown is not a pass, no reading
Disclosed with the read: one topology (a direction, never a magnitude), the hour-to-hour SD of g over h = 2..24 per contrast, g_1.

  long_horizon_s0_v1_read.py --out <gate out dir> --topo 16900 [--json read.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
from typing import Dict, List, Optional

GNN = ("ra_gnn_eng_nosplit", "ra_gnn_eng_cdxapply")
HAND = {"heavy": ("cd",), "moderate": ("cd", "cd_exactS")}
SEED = {"cd": 0, "cd_exactS": 0, "ra_gnn_eng_nosplit": 1, "ra_gnn_eng_cdxapply": 1}
BAR_PP = 2.0
HOURS = range(2, 25)


def load(out: str, topo: int, rung: str, arm: str) -> Optional[dict]:
    f = os.path.join(out, f"cc40s{topo}__L24{rung}__{arm}_s{SEED[arm]}.summary.json")
    return json.load(open(f)) if os.path.exists(f) else None


def ols_slope(xs: List[float], ys: List[float]) -> float:
    mx, my = st.mean(xs), st.mean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


def bins_by_hour(summary: dict) -> Dict[int, dict]:
    return {int(b["bin"]) + 1: b for b in summary["hourlyBins"]["bins"]}


def contrast(g: dict, h: dict) -> Dict[str, object]:
    bg, bh = bins_by_hour(g), bins_by_hour(h)
    gh = {}
    for hr in range(1, 25):
        if hr in bg and hr in bh and bg[hr]["n"] and bh[hr]["n"]:
            gh[hr] = 100.0 * (float(bg[hr]["latency"]) - float(bh[hr]["latency"])) / float(bh[hr]["latency"])
    missing = [hr for hr in HOURS if hr not in gh]
    if missing:
        return {"complete": False, "missing_hours": missing, "g": gh}
    xs = list(HOURS)
    ys = [gh[x] for x in xs]
    s = ols_slope(xs, ys) * 24.0
    return {"complete": True, "g": gh, "g_1": gh.get(1), "S_pp": s, "sd_g_pp": st.stdev(ys),
            "mean_g_h2_24": st.mean(ys), "block_means_pp": [st.mean(ys[i:i + 4]) for i in range(0, len(ys), 4)]}


def breakdown(summary: dict) -> List[dict]:
    keys = ("n", "latency", "queue", "cold", "exchange", "exec", "in_system_at_end", "last_arrival_done_minus_end_s")
    return [{"hour": int(b["bin"]) + 1, **{k: b.get(k) for k in keys}} for b in summary["hourlyBins"]["bins"]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--topo", type=int, default=16900)
    ap.add_argument("--json")
    a = ap.parse_args()
    failed = sorted(os.path.basename(f) for f in glob.glob(os.path.join(a.out, "*.failed.json")))
    res: Dict[str, object] = {"topology": a.topo, "failed_cells": failed, "contrasts": {}, "cells": {}}
    for rung in ("moderate", "heavy"):
        for arm in sorted(set(GNN) | set(HAND[rung])):
            s = load(a.out, a.topo, rung, arm)
            res["cells"][f"{rung}/{arm}"] = None if s is None else {
                "num_tasks": s.get("num_tasks"), "averageElapsedTime": s.get("averageElapsedTime"), "wallclock_s": s.get("wallclock_s"),
                "endTime": s.get("endTime"), "arrival_end": s.get("arrival_end"), "bins": breakdown(s)}
        for g in GNN:
            for h in HAND[rung]:
                sg, sh = load(a.out, a.topo, rung, g), load(a.out, a.topo, rung, h)
                key = f"{rung}: {g} vs {h}"
                res["contrasts"][key] = {"complete": False, "why": "cell missing"} if sg is None or sh is None else contrast(sg, sh)
    cs = res["contrasts"]
    if not all(c["complete"] for c in cs.values()):
        verdict = "INCOMPLETE"
    elif any(c["S_pp"] <= -BAR_PP for c in cs.values()):
        verdict = "HORIZON-GNN-GAINS"
    elif all(abs(c["S_pp"]) <= BAR_PP for c in cs.values()):
        verdict = "HORIZON-FLAT"
    else:
        verdict = "HORIZON-HAND-GAINS"
    res["verdict"] = verdict
    res["mixed_signs"] = len({(c["S_pp"] > 0) for c in cs.values() if c.get("complete")}) > 1
    for k, c in cs.items():
        if c.get("complete"):
            print(f"{k:48s} S {c['S_pp']:+6.2f} pp  g_1 {c['g_1']:+6.2f}  mean g(2..24) {c['mean_g_h2_24']:+6.2f}  "
                  f"SD {c['sd_g_pp']:.2f}  blocks {' '.join(f'{x:+.1f}' for x in c['block_means_pp'])}")
        else:
            print(f"{k:48s} INCOMPLETE {c.get('why') or c.get('missing_hours')}")
    print(f"failed cells: {failed or 'none'}")
    print(f"VERDICT {verdict}" + (" (mixed signs across contrasts)" if res["mixed_signs"] else ""))
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(res, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
