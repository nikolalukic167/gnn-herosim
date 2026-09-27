#!/usr/bin/env python3
"""capacity_sweep_v1 -- the C1 capacity and C2 keep-alive reads (docs/lineages/capacity_sweep_v1.md).

  capacity_sweep_v1_read.py --dirs <ladderjit dir> <capacity dir> --selection selected.json [--out read.json]

Rungs, in order: x1.0 (w0x10d*), x1.1, x1.2, x1.3 (w0x1{1,2,3}d*), x1.5 (w0x15d*); 4 draws each, the same draw for
every arm. The learned arm is xs1load_selfref seed 1 (the only seed run on the new rungs, and on the x1 draws).

C1 knee (per arm, per topology): the highest rung r such that at r and every lower rung the arm's queue share
(averageQueueTime / averageElapsedTime, median over the 4 draws) is <= 0.80 -- the program's admissibility
threshold applied to the arm itself. 0 = not even x1.0. Capacity index: the rung at which the median share crosses
0.80, linearly interpolated between rungs (1.5 when it never crosses). Label: CAPACITY-GAIN if the learned knee is
above CD's on >= 8 of the topologies with the non-tied topologies' exact two-sided sign test p < 0.05, else
NO-CAPACITY-GAIN.

C2 (x1 draws, keep_alive removed for every arm): per topology the median over draws of the paired % learned vs CD.
PLACEMENT-LEAD (median <= -5 %, Wilcoxon p < 0.05) / PLACEMENT-LEAD (direction only) (median < 0, p < 0.05) /
CHURN-LEAD (otherwise; the lead does not survive without replica churn).

Reported: per rung paired contrasts of the learned arm vs cd / cd_inflight / selfpredict / reactive (wherever both
exist), per rung the topologies where reactive's median share <= 0.80 (admissible), and the median last/first-quarter
queue drift per arm and rung.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from math import comb
from statistics import median
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cd_gap_v1_read as C  # noqa: E402
import fresh_topo_burst_v1_read as F  # noqa: E402

DRAWS = (1, 2, 3, 4)
RUNGS = (("x10", 1.0), ("x11", 1.1), ("x12", 1.2), ("x13", 1.3), ("x15", 1.5))
LEARNED = ("xs1load_selfref", 1)
SHARE_BAR = 0.80
KNEE_TOPOS = 8


def _share(r: dict) -> float:
    return float(r["averageQueueTime"]) / float(r["averageElapsedTime"])


def _win(rung: str, k: int, ka: bool = False) -> str:
    return f"w0{rung}d{k}" + ("ka" if ka else "")


def _key(arm: str):
    return LEARNED if arm == "learned" else (arm, 0)


def _get(s: Dict, t: int, w: str, arm: str) -> Optional[dict]:
    kind, seed = _key(arm)
    return s.get((t, w, kind, seed))


def median_share(s: Dict, t: int, rung: str, arm: str) -> Optional[float]:
    rows = [_get(s, t, _win(rung, k), arm) for k in DRAWS]
    if any(r is None for r in rows):
        return None
    return median(_share(r) for r in rows)


def knee(shares: List[Optional[float]]) -> Optional[float]:
    """Highest rung with share <= bar there and below; None if a needed rung is missing."""
    best = 0.0
    for (_, x), sh in zip(RUNGS, shares):
        if sh is None:
            return None
        if sh > SHARE_BAR:
            return best
        best = x
    return best


def capacity_index(shares: List[Optional[float]]) -> Optional[float]:
    pts = [(x, sh) for (_, x), sh in zip(RUNGS, shares)]
    if any(sh is None for _, sh in pts):
        return None
    if pts[0][1] > SHARE_BAR:
        return pts[0][0]
    for (x0, s0), (x1, s1) in zip(pts, pts[1:]):
        if s1 > SHARE_BAR:
            return x0 + (x1 - x0) * (SHARE_BAR - s0) / (s1 - s0)
    return pts[-1][0]


def sign_test(pos: int, neg: int) -> float:
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def paired(s: Dict, topos: List[int], rung: str, arm: str, ref: str, ka: bool = False) -> dict:
    vals, dropped = {}, []
    for t in topos:
        pcts = []
        for k in DRAWS:
            a, b = _get(s, t, _win(rung, k, ka), arm), _get(s, t, _win(rung, k, ka), ref)
            if a is None or b is None:
                break
            pcts.append(F._pct(F._el(a), F._el(b)))
        if len(pcts) == len(DRAWS):
            vals[t] = median(pcts)
        else:
            dropped.append(t)
    out = {"arm": arm, "ref": ref, "rung": rung, "keep_alive_removed": ka, "dropped": dropped}
    if len(vals) < F.MIN_TOPOLOGIES:
        return dict(out, verdict="DESIGN-SHORT", n=len(vals))
    out["read"] = F._read(vals, arm.upper(), ref.upper())
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    topos = json.load(open(a.selection))["topologies"]
    s = C._load(a.dirs)
    dirty = sorted({r["arm"] for r in s.values() if (r.get("code") or {}).get("dirty")})
    res: Dict = {"invalid_dirty_runs": dirty[:10]}

    arms = ("learned", "cd", "reactive", "selfpredict", "cd_inflight")
    shares = {arm: {t: [median_share(s, t, r, arm) for r, _ in RUNGS] for t in topos} for arm in arms}
    knees = {arm: {t: knee(shares[arm][t]) for t in topos} for arm in arms}
    cap = {arm: {t: capacity_index(shares[arm][t]) for t in topos} for arm in arms}
    res["shares"] = {arm: {str(t): v for t, v in d.items()} for arm, d in shares.items()}
    res["knees"] = {arm: {str(t): v for t, v in d.items()} for arm, d in knees.items()}
    res["capacity_index"] = {arm: {str(t): v for t, v in d.items()} for arm, d in cap.items()}

    both = [t for t in topos if knees["learned"][t] is not None and knees["cd"][t] is not None]
    up = sum(knees["learned"][t] > knees["cd"][t] for t in both)
    down = sum(knees["learned"][t] < knees["cd"][t] for t in both)
    p = sign_test(up, down)
    c1 = {"topologies": both, "learned_knee_higher": up, "learned_knee_lower": down,
          "tied": len(both) - up - down, "sign_p": p,
          "capacity_ratio_learned_over_cd": {str(t): (cap["learned"][t] / cap["cd"][t])
                                             for t in both if cap["learned"][t] and cap["cd"][t]}}
    ratios = list(c1["capacity_ratio_learned_over_cd"].values())
    c1["median_capacity_ratio"] = median(ratios) if ratios else None
    if len(both) < F.MIN_TOPOLOGIES:
        c1["label"] = "DESIGN-SHORT"
    else:
        c1["label"] = "CAPACITY-GAIN" if (up >= KNEE_TOPOS and p < 0.05) else "NO-CAPACITY-GAIN"
    res["C1"] = c1

    res["per_rung"] = {}
    for r, x in RUNGS:
        row = {"admissible_topologies": [t for t in topos
                                         if shares["reactive"][t][[rr for rr, _ in RUNGS].index(r)] is not None
                                         and shares["reactive"][t][[rr for rr, _ in RUNGS].index(r)] <= SHARE_BAR]}
        for ref in ("cd", "cd_inflight", "selfpredict", "reactive"):
            row[f"learned_vs_{ref}"] = paired(s, topos, r, "learned", ref)
        drift = {}
        for arm in arms:
            vals = [(_get(s, t, _win(r, k), arm) or {}).get("queue_drift") for t in topos for k in DRAWS]
            vals = [v["last_over_first"] for v in vals if v and v.get("last_over_first") is not None]
            drift[arm] = median(vals) if vals else None
        row["median_queue_drift_last_over_first"] = drift
        res["per_rung"][f"x{x}"] = row

    c2 = paired(s, topos, "x10", "learned", "cd", ka=True)
    rd = c2.get("read")
    if not rd:
        c2["label"] = c2.get("verdict", "NO-READ")
    elif rd["p"] < 0.05 and rd["median_pct"] <= -5.0:
        c2["label"] = "PLACEMENT-LEAD"
    elif rd["p"] < 0.05 and rd["median_pct"] < 0.0:
        c2["label"] = "PLACEMENT-LEAD (direction only)"
    else:
        c2["label"] = "CHURN-LEAD"
    c2["with_churn_same_draws"] = paired(s, topos, "x10", "learned", "cd")
    c2["learned_vs_selfpredict"] = paired(s, topos, "x10", "learned", "selfpredict", ka=True)
    c2["learned_vs_reactive"] = paired(s, topos, "x10", "learned", "reactive", ka=True)
    churn = {}
    for arm in ("learned", "cd", "selfpredict", "reactive"):
        per = []
        for t in topos:
            on = [_get(s, t, _win("x10", k), arm) for k in DRAWS]
            off = [_get(s, t, _win("x10", k, True), arm) for k in DRAWS]
            if all(on) and all(off):
                per.append(median(100.0 * (F._el(o) - F._el(f)) / F._el(o) for o, f in zip(on, off)))
        churn[arm] = median(per) if per else None
    c2["median_churn_share_pct"] = churn
    res["C2"] = c2

    print(json.dumps(res, indent=1))
    print(f"C1 {c1['label']}: learned knee higher on {up}, lower on {down}, tied {c1['tied']} (sign p={p:.4g}); "
          f"median capacity ratio {c1['median_capacity_ratio']}", file=sys.stderr)
    for x, row in res["per_rung"].items():
        for ref in ("cd", "cd_inflight", "selfpredict", "reactive"):
            rr = row[f"learned_vs_{ref}"].get("read") or {}
            print(f"  {x} learned vs {ref:11s} median {rr.get('median_pct', float('nan')):+7.2f} %  p={rr.get('p')}  "
                  f"faster {rr.get('a_faster')}/{rr.get('n')}", file=sys.stderr)
        print(f"  {x} admissible {len(row['admissible_topologies'])}/{len(topos)}  drift "
              f"{ {k: (round(v, 2) if v else v) for k, v in row['median_queue_drift_last_over_first'].items()} }",
              file=sys.stderr)
    r2 = c2.get("read") or {}
    print(f"C2 {c2['label']}: learned vs CD, keep_alive removed, median {r2.get('median_pct', float('nan')):+.2f} % "
          f"p={r2.get('p')} faster {r2.get('a_faster')}/{r2.get('n')}; churn share {churn}", file=sys.stderr)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
