#!/usr/bin/env python3
"""burst_ladder_v1 -- the live S0 read (docs/lineages/burst_ladder_v1.md).

  burst_ladder_v1_read.py --dir <ladder gate dir> --ref <ref_fresh_1ae90af dir> <xs1 dir> <xs1cd dir>
                          --selection selected.json [--out read.json]

Per rung (one window each: w0 at intensity x1, x1.5, x2) the statistic is cd_gap_v1_read.contrast's with that
single window: per environment the median over seeds of the paired %, exact two-sided Wilcoxon over topologies,
a topology missing any run of either arm dropped by name. Negative = the first-named arm is faster.

  A    xs1load_selfref vs cd            the learned lead
  B    xs1load_selfref vs cd_inflight   the lead over the in-flight-aware CD (the blocking contrast)
  I    cd_inflight vs cd                what the CD defect is worth
  share  per topology (cd - cd_inflight) / (cd - xs1load_selfref) in elapsed, seeds' median for the learned
         arm, over topologies where the learned arm leads; median over those topologies
Labels per rung, in this order: NO-LEAD (A not faster at p < 0.05) / CD-DEFECT (share >= 0.80, or B not faster)
/ LEARNED-LEAD-REAL (B faster at p < 0.05; "direction only" when |B| < 5 %).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from statistics import median
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cd_gap_v1_read as C  # noqa: E402
import fresh_topo_burst_v1_read as F  # noqa: E402

RUNGS = ("w0x10", "w0x15", "w0x20")
SHARE_BAR = 0.80
LEARNED = "xs1load_selfref"


def contrast(s: Dict, topos: List[int], window: str, arm: str, ref: str) -> dict:
    a_seeds, r_seeds = C._registered_seeds(arm), C._registered_seeds(ref)
    same_seed = a_seeds != [0] and r_seeds != [0]
    complete, dropped = [], {}
    for t in topos:
        missing = [(k, sd) for k, seeds in ((arm, a_seeds), (ref, r_seeds)) for sd in seeds
                   if (t, window, k, sd) not in s]
        (dropped.__setitem__(str(t), missing) if missing else complete.append(t))
    out = {"arm": arm, "ref": ref, "window": window, "topologies": complete, "dropped": dropped,
           "code_by_arm": {k: sorted({s[(t, window, k, sd)]["code"]["commit"][:7] for t in complete
                                      for sd in (a_seeds if k == arm else r_seeds)}) for k in (arm, ref)}}
    if any(s[(t, window, k, sd)]["code"]["dirty"] for t in complete
           for k, seeds in ((arm, a_seeds), (ref, r_seeds)) for sd in seeds):
        return dict(out, verdict="INVALID-CODE-STATE")
    if len(complete) < F.MIN_TOPOLOGIES:
        return dict(out, verdict="DESIGN-SHORT")

    def env(t):
        if same_seed:
            return median(F._pct(F._el(s[(t, window, arm, sd)]), F._el(s[(t, window, ref, sd)])) for sd in a_seeds)
        return median(F._pct(F._el(s[(t, window, arm, sd)]), F._el(s[(t, window, ref, 0)])) for sd in a_seeds)

    out["read"] = F._read({t: env(t) for t in complete}, arm.upper(), ref.upper())
    out["decomposition"] = {k: C._decomp([s[(t, window, k, sd)] for t in complete
                                          for sd in (a_seeds if k == arm else r_seeds)]) for k in (arm, ref)}
    return out


def share(s: Dict, topos: List[int], window: str) -> dict:
    per = {}
    for t in topos:
        try:
            cd = F._el(s[(t, window, "cd", 0)])
            fix = F._el(s[(t, window, "cd_inflight", 0)])
            learned = median(F._el(s[(t, window, LEARNED, sd)]) for sd in (1, 2, 3, 4))
        except KeyError:
            continue
        lead = cd - learned
        per[str(t)] = {"cd": cd, "cd_inflight": fix, "learned_median": learned, "lead_s": lead,
                       "share": (cd - fix) / lead if lead > 0 else None}
    shares = [v["share"] for v in per.values() if v["share"] is not None]
    return {"per_topology": per, "n_leading": len(shares), "median_share": median(shares) if shares else None}


def admissibility(s: Dict, topos: List[int], window: str) -> dict:
    per = {}
    for t in topos:
        r = s.get((t, window, "reactive", 0))
        per[str(t)] = None if r is None else r.get("queue_share")
    ok = [t for t, v in per.items() if v is not None and float(v) <= 0.80]
    return {"reactive_queue_share": per, "admissible": ok, "n_admissible": len(ok),
            "rule": "reactive queue share <= 0.80; a missing reactive run is unknown, not a pass"}


def label(a: dict, b: dict, sh: dict) -> str:
    ra, rb = a.get("read"), b.get("read")
    if not ra or not rb:
        return a.get("verdict") or b.get("verdict") or "NO-READ"
    if not (ra["p"] < 0.05 and ra["median_pct"] < 0):
        return "NO-LEAD"
    if sh["median_share"] is None:
        return "NO-LEAD"
    if sh["median_share"] >= SHARE_BAR or not (rb["p"] < 0.05 and rb["median_pct"] < 0):
        return "CD-DEFECT"
    return "LEARNED-LEAD-REAL" + ("" if rb["median_pct"] <= -5.0 else " (direction only)")


def witness(s: Dict, ref: Dict, topos: List[int]) -> dict:
    """x1 is the existing w0: every ladder run whose arm also ran at w0 in the reference gates must equal it."""
    same, diff = 0, []
    for (t, w, k, sd), r in s.items():
        if w != "w0x10" or t not in topos:
            continue
        o = ref.get((t, "w0", k, sd))
        if o is None:
            continue
        if float(o["total_rtt"]) == float(r["total_rtt"]):
            same += 1
        else:
            diff.append(f"{t}/{k}/s{sd}: {r['total_rtt']} vs {o['total_rtt']}")
    return {"equal": same, "differ": len(diff), "examples": diff[:8]}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--ref", nargs="*", default=[])
    ap.add_argument("--selection", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    topos = json.load(open(a.selection))["topologies"]
    s = C._load([a.dir])
    res: Dict = {"witness_x1_vs_w0": witness(s, C._load(a.ref), topos) if a.ref else None}
    for w in RUNGS:
        A = contrast(s, topos, w, LEARNED, "cd")
        B = contrast(s, topos, w, LEARNED, "cd_inflight")
        sh = share(s, topos, w)
        res[w] = {
            "A": A, "B": B, "I": contrast(s, topos, w, "cd_inflight", "cd"), "share": sh,
            "reported": {
                "cdapply_vs_cd": contrast(s, topos, w, "xs1load_cdapply", "cd"),
                "cdapply_vs_cd_inflight": contrast(s, topos, w, "xs1load_cdapply", "cd_inflight"),
                "selfpredict_vs_cd": contrast(s, topos, w, "selfpredict", "cd"),
                "learned_vs_selfpredict": contrast(s, topos, w, LEARNED, "selfpredict"),
            },
            "admissibility": admissibility(s, topos, w),
            "label": label(A, B, sh),
        }
    growth = [((res[w]["B"].get("read") or {}).get("median_pct")) for w in RUNGS]
    res["B_median_by_rung"] = dict(zip(RUNGS, growth))
    res["lead_grows"] = (all(g is not None for g in growth) and growth[0] > growth[1] > growth[2])
    print(json.dumps(res, indent=1))
    for w in RUNGS:
        r = res[w]
        for k in ("A", "B", "I"):
            rd = r[k].get("read") or {}
            print(f"{w} {k}  {r[k]['arm']} vs {r[k]['ref']}: median {rd.get('median_pct', float('nan')):+.2f} % "
                  f"p={rd.get('p')} faster {rd.get('a_faster')}/{rd.get('n')} dropped {sorted(r[k]['dropped'])} "
                  f"{r[k].get('verdict', '')}", file=sys.stderr)
        print(f"{w} share recovered {r['share']['median_share']} over {r['share']['n_leading']} leading topologies; "
              f"admissible {r['admissibility']['n_admissible']}/{len(topos)}; LABEL {r['label']}", file=sys.stderr)
    print(f"B by rung {res['B_median_by_rung']}  grows: {res['lead_grows']}  witness {res['witness_x1_vs_w0']}",
          file=sys.stderr)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
