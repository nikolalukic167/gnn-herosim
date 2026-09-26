#!/usr/bin/env python3
"""burst_ladder_v1 Amendment 1 -- read the perturbed-draw rerun (docs/lineages/burst_ladder_v1.md).

  burst_ladder_jit_read.py --dir <ladderjit gate dir> --selection selected.json [--out read.json]

Per rung (x1.5: draws d1-d4, rules x1 run per draw, xs1load_selfref seeds 1-4 per draw; x1: rules and seed 1),
per topology each arm's value is the median elapsed over all its runs (draws, and seeds for the learned arm);
the contrast is 100 * (arm / ref - 1) on those medians, exact two-sided Wilcoxon over topologies (a topology
missing any run of either arm is dropped by name). Negative = the first-named arm is faster.
Dispersion per arm and topology: max / min elapsed across draws (a rule) or across draws within each seed,
median over seeds (the learned arm), plus the spread over all its runs.
Label on the learned arm vs CD: OVERLOAD-LEAD-SURVIVES (median <= -10 %, p < 0.05) / DIRECTION-ONLY
(median < 0, p < 0.05) / DISSOLVES.
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

DRAWS = (1, 2, 3, 4)
LEARNED = "xs1load_selfref"
DESIGN = {"x15": {"rules": ("reactive", "cd", "cd_inflight", "selfpredict"), "seeds": (1, 2, 3, 4)},
          "x10": {"rules": ("cd", "cd_inflight"), "seeds": (1,)}}


def runs(s: Dict, t: int, rung: str, arm: str) -> Optional[List[float]]:
    seeds = DESIGN[rung]["seeds"] if arm == LEARNED else (0,)
    keys = [(t, f"w0{rung}d{k}", arm, sd) for k in DRAWS for sd in seeds]
    if any(k not in s for k in keys):
        return None
    return [F._el(s[k]) for k in keys]


def dispersion(s: Dict, t: int, rung: str, arm: str) -> Dict:
    seeds = DESIGN[rung]["seeds"] if arm == LEARNED else (0,)
    per_seed = []
    for sd in seeds:
        v = [F._el(s[(t, f"w0{rung}d{k}", arm, sd)]) for k in DRAWS]
        per_seed.append({"min": min(v), "max": max(v), "ratio": max(v) / min(v)})
    allv = runs(s, t, rung, arm)
    return {"draw_ratio_median_over_seeds": median(p["ratio"] for p in per_seed),
            "all_min": min(allv), "all_max": max(allv), "median": median(allv)}


def contrast(s: Dict, topos: List[int], rung: str, arm: str, ref: str) -> Dict:
    per, dropped = {}, []
    for t in topos:
        a, r = runs(s, t, rung, arm), runs(s, t, rung, ref)
        if a is None or r is None:
            dropped.append(t)
            continue
        per[t] = 100.0 * (median(a) / median(r) - 1.0)
    out = {"arm": arm, "ref": ref, "rung": rung, "dropped": dropped, "n": len(per)}
    if len(per) < F.MIN_TOPOLOGIES:
        return dict(out, verdict="DESIGN-SHORT")
    out["read"] = F._read(per, arm.upper(), ref.upper())
    return out


def label(c: Dict) -> str:
    r = c.get("read")
    if not r:
        return c.get("verdict", "NO-READ")
    if r["p"] < 0.05 and r["median_pct"] <= -10.0:
        return "OVERLOAD-LEAD-SURVIVES"
    if r["p"] < 0.05 and r["median_pct"] < 0.0:
        return "DIRECTION-ONLY"
    return "DISSOLVES"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    topos = json.load(open(a.selection))["topologies"]
    s = C._load([a.dir])
    if any(v["code"]["dirty"] for v in s.values()):
        raise SystemExit("FAIL LOUD: a run was served from a dirty tree")
    res: Dict = {"code": sorted({v["code"]["commit"][:7] for v in s.values()})}
    for rung, d in DESIGN.items():
        arms = (LEARNED,) + d["rules"]
        r = {f"{LEARNED}_vs_{ref}": contrast(s, topos, rung, LEARNED, ref) for ref in d["rules"]}
        r.update({f"{x}_vs_cd": contrast(s, topos, rung, x, "cd") for x in d["rules"] if x != "cd"})
        r["label"] = label(r[f"{LEARNED}_vs_cd"])
        r["dispersion"] = {str(t): {arm: dispersion(s, t, rung, arm) for arm in arms
                                    if runs(s, t, rung, arm) is not None} for t in topos}
        res[rung] = r
        for k, v in r.items():
            if isinstance(v, dict) and "arm" in v:
                q = v.get("read") or {}
                print(f"{rung} {k:30s} median {q.get('median_pct', float('nan')):+7.2f} %  p={q.get('p')}  "
                      f"faster {q.get('a_faster')}/{q.get('n')}  dropped {v['dropped']}", file=sys.stderr)
        print(f"{rung} label: {r['label']}", file=sys.stderr)
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
