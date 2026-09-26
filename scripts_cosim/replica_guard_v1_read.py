#!/usr/bin/env python3
"""replica_guard_v1 -- read the guard gate (docs/lineages/replica_guard_v1.md).

  replica_guard_v1_read.py --dir <guard gate dir> --selection selected.json [--out read.json]

x1 perturbed draws d1-d4 (burst_ladder_v1 Amendment 1): per topology each arm's value is the median elapsed
over all its runs (draws x seeds 1-4 for the learned arms, draws for CD); contrast 100 * (arm / ref - 1),
exact two-sided Wilcoxon over topologies, a topology missing any run dropped by name. Negative = first arm faster.
  K1  9434: guard vs twin per seed, on w0 as every gate ran it and on the draws
  K2  guard (xs1load_selfrefkw) vs CD on the draws, 12 topologies -- the bar
  K3  guard vs twin on the draws, the 11 topologies other than 9434; and w1-w3 (seeds 1-2, unperturbed)
  reported: twin vs CD on the draws; guard moves per 1,000 tasks
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

DRAWS = [f"w0x10d{k}" for k in (1, 2, 3, 4)]
GUARD, TWIN = "xs1load_selfrefkw", "xs1load_selfref"


def _runs(s: Dict, t: int, windows: List[str], arm: str, seeds) -> Optional[List[float]]:
    keys = [(t, w, arm, sd) for w in windows for sd in seeds]
    if any(k not in s for k in keys):
        return None
    return [F._el(s[k]) for k in keys]


def contrast(s, topos, windows, arm, ref, arm_seeds, ref_seeds) -> Dict:
    per, dropped = {}, []
    for t in topos:
        a, r = _runs(s, t, windows, arm, arm_seeds), _runs(s, t, windows, ref, ref_seeds)
        if a is None or r is None:
            dropped.append(t)
            continue
        per[t] = 100.0 * (median(a) / median(r) - 1.0)
    out = {"arm": arm, "ref": ref, "windows": windows, "dropped": dropped, "n": len(per)}
    if len(per) < F.MIN_TOPOLOGIES:
        return dict(out, verdict="DESIGN-SHORT", per_topology=per)
    out["read"] = F._read(per, arm.upper(), ref.upper())
    return out


def k2_label(c: Dict) -> str:
    r = c.get("read")
    if not r:
        return c.get("verdict", "NO-READ")
    if r["p"] < 0.05 and r["median_pct"] <= -5.0:
        return "BEATS-CD"
    if r["p"] < 0.05 and r["median_pct"] < 0.0:
        return "BEATS-CD (direction only)"
    if r["p"] < 0.05 and r["median_pct"] > 0.0:
        return "CD-FASTER"
    return "TIES"


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
    L = (1, 2, 3, 4)
    res: Dict = {"code": sorted({v["code"]["commit"][:7] for v in s.values()})}
    res["K1"] = {w: {str(sd): {arm: (F._el(s[(9434, w, arm, sd)]) if (9434, w, arm, sd) in s else None)
                               for arm in (GUARD, TWIN)} for sd in L} for w in ["w0"] + DRAWS}
    res["K1"]["cd_on_draws"] = {w: (F._el(s[(9434, w, "cd", 0)]) if (9434, w, "cd", 0) in s else None) for w in DRAWS}
    res["K2"] = contrast(s, topos, DRAWS, GUARD, "cd", L, (0,))
    res["K2_label"] = k2_label(res["K2"])
    others = [t for t in topos if t != 9434]
    res["K3_draws"] = contrast(s, others, DRAWS, GUARD, TWIN, L, L)
    for w in ("w1", "w2", "w3"):
        res[f"K3_{w}"] = contrast(s, topos, [w], GUARD, TWIN, (1, 2), (1, 2))
    res["reported_twin_vs_cd"] = contrast(s, topos, DRAWS, TWIN, "cd", L, (0,))
    moves = [s[k]["schedulerCounters"].get("keepwarm_moves", 0) / s[k]["num_tasks"] * 1000
             for k in s if k[2] == GUARD]
    res["guard_moves_per_1000_tasks"] = {"median": median(moves), "max": max(moves)} if moves else None
    for k, v in res.items():
        if isinstance(v, dict) and "arm" in v:
            q = v.get("read") or {}
            print(f"{k:22s} median {q.get('median_pct', float('nan')):+7.2f} %  p={q.get('p')}  "
                  f"faster {q.get('a_faster')}/{q.get('n')}  dropped {v['dropped']}", file=sys.stderr)
    print(f"K2: {res['K2_label']}  K1: {json.dumps(res['K1'])}", file=sys.stderr)
    print(json.dumps(res, indent=1))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
