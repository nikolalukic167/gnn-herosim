#!/usr/bin/env python3
"""grounded_workload_v1 -- the gate read (docs/lineages/grounded_workload_v1.md).

  grounded_workload_v1_read.py --gate <grounded dir> --capacity <capacity dir> --selection selected.json [--out read.json]

Cells are (topology, window) over the 12 study topologies and the Alibaba-grounded windows g0-g3. A cell is
ADMISSIBLE iff its reactive run completed with queue share (queue / elapsed) <= 0.80; a missing or failed
reactive run is not a pass. Only admissible cells enter a contrast.

Learned vs rule: per topology, the median over (admissible window, seed 1-4) of the paired % on the same cell.
Learned vs learned: the same, pairing the seed. Rule vs rule: the median over admissible windows. A pair with a
missing run is dropped by name and listed. Exact two-sided Wilcoxon over topologies; fewer than 8 topologies
reads DESIGN-SHORT.

Labels: CONFIRMED (median <= -5 %, p < 0.05) / DIRECTION-ONLY (median < 0, p < 0.05) /
REF-FASTER (median > 0, p < 0.05) / NOT-SEPARATED.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from statistics import median
from typing import Dict, List, Optional, Sequence, Tuple

from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fresh_topo_burst_v1_read as F  # noqa: E402

WINDOWS = ("g0", "g1", "g2", "g3")
SEEDS = (1, 2, 3, 4)
RULES = ("random", "reactive", "batched", "selfpredict", "cd", "decima")
LEARNED = ("xs1load_selfref", "xs1load", "gnnedge0", "mpoff")
PRIMARY = "xs1load_selfref"
SHARE_MAX = 0.80
BAR = 5.0
ALPHA = 0.05
MIN_TOPOLOGIES = 8
WITNESS = ("w0x11d1", (9119, 9420))

Key = Tuple[int, str, str, int]


def load(d: str) -> Dict[Key, dict]:
    return F._summaries(d)


def failures(d: str) -> Dict[Key, str]:
    out = {}
    for f in glob.glob(os.path.join(d, "*.failed.json")):
        name = os.path.basename(f)[: -len(".failed.json")]
        topo, window, rest = name.split("__")
        kind, seed = rest.rsplit("_s", 1)
        out[(int(topo[len("cc40s"):]), window, kind, int(seed))] = name
    return out


def share(r: dict) -> float:
    return float(r["averageQueueTime"]) / float(r["averageElapsedTime"])


def admissible(s: Dict[Key, dict], topos: Sequence[int]) -> Dict[Tuple[int, str], Optional[float]]:
    out = {}
    for t in topos:
        for w in WINDOWS:
            r = s.get((t, w, "reactive", 0))
            out[(t, w)] = share(r) if r else None
    return out


def _seeds(kind: str) -> Sequence[int]:
    return SEEDS if kind in LEARNED else (0,)


def contrast(s: Dict[Key, dict], adm: Dict[Tuple[int, str], Optional[float]], topos: Sequence[int],
             arm: str, ref: str) -> dict:
    same_seed = arm in LEARNED and ref in LEARNED
    per, dropped = {}, {}
    for t in topos:
        ds, miss = [], []
        for w in WINDOWS:
            sh = adm[(t, w)]
            if sh is None or sh > SHARE_MAX:
                continue
            for sd in _seeds(arm):
                rs = sd if same_seed else 0
                a, b = s.get((t, w, arm, sd)), s.get((t, w, ref, rs))
                if a is None or b is None:
                    miss.append(f"{w}/s{sd}")
                    continue
                ds.append(F._pct(F._el(a), F._el(b)))
        if miss:
            dropped[str(t)] = miss
        if ds:
            per[t] = median(ds)
    out = {"arm": arm, "ref": ref, "dropped": dropped, "n_topologies": len(per)}
    if len(per) < MIN_TOPOLOGIES:
        return dict(out, label="DESIGN-SHORT")
    xs = [per[t] for t in sorted(per)]
    p = float(wilcoxon(xs, method="exact").pvalue) if any(xs) else 1.0
    med = median(xs)
    out.update(median_pct=med, p=p, faster=sum(x < 0 for x in xs), per_topology={str(t): per[t] for t in sorted(per)},
               label=label(med, p))
    return out


def label(med: float, p: float) -> str:
    if p < ALPHA and med <= -BAR:
        return "CONFIRMED"
    if p < ALPHA and med < 0:
        return "DIRECTION-ONLY"
    if p < ALPHA and med > 0:
        return "REF-FASTER"
    return "NOT-SEPARATED"


def read(gate: str, capacity: str, topos: Sequence[int]) -> dict:
    s = load(gate)
    fails = failures(gate)
    adm = admissible(s, topos)
    res = {"admissible_cells": sorted(f"{t}/{w}" for (t, w), v in adm.items() if v is not None and v <= SHARE_MAX),
           "reactive_share": {f"{t}/{w}": v for (t, w), v in adm.items()}}
    cap = load(capacity)
    win, wtopos = WITNESS
    wit = {}
    for t in wtopos:
        k = (t, win, PRIMARY, 1)
        a, b = s.get(k), cap.get(k)
        wit[str(t)] = {"rerun": F._el(a) if a else None, "capacity": F._el(b) if b else None,
                       "equal": a is not None and b is not None and F._el(a) == F._el(b)}
    res["W"] = {"cells": wit, "pass": all(v["equal"] for v in wit.values())}
    res["G1"] = contrast(s, adm, topos, PRIMARY, "cd")
    res["G2"] = contrast(s, adm, topos, PRIMARY, "selfpredict")
    res["G3"] = {r: contrast(s, adm, topos, PRIMARY, r) for r in ("reactive", "random", "batched", "decima")}
    res["G4"] = contrast(s, adm, topos, "gnnedge0", "mpoff")
    res["reported"] = {f"{a}_vs_{r}": contrast(s, adm, topos, a, r)
                       for a in ("xs1load", "gnnedge0", "mpoff") for r in RULES}
    res["reported"].update({f"{r}_vs_reactive": contrast(s, adm, topos, r, "reactive") for r in RULES if r != "reactive"})
    res["failures"] = {k: sorted(n for kk, n in fails.items() if kk[2] == k and kk[1] in WINDOWS)
                       for k in RULES + LEARNED}
    res["primary_failures_on_admissible"] = sorted(
        n for kk, n in fails.items()
        if kk[2] == PRIMARY and kk[1] in WINDOWS and (adm.get((kk[0], kk[1])) or 9.9) <= SHARE_MAX)
    res["reliability_flag"] = len(res["primary_failures_on_admissible"]) > 1
    return res


def main(argv: List[str] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", required=True)
    ap.add_argument("--capacity", required=True)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    topos = json.load(open(a.selection))["topologies"]
    res = read(a.gate, a.capacity, topos)
    for key in ("G1", "G2", "G4"):
        c = res[key]
        print(f"{key} {c['arm']} vs {c['ref']}: {c.get('median_pct', float('nan')):+.2f} % p={c.get('p')} "
              f"faster {c.get('faster')}/{c['n_topologies']} -> {c['label']}", file=sys.stderr)
    for r, c in res["G3"].items():
        print(f"G3 vs {r}: {c.get('median_pct', float('nan')):+.2f} % p={c.get('p')} faster {c.get('faster')}/"
              f"{c['n_topologies']} -> {c['label']}", file=sys.stderr)
    print(f"W witness pass: {res['W']['pass']}  admissible cells {len(res['admissible_cells'])}/48  "
          f"reliability flag {res['reliability_flag']}", file=sys.stderr)
    if a.out:
        with open(a.out + ".partial", "w") as fh:
            json.dump(res, fh, indent=1)
        os.replace(a.out + ".partial", a.out)
    else:
        print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
