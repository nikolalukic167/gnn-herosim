#!/usr/bin/env python3
"""x11_confirm_v1 -- the confirmation read of capacity_sweep_v1's x1.1 result (docs/lineages/x11_confirm_v1.md).

  x11_confirm_v1_read.py --capacity <capacity dir> --confirm <x11confirm dir> --selection selected.json [--out read.json]

Rung w0 x1.1, perturbed draws d1-d4 (the same draw for every arm). CD and reactive run once per draw (seed 0);
the learned arm xs1load_selfref runs seeds 1-4 per draw. Per topology: the median, over every (seed, draw) of the
seed set, of the paired % learned vs CD on the same draw; exact two-sided Wilcoxon over topologies. A topology
missing any required run is dropped by name.

  Q1 (the bar)   seeds 2-4 only -- data nobody had seen when this lineage was registered.
                 CONFIRMED (median <= -5 %, p < 0.05) / DIRECTION-ONLY (median < 0, p < 0.05) / NOT-CONFIRMED
  Q2 (reported)  seeds 1-4.
  Q3 (reported)  each seed alone; reactive's median queue share per topology (admissible when <= 0.80).
  W  (witness)   seed 1 on w0x11d1 rerun at this lineage's commit must equal capacity_sweep_v1's run to the digit.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from statistics import median
from typing import Dict, List, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cd_gap_v1_read as C  # noqa: E402
import fresh_topo_burst_v1_read as F  # noqa: E402

DRAWS = (1, 2, 3, 4)
LEARNED = "xs1load_selfref"
BAR = 5.0
ALPHA = 0.05


def _win(k: int) -> str:
    return f"w0x11d{k}"


def contrast(s: Dict, topos: Sequence[int], seeds: Sequence[int]) -> dict:
    per, dropped = {}, {}
    for t in topos:
        missing = [(_win(k), "cd", 0) for k in DRAWS if (t, _win(k), "cd", 0) not in s] + \
                  [(_win(k), LEARNED, sd) for k in DRAWS for sd in seeds if (t, _win(k), LEARNED, sd) not in s]
        if missing:
            dropped[str(t)] = missing
            continue
        per[t] = median(F._pct(F._el(s[(t, _win(k), LEARNED, sd)]), F._el(s[(t, _win(k), "cd", 0)]))
                        for k in DRAWS for sd in seeds)
    out = {"seeds": list(seeds), "dropped": dropped}
    if len(per) < F.MIN_TOPOLOGIES:
        return dict(out, verdict="DESIGN-SHORT")
    out["read"] = F._read(per, "LEARNED", "CD")
    return out


def label(c: dict) -> str:
    r = c.get("read")
    if not r:
        return c.get("verdict", "NO-READ")
    if r["p"] < ALPHA and r["median_pct"] <= -BAR:
        return "CONFIRMED"
    if r["p"] < ALPHA and r["median_pct"] < 0:
        return "DIRECTION-ONLY"
    return "NOT-CONFIRMED"


def main(argv: List[str] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capacity", required=True, help="capacity_sweep_v1 gate dir (cd, reactive, seed 1)")
    ap.add_argument("--confirm", required=True, help="this lineage's gate dir (seeds 2-4, witness)")
    ap.add_argument("--selection", required=True)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    topos = json.load(open(a.selection))["topologies"]
    old, new = C._load([a.capacity]), C._load([a.confirm])
    s = dict(old)
    s.update({k: v for k, v in new.items() if k[3] != 1})

    witness = {}
    for k, r in new.items():
        if k[2] == LEARNED and k[3] == 1:
            ref = old.get(k)
            witness[f"{k[0]}/{k[1]}"] = {"rerun": F._el(r), "capacity": F._el(ref) if ref else None,
                                         "equal": ref is not None and F._el(r) == F._el(ref)}
    res = {"W": {"cells": witness, "pass": bool(witness) and all(v["equal"] for v in witness.values())},
           "Q1": contrast(s, topos, (2, 3, 4)), "Q2": contrast(s, topos, (1, 2, 3, 4)),
           "Q3": {f"seed{sd}": contrast(s, topos, (sd,)) for sd in (1, 2, 3, 4)}}
    res["Q1_label"] = label(res["Q1"])
    res["Q2_label"] = label(res["Q2"])
    shares = {}
    for t in topos:
        v = [float(s[(t, _win(k), "reactive", 0)]["averageQueueTime"]) /
             float(s[(t, _win(k), "reactive", 0)]["averageElapsedTime"])
             for k in DRAWS if (t, _win(k), "reactive", 0) in s]
        if v:
            shares[str(t)] = median(v)
    res["reactive_share"] = shares
    res["admissible"] = sorted(t for t, v in shares.items() if v <= 0.80)
    print(json.dumps(res, indent=1))
    for key in ("Q1", "Q2"):
        r = res[key].get("read") or {}
        print(f"{key} seeds {res[key]['seeds']}: median {r.get('median_pct', float('nan')):+.2f} % p={r.get('p')} "
              f"faster {r.get('a_faster')}/{r.get('n')} dropped {sorted(res[key]['dropped'])} -> {res[key + '_label']}",
              file=sys.stderr)
    for sd, c in res["Q3"].items():
        r = c.get("read") or {}
        print(f"Q3 {sd}: median {r.get('median_pct', float('nan')):+.2f} % p={r.get('p')} faster {r.get('a_faster')}/{r.get('n')}",
              file=sys.stderr)
    print(f"W witness pass: {res['W']['pass']}  admissible {len(res['admissible'])}/{len(shares)}", file=sys.stderr)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
