#!/usr/bin/env python3
"""Reader for the information-injection test (see queue_gap_inject_eval.py).

  queue_gap_inject_read.py <dir of <arm>_s<seed>.json> [--out read.json]

Per checkpoint the (lam_exch, lam_load) pair is chosen on the even-indexed half of VAL and the mean regret is reported on
the odd-indexed half. Reported per arm, averaged over the four seeds: no injection, best exchange only, best load only,
best of both.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from statistics import mean

ARMS = ("rawgnn", "rawmlp", "psignn")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--out")
    ap.add_argument("--stage", type=int, choices=(1, 2), default=1)
    a = ap.parse_args()
    names = ("exchange", "in-batch load") if a.stage == 1 else ("lookahead mass", "backlog seconds")
    suffix = "" if a.stage == 1 else "_st2"
    out = {}
    for arm in ARMS:
        rows = {"none": [], "exch": [], "load": [], "both": []}
        picks = {"exch": [], "load": [], "both": []}
        files = sorted(glob.glob(os.path.join(a.dir, f"{arm}_s?{suffix}.json")))
        if len(files) != 4:
            raise SystemExit(f"FAIL LOUD: {arm} has {len(files)} result files, expected 4")
        for f in files:
            r = json.load(open(f))
            if r.get("stage", 1) != a.stage:
                raise SystemExit(f"FAIL LOUD: {f} is stage {r.get('stage', 1)}, reading stage {a.stage}")
            reg = {tuple(float(x) for x in k.split("|")): v for k, v in r["regret"].items()}
            ev = lambda k, h: mean(reg[k][h::2])  # noqa: E731  (h=0 even half, h=1 odd half)
            rows["none"].append(ev((0.0, 0.0), 1))
            for name, ok in (("exch", lambda k: k[1] == 0.0), ("load", lambda k: k[0] == 0.0), ("both", lambda k: True)):
                best = min((k for k in reg if ok(k)), key=lambda k: ev(k, 0))
                rows[name].append(ev(best, 1))
                picks[name].append(best)
        out[arm] = {k: mean(v) for k, v in rows.items()}
        out[arm]["picks"] = {k: [list(p) for p in v] for k, v in picks.items()}
        out[arm]["per_seed"] = rows
    print("held-out half of VAL, mean regret (s), averaged over 4 seeds (lambda tuned on the other half):")
    print(f"stage {a.stage}: the two terms are {names[0]} and {names[1]}"
          + ("" if a.stage == 1 else " (exchange 0.3 and in-batch load 0.1 are always on; 'no injection' = those two only)"))
    print(f"{'arm':8s} {'no injection':>13s} {'+ ' + names[0]:>16s} {'+ ' + names[1]:>17s} {'+ both':>8s}   picked per seed, both")
    for arm in ARMS:
        r = out[arm]
        print(f"{arm:8s} {r['none']:13.2f} {r['exch']:16.2f} {r['load']:17.2f} {r['both']:8.2f}   {r['picks']['both']}")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
