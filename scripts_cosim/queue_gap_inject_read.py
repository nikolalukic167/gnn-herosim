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
    a = ap.parse_args()
    out = {}
    for arm in ARMS:
        rows = {"none": [], "exch": [], "load": [], "both": []}
        picks = {"exch": [], "load": [], "both": []}
        files = sorted(glob.glob(os.path.join(a.dir, f"{arm}_s*.json")))
        if len(files) != 4:
            raise SystemExit(f"FAIL LOUD: {arm} has {len(files)} result files, expected 4")
        for f in files:
            r = json.load(open(f))
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
    print(f"{'arm':8s} {'no injection':>13s} {'+ exchange':>11s} {'+ in-batch load':>16s} {'+ both':>8s}   picked (exch, load) per seed, both")
    for arm in ARMS:
        r = out[arm]
        print(f"{arm:8s} {r['none']:13.2f} {r['exch']:11.2f} {r['load']:16.2f} {r['both']:8.2f}   {r['picks']['both']}")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
