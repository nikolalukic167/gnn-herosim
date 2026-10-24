#!/usr/bin/env python3
"""live_link_features_v1 (b) re-encode arm on the sealed 12 (descriptive, 1 seed, 20 epochs): the (b) gnn against its GPU twin, the CD bar, the static e40 gnn
(seed 1 and pooled seeds 1-3) and the wide MLP. Uses the parent reader's pairing, windows, runaway rule and exact Wilcoxon unchanged (llf_confirm_read.Grid).

  llf_reenc_read.py --reenc DIR --final DIR --bar DIR --rulings confirm_rulings.json --out reenc_sealed12.json

Each DIR holds cell subdirs with cc40s<topology>__<window><rung>__<kind>_s<seed>.summary.json. Pins: reenc and bar 0ec7cdcb, static/wide (final) 02668ecf.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import llf_confirm_read as R  # noqa: E402

KINDS = {"re_gnn": "ra_gnn_eng_combfl_linkconv_nosplit_k100", "re_twin": "ra_twin_eng_combfl_link_gpu_nosplit_k100",
         "gnn": R.ARMS["gnn"], "wide": R.ARMS["wide"], "bar": R.BAR}
SRC = {"re_gnn": ("reenc", "0ec7cdcb", (1,)), "re_twin": ("reenc", "0ec7cdcb", (1,)), "gnn": ("final", "02668ecf", R.SEEDS),
       "wide": ("final", "02668ecf", R.SEEDS), "bar": ("bar", "0ec7cdcb", (0,))}
CONTRASTS = (("re_gnn", "re_twin", "(b) gnn vs its GPU twin", "seed1"),
             ("re_gnn", "bar", "(b) gnn vs CD bar", "seed1"),
             ("re_gnn", "gnn", "(b) gnn vs static e40 gnn, seed 1 only", "seed1"),
             ("re_gnn", "gnn", "(b) gnn vs static e40 gnn, pooled seeds 1-3", "pooled"),
             ("re_gnn", "wide", "(b) gnn vs wide MLP, seed 1 only", "seed1"),
             ("re_gnn", "wide", "(b) gnn vs wide MLP, pooled seeds 1-3 (extra)", "pooled"))


class ReGrid(R.Grid):
    def seeds(self, arm):
        return SRC[arm][2]


def load(dirs):
    want = {(KINDS[a], s, t, w, r) for a in KINDS for s in SRC[a][2] for t in R.IDS for w in R.WINDOWS for r in R.RUNGS}
    V, seen = {}, {}
    for a in KINDS:
        root, pin, _ = SRC[a]
        for f in sorted(glob.glob(os.path.join(dirs[root], "*", f"cc40s*__{KINDS[a]}_s*.summary.json"))):
            m = R.NAME.match(os.path.basename(f))
            if not m or m.group(4) != KINDS[a]:
                continue
            key = (m.group(4), int(m.group(5)), int(m.group(1)), m.group(2), m.group(3))
            if key in V:
                raise SystemExit(f"FAIL LOUD: {key} appears twice ({f})")
            if key not in want:
                raise SystemExit(f"FAIL LOUD: unexpected cell {key} ({f})")
            s = json.load(open(f))
            code = s.get("code") or {}
            if code.get("dirty") is not False or not str(code.get("commit", "")).startswith(pin) or s.get("servers") != 24:
                raise SystemExit(f"FAIL LOUD: {f} ran at {code.get('commit')} dirty={code.get('dirty')} servers={s.get('servers')}, not {pin} clean at 24 servers")
            a360 = ((s.get("steady_state") or {}).get("after_360s") or {}).get("mean")
            if s.get("averageElapsedTime") is None or a360 is None:
                raise SystemExit(f"FAIL LOUD: {f} has no averageElapsedTime / after_360s.mean")
            V[key] = {"whole": float(s["averageElapsedTime"]), "a360": float(a360)}
            seen.setdefault(a, set()).add(code["commit"])
    miss = sorted(want - set(V))
    if miss:
        raise SystemExit(f"FAIL LOUD: {len(miss)} of {len(want)} cells missing, first {miss[:3]}")
    return V, {a: sorted(c) for a, c in seen.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("reenc", "final", "bar", "rulings", "out"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"FAIL LOUD: {a.out} exists")
    rul = json.load(open(a.rulings))
    R.check_rulings_cd(rul)
    V, pins = load({"reenc": a.reenc, "final": a.final, "bar": a.bar})
    g = ReGrid(V, R.IDS, rul, KINDS)
    rows = {}
    for x, y, label, mode in CONTRASTS:
        for stat in R.STATS:
            for rung in R.RUNGS:
                for reading in ("all", "no_runaway"):
                    c = g.contrast(x, y, rung, stat, reading, seed=1 if mode == "seed1" else None)
                    rows[f"{label}|{stat}|{rung}|{reading}"] = {k: v for k, v in c.items() if k != "per_topology"}
    ra = g.runaways()
    counts = {}
    for r in ra:
        if r["arm"] in ("re_gnn", "re_twin") or r["arm"] in ("gnn", "wide", "bar"):
            k = f"{r['arm']}|{r['rung']}|{r['stat']}"
            counts[k] = counts.get(k, 0) + 1
    res = {"family": "llf_cd_confirm_v1 reviewer: (b) re-encode arm on the sealed 12 (descriptive; 1 seed, 20 epochs, best-val on the old-label metric, seen on the dev topologies)",
           "topologies": list(R.IDS), "pins": pins, "cells": len(V), "rulings": rul, "contrasts": rows, "runaway_cells": ra, "runaway_counts_by_arm_rung_stat": counts,
           "reader": "llf_confirm_read.Grid unchanged; seeds of the (b) arms = (1,)",
           "caveats": ["1 seed", "20 epochs", "best-val on the old-label metric", "already seen on the dev topologies", "each arm is charged its own measured decision time (cell env HEROSIM_DECISION_COST_MS_BY_BUCKET); the (b) gnn heavy charge is 8.29/6.90/10.74 ms (le10/11_50/gt50), the static e40 gnn 6.15/3.10/3.68"]}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1, default=str)
    fm = lambda x: "n/a" if x is None else f"{x:+.2f}"
    for stat in R.STATS:
        print(f"== {stat}")
        for x, y, label, mode in CONTRASTS:
            for rung in R.RUNGS:
                al, nr = rows[f"{label}|{stat}|{rung}|all"], rows[f"{label}|{stat}|{rung}|no_runaway"]
                print(f"{label:48s} {rung:8s} all {fm(al['median_pct'])}% {al['faster']}/{al['n']} p {al['p'] if al['p'] is None else round(al['p'], 4)} | no-runaway {fm(nr['median_pct'])}% {nr['faster']}/{nr['n']} p {nr['p'] if nr['p'] is None else round(nr['p'], 4)}")
    print("runaways:", json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
