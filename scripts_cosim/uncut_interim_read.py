#!/usr/bin/env python3
"""uncut_groups_s0_v1 INTERIM read (amendments 70022963 / 190e5771): descriptive only, labelled INTERIM-DESCRIPTIVE. No WIN/LOSS label,
no Holm p, never quoted as a result, never reused by the final read.

  uncut_interim_read.py --tools <checkout at 557d84bf> --cells <interim dir> --arm ra_twin_eng_arm1b_nosplit --seeds 1,2
                        --bar /share/nikola.lukic/uncut_groups/arm1_k1_bar_d6ae87c2 [--out x.json]

The cell statistic is the frozen reader's (uncut_arm_gate_read.contrast at 557d84bf, imported, not copied): per topology the median over
windows of 100 (arm - bar) / bar, the arm's cell mean the mean over its seeds. Reported per rung x {whole, after360}: median and mean over
topologies, n ahead (topologies with a negative value) of n, and the same per seed alone. Kills: the arm's .failed.json cells.
The arm's cells must all carry kind <arm>_k100, the bar is cd_ib0_k100 (the 869828 set), and only the bar's topologies are read.
"""
import argparse
import json
import os
import statistics as st
import sys

LABEL = "INTERIM-DESCRIPTIVE"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tools", required=True)
    ap.add_argument("--cells", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seeds", default="1,2")
    ap.add_argument("--bar", required=True)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    sys.path.insert(0, os.path.join(a.tools, "scripts_cosim"))
    import uncut_arm_gate_read as R

    seeds = [int(x) for x in a.seeds.split(",")]
    cells, failed = R.load(a.cells + "," + a.bar, "_k100")
    bar_topos = sorted({k[2] for k in cells if k[0] == "cd_ib0"})
    foreign = sorted({k[2] for k in list(cells) + list(failed) if k[0] == a.arm} - set(bar_topos))
    if not bar_topos or foreign:
        raise SystemExit(f"FAIL LOUD: bar topologies {bar_topos}; arm cells outside them {foreign}")
    out = {"label": LABEL, "arm": a.arm, "seeds": seeds, "cells": a.cells, "bar": a.bar, "tools": a.tools,
           "n_arm_cells": sum(1 for k in cells if k[0] == a.arm),
           "kills": sorted("__".join(map(str, k)) for k in failed if k[0] == a.arm), "rungs": {}}
    for r in R.RUNGS:
        for stat in R.STATS:
            row = {}
            for tag, sd in [("seeds_pooled", seeds)] + [(f"seed{s}", [s]) for s in seeds]:
                c = R.contrast(cells, a.arm, sd, "cd_ib0", [0], r, bar_topos, stat)
                vals = list(c["per_topology"].values())
                row[tag] = {"n_topologies": c["n_topologies"], "n_cells": c["n_cells"], "median_pct": c["median_pct"],
                            "mean_pct": st.mean(vals) if vals else None, "n_ahead": c["faster"], "per_topology": c["per_topology"]}
            out["rungs"][f"{r}/{stat}"] = row
    txt = json.dumps(out, indent=1)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(txt)
    print(f"[{LABEL}] {a.arm} vs cd_ib0_k100 (869828); kills {len(out['kills'])}")
    for k, row in out["rungs"].items():
        p = row["seeds_pooled"]
        print(f"[{LABEL}] {k}: median {p['median_pct']:+.2f} %  mean {p['mean_pct']:+.2f} %  ahead {p['n_ahead']}/{p['n_topologies']}  cells {p['n_cells']}"
              if p["median_pct"] is not None else f"[{LABEL}] {k}: no paired cells")
    return out


if __name__ == "__main__":
    main()
