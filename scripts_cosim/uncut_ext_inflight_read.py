#!/usr/bin/env python3
"""Seed extension, DESCRIPTIVE file-only read of new-seed cells against the in-flight hand bar (cd_ib0_inflightfull_k100). No label, no Holm, no family.

The registered read of an arm's new-seed cells is uncut_arm_gate_read.py at the arm's own pin against cd_ib0. The in-flight bar was adopted after the extension was
registered and its 12-topology cells sit at G8 a459dd98 (comb/cells_bar_g8; byte-identical to the 634cbd8d cells, comb/bar_identity.json), a pin chain that does not contain
the frozen-w pin of Arm 1b (W2b 59798c41). The reader's pin rule therefore cannot read the two together, and this tool does not pretend to: it applies the reader's statistic
(cell = mean over the seeds; per topology the median over the windows of the paired %; median over topologies; wins = topologies with a negative %; exact two-sided Wilcoxon,
unadjusted) to the cells as files, for the frozen charge (_k100) and the co-tenancy charge (_kr), on both rungs and both statistics.

  uncut_ext_inflight_read.py --gate <arm cell dir>[,..] --bar-gate <bar cell dir>[,..] --primary ra_gnn_eng_arm1b_nosplit --seeds 11,..,18 --topos 19537,..,19548 --out read.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import uncut_arm_gate_read as G  # noqa: E402

BAR_KIND = "cd_ib0_inflightfull"


def read(arm_cells_by_charge, bar_cells, primary, seeds, topos):
    out = {}
    for charge, cells in arm_cells_by_charge.items():
        both = {**bar_cells, **cells}
        rows = [G.contrast(both, primary, tuple(seeds), BAR_KIND, (0,), r, topos, s) for r in G.RUNGS for s in G.STATS]
        out[charge] = [{"rung": e["rung"], "stat": e["stat"], "n_topologies": e["n_topologies"], "n_cells": e["n_cells"], "median_pct": e["median_pct"],
                        "ahead": e["faster"], "of": e["n_topologies"], "p_unadjusted": e["p"], "per_topology": e["per_topology"]} for e in rows]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", required=True)
    ap.add_argument("--bar-gate", required=True)
    ap.add_argument("--primary", required=True)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--topos", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    seeds, topos = [int(x) for x in a.seeds.split(",")], [int(x) for x in a.topos.split(",")]
    k100, _ = G.load(a.gate, "_k100")
    kr, _ = G.load(a.gate, "_kr")
    bar, _ = G.load(a.bar_gate, "_k100")
    arm = {"_k100": {k: v for k, v in k100.items() if k[0] == a.primary}, "_kr": {k: v for k, v in kr.items() if k[0] == a.primary}}
    bar = {k: v for k, v in bar.items() if k[0] == BAR_KIND}
    stray = sorted({k[1] for c in arm.values() for k in c if k[1] not in seeds})
    if stray:
        raise SystemExit(f"FAIL LOUD: cells at seeds outside --seeds: {stray}")
    res = {"what": "DESCRIPTIVE file-only read against " + BAR_KIND + "_k100 (no label, no Holm)", "primary": a.primary, "seeds": seeds, "topos": topos,
           "gate": a.gate, "bar_gate": a.bar_gate, "n_cells": {c: len(v) for c, v in arm.items()}, "n_bar_cells": len(bar), "results": read(arm, bar, a.primary, seeds, topos)}
    json.dump(res, open(a.out, "w"), indent=1)
    for charge, rows in res["results"].items():
        for e in rows:
            print(f"{charge:5s} {e['rung']:8s} {e['stat']:8s} median {e['median_pct']:+.2f} %  ahead {e['ahead']}/{e['of']}  p {e['p_unadjusted']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
