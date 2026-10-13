#!/usr/bin/env python3
"""warm_surge_s0_v1 reads added after the 0b ruling (docs/lineages/warm_surge_s0_v1.md, "Rulings after 0b"). Descriptive unless said otherwise.

  collapse  --gate0b D --gate0a D --k1cd D [--uncapped D] --topos ... [--table out.md] [--out out.json]
      Collapse count per arm and condition (rule below), the per-cell table (cell, arm, post-switch mean, end - last arrival) for the heavy
      ×2/×3 classical arms and every collapsed learned cell, and the GNN lines against ALL THREE hand references (ib0, exactS, cd): median
      paired % of post-switch mean and wins, per condition. Heavy-rung medians against ib0 are collapse artefacts wherever ib0 collapses.
      COLLAPSE: post-switch mean > 3x the same cell's cd mean. TAIL FLAG (separate count): end - last arrival > 120 s. On the 0b data the two
      separate cleanly by ratio (non-collapsed max 2.4x, collapsed min 8.1x) but not by tail (non-collapsed max 100 s, one non-collapsed cell at 142 s
      with a normal mean), so the tail is reported beside the collapse count, not inside it.
  alpha     --gate0b D --gate0a D --extra D  (0c)
      cd_exactS at batch alpha 0 (cd_exactS_ib0, from gate_0b), 0.25, 0.5 (extra), 1.0 (cd_exactS, from gate_0b) on the heavy ×2 collapse cells
      19422, 19425, 19430 and the control 19420: post-switch mean, tail, collapse flag; verdict per the signed prediction and kill rule.
"""
import argparse
import json
import os
import statistics as st
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warm_surge_0b_read as B  # noqa: E402

T_S = 1200.0
COLLAPSE_X, COLLAPSE_TAIL = 3.0, 120.0
COLLAPSE_CELLS, CONTROL = (19422, 19425, 19430), 19420


def post_mean(rows):
    return B.window_mean(rows, T_S)


def tail_of(summary):
    return float(summary["endTime"]) - float(summary["arrival_end"]["last_arrival_s"])


def is_collapse(mean, cd_mean, tail=None):
    return bool(cd_mean and mean > COLLAPSE_X * cd_mean)


def tail_flag(tail):
    return tail > COLLAPSE_TAIL


def load_cell(path):
    s = json.load(open(path + ".summary.json"))
    return post_mean(np.load(path + ".tasks.npy")), tail_of(s)


def cell_path(c, topo, rung, k, arm, seed, g_override=None):
    if arm == "cd":
        return B.stem(c.k1 if k == 1 else c.a, topo, rung, k, arm, 0)
    return B.stem(g_override or c.g, topo, rung, k, arm, seed)


def collapse(a):
    topos = [int(x) for x in a.topos.split(",")]
    c = B.Cells(a.gate0b, a.gate0a, a.k1cd, T_S)
    sets = {"g4": (c.g, B.LEARNED)}
    if a.uncapped:
        sets["uncapped"] = (a.uncapped, B.LEARNED)
    out = {"counts": {}, "refs": {}, "table": []}
    for rung in B.RUNGS:
        for k in B.KS:
            key = f"{rung}k{k}"
            cdm = {t: load_cell(cell_path(c, t, rung, k, "cd", 0)) for t in topos}
            arms = [(B.IB0, 0, c.g), (B.EX, 0, c.g), ("cd", 0, None)]
            for name, (gdir, learned) in sets.items():
                if name == "uncapped" and k == 1 or (name == "uncapped" and rung == "heavy"):
                    continue
                arms += [(f"{ar}[{name}]", s, gdir) for ar in learned for s in B.SEEDS]
            for arm, seed, gdir in arms:
                base = arm.split("[")[0]
                n = ncell = nt = 0
                for t in topos:
                    p = cell_path(c, t, rung, k, base, seed, gdir)
                    if not os.path.exists(p + ".summary.json"):
                        continue
                    m, tl = load_cell(p)
                    flag = is_collapse(m, cdm[t][0], tl)
                    ncell += 1
                    n += flag
                    nt += tail_flag(tl)
                    if rung == "heavy" and k >= 2 and (base in (B.IB0, B.EX, "cd") or flag):
                        out["table"].append((key, t, f"{arm}_s{seed}", m, tl, flag))
                out["counts"][f"{key}|{arm}_s{seed}"] = {"collapsed": n, "tail_flag": nt, "cells": ncell}
            for gname, (gdir, learned) in sets.items():
                if gname == "uncapped" and (rung == "heavy" or k == 1):
                    continue
                for ar in learned:
                    for ref in (B.IB0, B.EX, "cd"):
                        ps = []
                        for t in topos:
                            vals = []
                            for s in B.SEEDS:
                                pa = cell_path(c, t, rung, k, ar, s, gdir)
                                if os.path.exists(pa + ".tasks.npy"):
                                    vals.append(post_mean(np.load(pa + ".tasks.npy")))
                            pr = cell_path(c, t, rung, k, ref, 0)
                            if len(vals) == len(B.SEEDS) and os.path.exists(pr + ".tasks.npy"):
                                ps.append(B.pct(sum(vals) / len(vals), post_mean(np.load(pr + ".tasks.npy"))))
                        out["refs"][f"{key}|{gname}|{ar}_vs_{ref}"] = B.summarize(ps)
    return out


def print_collapse(o):
    print("-- collapse counts (cells collapsed / cells), COLLAPSE = post-switch mean > 3x the cell's cd mean; tail flag = end - last arrival > 120 s")
    for k, v in o["counts"].items():
        print(f"  {k:60s} {v['collapsed']}/{v['cells']}  tail>120s: {v['tail_flag']}")
    print("-- GNN vs each hand reference: median paired % of post-switch mean, wins (GNN faster). Heavy vs ib0 are collapse artefacts where ib0 collapses")
    for k, v in o["refs"].items():
        if v["median"] is not None:
            print(f"  {k:70s} median {v['median']:+7.2f} % wins {v['wins']}/{v['n']}")


def write_table(o, path):
    rows = ["| condition | topology | arm | post-switch mean (s) | end - last arrival (s) | collapse |", "|---|---|---|---|---|---|"]
    for key, t, arm, m, tl, f in sorted(o["table"]):
        rows.append(f"| {key} | {t} | {arm} | {m:.2f} | {tl:.0f} | {'COLLAPSE' if f else ''} |")
    open(path, "w").write("\n".join(rows) + "\n")


def alpha(a):
    c = B.Cells(a.gate0b, a.gate0a, a.k1cd, T_S)
    out, verdict = {}, {}
    for t in (*COLLAPSE_CELLS, CONTROL):
        cdm, _ = load_cell(cell_path(c, t, "heavy", 2, "cd", 0))
        row = {}
        for al, arm, d in ((0.0, B.IB0, a.gate0b), (0.25, "cd_exactS_a025", a.extra), (0.5, "cd_exactS_a05", a.extra), (1.0, B.EX, a.gate0b)):
            p = B.stem(d, t, "heavy", 2, arm, 0)
            if not os.path.exists(p + ".summary.json"):
                row[al] = None
                continue
            m, tl = load_cell(p)
            row[al] = {"mean": m, "tail": tl, "collapse": is_collapse(m, cdm, tl)}
        out[t] = row
    ok = [all(out[t][al] is not None for al in (0.0, 0.25, 0.5, 1.0)) for t in out]
    if not all(ok):
        return {"cells": out, "verdict": "INCOMPLETE"}
    gone = {t: any(not out[t][al]["collapse"] for al in (0.25, 0.5)) for t in COLLAPSE_CELLS}
    still = sum(1 for t in COLLAPSE_CELLS if all(out[t][al]["collapse"] for al in (0.25, 0.5)))
    ctrl_worse = {al: B.pct(out[CONTROL][al]["mean"], out[CONTROL][0.0]["mean"]) for al in (0.25, 0.5, 1.0)}
    both_still = {t: all(out[t][al]["collapse"] for al in (0.25, 0.5)) for t in COLLAPSE_CELLS}
    # kill rule: alpha in {0.25, 0.5} still collapses on >= 2/3 cells -> BRITTLE-CAUSE-UNKNOWN (read per alpha)
    per_alpha_collapsed = {al: sum(out[t][al]["collapse"] for t in COLLAPSE_CELLS) for al in (0.25, 0.5)}
    killed = all(v >= 2 for v in per_alpha_collapsed.values())
    verdict = {"collapse_disappears_on_all_three_at_some_alpha_lt_1": all(gone.values()), "per_cell_gone": gone,
               "control_pct_vs_alpha0": ctrl_worse, "control_not_worse_gt_1pp": all(v is not None and v <= 1.0 for v in (ctrl_worse[0.25], ctrl_worse[0.5])),
               "collapsed_cells_per_alpha": per_alpha_collapsed, "KILL_BRITTLE_CAUSE_UNKNOWN": killed}
    return {"cells": out, "verdict": verdict}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for n in ("collapse", "alpha", "uncapped"):
        p = sub.add_parser(n)
        for x in ("--gate0b", "--gate0a", "--k1cd"):
            p.add_argument(x, required=True)
        p.add_argument("--topos", default=",".join(str(t) for t in range(19420, 19432)))
        p.add_argument("--out")
        if n == "uncapped":
            p.add_argument("--uncapped", required=True)
        elif n == "collapse":
            p.add_argument("--uncapped")
            p.add_argument("--table")
        elif n == "alpha":
            p.add_argument("--extra", required=True)
    a = ap.parse_args()
    o = {"collapse": collapse, "alpha": alpha, "uncapped": uncapped}[a.cmd](a)
    if a.cmd == "uncapped":
        for k, r in o.items():
            f = lambda s: "-" if s["median"] is None else f"{s['median']:+6.2f} {s['wins']}/{s['n']}"
            line = f"{k:62s} k1 {f(r['k1'])} | k2 {f(r['k2'])} Δ {f(r['delta_k2'])} moved={r['delta_k2']['moved']} | k3 {f(r['k3'])} Δ {f(r['delta_k3'])} moved={r['delta_k3']['moved']} | collapse {r['collapse']}"
            print(line)
    elif a.cmd == "collapse":
        print_collapse(o)
        if a.table:
            write_table(o, a.table)
    else:
        for t, row in o["cells"].items():
            print(t, {al: (None if v is None else (round(v["mean"], 2), round(v["tail"]), "COLLAPSE" if v["collapse"] else "ok")) for al, v in row.items()})
        print(json.dumps(o["verdict"], indent=1, default=str))
    if a.out:
        json.dump(o, open(a.out, "w"), indent=1, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())


def uncapped(a):
    """moderate ×1/×2/×3: the uncapped-slate learned arms (and the g4 arms beside them) against ib0, exactS and cd; Δ(k) = pct(k) - pct(1);
    registered secondary rule: moved = |median Δ| > 1 pp and >= 9 of the topologies sign-consistent. Collapse count per arm beside each line."""
    topos = [int(x) for x in a.topos.split(",")]
    c = B.Cells(a.gate0b, a.gate0a, a.k1cd, T_S)
    out = {}
    for gname, gdir in (("g4", a.gate0b), ("uncapped", a.uncapped)):
        for ar in B.LEARNED:
            for ref in (B.IB0, B.EX, "cd"):
                per = {}
                for k in B.KS:
                    ps = {}
                    for t in topos:
                        vals = []
                        for s in B.SEEDS:
                            p = cell_path(c, t, "moderate", k, ar, s, gdir)
                            if os.path.exists(p + ".tasks.npy"):
                                vals.append(post_mean(np.load(p + ".tasks.npy")))
                        pr = cell_path(c, t, "moderate", k, ref, 0)
                        ps[t] = B.pct(sum(vals) / len(vals), post_mean(np.load(pr + ".tasks.npy"))) if len(vals) == len(B.SEEDS) else None
                    per[k] = ps
                row = {}
                for k in B.KS:
                    s = B.summarize(list(per[k].values()))
                    row[f"k{k}"] = s
                    if k != 1:
                        d = [None if per[k][t] is None or per[1][t] is None else per[k][t] - per[1][t] for t in topos]
                        sd = B.summarize(d)
                        sd["moved"] = B.moved(sd, len(topos))
                        row[f"delta_k{k}"] = sd
                n_col = {}
                for k in B.KS:
                    n = m = 0
                    for t in topos:
                        cdm, _ = load_cell(cell_path(c, t, "moderate", k, "cd", 0))
                        for s in B.SEEDS:
                            p = cell_path(c, t, "moderate", k, ar, s, gdir)
                            if os.path.exists(p + ".summary.json"):
                                mm, tl = load_cell(p)
                                m += 1
                                n += is_collapse(mm, cdm)
                    n_col[f"k{k}"] = f"{n}/{m}"
                row["collapse"] = n_col
                out[f"{gname}|{ar}_vs_{ref}"] = row
    return out
