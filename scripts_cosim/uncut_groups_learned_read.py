#!/usr/bin/env python3
"""uncut_groups_s0_v1 learned one-pass read (F1/F2 at k = 1), under the pre-data record 506de44b (rp/uncut-k1learned). Every line is F1-CUT-TRAINED.

  uncut_groups_learned_read.py --hand <gate_k1> --learned <gate_k1_learned> --topos t1,...,t12 [--out read.json]

Counts first (finished/failed per arm x rung of 48), then per arm: COLLAPSE (mean > 3x the same topology x window x rung's cd_k100 mean, whole-run and
after-360 s separately), TAIL FLAG (end - last arrival > 120 s), wall-infeasible (watchdog-projected failures). Then the decision shares ((queue wait + busy)
/ tasks / mean latency, median over cells, in PERCENT). Then, for each statistic (whole-run, after 360 s) and rung, the paired % of mean latency per
topology (median over the four windows), the median over topologies, sign count and an exact two-sided Wilcoxon (negative = first arm faster):
  F1 (Holm over 4): nosplit vs cd_ib0 x 2 rungs; the two cdxapply hypotheses are NOT RUN (p = 1)
  F2 (Holm over 4): nosplit vs twin-nosplit x 2 rungs; nosplit vs MLP same-input one-pass x 2 rungs NOT RUN (p = 1, no checkpoint)
Descriptive: nosplit and twin vs cd, cd_exactS_ib0, cd_expand, prune (paired cells only, n reported). WIN needs median <= -2 % and Holm p < .05 at both
rungs on BOTH statistics, the arm's collapse count <= the bar's, and the anchor/F3 rule; any F1 WIN is WIN-VS-CD-ONLY.
UNCUT-ATTRIBUTED needs per-task batch sizes, which the cells do not carry: printed as NOT MEASURABLE with the pgb counters beside it.
"""
import argparse
import json
import statistics as st
import sys

import uncut_groups_hand_read as H
import r1_attribution_v1_read as R

NS, TW, BAR = "ra_gnn_eng_nosplit", "ra_twin_eng_nosplit", "cd_ib0"
HAND_DESC = ("cd", "cd_exactS_ib0", "cd_expand", "cd_exactS_ib0_prune")
SUF = "_k100"
RUNGS = ("moderate", "heavy")
ALPHA, WIN_MEDIAN = 0.05, -2.0


def paired(cells, a, b, rung, topos, stat):
    out = {}
    for t in topos:
        v = []
        for w in H.WINDOWS:
            ca, cb = cells.get((a + SUF, t, w, rung)), cells.get((b + SUF, t, w, rung))
            if ca is None or cb is None:
                continue
            ma, mb = H.means(ca)[stat], H.means(cb)[stat]
            if ma is None or not mb:
                continue
            v.append(100.0 * (ma - mb) / mb)
        if v:
            out[t] = (st.median(v), len(v))
    return out


def contrast(cells, a, b, rung, topos, stat):
    per = paired(cells, a, b, rung, topos, stat)
    if not per:
        return {"a": a, "b": b, "rung": rung, "stat": stat, "n_topologies": 0, "median_pct": None, "p": None, "wins": None, "n_cells": 0}
    v = [x[0] for x in per.values()]
    return {"a": a, "b": b, "rung": rung, "stat": stat, "n_topologies": len(v), "n_cells": sum(x[1] for x in per.values()), "median_pct": st.median(v),
            "p": R.exact_wilcoxon(v), "wins": sum(1 for x in v if x < 0), "per_topology": {str(t): round(x[0], 2) for t, x in sorted(per.items())}}


def holm4(entries):
    """entries: 4 dicts with `p` (None = NOT RUN, entered as p = 1). Holm over 4; returns the adjusted p in place."""
    ps = {i: (1.0 if e["p"] is None else e["p"]) for i, e in enumerate(entries)}
    adj = R.holm({str(i): p for i, p in ps.items()})
    for i, e in enumerate(entries):
        e["p_holm"] = adj[str(i)]
        e["not_run"] = e["p"] is None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hand", required=True)
    ap.add_argument("--learned", required=True)
    ap.add_argument("--topos", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    topos = [int(x) for x in a.topos.split(",")]
    hc, hf = H.load_gate(a.hand)
    lc, lf = H.load_gate(a.learned)
    cells = {**hc, **lc}
    failed = {**hf, **lf}
    arms = (NS, TW, "cd", BAR) + tuple(x for x in HAND_DESC if x != "cd")
    # collapse is read against cd_k100 of the same topology x window x rung; the H helpers key the reference as ("cd" + suffix)
    res = {"label": "F1-CUT-TRAINED", "counts": H.counts(cells, failed, arms, topos, SUF), "health": H.health_table(cells, arms, topos, SUF),
           "wall_infeasible": H.wall_infeasible(failed, arms, topos, SUF), "decision_share_pct": {}, "split_batches": H.split_counts(cells, [x + SUF for x in arms], topos)}
    for arm in arms:
        for rung in RUNGS:
            r = H.headroom(cells, arm, SUF, topos, rung)
            res["decision_share_pct"][f"{arm}|{rung}"] = {k: (None if r[k]["median"] is None else 100 * r[k]["median"]) for k in H.STAT} | {"n": r["whole"]["n"]}
    res["F1"], res["F2"], res["descriptive"], res["flags"] = {}, {}, {}, {}
    for stat in H.STAT:
        f1 = [contrast(cells, NS, BAR, r, topos, stat) for r in RUNGS] + [{"a": "cdxapply", "b": BAR, "rung": r, "stat": stat, "p": None, "median_pct": None} for r in RUNGS]
        f2 = [contrast(cells, NS, TW, r, topos, stat) for r in RUNGS] + [{"a": NS, "b": "ra_mlp_same", "rung": r, "stat": stat, "p": None, "median_pct": None} for r in RUNGS]
        holm4(f1)
        holm4(f2)
        res["F1"][stat], res["F2"][stat] = f1, f2
        res["descriptive"][stat] = [contrast(cells, x, b, r, topos, stat) for x in (NS, TW) for b in ("cd",) + HAND_DESC[1:] for r in RUNGS]
    win = {}
    for r_i, rung in enumerate(RUNGS):
        ok = True
        for stat in H.STAT:
            e = res["F1"][stat][r_i]
            ok = ok and e["median_pct"] is not None and e["median_pct"] <= WIN_MEDIAN and e["p_holm"] < ALPHA
        win[rung] = ok
    col = lambda arm, rung: res["health"][f"{arm}|{rung}"]["collapse_any"]
    res["flags"] = {"F1_WIN_LINE_EACH_RUNG": win, "F1_WIN_LINE_BOTH_RUNGS": all(win.values()),
                    "collapse_nosplit_le_bar": {r: col(NS, r) <= col(BAR, r) for r in RUNGS},
                    "nosplit_share_smaller_than_bar_so_F3_required": {r: res["decision_share_pct"][f"{NS}|{r}"]["whole"] < res["decision_share_pct"][f"{BAR}|{r}"]["whole"] for r in RUNGS},
                    "UNCUT_ATTRIBUTED": "NOT MEASURABLE: cells carry no per-task batch size; see pgb counters",
                    "label_if_win": "WIN-VS-CD-ONLY"}
    pg = {}
    for arm in (NS, TW, BAR):
        for rung in RUNGS:
            tot = {k: 0 for k in ("whole", "split")}
            for t in topos:
                s = cells.get((arm + SUF, t, "g0", rung))
                if s is None:
                    continue
                c = s["schedulerCounters"]
                tot["whole"] += H._pgb(c, arm, "whole")
                tot["split"] += H._pgb(c, arm, "split")
            pg[f"{arm}|{rung}"] = tot | {"whole_share_gt10_g0": tot["whole"] / (tot["whole"] + tot["split"]) if tot["whole"] + tot["split"] else None}
    res["pgb_g0_tasks_in_groups_gt10"] = pg
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    sys.exit(main())
