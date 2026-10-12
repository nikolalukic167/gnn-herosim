#!/usr/bin/env python3
"""Where does the hand bar lose to the live oracle?  Reads <oracle dir>/oracle_diag/<cell>/*.jsonl (written by plan_oracle.diag_write); no simulation.

Per oracle batch the diag holds the hand bar's plan (s_plan, its replay label s_label), the oracle's plan (best, best_label) and, for within-cap batches,
the whole slate-product table [plan, S, label].  Over-cap batches carry the local-search result only.  Both oracle modes search the DECLARED slate (plus
the couples of s_plan): the oracle plan is inside the slate by construction, so slate membership cannot be read from this file -- `slate_gap` (the
brute-force tables of the uncapped corpus) answers that.  Cold involvement is a PROXY: a platform no applied plan of the cell had used before.

  oracle_loss_diag.py diag <oracle dir> [--json OUT]
  oracle_loss_diag.py slate_gap <rows.jsonl of surrogate_gap_screen run> [--json OUT]   (brute-force tables: optimum outside the declared top-5 slate product?)
"""
import argparse
import glob
import json
import os
import re
import statistics as st
from collections import defaultdict

GAIN_EPS = 1e-9


def cell_of(path):
    m = re.search(r"cc40s(\d+)__g0(\w+?)__", path)
    return int(m.group(1)), m.group(2)


def load_cell(d):
    recs = []
    for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        with open(f) as fh:
            recs += [json.loads(l) for l in fh if l.strip()]
    recs.sort(key=lambda r: r["t"])
    return recs


def classify(recs):
    """One row per oracle batch."""
    used, rows = set(), []
    for r in recs:
        s, b = [tuple(x) for x in r["s_plan"]], [tuple(x) for x in r["best"]]
        gain = r["s_label"] - r["best_label"]
        row = {"n": r["n"], "over_cap": bool(r.get("search")), "gain": gain, "rel": gain / r["s_label"] if r["s_label"] > 0 else 0.0,
               "changed": s != b, "moved_tasks": sum(1 for x, y in zip(s, b) if x != y), "s_label": r["s_label"],
               "oracle_new_platform": any(c not in used for c in b), "hand_new_platform": any(c not in used for c in s),
               "colocation_gain": len(set(n for n, _ in s)) - len(set(n for n, _ in b)), "plans": r.get("plans"), "replays": r.get("replays")}
        if "table" in r:
            tab = r["table"]
            order = sorted(range(len(tab)), key=lambda i: (tab[i][1], i))      # slate-product plans by S, the scorer's own order
            idx = {tuple(map(tuple, t[0])): i for i, t in enumerate(tab)}
            rank = {i: k + 1 for k, i in enumerate(order)}
            row["oracle_rank_by_S"] = rank[idx[tuple(b)]]
            row["hand_rank_by_S"] = rank[idx[tuple(s)]]
            labs = sorted(t[2] for t in tab)
            row["slate_plans"] = len(tab)
            row["label_rank_of_hand"] = 1 + sum(1 for l in labs if l < r["s_label"] - GAIN_EPS)
            row["top2_gap_rel"] = (labs[1] - labs[0]) / labs[0] if len(labs) > 1 and labs[0] > 0 else None
        used.update(b)
        rows.append(row)
    return rows


def summarize(rows):
    n = len(rows)
    out = {"batches": n}
    if not n:
        return out
    ch = [r for r in rows if r["changed"] and r["gain"] > GAIN_EPS]
    tot = sum(r["gain"] for r in rows)
    out.update(batches_improved=len(ch), tasks=sum(r["n"] for r in rows), gain_seconds=tot, label_seconds=sum(r["s_label"] for r in rows),
               gain_share_of_label=tot / sum(r["s_label"] for r in rows))
    out["rel_gain_improved"] = {"median": st.median(r["rel"] for r in ch) if ch else None, "mean": st.fmean(r["rel"] for r in ch) if ch else None}
    out["tasks_moved_per_improved"] = st.fmean(r["moved_tasks"] for r in ch) if ch else None
    out["colocation_more_nodes_freed"] = sum(1 for r in ch if r["colocation_gain"] > 0)
    out["colocation_fewer_nodes_freed"] = sum(1 for r in ch if r["colocation_gain"] < 0)
    out["oracle_uses_new_platform"] = {"improved": sum(r["oracle_new_platform"] for r in ch), "of": len(ch)}
    wt = [r for r in rows if "oracle_rank_by_S" in r and r["changed"] and r["gain"] > GAIN_EPS]
    if wt:
        rk = [r["oracle_rank_by_S"] for r in wt]
        out["oracle_rank_by_S_among_improved"] = {"n": len(rk), **{f"top{k}": sum(x <= k for x in rk) for k in (1, 2, 4, 8, 16, 32)}, "median": st.median(rk), "max": max(rk)}
        out["hand_rank_by_S_among_improved_median"] = st.median(r["hand_rank_by_S"] for r in wt)
        out["gain_reached_by_S_topk"] = {f"top{k}": sum(r["gain"] for r in wt if r["oracle_rank_by_S"] <= k) / sum(r["gain"] for r in wt) for k in (2, 4, 8, 16, 32)}
    return out


def diag(a):
    cells = {}
    for d in sorted(glob.glob(os.path.join(a.oracle, "oracle_diag", "cc40s*__g0*__*"))):
        cells[cell_of(d)] = classify(load_cell(d))
    res = {}
    for rung in ("moderate", "heavy"):
        rows = [r for (t, g), v in cells.items() if g == rung for r in v]
        by = {"all": summarize(rows), "within_cap": summarize([r for r in rows if not r["over_cap"]]), "over_cap": summarize([r for r in rows if r["over_cap"]])}
        by["by_n"] = {str(n): summarize([r for r in rows if r["n"] == n]) for n in sorted({r["n"] for r in rows}) if n <= 12}
        by["by_n"]["13+"] = summarize([r for r in rows if r["n"] > 12])
        by["oracle_new_platform"] = {"batches_with": summarize([r for r in rows if r["oracle_new_platform"]]), "batches_without": summarize([r for r in rows if not r["oracle_new_platform"]])}
        by["hand_new_platform"] = {"batches_with": summarize([r for r in rows if r["hand_new_platform"]]), "batches_without": summarize([r for r in rows if not r["hand_new_platform"]])}
        res[rung] = by
        print(f"\n== {rung}: {len(rows)} oracle batches in {sum(1 for (t, g) in cells if g == rung)} cells")
        def line(name, s):
            if s["batches"]:
                print(f"  {name:22s} batches {s['batches']:6d}  improved {s['batches_improved']:5d}  gain {s['gain_seconds']:10.1f} s = {100 * s['gain_share_of_label']:.2f} % of hand label seconds"
                      + (f"  | oracle rank by S among improved: n={s['oracle_rank_by_S_among_improved']['n']} top1 {s['oracle_rank_by_S_among_improved']['top1']} top4 {s['oracle_rank_by_S_among_improved']['top4']} top8 {s['oracle_rank_by_S_among_improved']['top8']} top32 {s['oracle_rank_by_S_among_improved']['top32']}"
                         if s.get("oracle_rank_by_S_among_improved") else ""))
        for k in ("all", "within_cap", "over_cap"):
            line(k, by[k])
        for k, v in by["by_n"].items():
            line(f"n={k}", v)
        for k, v in by["oracle_new_platform"].items():
            line(f"oracle new-plat {k[6:]}", v)
        for k, v in by["hand_new_platform"].items():
            line(f"hand new-plat {k[6:]}", v)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


def slate_gap(a):
    """Per state of a surrogate_gap_screen run: the full-table optimum against the optimum over the declared top-5 slate product (the oracle's search space).
    The hand pick's regret over the full table (arm ib0_s, k = 0) is split into the part inside the slate and the part only a wider slate reaches."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import numpy as np

    from scripts_cosim import surrogate_gap_screen as G
    from scripts_cosim.extend_corpus_slate import slate_view
    from src.placement.declared_slate import TOP_K, prune_candidates

    rows = [json.loads(l) for l in open(a.rows) if l.strip()]
    out = []
    for r in rows:
        if "error" in r:
            continue
        ds = Path(r["ds"])
        combos, rtt = G.load_table(ds)
        v = slate_view(ds)
        tids = [int(g) for g in v["wl"]["trace_task_ids"]]
        raw_by = {int(t["task_id"]): t for t in v["raw"]["tasks"]}
        keep = [{(int(c["node_id"]), int(c["platform_id"])) for c in prune_candidates(raw_by[g]["candidates"], TOP_K)} for g in tids]
        plats = v["raw"]["fidelity"]["platforms"]       # the sim charges the table cold start unless the platform's previous task has the task's type
        cold = [{(int(c["node_id"]), int(c["platform_id"])) for c in raw_by[g]["candidates"]
                 if (plats.get(c["queue_key"]) or {}).get("prev") != raw_by[g]["task_type"] and float(c.get("cold_start_time", 0.0) or 0.0) > 0} for g in tids]
        inside = np.array([all(c[i] in keep[i] for i in range(len(tids))) for c in combos])
        opt = float(rtt.min())
        first = int(np.argmin(rtt))
        so = float(rtt[inside].min()) if inside.any() else None
        hand = r["arms"]["ib0_s"]["0"]
        out.append({"ds": r["ds"], "rung": r["rung"], "n": r["n"], "source": r["source"], "plans": len(rtt), "slate_plans": int(inside.sum()),
                    "opt_in_slate": bool(rtt[inside].min() <= opt * (1 + G.TIE)) if inside.any() else False,
                    "slate_regret": None if so is None else (so - opt) / opt, "hand_regret_full": hand,
                    
                                        "opt_has_cold": any(combos[first][i] in cold[i] for i in range(len(tids))), "opt_multi_node": r["multi_node_opt"]})
    res = {}
    for rung in ("moderate", "heavy"):
        for src in ("heldout", "all"):
            for n in (0, 2, 3, 4, 5, 6):
                x = [o for o in out if o["rung"] == rung and (n == 0 or o["n"] == n) and (src == "all" or o["source"] == src)]
                if not x:
                    continue
                sr = [o["slate_regret"] if o["slate_regret"] is not None else float("inf") for o in x]
                cell = {"states": len(x), "opt_outside_slate": sum(not o["opt_in_slate"] for o in x),
                        "slate_regret_gt1pct": sum(v > 0.01 for v in sr), "slate_regret_gt3pct": sum(v > 0.03 for v in sr),
                        "slate_regret_mean": st.fmean(sr), "slate_regret_median": st.median(sr), "hand_regret_full_mean": st.fmean(o["hand_regret_full"] for o in x),
                        "slate_regret_p90": sorted(sr)[min(len(sr) - 1, int(0.9 * len(sr)))],
                        "opt_has_cold": sum(o["opt_has_cold"] for o in x),
                        "outside_and_cold": sum(o["opt_has_cold"] and not o["opt_in_slate"] for o in x),
                        "outside_and_multi_node": sum(o["opt_multi_node"] and not o["opt_in_slate"] for o in x)}
                res[f"{rung}|{src}|{n}"] = cell
                print(f"{rung:9s} {src:8s} n={n or '2-6':>4}: {cell['states']:4d} states | optimum outside the slate {cell['opt_outside_slate']:3d} | slate-opt regret >1 % {cell['slate_regret_gt1pct']:3d}, >3 % {cell['slate_regret_gt3pct']:3d}, "
                      f"mean {100 * cell['slate_regret_mean']:.2f} % p90 {100 * cell['slate_regret_p90']:.2f} % | hand (full-candidate ib0 S pick) regret {100 * cell['hand_regret_full_mean']:.2f} % | opt plan charges a cold start {cell['opt_has_cold']}, outside&cold {cell['outside_and_cold']}, outside&multi-node {cell['outside_and_multi_node']}")
    if a.json:
        json.dump({"cells": res, "states": out}, open(a.json, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("diag"); d.add_argument("oracle"); d.add_argument("--json", default=None); d.set_defaults(fn=diag)
    g = sub.add_parser("slate_gap"); g.add_argument("rows"); g.add_argument("--json", default=None); g.set_defaults(fn=slate_gap)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
