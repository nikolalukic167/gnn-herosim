#!/usr/bin/env python3
"""Which term of S carries the label-minus-S residual where the oracle beats the hand bar? Reads oracle_diag/<cell>/*.jsonl written with ORACLE_DIAG_TERMS=1
(src/policy/peer_greedy_network/scheduler.py `_pg_terms_rec`); no simulation.

Per oracle record the hand plan (s_plan) and the oracle plan (best) carry per-task terms [drain, stack, inflight, cold, exec, lat, in_wait, exch, hold, ext, S, stack_raw, service]
('stack' is weighted by the running configuration, i.e. 0 under ib0; stack_raw is not).  For a batch: d_label = best_label - s_label (< 0 where the oracle gains), d_term = sum over
tasks of term(best) - term(s_plan).  S mispricing = d_label - d_S.  Every term is reported on three time bins: t < 360 s (the deploy transient), 360 s <= t < the 5th percentile of
task arrival over the cell's oracle batches, and steady state.  Unit of analysis: the batch; 'big' = relative gain (s_label - best_label)/s_label > 10 %.

  oracle_terms_read.py <oracle dir> [--json OUT] [--types cnn]
"""
import argparse
import glob
import json
import os
import re
import statistics as st
from collections import Counter, defaultdict

import numpy as np

COLS = ["drain", "stack", "inflight", "cold", "exec", "lat", "in_wait", "exch", "hold", "ext", "S", "stack_raw", "service"]
GROUP = {"queue": ["drain", "inflight"], "stack_raw": ["stack_raw"], "cold": ["cold"], "exec": ["exec"], "lat": ["lat", "in_wait"], "exch": ["exch"]}
T0 = 360.0
BIG = 0.10


def cell_of(path):
    m = re.search(r"cc40s(\d+)__g0(\w+?)__", path)
    return int(m.group(1)), m.group(2)


def load(d):
    recs = []
    for f in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
        with open(f) as fh:
            recs += [json.loads(l) for l in fh if l.strip()]
    recs.sort(key=lambda r: r["t"])
    return recs


def p5_time(recs):
    cum, tot = 0, sum(r["n"] for r in recs)
    for r in recs:
        cum += r["n"]
        if cum >= 0.05 * tot:
            return r["t"]
    return recs[-1]["t"]


def plan_terms(r, plan):
    key = tuple(map(tuple, plan))
    for k, rows in r["terms"]:
        if tuple(map(tuple, k)) == key:
            return np.array(rows, float)           # tasks x columns
    return None


def batch_row(r, cols):
    hand, orc = plan_terms(r, r["s_plan"]), plan_terms(r, r["best"])
    if hand is None or orc is None:
        return None
    ci = {c: i for i, c in enumerate(r["terms_cols"])}
    d = {c: float(orc[:, ci[c]].sum() - hand[:, ci[c]].sum()) for c in r["terms_cols"]}
    dl = r["best_label"] - r["s_label"]
    types, couples = r["ctx"]["task_types"], r["ctx"]["couples"]
    s, b = [tuple(x) for x in r["s_plan"]], [tuple(x) for x in r["best"]]
    moved = []
    for i, (x, y) in enumerate(zip(s, b)):
        if x != y:
            kx, ky = f"{x[0]}:{x[1]}", f"{y[0]}:{y[1]}"
            moved.append({"type": types[i], "from": couples[kx][1], "to": couples[ky][1], "same_node": x[0] == y[0],
                          "d": {c: float(orc[i, ci[c]] - hand[i, ci[c]]) for c in ("drain", "stack_raw", "cold", "exec", "lat", "exch", "S")}})
    nn_h, nn_o = len({x[0] for x in s}), len({y[0] for y in b})
    return {"t": r["t"], "n": r["n"], "d_label": dl, "rel": -dl / r["s_label"] if r["s_label"] > 0 else 0.0, "d": d, "d_S": d["S"], "moved": moved,
            "nodes_used": nn_o - nn_h, "search": bool(r.get("search"))}


def summarize(rows):
    out = {"batches": len(rows)}
    if not rows:
        return out
    big = [r for r in rows if r["rel"] > BIG]
    out["improved"] = sum(1 for r in rows if r["d_label"] < -1e-9)
    out["big"] = len(big)
    for name, sel in (("improved", [r for r in rows if r["d_label"] < -1e-9]), ("big", big)):
        if not sel:
            continue
        dl = np.array([r["d_label"] for r in sel])
        dS = np.array([r["d_S"] for r in sel])
        cell = {"n": len(sel), "mean_d_label": float(dl.mean()), "mean_d_S": float(dS.mean()), "mean_residual": float((dl - dS).mean()),
                "S_prefers_hand_share": float((dS > 0).mean()),
                "unstack(more nodes)": sum(r["nodes_used"] > 0 for r in sel), "stack(fewer nodes)": sum(r["nodes_used"] < 0 for r in sel), "same node count": sum(r["nodes_used"] == 0 for r in sel)}
        g = {}
        for k, cs in GROUP.items():
            g[k] = float(np.mean([sum(r["d"][c] for c in cs) for r in sel]))
        cell["mean_d_term_hand_to_oracle"] = g
        X = np.array([[sum(r["d"][c] for c in cs) for cs in GROUP.values()] for r in sel])
        if len(sel) > len(GROUP) + 5:
            beta, *_ = np.linalg.lstsq(X, dl, rcond=None)
            res = dl - X @ beta
            cell["ols_label_on_terms"] = {k: float(b) for k, b in zip(GROUP, beta)}
            cell["ols_r2"] = float(1 - (res ** 2).sum() / max(1e-12, ((dl - dl.mean()) ** 2).sum()))
        cell["share_with_cold_down"] = float(np.mean([r["d"]["cold"] < -1e-6 for r in sel]))
        cell["share_with_stack_raw_down"] = float(np.mean([r["d"]["stack_raw"] < -1e-6 for r in sel]))
        mv = Counter((m["type"], m["from"], m["to"]) for r in sel for m in r["moved"])
        cell["moved_task_top"] = [[list(k), v] for k, v in mv.most_common(8)]
        out[name] = cell
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("oracle"); ap.add_argument("--json", default=None)
    a = ap.parse_args()
    cells = {}
    for d in sorted(glob.glob(os.path.join(a.oracle, "oracle_diag", "cc40s*__g0*__*"))):
        recs = load(d)
        if not recs or "terms" not in recs[0]:
            continue
        p5 = p5_time(recs)
        rows = []
        for r in recs:
            if "terms" not in r:
                continue
            b = batch_row(r, COLS)
            if b:
                b["bin"] = "t<360" if r["t"] < T0 else ("360<=t<p5" if r["t"] < p5 else "steady")
                rows.append(b)
        cells[cell_of(d)] = (p5, rows)
    res = {}
    for rung in ("moderate", "heavy"):
        allrows = [r for (t, g), (p5, rs) in cells.items() if g == rung for r in rs]
        res[rung] = {"p5_by_topology": {str(t): p5 for (t, g), (p5, _) in cells.items() if g == rung}}
        for bn in ("t<360", "360<=t<p5", "steady", "all"):
            sel = allrows if bn == "all" else [r for r in allrows if r["bin"] == bn]
            res[rung][bn] = summarize(sel)
            s = res[rung][bn]
            print(f"\n== {rung} {bn}: {s['batches']} batches, improved {s.get('improved')}, big(>10 %) {s.get('big')}")
            for name in ("improved", "big"):
                c = s.get(name)
                if not c:
                    continue
                print(f"  [{name} n={c['n']}] mean d_label {c['mean_d_label']:.3f} s, d_S {c['mean_d_S']:.3f}, residual {c['mean_residual']:.3f}; S prefers the hand plan in {100 * c['S_prefers_hand_share']:.0f} %; "
                      f"nodes: unstack {c['unstack(more nodes)']} stack {c['stack(fewer nodes)']} same {c['same node count']}")
                print("     mean d_term hand->oracle: " + ", ".join(f"{k} {v:+.3f}" for k, v in c["mean_d_term_hand_to_oracle"].items()))
                if "ols_label_on_terms" in c:
                    print("     OLS d_label on d_terms (1.0 = S prices it right): " + ", ".join(f"{k} {v:.2f}" for k, v in c["ols_label_on_terms"].items()) + f"  R2 {c['ols_r2']:.2f}")
                print(f"     cold lower in oracle plan {100 * c['share_with_cold_down']:.0f} %, raw stacking lower {100 * c['share_with_stack_raw_down']:.0f} %; moved tasks (type, from, to): {c['moved_task_top'][:5]}")
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
