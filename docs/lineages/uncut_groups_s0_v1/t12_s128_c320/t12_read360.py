#!/usr/bin/env python3
"""AFTER-360 s companion of t12_read.py (SCIENTIST audit addition). task = tasks with arrival >= 360; req = requests whose first member arrives >= 360. Same cells, same statistic. 128 x 320 test read (12 topologies 30264-30275, g0-g3, moderate/heavy; one pin b092fad2 for every arm). Written once.
Cells: bar cd_ib0_inflightfull_k100 s0; FL gnn ra_gnn_eng_combfl_nosplit_k100 s1-2; FL MLP ra_mlp_same_combfl_nosplit_k100 s1-2 -> 480; reactive s0 96 (admission only).
Statistic = Table I's: per topology the median over the 4 windows of the paired % to the comparator in the same cell (a learned cell = the mean over seeds of the per-seed statistic),
then the median over the 12 topologies, topologies faster, exact two-sided Wilcoxon (r1_attribution_v1_read.exact_wilcoxon), raw p. Metrics: task mean (summary averageElapsedTime, whole run),
request mean (S4's request_stats.py definition: request = workload peer_group, fan-out >= 2, completion = last member completion - first arrival; whole run). Seeds 1 and 2 are also read alone.
Admission: reactive moderate-rung queue_share <= 0.80, unknown is not a pass; nothing dropped here."""
import json, os, sys, statistics as st
import numpy as np
sys.path.insert(0, os.path.expanduser("~/gnn-herosim-wt/s5_quick_b092fad2/scripts_cosim"))
from r1_attribution_v1_read import exact_wilcoxon
U = "/share/nikola.lukic/uncut_groups"; R = U + "/t12_s128_c320"; OUT = R + "/t12_read_after360.json"; PIN = "b092fad2"
assert not os.path.exists(OUT), "written once"
TOPOS = list(range(30264, 30276)); W = ("g0", "g1", "g2", "g3"); RUNGS = ("moderate", "heavy")
BAR, GNN, MLP = "cd_ib0_inflightfull_k100", "ra_gnn_eng_combfl_nosplit_k100", "ra_mlp_same_combfl_nosplit_k100"
SEEDS = {BAR: (0,), GNN: (1, 2), MLP: (1, 2)}
wl = {}
def workload(rung, w):
    if (rung, w) not in wl:
        ev = json.load(open(f"{R}/in/wf1_{rung}/wl/grounded_{w}_n50000.json"))["events"]
        g = np.array([e["peer_group"] for e in ev]); ts = np.array([e["timestamp"] for e in ev]); wl[(rung, w)] = (ts, g)
    return wl[(rung, w)]
def cell(kind, sd, t, w, rung):
    stem = f"{R}/cells_cells/cc40s{t}__{w}{rung}__{kind}_s{sd}"
    s = json.load(open(stem + ".summary.json"))
    assert s["code"]["commit"][:8] == PIN and not s["code"]["dirty"] and s["servers"] == 128, (stem, s["code"]["commit"], s["servers"])
    b = np.load(stem + ".tasks_bsz.npy").astype(np.float64); ts, g = workload(rung, w)
    ids = b[:, 0].astype(int); assert np.array_equal(np.sort(ids), np.arange(len(ts))), stem
    g = g[ids]; arr, comp = b[:, 1], b[:, 2]
    assert np.all(np.isfinite(comp)) and abs((comp - arr).mean() - s["averageElapsedTime"]) < 1e-6 * max(1, s["averageElapsedTime"]), stem
    u, inv, cnt = np.unique(g, return_inverse=True, return_counts=True)
    first = np.full(len(u), np.inf); last = np.full(len(u), -np.inf); np.minimum.at(first, inv, arr); np.maximum.at(last, inv, comp)
    m = arr >= 360; k = (cnt >= 2) & (first >= 360)
    return {"task": float((comp - arr)[m].mean()), "req": float((last - first)[k].mean()), "n_task": int(m.sum()), "n_req": int(k.sum())}
V = {}
for kind in (BAR, GNN, MLP):
    for sd in SEEDS[kind]:
        for t in TOPOS:
            for w in W:
                for r in RUNGS: V[(kind, sd, t, w, r)] = cell(kind, sd, t, w, r)
assert len(V) == 480, len(V)
def pct(a, b): return 100.0 * (a / b - 1.0)
def comp(A, B, metric, rung, seeds_a=None, seeds_b=None):
    per = []
    for t in TOPOS:
        vals = []
        for w in W:
            sa = [pct(V[(A, s, t, w, rung)][metric], V[(B, sb, t, w, rung)][metric]) for s in (seeds_a or SEEDS[A]) for sb in [None] if False]
            a = {s: V[(A, s, t, w, rung)][metric] for s in (seeds_a or SEEDS[A])}; b = {s: V[(B, s, t, w, rung)][metric] for s in (seeds_b or SEEDS[B])}
            if len(b) == 1: bb = list(b.values())[0]; per_seed = [pct(x, bb) for x in a.values()]
            elif len(a) == 1: aa = list(a.values())[0]; per_seed = [pct(aa, y) for y in b.values()]
            else: per_seed = [pct(a[s], b[s]) for s in a if s in b]
            vals.append(st.mean(per_seed))
        per.append(st.median(vals))
    return {"median_pct": st.median(per), "faster": sum(1 for x in per if x < 0), "n": len(per), "p": exact_wilcoxon(per), "per_topology": per}
out = {"pin": PIN, "topologies": TOPOS, "cells": len(V), "rows": {}}
PAIRS = (("gnn_vs_bar", GNN, BAR), ("mlp_vs_bar", MLP, BAR), ("gnn_vs_mlp", GNN, MLP))
for name, A, B in PAIRS:
    for metric in ("task", "req"):
        for rung in RUNGS:
            out["rows"][f"{name}|{metric}|{rung}|seedmean"] = comp(A, B, metric, rung)
            if name != "gnn_vs_mlp" or True:
                for s in (1, 2):
                    out["rows"][f"{name}|{metric}|{rung}|s{s}"] = comp(A, B, metric, rung, seeds_a=(s,) if A != BAR else None, seeds_b=(s,) if B != BAR else None)
out["admission"] = "see t12_read.json"
json.dump(out, open(OUT, "w"), indent=1)
for k, v in out["rows"].items(): print(f"{k:34s} median {v['median_pct']:+7.2f} %  faster {v['faster']}/{v['n']}  p {v['p']}")
#print("admitted topologies", sum(1 for a in adm.values() if a["admitted"]), "of", len(adm), "unknown", sum(1 for a in adm.values() if a["unknown"]))
