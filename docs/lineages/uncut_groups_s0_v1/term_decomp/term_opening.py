#!/usr/bin/env python3
"""Opening-only (first 360 s) breakdown, DESCRIPTIVE, filed cells only, addendum to term_decomp.
usage: term_opening.py <set a|b> <outdir>
Opening = tasks dispatched before 360 s (tasks_bsz col 1 < 360; reproduces summary steady_state.after_360s n and mean exactly: after-360 = dispatch >= 360).
What the filed artifacts carry per task: dispatch time, completion time (so elapsed), group size, served batch size (tasks_bsz) and the lock wait (tasks_lock col 3).
The other Eq. 2 terms (W, queue, ingress, pull, cold, rendezvous, exchange, input/output, exec) exist ONLY as whole-run summary means, and the summaries carry an
after-360 mean for elapsed only, so whole-minus-after cannot be formed for them: they are NOT available for the opening. Reported: elapsed, lock wait, task counts,
elapsed per 60 s dispatch bin of the opening, and elapsed per group-size bucket in the opening.
Per topology: mean over g0-g3 of the window-level mean (GNN: mean over seeds 1-2 for the pooled row; seed rows separate); faster/12 = GNN < bar; raw two-sided Wilcoxon over 12 topology means."""
import json, os, sys
import numpy as np
from scipy.stats import wilcoxon

U = "/share/nikola.lukic/uncut_groups"
SET, OUT = sys.argv[1], sys.argv[2]
GNN, BAR = "ra_gnn_eng_combfl_nosplit_k100", "cd_ib0_inflightfull_k100"
if SET == "a":
    TOPOS = list(range(19537, 19549)); BD = f"{U}/comb/cells_bar_g8"; GD = f"{U}/comb/cells_comb_fl"; NAME = "gate topologies 19537-19548"
else:
    TOPOS = list(range(30129, 30141)); BD = f"{U}/size64/cells_bar"; GD = f"{U}/size64/cells_learned"; NAME = "size-64 topologies 30129-30140"
WINDOWS = ("g0", "g1", "g2", "g3"); T0 = 360.0
BINS = [(0, 60), (60, 120), (120, 180), (180, 240), (240, 300), (300, 360)]
BK = (("le10", 0, 10), ("11_50", 11, 50), ("51_100", 51, 100), ("101_plus", 101, 10 ** 9))


def cell(p):
    a = np.load(p + ".tasks_bsz.npy"); l = np.load(p + ".tasks_lock.npy")
    o = np.argsort(a[:, 0], kind="stable"); a = a[o]; l = l[np.argsort(l[:, 0], kind="stable")]
    assert (a[:, 0] == l[:, 0]).all()
    lat = a[:, 2] - a[:, 1]; op = a[:, 1] < T0
    r = {"n_open": float(op.sum()), "open_share": float(op.mean()), "open_elapsed": lat[op].mean(), "after_elapsed": lat[~op].mean(), "whole_elapsed": lat.mean(),
         "open_lock": l[op, 3].mean(), "after_lock": l[~op, 3].mean()}
    s = json.load(open(p + ".summary.json"))["steady_state"]["after_360s"]
    assert s["n"] == int((~op).sum()) and abs(s["mean"] - lat[~op].mean()) < 1e-9, "after-360 does not reproduce the summary"
    for i, (lo, hi) in enumerate(BINS):
        m = (a[:, 1] >= lo) & (a[:, 1] < hi); r[f"bin{lo}_{hi}"] = lat[m].mean() if m.any() else np.nan
    g = a[:, 3]
    for name, lo, hi in BK:
        m = op & (g >= lo) & (g <= hi); r[f"bk_{name}_share"] = float(m.sum() / op.sum()); r[f"bk_{name}"] = lat[m].mean() if m.any() else np.nan
    return r


def agg(rows):
    return {k: float(np.nanmean([x[k] for x in rows])) for k in rows[0]}


def main():
    res, L = {}, [f"=== OPENING (dispatch < 360 s), {NAME}; mean s per task; delta = GNN - bar; faster = GNN < bar; Wilcoxon raw two-sided over 12 topology means"]
    for r in ("moderate", "heavy"):
        per = {"bar": [], "s1": [], "s2": [], "gnn": []}
        for t in TOPOS:
            rows = {"bar": [], "s1": [], "s2": [], "gnn": []}
            for w in WINDOWS:
                rows["bar"].append(cell(f"{BD}/cc40s{t}__{w}{r}__{BAR}_s0"))
                c1, c2 = (cell(f"{GD}/cc40s{t}__{w}{r}__{GNN}_s{s}") for s in (1, 2))
                rows["s1"].append(c1); rows["s2"].append(c2); rows["gnn"].append({k: float(np.nanmean([c1[k], c2[k]])) for k in c1})
            for a in per:
                per[a].append(agg(rows[a]))
        keys = list(per["bar"][0].keys()); R = {}
        L.append(f"--- {r}: {'quantity':22s} {'bar':>9s} | {'GNN s1-2':>9s} {'delta':>9s} {'fast/12':>8s} {'p':>7s} | {'s1 delta':>9s} {'fast':>5s} {'p':>7s} | {'s2 delta':>9s} {'fast':>5s} {'p':>7s}")

        def st(a, k):
            g = np.array([x[k] for x in per[a]]); b = np.array([x[k] for x in per["bar"]]); d = g - b
            try:
                p = float(wilcoxon(g, b).pvalue) if np.any(d != 0) else 1.0
            except ValueError:
                p = float("nan")
            return float(g.mean()), float(b.mean()), float(d.mean()), int((d < 0).sum()), p
        for k in keys:
            if k.endswith("_share") or k == "n_open" or k == "open_share":
                g = np.mean([x[k] for x in per["gnn"]]); b = np.mean([x[k] for x in per["bar"]])
                R[k] = {"gnn": float(g), "bar": float(b)}; L.append(f"        {k:22s} {b:9.4f} | {g:9.4f}"); continue
            R[k] = {a: dict(zip(("gnn", "bar", "delta", "faster_of_12", "wilcoxon_p"), st(a, k))) for a in ("gnn", "s1", "s2")}
            g, s1, s2 = R[k]["gnn"], R[k]["s1"], R[k]["s2"]
            L.append(f"        {k:22s} {g['bar']:9.4f} | {g['gnn']:9.4f} {g['delta']:+9.4f} {g['faster_of_12']:5d}/12 {g['wilcoxon_p']:7.4f} | {s1['delta']:+9.4f} {s1['faster_of_12']:3d}/12 {s1['wilcoxon_p']:7.4f} | {s2['delta']:+9.4f} {s2['faster_of_12']:3d}/12 {s2['wilcoxon_p']:7.4f}")
        res[r] = R
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f"{OUT}/term_opening_{SET}.json", "w"), indent=1); open(f"{OUT}/term_opening_{SET}.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))


main()
