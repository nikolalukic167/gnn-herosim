#!/usr/bin/env python3
"""ILS-live feasibility projection (amendment A2; descriptive, filed before any live ILS cell).

rho_sched = (sum over a cell's batches of evaluations x w_S + the bar's own charge) / the cell's run length, per load; median over the 12 topologies.
Batch sizes and counts are the bar's (dose_7cb483c5, cells_ref_{mod,heavy}070, the 0.7 target): per cell, count_n = tasks with servedBatchSize n / n, from
`.tasks_bsz.npy` column 4. Evaluations per batch are the MEDIAN n_evals of the label ILS runs on the K1(b) states of the same decision size bucket (11_50, gt50;
the ILS never runs on a batch of at most 10 tasks) at each budget: 200 = the filed K1(b) rows (rows_all.jsonl, 190 of 194 stopped on the budget), 10 and 50 =
this study's runs of `uncut_ils.py run --arm cd_ib0 --seed 0 --replay none` at the identity-checked pin e8c14124 (a wall cap of 1800 s per state; the number of
states that stopped on it is printed: a capped state is a lower bound on its evaluations). The states were captured under cd_ib0, whose S has no in-flight term;
the live arm's S has it, so the counts are an approximation, disclosed. w_S is each budget's own frozen per-bucket rate of decision_cost_w.json (A3; k = 1, per rung).
The bar's own charge is the cell's sched_busy_seconds. Run length: the arrival window (arrival_end.last_arrival_s), the time a single scheduler server has to
serve every batch; the cell's end time is printed beside it. rho_sched >= 1 at a load = "infeasible live at that load".

usage: ils_live_projection.py --u /share/nikola.lukic/uncut_groups --w decision_cost_w.json [--out OUT.json]   (A3: each budget uses its own w_S)"""
import argparse
import glob
import json
import os
import statistics as st

import numpy as np

TOPOS = list(range(19537, 19549))
WINDOWS = ("g0", "g1", "g2", "g3")
RUNGS = ("moderate", "heavy")
SHORT = {"moderate": "mod", "heavy": "heavy"}
BUDGETS = (10, 50, 200)


def dbucket(n):
    return "le10" if n <= 10 else "11_50" if n <= 50 else "gt50"


def evals_per_batch(u):
    """{budget: {bucket: (median evals, n states, n capped)}} from the runs."""
    out = {}
    src = {200: [f"/share/nikola.lukic/uncut_ils/k1b/rows_all.jsonl"],
           10: sorted(glob.glob(f"{u}/ils_proj/rows_B10_s*.jsonl")), 50: sorted(glob.glob(f"{u}/ils_proj/rows_B50_s*.jsonl"))}
    for b, files in src.items():
        rows = [json.loads(l) for f in files for l in open(f)]
        by = {}
        for r in rows:
            by.setdefault(dbucket(int(r["n_batch"])), []).append(r)
        out[b] = {k: (st.median(x["n_evals"] for x in v), len(v), sum(1 for x in v if x["stopped"] == "seconds")) for k, v in by.items()}
        out[b]["_states"] = len(rows)
    return out


def batch_counts(stem):
    a = np.load(stem + ".tasks_bsz.npy")
    sizes = a[:, 4].astype(int)
    out = {}
    for n in np.unique(sizes):
        if n > 0:
            out[int(n)] = int((sizes == n).sum()) // int(n)
    return out


KIND = {10: "cd_ib0_ils_p10", 50: "cd_ib0_ils_p50", 200: "cd_ib0_ils"}      # A3: each budget is charged with its own w_S


def project(u, w):
    ev = evals_per_batch(u)
    res = {}
    for r in RUNGS:
        wsb = {B: {b: float(v) for b, v in w["w_S_ms_per_plan_by_bucket"][KIND[B]][r].items()} for B in BUDGETS}
        per = {B: {} for B in BUDGETS}
        for t in TOPOS:
            for win in WINDOWS:
                stem = f"{u}/anomaly/dose_7cb483c5/cells_ref_{SHORT[r]}070/cc40s{t}__{win}{r}__cd_ib0_inflightfull_k100_s0"
                s = json.load(open(stem + ".summary.json"))
                cnt = batch_counts(stem)
                bar_s = float(s["schedulerCounters"]["sched_busy_seconds"])
                run = float(s["arrival_end"]["last_arrival_s"])
                for B in BUDGETS:
                    ils_s = sum(c * ev[B][dbucket(n)][0] * wsb[B][dbucket(n)] * 1e-3 for n, c in cnt.items() if n > 10)
                    per[B].setdefault(t, []).append(((ils_s + bar_s) / run, bar_s / run, ils_s / run, (ils_s + bar_s) / float(s["endTime"])))
        res[r] = {}
        for B in BUDGETS:
            tv = {t: [st.median(x[i] for x in v) for i in range(4)] for t, v in per[B].items()}
            res[r][B] = {"rho_median": st.median(v[0] for v in tv.values()), "rho_min": min(v[0] for v in tv.values()), "rho_max": max(v[0] for v in tv.values()),
                         "bar_part_median": st.median(v[1] for v in tv.values()), "ils_part_median": st.median(v[2] for v in tv.values()),
                         "rho_over_end_median": st.median(v[3] for v in tv.values()),
                         "topologies_ge1": sum(1 for v in tv.values() if v[0] >= 1.0), "per_topology": {str(t): round(v[0], 4) for t, v in tv.items()}}
    return ev, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--u", default="/share/nikola.lukic/uncut_groups")
    ap.add_argument("--w", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    w = json.load(open(a.w))
    ev, res = project(a.u, w)
    L = [__doc__.strip().split("\n\n")[0], "", f"w file {a.w}; w_S per budget kind and rung (ms per plan): " + json.dumps({str(B): w["w_S_ms_per_plan_by_bucket"][KIND[B]] for B in BUDGETS}), "",
         "evaluations per ILS batch (median; states; stopped on the wall cap), by decision size bucket:"]
    for B in BUDGETS:
        L.append(f"  budget {B:3d}: " + "; ".join(f"{k} {v[0]:.0f} ({v[1]} states, {v[2]} capped)" for k, v in sorted(ev[B].items()) if k != "_states"))
    L.append("")
    L.append("rho_sched = charged decision seconds / arrival-window length, median over 12 topologies (min - max); topologies with rho >= 1; the bar's part and the ILS part")
    for r in RUNGS:
        for B in BUDGETS:
            x = res[r][B]
            L.append(f"  {r:8s} budget {B:3d}: rho {x['rho_median']:.4f} ({x['rho_min']:.4f} - {x['rho_max']:.4f}); topologies >= 1: {x['topologies_ge1']}/12; "
                     f"bar {x['bar_part_median']:.4f} + ILS {x['ils_part_median']:.4f}; over the end time {x['rho_over_end_median']:.4f}  ->  "
                     + ("INFEASIBLE LIVE" if x["rho_median"] >= 1.0 else "feasible"))
    print("\n".join(L))
    if a.out:
        json.dump({"evals": {str(b): {k: v for k, v in d.items()} for b, d in ev.items()}, "rho": res}, open(a.out, "w"), indent=1, default=str)
        open(a.out.replace(".json", ".txt"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
