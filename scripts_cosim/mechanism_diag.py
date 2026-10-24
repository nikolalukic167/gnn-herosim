#!/usr/bin/env python3
"""llf_cd_confirm_v1 mechanism diagnostic on the SEALED 12 (already read; the fresh-24 cells are never opened): why does the static gnn beat the CD bar, and is heavy a scope limit or a fixable gnn defect?
Arms: static gnn and wide MLP (seeds 1-3, the -final.pt served) from d_conf_final, the bar (seed 0) from d_conf. Topologies 30285-30297 minus 30288 (main_* plus r1_* for 30297); windows g0, g1 (the only windows run on the sealed 12).

  cells    mechanism_cells.json: per cell, from the cell's summary and tasks.npy: exchange seconds per task, remote transfers per task, co-location rate of peer pairs (1 - remote transfers / (2 x peer pairs of the workload)),
           shared-transfer fraction, extra (contention) seconds per transfer, peak concurrent flows on one link, overlap share, cellular share of exchange seconds, queue share, backlog growth (in-system at 3/4 over 1/2), latency by
           arrival octile, latency decomposition (queue, wait, cold/init/pull, exec, exchange, decision per task). Then mechanism_contrasts.json: gnn vs bar and gnn vs wide per rung and window set (all, g0, g1), per topology the
           median over pairs, then the median over topologies, lower/12; two ways (all cells, without runaways = whole run > 10x the arm's own median in the rung, a pair dropped when either cell is flagged).
  rho      mechanism_rho.json: from the audit-trace re-runs (anomaly/fairshare/mech/<arm>/<window>_<rung>_<topology>.json written by mech_extract.py, each checked equal to its filed cell): offered usage per directed link (peer +
           ingress summed on a link), the share of cellular server uplinks with rho > 1, the busiest link's transfer share and rho, the share of peer bytes crossing a rho > 1 link, and the time structure and group concentration for the runaway cells.
usage: mechanism_diag.py cells|rho --out <dir>"""
import argparse, glob, json, math, os, statistics as st, sys
import numpy as np

D = "/share/nikola.lukic/uncut_groups/anomaly/fairshare"; IN = "/share/nikola.lukic/uncut_groups/xs/in_llf_conf_final"
TOPOS = [30285, 30286, 30287, 30289, 30290, 30291, 30292, 30293, 30294, 30295, 30296, 30297]; WINS = ("g0", "g1"); RUNGS = ("moderate", "heavy")
KINDS = {"gnn": "ra_gnn_eng_combfl_linkstatic_e40_nosplit_k100", "wide": "ra_mlp_wide_combfl_linkstatic_e40_nosplit_k100", "bar": "cd_ib0_inflightfull_k100"}

def cell_path(arm, rung, win, t, seed):
    g = "r1" if t == 30297 else "main"
    if arm == "bar": return f"{D}/d_conf/{g}_{win}{rung}_bar/cc40s{t}__{win}{rung}__{KINDS[arm]}_s0.summary.json"
    return f"{D}/d_conf_final/{g}_{win}{rung}_static_s{seed}/cc40s{t}__{win}{rung}__{KINDS[arm]}_s{seed}.summary.json"

def seeds_of(arm): return (0,) if arm == "bar" else (1, 2, 3)

def metrics(path, pairs):
    s = json.load(open(path)); N = s["num_tasks"]; fs = s.get("peerExchangeFairshare") or {}; ac = s.get("peerExchangeByAccessClass") or {}; ov = s.get("peerExchangeOverlap") or {}
    a = np.load(path.replace(".summary.json", ".tasks.npy")); lat = a[:, 1] - a[:, 0]; order = np.argsort(a[:, 0], kind="stable"); oct_ = [float(lat[i].mean()) for i in np.array_split(order, 8)]
    ex_s = sum(v["seconds"] for v in ac.values()) or 1.0; ex_b = sum(v["bytes"] for v in ac.values()) or 1.0
    cel_s = sum(v["seconds"] for k, v in ac.items() if "cellular" in k); cel_b = sum(v["bytes"] for k, v in ac.items() if "cellular" in k)
    tr = fs.get("transfers") or sum(v["transfers"] for v in ac.values())
    dt = s.get("decisionTiming") or {}; bp = (s.get("backlog_profile") or {}).get("in_system_at") or {}
    return {
        "avg_latency_s": s["averageElapsedTime"], "n_tasks": N, "end_time_s": s["endTime"],
        "exchange_s_per_task": s["totalPeerExchangeTime"] / N, "exchange_share_of_latency": s["totalPeerExchangeTime"] / s["total_rtt"],
        "transfers_per_task": tr / N, "colocation_rate": 1.0 - tr / (2.0 * pairs), "shared_fraction": (fs.get("shared_transfers", 0) / tr) if tr else None,
        "extra_s_per_transfer": (fs.get("extra_seconds", 0.0) / tr) if tr else None, "peak_flows_on_a_link": fs.get("peak_flows_on_a_link"),
        "overlap_seconds_share": ov.get("overlap_seconds_share"), "cellular_exchange_seconds_share": cel_s / ex_s, "cellular_exchange_bytes_share": cel_b / ex_b,
        "queue_share": s.get("queue_share"), "backlog_in_system_ratio": (bp["three_quarter"] / bp["half"]) if bp.get("half") else None,
        "latency_last_over_first_octile": oct_[-1] / oct_[0] if oct_[0] > 0 else None, "latency_by_octile": oct_,
        "decomp": {"queue": s["averageQueueTime"], "placement_wait": s["averageWaitTime"], "cold_init_pull": s["averageColdStartTime"] + s["averageInitializationTime"] + s["averagePullTime"],
                   "exec": s["averageExecutionTime"], "exchange": s["totalPeerExchangeTime"] / N, "decision": (dt.get("total_s", 0.0) / N) if dt else 0.0, "compute_stage": s["averageComputeTime"]},
        "scale_up_failures": sum((s.get("scaleOut") or {}).get("scale_up_failures_by_cause", {}).values()), "panic_ticks": (s.get("scaleOut") or {}).get("panic_ticks"),
    }

def wilcoxon(x):
    import itertools
    d = [v for v in x if v != 0]; n = len(d)
    if n == 0: return 1.0
    order = sorted(range(n), key=lambda i: abs(d[i])); ranks = [0.0] * n; i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]): j += 1
        for k in range(i, j + 1): ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    tot = sum(ranks); w = sum(r for r, v in zip(ranks, d) if v > 0); obs = abs(w - tot / 2); c = 0
    for sgn in itertools.product((0, 1), repeat=n): c += abs(sum(r for r, b in zip(ranks, sgn) if b) - tot / 2) >= obs - 1e-9
    return c / 2 ** n

RATE = {"colocation_rate", "shared_fraction", "overlap_seconds_share", "cellular_exchange_seconds_share", "cellular_exchange_bytes_share", "queue_share", "exchange_share_of_latency"}
CONTRAST_METRICS = ["avg_latency_s", "exchange_s_per_task", "transfers_per_task", "colocation_rate", "shared_fraction", "extra_s_per_transfer", "peak_flows_on_a_link", "overlap_seconds_share",
                    "cellular_exchange_seconds_share", "exchange_share_of_latency", "queue_share", "backlog_in_system_ratio", "latency_last_over_first_octile"]

def cmd_cells(out):
    cells = {}; pairs_of = {}
    for rung in RUNGS:
        for win in WINS:
            wl = json.load(open(f"{IN}/wf1_{rung}/wl/grounded_{win}_n50000.json")); pairs_of[(rung, win)] = wl["payload_cap_v1"]["pairs"]
    for arm in KINDS:
        for rung in RUNGS:
            for win in WINS:
                for t in TOPOS:
                    for sd in seeds_of(arm):
                        cells[(arm, rung, win, t, sd)] = metrics(cell_path(arm, rung, win, t, sd), pairs_of[(rung, win)])
    med = {(a, r): st.median(v["avg_latency_s"] for k, v in cells.items() if k[0] == a and k[1] == r) for a in KINDS for r in RUNGS}
    run = {k: v["avg_latency_s"] > 10 * med[(k[0], k[1])] for k, v in cells.items()}
    json.dump({"|".join(map(str, k)): dict(v, runaway=run[k]) for k, v in cells.items()}, open(f"{out}/mechanism_cells.json", "w"))
    flagged = sorted(k for k, v in run.items() if v)
    res = {"label": "sealed-12 confirmation cells, static e40 -final (seeds 1-3), bar s0; windows g0 and g1 only (g2/g3 were not run on these topologies)", "runaway_rule": "whole-run latency > 10x the arm's own median in the rung",
           "arm_medians": {f"{a}|{r}": med[(a, r)] for a in KINDS for r in RUNGS}, "runaway_cells": [{"arm": k[0], "rung": k[1], "window": k[2], "topology": k[3], "seed": k[4], "avg_latency_s": cells[k]["avg_latency_s"]} for k in flagged], "contrasts": {}}
    def pair_vals(A, B, rung, wins, metric, drop):
        vals = []
        for t in TOPOS:
            ps = []
            for w in wins:
                for sd in seeds_of(A):
                    ka = (A, rung, w, t, sd); kb = (B, rung, w, t, 0 if B == "bar" else sd)
                    if drop and (run[ka] or run[kb]): continue
                    x, y = cells[ka][metric], cells[kb][metric]
                    if x is None or y is None: continue
                    ps.append(100 * (x - y) if metric in RATE else (100 * (x - y) / y if y else None))
            ps = [p for p in ps if p is not None]
            if ps: vals.append(st.median(ps))
        return vals
    for B in ("bar", "wide"):
        for rung in RUNGS:
            for wlab, wins in (("g0+g1", WINS), ("g0", ("g0",)), ("g1", ("g1",))):
                for drop in (False, True):
                    key = f"gnn_vs_{B}|{rung}|{wlab}|{'without_runaways' if drop else 'all_cells'}"
                    res["contrasts"][key] = {}
                    for m in CONTRAST_METRICS:
                        v = pair_vals("gnn", B, rung, wins, m, drop)
                        if v: res["contrasts"][key][m] = {"unit": "pp difference" if m in RATE else "% difference", "median": st.median(v), "lower_of_n": f"{sum(x < 0 for x in v)}/{len(v)}", "p": wilcoxon(v), "per_topology": v}
    dec = {}
    for a in KINDS:
        for rung in RUNGS:
            for wl_ in ("g0", "g1"):
                ks = [k for k in cells if k[0] == a and k[1] == rung and k[2] == wl_]
                dec[f"{a}|{rung}|{wl_}"] = {c: st.median(cells[k]["decomp"][c] for k in ks) for c in cells[ks[0]]["decomp"]}
                dec[f"{a}|{rung}|{wl_}"]["n_cells"] = len(ks)
    res["latency_decomposition_median_of_cells_s_per_task"] = dec
    json.dump(res, open(f"{out}/mechanism_contrasts.json", "w"), indent=1)
    print("runaway cells:", len(flagged), {a: sum(1 for k in flagged if k[0] == a) for a in KINDS}, "| heavy g0:", sum(1 for k in flagged if k[1] == "heavy" and k[2] == "g0"))
    print("written", f"{out}/mechanism_cells.json", f"{out}/mechanism_contrasts.json")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=("cells", "rho")); ap.add_argument("--out", required=True); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.cmd == "cells": cmd_cells(a.out)
    else:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import mechanism_rho; mechanism_rho.run(a.out)

if __name__ == "__main__":
    main()
