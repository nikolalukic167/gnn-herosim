#!/usr/bin/env python3
"""Frozen extension of scale_diagnosis.py (EXPLORATORY; filed cells only; no label). usage: scale_diagnosis_ext.py <new output path>
Adds what SCIENTIST's plan has and the first read lacks: (1) cellular and Wi-Fi byte and second shares of peer exchange by size; (2) replicas, cap share,
cold and scale-ups by size; (3) the bar's id-decile placement share; (4) the per-term decomposition, whole run and after 360 s.
Same roots, topologies, windows, seeds and cell means as scale_diagnosis.py. Items the filed cells cannot support print NOT COMPUTABLE with the reason; nothing is rerun."""
import json, os, sys
import numpy as np

U = "/share/nikola.lukic/uncut_groups"
BAR = "cd_ib0_inflightfull_k100"; GNN = "ra_gnn_eng_combfl_nosplit_k100"; MLP = "ra_mlp_same_combfl_nosplit_k100"
W = ("g0", "g1", "g2", "g3")
SIZES = {
    24: dict(topos=range(19537, 19549), bar=f"{U}/comb/cells_bar_g8", gnn=f"{U}/comb/cells_comb_fl", mlp=f"{U}/comb/cells_comb"),
    64: dict(topos=range(30129, 30141), bar=f"{U}/size64/cells_bar", gnn=f"{U}/size64/cells_learned", mlp=None),
    128: dict(topos=range(30241, 30253), bar=f"{U}/size128/cells_bar", gnn=f"{U}/size128/cells_learned", mlp=f"{U}/size128/mlp_cells"),
}
ARMS = (("bar", BAR, (0,)), ("gnn", GNN, (1, 2)), ("mlp", MLP, (1, 2)))
TERMS = ("W_sched_wait", "Q_resid", "N_prop", "P_pull", "K_cold", "R_rendezvous", "X_exchange", "L_lock_wait", "E_exec")


def cell(p, servers):
    s = json.load(open(p)); n = s["num_tasks"]; base = p[:-len(".summary.json")]
    cls = s["peerExchangeByAccessClass"]
    d = {"elapsed": s["averageElapsedTime"], "W_sched_wait": s["averageWaitTime"], "Q_resid": s["averageQueueTime"] - s["averageNetworkLatency"] - s["averagePullTime"],
         "N_prop": s["averageNetworkLatency"], "P_pull": s["averagePullTime"], "K_cold": s["averageColdStartTime"], "R_rendezvous": s["totalPeerRendezvousWait"] / n,
         "X_exchange": s["totalPeerExchangeTime"] / n, "L_lock_wait": s["lock_wait"]["mean"], "E_exec": s["averageExecutionTime"]}
    sec = {k: v["seconds"] for k, v in cls.items()}; tr = {k: v["transfers"] for k, v in cls.items()}
    by = {k: v.get("bytes") for k, v in cls.items()}
    for tag in ("cellular", "wifi"):
        ks = [k for k in cls if tag in k]
        d[f"sec_{tag}"] = sum(sec[k] for k in ks) / max(sum(sec.values()), 1e-12)
        d[f"tr_{tag}"] = sum(tr[k] for k in ks) / max(sum(tr.values()), 1)
        d[f"bytes_{tag}"] = (sum(by[k] for k in ks) / sum(by.values())) if all(v is not None for v in by.values()) and sum(by.values()) > 0 else np.nan
    rc = s["replica_count_series"]; cap = 5 * servers; tot = np.asarray(rc["total"], float); so = s["scaleOut"]; sc = so["scale_ups_by_cause"]
    d.update(replica_mean=rc["time_mean"], replica_peak=rc["peak"], cap_share_mean=rc["time_mean"] / cap, time_at_cap=float((tot >= 0.95 * cap).mean()),
             cold_start_pct=s["cold_start_pct"], scale_ups_load=sc.get("load", 0), scale_ups_reach=sc.get("reachability", 0))
    a = np.load(base + ".tasks.npy"); lk = np.load(base + ".tasks_lock.npy")
    el = a[:, 1].astype(float) - a[:, 0]; m = a[:, 0] >= 360
    ss = s["steady_state"]["after_360s"]
    if m.sum() != ss["n"] or abs(el[m].mean() - ss["mean"]) > 1e-3 * ss["mean"]:
        raise SystemExit(f"FAIL LOUD: {base}: after-360 elapsed {el[m].mean()} n {m.sum()} != summary {ss}")
    d["elapsed_after360"] = float(el[m].mean()); d["elapsed_whole_arr"] = float(el.mean())
    m2 = lk[:, 1] >= 360
    d["lock_after360"] = float(lk[m2, 3].mean()); d["lock_whole_arr"] = float(lk[:, 3].mean())
    nodekeys = [k for k in s if "node" in k.lower() and k not in ("accessClasses",)] + ([] if a.shape[1] == 2 else ["tasks.npy cols"])
    d["_nodekeys"] = len(nodekeys)
    return d


def means(rows):
    return {k: float(np.nanmean([r[k] for r in rows])) for k in rows[0] if not k.startswith("_")}


def main():
    outp = sys.argv[1]
    if os.path.exists(outp): raise SystemExit(f"FAIL LOUD: {outp} exists")
    res = {}; out = []
    for size, S in SIZES.items():
        for rung in ("moderate", "heavy"):
            for arm, name, seeds in ARMS:
                root = S[arm]
                if not root: out.append(f"-- {size} {rung} {arm}: no filed cells"); continue
                tops = []
                for t in S["topos"]:
                    ws = [means([cell(f"{root}/cc40s{t}__{w}{rung}__{name}_s{sd}.summary.json", size) for sd in seeds]) for w in W]
                    tops.append({k: float(np.nanmean([x[k] for x in ws])) for k in ws[0]})
                res[(size, rung, arm)] = tops
    f = lambda sz, r, a, k: np.nanmean([x[k] for x in res[(sz, r, a)]])
    for rung in ("moderate", "heavy"):
        out.append(f"=== {rung}")
        out.append("  (1) peer-exchange share by access class touching cellular / wifi (pair classes a+b count under both); mean over topologies x windows")
        out.append(f"    {'arm size':14s}" + "".join(f"{h:>14s}" for h in ("sec_cellular", "sec_wifi", "tr_cellular", "tr_wifi", "bytes_cellular", "bytes_wifi")))
        for arm in ("bar", "gnn", "mlp"):
            for sz in SIZES:
                if (sz, rung, arm) in res:
                    v = [f(sz, rung, arm, k) for k in ("sec_cellular", "sec_wifi", "tr_cellular", "tr_wifi", "bytes_cellular", "bytes_wifi")]
                    out.append(f"    {arm + ' ' + str(sz):14s}" + "".join(f"{x:14.4f}" if x == x else f"{'ABSENT':>14s}" for x in v))
        out.append("      (bytes ABSENT = the filed cells of that root predate the byte telemetry)")
        out.append("  (2) replicas, cap share, cold, scale-ups by size")
        ks = ("replica_mean", "replica_peak", "cap_share_mean", "time_at_cap", "cold_start_pct", "K_cold", "scale_ups_load", "scale_ups_reach")
        out.append(f"    {'arm size':14s}" + "".join(f"{k:>16s}" for k in ks))
        for arm in ("bar", "gnn", "mlp"):
            for sz in SIZES:
                if (sz, rung, arm) in res:
                    out.append(f"    {arm + ' ' + str(sz):14s}" + "".join(f"{f(sz, rung, arm, k):16.4f}" for k in ks))
        out.append("  (3) bar id-decile placement share: NOT COMPUTABLE from filed cells; no summary key or per-task array carries the node a task was placed on (tasks.npy is (start, end); tasks_lock is (idx, start, end, lock)); it needs a node dump from a rerun, which this read does not do")
        out.append("  (4) per-term decomposition, whole run (cell means) and after 360 s (task start >= 360 s)")
        out.append(f"    {'arm size':14s}" + "".join(f"{k:>14s}" for k in ("elapsed",) + TERMS) + f"{'elapsed>360':>14s}{'lock>360':>12s}")
        for arm in ("bar", "gnn", "mlp"):
            for sz in SIZES:
                if (sz, rung, arm) in res:
                    out.append(f"    {arm + ' ' + str(sz):14s}" + "".join(f"{f(sz, rung, arm, k):14.4f}" for k in ("elapsed",) + TERMS)
                               + f"{f(sz, rung, arm, 'elapsed_after360'):14.4f}{f(sz, rung, arm, 'lock_after360'):12.4f}")
        out.append("      after 360 s only elapsed and lock wait can be computed from the filed per-task arrays; queue, pull, cold, rendezvous, exchange, exec after 360 s are NOT COMPUTABLE (no per-task array)")
        out.append("      after-360 elapsed was asserted equal to the summaries' steady_state.after_360s mean (1e-3 relative) in every cell")
    txt = "\n".join(out); print(txt); open(outp, "w").write(txt + "\n")


main()
