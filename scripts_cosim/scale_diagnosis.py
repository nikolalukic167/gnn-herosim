#!/usr/bin/env python3
"""Scale diagnosis (reviewer point 3), descriptive, from FILED summaries only. No label.

Why does the bar degrade 24 -> 64 -> 128 servers and why do FL gnn and FL MLP gain? Per size, arm and rung: per-task Eq. 2 terms,
co-location share, transfers per task by access class, replicas (mean, peak, share of the run at the platform cap), scale-ups by cause, and the
client-to-server ratio (160 clients at every size). Suspects: the fixed 160 clients, reachability-driven replicas, id-order placement.
Pattern: load_anomaly.py. Mean over topologies x 4 windows (x seeds); paired per topology against the bar of the same size.
Per-task node assignments are not in the filed summaries, so id-order placement is tested only through a proxy (class mix of transfers vs servers).
"""
import json, os, sys
import numpy as np

U = "/share/nikola.lukic/uncut_groups"
BAR = "cd_ib0_inflightfull_k100"; GNN = "ra_gnn_eng_combfl_nosplit_k100"; MLP = "ra_mlp_same_combfl_nosplit_k100"
W = ("g0", "g1", "g2", "g3")
SIZES = {
    24: dict(topos=range(19537, 19549), inp=f"{U}/comb/in_comball", bar=f"{U}/comb/cells_bar_g8", gnn=f"{U}/comb/cells_comb_fl", mlp=f"{U}/comb/cells_comb"),
    64: dict(topos=range(30129, 30141), inp=f"{U}/size64/in_size64", bar=f"{U}/size64/cells_bar", gnn=f"{U}/size64/cells_learned", mlp=None),
    128: dict(topos=range(30241, 30253), inp=f"{U}/size128/in_size128", bar=f"{U}/size128/cells_bar", gnn=f"{U}/size128/cells_learned", mlp=f"{U}/size128/mlp_cells"),
}
CLASSES = ("wired+wired", "wifi+wired", "cellular+wired", "wifi+wifi", "cellular+wifi", "cellular+cellular")


def one(p, rel, servers):
    s = json.load(open(p)); n = s["num_tasks"]
    cls = s["peerExchangeByAccessClass"]
    tr = sum(v["transfers"] for v in cls.values()); sec = sum(v["seconds"] for v in cls.values())
    so = s["scaleOut"]; sc = so["scale_ups_by_cause"]; fl = so.get("scale_up_failures_by_cause", {})
    rc = s["replica_count_series"]; cap = 5 * servers; tot = np.asarray(rc["total"], float)
    d = {"elapsed": s["averageElapsedTime"], "W_sched_wait": s["averageWaitTime"],
         "Q_resid": s["averageQueueTime"] - s["averageNetworkLatency"] - s["averagePullTime"], "N_prop": s["averageNetworkLatency"],
         "P_pull": s["averagePullTime"], "K_cold": s["averageColdStartTime"], "R_rendezvous": s["totalPeerRendezvousWait"] / n,
         "X_exchange": s["totalPeerExchangeTime"] / n, "L_lock_wait": s["lock_wait"]["mean"], "E_exec": s["averageExecutionTime"],
         "transfers_per_task": tr / n, "s_per_transfer": sec / tr if tr else 0.0, "coloc_share": 1 - tr / rel,
         "replica_mean": rc["time_mean"], "replica_peak": rc["peak"], "replica_share_of_cap_mean": rc["time_mean"] / cap,
         "time_at_cap_share": float((tot >= 0.95 * cap).mean()), "scale_ups_load": sc.get("load", 0), "scale_ups_reach": sc.get("reachability", 0),
         "scale_up_fail_load": fl.get("load", 0), "scale_up_fail_reach": fl.get("reachability", 0),
         "starved_create_calls": (so.get("starved") or {}).get("create_calls", 0), "cold_start_pct": s["cold_start_pct"],
         "duration_s": float(s.get("endTime") or 0), "placement_wait_mean": s["placement_wait"]["mean"]}
    for c in CLASSES:
        d["tr_" + c] = cls.get(c, {}).get("transfers", 0) / n
    pct = s.get("latency_percentiles") or {}
    for k in ("p50", "p90", "p99"):
        if k in pct: d["lat_" + k] = float(pct[k])
    srv = [v for k, v in s["accessClasses"].items() if k.startswith("node")]
    for c in ("wired", "wifi", "cellular"):
        d["servers_" + c] = sum(1 for v in srv if v == c) / len(srv)
    return d


def cells(root, topos, arm, seeds, rung, rel, servers):
    out = []
    for t in topos:
        rows = []
        for w in W:
            ds = []
            for sd in seeds:
                p = f"{root}/cc40s{t}__{w}{rung}__{arm}_s{sd}.summary.json"
                if not os.path.exists(p): return None
                ds.append(one(p, rel[w], SERVERS_CUR))
            rows.append({k: float(np.mean([x[k] for x in ds])) for k in ds[0]})
        out.append({k: float(np.mean([r[k] for r in rows])) for k in rows[0]})
    return out


def main():
    global SERVERS_CUR
    res = {}; out = []
    for size, S in SIZES.items():
        SERVERS_CUR = size
        for rung in ("moderate", "heavy"):
            rel = {w: 2 * len(json.load(open(f"{S['inp']}/wf1_{rung}/wl/grounded_{w}_n50000.json"))["peer_exchange"]) for w in W}
            for arm, root, name, seeds in (("bar", S["bar"], BAR, (0,)), ("gnn", S["gnn"], GNN, (1, 2)), ("mlp", S["mlp"], MLP, (1, 2))):
                r = cells(root, S["topos"], name, seeds, rung, rel, size) if root else None
                if r is None:
                    out.append(f"-- {size} servers {rung} {arm}: no filed cells (skipped)"); continue
                res[(size, rung, arm)] = r
    keys = list(next(iter(res.values()))[0])
    for rung in ("moderate", "heavy"):
        out.append(f"=== {rung}: mean over 12 topologies x 4 windows (x seeds); clients 160 at every size")
        out.append("    client:server ratio  " + "  ".join(f"{sz}: {160 / sz:.3f}" for sz in SIZES))
        for arm in ("bar", "gnn", "mlp"):
            have = [sz for sz in SIZES if (sz, rung, arm) in res]
            out.append(f"  -- arm {arm}  sizes {have}")
            out.append(f"    {'term':28s}" + "".join(f"{sz:>12d}" for sz in have) + ("   d(24->last)" if len(have) > 1 else ""))
            for k in keys:
                v = [np.mean([x[k] for x in res[(sz, rung, arm)]]) for sz in have]
                out.append(f"    {k:28s}" + "".join(f"{x:12.4f}" for x in v) + (f"{v[-1] - v[0]:+14.4f}" if len(v) > 1 else ""))
        out.append(f"  -- paired per topology vs the bar of the same size: elapsed gain % = 100*(arm-bar)/bar (negative = faster), n topologies ahead")
        for arm in ("gnn", "mlp"):
            for sz in SIZES:
                if (sz, rung, arm) in res:
                    a = np.array([x["elapsed"] for x in res[(sz, rung, arm)]]); b = np.array([x["elapsed"] for x in res[(sz, rung, "bar")]])
                    g = 100 * (a - b) / b
                    out.append(f"    {arm} {sz:4d}: mean {g.mean():+7.2f} %  median {np.median(g):+7.2f} %  ahead {(a < b).sum()}/12")
        out.append("  -- bar term growth 24 -> 128 (share of the elapsed delta): sum of terms vs elapsed")
        if (24, rung, "bar") in res and (128, rung, "bar") in res:
            e0 = np.mean([x["elapsed"] for x in res[(24, rung, "bar")]]); e1 = np.mean([x["elapsed"] for x in res[(128, rung, "bar")]])
            out.append(f"    elapsed {e0:.4f} -> {e1:.4f} ({e1 - e0:+.4f} s)")
            for k in ("W_sched_wait", "Q_resid", "N_prop", "P_pull", "K_cold", "R_rendezvous", "X_exchange", "L_lock_wait", "E_exec"):
                d = np.mean([x[k] for x in res[(128, rung, "bar")]]) - np.mean([x[k] for x in res[(24, rung, "bar")]])
                out.append(f"      {k:16s} {d:+9.4f} s")
        out.append("  -- suspect 3 proxy (id-order): transfer-class mix vs server class mix (bar); placement by node is not in the filed summaries")
        for sz in SIZES:
            if (sz, rung, "bar") in res:
                r = res[(sz, rung, "bar")]; tot = np.mean([sum(x["tr_" + c] for c in CLASSES) for x in r])
                mix = "  ".join(f"{c} {100 * np.mean([x['tr_' + c] for x in r]) / tot:5.1f}%" for c in CLASSES)
                sm = "  ".join(f"{c} {100 * np.mean([x['servers_' + c] for x in r]):5.1f}%" for c in ("wired", "wifi", "cellular"))
                out.append(f"    {sz:4d}: transfers {mix} | servers {sm}")
    txt = "\n".join(out); print(txt)
    od = f"{U}/anomaly"; open(f"{od}/scale_diagnosis.txt", "w").write(txt + "\n")
    json.dump({f"{k[0]}|{k[1]}|{k[2]}": v for k, v in res.items()}, open(f"{od}/scale_diagnosis.json", "w"), indent=1)


main()
