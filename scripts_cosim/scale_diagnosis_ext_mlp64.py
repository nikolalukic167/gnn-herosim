#!/usr/bin/env python3
"""One-root addendum to scale_diagnosis_ext.py (EXPLORATORY; filed cells only; no label). usage: scale_diagnosis_ext_mlp64.py <new output path>
Adds the FL MLP at 64 servers that scale_diagnosis_ext.py reported as missing: root size64/cells_desc, arm ra_mlp_same_combfl_nosplit_k100, seeds 1-2,
topologies 30129-30140, windows g0-g3, moderate and heavy = 192 cells, pin 74ddf70e. _k100 only: the uncharged (no _k100 suffix) and _kr cells in that root are
excluded and asserted absent from the read. The cell(), means() and table code is copied unchanged from scale_diagnosis_ext.py; nothing else changes."""
import json, os, sys
import numpy as np

U = "/share/nikola.lukic/uncut_groups"
BAR = "cd_ib0_inflightfull_k100"; GNN = "ra_gnn_eng_combfl_nosplit_k100"; MLP = "ra_mlp_same_combfl_nosplit_k100"
W = ("g0", "g1", "g2", "g3")
ROOT = f"{U}/size64/cells_desc"; TOPOS = range(30129, 30141); SERVERS = 64; PIN = "74ddf70e"
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
    out = []; res = {}
    for rung in ("moderate", "heavy"):
        tops = []
        for t in TOPOS:
            ws = []
            for w in W:
                rows = []
                for sd in (1, 2):
                    p = f"{ROOT}/cc40s{t}__{w}{rung}__{MLP}_s{sd}.summary.json"
                    s = json.load(open(p))
                    if s["arm"] != "ra_mlp_same_combfl_nosplit_k100" and not s["arm"].startswith("ra_mlp_same_combfl_nosplit"): raise SystemExit(f"FAIL LOUD: arm {s['arm']}")
                    if "_k100_" not in os.path.basename(p) or "_kr" in os.path.basename(p): raise SystemExit(f"FAIL LOUD: not a _k100 cell {p}")
                    if s["code"]["describe"] != PIN or s["servers"] != SERVERS or s["checkpoint_seed"] != sd: raise SystemExit(f"FAIL LOUD: provenance {p}")
                    rows.append(cell(p, SERVERS)); res.setdefault("n", []).append(p)
                ws.append(means(rows))
            tops.append({k: float(np.nanmean([x[k] for x in ws])) for k in ws[0]})
        res[rung] = tops
    if len(res["n"]) != 192 or len(set(res["n"])) != 192: raise SystemExit(f"FAIL LOUD: {len(res['n'])} cells, expect 192")
    out.append(f"192 cells read: {ROOT}, {MLP} s1-2, pin {PIN}, 12 topologies x 4 windows x 2 rungs; uncharged and _kr cells of that root excluded")
    f = lambda r, k: np.nanmean([x[k] for x in res[r]])
    for rung in ("moderate", "heavy"):
        out.append(f"=== {rung} (mlp 64)")
        ks = ("sec_cellular", "sec_wifi", "tr_cellular", "tr_wifi", "bytes_cellular", "bytes_wifi")
        out.append("  (1) " + "".join(f"{k}={f(rung, k):.4f}  " if f(rung, k) == f(rung, k) else f"{k}=ABSENT  " for k in ks))
        ks = ("replica_mean", "replica_peak", "cap_share_mean", "time_at_cap", "cold_start_pct", "K_cold", "scale_ups_load", "scale_ups_reach")
        out.append("  (2) " + "  ".join(f"{k}={f(rung, k):.4f}" for k in ks))
        out.append("  (4) " + "  ".join(f"{k}={f(rung, k):.4f}" for k in ("elapsed",) + TERMS) + f"  elapsed>360={f(rung, 'elapsed_after360'):.4f}  lock>360={f(rung, 'lock_after360'):.4f}")
    txt = "\n".join(out); print(txt); open(outp, "w").write(txt + "\n")


main()
