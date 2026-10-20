#!/usr/bin/env python3
"""Additions (a)-(c) to load_anomaly.py (md5 bb375fc5), same cells, same means (12 topologies x 4 windows; FL gnn = mean of s1, s2).
Descriptive, no label.
(a) after 360 s: elapsed from each summary's steady_state.after_360s; lock wait from the cell's filed .tasks_bsz.npy/.tasks_lock.npy
    sidecars (same rows, same order; rows dispatched >= 360 s). Every other term is a whole-run mean in the summary and cannot be split.
(b) share of run time at the replica cap from replica_count_series (uniform 120-step grid over endTime, value = live replicas at the
    last one-second event <= the grid time): fraction of grid points 1..120 with total >= cap; whole run and points at t >= 360 s.
(c) paired FL gnn vs bar at each rung, per topology: mean, n/12 with gnn < bar, Wilcoxon."""
import json, os
import numpy as np
from scipy.stats import wilcoxon

U = "/share/nikola.lukic/uncut_groups"
SETS = {
    "24 servers (gate 19537-19548)": dict(topos=range(19537, 19549), bar=f"{U}/comb/cells_bar_g8", gnn=f"{U}/comb/cells_comb_fl",
                                         inp=f"{U}/comb/in_comball", gnn_arm="ra_gnn_eng_combfl_nosplit_k100", cap=120),
    "64 servers (30129-30140)": dict(topos=range(30129, 30141), bar=f"{U}/size64/cells_bar", gnn=f"{U}/size64/cells_learned",
                                     inp=f"{U}/size64/in_size64", gnn_arm="ra_gnn_eng_combfl_nosplit_k100", cap=319),
}
BAR = "cd_ib0_inflightfull_k100"
W = ("g0", "g1", "g2", "g3")
T = 360.0


def one(stem, rel, cap):
    s = json.load(open(stem + ".summary.json")); n = s["num_tasks"]
    cls = s["peerExchangeByAccessClass"]
    tr = sum(v["transfers"] for v in cls.values()); sec = sum(v["seconds"] for v in cls.values())
    cel_tr = sum(v["transfers"] for k, v in cls.items() if "cellular" in k)
    cel_sec = sum(v["seconds"] for k, v in cls.items() if "cellular" in k)
    bsz = np.load(stem + ".tasks_bsz.npy"); lk = np.load(stem + ".tasks_lock.npy")
    assert bsz.shape[0] == lk.shape[0] == n and np.array_equal(bsz[:, 0], lk[:, 0]), stem
    lock = lk[:, 3]
    assert int(np.isnan(lock).sum()) == int(s["lock_wait"]["n_unstamped"]), stem
    assert abs(np.nanmean(lock) - s["lock_wait"]["mean"]) < 1e-9 * max(1.0, s["lock_wait"]["mean"]), stem
    after = bsz[:, 1] >= T
    ss = s["steady_state"]["after_360s"]
    assert ss["T_s"] == T and int(ss["n"]) == int(after.sum()), stem
    rc = s["replica_count_series"]; t = np.asarray(rc["t"][1:]); tot = np.asarray(rc["total"][1:])
    assert len(t) == 120, stem
    return {"elapsed_whole": s["averageElapsedTime"], "elapsed_after360": ss["mean"],
            "L_lock_wait_whole": s["lock_wait"]["mean"], "L_lock_wait_after360": float(np.nanmean(lock[after])),
            "tasks_after360_share": float(after.mean()),
            "at_cap_share_whole": float((tot >= cap).mean()), "at_cap_share_after360": float((tot[t >= T] >= cap).mean()),
            "grid_step_s": float(t[1] - t[0]),
            "cellular_transfers_per_task": cel_tr / n, "cellular_share_of_exchange_s": cel_sec / sec if sec else 0.0,
            "coloc_share": 1 - tr / rel, "transfers_per_task": tr / n, "s_per_transfer": sec / tr if tr else 0.0}


def stats(a, b):
    d = a - b
    try:
        p = float(wilcoxon(a, b).pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        p = float("nan")
    return d.mean(), int((d < 0).sum()), p


def main():
    out = [__doc__.strip(), ""]
    for name, S in SETS.items():
        per = {}
        for arm, root, suf in (("bar", S["bar"], [f"{BAR}_s0"]), ("gnn", S["gnn"], [f"{S['gnn_arm']}_s1", f"{S['gnn_arm']}_s2"])):
            for r in ("moderate", "heavy"):
                rel = {w: 2 * len(json.load(open(f"{S['inp']}/wf1_{r}/wl/grounded_{w}_n50000.json"))["peer_exchange"]) for w in W}
                rows_t = []
                for t in S["topos"]:
                    rows = []
                    for w in W:
                        ds = [one(f"{root}/cc40s{t}__{w}{r}__{x}", rel[w], S["cap"]) for x in suf]
                        rows.append({k: float(np.mean([x[k] for x in ds])) for k in ds[0]})
                    rows_t.append({k: float(np.mean([x[k] for x in rows])) for k in rows[0]})
                per[(arm, r)] = rows_t
        col = lambda arm, r, k: np.array([x[k] for x in per[(arm, r)]])
        hdr = f"    {'term':30s} {'moderate':>10s} {'heavy':>10s} {'delta':>10s} {'n/12':>5s} {'wilcoxon p':>11s}"
        for arm in ("bar", "gnn"):
            out.append(f"=== (a)+(b) {name}, {arm}: delta = heavy - moderate; n/12 = topologies with heavy < moderate")
            out.append(hdr)
            for k in ("elapsed_whole", "elapsed_after360", "L_lock_wait_whole", "L_lock_wait_after360", "tasks_after360_share",
                      "at_cap_share_whole", "at_cap_share_after360", "grid_step_s"):
                m, h = col(arm, "moderate", k), col(arm, "heavy", k)
                d, nn, p = stats(h, m)
                out.append(f"    {k:30s} {m.mean():10.4f} {h.mean():10.4f} {d:+10.4f} {nn:5d} {p:11.4f}")
            out.append("    not splittable at 360 s from filed data: W_sched_wait, Q_resid, N_prop, P_pull, K_cold, R_rendezvous, X_exchange,"
                       " E_exec, transfers/task, s/transfer, cellular terms, co-location share, scale-ups, cold_start_pct (whole-run"
                       " aggregates in the summary; no per-task column in the sidecars)")
        out.append(f"=== (c) {name}: paired FL gnn vs bar per rung; delta = gnn - bar; n/12 = topologies with gnn < bar")
        out.append(f"    {'rung / term':40s} {'bar':>10s} {'gnn':>10s} {'delta':>10s} {'n/12':>5s} {'wilcoxon p':>11s}")
        for r in ("moderate", "heavy"):
            for k in ("cellular_transfers_per_task", "cellular_share_of_exchange_s", "coloc_share", "transfers_per_task", "s_per_transfer"):
                b, g = col("bar", r, k), col("gnn", r, k)
                d, nn, p = stats(g, b)
                out.append(f"    {r + ' ' + k:40s} {b.mean():10.4f} {g.mean():10.4f} {d:+10.4f} {nn:5d} {p:11.4f}")
        out.append("")
    txt = "\n".join(out); print(txt)
    od = f"{U}/anomaly"; os.makedirs(od, exist_ok=True)
    dst = f"{od}/load_anomaly_ext.txt"
    assert not os.path.exists(dst), "written once"
    open(dst + ".partial", "w").write(txt + "\n"); os.replace(dst + ".partial", dst)


main()
