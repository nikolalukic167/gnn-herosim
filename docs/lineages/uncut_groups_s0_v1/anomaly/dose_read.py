#!/usr/bin/env python3
"""Dose read (S4), replica-count counterfactual. Descriptive, no label. Latency keys only (averageElapsedTime and the per-term fields,
steady_state.after_360s, the cells' lock sidecars); never total_rtt_plus_inference (includes measured decision wall time).
Arms (anomaly/dose_7cb483c5/cells_<arm>): ref_mod0294, ref_mod045, ref_mod070 (reference, moderate, KPA target 0.294/0.45/0.7),
ref_heavy070 (reference, heavy), gnn_mod070, gnn_heavy070 (FL gnn s1+s2, mean of the two seeds per cell).
Every value is the mean over 12 topologies (19537-19548) x 4 windows (g0-g3) of the per-cell value; the topology mean is formed first.
Checks before any number: pin (code.commit, clean), per-cell scaleOut.target equals the arm's registered value, cell counts, no failed cells,
and for the gnn cells the served checkpoint md5 (env.GNN_MODEL_PATH, hashed here) equals comb/md5_comball.json s1/s2.
Prediction (SCIENTIST, verbatim): "mean replicas rise toward 56; co-location share toward 0.78; exchange toward 0.89 s and cold start
toward 0.04 s, monotone in dose; lock wait does not rise; if exchange does not move at matched replica counts the replica reading is wrong
and rate itself (group overlap in time) is the remaining candidate". Rows are compared against the heavy reference (the target of the
dose), monotone = the three moderate arms ordered 0.7 -> 0.45 -> 0.294 move one way and toward heavy-0.7 at every step.
peerExchangeOverlap shares are approximate (intervals start at env.now plus the earlier peer terms of the task, include route latency,
sender out-link only)."""
import glob, hashlib, json, os
import numpy as np

U = "/share/nikola.lukic/uncut_groups"
D = f"{U}/anomaly/dose_7cb483c5"
PIN = "7cb483c5fd4f7a02fd76fd00cf6edb3abf2ac5b1"
INP = f"{U}/comb/in_comball"
GNN = "ra_gnn_eng_combfl_nosplit_k100"
BAR = "cd_ib0_inflightfull_k100"
CKPT = json.load(open(f"{U}/comb/md5_comball.json"))
TOPOS = range(19537, 19549)
W = ("g0", "g1", "g2", "g3")
T = 360.0
CAP = 120
ARMS = [  # name, root, rung, kind, seeds, registered target
    ("ref_mod0294", "cells_ref_mod0294", "moderate", BAR, ("s0",), 0.294),
    ("ref_mod045", "cells_ref_mod045", "moderate", BAR, ("s0",), 0.45),
    ("ref_mod070", "cells_ref_mod070", "moderate", BAR, ("s0",), 0.7),
    ("ref_heavy070", "cells_ref_heavy070", "heavy", BAR, ("s0",), 0.7),
    ("gnn_mod070", "cells_gnn_mod070", "moderate", GNN, ("s1", "s2"), 0.7),
    ("gnn_heavy070", "cells_gnn_heavy070", "heavy", GNN, ("s1", "s2"), 0.7),
]
EXPECT = {"ref_mod0294": 48, "ref_mod045": 48, "ref_mod070": 48, "ref_heavy070": 48, "gnn_mod070": 96, "gnn_heavy070": 96}
CLASSES = ["wired+wired", "wifi+wired", "wifi+wifi", "cellular+wired", "cellular+wifi", "cellular+cellular"]
_md5 = {}


def md5_of(path):
    if path not in _md5:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(1 << 22), b""):
                h.update(b)
        _md5[path] = h.hexdigest()
    return _md5[path]


def one(stem, rel, arm, seed, target):
    s = json.load(open(stem + ".summary.json")); n = s["num_tasks"]
    assert s["code"]["commit"] == PIN and not s["code"]["dirty"], (stem, s["code"]["commit"])
    assert abs(s["scaleOut"]["target"] - target) < 1e-12 and s["scaleOut"]["mode"] == "kpa", (stem, s["scaleOut"])
    if arm.startswith("gnn"):
        want = CKPT[f"ra_gnn_eng_combfl_nosplit|{seed}"]
        assert md5_of(s["env"]["GNN_MODEL_PATH"]) == want, (stem, s["env"]["GNN_MODEL_PATH"])
    cls = s["peerExchangeByAccessClass"]
    tr = sum(v["transfers"] for v in cls.values()); sec = sum(v["seconds"] for v in cls.values())
    cel = {k: v for k, v in cls.items() if "cellular" in k}
    cel_tr = sum(v["transfers"] for v in cel.values()); cel_sec = sum(v["seconds"] for v in cel.values())
    N, P = s["averageNetworkLatency"], s["averagePullTime"]
    bsz = np.load(stem + ".tasks_bsz.npy"); lk = np.load(stem + ".tasks_lock.npy")
    assert bsz.shape[0] == lk.shape[0] == n and np.array_equal(bsz[:, 0], lk[:, 0]), stem
    lock = lk[:, 3]
    assert abs(np.nanmean(lock) - s["lock_wait"]["mean"]) < 1e-9 * max(1.0, s["lock_wait"]["mean"]), stem
    after = bsz[:, 1] >= T
    ss = s["steady_state"]["after_360s"]
    assert ss["T_s"] == T and int(ss["n"]) == int(after.sum()), stem
    rc = s["replica_count_series"]; t = np.asarray(rc["t"][1:]); tot = np.asarray(rc["total"][1:])
    ov = s["peerExchangeOverlap"]
    d = {"elapsed_whole": s["averageElapsedTime"], "elapsed_after360": ss["mean"],
         "X_exchange_s_per_task": s["totalPeerExchangeTime"] / n, "transfers_per_task": tr / n, "s_per_transfer": sec / tr,
         "coloc_share": 1 - tr / rel, "replica_time_mean": rc["time_mean"], "at_cap_share_whole": float((tot >= CAP).mean()),
         "at_cap_share_after360": float((tot[t >= T] >= CAP).mean()),
         "K_cold_s": s["averageColdStartTime"], "cold_start_pct": s["cold_start_pct"],
         "L_lock_wait_whole": s["lock_wait"]["mean"], "L_lock_wait_after360": float(np.nanmean(lock[after])),
         "W_sched_wait": s["averageWaitTime"], "Q_resid": s["averageQueueTime"] - N - P, "R_rendezvous": s["totalPeerRendezvousWait"] / n,
         "E_exec": s["averageExecutionTime"], "N_prop": N, "P_pull": P,
         "cellular_transfers_per_task": cel_tr / n, "cellular_share_of_exchange_s": cel_sec / sec if sec else 0.0,
         "s_per_cellular_transfer": cel_sec / cel_tr if cel_tr else 0.0,
         "overlap_seconds_share": ov["overlap_seconds_share"], "transfers_with_overlap_share": ov["transfers_with_overlap_share"],
         "union_over_sum": ov["union_over_sum"]}
    for c in CLASSES:
        v = cls.get(c)
        d["MB_per_transfer " + c] = (v["bytes"] / v["transfers"] / 1e6) if v and v["transfers"] else float("nan")
    d["MB_per_transfer all"] = sum(v["bytes"] for v in cls.values()) / tr / 1e6
    return d


def main():
    out = [__doc__.strip(), ""]
    per = {}
    for arm, root, rung, kind, seeds, target in ARMS:
        S = sorted(glob.glob(f"{D}/{root}/*.summary.json")); F = glob.glob(f"{D}/{root}/*.failed.json")
        assert len(S) == EXPECT[arm] and not F, (arm, len(S), len(F))
        rel = {w: 2 * len(json.load(open(f"{INP}/wf1_{rung}/wl/grounded_{w}_n50000.json"))["peer_exchange"]) for w in W}
        rows_t = []
        for t in TOPOS:
            rows = []
            for w in W:
                ds = [one(f"{D}/{root}/cc40s{t}__{w}{rung}__{kind}_{sd}", rel[w], arm, sd, target) for sd in seeds]
                rows.append({k: float(np.nanmean([x[k] for x in ds])) for k in ds[0]})
            rows_t.append({k: float(np.nanmean([x[k] for x in rows])) for k in rows[0]})
        per[arm] = rows_t
        out.append(f"checked {arm}: {len(S)}/{EXPECT[arm]} summaries, 0 failed, pin {PIN[:8]} clean, scaleOut mode kpa target {target}"
                   + (", checkpoint md5s equal comb/md5_comball.json" if arm.startswith("gnn") else ""))
    out.append("")
    keys = list(per["ref_heavy070"][0])
    col = lambda a, k: np.array([x[k] for x in per[a]])
    ref = ["ref_mod070", "ref_mod045", "ref_mod0294"]
    out.append("=== Reference dose series (moderate), mean over 12 topologies x 4 windows. dose order 0.7 -> 0.45 -> 0.294; heavy-0.7 is the target.")
    out.append(f"    {'term':36s} {'mod 0.7':>10s} {'mod 0.45':>10s} {'mod 0.294':>10s} {'heavy 0.7':>10s} {'monotone':>9s} {'n/12 mono':>10s}")
    for k in keys:
        m = [col(a, k).mean() for a in ref]; h = col("ref_heavy070", k).mean()
        v = [col(a, k) for a in ref]
        toward = (np.sign(h - m[0]) != 0) and all(np.sign((m[i + 1] - m[i]) * (h - m[0])) > 0 for i in range(2))
        nmono = int(sum(all((v[i + 1][j] - v[i][j]) * (h - m[0]) > 0 for i in range(2)) for j in range(12)))
        out.append(f"    {k:36s} {m[0]:10.4f} {m[1]:10.4f} {m[2]:10.4f} {h:10.4f} {('yes' if toward else 'no'):>9s} {nmono:10d}")
    out.append("")
    out.append("=== FL gnn (s1+s2), 0.7: moderate vs heavy; delta = heavy - moderate")
    out.append(f"    {'term':36s} {'moderate':>10s} {'heavy':>10s} {'delta':>10s}")
    for k in keys:
        a, b = col("gnn_mod070", k).mean(), col("gnn_heavy070", k).mean()
        out.append(f"    {k:36s} {a:10.4f} {b:10.4f} {b - a:+10.4f}")
    out.append("")
    out.append("=== Prediction rows (SCIENTIST), reference series")
    P = [("replica_time_mean", "rises toward 56 (heavy value)"), ("coloc_share", "toward 0.78"), ("X_exchange_s_per_task", "toward 0.89 s"),
         ("K_cold_s", "toward 0.04 s"), ("L_lock_wait_whole", "does not rise"), ("L_lock_wait_after360", "does not rise")]
    for k, claim in P:
        m = [col(a, k).mean() for a in ref]; h = col("ref_heavy070", k).mean()
        out.append(f"    {k:26s} 0.7 {m[0]:.4f} -> 0.45 {m[1]:.4f} -> 0.294 {m[2]:.4f}; heavy {h:.4f}; predicted: {claim}")
    out.append("    fallback (exchange does not move at matched replica counts -> rate / group overlap in time is the remaining candidate): "
               "the exchange row above and the replica row are the two inputs; no label here.")
    txt = "\n".join(out); print(txt)
    dst = f"{U}/anomaly/dose_read.txt"
    assert not os.path.exists(dst), "written once"
    open(dst + ".partial", "w").write(txt + "\n"); os.replace(dst + ".partial", dst)


main()
