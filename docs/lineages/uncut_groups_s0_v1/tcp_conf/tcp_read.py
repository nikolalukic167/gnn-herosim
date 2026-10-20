#!/usr/bin/env python3
"""TCP-transfer sensitivity read (uncut_groups_s0_v1, HEROSIM_EXCHANGE_TCP; SCIENTIST's prediction and reading rules (a)-(c)
filed before any cell). Descriptive, no label.
Switch on  = anomaly/cells_tcp_{ref,gnn}_{mod,heavy} at pin e8b3f53c (anomaly/tcp_ident is never opened).
Switch off = S4's 0.7 arms at pin 7cb483c5, anomaly/dose_7cb483c5/cells_{ref_mod070,ref_heavy070,gnn_mod070,gnn_heavy070}.
Latency keys only: averageElapsedTime, steady_state.after_360s.mean and the per-term fields; never total_rtt_plus_inference.
Per topology: mean over g0-g3 (FL gnn: and over seeds 1-2). Wilcoxon over the 12 topologies, paired, two-sided, p unadjusted.
  R1  paired FL-vs-bar % (100 (gnn - bar) / bar, per topology), median over 12, per rung x {whole, after-360}: off, on, shift pp
  R2  the bar's heavy - moderate gap per load_anomaly term, off and on, and the change on - off of that gap
  R3  per arm x rung, off -> on: exchange s per task, transfers per task, s per transfer, lock wait, mean replicas, cold start
  Rules (a)-(c) applied to R1 as filed. SCIENTIST's prediction is printed beside each row; nothing here is a label."""
import hashlib, json, os
import numpy as np
from scipy.stats import wilcoxon

U = "/share/nikola.lukic/uncut_groups"; A = f"{U}/anomaly"; IN = f"{U}/comb/in_comball"
TOPOS = range(19537, 19549); W = ("g0", "g1", "g2", "g3")
BAR, GNN = "cd_ib0_inflightfull_k100", "ra_gnn_eng_combfl_nosplit_k100"
PIN = {"on": "e8b3f53c8b2e1e3d8d5a586064fc4f15b39b2e50", "off": "7cb483c5fd4f7a02fd76fd00cf6edb3abf2ac5b1"}
ROOT = {("off", "bar", "moderate"): "dose_7cb483c5/cells_ref_mod070", ("off", "bar", "heavy"): "dose_7cb483c5/cells_ref_heavy070",
        ("off", "gnn", "moderate"): "dose_7cb483c5/cells_gnn_mod070", ("off", "gnn", "heavy"): "dose_7cb483c5/cells_gnn_heavy070",
        ("on", "bar", "moderate"): "cells_tcp_ref_mod", ("on", "bar", "heavy"): "cells_tcp_ref_heavy",
        ("on", "gnn", "moderate"): "cells_tcp_gnn_mod", ("on", "gnn", "heavy"): "cells_tcp_gnn_heavy"}
TCP_BLOCK = {"model": "slow_start_v1", "mss": 1448, "frame": 1514, "initcwnd_segments": 10, "per_partner": True}
T360 = 360.0
CKPT = json.load(open(f"{U}/comb/md5_comball.json"))
_MD5 = {}
NCELLS = {}


def ckpt_ok(s, seed, stem):
    path = s["env"]["GNN_MODEL_PATH"]
    assert path == f"{IN}/models/uncut-comb-fl-gnn-eng-seed{seed[1:]}.pt", f"FAIL LOUD: model path {stem}: {path}"
    if path not in _MD5:
        _MD5[path] = hashlib.md5(open(path, "rb").read()).hexdigest()
    assert _MD5[path] == CKPT[f"ra_gnn_eng_combfl_nosplit|{seed}"], f"FAIL LOUD: checkpoint md5 {stem}"


def one(stem, rel, sw, seed):
    assert "tcp_ident" not in stem
    assert not os.path.exists(stem + ".failed.json"), f"FAIL LOUD: failed cell {stem}"
    s = json.load(open(stem + ".summary.json")); n = s["num_tasks"]
    if seed != "s0":
        ckpt_ok(s, seed, stem)
    assert s["code"]["commit"] == PIN[sw] and not s["code"]["dirty"], f"FAIL LOUD: pin {stem}"
    if sw == "on":
        b = s.get("peerExchangeTcp")
        assert b is not None and all(b[k] == v for k, v in TCP_BLOCK.items()) and b["transfers"] > 0, f"FAIL LOUD: TCP block {stem}"
    else:
        assert "peerExchangeTcp" not in s, f"FAIL LOUD: TCP block in a switch-off cell {stem}"
    env = s["env"]
    bad = [k for k in env if k in ("HEROSIM_SNAPSHOT_FIDELITY", "HEROSIM_PG_ORACLE", "HEROSIM_KPA_KNATIVE_PANIC") or k.startswith("LIVE_AUDIT_")]
    assert not bad, f"FAIL LOUD: env {stem}: {bad}"
    assert "knative_panic" not in s["scaleOut"] and s["scaleOut"]["target"] == 0.7, f"FAIL LOUD: scaleOut {stem}"
    q, N, P = s["averageQueueTime"], s["averageNetworkLatency"], s["averagePullTime"]
    cls = s["peerExchangeByAccessClass"]
    tr = sum(v["transfers"] for v in cls.values()); sec = sum(v["seconds"] for v in cls.values())
    bsz = np.load(stem + ".tasks_bsz.npy"); lk = np.load(stem + ".tasks_lock.npy")
    assert bsz.shape[0] == lk.shape[0] == n and np.array_equal(bsz[:, 0], lk[:, 0])
    after = bsz[:, 1] >= T360
    NCELLS[(sw, stem.split("/")[-2])] = NCELLS.get((sw, stem.split("/")[-2]), 0) + 1
    return {"elapsed": s["averageElapsedTime"], "elapsed_after360": s["steady_state"]["after_360s"]["mean"],
            "W_sched_wait": s["averageWaitTime"], "Q_resid": q - N - P, "N_prop": N, "P_pull": P,
            "K_cold": s["averageColdStartTime"], "R_rendezvous": s["totalPeerRendezvousWait"] / n,
            "X_exchange": s["totalPeerExchangeTime"] / n, "L_lock_wait": s["lock_wait"]["mean"],
            "L_lock_wait_after360": float(np.nanmean(lk[after, 3])), "E_exec": s["averageExecutionTime"],
            "transfers_per_task": tr / n, "s_per_transfer": sec / tr if tr else 0.0, "coloc_share": 1 - tr / rel,
            "replica_time_mean": s["replica_count_series"]["time_mean"], "cold_start_pct": s["cold_start_pct"],
            "tcp_extra_s_per_task": (s["peerExchangeTcp"]["extra_seconds"] / n) if sw == "on" else 0.0}


def load(sw, arm, rung):
    rel = {w: 2 * len(json.load(open(f"{IN}/wf1_{rung}/wl/grounded_{w}_n50000.json"))["peer_exchange"]) for w in W}
    seeds = ["s0"] if arm == "bar" else ["s1", "s2"]; kind = BAR if arm == "bar" else GNN
    root = f"{A}/{ROOT[(sw, arm, rung)]}"
    per = []
    for t in TOPOS:
        rows = []
        for w in W:
            ds = [one(f"{root}/cc40s{t}__{w}{rung}__{kind}_{sd}", rel[w], sw, sd) for sd in seeds]
            rows.append({k: float(np.mean([d[k] for d in ds])) for k in ds[0]})
        per.append({k: float(np.mean([r[k] for r in rows])) for k in rows[0]})
    return per


def wx(a, b):
    d = np.asarray(a) - np.asarray(b)
    try:
        return float(wilcoxon(a, b).pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        return float("nan")


def main():
    out = [__doc__.strip(), ""]
    data = {(sw, arm, r): load(sw, arm, r) for sw in ("off", "on") for arm in ("bar", "gnn") for r in ("moderate", "heavy")}
    out.append(f"=== cells read (all asserted): {sorted(NCELLS.items())}")
    col = lambda sw, arm, r, k: np.array([x[k] for x in data[(sw, arm, r)]])
    # R1
    pred = {("moderate", "whole"): "toward the bar by 3-6 pp, sign not predicted", ("heavy", "whole"): "toward the bar by 2-5 pp",
            ("moderate", "after-360"): "same direction as heavy after-360, size not predicted",
            ("heavy", "after-360"): "same direction as moderate after-360, size not predicted"}
    out.append("=== R1 paired FL-vs-bar %, median over 12 topologies (n/12 = FL faster); shift = median(on) - median(off) in pp (positive = toward the bar)")
    r1 = {}
    for r in ("moderate", "heavy"):
        for k, lab in (("elapsed", "whole"), ("elapsed_after360", "after-360")):
            po = 100 * (col("off", "gnn", r, k) - col("off", "bar", r, k)) / col("off", "bar", r, k)
            pn = 100 * (col("on", "gnn", r, k) - col("on", "bar", r, k)) / col("on", "bar", r, k)
            r1[(r, lab)] = (float(np.median(po)), float(np.median(pn)))
            out.append(f"    {r:8s} {lab:9s} off {np.median(po):+7.2f} % ({int((po < 0).sum())}/12)  on {np.median(pn):+7.2f} % ({int((pn < 0).sum())}/12)"
                       f"  shift {np.median(pn) - np.median(po):+6.2f} pp  per-topology on-off p {wx(pn, po):.4f}  | predicted: {pred[(r, lab)]}")
    # rules
    mo = r1[("moderate", "whole")]
    shifts = {k: v[1] - v[0] for k, v in r1.items()}
    flips = [f"{k[0]} {k[1]}" for k, v in r1.items() if np.sign(v[0]) != np.sign(v[1])]
    out.append("=== Reading rules (filed before any cell), applied to R1")
    out.append(f"    (a) moderate whole under the switch {mo[1]:+.2f} % {'>' if mo[1] > -2 else '<='} -2 % -> {'FIRES' if mo[1] > -2 else 'does not fire'}")
    out.append(f"    (b) moderate whole sign off {mo[0]:+.2f} -> on {mo[1]:+.2f} -> {'FIRES (sign flips)' if np.sign(mo[0]) != np.sign(mo[1]) else 'does not fire'};"
               f" sign flips on any test: {flips or 'none'}")
    big = {f"{k[0]} {k[1]}": round(v, 2) for k, v in shifts.items() if abs(v) >= 1.5}
    out.append(f"    (c) |shift| < 1.5 pp on every test -> {'FIRES' if not big else 'does not fire'}; tests at >= 1.5 pp: {big or 'none'}")
    # R2
    out.append("=== R2 bar heavy - moderate (mean over 12 topologies); n/12 = topologies with heavy < moderate; change = gap(on) - gap(off)")
    out.append("    predicted: gap widens (more negative) by 0.15-0.45 s whole; moderate has more transfers per task")
    out.append(f"    {'term':24s} {'gap off':>9s} {'n/12':>4s} {'gap on':>9s} {'n/12':>4s} {'change':>9s} {'p':>7s} | {'mod on-off':>10s} {'p':>7s} {'heavy on-off':>12s} {'p':>7s}")
    for k in data[("off", "bar", "moderate")][0]:
        go = col("off", "bar", "heavy", k) - col("off", "bar", "moderate", k)
        gn = col("on", "bar", "heavy", k) - col("on", "bar", "moderate", k)
        dm = col("on", "bar", "moderate", k) - col("off", "bar", "moderate", k)
        dh = col("on", "bar", "heavy", k) - col("off", "bar", "heavy", k)
        out.append(f"    {k:24s} {go.mean():+9.4f} {int((go < 0).sum()):4d} {gn.mean():+9.4f} {int((gn < 0).sum()):4d} {(gn - go).mean():+9.4f} {wx(gn, go):7.4f} | "
                   f"{dm.mean():+10.4f} {wx(col('on', 'bar', 'moderate', k), col('off', 'bar', 'moderate', k)):7.4f} {dh.mean():+12.4f} "
                   f"{wx(col('on', 'bar', 'heavy', k), col('off', 'bar', 'heavy', k)):7.4f}")
    # R3
    out.append("=== R3 per arm x rung, off -> on (mean over 12 topologies), change, % change, n/12 rising, p")
    out.append("    predicted: exchange per task rises for every arm at both rungs; lock wait and mean replicas rise")
    for k in ("elapsed", "elapsed_after360", "X_exchange", "transfers_per_task", "s_per_transfer", "coloc_share", "L_lock_wait",
              "L_lock_wait_after360", "replica_time_mean", "K_cold", "cold_start_pct", "tcp_extra_s_per_task"):
        out.append(f"  -- {k}")
        for arm in ("bar", "gnn"):
            for r in ("moderate", "heavy"):
                o, n = col("off", arm, r, k), col("on", arm, r, k)
                pc = 100 * (n.mean() - o.mean()) / o.mean() if o.mean() else float("nan")
                out.append(f"    {arm:3s} {r:8s} {o.mean():10.4f} -> {n.mean():10.4f}  {n.mean() - o.mean():+9.4f}  {pc:+7.2f} %  {int((n > o).sum())}/12  p {wx(n, o):.4f}")
    out.append(f"=== checkpoints asserted per learned cell (both switch states): {_MD5}")
    txt = "\n".join(out); print(txt)
    dst = f"{A}/tcp_read.txt"
    assert not os.path.exists(dst), "written once"
    open(dst + ".partial", "w").write(txt + "\n"); os.replace(dst + ".partial", dst)


main()
