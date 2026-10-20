#!/usr/bin/env python3
"""KPA closed-loop sensitivity read (uncut_groups_s0_v1, prediction line filed before the cells). Descriptive, no label.
Switch off = S4's 0.7 arms at the shared pin 90b6afe5 (anomaly/cells_{ref,gnn}_{mod,heavy}070); switch on = anomaly/cells_knpanic_*.
Per topology: mean over g0-g3 (FL gnn: and over seeds 1-2). Wilcoxon over the 12 topologies, paired, two-sided, p unadjusted.
  R0  the smoke cell (kpa_conf/smoke_65db9c82_on) against its grid twin, field for field (provenance / wall clock aside) + arrays
  R1  bar heavy - moderate per load_anomaly term, off and on, and the change on - off of that gap; elapsed and lock after 360 s
  R2  paired FL-vs-bar % (100 (gnn - bar) / bar, per topology), median over 12, per rung x {whole, after-360}: off, on, shift
  R3  replica_count_series.time_mean per arm x rung: off, on, % change
  R4  scaleOut.panic_ticks per arm x rung: off, on
The prediction (SCIENTIST's, verbatim in the line) is printed beside the measured value; nothing here is a label."""
import hashlib, json, os, sys
import numpy as np
from scipy.stats import wilcoxon

U = "/share/nikola.lukic/uncut_groups"; A = f"{U}/anomaly"; IN = f"{U}/comb/in_comball"
TOPOS = range(19537, 19549); W = ("g0", "g1", "g2", "g3")
BAR, GNN = "cd_ib0_inflightfull_k100", "ra_gnn_eng_combfl_nosplit_k100"
ROOT = {("off", "bar", "moderate"): "cells_ref_mod070", ("off", "bar", "heavy"): "cells_ref_heavy070",
        ("off", "gnn", "moderate"): "cells_gnn_mod070", ("off", "gnn", "heavy"): "cells_gnn_heavy070",
        ("on", "bar", "moderate"): "cells_knpanic_ref_mod", ("on", "bar", "heavy"): "cells_knpanic_ref_heavy",
        ("on", "gnn", "moderate"): "cells_knpanic_gnn_mod", ("on", "gnn", "heavy"): "cells_knpanic_gnn_heavy"}
VOL = ("wallclock_s", "code", "python_env", "host", "env_fingerprint", "decisionTiming", "total_rtt_plus_inference")
T360 = 360.0
CKPT = json.load(open(f"{U}/comb/md5_comball.json"))
_MD5 = {}


def ckpt_ok(s, seed, stem):
    """As the frozen readers: the learned cell's served checkpoint (env.GNN_MODEL_PATH) has the filed md5 for its seed."""
    path = s["env"]["GNN_MODEL_PATH"]
    assert path == f"{IN}/models/uncut-comb-fl-gnn-eng-seed{seed[1:]}.pt", f"FAIL LOUD: model path {stem}: {path}"
    if path not in _MD5:
        _MD5[path] = hashlib.md5(open(path, "rb").read()).hexdigest()
    assert _MD5[path] == CKPT[f"ra_gnn_eng_combfl_nosplit|{seed}"], f"FAIL LOUD: checkpoint md5 {stem}"


def one(stem, rel, want_on, seed):
    s = json.load(open(stem + ".summary.json")); n = s["num_tasks"]
    if seed != "s0":
        ckpt_ok(s, seed, stem)
    assert ("knative_panic" in s["scaleOut"]) == want_on, f"FAIL LOUD: switch state {stem}"
    assert s["code"]["commit"].startswith("90b6afe5") and not s["code"]["dirty"], f"FAIL LOUD: pin {stem}"
    q, N, P = s["averageQueueTime"], s["averageNetworkLatency"], s["averagePullTime"]
    cls = s["peerExchangeByAccessClass"]
    tr = sum(v["transfers"] for v in cls.values()); sec = sum(v["seconds"] for v in cls.values())
    bsz = np.load(stem + ".tasks_bsz.npy"); lk = np.load(stem + ".tasks_lock.npy")
    assert bsz.shape[0] == lk.shape[0] == n and np.array_equal(bsz[:, 0], lk[:, 0])
    after = bsz[:, 1] >= T360
    so = s["scaleOut"]["scale_ups_by_cause"]
    return {"elapsed": s["averageElapsedTime"], "elapsed_after360": s["steady_state"]["after_360s"]["mean"],
            "W_sched_wait": s["averageWaitTime"], "Q_resid": q - N - P, "N_prop": N, "P_pull": P,
            "K_cold": s["averageColdStartTime"], "R_rendezvous": s["totalPeerRendezvousWait"] / n,
            "X_exchange": s["totalPeerExchangeTime"] / n, "L_lock_wait": s["lock_wait"]["mean"],
            "L_lock_wait_after360": float(np.nanmean(lk[after, 3])), "E_exec": s["averageExecutionTime"],
            "transfers_per_task": tr / n, "s_per_transfer": sec / tr if tr else 0.0, "coloc_share": 1 - tr / rel,
            "replica_time_mean": s["replica_count_series"]["time_mean"], "panic_ticks": s["scaleOut"]["panic_ticks"],
            "panic_entries": s["scaleOut"]["panic_entries"], "scale_ups_load": so.get("load", 0),
            "scale_ups_reach": so.get("reachability", 0), "cold_start_pct": s["cold_start_pct"]}


def load(sw, arm, rung):
    rel = {w: 2 * len(json.load(open(f"{IN}/wf1_{rung}/wl/grounded_{w}_n50000.json"))["peer_exchange"]) for w in W}
    seeds = ["s0"] if arm == "bar" else ["s1", "s2"]; kind = BAR if arm == "bar" else GNN
    root = f"{A}/{ROOT[(sw, arm, rung)]}"
    per = []
    for t in TOPOS:
        rows = []
        for w in W:
            ds = [one(f"{root}/cc40s{t}__{w}{rung}__{kind}_{sd}", rel[w], sw == "on", sd) for sd in seeds]
            rows.append({k: float(np.mean([d[k] for d in ds])) for k in ds[0]})
        per.append({k: float(np.mean([r[k] for r in rows])) for k in rows[0]})
    return per


def wx(a, b):
    d = np.asarray(a) - np.asarray(b)
    try:
        return float(wilcoxon(a, b).pvalue) if np.any(d != 0) else 1.0
    except ValueError:
        return float("nan")


def walk(a, b, p, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(p + [k])
            else:
                walk(a[k], b[k], p + [k], out)
    elif a != b:
        out.append(p)


def main():
    out = [__doc__.strip(), ""]
    # R0
    sm = f"{U}/kpa_conf/smoke_65db9c82_on/cc40s19537__g0heavy__{BAR}_s0"; tw = f"{A}/cells_knpanic_ref_heavy/cc40s19537__g0heavy__{BAR}_s0"
    d = []; walk(json.load(open(sm + ".summary.json")), json.load(open(tw + ".summary.json")), [], d)
    real = [".".join(map(str, x)) for x in d if x[0] not in VOL]
    arr = [hashlib.md5(open(sm + e, "rb").read()).hexdigest() == hashlib.md5(open(tw + e, "rb").read()).hexdigest()
           for e in (".tasks.npy", ".tasks_bsz.npy", ".tasks_lock.npy")]
    out.append(f"=== R0 smoke cell vs grid twin: non-volatile differing fields {len(real)} {real[:10]}; arrays equal {arr}")
    data = {(sw, arm, r): load(sw, arm, r) for sw in ("off", "on") for arm in ("bar", "gnn") for r in ("moderate", "heavy")}
    col = lambda sw, arm, r, k: np.array([x[k] for x in data[(sw, arm, r)]])
    # R1
    out.append("=== R1 bar heavy - moderate (mean over 12 topologies); n/12 = topologies with heavy < moderate; change = gap(on) - gap(off)")
    out.append(f"    {'term':24s} {'gap off':>9s} {'n/12':>4s} {'gap on':>9s} {'n/12':>4s} {'change':>9s} {'p':>7s} | {'mod on-off':>10s} {'p':>7s} {'heavy on-off':>12s} {'p':>7s}")
    for k in data[("off", "bar", "moderate")][0]:
        go = col("off", "bar", "heavy", k) - col("off", "bar", "moderate", k)
        gn = col("on", "bar", "heavy", k) - col("on", "bar", "moderate", k)
        dm = col("on", "bar", "moderate", k) - col("off", "bar", "moderate", k)
        dh = col("on", "bar", "heavy", k) - col("off", "bar", "heavy", k)
        out.append(f"    {k:24s} {go.mean():+9.4f} {int((go < 0).sum()):4d} {gn.mean():+9.4f} {int((gn < 0).sum()):4d} {(gn - go).mean():+9.4f} {wx(gn, go):7.4f} | "
                   f"{dm.mean():+10.4f} {wx(col('on', 'bar', 'moderate', k), col('off', 'bar', 'moderate', k)):7.4f} {dh.mean():+12.4f} "
                   f"{wx(col('on', 'bar', 'heavy', k), col('off', 'bar', 'heavy', k)):7.4f}")
    # R2
    out.append("=== R2 paired FL-vs-bar %, median over 12 topologies (n/12 = FL faster); shift = median(on) - median(off) in pp")
    for r in ("moderate", "heavy"):
        for k, lab in (("elapsed", "whole"), ("elapsed_after360", "after-360")):
            po = 100 * (col("off", "gnn", r, k) - col("off", "bar", r, k)) / col("off", "bar", r, k)
            pn = 100 * (col("on", "gnn", r, k) - col("on", "bar", r, k)) / col("on", "bar", r, k)
            out.append(f"    {r:8s} {lab:9s} off {np.median(po):+7.2f} % ({int((po < 0).sum())}/12)  on {np.median(pn):+7.2f} % ({int((pn < 0).sum())}/12)"
                       f"  shift {np.median(pn) - np.median(po):+6.2f} pp  per-topology on-off p {wx(pn, po):.4f}")
    # R3, R4
    for lab, k in (("R3 mean replicas (time mean)", "replica_time_mean"), ("R4 panic ticks", "panic_ticks"), ("    panic entries", "panic_entries")):
        out.append(f"=== {lab}: off -> on (mean over 12 topologies), % change, n/12 rising, p")
        for arm in ("bar", "gnn"):
            for r in ("moderate", "heavy"):
                o, n = col("off", arm, r, k), col("on", arm, r, k)
                out.append(f"    {arm:3s} {r:8s} {o.mean():10.2f} -> {n.mean():10.2f}  {100 * (n.mean() - o.mean()) / o.mean():+6.2f} %  {int((n > o).sum())}/12  p {wx(n, o):.4f}")
    out.append(f"=== checkpoints asserted per learned cell (both switch states): {_MD5}")
    txt = "\n".join(out); print(txt)
    dst = f"{A}/knpanic_read.txt"
    assert not os.path.exists(dst), "written once"
    open(dst + ".partial", "w").write(txt + "\n"); os.replace(dst + ".partial", dst)


main()
