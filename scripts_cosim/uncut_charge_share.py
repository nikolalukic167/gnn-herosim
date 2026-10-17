#!/usr/bin/env python3
"""uncut_groups_s0_v1 calibration protocol (SMT, 2026-10-14): the hand q_n bound, the learned charge share and the H3 drift stop.

  uncut_charge_share.py q     --new <dir>[,<dir>] --arm cd_ib0 --bar <869828 dir> (--fit <fit.json> | --ub) [--k 1.468] [--w-file ...] [--out q.json]
  uncut_charge_share.py share --cells <dir>[,<dir>] --arm ra_gnn_eng_arm1_nosplit [--w-file <W1's w file>] [--out s.json]
  uncut_charge_share.py felt  --bar <869828 dir> --arm cd_ib0 (--fit <fit.json> | --ub) [--k 1.468] [--out f.json]
  uncut_charge_share.py drift --start <dir> --end <dir> --arm cd_ib0 [--out d.json]

q: per rung and served-batch bucket n, q_n = w_new_n / w_frozen (the frozen w is the arm's scalar at the rung, or its bucket rate where it
has one), w_new pooled over the re-measure's cells (sum total_s / sum plan_evals, every node's rep together). share_n = w_frozen_n x the
bar cells' plan evals in n / their summed task latency (num_tasks x mean), pooled over the arm's bar cells at the rung (suffix _k100).
bound = sum_n share_n x |q_n - 1|. KEEP if bound < 0.1 % of mean latency on both rungs, else REFREEZE (the 192 hand cells rerun).
Felt share (hand-F rule, 2026-10-14, filed before 871538): share_n above is the FELT share F_n, every task of a batch waiting the batch's
whole charge: F_n = w_n x E_n x R_n / latency, E_n the bar cells' charged plan evals in n, R_n = sum_b n_b e(n_b) / sum_b e(n_b) over the
cell's batches in n. --fit gives e(n) = n^beta per rung and bucket ({rung: {bucket: {"beta": b}}}, from the per-batch dump of the diagnostic cells; the batch-size counts per cell come
from .tasks_bsz.npy, tasks with servedBatchSize n / n); --ub sets R_n = the largest batch in n (10, 50, the cell's largest), the upper bound
that applies when the fit does not. The bound and the verdict use F; the pooled share and its bound are kept beside it (share_pooled,
bound_pooled). worst_cell: the largest per-cell sum_n F_n |q_n - 1|. --k: the co-tenancy hand-bias bound F x (1 - 1/k), pooled and worst cell.
felt: the hand F alone (per rung and bucket, pooled, median and worst cell; --k adds the hand-bias bound), no re-measure needed.
share: the arm's charge share per rung and bucket, w_n x plan evals in n / summed task latency, over its k = 0 calibration cells.
drift: pooled w of the same arm at the start and the end of a node's series; STOP (exit 1) if they differ by more than 2 % at a rung.
Every cell read (not the bar's, filed before the host block) must carry the gate's host block, in one CPU model and one SMT state.
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

BUCKETS = ("le10", "11_50", "gt50")
BOUND, DRIFT = 0.001, 0.02
W_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "decision_cost_w.json")
NAME = re.compile(r"^cc40s(\d+)__(g\d)(moderate|heavy)__(.+)_s(\d+)\.summary\.json$")


def cells(dirs, arm, host=True):
    out = []
    for d in [x for x in dirs.split(",") if x]:
        for p in sorted(glob.glob(os.path.join(d, "*.summary.json"))):
            m = NAME.match(os.path.basename(p))
            if m and m.group(4) == arm:
                s = json.load(open(p))
                s["_rung"], s["_path"] = m.group(3), p
                out.append(s)
    if not out:
        raise SystemExit(f"FAIL LOUD: no {arm} cells in {dirs}")
    if host:
        venue = set()
        for s in out:
            h = s.get("host")
            if not isinstance(h, dict) or not h.get("cpu_model") or not h.get("smt_control"):
                raise SystemExit(f"FAIL LOUD: {s['_path']} has no host block")
            venue.add((h["cpu_model"], h["smt_control"]))
        if len(venue) != 1:
            raise SystemExit(f"FAIL LOUD: {dirs} mixes CPU model / SMT states {sorted(venue)}")
    return out


def pooled_w(cs):
    """{rung: {"all": ms/plan, bucket: ms/plan}} over the cells."""
    acc = {}
    for s in cs:
        a = acc.setdefault(s["_rung"], {b: [0.0, 0] for b in ("all",) + BUCKETS})
        bb = (s.get("decisionTiming") or {}).get("by_bucket")
        if bb is None:
            raise SystemExit(f"FAIL LOUD: {s['_path']} has no decisionTiming.by_bucket")
        for b in BUCKETS:
            a[b][0] += bb[b]["total_s"]
            a[b][1] += bb[b]["plan_evals"]
        a["all"][0] += s["decisionTiming"]["total_s"]
        a["all"][1] += s["schedulerCounters"]["sched_plan_evals"]
    return {r: {b: (1000.0 * t / n if n else None) for b, (t, n) in a.items()} for r, a in acc.items()}


def evals_latency(cs):
    acc = {}
    for s in cs:
        a = acc.setdefault(s["_rung"], {"lat": 0.0, **{b: 0 for b in BUCKETS}})
        a["lat"] += float(s["averageElapsedTime"]) * int(s["num_tasks"])
        for b in BUCKETS:
            a[b] += s["decisionTiming"]["by_bucket"][b]["plan_evals"]
    return acc


def frozen(w, arm, rung):
    byb = (w.get("w_ms_per_plan_by_bucket", {}).get(arm) or {}).get(rung)
    if byb is not None:
        return {b: float(byb[b]) for b in BUCKETS}
    v = w.get("w_ms_per_plan", {}).get(arm, {}).get(rung)
    if v is None:
        raise SystemExit(f"FAIL LOUD: no frozen w for {arm} at {rung}")
    return {b: float(v) for b in BUCKETS}


RANGES = {"le10": (1, 10), "11_50": (11, 50), "gt50": (51, 10 ** 9)}


def felt_ratio(s, fit, ub):
    """{bucket: R_n} for one cell: the task-weighted batch size of its charge in n (see the docstring)."""
    sbs = np.load(s["_path"][: -len(".summary.json")] + ".tasks_bsz.npy")[:, 4]
    if (sbs < 1).any():
        raise SystemExit(f"FAIL LOUD: {s['_path']} has tasks without a servedBatchSize")
    out = {}
    for b, (lo, hi) in RANGES.items():
        m = (sbs >= lo) & (sbs <= hi)
        if not m.any():
            out[b] = float(hi if hi < 10 ** 9 else lo)
            continue
        if ub:
            out[b] = float(hi if hi < 10 ** 9 else sbs[m].max())
            continue
        n, tasks = np.unique(sbs[m], return_counts=True)
        nb = tasks / n
        e = n ** fit[b]["beta"]
        out[b] = float((nb * n * e).sum() / (nb * e).sum())
    return out


def q_mode(a, w):
    if bool(a.fit) == bool(a.ub):
        raise SystemExit("FAIL LOUD: q needs exactly one of --fit (exact-F fit) or --ub (the upper bound)")
    fits = json.load(open(a.fit)) if a.fit else None
    new = pooled_w(cells(a.new, a.arm))
    bar_cells = cells(a.bar, a.arm + a.suffix, host=False)
    bar = evals_latency(bar_cells)
    out = {"arm": a.arm, "new": a.new, "bar": a.bar, "estimator": "UB" if a.ub else f"fit {os.path.abspath(a.fit)}", "rungs": {}}
    for r in sorted(bar):
        if r not in new:
            raise SystemExit(f"FAIL LOUD: the re-measure has no {r} cells")
        wf = frozen(w, a.arm, r)
        q = {b: new[r][b] / wf[b] if new[r][b] is not None else None for b in BUCKETS}
        filled = []
        for b in BUCKETS:
            if q[b] is None and bar[r][b]:
                if a.fill_missing_dq is None:
                    raise SystemExit(f"FAIL LOUD: {a.arm} {r} bucket {b} has bar plan evals but no re-measured w")
                filled.append(b)
        dq = {b: (a.fill_missing_dq if b in filled else abs((q[b] if q[b] is not None else 1.0) - 1.0)) for b in BUCKETS}
        felt_c, worst, hb_worst = {b: 0.0 for b in BUCKETS}, None, None
        for s in [x for x in bar_cells if x["_rung"] == r]:
            R = felt_ratio(s, None if a.ub else fits[r], a.ub)
            lat = float(s["averageElapsedTime"]) * int(s["num_tasks"])
            fc = {b: wf[b] * 1e-3 * s["decisionTiming"]["by_bucket"][b]["plan_evals"] * R[b] for b in BUCKETS}
            for b in BUCKETS:
                felt_c[b] += fc[b]
            cb = sum(fc[b] / lat * dq[b] for b in BUCKETS)
            if worst is None or cb > worst["bound"]:
                worst = {"cell": os.path.basename(s["_path"]), "bound": cb, "F": sum(fc.values()) / lat}
            if a.k:
                hb = sum(fc.values()) / lat * (1.0 - 1.0 / a.k)
                if hb_worst is None or hb > hb_worst["bound"]:
                    hb_worst = {"cell": os.path.basename(s["_path"]), "bound": hb}
        rows, bound, bound_pooled = {}, 0.0, 0.0
        for b in BUCKETS:
            share = wf[b] * 1e-3 * bar[r][b] / bar[r]["lat"]
            F = felt_c[b] / bar[r]["lat"]
            bound += F * dq[b]
            bound_pooled += share * dq[b]
            rows[b] = {"w_frozen": wf[b], "w_new": new[r][b], "q": q[b], "F": F, "share_pooled": share}
            if b in filled:
                rows[b]["dq_filled"] = a.fill_missing_dq
        F_total = sum(v["F"] for v in rows.values())
        out["rungs"][r] = {"by_bucket": rows, "F_total": F_total, "share_pooled_total": sum(v["share_pooled"] for v in rows.values()),
                           "bound": bound, "bound_pooled": bound_pooled, "worst_cell": worst, "w_new_pooled": new[r]["all"]}
        if a.k:
            out["rungs"][r]["hand_bias"] = {"k": a.k, "pooled": F_total * (1.0 - 1.0 / a.k), "worst_cell": hb_worst}
    out["verdict"] = "KEEP" if all(v["bound"] < BOUND for v in out["rungs"].values()) and len(out["rungs"]) == 2 else "REFREEZE"
    out["verdict_pooled"] = "KEEP" if all(v["bound_pooled"] < BOUND for v in out["rungs"].values()) and len(out["rungs"]) == 2 else "REFREEZE"
    return out


def felt_cells(a, w, cs, r):
    """[(cell, {bucket: felt charge s}, latency s)] for the rung's cells."""
    fits = json.load(open(a.fit)) if a.fit else None
    wf = frozen(w, a.arm, r)
    out = []
    for s in [x for x in cs if x["_rung"] == r]:
        R = felt_ratio(s, None if a.ub else fits[r], a.ub)
        fc = {b: wf[b] * 1e-3 * s["decisionTiming"]["by_bucket"][b]["plan_evals"] * R[b] for b in BUCKETS}
        out.append((os.path.basename(s["_path"]), fc, float(s["averageElapsedTime"]) * int(s["num_tasks"])))
    return out


def felt_mode(a, w):
    if bool(a.fit) == bool(a.ub):
        raise SystemExit("FAIL LOUD: felt needs exactly one of --fit or --ub")
    cs = cells(a.bar, a.arm + a.suffix, host=False)
    out = {"arm": a.arm, "bar": a.bar, "estimator": "UB" if a.ub else f"fit {os.path.abspath(a.fit)}", "rungs": {}}
    for r in sorted({s["_rung"] for s in cs}):
        rows = felt_cells(a, w, cs, r)
        lat = sum(x[2] for x in rows)
        per = sorted((sum(fc.values()) / l, c) for c, fc, l in rows)
        o = {"n_cells": len(rows), "F_by_bucket": {b: sum(fc[b] for _, fc, _ in rows) / lat for b in BUCKETS},
             "F_total": sum(sum(fc.values()) for _, fc, _ in rows) / lat, "F_cell_median": float(np.median([p[0] for p in per])),
             "F_cell_max": per[-1][0], "worst_cell": per[-1][1],
             "F_cell_max_by_bucket": {b: max(fc[b] / l for _, fc, l in rows) for b in BUCKETS}}
        if a.k:
            o["hand_bias"] = {"k": a.k, "pooled": o["F_total"] * (1 - 1 / a.k), "worst_cell": per[-1][0] * (1 - 1 / a.k)}
        out["rungs"][r] = o
    return out


def share_mode(a, w):
    el = evals_latency(cells(a.cells, a.arm))
    out = {"arm": a.arm, "cells": a.cells, "basis": "k = 0 calibration cells (latency without the charge)", "rungs": {}}
    for r in sorted(el):
        wf = frozen(w, a.arm, r)
        rows = {b: wf[b] * 1e-3 * el[r][b] / el[r]["lat"] for b in BUCKETS}
        out["rungs"][r] = {"by_bucket": rows, "share_total": sum(rows.values())}
    return out


def drift_mode(a):
    s, e = pooled_w(cells(a.start, a.arm)), pooled_w(cells(a.end, a.arm))
    out = {"arm": a.arm, "start": a.start, "end": a.end, "rungs": {}}
    for r in sorted(s):
        d = e[r]["all"] / s[r]["all"] - 1.0
        out["rungs"][r] = {"w_start": s[r]["all"], "w_end": e[r]["all"], "drift": d}
    out["verdict"] = "STOP" if any(abs(v["drift"]) > DRIFT for v in out["rungs"].values()) or set(s) != set(e) else "OK"
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("q", "share", "felt", "drift"))
    ap.add_argument("--arm", required=True)
    ap.add_argument("--new")
    ap.add_argument("--bar")
    ap.add_argument("--suffix", default="_k100")
    ap.add_argument("--cells")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--w-file", default=W_FILE)
    ap.add_argument("--fit")
    ap.add_argument("--ub", action="store_true")
    ap.add_argument("--k", type=float)
    ap.add_argument("--fill-missing-dq", type=float, help="q: |q - 1| for a bucket the re-measure has no evals in but the bar does "
                    "(ruling (c), 2026-10-17: the worst |q - 1| of the series); unset, such a bucket fails loud")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    w = json.load(open(a.w_file))
    need = {"q": ("new", "bar"), "share": ("cells",), "felt": ("bar",), "drift": ("start", "end")}[a.mode]
    if any(not getattr(a, k) for k in need):
        raise SystemExit(f"FAIL LOUD: {a.mode} needs --{' --'.join(need)}")
    out = (q_mode(a, w) if a.mode == "q" else share_mode(a, w) if a.mode == "share" else felt_mode(a, w) if a.mode == "felt"
           else drift_mode(a))
    out["w_file"] = os.path.abspath(a.w_file)
    txt = json.dumps(out, indent=1)
    if a.out:
        open(a.out, "w").write(txt)
    print(txt)
    return out


if __name__ == "__main__":
    sys.exit(1 if main().get("verdict") == "STOP" else 0)
