#!/usr/bin/env python3
"""uncut_groups_s0_v1 calibration protocol (SMT, 2026-10-14): the hand q_n bound, the learned charge share and the H3 drift stop.

  uncut_charge_share.py q     --new <dir>[,<dir>] --arm cd_ib0 --bar <869828 dir> [--w-file decision_cost_w.json] [--out q.json]
  uncut_charge_share.py share --cells <dir>[,<dir>] --arm ra_gnn_eng_arm1_nosplit [--w-file <W1's w file>] [--out s.json]
  uncut_charge_share.py drift --start <dir> --end <dir> --arm cd_ib0 [--out d.json]

q: per rung and served-batch bucket n, q_n = w_new_n / w_frozen (the frozen w is the arm's scalar at the rung, or its bucket rate where it
has one), w_new pooled over the re-measure's cells (sum total_s / sum plan_evals, every node's rep together). share_n = w_frozen_n x the
bar cells' plan evals in n / their summed task latency (num_tasks x mean), pooled over the arm's bar cells at the rung (suffix _k100).
bound = sum_n share_n x |q_n - 1|. KEEP if bound < 0.1 % of mean latency on both rungs, else REFREEZE (the 192 hand cells rerun).
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


def q_mode(a, w):
    new = pooled_w(cells(a.new, a.arm))
    bar = evals_latency(cells(a.bar, a.arm + a.suffix, host=False))
    out = {"arm": a.arm, "new": a.new, "bar": a.bar, "rungs": {}}
    for r in sorted(bar):
        if r not in new:
            raise SystemExit(f"FAIL LOUD: the re-measure has no {r} cells")
        wf = frozen(w, a.arm, r)
        rows, bound = {}, 0.0
        for b in BUCKETS:
            q = new[r][b] / wf[b] if new[r][b] is not None else None
            share = wf[b] * 1e-3 * bar[r][b] / bar[r]["lat"]
            if q is None and bar[r][b]:
                raise SystemExit(f"FAIL LOUD: {a.arm} {r} bucket {b} has bar plan evals but no re-measured w")
            bound += share * abs((q if q is not None else 1.0) - 1.0)
            rows[b] = {"w_frozen": wf[b], "w_new": new[r][b], "q": q, "share": share}
        out["rungs"][r] = {"by_bucket": rows, "share_total": sum(v["share"] for v in rows.values()), "bound": bound,
                           "w_new_pooled": new[r]["all"]}
    out["verdict"] = "KEEP" if all(v["bound"] < BOUND for v in out["rungs"].values()) and len(out["rungs"]) == 2 else "REFREEZE"
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
    ap.add_argument("mode", choices=("q", "share", "drift"))
    ap.add_argument("--arm", required=True)
    ap.add_argument("--new")
    ap.add_argument("--bar")
    ap.add_argument("--suffix", default="_k100")
    ap.add_argument("--cells")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--w-file", default=W_FILE)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    w = json.load(open(a.w_file))
    need = {"q": ("new", "bar"), "share": ("cells",), "drift": ("start", "end")}[a.mode]
    if any(not getattr(a, k) for k in need):
        raise SystemExit(f"FAIL LOUD: {a.mode} needs --{' --'.join(need)}")
    out = q_mode(a, w) if a.mode == "q" else share_mode(a, w) if a.mode == "share" else drift_mode(a)
    out["w_file"] = os.path.abspath(a.w_file)
    txt = json.dumps(out, indent=1)
    if a.out:
        open(a.out, "w").write(txt)
    print(txt)
    return out


if __name__ == "__main__":
    sys.exit(1 if main().get("verdict") == "STOP" else 0)
