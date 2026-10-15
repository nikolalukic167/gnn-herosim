#!/usr/bin/env python3
"""uncut_groups_s0_v1: learned charge share per served-batch bucket (<=10, 11-50, 51-100, 101-256, >256) and in the worst cell.

  uncut_interim_share.py --cells <dir>[,<dir>] --arm ra_twin_eng_arm1b_nosplit_k100 --w-file <the cells' decision_cost_w.json> [--out x.json]

Charge = sum over buckets of w(bucket) x plan evals in it; share = charge / summed task latency (num_tasks x averageElapsedTime), pooled
over the cells per rung, and per cell (the worst cell is the largest total share). The gate's w buckets are le10 / 11_50 / gt50; 51-100,
101-256 and >256 split gt50 and are charged its rate. le10, 11_50 and gt50 plan evals are decisionTiming.by_bucket's (the charged ones).
gt50's charged plan evals are split APPROXIMATELY in the proportions of the .tasks_bsz.npy sidecar (servedBatchSize, column 4; tasks
above 50 by sub-bucket). The sidecar holds each task's LAST served batch, so its count above 50 differs from the charged gt50 count (look 1:
147 of 192 cells, by -1 to +373 tasks); the per-cell relative gap is reported (gt50_sidecar_gap). A one-pass learned arm evaluates one
plan per served task; any other arm fails loud.
Felt share (the large-group view): a task served in a batch of n waits the batch's whole charge, w(n) x n; per bucket, the sum of that over
the bucket's tasks / their summed latency (doneTime - dispatchedTime), from the sidecar's last served batch. Pooled per rung, and in the
cell with the largest felt share in any bucket. felt_total: the same over every task, pooled per rung (the primary figure, 2026-10-14);
felt_total_worst: the largest per-cell felt_total.
--evals-per-task-from-buckets (Arm 2 / K0, which re-encode: plan evals = tasks + arm2_extra_evals): the one-plan-per-task check is off;
pooled share is unchanged and exact (the charged bucket totals). Felt is APPROXIMATE (bucket-mean evals per task; per-batch evals not in
sidecar): a task in a batch of n waits w(n) x n x ratio, ratio = the gate bucket's plan evals / its sidecar task count. With ratio 1 it
equals the default.
"""
import argparse
import glob
import json
import os
import re

import numpy as np

NAME = re.compile(r"^cc40s(\d+)__(g\d)(moderate|heavy)__(.+)_s(\d+)\.summary\.json$")
FINE = (("le10", 1, 10, "le10"), ("11_50", 11, 50, "11_50"), ("51_100", 51, 100, "gt50"), ("101_256", 101, 256, "gt50"),
        ("gt256", 257, 10 ** 9, "gt50"))


def cell(p, w, per_task=False):
    s = json.load(open(p))
    rung = NAME.match(os.path.basename(p)).group(3)
    arr = np.load(p[: -len(".summary.json")] + ".tasks_bsz.npy")
    sbs, tlat = arr[:, 4], arr[:, 2] - arr[:, 1]
    if (sbs < 1).any():
        raise SystemExit(f"FAIL LOUD: {p} has {(sbs < 1).sum()} tasks without a servedBatchSize")
    n = {f: int(((sbs >= lo) & (sbs <= hi)).sum()) for f, lo, hi, _ in FINE}
    bb = s["decisionTiming"]["by_bucket"]
    if not per_task and sum(v["plan_evals"] for v in bb.values()) != int(s["decisionTiming"]["tasks"]):
        raise SystemExit(f"FAIL LOUD: {p}: plan evals != tasks; not a one-plan-per-task arm")
    above, e = n["51_100"] + n["101_256"] + n["gt256"], bb["gt50"]["plan_evals"]
    side = {"le10": n["le10"], "11_50": n["11_50"], "gt50": above}
    ratio = {g: (bb[g]["plan_evals"] / side[g] if per_task and side[g] else 1.0) for g in side}
    gap = (above - e) / e if e else 0.0
    for f in ("51_100", "101_256", "gt256"):
        n[f] = e * n[f] / above if above else 0.0
    n["le10"], n["11_50"] = bb["le10"]["plan_evals"], bb["11_50"]["plan_evals"]
    rate = w[rung]
    lat = float(s["averageElapsedTime"]) * int(s["num_tasks"])
    charge = {f: rate[g] * 1e-3 * n[f] for f, _, _, g in FINE}
    felt = {}
    for f, lo, hi, g in FINE:
        m = (sbs >= lo) & (sbs <= hi)
        felt[f] = (float((rate[g] * 1e-3 * sbs[m]).sum()) * ratio[g], float(tlat[m].sum()))
    return rung, lat, charge, n, gap, felt, ratio


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--w-file", required=True)
    ap.add_argument("--out")
    ap.add_argument("--evals-per-task-from-buckets", action="store_true")
    a = ap.parse_args(argv)
    base = re.sub(r"_(k\d+|kr)$", "", a.arm)
    wf = json.load(open(a.w_file))
    w = wf.get("w_ms_per_plan_by_bucket", {}).get(base) or {r: {g: v for g in ("le10", "11_50", "gt50")}
                                                         for r, v in wf["w_ms_per_plan"][base].items()}
    paths = [p for d in a.cells.split(",") if d for p in sorted(glob.glob(os.path.join(d, "*.summary.json")))
             if NAME.match(os.path.basename(p)) and NAME.match(os.path.basename(p)).group(4) == a.arm]
    if not paths:
        raise SystemExit(f"FAIL LOUD: no {a.arm} cells in {a.cells}")
    out = {"arm": a.arm, "w_file": os.path.abspath(a.w_file), "cells": a.cells, "n_cells": len(paths), "rungs": {}}
    if a.evals_per_task_from_buckets:
        out["felt_label"] = "approximate (bucket-mean evals per task; per-batch evals not in sidecar)"
    for p in paths:
        rung, lat, charge, n, gap, felt, ratio = cell(p, w, a.evals_per_task_from_buckets)
        r = out["rungs"].setdefault(rung, {"lat": 0.0, "charge": {f[0]: 0.0 for f in FINE}, "evals": {f[0]: 0 for f in FINE}, "worst": None,
                                             "gt50_sidecar_gap_max": 0.0})
        r["gt50_sidecar_gap_max"] = max(r["gt50_sidecar_gap_max"], abs(gap))
        if a.evals_per_task_from_buckets:
            rr = r.setdefault("evals_per_task_ratio", {g: [None, None] for g in ratio})
            for g, v in ratio.items():
                rr[g] = [v if rr[g][0] is None else min(rr[g][0], v), v if rr[g][1] is None else max(rr[g][1], v)]
        fa = r.setdefault("_felt", {f[0]: [0.0, 0.0] for f in FINE})
        for f, (c, l) in felt.items():
            fa[f][0] += c
            fa[f][1] += l
        cf = {f: (c / l if l else None) for f, (c, l) in felt.items()}
        ft = sum(c for c, _ in felt.values()) / sum(l for _, l in felt.values())
        if r.get("felt_total_worst") is None or ft > r["felt_total_worst"]["felt_total"]:
            r["felt_total_worst"] = {"cell": os.path.basename(p), "felt_total": ft}
        top = max(v for v in cf.values() if v is not None)
        if r.get("felt_worst") is None or top > r["felt_worst"]["max"]:
            r["felt_worst"] = {"cell": os.path.basename(p), "max": top, "by_bucket": cf}
        r["lat"] += lat
        for f in charge:
            r["charge"][f] += charge[f]
            r["evals"][f] += n[f]
        tot = sum(charge.values()) / lat
        if r["worst"] is None or tot > r["worst"]["share_total"]:
            r["worst"] = {"cell": os.path.basename(p), "share_total": tot, "by_bucket": {f: c / lat for f, c in charge.items()}}
    for r in out["rungs"].values():
        r["share_by_bucket"] = {f: c / r["lat"] for f, c in r["charge"].items()}
        r["share_total"] = sum(r["share_by_bucket"].values())
        r["evals_share"] = {f: e / sum(r["evals"].values()) for f, e in r["evals"].items()}
        del r["charge"]
        fe = r.pop("_felt")
        r["felt_by_bucket"] = {f: (c / l if l else None) for f, (c, l) in fe.items()}
        r["felt_total"] = sum(c for c, _ in fe.values()) / sum(l for _, l in fe.values())
    txt = json.dumps(out, indent=1)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(txt)
    for rung, r in sorted(out["rungs"].items()):
        print(f"{rung}: pooled {100 * r['share_total']:.3f} %  " + "  ".join(f"{f} {100 * v:.3f} %" for f, v in r["share_by_bucket"].items())
              + f"  | worst {r['worst']['cell']} {100 * r['worst']['share_total']:.3f} %")
        print(f"{rung} felt total {100 * r['felt_total']:.3f} % (worst cell {r['felt_total_worst']['cell']} {100 * r['felt_total_worst']['felt_total']:.3f} %)")
        print(f"{rung} felt: " + "  ".join(f"{f} {100 * v:.3f} %" for f, v in r["felt_by_bucket"].items() if v is not None)
              + f"  | worst {r['felt_worst']['cell']} " + "  ".join(f"{f} {100 * v:.3f} %" for f, v in r["felt_worst"]["by_bucket"].items() if v is not None))
    return out


if __name__ == "__main__":
    main()
