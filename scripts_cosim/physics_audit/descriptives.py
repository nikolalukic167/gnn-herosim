#!/usr/bin/env python3
"""physics_audit_v1 descriptive outputs (no tests), from traces written with HEROSIM_AUDIT_TRACE.

Per run: per-replica busy fraction, utilisation per platform type, latency shares, replica count over time, and the
concentration-vs-saturation summary. A replica is (function, node:platform, incarnation); its life runs from
rep_up to rep_down (or the end of the run). Two busy fractions, both over the replica's life:
  compute   seconds with the compute lock held (compute_start -> exec_end) -- the CPU/accelerator being used
  occupied  seconds with at least one task past its pop and not yet done -- the replica "has work"
Latency is done - dispatched per task; its parts are sched (scheduled - dispatched: waiting at the scheduler), queue
(pop - scheduled: waiting in the replica's queue), ingress (arrived - pop: the transfer in), cold, rendezvous,
exchange, exec, and `other` (what is left: storage waits, input/output I/O beyond exchange, output).

  descriptives.py --dir D [--dir-label NAME] --out OUT.json     (files <topo>_<rung>_<arm>.trace.jsonl)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
from collections import defaultdict
from typing import Any, Dict, List, Tuple

PARTS = ("sched", "queue", "ingress", "cold", "rendezvous", "exchange", "exec", "other")


def pct(vals: List[float], q: float) -> float:
    s = sorted(vals)
    return s[min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))]


def union_length(iv: List[Tuple[float, float]]) -> float:
    total, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(iv):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def load(path: str) -> Dict[str, List[Dict[str, Any]]]:
    rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            rows[r["k"]].append(r)
    return rows


def describe(path: str, series_points: int = 60) -> Dict[str, Any]:
    rows = load(path)
    end_t = rows["end"][-1]["t"] if rows["end"] else max(r["t"] for r in rows["svc"])
    life: Dict[Tuple[str, str, int], List[float]] = {}
    for r in rows["rep_up"]:
        life[(r["fn"], r["q"], r["inc"])] = [r["t"], end_t]
    for r in rows["rep_down"]:
        k = (r["fn"], r["q"], r["inc"])
        if k in life:
            life[k][1] = r["t"]
    by_rep: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = defaultdict(list)
    ptype_of: Dict[Tuple[str, str, int], str] = {}
    for r in rows["svc"]:
        k = (r["type"], r["q"], r["incarnation"])
        by_rep[k].append(r)
        ptype_of[k] = r["ptype"]
    for k, r in ((k, r) for k, rs in by_rep.items() for r in rs[:1]):
        if k not in life:  # a replica the run started with and never logged a creation for
            life[k] = [0.0, end_t]
    reps = []
    for k, (t0, t1) in life.items():
        span = t1 - t0
        if span <= 0:
            continue
        rs = by_rep.get(k, [])
        compute = sum(r["exec_end"] - r["compute_start"] for r in rs)
        occupied = union_length([(r["pop"], r["done"]) for r in rs])
        reps.append({"key": k, "span": span, "compute": min(1.0, compute / span), "occupied": min(1.0, occupied / span),
                     "compute_s": compute, "tasks": len(rs), "ptype": ptype_of.get(k)})
    def busy_stats(population: List[Dict[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {"replicas": len(population)}
        if not population:
            return out
        for kind in ("compute", "occupied"):
            v = [r[kind] for r in population]
            top = sorted(((r[kind] * r["span"]) for r in population), reverse=True)
            k10 = max(1, len(top) // 10)
            out[kind] = {"median": statistics.median(v), "p90": pct(v, 0.9), "max": max(v), "mean": statistics.fmean(v),
                         "frac_above_0.5": sum(x > 0.5 for x in v) / len(v), "frac_above_0.8": sum(x > 0.8 for x in v) / len(v),
                         "top10pct_share_of_busy_seconds": sum(top[:k10]) / sum(top) if sum(top) > 0 else None,
                         "time_weighted_mean": sum(r[kind] * r["span"] for r in population) / sum(r["span"] for r in population)}
        return out

    # replicas alive under a minute are mostly the reachability-driven one-task replicas and make a fraction of a short
    # life look like a busy one, so the long-lived population is reported next to all of them
    busy = {"all": busy_stats(reps), "ge60s": busy_stats([r for r in reps if r["span"] >= 60.0])}
    plat: Dict[str, Dict[str, float]] = defaultdict(lambda: {"replicas": 0, "span": 0.0, "compute_s": 0.0, "tasks": 0,
                                                                "occupied_s": 0.0})
    for r in reps:
        d = plat[r["ptype"] or "unserved"]
        d["replicas"] += 1
        d["span"] += r["span"]
        d["compute_s"] += r["compute_s"]
        d["occupied_s"] += r["occupied"] * r["span"]
        d["tasks"] += r["tasks"]
    util = {p: {"replicas": d["replicas"], "tasks": d["tasks"], "compute_util": d["compute_s"] / d["span"],
                "occupied_util": d["occupied_s"] / d["span"]} for p, d in plat.items()}
    parts = {p: 0.0 for p in PARTS}
    total = 0.0
    for r in rows["svc"]:
        lat = r["done"] - r["dispatched"]
        got = {"sched": r["scheduled"] - r["dispatched"], "queue": r["pop"] - r["scheduled"],
               "ingress": r["arrived"] - r["pop"], "cold": r["cold"] or 0.0,
               "rendezvous": r["rendezvous"] or 0.0, "exchange": r["exchange"] or 0.0, "exec": r["exec"] or 0.0}
        got["other"] = lat - sum(got.values())
        for p in PARTS:
            parts[p] += got[p]
        total += lat
    shares = {p: parts[p] / total for p in PARTS}
    shares["latency_mean_s"] = total / len(rows["svc"])
    ups = [(r["t"], 1, r["cause"]) for r in rows["rep_up"]]
    downs = [(r["t"], -1, None) for r in rows["rep_down"]]
    initial = sum(1 for r in rows["rep_up"] if r["cause"] == "initial")
    events = sorted(ups + downs, key=lambda e: (e[0], e[1]))
    grid = [end_t * i / series_points for i in range(series_points + 1)]
    series, n, j = [], 0, 0
    for g in grid:
        while j < len(events) and events[j][0] <= g:
            n += events[j][1]
            j += 1
        series.append(n)
    area, n, last = 0.0, 0, 0.0
    for t, d, _c in events:
        area += n * (t - last)
        n += d
        last = t
    area += n * (end_t - last)
    counts = {"end_t": end_t, "grid_s": grid, "alive": series, "peak": max(series), "time_mean": area / end_t,
              "creations": len(ups) - initial, "initial": initial,
              "by_cause": {c: sum(1 for u in ups if u[2] == c) for c in {u[2] for u in ups}}}
    return {"replicas": len(reps), "busy": busy, "util_by_platform_type": util, "latency_shares": shares, "replica_counts": counts}


def aggregate(runs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Across topologies, per (rung, arm): the median of the per-topology figures; shares are the median share."""
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for name, d in runs.items():
        _topo, rung, arm = name.split("_")
        groups[(rung, arm)].append(d)
    out = {}
    for (rung, arm), ds in sorted(groups.items()):
        def med(f):
            vals = [f(d) for d in ds]
            vals = [v for v in vals if v is not None]
            return statistics.median(vals) if vals else None
        row = {"topologies": len(ds), "replicas": med(lambda d: d["replicas"]),
               "alive_time_mean": med(lambda d: d["replica_counts"]["time_mean"]),
               "alive_peak": med(lambda d: d["replica_counts"]["peak"]), "creations": med(lambda d: d["replica_counts"]["creations"])}
        for pop in ("all", "ge60s"):
            row[f"{pop}_replicas"] = med(lambda d: d["busy"][pop]["replicas"])
            for kind in ("compute", "occupied"):
                for stat in ("median", "p90", "max", "mean", "frac_above_0.5", "frac_above_0.8", "top10pct_share_of_busy_seconds",
                             "time_weighted_mean"):
                    row[f"{pop}_{kind}_{stat}"] = med(lambda d: d["busy"][pop].get(kind, {}).get(stat))
        for p in PARTS + ("latency_mean_s",):
            row[f"share_{p}"] = med(lambda d: d["latency_shares"][p])
        ptypes = sorted({p for d in ds for p in d["util_by_platform_type"]})
        row["util"] = {p: {"compute": med(lambda d: d["util_by_platform_type"].get(p, {}).get("compute_util")),
                           "occupied": med(lambda d: d["util_by_platform_type"].get(p, {}).get("occupied_util")),
                           "replicas": med(lambda d: d["util_by_platform_type"].get(p, {}).get("replicas"))} for p in ptypes}
        out[f"{rung}_{arm}"] = row
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = {}
    for tp in sorted(glob.glob(os.path.join(a.dir, "*.trace.jsonl"))):
        runs[os.path.basename(tp)[: -len(".trace.jsonl")]] = describe(tp)
    agg = aggregate(runs)
    json.dump({"per_run": runs, "aggregate": agg}, open(a.out, "w"), indent=1, default=str)
    for key, r in agg.items():
        print(f"{key}: lives {r['replicas']:.0f} (>=60s: {r['ge60s_replicas']:.0f}) alive mean {r['alive_time_mean']:.1f} peak {r['alive_peak']:.0f}")
        for pop in ("all", "ge60s"):
            print(f"    {pop:5s} occupied med {r[pop+'_occupied_median']} p90 {r[pop+'_occupied_p90']} max {r[pop+'_occupied_max']} "
                  f"| compute med {r[pop+'_compute_median']} p90 {r[pop+'_compute_p90']} max {r[pop+'_compute_max']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
