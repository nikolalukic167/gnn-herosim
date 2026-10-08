#!/usr/bin/env python3
"""load_recalibration_v1: arrival multipliers at which CD's median queue share sits in the light / moderate / heavy band.

Registered protocol (docs/lineages/load_recalibration_v1.md and its 2026-10-08 pre-run amendments): the calibration
topologies x two windows, WF1 (W2 payloads, W3 access classes, W4 mix and repair), batching window 1 s, R1 flags with policy
time scale 1.0, at most 8 evaluated multipliers per rung. A multiplier m scales the x1 arrival rate (timestamps by 1/m).
Every evaluation runs CD and Knative (reactive) on every cell; the statistic is the median CD queue share.

CD guards (a rung where CD fails one is not allowed): every cell finishes; request failures <= 1 % of tasks; per-task
latency p95 <= 300 s; run end <= 1.25 x last arrival (all per cell, worst cell reported). Rung-specific, from the
definition table: light needs the median mean-per-replica busy fraction < 0.3; heavy needs end-of-run backlog <= 2 x
mid-run backlog (per task placement wait + queue time, by arrival: mean of the last quarter over the mean of the middle two
quarters, worst cell) AND, in cells with in-system(1/2) >= 20, the in-system count at 3/4 over 1/2 of the last arrival <= 2.
The same guards are computed for Knative and reported, never disqualifying.

Search per rung: log-bisection between the shared end points. An evaluation whose CD guards fail counts as above target.
It stops early when the median share is inside the band with the guards passing. If a band is never reached, the answer
is the evaluated multiplier with the closest share among those passing the guards, flagged; heavy falls back to the
highest multiplier passing the guards, recorded with its achieved share.

  load_recalibration_v1_bisect.py --work <dir> --grounded-wl <x1 wl> --cfg-dir <cfg_w4> --topologies 9601 9602 9607 9608 \\
      --windows g0 g1 --log <steps.json> --map 0.2666 11.6139     (run on datalab, R1 environment exported)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import statistics as st
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
BANDS = {"light": (0.05, 0.10), "moderate": (0.20, 0.30), "heavy": (0.40, 0.50)}
MIN_IN_SYSTEM = 20  # the in-system ratio binds only when in-system(1/2) is at least this (small counts are noise); always reported
LIMITS = {"request_failure_pct": 1.0, "p95_s": 300.0, "end_over_last_arrival": 1.25, "busy_fraction": 0.3, "backlog_ratio": 2.0}


def tag_for(m: float) -> str:
    return "m" + f"{m:.4f}".replace(".", "p")


def next_multiplier(lo: float, hi: float) -> float:
    return round(math.exp(0.5 * (math.log(lo) + math.log(hi))), 4)


def cell_metrics(s: dict) -> dict:
    """Guard inputs from one gate summary."""
    n = float(s["num_tasks"])
    rf = s.get("requestFailures") or 0
    rf = rf if isinstance(rf, (int, float)) else len(rf)
    lp, ae, qd = s.get("latency_percentiles") or {}, s.get("arrival_end") or {}, s.get("queue_drift") or {}
    rc = s.get("replica_count_series") or {}
    busy = None
    if rc.get("time_mean") and s.get("endTime") and s.get("averageExecutionTime") is not None:
        busy = float(s["averageExecutionTime"]) * n / (float(rc["time_mean"]) * float(s["endTime"]))
    bp = s.get("backlog_profile") or {}
    backlog = bp.get("last_over_mid")
    pw = s.get("placement_wait") or {}
    return {"queue_share": s["queue_share"], "latency_s": s["averageElapsedTime"], "request_failure_pct": 100.0 * rf / n,
            "p95_s": lp.get("p95"), "p99_s": lp.get("p99"), "end_over_last_arrival": ae.get("end_over_last_arrival"),
            "busy_fraction": busy, "backlog_ratio": backlog, "in_system_ratio": bp.get("in_system_ratio"),
            "in_system_half": (bp.get("in_system_at") or {}).get("half"), "wait_mean_s": pw.get("mean"), "wait_p95_s": pw.get("p95"),
            "wait_max_s": pw.get("max")}


def guards(cells: List[dict], expected: int, rung: Optional[str] = None) -> dict:
    """Every guard, pass/fail, for one arm at one multiplier. ``rung`` None reports all; the rung-specific two are
    only decisive for light (busy) and heavy (backlog)."""
    def worst(k):
        v = [c[k] for c in cells if c[k] is not None]
        return max(v) if v else None

    busy = [c["busy_fraction"] for c in cells if c["busy_fraction"] is not None]
    g = {"finished": len(cells) == expected,
         "request_failures": bool(cells) and worst("request_failure_pct") <= LIMITS["request_failure_pct"],
         "p95": bool(cells) and worst("p95_s") is not None and worst("p95_s") <= LIMITS["p95_s"],
         "run_end": bool(cells) and worst("end_over_last_arrival") is not None
         and worst("end_over_last_arrival") <= LIMITS["end_over_last_arrival"],
         "busy": bool(busy) and st.median(busy) < LIMITS["busy_fraction"],
         "backlog": bool(cells) and worst("backlog_ratio") is not None and worst("backlog_ratio") <= LIMITS["backlog_ratio"]
         and all(c["in_system_ratio"] is not None and c["in_system_ratio"] <= LIMITS["backlog_ratio"]
                 for c in cells if (c["in_system_half"] or 0) >= MIN_IN_SYSTEM)}
    g["values"] = {"worst_request_failure_pct": worst("request_failure_pct") if cells else None, "worst_p95_s": worst("p95_s") if cells else None,
                   "worst_end_over_last_arrival": worst("end_over_last_arrival") if cells else None,
                   "median_busy_fraction": st.median(busy) if busy else None,
                   "worst_backlog_ratio": worst("backlog_ratio") if cells else None,
                   "worst_in_system_ratio": worst("in_system_ratio") if cells else None}
    return g


def allowed(g: dict, rung: str) -> bool:
    base = g["finished"] and g["request_failures"] and g["p95"] and g["run_end"]
    return bool(base and (g["busy"] if rung == "light" else g["backlog"] if rung == "heavy" else True))


def search(rung: str, evaluate, m_lo: float, m_hi: float, max_steps: int) -> dict:
    """``evaluate(m)`` -> (share, allowed). Returns the step list and the rung's answer."""
    band_lo, band_hi = BANDS[rung]
    target = (band_lo + band_hi) / 2
    steps: List[dict] = []

    def step(m: float) -> str:
        share, ok = evaluate(m)
        pos = "inf" if share is None or not ok else "low" if share < band_lo else "high" if share > band_hi else "in"
        steps.append({"step": len(steps) + 1, "m": m, "share": share, "allowed": ok, "position": pos})
        return pos

    lo, hi = m_lo, m_hi
    p_lo, p_hi = step(lo), step(hi)
    out: dict = {"rung": rung, "band": [band_lo, band_hi], "target": target, "steps": steps}
    if p_lo == "in" or p_hi == "in":
        out["status"] = "BRACKET-END-IN-BAND"
    elif p_lo in ("high", "inf"):
        out["status"] = "UNBRACKETED-LOW"
    elif p_hi == "low":
        out["status"] = "UNBRACKETED-HIGH"
    else:
        out["status"] = "BRACKETED"
        while len(steps) < max_steps:
            mid = next_multiplier(lo, hi)
            if mid <= lo or mid >= hi:
                break
            p = step(mid)
            if p == "in":
                break
            lo, hi = (mid, hi) if p == "low" else (lo, mid)
        out["bracket"] = [lo, hi]
    inside = [s for s in steps if s["position"] == "in"]
    passing = [s for s in steps if s["allowed"] and s["share"] is not None]
    if inside:
        out["answer"] = {**inside[0], "kind": "IN-BAND"}
    elif rung == "heavy" and passing:
        out["answer"] = {**max(passing, key=lambda s: s["m"]), "kind": "HIGHEST-STABLE"}
    elif passing:
        out["answer"] = {**min(passing, key=lambda s: abs(s["share"] - target)), "kind": "CLOSEST-ALLOWED-NOT-IN-BAND"}
    else:
        out["answer"] = None
    return out


class Evaluator:
    def __init__(self, a: argparse.Namespace):
        self.a = a
        self.cache: Dict[float, dict] = {}
        self.locks: Dict[float, threading.Lock] = {}
        self.guard = threading.Lock()
        self.expected = len(a.topologies) * len(a.windows)

    def record(self, m: float) -> dict:
        with self.guard:
            lock = self.locks.setdefault(m, threading.Lock())
        with lock:
            if m not in self.cache:
                self.cache[m] = self.run(m)
            return self.cache[m]

    def run(self, m: float) -> dict:
        a, tag = self.a, tag_for(m)
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        subprocess.run([sys.executable, str(ROOT / "scripts_cosim/workload_fix_v1_build.py"), "--grounded-wl", str(a.grounded_wl),
                        "--cfg-dir", str(a.cfg_dir), "--topologies", *map(str, a.topologies), "--rung", f"{tag}={1.0 / m!r}",
                        "--payload-sampler", "wf1_v1", "--task-mix", "wf1_v1", "--batch-timeout-fixed", "1.0",
                        "--out", str(a.work / tag)], check=True, cwd=ROOT, env=env)
        out = a.work / tag / "gate"
        genv = {**env, "WF1_RUNGS": tag, "WF1_TOPOS": ",".join(map(str, a.topologies)), "WF1_CAL_WINDOWS": ",".join(a.windows),
                "WF1_CAL_KINDS": "cd,reactive"}
        cmd = [sys.executable, str(ROOT / "scripts_cosim/fresh_topo_burst_v1_gate.py"), "wf1cal", "--inputs", str(a.work / tag),
               "--out", str(out), "--parallel", str(a.parallel), "--mem", "4G", "--no-scope"]
        subprocess.run(cmd + ["--timeout", str(a.timeout)], check=False, cwd=ROOT, env=genv, stdout=subprocess.DEVNULL)
        failed = sorted(out.glob("*.failed.json"))
        if failed:  # registered failure rule: one rerun of a failed run at 3x the timeout; a second failure stands
            (out / "first_pass_failed").mkdir(exist_ok=True)
            for f in failed:
                shutil.move(str(f), str(out / "first_pass_failed" / f.name))
            subprocess.run(cmd + ["--timeout", str(3 * a.timeout)], check=False, cwd=ROOT, env=genv, stdout=subprocess.DEVNULL)
        arms: Dict[str, list] = {"cd": [], "reactive": []}
        names = []
        for t in a.topologies:
            for w in a.windows:
                for arm in arms:
                    f = out / f"cc40s{t}__{w}{tag}__{arm}_s0.summary.json"
                    if f.exists():
                        arms[arm].append({"topology": t, "window": w, **cell_metrics(json.loads(f.read_text()))})
                    else:
                        names.append(f.name)
        rec = {"m": m, "tag": tag, "missing": names, "cells": arms,
               "cd_median_share": st.median([c["queue_share"] for c in arms["cd"]]) if arms["cd"] else None,
               "reactive_median_share": st.median([c["queue_share"] for c in arms["reactive"]]) if arms["reactive"] else None,
               "guards": {arm: guards(cs, self.expected) for arm, cs in arms.items()}}
        print(f"[eval] m={m} {tag}: CD {len(arms['cd'])}/{self.expected} cells, median share {rec['cd_median_share']}; "
              f"Knative {len(arms['reactive'])}/{self.expected}, median share {rec['reactive_median_share']}", flush=True)
        return rec

    def for_rung(self, rung: str):
        def evaluate(m: float):
            r = self.record(m)
            return r["cd_median_share"], allowed(r["guards"]["cd"], rung)
        return evaluate


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--grounded-wl", type=Path, required=True)
    ap.add_argument("--cfg-dir", type=Path, required=True)
    ap.add_argument("--topologies", type=int, nargs="+", required=True)
    ap.add_argument("--windows", nargs="+", default=["g0", "g1"])
    ap.add_argument("--m-lo", type=float, default=0.05)
    ap.add_argument("--m-hi", type=float, default=64.0)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--rungs", nargs="+", default=list(BANDS))
    ap.add_argument("--map", type=float, nargs="*", default=[], help="multipliers evaluated as context (the provisional rungs)")
    ap.add_argument("--parallel", type=int, default=16)
    ap.add_argument("--timeout", type=int, default=2700)
    ap.add_argument("--log", type=Path, required=True)
    a = ap.parse_args()
    for k, v in (("HEROSIM_TRANSFER_MODEL", "pipelined"), ("HEROSIM_REPLICA_RELEASE", "1"), ("HEROSIM_SCALEOUT", "kpa"),
                 ("GATE_FIXED_POLICY_TIME_SCALE", "1.0")):
        if os.environ.get(k) != v:
            raise SystemExit(f"FAIL LOUD: {k}={os.environ.get(k)!r}; load_recalibration_v1 runs on R1 ({k}={v})")
    a.work.mkdir(parents=True, exist_ok=True)
    ev = Evaluator(a)
    with ThreadPoolExecutor(max_workers=len(a.rungs) + len(a.map)) as ex:
        searches = [ex.submit(search, r, ev.for_rung(r), a.m_lo, a.m_hi, a.max_steps) for r in a.rungs]
        mapped = [ex.submit(ev.record, m) for m in a.map]
        results = [f.result() for f in searches]
        for f in mapped:
            f.result()
    doc = {"topologies": a.topologies, "windows": a.windows, "m_range": [a.m_lo, a.m_hi], "max_steps": a.max_steps,
           "bands": BANDS, "limits": LIMITS, "results": results, "mapped": [ev.cache[m] for m in a.map],
           "evaluations": {r["tag"]: r for r in ev.cache.values()}}
    a.log.write_text(json.dumps(doc, indent=1, default=str))
    for r in results:
        print(json.dumps({k: r[k] for k in ("rung", "status", "answer")}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
