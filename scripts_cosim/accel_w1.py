#!/usr/bin/env python3
"""accel_replica_v1 W1: how much of CD's work runs on an accelerator, with and without the rule, on the SAME calibrated cells.

For one calibrated heavy rung (a tag from scale_rung_bisect.py's inputs) this runs peer_greedy_network_cd on every seed x window cell twice -- the cell as
minted (preinit.replica_placement_rule = fastest_compatible) and a copy with the key removed (first_compatible) -- through physics_audit/run_cell.sh with
SIM_FORCE_FULL_STATS=1, and reads taskResults: the share of tasks on xavierGpu / xavierDla / pynqFpga, and which task types ran on xavierGpu.
Bar (amended 2026-10-10): accel share (fastest) >= accel share (first) + 15 points, AND xavierGpu hosts >= 1 task type. Medians over the cells.

  accel_w1.py --root <calib root with inputs/ and cfg_src/> --tag m27p6227 --out <dir> --wt <worktree> --seeds 9905 9906 9907 9908 --windows g0 g1
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ACC = ("xavierGpu", "xavierDla", "pynqFpga")

ap = argparse.ArgumentParser()
ap.add_argument("--root", type=Path, required=True); ap.add_argument("--tag", required=True); ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--wt", type=Path, required=True); ap.add_argument("--seeds", type=int, nargs="+", required=True)
ap.add_argument("--windows", nargs="+", default=["g0", "g1"]); ap.add_argument("--parallel", type=int, default=8); ap.add_argument("--limit", type=int, default=2700)
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
src = a.root / "inputs" / f"grounded_{a.tag}"
for rule in ("first_compatible", "fastest_compatible"):
    d = a.out / f"inputs_{rule}" / f"grounded_{a.tag}"
    (d / "cfg").mkdir(parents=True, exist_ok=True)
    if not (d / "wl").exists():
        (d / "wl").symlink_to((src / "wl").resolve())
    for s in a.seeds:
        c = json.loads((src / "cfg" / f"cc40s{s}.json").read_text())
        pre = c.setdefault("preinit", {})
        if rule == "first_compatible":
            pre.pop("replica_placement_rule", None)
        else:
            assert pre.get("replica_placement_rule") == "fastest_compatible", f"cell {s} was not minted under fastest_compatible"
        (d / "cfg" / f"cc40s{s}.json").write_text(json.dumps(c, indent=1))


def run(job):
    rule, s, w = job
    raw = a.out / f"{rule}_{s}_{w}.raw.json"
    if not raw.exists():
        env = dict(os.environ, HEROSIM_PY=sys.executable, HEROSIM_AUDIT_INPUTS=str(a.out / f"inputs_{rule}"), TS="1.0", SIM_FORCE_FULL_STATS="1",
                   HEROSIM_SHARED_AUTOSCALER="0", GATE_FIXED_POLICY_TIME_SCALE="1.0", OMP_NUM_THREADS="1")
        subprocess.run(["timeout", "--kill-after=60", str(a.limit), "bash", "scripts_cosim/physics_audit/run_cell.sh", str(s), w, a.tag, "peer_greedy_network_cd", str(raw), "50000"],
                       cwd=a.wt, env=env, stdout=open(a.out / f"{rule}_{s}_{w}.log", "w"), stderr=subprocess.STDOUT)
    d = json.load(open(raw))
    tr = d["stats"]["taskResults"]
    n = len(tr)
    by_plat: dict = {}
    gpu_types: dict = {}
    for t in tr:
        p = t["platform"]["shortName"]; ty = t["taskType"]["name"]
        by_plat[p] = by_plat.get(p, 0) + 1
        if p == "xavierGpu":
            gpu_types[ty] = gpu_types.get(ty, 0) + 1
    raw.unlink()
    return dict(rule=rule, seed=s, window=w, n=n, accel_share=sum(by_plat.get(p, 0) for p in ACC) / n, by_platform=by_plat, xavierGpu_types=gpu_types,
                elapsed=d["stats"]["averageElapsedTime"], end=d["stats"]["endTime"])


jobs = [(r, s, w) for r in ("first_compatible", "fastest_compatible") for s in a.seeds for w in a.windows]
with ThreadPoolExecutor(a.parallel) as ex:
    cells = list(ex.map(run, jobs))
res = {"tag": a.tag, "cells": cells}
for r in ("first_compatible", "fastest_compatible"):
    cs = [c for c in cells if c["rule"] == r]
    res[r] = {"median_accel_share": st.median(c["accel_share"] for c in cs), "min": min(c["accel_share"] for c in cs), "max": max(c["accel_share"] for c in cs),
              "xavierGpu_hosts_types": sorted({t for c in cs for t in c["xavierGpu_types"]}), "median_elapsed": st.median(c["elapsed"] for c in cs)}
f, fa = res["first_compatible"], res["fastest_compatible"]
res["delta_points"] = 100 * (fa["median_accel_share"] - f["median_accel_share"])
res["W1_pass"] = bool(res["delta_points"] >= 15.0 and fa["xavierGpu_hosts_types"])
(a.out / "w1.json").write_text(json.dumps(res, indent=1))
print(json.dumps({k: res[k] for k in ("tag", "first_compatible", "fastest_compatible", "delta_points", "W1_pass")}, indent=1))
