#!/usr/bin/env python3
"""accel_replica_v1 W1: how much of CD's work runs on an accelerator, with and without the rule, on the SAME calibrated cells.

For one calibrated heavy rung (a tag from scale_rung_bisect.py's inputs) this runs peer_greedy_network_cd on every seed x window cell twice -- the cell as
minted (preinit.replica_placement_rule = fastest_compatible) and a copy with the key removed (first_compatible) -- through physics_audit/run_cell.sh with
SIM_FORCE_FULL_STATS=1, and reads taskResults: the share of tasks on xavierGpu / xavierDla / pynqFpga, and which task types ran on xavierGpu.
Bar (amended 2026-10-10): accel share (fastest at its calibrated heavy rung) >= accel share (first at ITS calibrated heavy rung) + 15 points, AND xavierGpu
hosts >= 1 task type. Medians over the cells. The equal-load comparison (both rules at one rung) is descriptive. Conditions are `<rule>:<tag>`.

  accel_w1.py --root <calib root with inputs/> --conditions first_compatible:m23p2278 fastest_compatible:m27p6227 fastest_compatible:m23p2278 --out <dir> --wt <worktree> --seeds 9905 9906 9907 9908
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
ap.add_argument("--root", type=Path, required=True); ap.add_argument("--conditions", nargs="+", required=True); ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--wt", type=Path, required=True); ap.add_argument("--seeds", type=int, nargs="+", required=True)
ap.add_argument("--windows", nargs="+", default=["g0", "g1"]); ap.add_argument("--parallel", type=int, default=8); ap.add_argument("--limit", type=int, default=2700)
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
CONDS = [tuple(c.split(":")) for c in a.conditions]
for rule, tag in CONDS:
    src = a.root / "inputs" / f"grounded_{tag}"
    d = a.out / f"inputs_{rule}" / f"grounded_{tag}"
    (d / "cfg").mkdir(parents=True, exist_ok=True)
    if not (d / "wl").exists():
        (d / "wl").symlink_to((src / "wl").resolve())
    for sd in a.seeds:
        c = json.loads((src / "cfg" / f"cc40s{sd}.json").read_text())
        pre = c.setdefault("preinit", {})
        assert pre.get("replica_placement_rule") == "fastest_compatible", f"cell {sd} was not minted under fastest_compatible"
        if rule == "first_compatible":
            pre.pop("replica_placement_rule")
        (d / "cfg" / f"cc40s{sd}.json").write_text(json.dumps(c, indent=1))


def run(job):
    rule, tag, s, w = job
    raw = a.out / f"{rule}_{tag}_{s}_{w}.raw.json"
    if not raw.exists():
        env = dict(os.environ, HEROSIM_PY=sys.executable, HEROSIM_AUDIT_INPUTS=str(a.out / f"inputs_{rule}"), TS="1.0", SIM_FORCE_FULL_STATS="1",
                   HEROSIM_SHARED_AUTOSCALER="0", GATE_FIXED_POLICY_TIME_SCALE="1.0", OMP_NUM_THREADS="1")
        subprocess.run(["timeout", "--kill-after=60", str(a.limit), "bash", "scripts_cosim/physics_audit/run_cell.sh", str(s), w, tag, "peer_greedy_network_cd", str(raw), "50000"],
                       cwd=a.wt, env=env, stdout=open(a.out / f"{rule}_{tag}_{s}_{w}.log", "w"), stderr=subprocess.STDOUT)
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
    return dict(rule=rule, tag=tag, seed=s, window=w, n=n, accel_share=sum(by_plat.get(p, 0) for p in ACC) / n, by_platform=by_plat, xavierGpu_types=gpu_types,
                elapsed=d["stats"]["averageElapsedTime"], end=d["stats"]["endTime"])


jobs = [(r, t, sd, w) for r, t in CONDS for sd in a.seeds for w in a.windows]
with ThreadPoolExecutor(a.parallel) as ex:
    cells = list(ex.map(run, jobs))
res = {"conditions": a.conditions, "cells": cells}
for r, t in CONDS:
    cs = [c for c in cells if c["rule"] == r and c["tag"] == t]
    res[f"{r}:{t}"] = {"median_accel_share": st.median(c["accel_share"] for c in cs), "min": min(c["accel_share"] for c in cs), "max": max(c["accel_share"] for c in cs),
                       "xavierGpu_hosts_types": sorted({ty for c in cs for ty in c["xavierGpu_types"]}), "median_elapsed": st.median(c["elapsed"] for c in cs),
                       "pooled_by_platform": {p: sum(c["by_platform"].get(p, 0) for c in cs) for p in sorted({p for c in cs for p in c["by_platform"]})}}
(a.out / "w1.json").write_text(json.dumps(res, indent=1))
print(json.dumps({k: v for k, v in res.items() if k != "cells"}, indent=1))
