#!/usr/bin/env python3
"""kpa_scaleout_v1 on R1.1 against the published gates, one condition (docs/lineages/workload_fix_v1.md, re-measure scope).

For condition COND: (1) completion, failed and timed-out runs and request failures per arm x rung in the _r11 directory;
(2) the r11 vs-CD table (kpa_scaleout_v1_read.py --suffix _r11 --conds COND; Holm over that condition's family only) next to
the published kpa_read.json row and label (Holm over the registered 60); (3) per arm x rung the paired r11/published
latency change over cells (topology, window, seed): median %, max |%|, cells that differ at all.

  kpa_r11_compare.py --cond pipe_release --out kpa_r11_pipe_release_read.json      (run on datalab)
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import statistics as st
import subprocess
import sys
import tempfile

B = "/home/nikola.lukic/gnn-herosim/simulation_data/"
HERE = os.path.dirname(os.path.abspath(__file__))
DIRS = {"sf_held": "gate_replay_kpa", "pipe": "gate_pipe_kpa", "release": "gate_release_kpa", "pipe_release": "gate_pipe_release_kpa"}
RUNGS = ("20", "30", "50")


def cells(directory: str, pattern: str):
    out = {}
    for f in glob.glob(f"{B}kpa_scaleout_v1/{directory}/{pattern}"):
        r = json.load(open(f))
        arm = r["arm"].split("__")
        out[(int(r["topology"]), r["window"], arm[2].rsplit("_s", 1)[0], int(r.get("checkpoint_seed", 0)))] = r
    return out


def rung_of(window: str) -> str:
    return window.split("x")[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cond", default="pipe_release", choices=sorted(DIRS))
    ap.add_argument("--suffix", default="_r11")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    new_ok, new_bad = cells(DIRS[a.cond] + a.suffix, "*.summary.json"), cells(DIRS[a.cond] + a.suffix, "*.failed.json")
    old_ok = cells(DIRS[a.cond], "*.summary.json")
    with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
        subprocess.run([sys.executable, os.path.join(HERE, "kpa_scaleout_v1_read.py"), "--suffix", a.suffix, "--conds", a.cond,
                        "--out", tmp.name], check=True, stdout=subprocess.DEVNULL)
        new_read = json.load(open(tmp.name))
    published = json.load(open(B + "kpa_scaleout_v1/kpa_read.json"))
    result = {"cond": a.cond, "suffix": a.suffix, "completion": {}, "vs_cd": {}, "paired_change": {}}
    arms = sorted({k[2] for k in list(new_ok) + list(old_ok) + list(new_bad)})
    for rung in RUNGS:
        for arm in arms:
            key = f"{rung}/{arm}"
            ok = [r for k, r in new_ok.items() if k[2] == arm and rung_of(k[1]) == rung]
            bad = [r for k, r in new_bad.items() if k[2] == arm and rung_of(k[1]) == rung]
            n_old = sum(1 for k in old_ok if k[2] == arm and rung_of(k[1]) == rung)
            rf = [r.get("requestFailures") for r in ok]
            result["completion"][key] = {
                "completed": len(ok), "published_completed": n_old, "failed": len(bad),
                "timed_out": sum(1 for r in bad if r.get("why") == "timeout"),
                "request_failures": None if any(x is None for x in rf) else int(sum(rf)),
                "runs_with_request_failure": None if any(x is None for x in rf) else sum(x > 0 for x in rf)}
            deltas = [100 * (r["averageElapsedTime"] / old_ok[k]["averageElapsedTime"] - 1)
                      for k, r in new_ok.items() if k[2] == arm and rung_of(k[1]) == rung and k in old_ok]
            result["paired_change"][key] = (
                {"n_cells": len(deltas), "median_pct": st.median(deltas), "max_abs_pct": max(abs(d) for d in deltas),
                 "n_differ": sum(d != 0 for d in deltas)} if deltas else None)
            n, o = new_read[a.cond][rung].get(arm), (published.get(a.cond) or {}).get(rung, {}).get(arm)
            keep = ("lat", "qshare", "vs_cd", "faster", "n_pc", "p", "p_holm", "label")
            result["vs_cd"][key] = {"r11": {k: n[k] for k in keep if n and k in n}, "published": {k: o[k] for k in keep if o and k in o}}
    result["_note"] = ("r11 Holm is over this condition's 15 tests; published Holm is over the registered 60, so adjusted p are "
                       "not on one scale; compare vs_cd and the raw p")
    json.dump(result, open(a.out, "w"), indent=1)
    for key in result["completion"]:
        c, pc, v = result["completion"][key], result["paired_change"][key], result["vs_cd"][key]
        fmt = lambda d: (f"{d['vs_cd']:+6.1f}% {d['faster']}/{d['n_pc']} {d['label']}" if "vs_cd" in d else "(CD)")
        print(f"{key:24s} done {c['completed']:3d}/{c['published_completed']:3d} fail {c['failed']} reqfail {c['request_failures']} | "
              f"r11 {fmt(v['r11'])} | pub {fmt(v['published'])} | dlat med {pc['median_pct']:+.3f}% max {pc['max_abs_pct']:.3f}% differ {pc['n_differ']}/{pc['n_cells']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
