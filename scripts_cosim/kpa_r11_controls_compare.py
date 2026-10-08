#!/usr/bin/env python3
"""kpa_scaleout_v1 shared-autoscaler controls on R1.1 against the published shared_control_read.json (run on datalab).

Per condition x rung x arm (reactive, selfpredict): completion of the *_legacy_shared_r11 directory; the swap / kpa / vs-CD
numbers from kpa_shared_control_read.py --suffix _r11 next to the published ones, with a descriptive direction label from
the raw exact Wilcoxon p (< 0.05: FASTER/SLOWER by sign; else NOT-SEPARATED; no multiplicity correction, the control is not
a registered family); and the paired r11/published latency change per cell (topology, window) of the shared and kpa directories.

  kpa_r11_controls_compare.py --out kpa_r11_controls_read.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
import subprocess
import sys
import tempfile

B = "/home/nikola.lukic/gnn-herosim/simulation_data/"
HERE = os.path.dirname(os.path.abspath(__file__))
SHARED = {"sf_held": "gate_replay_legacy_shared", "pipe": "gate_pipe_legacy_shared", "release": "gate_release_legacy_shared",
          "pipe_release": "gate_pipe_release_legacy_shared"}
KPA = {"sf_held": "gate_replay_kpa", "pipe": "gate_pipe_kpa", "release": "gate_release_kpa", "pipe_release": "gate_pipe_release_kpa"}
KEYS = ("swap", "kpa", "vs_cd_legacy_shared", "vs_cd_legacy_original", "vs_cd_kpa")


def label(x):
    if not x:
        return None
    return "NOT-SEPARATED" if x["p"] >= 0.05 else ("FASTER" if x["median"] < 0 else "SLOWER")


def cells(directory):
    out = {}
    for f in glob.glob(f"{B}kpa_scaleout_v1/{directory}/*.summary.json"):
        r = json.load(open(f))
        out[(int(r["topology"]), r["window"], r["arm"].split("__")[2].rsplit("_s", 1)[0], int(r.get("checkpoint_seed", 0)))] = r
    return out


def failed(directory):
    return glob.glob(f"{B}kpa_scaleout_v1/{directory}/*.failed.json")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suffix", default="_r11")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
        subprocess.run([sys.executable, os.path.join(HERE, "kpa_shared_control_read.py"), "--suffix", a.suffix, "--out", tmp.name],
                       check=True, stdout=subprocess.DEVNULL)
        new = json.load(open(tmp.name))
    pub = json.load(open(B + "kpa_scaleout_v1/shared_control_read.json"))
    res = {"suffix": a.suffix, "completion": {}, "read": {}, "paired_change": {}}
    for cond in SHARED:
        n_ok, o_ok = cells(SHARED[cond] + a.suffix), cells(SHARED[cond])
        bad = failed(SHARED[cond] + a.suffix)
        rf = [r.get("requestFailures") for r in n_ok.values()]
        res["completion"][cond] = {"completed": len(n_ok), "published_completed": len(o_ok), "failed": len(bad),
                                   "request_failures": None if any(x is None for x in rf) else int(sum(rf))}
        kn_ok, ko_ok = cells(KPA[cond] + a.suffix), cells(KPA[cond])
        for which, nn, oo in (("shared", n_ok, o_ok), ("kpa", kn_ok, ko_ok)):
            for rung in ("20", "30", "50"):
                for arm in ("reactive", "selfpredict"):
                    d = [100 * (r["averageElapsedTime"] / oo[k]["averageElapsedTime"] - 1) for k, r in nn.items()
                         if k[2] == arm and k[1].endswith(f"x{rung}") and k in oo and k[3] == 0]
                    res["paired_change"][f"{cond}/{which}/{rung}/{arm}"] = (
                        {"n_cells": len(d), "median_pct": st.median(d), "max_abs_pct": max(abs(x) for x in d), "n_differ": sum(x != 0 for x in d)}
                        if d else None)
        for rung in ("20", "30", "50"):
            for arm in ("reactive", "selfpredict"):
                n, o = new[cond][rung][arm], pub[cond][rung][arm]
                res["read"][f"{cond}/{rung}/{arm}"] = {k: {"r11": n[k], "published": o[k], "r11_label": label(n[k]), "published_label": label(o[k])}
                                                       for k in KEYS if k in n}
    res["_note"] = "labels are descriptive: raw exact Wilcoxon p < 0.05 -> FASTER/SLOWER by sign, else NOT-SEPARATED; no Holm"
    json.dump(res, open(a.out, "w"), indent=1)
    f = lambda x: "—" if not x else f"{x['median']:+6.1f}%({x['faster']}/{x['n']}){label(x)[:3]}"
    for key, v in res["read"].items():
        print(f"{key:26s}" + " | ".join(f"{k} {f(v[k]['r11'])} vs {f(v[k]['published'])}" for k in ("swap", "kpa", "vs_cd_legacy_shared")))
    for c, v in res["completion"].items():
        print(c, v)
    for k, v in res["paired_change"].items():
        if v and v["n_differ"]:
            print("differ", k, v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
