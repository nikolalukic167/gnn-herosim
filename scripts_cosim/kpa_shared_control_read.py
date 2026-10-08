#!/usr/bin/env python3
"""kpa_scaleout_v1 A4 control: separate the autoscaler swap from KPA for the two arms that changed autoscaler.

For reactive and selfpredict, per condition x rung, paired over (topology, window), median over topologies:
  swap   = legacy scale-out, shared autoscaler   vs legacy scale-out, original pairing (transfer_physics_v1)
  kpa    = kpa (shared)                          vs legacy scale-out, shared autoscaler
  vs_cd  = the arm vs CD under legacy + shared (CD never changed autoscaler), with the exact Wilcoxon p
Reported next to kpa_read.json's vs-CD numbers. Descriptive; not part of the registered family.

  kpa_shared_control_read.py [--out control.json]     (run on datalab)
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st

from scipy.stats import wilcoxon

B = "/home/nikola.lukic/gnn-herosim/simulation_data/"
TOPOS = json.load(open(B + "client_local_v1/inputs_so_server/selected.json"))["topologies"]
LEGACY = {"sf_held": ["client_local_v1/gate_so_server", "small_batch_so_v1/gate"],
          "pipe": ["transfer_physics_v1/gate_pipe"], "release": ["transfer_physics_v1/gate_release"],
          "pipe_release": ["transfer_physics_v1/gate_pipe_release"]}
SHARED = {"sf_held": "kpa_scaleout_v1/gate_replay_legacy_shared", "pipe": "kpa_scaleout_v1/gate_pipe_legacy_shared",
          "release": "kpa_scaleout_v1/gate_release_legacy_shared",
          "pipe_release": "kpa_scaleout_v1/gate_pipe_release_legacy_shared"}
KPA = {"sf_held": "kpa_scaleout_v1/gate_replay_kpa", "pipe": "kpa_scaleout_v1/gate_pipe_kpa",
       "release": "kpa_scaleout_v1/gate_release_kpa", "pipe_release": "kpa_scaleout_v1/gate_pipe_release_kpa"}


def load(dirs):
    s = {}
    for d in dirs:
        for f in glob.glob(B + d + "/*.summary.json"):
            r = json.load(open(f))
            k = r["arm"].split("__")[2].rsplit("_s", 1)[0]
            if int(r["checkpoint_seed"]) == 0:
                s[(int(r["topology"]), r["window"], k)] = r["averageElapsedTime"]
    return s


def paired(a, b, arm_a, arm_b, ws):
    per = []
    for t in TOPOS:
        p = [100 * (a[(t, w, arm_a)] / b[(t, w, arm_b)] - 1) for w in ws if (t, w, arm_a) in a and (t, w, arm_b) in b]
        if p:
            per.append(st.median(p))
    if not per:
        return None
    return {"median": st.median(per), "faster": sum(x < 0 for x in per), "n": len(per),
            "p": float(wilcoxon(per).pvalue) if any(per) else 1.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--suffix", default="", help="appended to the shared and kpa directory names (R1.1 re-measure: _r11); the legacy references are never suffixed")
    a = ap.parse_args()
    out = {}
    for cond in LEGACY:
        leg, sh, kp = load(LEGACY[cond]), load([SHARED[cond] + a.suffix]), load([KPA[cond] + a.suffix])
        out[cond] = {}
        for rung in ("20", "30", "50"):
            ws = [f"g{i}x{rung}" for i in range(4)]
            out[cond][rung] = {arm: {"swap": paired(sh, leg, arm, arm, ws), "kpa": paired(kp, sh, arm, arm, ws),
                                     "vs_cd_legacy_shared": paired(sh, leg, arm, "cd", ws),
                                     "vs_cd_legacy_original": paired(leg, leg, arm, "cd", ws),
                                     "vs_cd_kpa": paired(kp, kp, arm, "cd", ws)}
                               for arm in ("reactive", "selfpredict")}
            for arm, r in out[cond][rung].items():
                f = lambda x: "—" if x is None else f"{x['median']:+6.1f}% ({x['faster']}/{x['n']}, p {x['p']:.2g})"
                print(f"{cond:12s} x{rung[0]}.{rung[1]} {arm:11s} swap {f(r['swap'])} | kpa {f(r['kpa'])} | vs CD: "
                      f"legacy {f(r['vs_cd_legacy_original'])} shared {f(r['vs_cd_legacy_shared'])} kpa {f(r['vs_cd_kpa'])}")
    if a.out:
        open(a.out, "w").write(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
