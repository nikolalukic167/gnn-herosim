#!/usr/bin/env python3
"""client_local_v1 feasibility reader: does letting a call run on its own client change latency?

  client_local_v1_read.py --gate <client_local_v1 gate dir> [--out]

Per rule: its client-enabled run vs its server-only run on the same (topology, window) (server-only runs from
small_batch_confirm_v1 and rule_baselines_v1); per topology the median paired %, median over the 19 topologies, exact
Wilcoxon. Also each client-enabled rule vs client-enabled CD, and the share of calls each rule ran locally.
Client-enabled summaries carry kind names of their own here (suffix "@cl") so the two runs of a rule do not collide.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from statistics import median

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fresh_topo_burst_v1_read as F  # noqa: E402
import local_features_v1_read as L  # noqa: E402
from peak_controls_v1_read import RUNGS, describe  # noqa: E402
from peak_load_v1_read import contrast  # noqa: E402

D = "/home/nikola.lukic/gnn-herosim/simulation_data"
RULES = ("reactive", "cd", "locality", "selfpredict", "batched", "offload", "localfirst")
SERVER_ONLY = ("reactive", "cd", "locality", "selfpredict", "batched")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    s = {}
    for d in (f"{D}/small_batch_confirm_v1/gate", f"{D}/local_features_v1/ref_retry", f"{D}/rule_baselines_v1/gate"):
        s.update(F._summaries(d))
    for (t, w, k, sd), r in F._summaries(a.gate).items():
        s[(t, w, f"{k}@cl", sd)] = r
    topos = json.load(open(f"{D}/small_batch_confirm_v1/inputs/selected.json"))["topologies"]
    out = {"vs_server_only": {}, "vs_cd_client": {}, "local_share_pct": {}, "arms": {}}
    for r in RUNGS:
        ws = L._windows(r)
        for k in SERVER_ONLY:
            out["vs_server_only"][f"{r}/{k}"] = contrast(s, topos, ws, f"{k}@cl", k)
        for k in RULES:
            if k != "cd":
                out["vs_cd_client"][f"{r}/{k}"] = contrast(s, topos, ws, f"{k}@cl", "cd@cl")
            rows = [v for (t, w, kk, _), v in s.items() if kk == f"{k}@cl" and t in topos and w in ws]
            out["local_share_pct"][f"{r}/{k}"] = (median(100.0 - float(v.get("offloadingRate") or 100.0) for v in rows)
                                                  if rows else None)
        out["arms"][r] = {x: describe(s, topos, ws, x) for x in [f"{k}@cl" for k in RULES] + list(SERVER_ONLY) + [L.SB1]}
    for name in ("vs_server_only", "vs_cd_client"):
        for k, c in out[name].items():
            print(f"[{name}] {k:22s} {c.get('median_pct', math.nan):+8.2f} %  p={c.get('p', math.nan):.4f}  "
                  f"{c.get('faster')}/{c['n_topologies']}  {c['label']}", file=sys.stderr)
    for k, v in out["local_share_pct"].items():
        print(f"[local %] {k:22s} {v if v is None else round(v, 2)}", file=sys.stderr)
    for r, d in out["arms"].items():
        print(r, "  ".join(f"{x}:{v.get('median_latency', math.nan):.0f}" for x, v in d.items()), file=sys.stderr)
    if a.out:
        with open(a.out + ".partial", "w") as fh:
            json.dump(out, fh, indent=1)
        os.replace(a.out + ".partial", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
