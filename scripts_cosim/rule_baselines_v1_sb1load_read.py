#!/usr/bin/env python3
"""rule_baselines_v1, descriptive addendum: sb1load (small_batch_confirm_v1, 8 seeds) vs CD, cdextr, reactive and the six
rules on the same 19 topologies; Holm across 27. Run from the repo root on datalab."""
import json, math, sys
sys.path.insert(0, "scripts_cosim")
import local_features_v1_read as L
import fresh_topo_burst_v1_read as F
from peak_controls_v1_read import RUNGS, describe, holm
from peak_load_v1_read import label
D = "/home/nikola.lukic/gnn-herosim/simulation_data"
s = {}
for d in (f"{D}/small_batch_confirm_v1/gate", f"{D}/local_features_v1/ref_retry", f"{D}/rule_baselines_v1/gate"):
    s.update(F._summaries(d))
topos = json.load(open(f"{D}/small_batch_confirm_v1/inputs/selected.json"))["topologies"]
REFS = ("cd", "cdextr", "reactive", "random", "drain", "locality", "decima", "batched", "selfpredict")
tests = {f"{r}/{k}": L._c(L.EIGHT, s, topos, L._windows(r), L.SB1, k) for r in RUNGS for k in REFS}
adj = holm({k: c["p"] for k, c in tests.items() if c.get("p") is not None})
for k, c in tests.items():
    print(f"{k:16s} {c.get('median_pct', math.nan):+8.2f} %  holm={adj.get(k, math.nan):.4f}  {c.get('faster')}/{c['n_topologies']}  "
          f"{label(c['median_pct'], adj[k]) if k in adj else c['label']}")
for r in RUNGS:
    d = {a: describe(s, topos, L._windows(r), a) for a in (L.SB1,) + REFS}
    print(r, "  ".join(f"{a.replace('_selfref','')}:{v['median_latency']:.0f}/{v['median_queue']:.0f}" for a, v in d.items()))
