#!/usr/bin/env python3
"""small_batch_v1 Phase D read (docs/lineages/small_batch_v1.md): sb1load / sb1mpoff vs CD, cdextr, xs1load, xs1mpoff on
the 12 development topologies, grounded x2/x3/x5. Run from a worktree root on datalab; writes gates/sb1dev_read.json."""
import json, sys, os
sys.path.insert(0, "scripts_cosim")
import raw_plan_v2_read as R
import peak_load_v1_read as P
from peak_load_v1_read import contrast
P.LEARNED = tuple(P.LEARNED) + ("sb1load_selfref", "sb1mpoff_selfref", "xs1mpoff_selfref")
G = os.path.expanduser("~/gnn-herosim/simulation_data/backlog_corpus_v1/gates")
s = R._summaries([f"{G}/{d}" for d in ("groundedladder", "peakctl", "peakmlp", "sb1dev")])
topos = json.load(open(f"{G}/inputs/selected.json"))["topologies"]
arms = sorted({k[2] for k in s})
print("arms:", arms)
fam = [("sb1load_selfref","cd"),("sb1load_selfref","cdextr"),("sb1load_selfref","sb1mpoff_selfref"),
       ("sb1load_selfref","xs1load_selfref"),("sb1mpoff_selfref","xs1mpoff_selfref"),("sb1mpoff_selfref","cd"),
       ("sb1mpoff_selfref","cdextr"),("xs1load_selfref","xs1mpoff_selfref")]
fam += [("sb1load_selfref", a) for a in arms if "mlp" in a and "raw" not in a]
out = {}
for r in R.RUNGS:
    print("---", r)
    for a, b in fam:
        c = contrast(s, topos, R._windows(r), a, b)
        out[f"{r}:{a}_vs_{b}"] = c
        print(f"{a:18s} vs {b:20s} {c.get('median_pct', float('nan')):+8.2f} %  p={c.get('p', float('nan')):.4f}  {c.get('faster')}/{c['n_topologies']}  {c['label']}")
json.dump(out, open(f"{G}/sb1dev_read.json", "w"), indent=1)
