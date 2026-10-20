#!/usr/bin/env python3
"""Per-cell assertion that scaleOut.target equals the arm's REGISTERED value (a table here, not the environment), plus counts.
usage: dose_check.py <dose_dir>"""
import glob, json, os, sys
D = sys.argv[1]
REG = {"cells_ref_mod070": 0.7, "cells_gnn_mod070": 0.7, "cells_ref_mod045": 0.45, "cells_ref_mod0294": 0.294, "cells_ref_heavy070": 0.7, "cells_gnn_heavy070": 0.7}
EXP = {"cells_ref_mod070": 48, "cells_gnn_mod070": 96, "cells_ref_mod045": 48, "cells_ref_mod0294": 48, "cells_ref_heavy070": 48, "cells_gnn_heavy070": 96}
bad = 0
for root, want in REG.items():
    S = sorted(glob.glob(f"{D}/{root}/*.summary.json")); F = glob.glob(f"{D}/{root}/*.failed.json")
    wrong = [os.path.basename(p) for p in S if abs(json.load(open(p))["scaleOut"]["target"] - want) > 1e-12]
    print(f"{root}: summaries {len(S)}/{EXP[root]}, failed {len(F)}, target {want} wrong in {len(wrong)}" + (f" e.g. {wrong[:2]}" if wrong else ""))
    bad += len(wrong) + len(F)
print("ASSERT OK" if not bad else f"ASSERT FAILED ({bad})")
sys.exit(1 if bad else 0)
