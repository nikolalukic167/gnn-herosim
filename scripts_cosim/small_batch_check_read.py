#!/usr/bin/env python3
"""small_batch_v1 check A reader (docs/lineages/small_batch_v1.md).

  small_batch_check_read.py --old-diag <qgap dir, 10-task VAL> --new-diag <qgap dir, small-batch VAL>
      --old-rules <json> --new-rules <json> --old-root <simulation_data of the 10-task corpus>
      --new-root <simulation_data of the small-batch corpus> --cells 9207 9208 9209 [--out read.json]

Both sides are restricted to the same validation cells (the 10-task VAL also holds 9213, which the small-batch capture
lost to a hung run). Per arm: mean val regret (4 seeds averaged first, then graphs) on each VAL, and the ratio to the
CD-replica regret on the same graphs. Bar (signed before the data): xs1load's (psignn) ratio rising by more than 25 %
from 10-task to small batches is DEGRADES, within +-10 % ROBUST, otherwise MIXED; the same call is made for every arm.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src" / "notebooks"))
ARMS = ("psignn", "psimlp", "rawgnn", "rawmlp")


def cell_of(root: str, gid: str) -> int:
    from non_unique_lib.training_contract import canonical_parent_id

    parent = canonical_parent_id(gid)
    meta = json.load(open(os.path.join(root, parent, "infrastructure.json"))).get("metadata") or {}
    return int((meta.get("warm_snapshot") or {})["cell_seed"])


def load_diag(d: str, refine: int):
    out = defaultdict(dict)
    for f in glob.glob(os.path.join(d, f"*_r{refine}.json")):
        arm, s, _ = os.path.basename(f)[:-5].split("_")
        if arm in ARMS:
            out[arm][s] = {r["id"]: r["regret"] for r in json.load(open(f))["rows"]}
    for arm, seeds in out.items():
        if len(seeds) != 4:
            raise SystemExit(f"FAIL LOUD: {d} {arm} has seeds {sorted(seeds)}, expected 4")
    return out


def side(diag_dir, rules_json, root, cells, refine):
    diag = load_diag(diag_dir, refine)
    rules = {r["id"]: r for r in json.load(open(rules_json))["rows"]}
    keep = {i for i in rules if cell_of(root, i) in cells}
    res = {"n_graphs": len(keep), "cd": mean(rules[i]["regret_cd"] for i in keep),
           "indep": mean(rules[i]["regret_indep"] for i in keep),
           "mean_tasks": mean(rules[i]["n_tasks"] for i in keep), "arms": {}}
    for arm, seeds in diag.items():
        ids = keep & set.intersection(*[set(v) for v in seeds.values()])
        res["arms"][arm] = {"regret": mean(mean(seeds[s][i] for s in seeds) for i in ids), "n": len(ids)}
    return res


def call(change):
    return "DEGRADES" if change > 0.25 else "ROBUST" if abs(change) <= 0.10 else "MIXED"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("old-diag", "new-diag", "old-rules", "new-rules", "old-root", "new-root"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--cells", type=int, nargs="+", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    cells = set(a.cells)
    res = {}
    for refine in (0, 3):
        old = side(a.old_diag, a.old_rules, a.old_root, cells, refine)
        new = side(a.new_diag, a.new_rules, a.new_root, cells, refine)
        r = {"old": old, "new": new, "ratio": {}}
        print(f"\n=== self-refine {refine}: 10-task VAL {old['n_graphs']} graphs (mean {old['mean_tasks']:.1f} tasks), "
              f"small-batch VAL {new['n_graphs']} graphs (mean {new['mean_tasks']:.1f} tasks)")
        print(f"CD-replica regret {old['cd']:.2f} -> {new['cd']:.2f} s;  independent argmin {old['indep']:.2f} -> {new['indep']:.2f} s")
        for arm in ARMS:
            if arm in old["arms"] and arm in new["arms"]:
                o, n = old["arms"][arm]["regret"], new["arms"][arm]["regret"]
                ro, rn = o / old["cd"], n / new["cd"]
                r["ratio"][arm] = {"regret_old": o, "regret_new": n, "ratio_old": ro, "ratio_new": rn,
                                   "change": rn / ro - 1, "call": call(rn / ro - 1)}
                print(f"  {arm:7s} regret {o:6.2f} -> {n:6.2f} s   ratio to CD-replica {ro:5.2f} -> {rn:5.2f}  "
                      f"({100 * (rn / ro - 1):+.0f} %)  {call(rn / ro - 1)}")
        res[str(refine)] = r
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
