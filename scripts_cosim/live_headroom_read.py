#!/usr/bin/env python3
"""live_headroom_v1 reader (docs/lineages/live_headroom_v1.md).

  live_headroom_read.py --diag <dir of <arm>_s<seed>_r<refine>.json> --rules <json> --cache <graph cache>
                        --root <simulation_data holding the corpus> [--out read.json]

Per rung (x20 / x30 / x50, from each dataset's source tag): the optimum batch RTT, and the regret of the CD cost model's plan,
the independent argmin and each learned arm (self-refine 0 and 3, 4 seeds averaged first) against the full sweep, in seconds
and as % of the optimum, on live states (no synthetic backlog). Headroom reading: CD-replica regret as % of the optimum.
Pre-registered calls (signed before the data): CLOSED if the CD replica is within 5 % of the optimum at every rung,
OPEN if it is above 10 % at any rung, otherwise MIDDLE. Offline, per batch, the CD replica rather than the live CD arm.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src" / "notebooks"))
ARMS = ("psignn", "psimlp", "rawgnn", "rawmlp")


def rung_of(root: str, gid: str) -> str:
    from non_unique_lib.training_contract import canonical_parent_id

    parent = canonical_parent_id(gid)
    meta = json.load(open(os.path.join(root, parent, "infrastructure.json"))).get("metadata") or {}
    tag = ((meta.get("warm_snapshot") or {}).get("source_tag")) or ""
    for r in ("x20", "x30", "x50"):
        if f"_{r}_" in tag:
            return r
    raise SystemExit(f"FAIL LOUD: no rung in source tag {tag!r} of {gid}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("diag", "rules", "cache", "root"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    rules = {r["id"]: r for r in json.load(open(a.rules))["rows"]}
    opt = pickle.load(open(os.path.join(a.cache, "optimal_rtt.pkl"), "rb"))
    rung = {i: rung_of(a.root, i) for i in rules}
    diag = defaultdict(lambda: defaultdict(dict))
    for f in glob.glob(os.path.join(a.diag, "*.json")):
        arm, s, r = os.path.basename(f)[:-5].split("_")
        if arm in ARMS:
            diag[(arm, int(r[1:]))][s] = {x["id"]: x["regret"] for x in json.load(open(f))["rows"]}
    for k, seeds in diag.items():
        if len(seeds) != 4:
            raise SystemExit(f"FAIL LOUD: {k} has seeds {sorted(seeds)}, expected 4")
    res = {}
    for r in ("x20", "x30", "x50"):
        ids = [i for i in rules if rung[i] == r]
        o = [float(opt.get(i, opt.get(i.split("@seq", 1)[0]))) for i in ids]
        row = {"n": len(ids), "mean_tasks": mean(rules[i]["n_tasks"] for i in ids), "opt_rtt": mean(o),
               "cd": {"regret": mean(rules[i]["regret_cd"] for i in ids)},
               "indep": {"regret": mean(rules[i]["regret_indep"] for i in ids)}, "arms": {}}
        for nm in ("cd", "indep"):
            row[nm]["pct_of_opt"] = 100 * row[nm]["regret"] / row["opt_rtt"]
        for (arm, refine), seeds in diag.items():
            keep = [i for i in ids if all(i in v for v in seeds.values())]
            g = mean(mean(seeds[s][i] for s in seeds) for i in keep)
            row["arms"][f"{arm}_r{refine}"] = {"regret": g, "pct_of_opt": 100 * g / row["opt_rtt"], "n": len(keep)}
        res[r] = row
        print(f"--- {r}: {row['n']} batches, mean {row['mean_tasks']:.1f} tasks, optimum batch RTT {row['opt_rtt']:.1f} s")
        print(f"   CD-replica regret {row['cd']['regret']:6.2f} s = {row['cd']['pct_of_opt']:5.1f} % of optimum   "
              f"independent argmin {row['indep']['regret']:6.2f} s = {row['indep']['pct_of_opt']:5.1f} %")
        for k, v in sorted(row["arms"].items()):
            print(f"   {k:10s} {v['regret']:6.2f} s = {v['pct_of_opt']:5.1f} %")
    pct = [res[r]["cd"]["pct_of_opt"] for r in res]
    res["call"] = "CLOSED" if max(pct) <= 5 else "OPEN" if max(pct) > 10 else "MIDDLE"
    print("headroom call (CD replica vs optimum, all rungs):", res["call"], [round(p, 1) for p in pct])
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
