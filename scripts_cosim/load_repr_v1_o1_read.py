#!/usr/bin/env python3
"""load_repr_v1 O1 -- offline held-out regret of v4load / v4twin next to CD and jb2 gnnedge0
(docs/lineages/load_repr_v1.md). Offline: orders nothing, closes nothing (rule 6).

Protocol is cd_gap_v1 A1's: per held-out group, regret % = 100 x (decoded rtt - sweep optimum) / optimum,
learned arm = median over its seeds; CD from the D0 engine replay.

  load_repr_v1_o1_read.py --lr1-dir DIR --d0-dir DIR [--out F]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from statistics import mean, median

JB2 = (1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 14, 15, 16)


def _test_rows(path):
    return {r["dataset_id"]: r for r in json.load(open(path))["per_dataset"] if r.get("split") == "test"}


def _pct(row):
    opt = float(row["constrained_optimum"]["mean_tied"])
    return 100.0 * (float(row["decoded_rtt"]) - opt) / opt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lr1-dir", required=True)
    ap.add_argument("--d0-dir", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    arms = {
        "v4load": [_test_rows(os.path.join(a.lr1_dir, f"eval_v4load_s{s}.json")) for s in (1, 2, 3, 4)],
        "v4twin": [_test_rows(os.path.join(a.lr1_dir, f"eval_v4twin_s{s}.json")) for s in (1, 2, 3, 4)],
        "gnnedge0": [_test_rows(os.path.join(a.d0_dir, f"eval_gnnedge0_s{s}.json")) for s in JB2],
    }
    ids = sorted(arms["v4load"][0])
    for name, reps in arms.items():
        for rows in reps:
            if sorted(rows) != ids:
                raise SystemExit(f"FAIL LOUD: {name} held-out groups differ from v4load s1's")
    rule = {r["dataset"]: r for r in json.load(open(os.path.join(a.d0_dir, "offline_rule_regret_jb2_heldout.json")))["rows"]}
    per = {"cd": {g: 100.0 * float(rule[g.split("/")[-1]]["cd_greedy"]["regret"])
                  / float(rule[g.split("/")[-1]]["optimal_rtt"]) for g in ids}}
    for name, reps in arms.items():
        per[name] = {g: median(_pct(rows[g]) for rows in reps) for g in ids}
    out = {"n_groups": len(ids),
           "regret_pct_median": {k: median(v.values()) for k, v in per.items()},
           "regret_pct_mean": {k: mean(v.values()) for k, v in per.items()},
           "per_seed_median": {k: [median(_pct(rows[g]) for g in ids) for rows in reps] for k, reps in arms.items()},
           "paired_median_pp": {
               "v4load_minus_v4twin": median(per["v4load"][g] - per["v4twin"][g] for g in ids),
               "v4load_minus_cd": median(per["v4load"][g] - per["cd"][g] for g in ids),
               "v4load_minus_gnnedge0": median(per["v4load"][g] - per["gnnedge0"][g] for g in ids)},
           "paired_mean_pp": {
               "v4load_minus_v4twin": mean(per["v4load"][g] - per["v4twin"][g] for g in ids),
               "v4load_minus_cd": mean(per["v4load"][g] - per["cd"][g] for g in ids)},
           "v4load_better_than_v4twin_groups": sum(per["v4load"][g] < per["v4twin"][g] for g in ids),
           "v4load_worse_than_v4twin_groups": sum(per["v4load"][g] > per["v4twin"][g] for g in ids)}
    print(json.dumps(out, indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
