#!/usr/bin/env python3
"""cd_gap_v1 A1 -- does the CD imitator reproduce CD offline? (docs/lineages/cd_gap_v1.md)

On the held-out (test-split) groups: each imitator seed's regret % against the true sweep (its
eval_route_b_stage2_arm.py report) and its exact plan agreement with the CD label (any of CD's tied
plans), next to CD's own regret and jb2 gnnedge0's (D0). Registered bar: IMITATES-CD iff the
imitator's median regret (per group median over seeds) is within 1 pp of CD's; else CANNOT-FIT-CD.

  cd_gap_v1_a1_read.py --a-dir DIR --d0-dir DIR [--out F]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from statistics import mean, median

SEEDS = (1, 2, 3, 4)
JB2 = (1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 14, 15, 16)


def _test_rows(path):
    return {r["dataset_id"]: r for r in json.load(open(path))["per_dataset"] if r.get("split") == "test"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a-dir", required=True)
    ap.add_argument("--d0-dir", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    labels = json.load(open(os.path.join(a.a_dir, "cd_labels_jb2.json")))
    rule = {r["dataset"]: r for r in json.load(open(os.path.join(a.d0_dir, "offline_rule_regret_jb2_heldout.json")))["rows"]}
    imit = [_test_rows(os.path.join(a.a_dir, f"eval_cdimit_s{s}.json")) for s in SEEDS]
    jb2 = [_test_rows(os.path.join(a.d0_dir, f"eval_gnnedge0_s{s}.json")) for s in JB2]
    ids = sorted(imit[0])
    for rows in imit + jb2:
        if sorted(rows) != ids:
            raise SystemExit("FAIL LOUD: test groups differ between reports")

    def pct(row):
        opt = float(row["constrained_optimum"]["mean_tied"])
        return 100.0 * (float(row["decoded_rtt"]) - opt) / opt

    out = {"n_groups": len(ids), "per_seed": {}}
    agree_all = []
    for s, rows in zip(SEEDS, imit):
        agree = []
        for gid in ids:
            plans = {tuple(map(tuple, p)) for p in labels[gid]["plans"]}
            agree.append(tuple(tuple(int(x) for x in c) for c in rows[gid]["decoded_combo"]) in plans)
        agree_all.append(agree)
        out["per_seed"][s] = {"regret_pct_median": median(pct(rows[g]) for g in ids),
                              "regret_pct_mean": mean(pct(rows[g]) for g in ids),
                              "cd_plan_agreement": sum(agree) / len(agree)}
    cd = {g: 100.0 * float(rule[g.split("/")[-1]]["cd_greedy"]["regret"]) / float(rule[g.split("/")[-1]]["optimal_rtt"])
          for g in ids}
    im = {g: median(pct(rows[g]) for rows in imit) for g in ids}
    gn = {g: median(pct(rows[g]) for rows in jb2) for g in ids}
    out["regret_pct_median"] = {"cd": median(cd.values()), "cdimit": median(im.values()), "gnnedge0": median(gn.values())}
    out["regret_pct_mean"] = {"cd": mean(cd.values()), "cdimit": mean(im.values()), "gnnedge0": mean(gn.values())}
    out["paired_median_pp"] = {"cdimit_minus_cd": median(im[g] - cd[g] for g in ids),
                               "gnnedge0_minus_cd": median(gn[g] - cd[g] for g in ids)}
    out["cd_plan_agreement_mean_over_seeds"] = mean(sum(x) / len(x) for x in agree_all)
    out["label_stats"] = {"groups": len(labels), "tied": sum(1 for v in labels.values() if len(v["plans"]) > 1),
                          "cd_at_optimum": sum(1 for v in labels.values()
                                               if abs(v["cd_rtt"] - v["opt_rtt"]) <= 1e-9 * max(1.0, v["opt_rtt"]))}
    out["A1"] = "IMITATES-CD" if out["regret_pct_median"]["cdimit"] - out["regret_pct_median"]["cd"] <= 1.0 else "CANNOT-FIT-CD"
    print(json.dumps(out, indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
