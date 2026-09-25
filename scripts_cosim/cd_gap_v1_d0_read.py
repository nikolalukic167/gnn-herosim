#!/usr/bin/env python3
"""cd_gap_v1 D0 -- offline regret of the jb2 learned arms vs the CD greedy on the 480 held-out groups.

Inputs (all from datalab, see docs/lineages/cd_gap_v1.md): the engine replay of the hand rules
(`joint_burst_v1_offline_rule_regret.py` output), one `eval_route_b_stage2_arm.py` report per
checkpoint, and a dataset -> (cell, window) map built from each dataset's generation_provenance.json.

Per group: regret % = 100 * (rtt - sweep optimum) / optimum. A learned arm's group value is the
median over its checkpoints. The optimum of both sources must agree on every group (fail loud).

  cd_gap_v1_d0_read.py --dir <d0 dir> [--optima-plans plans.json] [--out read.json]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys
from statistics import mean, median
from typing import Dict, List, Optional

SEEDS = (1, 2, 3, 4, 5, 7, 8, 9, 10, 12, 14, 15, 16)


def _stack(combo: List[List[int]]) -> Dict[str, float]:
    per_plat = collections.Counter((int(n), int(p)) for n, p in combo)
    per_node = collections.Counter(int(n) for n, _p in combo)
    pairs_same_node = sum(c * (c - 1) // 2 for c in per_node.values())
    pairs_same_plat = sum(c * (c - 1) // 2 for c in per_plat.values())
    return {"max_on_platform": max(per_plat.values()), "distinct_platforms": len(per_plat),
            "pairs_same_node": pairs_same_node, "pairs_same_platform": pairs_same_plat}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--optima-plans", default=None, help="dataset -> sweep-optimal combo (best.json), optional")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    rule = {r["dataset"]: r for r in json.load(open(os.path.join(a.dir, "offline_rule_regret_jb2_heldout.json")))["rows"]}
    cells = json.load(open(os.path.join(a.dir, "heldout_cells.json")))
    learned: Dict[str, Dict[str, List[float]]] = {"gnnedge0": collections.defaultdict(list),
                                                  "mpoff": collections.defaultdict(list)}
    stack: Dict[str, List[dict]] = {"gnnedge0": [], "mpoff": []}
    for kind in learned:
        for s in SEEDS:
            rep = json.load(open(os.path.join(a.dir, f"eval_{kind}_s{s}.json")))
            n = 0
            for row in rep["per_dataset"]:
                if row.get("split") != "test":
                    continue
                ds = row["dataset_id"].split("/")[-1]
                if ds not in rule:
                    raise SystemExit(f"FAIL LOUD: {ds} in eval but not in the rule replay")
                opt = float(rule[ds]["optimal_rtt"])
                eo = float(row["constrained_optimum"]["mean_tied"])
                if abs(eo - opt) > 1e-6 * max(1.0, opt):
                    raise SystemExit(f"FAIL LOUD: optimum mismatch on {ds}: eval {eo} vs replay {opt}")
                if row.get("infeasible") or (row.get("decoder") or {}).get("relaxed"):
                    raise SystemExit(f"FAIL LOUD: {kind} s{s} {ds} infeasible/relaxed")
                learned[kind][ds].append(100.0 * (float(row["decoded_rtt"]) - opt) / opt)
                stack[kind].append(_stack(row["decoded_combo"]))
                n += 1
            if n != len(rule):
                raise SystemExit(f"FAIL LOUD: {kind} s{s} has {n} test groups, replay has {len(rule)}")
    groups = sorted(rule)
    out: Dict[str, object] = {"n_groups": len(groups)}
    cd = {ds: 100.0 * float(rule[ds]["cd_greedy"]["regret"]) / float(rule[ds]["optimal_rtt"]) for ds in groups}
    onep = {ds: 100.0 * float(rule[ds]["batched_greedy"]["regret"]) / float(rule[ds]["optimal_rtt"]) for ds in groups}
    g = {ds: median(learned["gnnedge0"][ds]) for ds in groups}
    m = {ds: median(learned["mpoff"][ds]) for ds in groups}

    def summ(x):
        v = [x[ds] for ds in groups]
        return {"median": median(v), "mean": mean(v)}

    out["regret_pct"] = {"cd": summ(cd), "1pass": summ(onep), "gnnedge0": summ(g), "mpoff": summ(m)}
    units = collections.defaultdict(list)
    for ds in groups:
        units[tuple(cells[ds])].append(ds)
    per_unit = {}
    for u, dss in sorted(units.items()):
        per_unit[f"cc40s{u[0]}_{u[1]}"] = {
            "n": len(dss),
            "cd_median": median(cd[d] for d in dss), "gnnedge0_median": median(g[d] for d in dss),
            "mpoff_median": median(m[d] for d in dss), "1pass_median": median(onep[d] for d in dss),
            "gnnedge0_minus_cd_paired_median": median(g[d] - cd[d] for d in dss),
            "gnnedge0_minus_mpoff_paired_median": median(g[d] - m[d] for d in dss),
            "gnnedge0_beats_cd_frac": sum(g[d] < cd[d] for d in dss) / len(dss),
        }
    out["per_cell_window"] = per_unit
    diff = median(g[d] - cd[d] for d in groups)
    below_everywhere = all(v["gnnedge0_median"] <= v["cd_median"] - 1.0 for v in per_unit.values())
    gm, cm = out["regret_pct"]["gnnedge0"]["median"], out["regret_pct"]["cd"]["median"]
    if gm >= cm:
        verdict = "FIT-GAP"
    elif cm - gm >= 1.0 and below_everywhere:
        verdict = "OFFLINE-LIVE-GAP"
    else:
        verdict = "MIXED"
    out["gnnedge0_minus_cd_paired_median_pp"] = diff
    out["D0"] = verdict
    out["D0b_gnnedge0_minus_mpoff_paired_median_pp"] = median(g[d] - m[d] for d in groups)
    out["stacking_decoded"] = {k: {f: mean(r[f] for r in v) for f in v[0]} for k, v in stack.items()}
    if a.optima_plans:
        plans = json.load(open(a.optima_plans))
        rows = [_stack(plans[ds]) for ds in groups if ds in plans]
        out["stacking_sweep_optimum"] = {f: mean(r[f] for r in rows) for f in rows[0]} | {"n": len(rows)}
    print(json.dumps(out, indent=1))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
