#!/usr/bin/env python3
"""cd_gap_v1 D3 -- does the sum label trade the NEXT groups' queue for its own sum? (offline)

On each held-out group the sweep rows carry `task_times` ([[task, dispatched, done], ...]) and the
placement plan. For four plans on the same snapshot -- the label (sweep min of the summed elapsed),
the CD greedy's, the 1-pass greedy's (both matched to their sweep row by the engine replay's
total_rtt), and each jb2 `gnnedge0` checkpoint's decoded combo (matched by plan) -- report:

  sum      the group's summed elapsed (the label's objective; = the row's rtt)
  span     makespan: last done - first dispatched
  backlog  sum over platforms the plan uses of (that platform's last done - first dispatched):
           the busy time the group leaves on the platforms the next groups will find

A plan whose rtt matches several rows is ambiguous: every metric is reported as the mean over the
matching rows and the count of ambiguous groups is printed. A plan with no matching row is counted
and excluded, never guessed.

  cd_gap_v1_externality.py --heldout DIR --rule-json F --eval-glob 'DIR/eval_gnnedge0_s*.json' --out F
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys
from statistics import mean, median
from typing import Dict, List, Tuple


def _metrics(row: dict) -> Dict[str, float]:
    tt = row.get("task_times")
    if not tt:
        raise SystemExit("FAIL LOUD: sweep row without task_times")
    plan = row["placement_plan"]
    disp = {int(t): float(d) for t, d, _e in tt}
    done = {int(t): float(e) for t, _d, e in tt}
    per_plat: Dict[Tuple[int, int], List[int]] = collections.defaultdict(list)
    for t, np_ in plan.items():
        per_plat[(int(np_[0]), int(np_[1]))].append(int(t))
    t0 = min(disp.values())
    backlog = sum(max(done[t] for t in ts) - t0 for ts in per_plat.values())
    return {"sum": float(row["rtt"]), "span": max(done.values()) - t0, "backlog": backlog,
            "max_on_platform": max(len(v) for v in per_plat.values())}


def _combo_key(combo) -> Tuple[Tuple[int, int], ...]:
    return tuple((int(a), int(b)) for a, b in combo)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--rule-json", required=True)
    ap.add_argument("--eval-glob", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rule = {r["dataset"]: r for r in json.load(open(a.rule_json))["rows"]}
    evals = [json.load(open(p)) for p in sorted(glob.glob(a.eval_glob))]
    if not evals:
        raise SystemExit("FAIL LOUD: no eval reports")
    decoded: Dict[str, List[Tuple[Tuple[int, int], ...]]] = collections.defaultdict(list)
    for rep in evals:
        for row in rep["per_dataset"]:
            if row.get("split") == "test":
                decoded[row["dataset_id"].split("/")[-1]].append(_combo_key(row["decoded_combo"]))
    per_group = {}
    counts = collections.Counter()
    for ds in sorted(rule):
        rows = [json.loads(l) for l in open(os.path.join(a.heldout, ds, "placements", "placements.jsonl")) if l.strip()]
        by_plan = {}
        by_rtt = collections.defaultdict(list)
        for r in rows:
            pl = r["placement_plan"]
            key = tuple((int(pl[k][0]), int(pl[k][1])) for k in sorted(pl, key=int))
            by_plan[key] = r
            by_rtt[round(float(r["rtt"]), 6)].append(r)
        best = min(rows, key=lambda r: float(r["rtt"]))
        g = {"label": _metrics(best)}
        for arm in ("cd_greedy", "batched_greedy"):
            rtt = round(float(rule[ds][arm]["total_rtt"]), 6)
            match = by_rtt.get(rtt, [])
            if not match:
                counts[f"{arm}_unmatched"] += 1
                continue
            if len(match) > 1:
                counts[f"{arm}_ambiguous"] += 1
            ms = [_metrics(r) for r in match]
            g[arm] = {k: mean(m[k] for m in ms) for k in ms[0]}
        gm = []
        for key in decoded.get(ds, []):
            r = by_plan.get(key)
            if r is None:
                counts["gnnedge0_unmatched"] += 1
                continue
            gm.append(_metrics(r))
        if gm:
            g["gnnedge0"] = {k: median(m[k] for m in gm) for k in gm[0]}
        per_group[ds] = g
    out = {"n_groups": len(per_group), "counts": dict(counts), "summary": {}, "paired_vs_cd": {}}
    arms = ("label", "cd_greedy", "batched_greedy", "gnnedge0")
    for arm in arms:
        vals = [g[arm] for g in per_group.values() if arm in g]
        out["summary"][arm] = {k: {"mean": mean(v[k] for v in vals), "median": median(v[k] for v in vals)}
                               for k in ("sum", "span", "backlog", "max_on_platform")} | {"n": len(vals)}
    for arm in ("label", "gnnedge0", "batched_greedy"):
        both = [g for g in per_group.values() if arm in g and "cd_greedy" in g]
        out["paired_vs_cd"][arm] = {
            k: {"median_diff": median(g[arm][k] - g["cd_greedy"][k] for g in both),
                "mean_diff": mean(g[arm][k] - g["cd_greedy"][k] for g in both),
                "frac_arm_higher": sum(g[arm][k] > g["cd_greedy"][k] + 1e-9 for g in both) / len(both)}
            for k in ("sum", "span", "backlog")} | {"n": len(both)}
    json.dump({"out": out, "per_group": per_group}, open(a.out, "w"))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
