#!/usr/bin/env python3
"""wide_choice_s0_v1 read (docs/lineages/wide_choice_s0_v1.md): W1-W3 on tier A, the same reads
reported on tier B, and CD regret by slate width.

Usage: wide_choice_s0_v1_read.py --tier A rowsA.json manifestA.jsonl --tier B rowsB.json manifestB.jsonl --out read.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from typing import Any, Dict, List

SEEDS = (1, 2, 3, 4)
MODES = ("onepass", "selfref", "cdapply")


def _q(xs: List[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def _draws(manifest: str) -> Dict[str, Dict[str, Any]]:
    out = {}
    for line in open(manifest):
        d = json.loads(line)
        if d.get("status") == "success":
            out[d["dataset_id"]] = d["candidate_draw"]
    return out


def read_tier(rows: List[Dict[str, Any]], draws: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        raise SystemExit("FAIL LOUD: a tier has no rows")
    cd = [r["cd_greedy"]["regret_pct"] for r in rows]
    res: Dict[str, Any] = {"n_groups": len(rows)}
    cands = []
    for r in rows:
        d = draws.get(r["dataset"])
        if d is None:
            raise SystemExit(f"FAIL LOUD: {r['dataset']} has no manifest draw")
        pt = d["per_task_candidates"]
        cands.append(sum(pt) / len(pt))
        r["_cands"] = cands[-1]
    res["slate"] = {"cands_per_task_median": statistics.median(cands), "cands_per_task_q10_q90": [_q(cands, .1), _q(cands, .9)],
                    "n_plans_median": statistics.median(r["n_plans"] for r in rows),
                    "near5_share_median": statistics.median(r["near5_share"] for r in rows),
                    "batches_per_group_learned_max": max(r[f"xs1load_onepass_s{s}"]["n_batches"] for r in rows for s in SEEDS)}

    def arm(key_fn) -> Dict[str, Any]:
        v = [key_fn(r) for r in rows]
        return {"median": statistics.median(v), "mean": statistics.fmean(v), "share_ge20": sum(x >= 20 for x in v) / len(v)}

    res["cd"] = arm(lambda r: r["cd_greedy"]["regret_pct"])
    res["onepass_greedy"] = arm(lambda r: r["batched_greedy"]["regret_pct"])
    for m in MODES:
        res[f"xs1load_{m}"] = arm(lambda r, m=m: statistics.median(r[f"xs1load_{m}_s{s}"]["regret_pct"] for s in SEEDS))
    w1 = res["cd"]["median"]
    res["W1"] = {"cd_median_regret_pct": w1,
                 "label": "CD-WEAK" if w1 >= 20.0 else "CD-MIDDLE" if w1 >= 10.0 else "CD-STRONG"}
    for m in MODES:
        diffs = [statistics.median(r[f"xs1load_{m}_s{s}"]["regret_pct"] for s in SEEDS) - r["cd_greedy"]["regret_pct"]
                 for r in rows]
        res[f"W2_{m}"] = {"median_pp": statistics.median(diffs), "mean_pp": statistics.fmean(diffs),
                          "share_learned_better": sum(d < 0 for d in diffs) / len(diffs),
                          "share_cd_better": sum(d > 0 for d in diffs) / len(diffs),
                          "label": "LEARNED-AT-OR-BELOW-CD" if statistics.median(diffs) <= 0.0 else "LEARNED-ABOVE-CD"}
    res["W2"] = res["W2_onepass"]
    better = []
    for r in rows:
        l_rtt = statistics.median(r[f"xs1load_onepass_s{s}"]["rtt"] for s in SEEDS)
        better.append(min(l_rtt, r["cd_greedy"]["rtt"]) <= 0.9 * r["cd_greedy"]["rtt"])
    w3 = sum(better) / len(better)
    res["W3"] = {"share": w3, "label": "COMPLEMENTARY" if w3 >= 0.25 else "PARTIAL" if w3 >= 0.10 else "REDUNDANT"}
    by = {}
    for lo, hi in ((0, 2.0), (2.0, 2.75), (2.75, 3.25), (3.25, 99)):
        sub = [r for r in rows if lo <= r["_cands"] < hi]
        if sub:
            by[f"{lo}-{hi}"] = {"n": len(sub), "cd_median": statistics.median(r["cd_greedy"]["regret_pct"] for r in sub),
                                "xs1load_onepass_median": statistics.median(
                                    statistics.median(r[f"xs1load_onepass_s{s}"]["regret_pct"] for s in SEEDS) for r in sub)}
    res["by_cands_per_task"] = by
    for r in rows:
        r.pop("_cands", None)
    res["S0"] = "PASS" if (res["W1"]["label"] == "CD-WEAK" and res["W2"]["label"] == "LEARNED-AT-OR-BELOW-CD") else "FAIL"
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", nargs=3, action="append", metavar=("NAME", "ROWS", "MANIFEST"), required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {name: read_tier(json.load(open(rows))["rows"], _draws(man)) for name, rows, man in a.tier}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
