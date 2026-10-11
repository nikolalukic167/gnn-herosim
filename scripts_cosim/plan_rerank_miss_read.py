#!/usr/bin/env python3
"""Read plan_rerank_miss.py output: per rung, the cause of each argmin-S miss = the component group with the largest positive
(S-plan minus optimal-plan) difference in replayed seconds; the gap distribution in seconds per cause and each cause's share of the
total regret seconds. Also the replay fidelity (replayed vs label) and wall seconds per replayed plan.

  plan_rerank_miss_read.py miss.jsonl [--out read.json]
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("miss")
    ap.add_argument("--out")
    a = ap.parse_args()
    R = [json.loads(l) for l in open(a.miss)]
    err = [r for r in R if "error" in r]
    ok = [r for r in R if "error" not in r]
    print(f"rows {len(R)}, errors {len(err)}")
    for r in err[:5]:
        print("  ERR", r["ds"], r["error"][:200])
    out = {}
    for rung in ("all", "moderate", "heavy"):
        v = [r for r in ok if rung == "all" or r["rung"] == rung]
        if not v:
            continue
        fid = [max(abs(r["s_replayed"] - r["s_label"]) / r["s_label"], abs(r["opt_replayed"] - r["opt"]) / r["opt"]) for r in v]
        bad = sum(f > 0.05 for f in fid)
        by = defaultdict(list)
        for r in v:
            d = r["delta"]
            g = max((k for k in d if k != "elapsed"), key=lambda k: d[k])
            if d[g] <= 0:
                g = "none_positive"
            r["cause"] = g
            by[g].append(r)
        tot = sum(r["gap_label"] for r in v)
        res = {"n": len(v), "regret_s": tot, "replay_off_gt5pct": bad, "fid_median": st.median(fid),
               "stack_s_gt_opt": sum(r["s_stack"] > r["opt_stack"] for r in v) / len(v),
               "stack_s_lt_opt": sum(r["s_stack"] < r["opt_stack"] for r in v) / len(v),
               "mean_delta_s": {k: st.mean(r["delta"][k] for r in v) for k in v[0]["delta"]},
               "sum_delta_s": {k: sum(r["delta"][k] for r in v) for k in v[0]["delta"]},
               "causes": {}}
        for g, rs in sorted(by.items(), key=lambda kv: -sum(r["gap_label"] for r in kv[1])):
            gaps = [r["gap_label"] for r in rs]
            res["causes"][g] = {"n": len(rs), "share_tables": len(rs) / len(v), "share_regret_s": sum(gaps) / tot,
                                "gap_s_median": st.median(gaps), "gap_s_p10": q(gaps, .1), "gap_s_p90": q(gaps, .9), "gap_s_max": max(gaps),
                                "stack_s_gt_opt": sum(r["s_stack"] > r["opt_stack"] for r in rs) / len(rs)}
        walls = [w for r in v for w in r["wall_per_plan"]]
        res["wall_per_plan_s"] = {"median": st.median(walls), "p90": q(walls, .9), "max": max(walls),
                                  "setup_median": st.median(r["wall_setup"] for r in v)}
        out[rung] = res
        print(f"== {rung}: {len(v)} misses, regret {tot:.2f} s, replay off >5 %: {bad} (median rel diff {res['fid_median']:.2e}); "
              f"S-plan stacks more tasks on one platform than opt in {100 * res['stack_s_gt_opt']:.0f} % (fewer {100 * res['stack_s_lt_opt']:.0f} %); "
              f"wall per replayed plan median {res['wall_per_plan_s']['median']:.2f} s p90 {res['wall_per_plan_s']['p90']:.2f} s, setup {res['wall_per_plan_s']['setup_median']:.2f} s")
        print("   sum delta (S-plan - opt) s: " + ", ".join(f"{k} {x:+.2f}" for k, x in res["sum_delta_s"].items()))
        for g, c in res["causes"].items():
            print(f"   {g:14s} n {c['n']:3d} ({100 * c['share_tables']:4.0f} %) regret share {100 * c['share_regret_s']:5.1f} % | gap s median {c['gap_s_median']:.3f} "
                  f"p10 {c['gap_s_p10']:.3f} p90 {c['gap_s_p90']:.3f} max {c['gap_s_max']:.3f} | S stacks more {100 * c['stack_s_gt_opt']:.0f} %")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
