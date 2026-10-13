#!/usr/bin/env python3
"""uncut_groups_s0_v1 w(n) table: ms per plan by SERVED batch-size bucket from decisionTiming.by_bucket ({le10, 11_50, gt50}: calls, total_s,
plan_evals), per arm and rung, with each bucket's ratio to the le10 bucket and to the calibrated w in decision_cost_w.json. Registered check: if any
bucket differs from le10 by more than 1.5x, S6 freezes w(n) per bucket; otherwise the scalar w stands. Cells without the key fail loud.

  uncut_groups_wn_table.py --gate <dir> [--w scripts_cosim/decision_cost_w.json] [--out table.json]
"""
import argparse
import glob
import json
import os
import re
import sys

BUCKETS = ("le10", "11_50", "gt50")
NAME = re.compile(r"^cc40s(\d+)__(g\d)(moderate|heavy)__(.+)_s(\d+)\.summary\.json$")
RATIO_MAX = 1.5


def table(gate, w_path):
    w = json.load(open(w_path))["w_ms_per_plan"]
    acc = {}
    for p in sorted(glob.glob(os.path.join(gate, "*.summary.json"))):
        m = NAME.match(os.path.basename(p))
        if not m:
            continue
        s = json.load(open(p))
        bb = (s.get("decisionTiming") or {}).get("by_bucket")
        if bb is None:
            raise SystemExit(f"FAIL LOUD: decisionTiming.by_bucket missing in {os.path.basename(p)}")
        arm, rung = m.group(4), m.group(3)
        tot = acc.setdefault((arm, rung), {b: {"calls": 0, "total_s": 0.0, "plan_evals": 0} for b in BUCKETS})
        for b in BUCKETS:
            for k in tot[b]:
                tot[b][k] += bb[b][k]
        if sum(bb[b]["plan_evals"] for b in BUCKETS) != int(s["schedulerCounters"].get("sched_plan_evals", 0)):
            raise SystemExit(f"FAIL LOUD: buckets do not sum to sched_plan_evals in {os.path.basename(p)}")
    out = {}
    for (arm, rung), tot in sorted(acc.items()):
        ms = {b: (1000.0 * tot[b]["total_s"] / tot[b]["plan_evals"] if tot[b]["plan_evals"] else None) for b in BUCKETS}
        cal = (w.get(arm) or {}).get(rung)
        row = {"ms_per_plan": ms, "plan_evals": {b: tot[b]["plan_evals"] for b in BUCKETS}, "calls": {b: tot[b]["calls"] for b in BUCKETS}, "calibrated_w": cal,
               "ratio_to_le10": {b: (ms[b] / ms["le10"] if ms[b] and ms["le10"] else None) for b in BUCKETS},
               "ratio_to_calibrated": {b: (ms[b] / cal if ms[b] and cal else None) for b in BUCKETS}}
        row["exceeds_1_5x"] = [b for b in ("11_50", "gt50") if row["ratio_to_le10"][b] is not None and not (1 / RATIO_MAX <= row["ratio_to_le10"][b] <= RATIO_MAX)]
        out[f"{arm}|{rung}"] = row
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", required=True)
    ap.add_argument("--w", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "decision_cost_w.json"))
    ap.add_argument("--out")
    a = ap.parse_args()
    o = table(a.gate, a.w)
    for k, r in o.items():
        f = lambda v: "-" if v is None else f"{v:.5f}"
        g = lambda v: "-" if v is None else f"{v:.2f}x"
        print(f"{k:32s} ms/plan le10 {f(r['ms_per_plan']['le10'])} 11_50 {f(r['ms_per_plan']['11_50'])} gt50 {f(r['ms_per_plan']['gt50'])} | "
              f"vs le10 {g(r['ratio_to_le10']['11_50'])} {g(r['ratio_to_le10']['gt50'])} | vs calibrated w {f(r['calibrated_w'])}: "
              f"{g(r['ratio_to_calibrated']['le10'])} {g(r['ratio_to_calibrated']['11_50'])} {g(r['ratio_to_calibrated']['gt50'])} | exceeds 1.5x: {r['exceeds_1_5x']}")
    if a.out:
        json.dump(o, open(a.out, "w"), indent=1)
    print("ANY_BUCKET_EXCEEDS_1_5X", any(r["exceeds_1_5x"] for r in o.values()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
