#!/usr/bin/env python3
"""uncut_groups_s0_v1 co-tenancy r (pre-registered 2026-10-14, Note A): one arm's k = 0 cells run --exclusive at PAR = 1, once alone and once
beside the fixed training-like load (N = 8 x 8 threads). Per rung: pooled r = (sum total_s / sum plan_evals) shared / exclusive; per served-batch
bucket r_b the same on decisionTiming.by_bucket. If any bucket with >= MIN_EVALS plan evals in both runs has r_b > 1.25 x pooled, the rung's r is
the max r_b over such buckets, else the pooled r. Both runs must be sim-identical per cell (mean latency, plan evals), or it fails loud.

  uncut_cotenancy_r.py --excl <dir> --shared <dir> --arm cd_ib0 [--out r.json]
"""
import argparse
import glob
import json
import os
import re
import sys

BUCKETS = ("le10", "11_50", "gt50")
MIN_EVALS, BUCKET_X = 200, 1.25
NAME = re.compile(r"^cc40s(\d+)__(g\d)(moderate|heavy)__(.+)_s(\d+)\.summary\.json$")


def pooled(d, arm):
    acc, sim = {}, {}
    for p in sorted(glob.glob(os.path.join(d, "*.summary.json"))):
        m = NAME.match(os.path.basename(p))
        if not m or m.group(4) != arm:
            continue
        s = json.load(open(p))
        bb = (s.get("decisionTiming") or {}).get("by_bucket")
        if bb is None:
            raise SystemExit(f"FAIL LOUD: decisionTiming.by_bucket missing in {p}")
        r = m.group(3)
        a = acc.setdefault(r, {b: [0.0, 0] for b in ("all",) + BUCKETS})
        for b in BUCKETS:
            a[b][0] += bb[b]["total_s"]
            a[b][1] += bb[b]["plan_evals"]
        a["all"][0] += s["decisionTiming"]["total_s"]
        a["all"][1] += s["schedulerCounters"]["sched_plan_evals"]
        sim[(m.group(1), m.group(2), r, m.group(5))] = (s["averageElapsedTime"], s["schedulerCounters"]["sched_plan_evals"])
    if not acc:
        raise SystemExit(f"FAIL LOUD: no {arm} cells in {d}")
    return acc, sim


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--excl", required=True)
    ap.add_argument("--shared", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    ex, sim_ex = pooled(a.excl, a.arm)
    sh, sim_sh = pooled(a.shared, a.arm)
    if sim_ex != sim_sh:
        raise SystemExit("FAIL LOUD: exclusive and shared cells are not sim-identical")
    out = {"arm": a.arm, "excl": os.path.abspath(a.excl), "shared": os.path.abspath(a.shared), "rungs": {}}
    for r in sorted(ex):
        ms = lambda acc, b: 1000.0 * acc[r][b][0] / acc[r][b][1] if acc[r][b][1] else None
        pool = ms(sh, "all") / ms(ex, "all")
        by = {}
        for b in BUCKETS:
            n = min(ex[r][b][1], sh[r][b][1])
            rb = ms(sh, b) / ms(ex, b) if n else None
            by[b] = {"plan_evals": n, "w_excl": ms(ex, b), "w_shared": ms(sh, b), "r": rb, "eligible": n >= MIN_EVALS}
        hi = [v["r"] for v in by.values() if v["eligible"] and v["r"] > BUCKET_X * pool]
        el = [v["r"] for v in by.values() if v["eligible"]]
        out["rungs"][r] = {"w_excl": ms(ex, "all"), "w_shared": ms(sh, "all"), "r_pooled": pool, "by_bucket": by,
                           "r": max(el) if hi else pool, "rule": "max eligible bucket r" if hi else "pooled"}
    out["r"] = max(v["r"] for v in out["rungs"].values())
    txt = json.dumps(out, indent=1)
    if a.out:
        open(a.out, "w").write(txt)
    print(txt)
    return out


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
