#!/usr/bin/env python3
"""uncut_groups_s0_v1 co-tenancy r (pre-registered 2026-10-14, Note A): one arm's k = 0 cells run --exclusive at PAR = 1, once alone and once
beside the fixed training-like load (N = 8 x 8 threads). Per rung: pooled r = (sum total_s / sum plan_evals) shared / exclusive; per served-batch
bucket r_b the same on decisionTiming.by_bucket. If any bucket with >= MIN_EVALS plan evals in both runs has r_b > 1.25 x pooled, the rung's r is
the max r_b over such buckets, else the pooled r. Both runs must be sim-identical per cell (mean latency, plan evals), or it fails loud.

  uncut_cotenancy_r.py --excl <dir>[,<dir>] --shared <dir>[,<dir>] --arm cd_ib0 [--out r.json] [--legacy-no-host]

Several dirs per condition are the node-balanced reps (node A and node B, the same cells on each): pooled together, and every dir must
hold the same cells, sim-identical to every other dir of both conditions.

Calibration protocol (SMT, 2026-10-14): every cell carries the gate's host block (114cc0e7); all cells of both runs share one CPU model
and one SMT state, and a cell whose torch threads exceed its cpuset's nproc is void (fails). --legacy-no-host reads the SMT-on runs
filed before the host block (869835/869836, 869839/869840) and marks the output so.
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


def pooled(d, arm, hosts=None):
    acc, sim = {}, {}
    for p in sorted(glob.glob(os.path.join(d, "*.summary.json"))):
        m = NAME.match(os.path.basename(p))
        if not m or m.group(4) != arm:
            continue
        s = json.load(open(p))
        bb = (s.get("decisionTiming") or {}).get("by_bucket")
        if bb is None:
            raise SystemExit(f"FAIL LOUD: decisionTiming.by_bucket missing in {p}")
        if hosts is not None:
            h = s.get("host")
            th = (s.get("python_env") or {}).get("torch_num_threads")
            if not isinstance(h, dict) or not h.get("cpu_model") or not h.get("smt_control") or not h.get("nproc"):
                raise SystemExit(f"FAIL LOUD: {p} has no host block (cpu_model, smt_control, nproc)")
            if not isinstance(th, int) or th > h["nproc"]:
                raise SystemExit(f"FAIL LOUD: {p} torch threads {th!r} vs nproc {h['nproc']}: void (threads > nproc)")
            hosts.append({"cell": os.path.basename(p), "hostname": h.get("hostname"), "cpu_model": h["cpu_model"], "smt_control": h["smt_control"],
                          "threads_per_core": h.get("threads_per_core"), "nproc": h["nproc"], "cpuset": h.get("cpuset"), "torch_threads": th})
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


def merged(dirs, arm, hosts):
    acc, sims = {}, []
    for d in [x for x in dirs.split(",") if x]:
        a, sim = pooled(d, arm, hosts)
        sims.append(sim)
        for r, bs in a.items():
            t = acc.setdefault(r, {b: [0.0, 0] for b in bs})
            for b, (x, n) in bs.items():
                t[b][0] += x
                t[b][1] += n
    return acc, sims


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--excl", required=True)
    ap.add_argument("--shared", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out")
    ap.add_argument("--legacy-no-host", action="store_true")
    a = ap.parse_args(argv)
    hx, hs = (None, None) if a.legacy_no_host else ([], [])
    ex, sims = merged(a.excl, a.arm, hx)
    sh, sims_sh = merged(a.shared, a.arm, hs)
    if any(x != sims[0] for x in sims + sims_sh):
        raise SystemExit("FAIL LOUD: the runs (exclusive, shared, every node's rep) are not sim-identical on the same cells")
    out = {"arm": a.arm, "excl": os.path.abspath(a.excl), "shared": os.path.abspath(a.shared), "rungs": {}}
    if a.legacy_no_host:
        out["venue"] = "LEGACY: no host block (filed before 114cc0e7)"
    else:
        venue = {(h["cpu_model"], h["smt_control"]) for h in hx + hs}
        if len(venue) != 1:
            raise SystemExit(f"FAIL LOUD: the runs mix CPU model / SMT states {sorted(venue)}; one venue per calibration")
        (model, smt), = venue
        out["venue"] = {"cpu_model": model, "smt_control": smt, "hosts": sorted({h["hostname"] for h in hx + hs}),
                        "excl": hx, "shared": hs}
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
