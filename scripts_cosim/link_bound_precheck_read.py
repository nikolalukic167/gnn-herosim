"""link_bound_s0_v1 load pre-check reader. One output directory per bandwidth (pre_bw<tag>), CD only.

A bandwidth is admissible only if every one of its 64 cells (8 topologies x g0-g3 x moderate/heavy) finished and has queue share <= 0.80
and end/last-arrival <= 1.05; a missing, failed or hung cell is not a pass. Queue share is reported both raw (averageQueueTime over
averageElapsedTime) and effective (lock_wait.effective_queue_share); the rule is applied to the larger of the two.
Per bandwidth and rung: median core-link utilisation (the median over cells of each cell's median core link), the median over cells of the
cell's maximum core link, the same for access links, and the median over cells of each cell's median per-task link wait.

  link_bound_precheck_read.py --dirs 1000=<dir> 250=<dir> ... --topos 17100,... [--out read.json]
"""
import argparse
import glob
import json
import statistics as st

RUNGS = ("moderate", "heavy")
WINDOWS = ("g0", "g1", "g2", "g3")
QS_MAX, EL_MAX = 0.80, 1.05


def load(d):
    cells, failed = {}, []
    for f in glob.glob(f"{d}/*.summary.json"):
        s = json.load(open(f))
        w = s["window"]
        win = next(x for x in WINDOWS if w.startswith(x))
        rung = next(x for x in RUNGS if w.endswith(x))
        cells[(int(s["topology"]), win, rung)] = s
    for f in glob.glob(f"{d}/*.failed.json"):
        failed.append(f.rsplit("/", 1)[1])
    return cells, failed


def med(v):
    v = [x for x in v if x is not None]
    return st.median(v) if v else None


def read(dirs, topos):
    out = {}
    for bw, d in dirs.items():
        cells, failed = load(d)
        extra = sorted({k[0] for k in cells} - set(topos))
        if extra:
            raise SystemExit(f"FAIL LOUD: {bw}: topologies outside --topos: {extra}")
        rec = {"dir": d, "finished": len(cells), "failed": failed, "expected": len(topos) * len(WINDOWS) * len(RUNGS), "rungs": {}}
        bad = []
        for rung in RUNGS:
            cs = {k: s for k, s in cells.items() if k[2] == rung}
            qs = {k: max(s["queue_share"] or 0.0, s.get("effective_queue_share") or 0.0) for k, s in cs.items()}
            el = {k: s["arrival_end"]["end_over_last_arrival"] for k, s in cs.items()}
            for k in cs:
                if qs[k] > QS_MAX or el[k] is None or el[k] > EL_MAX:
                    bad.append({"cell": list(k), "queue_share": qs[k], "end_over_last_arrival": el[k]})
            tel = [s["link_telemetry"] for s in cs.values()]
            rec["rungs"][rung] = {
                "cells": len(cs), "queue_share_max": max(qs.values(), default=None), "end_over_last_arrival_max": max((v for v in el.values() if v), default=None),
                "mean_latency_median_s": med([s["averageElapsedTime"] for s in cs.values()]),
                "core_util_median": med([t["core_util_median"] for t in tel]), "core_util_max_median": med([t["core_util_max"] for t in tel]),
                "core_util_max_max": max((t["core_util_max"] for t in tel), default=None),
                "access_util_median": med([t["access_util_median"] for t in tel]), "access_util_max_median": med([t["access_util_max"] for t in tel]),
                "link_wait_median_s": med([t["link_wait"]["median"] for t in tel]), "link_wait_mean_s": med([t["link_wait"]["mean"] for t in tel]),
                "link_wait_p95_s": med([t["link_wait"]["p95"] for t in tel])}
        rec["inadmissible_cells"] = bad
        rec["admissible"] = rec["finished"] == rec["expected"] and not failed and not bad
        out[bw] = rec
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dirs", nargs="+", required=True, help="<bandwidth MB/s>=<dir>")
    ap.add_argument("--topos", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    topos = [int(x) for x in a.topos.split(",") if x]
    r = read(dict(x.split("=", 1) for x in a.dirs), topos)
    if a.out:
        json.dump(r, open(a.out, "w"), indent=1)
    f = lambda x, n=3: "-" if x is None else f"{x:.{n}f}"
    print(f"{'MB/s':>6} {'rung':9} {'cells':>5} {'qs max':>7} {'e/la max':>8} {'core u med':>10} {'core u max':>10} {'acc u med':>9} {'acc u max':>9} {'lw med s':>9} {'lw mean s':>9} {'lat s':>6}  admissible")
    for bw, rec in r.items():
        for rung, x in rec["rungs"].items():
            print(f"{bw:>6} {rung:9} {x['cells']:5d} {f(x['queue_share_max'], 2):>7} {f(x['end_over_last_arrival_max'], 3):>8} {f(x['core_util_median'], 4):>10} "
                  f"{f(x['core_util_max_max'], 4):>10} {f(x['access_util_median'], 4):>9} {f(x['access_util_max_median'], 3):>9} {f(x['link_wait_median_s'], 4):>9} "
                  f"{f(x['link_wait_mean_s'], 4):>9} {f(x['mean_latency_median_s'], 2):>6}  {rec['admissible']}")
        print(f"{'':>6} finished {rec['finished']}/{rec['expected']}, failed {len(rec['failed'])}, inadmissible cells {len(rec['inadmissible_cells'])}")


if __name__ == "__main__":
    main()
