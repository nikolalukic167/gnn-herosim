"""sparse_obs_s0_v1: byte-identity of two sets of summaries cell by cell. Every key the OLD summary carries must be equal in the NEW one (total_rtt, endTime, averageElapsedTime,
and every old schedulerCounters entry); keys only the new one has are listed, not compared.

  sparse_obs_ident.py OLD_DIR NEW_DIR [--arm cd_exactS_ib0]
"""
import argparse
import glob
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--arm", default="cd_exactS_ib0")
    a = ap.parse_args()
    n = bad = 0
    added = set()
    for f in sorted(glob.glob(os.path.join(a.new, f"*__{a.arm}_s*.summary.json"))):
        g = os.path.join(a.old, os.path.basename(f))
        if not os.path.exists(g):
            continue
        o, w = json.load(open(g)), json.load(open(f))
        n += 1
        d = [k for k in ("total_rtt", "endTime", "averageElapsedTime", "averageQueueTime", "averageColdStartTime", "num_tasks") if o.get(k) != w.get(k)]
        co, cw = o["schedulerCounters"], w["schedulerCounters"]
        d += [f"counter:{k}" for k in co if co.get(k) != cw.get(k)]
        added |= set(cw) - set(co)
        if d:
            bad += 1
            print("DIFF", os.path.basename(f), d)
    print(f"{n} cells compared, {bad} differ; counters only in new: {sorted(added)}")
    print("IDENTITY_OK" if n and not bad else "IDENTITY_FAIL")
    return 0 if n and not bad else 1


if __name__ == "__main__":
    sys.exit(main())
