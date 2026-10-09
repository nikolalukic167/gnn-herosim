"""Starved-client spin columns for an r1_attribution_v1 gate read (descriptive, outside the families).

Per arm and rung: median prefix_tasks_deferred, median reachability scale-up failures and median averageWaitTime, and the cells whose
deferrals exceed FLAG_X times CD's on the same (topology, window, rung). Arms without the counter (per-arrival rules) print n/a.
  r1a_gate_spin.py --gate <dir> [<dir> ...] [--arms a,b,c]
"""
import argparse
import signal
import statistics as st
import sys
from collections import defaultdict

import r1_attribution_v1_read as R

FLAG_X = 5.0


def deferred(s):
    v = (s.get("schedulerCounters") or {}).get("prefix_tasks_deferred")
    return None if v is None else float(v)


def reach_failures(s):
    v = ((s.get("scaleOut") or {}).get("scale_up_failures_by_cause") or {}).get("reachability")
    return None if v is None else float(v)


def med(xs):
    xs = [x for x in xs if x is not None]
    return st.median(xs) if xs else None


def fmt(x):
    return "n/a" if x is None else (f"{x:,.0f}" if x >= 100 else f"{x:.3g}")


def main() -> int:
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", nargs="+", required=True)
    ap.add_argument("--arms", default="")
    a = ap.parse_args()
    cells, _ = R.load(a.gate)
    want = [k for k in a.arms.split(",") if k] or sorted({k[0] for k in cells})
    print(f"{'arm':22s} {'rung':9s} {'n':>4s} {'deferred med':>13s} {'reach-fail med':>15s} {'avgWait s':>10s} {'>%gx CD' % FLAG_X:>9s}")
    flagged = defaultdict(list)
    for kind in want:
        for rung in R.RUNGS:
            cs = {k: s for k, s in cells.items() if k[0] == kind and k[4] == rung}
            if not cs:
                continue
            nflag = 0
            for (_, seed, topo, win, r), s in cs.items():
                d, ref = deferred(s), cells.get(("cd", 0, topo, win, rung))
                dc = deferred(ref) if ref else None
                if kind != "cd" and d is not None and dc is not None and d > FLAG_X * max(dc, 1.0):
                    nflag += 1
                    flagged[kind].append((topo, win, rung, seed, d, dc))
            print(f"{kind:22s} {rung:9s} {len(cs):4d} {fmt(med(deferred(s) for s in cs.values())):>13s} "
                  f"{fmt(med(reach_failures(s) for s in cs.values())):>15s} {fmt(med(float(s['averageWaitTime']) for s in cs.values())):>10s} "
                  f"{nflag if med(deferred(s) for s in cs.values()) is not None else 'n/a':>9}")
    for kind, rows in flagged.items():
        print(f"\nflagged {kind}: {len(rows)} cells; worst 5 (topo window rung seed deferred / CD deferred)")
        for t, w, r, sd, d, dc in sorted(rows, key=lambda x: -x[4] / max(x[5], 1.0))[:5]:
            print(f"  {t} {w} {r} s{sd}: {d:,.0f} / {dc:,.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
