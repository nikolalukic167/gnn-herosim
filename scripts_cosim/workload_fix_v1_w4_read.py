#!/usr/bin/env python3
"""workload_fix_v1 W2+W3+W4 read: the W2 reader's statistic on the W4 gate, plus the W3-read descriptives.

Same statistic and descriptives as `workload_fix_v1_w3_read.py` (paired % vs CD, Holm over {selfpredict, locality,
batched} x rungs, CD stage decomposition, exchange by arm and wired-pair share), with the comparison gate being the
W3 gate (W4 minus W3, paired on (topology, window, arm)). Unlike the W3 wrapper it does not refuse an incomplete gate:
a run that is hung or failed stays out of both its arm's statistic and the pairing (the reader counts it as failed),
and the numbers of summaries and failures are written to `_inputs`. `--exclude topo:window:rung ...` drops a
(topology, window, rung) cell for every arm in both gates (the live-feasibility exclusion; none were infeasible).

  workload_fix_v1_w4_read.py --gate-dir <w4 gate> --w3-gate-dir <w3 gate> --selected <selected.json> --rungs lo,hi --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import workload_fix_v1_read as R  # noqa: E402
from workload_fix_v1_w3_read import exchange_by_arm, w3_minus_w2  # noqa: E402


def drop_cells(ok: dict, bad: dict, excluded: set):
    keep = lambda k: (k[0], k[1][:2], k[1][2:]) not in excluded  # noqa: E731
    return {k: v for k, v in ok.items() if keep(k)}, {k: v for k, v in bad.items() if keep(k)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate-dir", required=True)
    ap.add_argument("--w3-gate-dir", required=True)
    ap.add_argument("--exclude", nargs="*", default=[], help="topology:window:rung, e.g. 9565:g2:hi")
    ap.add_argument("--selected", required=True)
    ap.add_argument("--rungs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    topos = json.load(open(a.selected))["topologies"]
    rungs = [r for r in a.rungs.split(",") if r]
    ok, bad = R.load(a.gate_dir)
    ok2, bad2 = R.load(a.w3_gate_dir)
    excluded = {(int(t), w, r) for t, w, r in (x.split(":") for x in a.exclude)}
    ok, bad = drop_cells(ok, bad, excluded)
    ok2, bad2 = drop_cells(ok2, bad2, excluded)
    result = R.compute(ok, bad, topos, rungs)
    result["_cd_decomposition"] = R.decompose(ok, topos, rungs)
    result["_exchange_by_arm"] = exchange_by_arm(ok, topos, rungs)
    result["_w3_minus_w2"] = w3_minus_w2(ok, ok2, topos, rungs)
    result["_inputs"] = {"w4_gate": a.gate_dir, "w3_gate": a.w3_gate_dir, "n_w4": len(ok), "n_w4_failed": len(bad),
                         "n_w3": len(ok2), "n_w3_failed": len(bad2), "excluded": sorted(a.exclude)}
    Path(a.out).write_text(json.dumps(result, indent=1))

    for rung in rungs:
        print(f"== {rung}  (CD first: {result[rung]['_cd_first']})")
        for arm in R.ARMS:
            r = result[rung][arm]
            q = r["request_failures"]
            rf = f"{q['tasks']}/{q['runs_with_failure']}runs" if q["available"] else "n/a"
            vs = (f"vs CD {r['vs_cd']:+6.1f}% ({r['faster']}/{r['n_pc']} faster, p {r['p']:.2g}"
                  + (f", Holm {r['p_holm']:.2g}, {r['label']})" if "p_holm" in r else ", context)")) if "vs_cd" in r else ""
            print(f"  {arm:12s} n={r['n_topologies']:2d} fail={r['n_failed']:2d} lat {r['lat']:7.2f} q {r['qshare']:.2f} "
                  f"exch/task {r['exch_per_task']:.2f}s ({100 * r['exch_share']:.1f}% of latency) req-fail {rf} {vs}")
        d = result["_cd_decomposition"][rung]
        print("  CD stages: " + ", ".join(f"{k} {d[k]:.3f}" for k in ("latency",) + R.COMPONENTS + ("other",)))
        for arm in R.ARMS:
            e = result["_exchange_by_arm"][rung][arm]
            c = result["_w3_minus_w2"][rung][arm]
            print(f"  {arm:12s} exch/task {e['exch_per_task']:.2f}s  wired+wired: {100 * e['share_transfers_wired_wired']:.1f}% of transfers, "
                  f"{100 * e['share_seconds_wired_wired']:.1f}% of exchange s | W4-W3 latency {c['median_pct_cells']:+.1f}% cells, "
                  f"{c['median_pct_topologies']:+.1f}% topologies ({c['n_topologies_slower']}/{c['n_topologies']} slower), "
                  f"exch/task {c['median_exch_per_task_diff_s']:+.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
