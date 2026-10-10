"""reactive_conc reader (accel_nosplit_v1 side study): a faithful-Knative arm whose least-connected key counts in-flight concurrency
(len(queue.items) + compute-lock waiters + running, HEROSIM_KN_CONC=1) against the same cells' other arms. Descriptive, never the bar.

Statistic as r1_attribution_v1_read.py: per topology the median over windows of the paired % of reactive_conc's mean latency against the
reference arm on the same (topology, window); then the median over topologies and an exact two-sided Wilcoxon. Negative = reactive_conc faster.
References: CD, reactive (the key-blind Knative), the no-split GNN arms, batched, locality, self-predict.
Collapse metrics per arm and rung: cells finished, cells with run end > 1.5 x last arrival, cells with effective queue share > 0.80,
median / worst p95, median end / last arrival, median effective share.

  reactive_conc_read.py --gate <dir> [<dir> ...] [--out read.json]
"""
import argparse
import json
import signal
import statistics as st
import sys

import r1_attribution_v1_read as R

ARM = "reactive_conc"
REFERENCES = ("cd", "reactive", "ra_gnn_eng_nosplit", "ra_gnn_eng_physmp_nosplit", "batched", "locality", "selfpredict")
RUNGS = ("moderate", "heavy")
END_X, SHARE_X = 1.5, 0.80


def collapse(cells: dict) -> dict:
    out = {}
    for kind in sorted({k[0] for k in cells}):
        for rung in RUNGS:
            cs = [s for k, s in cells.items() if k[0] == kind and k[4] == rung]
            if not cs:
                continue
            end = [(s.get("arrival_end") or {}).get("end_over_last_arrival") for s in cs]
            share = [s.get("effective_queue_share") for s in cs]
            p95 = [(s.get("latency_percentiles") or {}).get("p95") for s in cs]
            known = lambda xs: [x for x in xs if x is not None]
            out[f"{kind}|{rung}"] = {
                "cells": len(cs), "end_gt_1.5": sum(1 for x in known(end) if x > END_X), "share_gt_0.80": sum(1 for x in known(share) if x > SHARE_X),
                "unknown": sum(1 for x in end if x is None) + sum(1 for x in share if x is None),
                "median_p95_s": st.median(known(p95)) if known(p95) else None, "worst_p95_s": max(known(p95)) if known(p95) else None,
                "median_end_over_last_arrival": st.median(known(end)) if known(end) else None,
                "median_effective_share": st.median(known(share)) if known(share) else None}
    return out


def read(gate) -> dict:
    cells, failed = R.load(gate)
    topos = sorted({k[2] for k in cells if k[0] == ARM})
    r = {"topologies": topos, "cells": {ARM: sum(1 for k in cells if k[0] == ARM), "failed": sum(1 for k in failed if k[0] == ARM)},
         "contrasts": {}, "collapse": collapse(cells)}
    for ref in REFERENCES:
        for rung in RUNGS:
            c = R.contrast(cells, ARM, ref, rung, topos)
            r["contrasts"][f"{ARM} vs {ref}|{rung}"] = {k: c.get(k) for k in ("n_topologies", "median_pct", "p", "wins", "per_topology")}
    return r


def main() -> int:
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    r = read(a.gate)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(r, fh, indent=1)
    print(f"reactive_conc cells {r['cells'][ARM]}, failed {r['cells']['failed']}, topologies {len(r['topologies'])}")
    print("-- collapse metrics")
    for k, c in r["collapse"].items():
        mp = "-" if c["median_p95_s"] is None else f"{c['median_p95_s']:.1f}"
        print(f"  {k:36s} cells {c['cells']:3d} end>1.5 {c['end_gt_1.5']:3d} share>0.8 {c['share_gt_0.80']:3d} unknown {c['unknown']} | median p95 {mp} s, median end/last {c['median_end_over_last_arrival']:.2f}, median share {c['median_effective_share']:.2f}")
    print("-- paired (negative = reactive_conc faster; median over topologies of the per-topology median over windows)")
    for k, c in r["contrasts"].items():
        m = "-" if c["median_pct"] is None else f"{c['median_pct']:+.2f} %"
        print(f"  {k:52s} median {m:>10s} n={c['n_topologies']} wins={c['wins']} p={'-' if c['p'] is None else format(c['p'], '.4f')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
