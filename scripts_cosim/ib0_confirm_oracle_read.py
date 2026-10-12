"""ib0_confirm_v1 full-coverage oracle reader (the node's "Full-coverage oracle" stage). Written before the data lands.

Cells: `cd_oracle_ib0` on 6 topologies (19405-19410), window g0, both rungs = 12 cells, against the classical gate's `cd_exactS_ib0` (IB0) and `cd_exactS` (E) g0 cells.
Unit: the topology (one g0 cell each); median over the 6 topologies of the paired % of mean latency; negative = the oracle faster.
Coverage rule: replay-labelled tasks (exact-covered `pg_oracle_tasks` + searched over-cap `pg_oracle_search_tasks`) must be >= 95 % of `pg_oracle_seen_tasks`,
pooled per rung; else INCONCLUSIVE. Closing rule: oracle < 3 % faster than ib0 at MODERATE (median > -3 %) -> the supervised batch-label family closes;
>= 3 % keeps it open; heavy is reported and cannot rescue moderate. Descriptive (no bar): oracle vs cd_exactS, ib0 vs cd_exactS on the same cells.
Counts first; a failed.json is a count, never a rerun.

  ib0_confirm_oracle_read.py --gate <oracle dir> <classical dir> [--out read_oracle.json]
"""
import argparse
import json
import signal
import statistics as st
import sys

import r1_attribution_v1_read as R

ORC, IB0, E = "cd_oracle_ib0", "cd_exactS_ib0", "cd_exactS"
TOPOS = (19405, 19406, 19407, 19408, 19409, 19410)
RUNGS = ("moderate", "heavy")
WINDOW = "g0"
COVERAGE_MIN, CLOSE_PCT = 0.95, -3.0
BOOKS = ("pg_oracle_seen_batches", "pg_oracle_seen_tasks", "pg_oracle_batches", "pg_oracle_tasks", "pg_oracle_plans", "pg_oracle_changed",
         "pg_oracle_tasks_moved", "pg_oracle_over_cap_batches", "pg_oracle_over_cap_tasks", "pg_oracle_unplaced_batches", "pg_oracle_unplaced_tasks",
         "pg_oracle_search_batches", "pg_oracle_search_tasks", "pg_oracle_search_replays", "pg_oracle_search_steps", "pg_oracle_search_improved",
         "pg_oracle_search_exhausted", "pg_oracle_search_gain_seconds", "pg_oracle_search_seconds", "pg_oracle_gain_seconds", "pg_oracle_seconds")


def coverage(cells, rung):
    rows = {t: (cells.get((ORC, 1, t, WINDOW, rung)) or cells.get((ORC, 0, t, WINDOW, rung))) for t in TOPOS}
    tot = {n: 0.0 for n in BOOKS}
    per = {}
    for t, s in rows.items():
        if s is None:
            continue
        c = s.get("schedulerCounters") or {}
        for n in BOOKS:
            tot[n] += float(c.get(n) or 0)
        seen = float(c.get("pg_oracle_seen_tasks") or 0)
        lab = float(c.get("pg_oracle_tasks") or 0) + float(c.get("pg_oracle_search_tasks") or 0)
        per[str(t)] = round(lab / seen, 4) if seen else None
    seen, lab = tot["pg_oracle_seen_tasks"], tot["pg_oracle_tasks"] + tot["pg_oracle_search_tasks"]
    books_close = tot["pg_oracle_seen_batches"] == (tot["pg_oracle_batches"] + tot["pg_oracle_search_batches"]
                                                    + tot["pg_oracle_over_cap_batches"] + tot["pg_oracle_unplaced_batches"]) if seen else None
    return {"cells": sum(1 for s in rows.values() if s is not None), "labelled_share": (lab / seen) if seen else None,
            "served_share": (tot["pg_oracle_tasks"] / seen) if seen else None, "searched_share": (tot["pg_oracle_search_tasks"] / seen) if seen else None,
            "unplaced_share": (tot["pg_oracle_unplaced_tasks"] / seen) if seen else None,
            "over_cap_unsearched_share": (tot["pg_oracle_over_cap_tasks"] / seen) if seen else None,
            "per_topology_labelled_share": per, "books_close": books_close, "totals": tot}


def read(gates):
    cells, failed = R.load(list(gates))
    keep = lambda k: k[0] in (ORC, IB0, E) and k[3] == WINDOW and k[4] in RUNGS and k[2] in TOPOS
    cells = {k: v for k, v in cells.items() if keep(k)}
    failed = {k: v for k, v in failed.items() if keep(k)}
    counts = {f"{kind}|{r}": {"finished": sum(1 for k in cells if k[0] == kind and k[4] == r),
                              "failed": sum(1 for k in failed if k[0] == kind and k[4] == r), "expected": len(TOPOS)}
              for kind in (ORC, IB0, E) for r in RUNGS}
    contrasts = {f"{a} vs {b}|{r}": R.contrast(cells, a, b, r, list(TOPOS)) for a, b in ((ORC, IB0), (ORC, E), (IB0, E)) for r in RUNGS}
    cov = {r: coverage(cells, r) for r in RUNGS}
    verdict = {}
    complete = all(c["finished"] == c["expected"] for c in counts.values())
    cov_ok = {r: cov[r]["labelled_share"] is not None and cov[r]["labelled_share"] >= COVERAGE_MIN for r in RUNGS}
    verdict["complete"] = complete
    verdict["coverage_ok"] = cov_ok
    m = contrasts[f"{ORC} vs {IB0}|moderate"]["median_pct"]
    if not complete:
        verdict["ORACLE"] = "INCOMPLETE"
    elif not all(cov_ok.values()):
        verdict["ORACLE"] = "INCONCLUSIVE"
    elif m is not None and m > CLOSE_PCT:
        verdict["ORACLE"] = "FAMILY-CLOSES"
    else:
        verdict["ORACLE"] = "FAMILY-OPEN"
    verdict["moderate_median_pct"] = m
    verdict["heavy_median_pct"] = contrasts[f"{ORC} vs {IB0}|heavy"]["median_pct"]
    return {"gate": list(gates), "topologies": list(TOPOS), "counts": counts, "failed_why": sorted({str(f.get("why", "?"))[:80] for f in failed.values()}),
            "contrasts": contrasts, "coverage": cov, "verdict": verdict}


def print_report(r):
    f = lambda x: "-" if x is None else f"{x:.4f}"
    print("-- counts (finished / failed / expected)")
    for k, c in r["counts"].items():
        print(f"  {k:28s} {c['finished']}/{c['failed']}/{c['expected']}")
    print(f"failed why: {r['failed_why']}")
    for rung, c in r["coverage"].items():
        print(f"-- coverage {rung}: labelled {f(c['labelled_share'])} (served {f(c['served_share'])}, searched {f(c['searched_share'])}, "
              f"unplaced {f(c['unplaced_share'])}, over-cap unsearched {f(c['over_cap_unsearched_share'])}) books_close={c['books_close']} per-topology {c['per_topology_labelled_share']}")
        t = c["totals"]
        print(f"   search: replays {t['pg_oracle_search_replays']:.0f} steps {t['pg_oracle_search_steps']:.0f} improved {t['pg_oracle_search_improved']:.0f}/"
              f"{t['pg_oracle_search_batches']:.0f} batches, exhausted {t['pg_oracle_search_exhausted']:.0f}; exact changed {t['pg_oracle_changed']:.0f}/{t['pg_oracle_batches']:.0f}")
    print("-- paired % (g0, median over topologies; negative = first faster)")
    for k, c in r["contrasts"].items():
        m = "-" if c["median_pct"] is None else format(c["median_pct"], "+.2f") + " %"
        print(f"  {k:40s} median {m:>10s} n={c['n_topologies']} wins={c['wins']} p={f(c['p'])}")
        if c.get("per_topology"):
            print("      " + " ".join(f"{t}:{v:+.1f}" for t, v in c["per_topology"].items()))
    print(f"-- verdict {r['verdict']}")


def main():
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    r = read(a.gate)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(r, fh, indent=1)
    print_report(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
