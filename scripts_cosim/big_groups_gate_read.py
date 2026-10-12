"""big_groups_s0_v1 live gate reader (rule 6, G = 8). Written before the data lands.

Arms on the 12 fresh topologies, 4 windows, both rungs: ra_gnn_eng_selfsearch (seed 1), cd (seed 0), cd_exactS (seed 0).
Counts first: summaries and failures per arm and rung, with the expected 48 cells per arm and rung. Then, per rung, the paired % of mean latency
per topology (median over the topology's windows), its median over the 12 topologies, the sign count (wins = topologies where the first arm is
faster) and an exact two-sided Wilcoxon, for
  self-search vs cd (the bar), self-search vs cd_exactS, cd_exactS vs cd.
Negative = the first arm is faster. The self-search batches are split into exact and ascent, and cd_exactS's fallback share is its fallbacks over its
batches. The signed trigger of accel_nosplit_v1 (both rungs <= -5 % vs cd) is printed as a flag only. A topology outside --topos is refused, and so
is a window set other than g0-g3.

  big_groups_gate_read.py --gate <dir> [<dir> ...] --topos 16900,16901,... [--out read.json]
"""
import argparse
import json
import signal
import statistics as st
import sys

import r1_attribution_v1_read as R

ARM = "ra_gnn_eng_selfsearch"
CD = "cd"
EXACT = "cd_exactS"
RUNGS = ("moderate", "heavy")
WINDOWS = ("g0", "g1", "g2", "g3")
PAIRS = ((ARM, CD), (ARM, EXACT), (EXACT, CD))
BAR_PCT = -5.0


def check(cells, failed, topos):
    seen = {k[2] for k in list(cells) + list(failed)}
    bad = sorted(seen - set(topos))
    if bad:
        raise SystemExit(f"FAIL LOUD: topologies outside --topos: {bad}")
    wins = {k[3] for k in list(cells) + list(failed)}
    bad_w = sorted(w for w in wins if not any(w.startswith(x) for x in WINDOWS))
    if bad_w:
        raise SystemExit(f"FAIL LOUD: windows outside {WINDOWS}: {bad_w}")


def counts(cells, failed, topos):
    exp = len(topos) * len(WINDOWS)
    out = {}
    for kind in (ARM, CD, EXACT):
        for rung in RUNGS:
            fin = sum(1 for k in cells if k[0] == kind and k[4] == rung)
            bad = sum(1 for k in failed if k[0] == kind and k[4] == rung)
            out[f"{kind}|{rung}"] = {"finished": fin, "failed": bad, "expected": exp, "complete": fin == exp}
    return out


def _tot(cs, name):
    return sum(int(c.get(name) or 0) for c in cs)


def search_counts(cells, rung):
    cs = [s["schedulerCounters"] for k, s in cells.items() if k[0] == ARM and k[4] == rung]
    b = _tot(cs, "ss_batches")
    return {"cells": len(cs), "batches": b, "exact_batches": _tot(cs, "ss_exact_batches"), "ascent_batches": _tot(cs, "ss_ascent_batches"),
            "exact_share": _tot(cs, "ss_exact_batches") / b if b else None, "ascent_share": _tot(cs, "ss_ascent_batches") / b if b else None,
            "changed_batch_share": _tot(cs, "ss_changed_batches") / b if b else None,
            "plans_scored": _tot(cs, "ss_scored")}


def fallback_counts(cells, rung):
    cs = [s["schedulerCounters"] for k, s in cells.items() if k[0] == EXACT and k[4] == rung]
    b, ex, fb = _tot(cs, "pg_batches"), _tot(cs, "pg_exact_batches"), _tot(cs, "pg_exact_fallbacks")
    return {"cells": len(cs), "batches": b, "exact_batches": ex, "fallbacks": fb, "fallback_share_of_batches": fb / b if b else None,
            "plans_enumerated": _tot(cs, "pg_exact_plans")}


def read(gate, topos):
    cells, failed = R.load(gate)
    cells = {k: v for k, v in cells.items() if k[4] in RUNGS}
    failed = {k: v for k, v in failed.items() if k[4] in RUNGS}
    check(cells, failed, topos)
    out = {"gate": gate, "topologies": sorted(topos), "n_summaries": len(cells), "n_failed": len(failed), "counts": counts(cells, failed, topos),
           "search": {r: search_counts(cells, r) for r in RUNGS}, "cd_exactS": {r: fallback_counts(cells, r) for r in RUNGS},
           "contrasts": {f"{a} vs {b}|{r}": R.contrast(cells, a, b, r, sorted(topos)) for a, b in PAIRS for r in RUNGS}}
    out["trigger_vs_cd"] = all((out["contrasts"][f"{ARM} vs {CD}|{r}"]["median_pct"] or 0) <= BAR_PCT for r in RUNGS)
    return out


def print_report(r):
    f = lambda x, n=4: "-" if x is None else f"{x:.{n}f}"
    print(f"{r['n_summaries']} summaries, {r['n_failed']} failed, {len(r['topologies'])} topologies")
    print("-- cells per arm and rung (finished / failed / expected)")
    for k, c in r["counts"].items():
        print(f"  {k:36s} {c['finished']}/{c['failed']}/{c['expected']}{'' if c['complete'] else '   INCOMPLETE'}")
    print("-- self-search batches")
    for rung, c in r["search"].items():
        print(f"  {rung}: {c}")
    print("-- cd_exactS batches and fallbacks")
    for rung, c in r["cd_exactS"].items():
        print(f"  {rung}: {c}")
    print("-- paired % per topology (median over windows), median over topologies; negative = first arm faster")
    for k, c in r["contrasts"].items():
        m = "-" if c["median_pct"] is None else format(c["median_pct"], "+.2f") + " %"
        print(f"  {k:46s} median {m:>10s} n={c['n_topologies']:2d} wins={c['wins']} p={f(c['p'])}")
        if c.get("per_topology"):
            print("      " + " ".join(f"{t}:{v:+.1f}" for t, v in c["per_topology"].items()))
    print(f"-- flag (both rungs <= {BAR_PCT} % vs cd): {r['trigger_vs_cd']}")


def main():
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", nargs="+", required=True)
    ap.add_argument("--topos", required=True, help="comma-separated topology ids (the 12 fresh ones)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    topos = [int(x) for x in a.topos.split(",") if x]
    if len(topos) != 12:
        raise SystemExit(f"FAIL LOUD: expected 12 topologies, got {len(topos)}")
    r = read(a.gate, topos)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(r, fh, indent=1)
    print_report(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
