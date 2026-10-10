"""accel_nosplit_v1 classical-reference addendum reader (docs/lineages/accel_nosplit_v1.md). Descriptive; it cannot change the verdict.

The classical arms reactive (Knative), random, batched (the one-pass greedy) and locality, seed 0, on the 24 topologies 16345-16368, 4 windows,
both rungs, at pin acbea783. Per topology the median over (window, reference seed) of 100 (A - R) / R on mean latency, A the classical cell at
seed 0 and R a reference cell at the same (topology, window, rung): every seed the reference has (CD seed 0; the no-split GNN seeds 1 and 2).
Then the median over topologies and an exact two-sided Wilcoxon. Three families, each Holm over (4 arms x 2 rungs) = 8: vs ra_gnn_eng_nosplit,
vs ra_gnn_eng_physmp_nosplit, vs cd. Positive = the classical arm is slower. Per-rung label: SLOWER / FASTER (Holm p < .05) else NOT-SEPARATED.
A missing arm counts as p = 1. Knative collapse is recorded, never dropped: per arm and rung the finished/failed cells, p95, run end over last
arrival and effective queue share (median and worst), and the number of cells beyond end/last-arrival 1.5 or effective share 0.80 (display
thresholds only). Watchdog kills count as failed. The reader refuses any topology outside 16345-16368 and prints counts first.

  accel_nosplit_classical_read.py --gate <dir> [<dir> ...] [--out read.json]
"""
import argparse
import json
import signal
import statistics as st
import sys
from typing import Dict, List, Sequence

import r1_attribution_v1_read as R

RUNGS = ("moderate", "heavy")
ARMS = ("reactive", "random", "batched", "locality")
REFS = ("ra_gnn_eng_nosplit", "ra_gnn_eng_physmp_nosplit", "cd")
TOPO_RANGE = (16345, 16368)
ALPHA = 0.05
COLLAPSE_END, COLLAPSE_SHARE = 1.5, 0.80


def check_topologies(keys) -> List[int]:
    topos = sorted({k[2] for k in keys})
    out = [t for t in topos if not TOPO_RANGE[0] <= t <= TOPO_RANGE[1]]
    if out:
        raise SystemExit(f"FAIL LOUD: topologies outside {TOPO_RANGE[0]}-{TOPO_RANGE[1]} in the gate directories: {out}")
    return topos


def contrast(cells: dict, a: str, ref: str, rung: str, topos: Sequence[int]) -> dict:
    per = {}
    for t in topos:
        vals = []
        for (kind, seed, topo, win, r), s in cells.items():
            if kind != a or topo != t or r != rung:
                continue
            for (rk, rs, rt, rw, rr), c in cells.items():
                if rk == ref and rt == t and rw == win and rr == rung:
                    vals.append(100.0 * (float(s["averageElapsedTime"]) - float(c["averageElapsedTime"])) / float(c["averageElapsedTime"]))
        if vals:
            per[t] = st.median(vals)
    if not per:
        return {"a": a, "ref": ref, "rung": rung, "n_topologies": 0, "median_pct": None, "p": None, "wins": None}
    v = list(per.values())
    return {"a": a, "ref": ref, "rung": rung, "n_topologies": len(v), "median_pct": st.median(v), "p": R.exact_wilcoxon(v),
            "wins": sum(1 for x in v if x < 0), "per_topology": {str(t): round(x, 3) for t, x in sorted(per.items())}}


def rung_label(c: dict) -> str:
    if c["median_pct"] is None or c.get("holm_p") is None or c["holm_p"] >= ALPHA:
        return "NOT-SEPARATED"
    return "SLOWER" if c["median_pct"] > 0 else "FASTER"


def family(cells: dict, topos: List[int], arms: Sequence[str], ref: str) -> dict:
    tests = {f"{a}|{r}": contrast(cells, a, ref, r, topos) for a in arms for r in RUNGS}
    adj = R.holm({k: c["p"] for k, c in tests.items()})
    for k, c in tests.items():
        c["holm_p"] = adj[k]
    return {"ref": ref, "holm_over": len(tests), "tests": tests, "labels": {k: rung_label(c) for k, c in tests.items()}}


def collapse(cells: dict, failed: dict, arms: Sequence[str]) -> dict:
    """Per classical arm and rung: how many cells ran past the display thresholds. Nothing is dropped."""
    out = {}
    for a in arms:
        for r in RUNGS:
            cs = [s for k, s in cells.items() if k[0] == a and k[4] == r]
            end = [(s.get("arrival_end") or {}).get("end_over_last_arrival") for s in cs]
            sh = [s.get("effective_queue_share") for s in cs]
            out[f"{a}|{r}"] = {
                "finished": len(cs), "failed": sum(1 for k in failed if k[0] == a and k[4] == r),
                "end_over_last_arrival_gt_1.5": sum(1 for x in end if x is not None and x > COLLAPSE_END),
                "effective_share_gt_0.80": sum(1 for x in sh if x is not None and x > COLLAPSE_SHARE),
                "unknown_end_or_share": sum(1 for e, x in zip(end, sh) if e is None or x is None)}
    return out


def read(gate, arms: Sequence[str] = ARMS, refs: Sequence[str] = REFS) -> dict:
    cells, failed = R.load(gate)
    cells = {k: v for k, v in cells.items() if k[4] in RUNGS}
    failed = {k: v for k, v in failed.items() if k[4] in RUNGS}
    topos = check_topologies(list(cells) + list(failed))
    bad = sorted({k[2] for k in failed if k[0] in arms})
    sens = {k: v for k, v in cells.items() if k[2] not in bad}
    keep = [t for t in topos if t not in bad]
    return {"gate": gate, "arms": list(arms), "topologies": topos, "n_summaries": len(cells), "n_failed": len(failed),
            "arm_table": R.arm_table(cells, failed, {}), "collapse": collapse(cells, failed, arms),
            "families": {ref: family(cells, topos, arms, ref) for ref in refs},
            "sensitivity": {"excluded_topologies": bad, "families": {ref: family(sens, keep, arms, ref) for ref in refs}}}


def print_report(r: dict) -> None:
    f = lambda x, n=4: "-" if x is None else f"{x:.{n}f}"
    print(f"{r['n_summaries']} summaries, {r['n_failed']} failed, {len(r['topologies'])} topologies")
    print("-- cells per arm and rung (finished/failed); p95 s, end/last arrival, effective share (median/worst)")
    for k, a in r["arm_table"].items():
        if k.split("|")[0] in r["arms"]:
            print(f"  {k:22s} {a['finished']}/{a['failed']}  p95 {f(a['median_p95_s'], 1)}/{f(a['worst_p95_s'], 1)}  "
                  f"end {f(a['median_end_over_last_arrival'], 2)}/{f(a['worst_end_over_last_arrival'], 2)}  "
                  f"share {f(a['median_effective_share'], 2)}/{f(a['worst_effective_share'], 2)}" + (f"  failed why {a['failed_why']}" if a["failed"] else ""))
    print("-- collapse record (cells beyond end/last-arrival 1.5, effective share 0.80; display thresholds)")
    for k, c in r["collapse"].items():
        print(f"  {k:22s} {c}")
    for ref, fam in r["families"].items():
        print(f"-- vs {ref} (Holm over {fam['holm_over']}; positive = the classical arm is slower)")
        for k, c in fam["tests"].items():
            m = "-" if c["median_pct"] is None else format(c["median_pct"], "+.2f") + " %"
            print(f"  {k:22s} median {m:>10s} n={c['n_topologies']:2d} wins={c['wins']} p={f(c['p'])} holm={f(c.get('holm_p'))}  {fam['labels'][k]}")
    if r["sensitivity"]["excluded_topologies"]:
        print(f"-- sensitivity: families repeated without topologies {r['sensitivity']['excluded_topologies']} (in --out json)")


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
    print_report(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
