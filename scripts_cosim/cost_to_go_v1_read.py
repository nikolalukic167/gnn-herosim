"""cost_to_go_v1 live-gate reader (docs/lineages/cost_to_go_v1.md).

Statistic as accel_nosplit_v1_read.py: per topology the median over (window, seed) of the paired % of a gated arm's mean latency against the
reference cd_exactS on the same (topology, window, seed-or-seed-0); then the median over the 24 topologies and an exact two-sided Wilcoxon.
  * primary family, Holm over (arms x 2 rungs): each gated V arm vs cd_exactS x {moderate, heavy}; a missing arm counts as p = 1;
  * arm label: REF-FASTER (the reference is Holm-confirmed faster, median > 0 and Holm p < .05, at some rung), else WIN (median <= -5 % with
    Holm p < .05 at BOTH rungs), else DIRECTION-ONLY (median < 0 at both), else NOT-SEPARATED. Per-rung: CONFIRMED / REF-FASTER /
    DIRECTION-ONLY / NOT-SEPARATED;
  * descriptive, outside the family and never the bar: each gated arm vs each descriptive reference (default cd, cd_expand, cdxapply).
Arm and reference names are parameters (--arms, --ref, --descriptive-refs); the registered gate fixes them. The reader refuses any topology
outside 16369-16392, prints the counts and the failures first, and repeats the family without every topology that has a failure (`sensitivity`).

  cost_to_go_v1_read.py --gate <dir> [<dir> ...] --arms cdexs_handv [mlpv ...] [--ref cd_exactS] [--out read.json]
"""
import argparse
import json
import signal
import sys
from typing import Dict, List, Sequence

import r1_attribution_v1_read as R

RUNGS = ("moderate", "heavy")
TOPO_RANGE = (16369, 16392)
REF = "cd_exactS"
DESCRIPTIVE_REFS = ("cd", "cd_expand", "cdxapply")
BAR_PCT, ALPHA = -5.0, 0.05


def rung_label(c: dict) -> str:
    if c["median_pct"] is None:
        return "NOT-SEPARATED"
    if c["median_pct"] > 0 and c["holm_p"] < ALPHA:
        return "REF-FASTER"
    if c["median_pct"] <= BAR_PCT and c["holm_p"] < ALPHA:
        return "CONFIRMED"
    return "DIRECTION-ONLY" if c["median_pct"] < 0 else "NOT-SEPARATED"


def label(tests: Dict[str, dict]) -> str:
    """tests: rung -> contrast with holm_p."""
    cs = [tests[r] for r in RUNGS]
    if any(c["median_pct"] is None for c in cs):
        return "NOT-SEPARATED"
    if any(c["median_pct"] > 0 and c["holm_p"] < ALPHA for c in cs):
        return "REF-FASTER"
    if all(c["median_pct"] <= BAR_PCT and c["holm_p"] < ALPHA for c in cs):
        return "WIN"
    if all(c["median_pct"] < 0 for c in cs):
        return "DIRECTION-ONLY"
    return "NOT-SEPARATED"


def check_topologies(keys) -> List[int]:
    topos = sorted({k[2] for k in keys})
    out = [t for t in topos if not TOPO_RANGE[0] <= t <= TOPO_RANGE[1]]
    if out:
        raise SystemExit(f"FAIL LOUD: topologies outside {TOPO_RANGE[0]}-{TOPO_RANGE[1]} in the gate directories: {out}")
    return topos


def family(cells: dict, topos: List[int], arms: Sequence[str], ref: str, drefs: Sequence[str]) -> dict:
    tests = {(a, r): R.contrast(cells, a, ref, r, topos) for a in arms for r in RUNGS}
    adj = R.holm({f"{a}|{r}": c["p"] for (a, r), c in tests.items()})
    for (a, r), c in tests.items():
        c["holm_p"] = adj[f"{a}|{r}"]
    return {"holm_over": len(tests), "ref": ref,
            "tests": {f"{a}|{r}": c for (a, r), c in tests.items()},
            "labels": {a: label({r: tests[(a, r)] for r in RUNGS}) for a in arms},
            "rung_labels": {f"{a}|{r}": rung_label(c) for (a, r), c in tests.items()},
            "descriptive": {f"{a} vs {d}|{r}": R.contrast(cells, a, d, r, topos) for a in arms for d in drefs for r in RUNGS}}


def read(gate, arms: Sequence[str], ref: str = REF, drefs: Sequence[str] = DESCRIPTIVE_REFS) -> dict:
    cells, failed = R.load(gate)
    cells = {k: v for k, v in cells.items() if k[4] in RUNGS}
    failed = {k: v for k, v in failed.items() if k[4] in RUNGS}
    topos = check_topologies(list(cells) + list(failed))
    bad = sorted({k[2] for k in failed})
    sens = {k: v for k, v in cells.items() if k[2] not in bad}
    return {"gate": gate, "arms_gated": list(arms), "ref": ref, "topologies": topos, "n_summaries": len(cells), "n_failed": len(failed),
            "arms": R.arm_table(cells, failed, {}),
            "main": family(cells, topos, arms, ref, drefs),
            "sensitivity": {"excluded_topologies": bad, "family": family(sens, [t for t in topos if t not in bad], arms, ref, drefs)}}


def print_report(r: dict) -> None:
    f = lambda x, n=4: "-" if x is None else f"{x:.{n}f}"
    print(f"{r['n_summaries']} summaries, {r['n_failed']} failed, {len(r['topologies'])} topologies")
    print("-- cells per arm and rung (finished/failed)")
    for k, a in r["arms"].items():
        print(f"  {k:34s} {a['finished']}/{a['failed']}" + (f"  failed why {a['failed_why']}" if a["failed"] else ""))
    for name, fam in (("main", r["main"]), ("sensitivity", r["sensitivity"]["family"])):
        print(f"-- primary family ({name}, vs {fam['ref']}, Holm over {fam['holm_over']}); labels {fam['labels']}")
        for k, c in {**fam["tests"], **fam["descriptive"]}.items():
            print(f"  {k:44s} median {('-' if c['median_pct'] is None else format(c['median_pct'], '+.2f') + ' %'):>10s} n={c['n_topologies']:2d} "
                  f"wins={c['wins']} p={f(c['p'])} holm={f(c.get('holm_p'))}")
    print("-- per-rung labels (main): " + ", ".join(f"{k} {v}" for k, v in r["main"]["rung_labels"].items()))


def main() -> int:
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", nargs="+", required=True)
    ap.add_argument("--arms", nargs="+", required=True, help="the gated V arms (fixed when the gate is registered)")
    ap.add_argument("--ref", default=REF)
    ap.add_argument("--descriptive-refs", nargs="*", default=list(DESCRIPTIVE_REFS))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    r = read(a.gate, a.arms, a.ref, a.descriptive_refs)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(r, fh, indent=1)
    print_report(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
