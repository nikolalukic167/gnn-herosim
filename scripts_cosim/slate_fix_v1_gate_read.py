#!/usr/bin/env python3
"""slate_fix_v1 live-gate reader (docs/lineages/slate_fix_v1.md). Written before the data.

Topologies 19420-19431 (anything else is refused), windows g0-g1, moderate and heavy. Two gate submissions run the same harness kinds on
different checkpoints, so the reader renames a learned kind by the directory it came from: --new (R1A_STEM=slate-fix-v1, the uncapped retrain)
-> "new:<kind>", --old (R1A_STEM=accel-replica-v1, the capped g4 checkpoints) -> "old:<kind>". The hand arms (cd, cd_exactS) keep their names
and must appear in exactly one directory. Gate B2 (cd_exactS_ib0, run at d192081f, the commit that defines it) is passed as a second --old dir.

Statistic as every gate of the line (r1_attribution_v1_read): per topology the median over (window, seed) of the paired % of an arm's mean
latency against the reference (the reference's own seed if seeded, else its seed 0), then the median over topologies and an exact two-sided
Wilcoxon. Negative = the arm is faster.
  * primary, Holm over 8 (amended 2026-10-12 before any gate read: the hand bar is cd_exactS_ib0 from ib0_confirm_v1; cd_exactS stays as the
    secondary reference): new gnn_eng self-search and new gnn_eng no-split vs cd_exactS_ib0 and vs cd_exactS x {moderate, heavy}. Per arm and
    reference: E-FASTER (the reference Holm-confirmed faster at some rung), else WIN (median <= -2 % with Holm p < .05 at BOTH rungs), else
    DIRECTION-ONLY (median < 0 at both), else NOT-SEPARATED. Per rung: CONFIRMED / E-FASTER / DIRECTION-ONLY / NOT-SEPARATED;
  * secondary "data effect", Holm over 4: new vs old checkpoint, self-search and no-split x 2 rungs;
  * secondary "MP", Holm over 4: new gnn_eng vs new twin_eng, self-search and no-split x 2 rungs;
  * descriptive, outside every family: the gap change (new vs cd_exactS minus old vs cd_exactS, pp, per topology then median), old arms vs
    cd_exactS, new cdxapply vs cd_exactS and vs CD, cd_exactS vs CD (the comparability check: cd_exactS must be faster than CD at both rungs,
    else the gate is NOT-COMPARABLE with the seat selfsearch_confirm_v1 measured), new self-search vs new no-split.
A missing arm counts as p = 1 in its family. Counts come first; a second pass repeats every family without the topologies that have a failure.

  slate_fix_v1_gate_read.py --new <dir> [<dir> ...] --old <dir> [<dir> ...] [--out read.json]
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import os
import sys
from typing import Dict, List, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r1_attribution_v1_read as R  # noqa: E402

RUNGS = ("moderate", "heavy")
WINDOWS = ("g0", "g1")
TOPO_RANGE = (19420, 19431)
E, CD, IB0 = "cd_exactS", "cd", "cd_exactS_ib0"
HAND = (E, CD, IB0)
SS, NS = "ra_gnn_eng_selfsearch", "ra_gnn_eng_nosplit"
TWIN_SS, TWIN_NS = "ra_twin_eng_selfsearch", "ra_twin_eng_nosplit"
CDX = "ra_gnn_eng_cdxapply"
BAR_PCT, ALPHA = -2.0, 0.05


def rung_label(c: dict, faster: str = "E-FASTER") -> str:
    if c["median_pct"] is None or c.get("holm_p") is None:
        return "NOT-SEPARATED"
    if c["median_pct"] > 0 and c["holm_p"] < ALPHA:
        return faster
    if c["median_pct"] <= BAR_PCT and c["holm_p"] < ALPHA:
        return "CONFIRMED"
    return "DIRECTION-ONLY" if c["median_pct"] < 0 else "NOT-SEPARATED"


def arm_label(tests: Sequence[dict]) -> str:
    if any(c["median_pct"] is None or c.get("holm_p") is None for c in tests):
        return "NOT-SEPARATED"
    if any(c["median_pct"] > 0 and c["holm_p"] < ALPHA for c in tests):
        return "E-FASTER"
    if all(c["median_pct"] <= BAR_PCT and c["holm_p"] < ALPHA for c in tests):
        return "WIN"
    return "DIRECTION-ONLY" if all(c["median_pct"] < 0 for c in tests) else "NOT-SEPARATED"


def load(new: Sequence[str], old: Sequence[str]):
    cells, failed = {}, {}
    for tag, dirs in (("new", new), ("old", old)):
        c, f = R.load([dirs] if isinstance(dirs, str) else list(dirs))
        for src, dst in ((c, cells), (f, failed)):
            for (kind, seed, topo, win, rung), v in src.items():
                k = (f"{tag}:{kind}" if kind.startswith("ra_") else kind, seed, topo, win, rung)
                if k in cells or k in failed:
                    raise SystemExit(f"FAIL LOUD: cell {k} appears in both the --new and the --old directories")
                dst[k] = v
    return cells, failed


def check_grid(keys) -> List[int]:
    topos = sorted({k[2] for k in keys})
    out = [t for t in topos if not TOPO_RANGE[0] <= t <= TOPO_RANGE[1]]
    if out:
        raise SystemExit(f"FAIL LOUD: topologies outside {TOPO_RANGE[0]}-{TOPO_RANGE[1]} in the gate directories: {out}")
    wins = sorted({k[3] for k in keys} - set(WINDOWS))
    if wins:
        raise SystemExit(f"FAIL LOUD: windows {wins} are not in the registered {WINDOWS}")
    return topos


def fam(cells: dict, topos: List[int], pairs: Sequence[tuple], faster: str) -> dict:
    """pairs: (arm, reference, name). Holm over len(pairs) x 2 rungs."""
    tests = {f"{n}|{r}": R.contrast(cells, a, b, r, topos) for a, b, n in pairs for r in RUNGS}
    adj = R.holm({k: c["p"] for k, c in tests.items()})
    for k, c in tests.items():
        c["holm_p"] = adj[k]
    return {"holm_over": len(tests), "tests": tests, "rung_labels": {k: rung_label(c, faster) for k, c in tests.items()}}


def gap_change(cells: dict, topos: List[int], arm: str, rung: str) -> dict:
    """Per topology (new arm vs E) - (old arm vs E) in pp; median over the topologies that have both. Negative = the gap shrank."""
    new = R.paired(cells, f"new:{arm}", E, rung, topos, set())
    old = R.paired(cells, f"old:{arm}", E, rung, topos, set())
    d = {t: new[t]["median_pct"] - old[t]["median_pct"] for t in topos if t in new and t in old}
    if not d:
        return {"n_topologies": 0, "median_pp": None}
    v = list(d.values())
    return {"n_topologies": len(v), "median_pp": st.median(v), "shrank": sum(1 for x in v if x < 0),
            "per_topology": {str(t): round(x, 3) for t, x in sorted(d.items())}}


def families(cells: dict, topos: List[int]) -> dict:
    pp = [(f"new:{SS}", IB0, "new self-search vs cd_exactS_ib0"), (f"new:{NS}", IB0, "new no-split vs cd_exactS_ib0"),
          (f"new:{SS}", E, "new self-search vs cd_exactS"), (f"new:{NS}", E, "new no-split vs cd_exactS")]
    prim = fam(cells, topos, pp, "E-FASTER")
    prim["labels"] = {n: arm_label([prim["tests"][f"{n}|{r}"] for r in RUNGS]) for _, _, n in pp}
    data = fam(cells, topos, [(f"new:{SS}", f"old:{SS}", "self-search new vs old"), (f"new:{NS}", f"old:{NS}", "no-split new vs old")], "OLD-FASTER")
    mp = fam(cells, topos, [(f"new:{SS}", f"new:{TWIN_SS}", "self-search gnn vs twin"), (f"new:{NS}", f"new:{TWIN_NS}", "no-split gnn vs twin")],
             "TWIN-FASTER")
    desc = {}
    for a, b, n in ((f"old:{SS}", E, "old self-search vs cd_exactS"), (f"old:{NS}", E, "old no-split vs cd_exactS"),
                    (f"new:{CDX}", E, "new cdxapply vs cd_exactS"), (f"new:{CDX}", CD, "new cdxapply vs cd"), (E, CD, "cd_exactS vs cd"), (IB0, E, "cd_exactS_ib0 vs cd_exactS"), (IB0, CD, "cd_exactS_ib0 vs cd"),
                    (f"new:{SS}", CD, "new self-search vs cd"), (f"new:{SS}", f"new:{NS}", "new self-search vs new no-split")):
        for r in RUNGS:
            desc[f"{n}|{r}"] = R.contrast(cells, a, b, r, topos)
    gaps = {f"{arm}|{r}": gap_change(cells, topos, arm, r) for arm in (SS, NS) for r in RUNGS}
    ce = [desc[f"cd_exactS vs cd|{r}"]["median_pct"] for r in RUNGS]
    comparable = all(x is not None and x < 0 for x in ce)
    return {"primary": prim, "data_effect": data, "mp": mp, "descriptive": desc, "gap_change_pp": gaps,
            "comparable": comparable, "verdict": "NOT-COMPARABLE" if not comparable else None}


def slate_counts(cells: dict) -> dict:
    """Per learned arm and rung: declared-slate batches, batches whose slate was pruned (0 under the uncapped rule), sub-batched and unsplit."""
    out = {}
    for arm in sorted({k[0] for k in cells if ":" in k[0]}):
        for r in RUNGS:
            cs = [s for k, s in cells.items() if k[0] == arm and k[4] == r]
            if not cs:
                continue
            tot = lambda n: sum(int((s.get("schedulerCounters") or {}).get(n) or 0) for s in cs)
            out[f"{arm}|{r}"] = {"cells": len(cs), **{n: tot(n) for n in ("slate_declared_batches", "slate_declared_pruned",
                                                                         "slate_declared_sub_batched", "slate_declared_unsplit",
                                                                         "ss_batches", "ss_exact_batches", "ss_ascent_batches")}}
    return out


def read(new: Sequence[str], old: Sequence[str]) -> dict:
    cells, failed = load(new, old)
    cells = {k: v for k, v in cells.items() if k[4] in RUNGS}
    failed = {k: v for k, v in failed.items() if k[4] in RUNGS}
    topos = check_grid(list(cells) + list(failed))
    bad = sorted({k[2] for k in failed})
    sens = {k: v for k, v in cells.items() if k[2] not in bad}
    new, old = ([d] if isinstance(d, str) else list(d) for d in (new, old))
    return {"new": new, "old": old, "topologies": topos, "n_summaries": len(cells), "n_failed": len(failed),
            "arm_table": R.arm_table(cells, failed, {}), "slate": slate_counts(cells), "main": families(cells, topos),
            "sensitivity": {"excluded_topologies": bad, "families": families(sens, [t for t in topos if t not in bad])}}


def print_report(r: dict) -> None:
    f = lambda x, n=2: "-" if x is None else f"{x:.{n}f}"
    m = r["main"]
    print(f"{r['n_summaries']} summaries, {r['n_failed']} failed, {len(r['topologies'])} topologies"
          + (" | NOT-COMPARABLE: cd_exactS is not faster than CD at both rungs" if not m["comparable"] else ""))
    print("-- cells per arm and rung (finished/failed, median wall s)")
    for k, a in r["arm_table"].items():
        print(f"  {k:40s} {a['finished']:4d}/{a['failed']:<3d} wall {f(a['median_wallclock_s'], 0)}")
    for fam_name in ("primary", "data_effect", "mp"):
        fm = m[fam_name]
        print(f"-- {fam_name} (Holm over {fm['holm_over']})")
        for k, c in fm["tests"].items():
            print(f"  {k:44s} {f(c['median_pct'])} % ({c['wins']}/{c['n_topologies']}) p {f(c['p'], 4)} Holm {f(c['holm_p'], 4)}  "
                  f"{fm['rung_labels'][k]}")
    for n, lab in m["primary"]["labels"].items():
        print(f"  ARM {n}: {lab}")
    print("-- gap change, pp (new vs cd_exactS minus old vs cd_exactS; negative = shrank)")
    for k, g in m["gap_change_pp"].items():
        print(f"  {k:40s} {f(g['median_pp'])} pp (shrank on {g.get('shrank')}/{g['n_topologies']})")
    print("-- descriptive")
    for k, c in m["descriptive"].items():
        print(f"  {k:44s} {f(c['median_pct'])} % ({c['wins']}/{c['n_topologies']}) p {f(c['p'], 4)}")
    if r["sensitivity"]["excluded_topologies"]:
        print(f"-- sensitivity without {r['sensitivity']['excluded_topologies']}: primary labels "
              f"{r['sensitivity']['families']['primary']['labels']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--new", nargs="+", required=True)
    ap.add_argument("--old", nargs="+", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    r = read(a.new, a.old)
    if a.out:
        with open(a.out + ".partial", "w") as fh:
            json.dump(r, fh, indent=1)
        os.replace(a.out + ".partial", a.out)
    print_report(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
