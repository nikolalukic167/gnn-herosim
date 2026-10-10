"""selfsearch_confirm_v1 live-gate reader (docs/lineages/selfsearch_confirm_v1.md). Written before the data.

Topologies 16490-16513 (anything else is refused), g0-g3, moderate and heavy. Statistic as every gate of the line: per topology the median over
(window, seed) of the paired % of an arm's mean latency against the reference (the reference's own seed if seeded, else its seed 0), then the median
over topologies and an exact two-sided Wilcoxon. Negative = the arm is faster.
  * primary, Holm over 4: {gnn_eng, gnn_eng_physmp} self-search vs CD x {moderate, heavy}. Arm label: CD-FASTER (CD Holm-confirmed faster at some
    rung), else WIN (median <= -5 % with Holm p < .05 at BOTH rungs), else DIRECTION-ONLY (median < 0 at both), else NOT-SEPARATED. Per-rung:
    CONFIRMED / CD-FASTER / DIRECTION-ONLY / NOT-SEPARATED;
  * secondary "does the search add", Holm over 4: self-search vs the same checkpoint's no-split x 2 checkpoints x 2 rungs;
  * secondary "MP", Holm over 2: gnn_eng self-search vs the MP-OFF twin's self-search x 2 rungs. Recorded as NOT RUN when the twin arm has no cells;
  * descriptive, outside every family: each self-search arm vs cd_exactS and vs cdxapply; the no-split arms vs CD.
A missing arm counts as p = 1 in its family. Counts come first (summaries and failures per arm and rung, then exact vs ascent batches, plan-changed
share and decision time per self-search arm; decision time is reported, never scored). A second pass repeats every family without the topologies that
have a failure (`sensitivity`).

  selfsearch_confirm_v1_read.py --gate <dir> [<dir> ...] [--out read.json]
"""
import argparse
import json
import signal
import statistics as st
import sys
from typing import Dict, List, Sequence

import r1_attribution_v1_read as R

RUNGS = ("moderate", "heavy")
TOPO_RANGE = (16490, 16513)
SS = {"ra_gnn_eng": "ra_gnn_eng_selfsearch", "ra_gnn_eng_physmp": "ra_gnn_eng_physmp_selfsearch"}
NOSPLIT = {"ra_gnn_eng": "ra_gnn_eng_nosplit", "ra_gnn_eng_physmp": "ra_gnn_eng_physmp_nosplit"}
TWIN = "ra_twin_eng_selfsearch"
CD, HAND = "cd", ("cd_exactS", "ra_gnn_eng_cdxapply")
BAR_PCT, ALPHA = -5.0, 0.05


def rung_label(c: dict) -> str:
    if c["median_pct"] is None or c.get("holm_p") is None:
        return "NOT-SEPARATED"
    if c["median_pct"] > 0 and c["holm_p"] < ALPHA:
        return "CD-FASTER"
    if c["median_pct"] <= BAR_PCT and c["holm_p"] < ALPHA:
        return "CONFIRMED"
    return "DIRECTION-ONLY" if c["median_pct"] < 0 else "NOT-SEPARATED"


def label(tests: Dict[str, dict]) -> str:
    cs = [tests[r] for r in RUNGS]
    if any(c["median_pct"] is None or c.get("holm_p") is None for c in cs):
        return "NOT-SEPARATED"
    if any(c["median_pct"] > 0 and c["holm_p"] < ALPHA for c in cs):
        return "CD-FASTER"
    if all(c["median_pct"] <= BAR_PCT and c["holm_p"] < ALPHA for c in cs):
        return "WIN"
    return "DIRECTION-ONLY" if all(c["median_pct"] < 0 for c in cs) else "NOT-SEPARATED"


def check_topologies(keys) -> List[int]:
    topos = sorted({k[2] for k in keys})
    out = [t for t in topos if not TOPO_RANGE[0] <= t <= TOPO_RANGE[1]]
    if out:
        raise SystemExit(f"FAIL LOUD: topologies outside {TOPO_RANGE[0]}-{TOPO_RANGE[1]} in the gate directories: {out}")
    return topos


def fam(cells: dict, topos: List[int], pairs: Sequence[tuple]) -> dict:
    """pairs: (arm, reference, name). Holm over len(pairs) x 2 rungs."""
    tests = {f"{n}|{r}": R.contrast(cells, a, b, r, topos) for a, b, n in pairs for r in RUNGS}
    adj = R.holm({k: c["p"] for k, c in tests.items()})
    for k, c in tests.items():
        c["holm_p"] = adj[k]
    return {"holm_over": len(tests), "tests": tests, "rung_labels": {k: rung_label(c) for k, c in tests.items()}}


def families(cells: dict, topos: List[int]) -> dict:
    prim = fam(cells, topos, [(a, CD, f"{ck} self-search vs cd") for ck, a in SS.items()])
    prim["labels"] = {ck: label({r: prim["tests"][f"{ck} self-search vs cd|{r}"] for r in RUNGS}) for ck in SS}
    add = fam(cells, topos, [(SS[ck], NOSPLIT[ck], f"{ck} self-search vs no-split") for ck in SS])
    twin_on = any(k[0] == TWIN for k in cells)
    mp = fam(cells, topos, [(SS["ra_gnn_eng"], TWIN, "gnn_eng self-search vs twin self-search")]) if twin_on else {"not_run": True}
    desc = {f"{n} vs {b}|{r}": R.contrast(cells, a, b, r, topos)
            for ck, a in SS.items() for b in HAND for n in [f"{ck} self-search"] for r in RUNGS}
    desc.update({f"{ck} no-split vs cd|{r}": R.contrast(cells, NOSPLIT[ck], CD, r, topos) for ck in NOSPLIT for r in RUNGS})
    return {"primary": prim, "search_adds": add, "mp": mp, "descriptive": desc}


def search_counts(cells: dict) -> dict:
    out = {}
    for arm in (*SS.values(), TWIN):
        for r in RUNGS:
            cs = [s for k, s in cells.items() if k[0] == arm and k[4] == r]
            if not cs:
                continue
            tot = lambda n: sum(int((s.get("schedulerCounters") or {}).get(n) or 0) for s in cs)
            b, t = tot("ss_batches"), tot("ss_tasks")
            dt = [s["decisionTiming"] for s in cs if s.get("decisionTiming")]
            out[f"{arm}|{r}"] = {"cells": len(cs), "batches": b, "slate_declared_batches": tot("slate_declared_batches"),
                                 "exact_batches": tot("ss_exact_batches"), "ascent_batches": tot("ss_ascent_batches"),
                                 "changed_batch_share": tot("ss_changed_batches") / b if b else None,
                                 "changed_task_share": tot("ss_changed_tasks") / t if t else None,
                                 "decision_per_task_median_s": st.median(d["per_task_median_s"] for d in dt) if dt else None}
    return out


def read(gate) -> dict:
    cells, failed = R.load(gate)
    cells = {k: v for k, v in cells.items() if k[4] in RUNGS}
    failed = {k: v for k, v in failed.items() if k[4] in RUNGS}
    topos = check_topologies(list(cells) + list(failed))
    bad = sorted({k[2] for k in failed})
    sens = {k: v for k, v in cells.items() if k[2] not in bad}
    return {"gate": gate, "topologies": topos, "n_summaries": len(cells), "n_failed": len(failed),
            "arm_table": R.arm_table(cells, failed, {}), "search": search_counts(cells),
            "main": families(cells, topos), "twin_run": any(k[0] == TWIN for k in cells),
            "sensitivity": {"excluded_topologies": bad, "families": families(sens, [t for t in topos if t not in bad])}}


def print_report(r: dict) -> None:
    f = lambda x, n=4: "-" if x is None else f"{x:.{n}f}"
    print(f"{r['n_summaries']} summaries, {r['n_failed']} failed, {len(r['topologies'])} topologies")
    print("-- cells per arm and rung (finished/failed)")
    for k, a in r["arm_table"].items():
        print(f"  {k:40s} {a['finished']}/{a['failed']}" + (f"  failed why {a['failed_why']}" if a["failed"] else ""))
    print("-- search counts (decision time reported, not scored)")
    for k, c in r["search"].items():
        print(f"  {k}: {c}")
    print(f"-- MP twin arm: {'run' if r['twin_run'] else 'NOT RUN (no cells); no MP claim possible'}")
    for name, F in (("main", r["main"]), ("sensitivity", r["sensitivity"]["families"])):
        print(f"== {name}")
        for fname in ("primary", "search_adds", "mp"):
            fm = F[fname]
            if fm.get("not_run"):
                print(f"-- {fname}: not run")
                continue
            print(f"-- {fname} (Holm over {fm['holm_over']})" + (f"; labels {fm['labels']}" if "labels" in fm else ""))
            for k, c in fm["tests"].items():
                m = "-" if c["median_pct"] is None else format(c["median_pct"], "+.2f") + " %"
                print(f"  {k:52s} median {m:>10s} n={c['n_topologies']:2d} wins={c['wins']} p={f(c['p'])} holm={f(c.get('holm_p'))}  {fm['rung_labels'][k]}")
        print("-- descriptive (no family)")
        for k, c in F["descriptive"].items():
            m = "-" if c["median_pct"] is None else format(c["median_pct"], "+.2f") + " %"
            print(f"  {k:52s} median {m:>10s} n={c['n_topologies']:2d} wins={c['wins']} p={f(c['p'])}")


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
