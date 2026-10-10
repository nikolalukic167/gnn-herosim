#!/usr/bin/env python3
"""cost_to_go_v1 live-gate reader (docs/lineages/cost_to_go_v1.md, "Gate"), written before any gate cell ran.

Statistic (the r1_attribution_v1 reader's, imported): per topology, the median over windows of the paired % of an arm's mean
latency against its reference on the same (topology, window, rung); then the median over topologies; an exact two-sided Wilcoxon
signed-rank test over the topologies. Negative = the arm is faster.
  * primary family, Holm over 4: cd_exactS + hand V vs cd_exactS and cd_exactS + MLP V vs cd_exactS, at the moderate and heavy
    rungs. An arm WINS if, at BOTH rungs, its median is <= -5 % and its Holm-adjusted p < .05.
  * descriptive (no family): plain CD vs cd_exactS; each V arm vs plain CD; the MLP V vs the hand V.
A failed cell drops out of that arm's tests; failures are counted per arm and rung, and the read is repeated with every topology
that has a failure excluded for every arm (`sensitivity`). Per arm and rung it reports the median effective queue share, p95, run
end over the last arrival, and the value re-rank's own counters (batches re-ranked, plans changed).

  cost_to_go_v1_gate_read.py --gate DIR [DIR ...] --out read.json
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import statistics as st
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r1_attribution_v1_read import arm_table, contrast, holm, load  # noqa: E402

RUNGS = ("moderate", "heavy")
BAR = "cd_exactS"
V_ARMS = ("cd_exactS_hv", "cd_exactS_mv")
WIN_PCT = -5.0
ALPHA = 0.05


def families(cells: dict, topos: List[int]) -> dict:
    prim = {f"{a} vs {BAR}|{r}": contrast(cells, a, BAR, r, topos) for a in V_ARMS for r in RUNGS}
    adj = holm({k: c["p"] for k, c in prim.items()})
    for k, c in prim.items():
        c["holm_p"] = adj[k]
        if c["n_topologies"] == 0:
            c["note"] = "not run (p = 1 in the family)"
    verdict = {}
    for a in V_ARMS:
        ok = [prim[f"{a} vs {BAR}|{r}"] for r in RUNGS]
        win = all(c["median_pct"] is not None and c["median_pct"] <= WIN_PCT and c["holm_p"] is not None and c["holm_p"] < ALPHA for c in ok)
        verdict[a] = "WIN" if win else "NO-WIN"
    desc = {f"cd vs {BAR}|{r}": contrast(cells, "cd", BAR, r, topos) for r in RUNGS}
    desc.update({f"{a} vs cd|{r}": contrast(cells, a, "cd", r, topos) for a in V_ARMS for r in RUNGS})
    desc.update({f"cd_exactS_mv vs cd_exactS_hv|{r}": contrast(cells, "cd_exactS_mv", "cd_exactS_hv", r, topos) for r in RUNGS})
    return {"primary": {"holm_over": len(prim), "tests": prim, "verdict": verdict}, "descriptive": desc}


def v_counters(cells: dict) -> Dict[str, dict]:
    out = {}
    for a in V_ARMS:
        for r in RUNGS:
            cs = [s for k, s in cells.items() if k[0] == a and k[4] == r]
            if not cs:
                continue
            get = lambda name: [int((s.get("schedulerCounters") or s.get("counters") or {}).get(name) or 0) for s in cs]
            vb, vc, ex = get("pg_v_batches"), get("pg_v_changed"), get("pg_exact_batches")
            out[f"{a}|{r}"] = {"cells": len(cs), "v_batches": sum(vb), "v_changed": sum(vc), "exact_batches": sum(ex),
                               "changed_share": sum(vc) / sum(vb) if sum(vb) else None}
    return out


def read(gate) -> dict:
    cells, failed = load(gate)
    topos = sorted({k[2] for k in list(cells) + list(failed)})
    bad = sorted({k[2] for k in failed})
    sens = {k: v for k, v in cells.items() if k[2] not in bad}
    return {"gate": gate, "topologies": topos, "n_summaries": len(cells), "n_failed": len(failed),
            "code_versions": sorted({str(s.get("code")) for s in cells.values()}), "arms": arm_table(cells, failed, {}),
            "v_counters": v_counters(cells), "main": families(cells, topos),
            "sensitivity": {"excluded_topologies": bad, "families": families(sens, [t for t in topos if t not in bad])}}


def print_report(r: dict) -> None:
    f = lambda x, n=3: "-" if x is None else f"{x:.{n}f}"
    print(f"{r['n_summaries']} summaries, {r['n_failed']} failed, {len(r['topologies'])} topologies, code {r['code_versions']}")
    for v in r["arms"].values():
        print(f"  {v['kind']:16s} {v['rung']:9s} {v['finished']}/{v['failed']} eff.share {f(v['median_effective_share'])} "
              f"p95 {f(v['median_p95_s'], 1)} end/last arrival worst {f(v['worst_end_over_last_arrival'])} latency {f(v['median_latency_s'], 2)} s")
    for k, c in r["v_counters"].items():
        print(f"  value re-rank {k}: {c['v_batches']} batches re-ranked, {c['v_changed']} changed ({f(c['changed_share'])}), exact batches {c['exact_batches']}")
    for label, fam in (("main", r["main"]), (f"sensitivity (excluding {r['sensitivity']['excluded_topologies']})", r["sensitivity"]["families"])):
        print(f"== {label}")
        print(f"-- primary, Holm over {fam['primary']['holm_over']}; WIN = median <= {WIN_PCT} % and Holm p < {ALPHA} at both rungs")
        for k, c in fam["primary"]["tests"].items():
            print(f"  {k:36s} median {('-' if c['median_pct'] is None else format(c['median_pct'], '+.2f') + ' %'):>10s} n={c['n_topologies']:2d} "
                  f"wins={c['wins']} p={f(c['p'], 4)} holm={f(c.get('holm_p'), 4)} {c.get('note', '')}")
        print(f"  verdict: {fam['primary']['verdict']}")
        print("-- descriptive")
        for k, c in fam["descriptive"].items():
            print(f"  {k:36s} median {('-' if c['median_pct'] is None else format(c['median_pct'], '+.2f') + ' %'):>10s} n={c['n_topologies']:2d} "
                  f"wins={c['wins']} p={f(c['p'], 4)}")


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
    raise SystemExit(main())
