#!/usr/bin/env python3
"""r1_attribution_v1 live-gate reader (docs/lineages/r1_attribution_v1.md, "Arms" / "Primary family" / "Secondary families").

Statistic (as transfer_physics_v1 / small_batch_so_v1): per topology, the median over (window, checkpoint seed) of the paired % of an
arm's mean latency against its reference on the same (topology, window); then the median over topologies; an exact Wilcoxon
signed-rank test over the topologies (two-sided; zero differences dropped; mid-ranks for ties). Negative = the arm is faster.
  * primary family, Holm over 3: the best learned arm (declared from validation before any test read: --best-arm) vs CD, one test per rung;
  * S1, Holm over 15: GNN-eng vs Twin-eng, GNN-raw vs Twin-raw, GNN-eng vs MLP-same, GNN-eng vs Set-transformer, GNN-eng-physMP vs GNN-eng;
  * S2, Holm over 9: CD<-GNN vs CD, CD<-GNN vs CD<-Twin, CD<-random vs CD. CD<-random is `cd_random_seed` (seeded per cell); a test with no cells stays in the family at p = 1 (the registered family size is kept);
  * CD-declared (descriptive) and Knative (context) are reported against CD and enter no family.
A cell that failed (failed.json) drops out of that arm's tests only; failures are counted per arm and rung. The whole read is repeated
with every topology that has any remaining failure excluded for every arm (`sensitivity`). Per arm and rung it also reports the
median effective queue share, p95 and run end over last arrival.

  r1_attribution_v1_read.py --gate <dir with *.summary.json> --best-arm ra_gnn_eng --out read.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import statistics as st
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

RUNGS = ("light", "moderate", "heavy")
NAME = re.compile(r"cc40s(?P<topo>\d+)__(?P<win>g\d)(?P<rung>light|moderate|heavy)__(?P<kind>.+)_s(?P<seed>\d+)$")
LEARNED = ("ra_gnn_eng", "ra_twin_eng", "ra_mlp_same", "ra_gnn_raw", "ra_twin_raw", "ra_gnn_eng_physmp", "ra_set_transformer")
CD_GNN, CD_TWIN, CD_RANDOM = "ra_gnn_eng_cdapply", "ra_twin_eng_cdapply", "cd_random_seed"
S1 = (("ra_gnn_eng", "ra_twin_eng"), ("ra_gnn_raw", "ra_twin_raw"), ("ra_gnn_eng", "ra_mlp_same"),
      ("ra_gnn_eng", "ra_set_transformer"), ("ra_gnn_eng_physmp", "ra_gnn_eng"))
S2 = ((CD_GNN, "cd"), (CD_GNN, CD_TWIN), (CD_RANDOM, "cd"))
DESCRIPTIVE = ("cd_declared", "reactive")  # reported against CD, in no family
CLASSICAL = ("cd", "cd_declared", "locality", "batched", "selfpredict", "reactive")


def load(gate: str) -> Tuple[dict, dict]:
    """cells[(kind, seed, topo, win, rung)] = summary dict; failed[(kind, seed, topo, win, rung)] = failed.json dict."""
    cells, failed = {}, {}
    for f in glob.glob(os.path.join(gate, "*.summary.json")) + glob.glob(os.path.join(gate, "*.failed.json")):
        base = os.path.basename(f).rsplit(".", 2)[0]
        m = NAME.match(base)
        if not m:
            continue
        key = (m["kind"], int(m["seed"]), int(m["topo"]), m["win"], m["rung"])
        (cells if f.endswith(".summary.json") else failed)[key] = json.load(open(f))
    return cells, failed


def exact_wilcoxon(diffs: List[float]) -> Optional[float]:
    """Two-sided exact signed-rank p over the non-zero differences (mid-ranks; the null enumerates all sign patterns)."""
    d = [x for x in diffs if x != 0]
    n = len(d)
    if n == 0:
        return None
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    r2 = [int(round(2 * r)) for r in ranks]  # integers: DP over the sum of doubled ranks
    total = sum(r2)
    counts = defaultdict(int)
    counts[0] = 1
    for r in r2:
        nxt = defaultdict(int)
        for s, c in counts.items():
            nxt[s] += c
            nxt[s + r] += c
        counts = nxt
    w = sum(r for r, x in zip(r2, d) if x > 0)
    dev = abs(2 * w - total)
    tail = sum(c for s, c in counts.items() if abs(2 * s - total) >= dev)
    return min(1.0, tail / 2 ** n)


def holm(ps: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    """Holm step-down adjusted p over the family; a test with no p (not run) counts as p = 1."""
    items = sorted(ps.items(), key=lambda kv: 1.0 if kv[1] is None else kv[1])
    m, out, run = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * (1.0 if p is None else p))
        run = max(run, adj)
        out[k] = None if p is None else run
    return out


def paired(cells: dict, a: str, b: str, rung: str, topos: List[int], drop: set) -> Dict[int, dict]:
    """Per topology: median over (window, seed) of 100 (A - B) / B on mean latency. Reference seed: B's own seed when B is
    seeded, else its seed 0. A pair is used only when both cells finished."""
    out = {}
    for t in topos:
        vals = []
        for (kind, seed, topo, win, r), s in cells.items():
            if kind != a or topo != t or r != rung or (a, seed, t, win, r) in drop:
                continue
            ref = cells.get((b, seed, t, win, r)) or cells.get((b, 0, t, win, r))
            if ref is None:
                continue
            vals.append(100.0 * (float(s["averageElapsedTime"]) - float(ref["averageElapsedTime"])) / float(ref["averageElapsedTime"]))
        if vals:
            out[t] = {"median_pct": st.median(vals), "n": len(vals)}
    return out


def contrast(cells: dict, a: str, b: str, rung: str, topos: List[int]) -> dict:
    per = paired(cells, a, b, rung, topos, set())
    if not per:
        return {"a": a, "b": b, "rung": rung, "n_topologies": 0, "median_pct": None, "p": None, "wins": None}
    v = [x["median_pct"] for x in per.values()]
    return {"a": a, "b": b, "rung": rung, "n_topologies": len(v), "median_pct": st.median(v), "p": exact_wilcoxon(v),
            "wins": sum(1 for x in v if x < 0), "per_topology": {str(t): round(x["median_pct"], 3) for t, x in sorted(per.items())}}


def arm_table(cells: dict, failed: dict, expected: Dict[str, int]) -> dict:
    """Per arm and rung: cells finished/failed, median and worst effective share, p95, run end, mean latency, wall time."""
    out: Dict[str, dict] = {}
    kinds = sorted({k[0] for k in list(cells) + list(failed)})
    for kind in kinds:
        for rung in RUNGS:
            cs = [s for k, s in cells.items() if k[0] == kind and k[4] == rung]
            fs = [(k, f) for k, f in failed.items() if k[0] == kind and k[4] == rung]
            if not cs and not fs:
                continue
            get = lambda f: [x for x in (f(s) for s in cs) if x is not None]
            share = get(lambda s: s.get("effective_queue_share"))
            p95 = get(lambda s: (s.get("latency_percentiles") or {}).get("p95"))
            end = get(lambda s: (s.get("arrival_end") or {}).get("end_over_last_arrival"))
            rf = get(lambda s: 100.0 * (s.get("requestFailures") if isinstance(s.get("requestFailures"), (int, float)) else len(s.get("requestFailures") or [])) / s["num_tasks"])
            why: Dict[str, int] = defaultdict(int)
            for _, f in fs:
                why[str(f.get("why", "?")).split(":")[0].split(";")[0][:40]] += 1
            out[f"{kind}|{rung}"] = {
                "kind": kind, "rung": rung, "finished": len(cs), "failed": len(fs), "expected": expected.get(kind), "failed_why": dict(why),
                "median_effective_share": st.median(share) if share else None, "worst_effective_share": max(share) if share else None,
                "median_p95_s": st.median(p95) if p95 else None, "worst_p95_s": max(p95) if p95 else None,
                "median_end_over_last_arrival": st.median(end) if end else None, "worst_end_over_last_arrival": max(end) if end else None,
                "worst_request_failure_pct": max(rf) if rf else None,
                "median_latency_s": st.median(float(s["averageElapsedTime"]) for s in cs) if cs else None,
                "median_wallclock_s": st.median(float(s["wallclock_s"]) for s in cs) if cs else None}
    return out


def families(cells: dict, topos: List[int], best_arm: Optional[str]) -> dict:
    fam: Dict[str, dict] = {}
    prim = {}
    if best_arm:
        for r in RUNGS:
            prim[r] = contrast(cells, best_arm, "cd", r, topos)
        adj = holm({r: c["p"] for r, c in prim.items()})
        for r, c in prim.items():
            c["holm_p"] = adj[r]
    fam["primary"] = {"best_arm": best_arm, "tests": prim}
    for name, pairs in (("S1", S1), ("S2", S2)):
        tests = {f"{a} vs {b}|{r}": contrast(cells, a, b, r, topos) for a, b in pairs for r in RUNGS}
        adj = holm({k: c["p"] for k, c in tests.items()})
        for k, c in tests.items():
            c["holm_p"] = adj[k]
            if c["n_topologies"] == 0:
                c["note"] = "not run"
        fam[name] = {"holm_over": len(tests), "tests": tests}
    fam["descriptive"] = {f"{a}|{r}": contrast(cells, a, "cd", r, topos) for a in DESCRIPTIVE for r in RUNGS}
    fam["classical_vs_cd"] = {f"{a}|{r}": contrast(cells, a, "cd", r, topos) for a in ("locality", "batched", "selfpredict") for r in RUNGS}
    return fam


def read(gate: str, best_arm: Optional[str]) -> dict:
    cells, failed = load(gate)
    topos = sorted({k[2] for k in list(cells) + list(failed)})
    expected: Dict[str, int] = {}
    for k in list(cells) + list(failed):
        expected[k[0]] = expected.get(k[0], 0) + 1
    bad_topos = sorted({k[2] for k in failed})
    sens_cells = {k: v for k, v in cells.items() if k[2] not in bad_topos}
    sens_topos = [t for t in topos if t not in bad_topos]
    codes = sorted({str(s.get("code")) for s in cells.values()})
    return {"gate": gate, "topologies": topos, "n_summaries": len(cells), "n_failed": len(failed), "code_versions": codes,
            "arms": arm_table(cells, failed, {}), "main": families(cells, topos, best_arm),
            "sensitivity": {"excluded_topologies": bad_topos, "families": families(sens_cells, sens_topos, best_arm)}}


def print_report(r: dict) -> None:
    print(f"{r['n_summaries']} summaries, {r['n_failed']} failed, code {r['code_versions']}")
    print(f"{'arm':26s} {'rung':9s} {'ok/fail':8s} {'eff.share':>9s} {'p95 med/worst':>16s} {'end/arr worst':>13s} {'lat s':>7s}")
    for v in r["arms"].values():
        f = lambda x, n=3: "-" if x is None else f"{x:.{n}f}"
        print(f"{v['kind']:26s} {v['rung']:9s} {v['finished']}/{v['failed']:<6d} {f(v['median_effective_share']):>9s} "
              f"{f(v['median_p95_s'], 1):>7s}/{f(v['worst_p95_s'], 1):<8s} {f(v['worst_end_over_last_arrival']):>13s} {f(v['median_latency_s'], 2):>7s}")
    for name in ("primary", "S1", "S2", "descriptive", "classical_vs_cd"):
        fam = r["main"][name]
        tests = fam.get("tests", fam)
        print(f"-- {name}" + (f" (best arm {fam['best_arm']})" if name == "primary" else ""))
        for k, c in tests.items():
            f = lambda x: "-" if x is None else f"{x:.4f}"
            print(f"  {k or c['rung']:44s} median {('-' if c['median_pct'] is None else format(c['median_pct'], '+.2f') + ' %'):>10s}  n={c['n_topologies']:2d}  "
                  f"wins={c['wins']}  p={f(c['p'])}  holm={f(c.get('holm_p'))}  {c.get('note', '')}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate", required=True)
    ap.add_argument("--best-arm", default=None, help="the best learned arm, declared from validation topologies before any test read")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.best_arm is not None and a.best_arm not in LEARNED:
        raise SystemExit(f"FAIL LOUD: --best-arm {a.best_arm!r}; learned arms are {LEARNED}")
    r = read(a.gate, a.best_arm)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(r, fh, indent=1)
    print_report(r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
