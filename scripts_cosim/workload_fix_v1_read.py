#!/usr/bin/env python3
"""workload_fix_v1 read (docs/lineages/workload_fix_v1.md): CD against each other classical arm, one stage.

Per rung and arm: latency, queue share, exchange per task and exchange share of latency (median over topologies of
per-topology means over windows). Against CD: the paired % per topology (median over windows of
100 * (arm / CD - 1) on averageElapsedTime), the median over topologies, an exact two-sided Wilcoxon over
topologies (as kpa_scaleout_v1_read.py), Holm within the stage.

Family = {selfpredict, locality, batched} x rungs. Knative (`reactive`) is context only: reported with its paired %
and unadjusted p, outside the Holm family. A run that failed drops out of that arm's paired tests only; the
sensitivity block re-reads with every topology that has a remaining failure excluded for every arm.

Exchange share by access-class pair (for the W3 stage): the summed exchange seconds of each unordered class pair of
the two servers (`wired+wired`, `wifi+wired`, `cellular+wifi`, ...) over the summed latency seconds of the rung's
tasks, pooled over topologies and windows. Present only when the summaries carry `peerExchangeByAccessClass`
(topologies with `network.backbone.access_classes`); on uniform links it is reported as unavailable, not zero.

  workload_fix_v1_read.py --gate-dir <dir> --selected <selected.json> --rungs lo,hi [--out read.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
from typing import Any, Dict, List, Sequence, Tuple

from scipy.stats import wilcoxon

ARMS = ("reactive", "selfpredict", "locality", "batched", "cd")
FAMILY = ("selfpredict", "locality", "batched")
WINDOWS = ("g0", "g1", "g2", "g3")


def holm(ps: Sequence[float]) -> List[float]:
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def load(gate_dir: str) -> Tuple[Dict[Tuple[int, str, str], dict], Dict[Tuple[int, str, str], dict]]:
    """(summaries, failures) keyed (topology, window label, arm kind)."""
    def key(arm: str) -> Tuple[int, str, str]:
        cell, window, kind = arm.split("__")
        return int(cell[len("cc40s"):]), window, kind.rsplit("_s", 1)[0]

    ok = {key(json.load(open(f))["arm"]): json.load(open(f)) for f in glob.glob(os.path.join(gate_dir, "*.summary.json"))}
    bad = {key(json.load(open(f))["arm"]): json.load(open(f)) for f in glob.glob(os.path.join(gate_dir, "*.failed.json"))}
    return ok, bad


COMPONENTS = ("batching_wait", "queue", "exchange", "rendezvous", "cold_start")


def decompose(ok: dict, topos: Sequence[int], rungs: Sequence[str], arm: str = "cd") -> Dict[str, Any]:
    """Mean seconds per task of one arm's latency by stage, per rung (median over topologies of per-topology means).

    batching_wait = averageWaitTime (dispatch -> scheduled: peer-group assembly and the decision), queue =
    averageQueueTime (scheduled -> on the replica's queue), exchange / rendezvous = the peer totals per task,
    cold_start = averageColdStartTime; other = latency minus those. Needs the stage times that the wf1 summaries
    carry (averageColdStartTime), so a summary without it is reported as missing, not zero."""
    out: Dict[str, Any] = {}
    for rung in rungs:
        per_topo: List[Dict[str, float]] = []
        for t in topos:
            rs = [ok[(t, f"{w}{rung}", arm)] for w in WINDOWS if (t, f"{w}{rung}", arm) in ok]
            if not rs:
                continue
            if any(r.get("averageColdStartTime") is None for r in rs):
                return {"available": False, "reason": "summaries lack averageColdStartTime (run before the wf1 stage fields)"}
            comp = {
                "latency": st.mean(r["averageElapsedTime"] for r in rs),
                "batching_wait": st.mean(r["averageWaitTime"] for r in rs),
                "queue": st.mean(r["averageQueueTime"] for r in rs),
                "exchange": st.mean(r["totalPeerExchangeTime"] / r["num_tasks"] for r in rs),
                "rendezvous": st.mean(r["totalPeerRendezvousWait"] / r["num_tasks"] for r in rs),
                "cold_start": st.mean(r["averageColdStartTime"] for r in rs),
            }
            comp["other"] = comp["latency"] - sum(comp[k] for k in COMPONENTS)
            per_topo.append(comp)
        out[rung] = {"available": True, "n_topologies": len(per_topo),
                     **{k: st.median(c[k] for c in per_topo) for k in ("latency",) + COMPONENTS + ("other",)}} if per_topo else None
    return out


def tune(ok: dict, bad: dict, topos: Sequence[int], rungs: Sequence[str], windows: Sequence[int],
         workload_windows: Sequence[str] = WINDOWS) -> Dict[str, Any]:
    """Amendment WB2: CD at each fixed batching window. Per (rung, window) the median CD latency over the
    (topology, workload window) cells, a hung or missing run counting as infinite; the geometric mean over the rungs
    of those medians; the window that minimises it. Cells are keyed by the tag ``<rung>b<seconds>``."""
    table: Dict[int, Dict[str, Any]] = {}
    for w in windows:
        row: Dict[str, Any] = {}
        for rung in rungs:
            tag = f"{rung}b{w}"
            lat, hung, missing = [], 0, 0
            for t in topos:
                for g in workload_windows:
                    key = (t, f"{g}{tag}", "cd")
                    if key in ok:
                        lat.append(ok[key]["averageElapsedTime"])
                    else:
                        lat.append(float("inf"))
                        hung += 1
                        missing += key not in bad
            finite = [x for x in lat if x != float("inf")]
            row[rung] = {"median_latency": st.median(lat), "n_cells": len(lat), "n_hung": hung,
                         "n_missing_not_failed": missing, "mean_latency_finite": st.mean(finite) if finite else None}
        gm = 1.0
        for rung in rungs:
            gm *= row[rung]["median_latency"]
        row["geomean"] = gm ** (1.0 / len(rungs)) if gm != float("inf") else float("inf")
        table[w] = row
    best = min(windows, key=lambda w: (table[w]["geomean"], w))
    return {"rungs": list(rungs), "table": table, "best_window_s": best,
            "tie": sorted(w for w in windows if table[w]["geomean"] == table[best]["geomean"]) if len(
                [w for w in windows if table[w]["geomean"] == table[best]["geomean"]]) > 1 else None}


def exchange_share_by_class(rows: Sequence[dict]) -> Dict[str, Any]:
    with_classes = [r for r in rows if r.get("peerExchangeByAccessClass") is not None]
    if not with_classes:
        return {"available": False, "reason": "no access classes on these topologies (uniform links)"}
    latency_s = sum(r["num_tasks"] * r["averageElapsedTime"] for r in with_classes)
    pairs: Dict[str, Dict[str, float]] = {}
    for r in with_classes:
        for pair, v in r["peerExchangeByAccessClass"].items():
            slot = pairs.setdefault(pair, {"seconds": 0.0, "transfers": 0})
            slot["seconds"] += v["seconds"]
            slot["transfers"] += v["transfers"]
    return {"available": True, "n_runs": len(with_classes), "latency_seconds": latency_s,
            "by_pair": {p: {**v, "share_of_latency": v["seconds"] / latency_s} for p, v in sorted(pairs.items())}}


def compute(ok: dict, bad: dict, topos: Sequence[int], rungs: Sequence[str], excluded: Sequence[int] = ()) -> dict:
    topos = [t for t in topos if t not in set(excluded)]
    out: Dict[str, Any] = {}
    tests: List[Tuple[str, str]] = []
    for rung in rungs:
        ws = [f"{w}{rung}" for w in WINDOWS]
        rows: Dict[str, Any] = {}
        for arm in ARMS:
            lat, qs, ex, exs, cold, runs = [], [], [], [], [], []
            for t in topos:
                rs = [ok[(t, w, arm)] for w in ws if (t, w, arm) in ok]
                if not rs:
                    continue
                runs += rs
                lat.append(st.mean(r["averageElapsedTime"] for r in rs))
                qs.append(st.mean(r["queue_share"] for r in rs))
                ex.append(st.mean(r["totalPeerExchangeTime"] / r["num_tasks"] for r in rs))
                exs.append(st.mean(r["totalPeerExchangeTime"] / r["num_tasks"] / r["averageElapsedTime"] for r in rs))
                cold.append(st.mean(r["cold_start_pct"] for r in rs if r.get("cold_start_pct") is not None) if any(
                    r.get("cold_start_pct") is not None for r in rs) else None)
            row: Dict[str, Any] = {
                "n_topologies": len(lat), "n_runs": len(runs),
                "n_failed": sum(1 for t in topos for w in ws if (t, w, arm) in bad),
                "lat": st.median(lat) if lat else None, "qshare": st.median(qs) if qs else None,
                "exch_per_task": st.median(ex) if ex else None, "exch_share": st.median(exs) if exs else None,
                "cold_start_pct": st.median([c for c in cold if c is not None]) if any(c is not None for c in cold) else None,
                "exchange_by_access_class": exchange_share_by_class(runs),
            }
            if arm != "cd":
                pc = []
                for t in topos:
                    p = [100 * (ok[(t, w, arm)]["averageElapsedTime"] / ok[(t, w, "cd")]["averageElapsedTime"] - 1)
                         for w in ws if (t, w, arm) in ok and (t, w, "cd") in ok]
                    if p:
                        pc.append(st.median(p))
                if pc:
                    row.update(vs_cd=st.median(pc), faster=sum(x < 0 for x in pc), n_pc=len(pc),
                               p=float(wilcoxon(pc).pvalue) if any(pc) else 1.0)
                    if arm in FAMILY:
                        tests.append((rung, arm))
            rows[arm] = row
        lats = {a: r["lat"] for a, r in rows.items() if r["lat"] is not None}
        rows["_cd_first"] = bool(lats) and min(lats, key=lats.get) == "cd"
        out[rung] = rows
    adj = holm([out[r][a]["p"] for r, a in tests])
    for (r, a), pa in zip(tests, adj):
        row = out[r][a]
        row["p_holm"] = pa
        row["label"] = ("CONFIRMED" if row["vs_cd"] <= -5 and pa < 0.05 else
                        "CD-FASTER" if row["vs_cd"] > 0 and pa < 0.05 else
                        "DIRECTION" if pa < 0.05 else "NOT-SEPARATED")
    out["_meta"] = {"family": list(FAMILY), "family_size": len(tests), "context_only": ["reactive"],
                    "topologies": list(topos), "excluded": sorted(excluded)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate-dir", required=True)
    ap.add_argument("--tune", nargs="*", type=int, metavar="SECONDS",
                    help="amendment WB2 mode: --selected lists the calibration topologies, --rungs the rung names, "
                         "--gate-dir holds cells tagged <rung>b<seconds>; the listed windows are compared")
    ap.add_argument("--override-dir", action="append", default=[],
                    help="amendment WB: summaries here replace --gate-dir's for the arms they contain (repeatable)")
    ap.add_argument("--decompose", nargs=2, metavar=("LADDER_DIR", "FIXED_DIR"),
                    help="CD latency by stage under the ladder window (LADDER_DIR) and the fixed window (FIXED_DIR)")
    ap.add_argument("--selected", required=True)
    ap.add_argument("--rungs", required=True, help="comma-separated rung tags, as in WF1_RUNGS")
    ap.add_argument("--out")
    a = ap.parse_args()
    topos = json.load(open(a.selected))["topologies"]
    rungs = [r for r in a.rungs.split(",") if r]
    ok, bad = load(a.gate_dir)
    if a.tune is not None:
        res = tune(ok, bad, topos, rungs, a.tune or [1, 2, 4, 8, 16])
        for w, row in res["table"].items():
            print(f"window {w:>2d} s: " + "  ".join(f"{r} median {row[r]['median_latency']:.4f} (hung {row[r]['n_hung']}/{row[r]['n_cells']})"
                                                  for r in rungs) + f"  geomean {row['geomean']:.4f}")
        print(f"minimum: {res['best_window_s']} s" + (f" (tie {res['tie']})" if res["tie"] else ""))
        if a.out:
            with open(a.out, "w") as fh:
                json.dump(res, fh, indent=1)
        return 0
    for d in a.override_dir:
        o_ok, o_bad = load(d)
        arms = {k[2] for k in o_ok} | {k[2] for k in o_bad}
        ok = {k: v for k, v in ok.items() if k[2] not in arms}
        bad = {k: v for k, v in bad.items() if k[2] not in arms}
        ok.update(o_ok)
        bad.update(o_bad)
    result = compute(ok, bad, topos, rungs)
    failing = sorted({t for (t, w, k) in bad if t in topos})
    result["_sensitivity_excluding_failed_topologies"] = compute(ok, bad, topos, rungs, failing) if failing else None
    if a.decompose:
        d_ladder, d_fixed = load(a.decompose[0])[0], load(a.decompose[1])[0]
        complete = [t for t in topos if t not in failing]
        result["_cd_decomposition"] = {"ladder_window": decompose(d_ladder, topos, rungs),
                                       "fixed_window": decompose(d_fixed, topos, rungs),
                                       "ladder_window_excluding_failed_topologies": decompose(d_ladder, complete, rungs),
                                       "fixed_window_excluding_failed_topologies": decompose(d_fixed, complete, rungs)}
        for which, per in result["_cd_decomposition"].items():
            for rung, d in per.items():
                print(f"-- CD latency, {which}, {rung}: " + (", ".join(f"{k} {d[k]:.3f}" for k in ("latency",) + COMPONENTS + ("other",))
                                                           if d and d.get("available") else str(d)))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(result, fh, indent=1)
    for rung in rungs:
        print(f"== {rung}  (CD first: {result[rung]['_cd_first']})")
        for arm in ARMS:
            r = result[rung][arm]
            vs = (f"vs CD {r['vs_cd']:+6.1f}% ({r['faster']}/{r['n_pc']} faster, p {r['p']:.2g}"
                  + (f", Holm {r['p_holm']:.2g}, {r['label']})" if "p_holm" in r else ", context)")) if "vs_cd" in r else ""
            print(f"  {arm:12s} n={r['n_topologies']:2d} fail={r['n_failed']:2d} lat {r['lat'] or float('nan'):7.2f} "
                  f"q {r['qshare'] or float('nan'):.2f} exch/task {r['exch_per_task'] or float('nan'):.2f}s "
                  f"({100 * (r['exch_share'] or 0):.1f}% of latency) {vs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
