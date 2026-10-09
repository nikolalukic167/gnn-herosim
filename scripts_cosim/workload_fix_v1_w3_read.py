#!/usr/bin/env python3
"""workload_fix_v1 W2+W3 read: the W2 reader's statistic on the W3 gate, plus three descriptives.

Runs `workload_fix_v1_read.compute` unchanged on <gate-dir> (per arm vs CD: paired % per topology, median over
windows, median over topologies, exact Wilcoxon, Holm over {selfpredict, locality, batched} x rungs; reactive is
context), then adds
  (a) the CD stage decomposition per rung (`workload_fix_v1_read.decompose`);
  (b) per arm and rung: exchange seconds per task (median over topologies of per-topology means) and the share of the
      arm's transfers, and of its exchange seconds, that sit on wired+wired server pairs (pooled over runs);
  (c) the change from the W2 gate to the W3 gate per arm and rung, paired on (topology, window, arm): per-cell
      100 * (W3 / W2 - 1) on averageElapsedTime and the difference in exchange seconds per task, medians over cells and
      the median over topologies of the per-topology median. Descriptive; no test.

  workload_fix_v1_w3_read.py --gate-dir <w3 gate> --w2-gate-dir <w2 gate> --selected <selected.json> --rungs lo,hi --out <json>
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import workload_fix_v1_read as R  # noqa: E402

WIRED = "wired+wired"


def exchange_by_arm(ok: dict, topos: Sequence[int], rungs: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for rung in rungs:
        out[rung] = {}
        for arm in R.ARMS:
            runs = [ok[(t, f"{w}{rung}", arm)] for t in topos for w in R.WINDOWS if (t, f"{w}{rung}", arm) in ok]
            if not runs or any(r.get("peerExchangeByAccessClass") is None for r in runs):
                out[rung][arm] = None
                continue
            per_topo = []
            for t in topos:
                rs = [ok[(t, f"{w}{rung}", arm)] for w in R.WINDOWS if (t, f"{w}{rung}", arm) in ok]
                if rs:
                    per_topo.append(st.mean(r["totalPeerExchangeTime"] / r["num_tasks"] for r in rs))
            tr = sec = tr_w = sec_w = 0.0
            for r in runs:
                for pair, v in r["peerExchangeByAccessClass"].items():
                    tr += v["transfers"]
                    sec += v["seconds"]
                    if pair == WIRED:
                        tr_w += v["transfers"]
                        sec_w += v["seconds"]
            out[rung][arm] = {"n_runs": len(runs), "exch_per_task": st.median(per_topo),
                              "transfers": int(tr), "transfers_wired_wired": int(tr_w),
                              "share_transfers_wired_wired": tr_w / tr, "share_seconds_wired_wired": sec_w / sec,
                              "seconds_per_transfer": sec / tr}
    return out


def w3_minus_w2(ok3: dict, ok2: dict, topos: Sequence[int], rungs: Sequence[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for rung in rungs:
        out[rung] = {}
        for arm in R.ARMS:
            pct: List[float] = []
            dex: List[float] = []
            per_topo_pct: List[float] = []
            for t in topos:
                tp = []
                for w in R.WINDOWS:
                    k = (t, f"{w}{rung}", arm)
                    if k in ok3 and k in ok2:
                        a, b = ok3[k], ok2[k]
                        tp.append(100 * (a["averageElapsedTime"] / b["averageElapsedTime"] - 1))
                        dex.append(a["totalPeerExchangeTime"] / a["num_tasks"] - b["totalPeerExchangeTime"] / b["num_tasks"])
                pct += tp
                if tp:
                    per_topo_pct.append(st.median(tp))
            out[rung][arm] = ({"n_pairs": len(pct), "median_pct_cells": st.median(pct),
                               "median_pct_topologies": st.median(per_topo_pct),
                               "n_topologies_slower": sum(x > 0 for x in per_topo_pct), "n_topologies": len(per_topo_pct),
                               "median_exch_per_task_diff_s": st.median(dex)} if pct else None)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate-dir", required=True)
    ap.add_argument("--w2-gate-dir", required=True)
    ap.add_argument("--selected", required=True)
    ap.add_argument("--rungs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    topos = json.load(open(a.selected))["topologies"]
    rungs = [r for r in a.rungs.split(",") if r]
    ok, bad = R.load(a.gate_dir)
    ok2, bad2 = R.load(a.w2_gate_dir)
    n_expected = len(topos) * len(rungs) * len(R.WINDOWS) * len(R.ARMS)
    if len(ok) != n_expected or bad or len(ok2) != n_expected or bad2:
        raise SystemExit(f"FAIL LOUD: expected {n_expected} summaries and no failures in each gate; "
                         f"W3 {len(ok)}/{len(bad)}, W2 {len(ok2)}/{len(bad2)}")
    result = R.compute(ok, bad, topos, rungs)
    result["_cd_decomposition"] = R.decompose(ok, topos, rungs)
    result["_exchange_by_arm"] = exchange_by_arm(ok, topos, rungs)
    result["_w3_minus_w2"] = w3_minus_w2(ok, ok2, topos, rungs)
    result["_inputs"] = {"w3_gate": a.gate_dir, "w2_gate": a.w2_gate_dir, "n_w3": len(ok), "n_w2": len(ok2)}
    Path(a.out).write_text(json.dumps(result, indent=1))

    for rung in rungs:
        print(f"== {rung}  (CD first: {result[rung]['_cd_first']})")
        for arm in R.ARMS:
            r = result[rung][arm]
            q = r["request_failures"]
            rf = f"{q['tasks']}/{q['runs_with_failure']}runs" if q["available"] else "n/a"
            vs = (f"vs CD {r['vs_cd']:+6.1f}% ({r['faster']}/{r['n_pc']} faster, p {r['p']:.2g}"
                  + (f", Holm {r['p_holm']:.2g}, {r['label']})" if "p_holm" in r else ", context)")) if "vs_cd" in r else ""
            print(f"  {arm:12s} n={r['n_topologies']:2d} fail={r['n_failed']:2d} lat {r['lat']:7.2f} q {r['qshare']:.2f} "
                  f"exch/task {r['exch_per_task']:.2f}s ({100 * r['exch_share']:.1f}% of latency) req-fail {rf} {vs}")
        d = result["_cd_decomposition"][rung]
        print("  CD stages: " + ", ".join(f"{k} {d[k]:.3f}" for k in ("latency",) + R.COMPONENTS + ("other",)))
        for arm in R.ARMS:
            e = result["_exchange_by_arm"][rung][arm]
            c = result["_w3_minus_w2"][rung][arm]
            print(f"  {arm:12s} exch/task {e['exch_per_task']:.2f}s  wired+wired: {100 * e['share_transfers_wired_wired']:.1f}% of transfers, "
                  f"{100 * e['share_seconds_wired_wired']:.1f}% of exchange s | W3-W2 latency {c['median_pct_cells']:+.1f}% cells, "
                  f"{c['median_pct_topologies']:+.1f}% topologies ({c['n_topologies_slower']}/{c['n_topologies']} slower), "
                  f"exch/task {c['median_exch_per_task_diff_s']:+.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
