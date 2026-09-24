#!/usr/bin/env python3
"""hidden_exec_s0_v1 -- read the headroom screen. See docs/lineages/hidden_exec_s0_v1.md.

Pairs `oracle` against `table` per (rung, env) on total_rtt under hidden_node_v1 and applies the
registered bar. Optionally discloses the physics cost against the selfpredict_bar_v1 gate's
table_v0 summaries of the same arm.

  python3 scripts_cosim/hidden_exec_s0_v1_read.py --gate-dir DIR [--table-v0-dir DIR] [--out read.json]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
from typing import Dict, Tuple

from scipy.stats import wilcoxon

BAR = 0.05
ALPHA = 0.05
N_ENVS = 16


def _load(gate_dir: str) -> Dict[Tuple[str, int, str, str], dict]:
    rows = {}
    for path in glob.glob(os.path.join(gate_dir, "*.summary.json")):
        s = json.load(open(path))
        rows[(s["rung"], int(s["topology"]), s["window"], s["arm"])] = s
    codes = {(s["code"].get("commit"), s["code"].get("dirty"), s["code"].get("diff_sha256"))
             for s in rows.values()}
    if len(codes) != 1:
        raise SystemExit(f"FAIL LOUD: summaries span {len(codes)} code states: {codes}")
    if next(iter(codes))[1]:
        raise SystemExit(f"FAIL LOUD: gate ran on a dirty tree: {codes}")
    return rows


def _paired(rows, rung: str, a: str, b: str):
    envs = sorted({(t, w) for (r, t, w, arm) in rows if r == rung and arm == a})
    deltas = []
    for t, w in envs:
        ra, rb = rows.get((rung, t, w, a)), rows.get((rung, t, w, b))
        if ra is None or rb is None:
            raise SystemExit(f"FAIL LOUD: {rung} {t} {w} is missing {a if ra is None else b}")
        deltas.append((float(ra["total_rtt"]) - float(rb["total_rtt"])) / float(rb["total_rtt"]))
    return deltas


def _verdict(deltas) -> Tuple[str, float, float, int]:
    if len(deltas) != N_ENVS:
        raise SystemExit(f"FAIL LOUD: {len(deltas)} paired environments, the design is {N_ENVS}")
    med = statistics.median(deltas)
    p = float(wilcoxon(deltas).pvalue) if any(deltas) else 1.0
    faster = sum(d < 0 for d in deltas)
    if med <= -BAR and p < ALPHA:
        return "HEADROOM", med, p, faster
    if med >= BAR and p < ALPHA:
        return "ORACLE-SLOWER", med, p, faster
    return "NO-HEADROOM", med, p, faster


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-dir", required=True)
    ap.add_argument("--table-v0-dir", default=None,
                    help="selfpredict_bar_v1 gate summaries (disclosed physics cost only)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = _load(args.gate_dir)

    out = {"bar": {"median": -BAR, "p": ALPHA, "n": N_ENVS}, "rungs": {}}
    print("=== hidden_exec_s0_v1 HEADROOM SCREEN (oracle vs table, hidden_node_v1) ===")
    verdicts = []
    for rung in ("C40", "C80"):
        deltas = _paired(rows, rung, "oracle", "table")
        v, med, p, faster = _verdict(deltas)
        verdicts.append(v)
        exec_t = statistics.mean(float(rows[k]["averageExecutionTime"]) for k in rows
                                 if k[0] == rung and k[3] == "table")
        elapsed_t = statistics.mean(float(rows[k]["averageElapsedTime"]) for k in rows
                                    if k[0] == rung and k[3] == "table")
        rec = {"verdict": v, "median_delta": med, "wilcoxon_p": p, "oracle_faster": faster,
               "n": len(deltas), "deltas": deltas, "table_mean_exec_s": exec_t,
               "table_mean_elapsed_s": elapsed_t}
        print(f"{rung}: oracle vs table median {med * 100:+.2f} % (p = {p:.4f}, oracle faster "
              f"{faster}/{len(deltas)}) -> {v}; table exec {exec_t:.4f} s of {elapsed_t:.3f} s/task")
        if args.table_v0_dir:
            cost = []
            for (r, t, w, arm), s in rows.items():
                if r != rung or arm != "table":
                    continue
                path = os.path.join(args.table_v0_dir,
                                    f"cc{rung[1:]}s{t}__{w}__peer_greedy_selfpredict_network_s0.summary.json")
                base = json.load(open(path))
                cost.append((float(s["total_rtt"]) - float(base["total_rtt"])) / float(base["total_rtt"]))
            rec["physics_cost_median"] = statistics.median(cost)
            print(f"      disclosed: hidden physics costs the rule {statistics.median(cost) * 100:+.2f} % "
                  f"(median, vs table_v0 in selfpredict_bar_v1)")
        out["rungs"][rung] = rec
    out["verdict"] = "GO-P0B" if all(v == "HEADROOM" for v in verdicts) else "NO-HEADROOM"
    print(f"VERDICT: {out['verdict']}")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
