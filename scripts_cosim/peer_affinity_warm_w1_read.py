#!/usr/bin/env python3
"""peer_affinity_warm_v1 -- W1 live-gate read (docs/lineages/peer_affinity_warm_v1.md, W1 + Amendment 1).

Inputs: `<arm>.summary.json` files written by peer_affinity_v1_stage3_live_gate.sbatch --
the warm arms (gnn_s<N>, mpoff_s<N>) in a capped and an uncapped results dir, the landed
T1b arms in their capped / uncapped dirs, and the Knative re-run check dir.

Statistic: total_rtt over every task of the trace; improvement of A over B in percent is
100 * (B - A) / B (positive = A has lower total latency). Paired by training seed, exact
two-sided Wilcoxon, alpha 0.05. Registered contrasts (primary configuration = capped):
  L1  warm gnn vs T1b gnn      WARM-HELPS  median >= +3 % and p < 0.05 | WARM-HURTS mirror | NO-EFFECT
  L2  warm gnn vs warm mpoff   GNN-NEEDED-LIVE median >= +1 %, p < 0.05, gnn ahead >= 11/16 | POINTWISE-BETTER mirror | TIE
  L3  warm gnn vs knative_network  BEATS-KNATIVE  gnn lower on >= 12/16 seeds and median >= +3 %
Headline WINNING-GNN only if L2 = GNN-NEEDED-LIVE and L3 = BEATS-KNATIVE on the primary configuration.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts_cosim.peer_affinity_stage3_read import _stats  # noqa: E402
from scripts_cosim.peer_affinity_t1_read import wilcoxon_exact  # noqa: E402

ALPHA = 0.05
BARS = {"l1_median_pct": 3.0, "l2_median_pct": 1.0, "l2_seeds": 11, "l3_seeds": 12, "l3_median_pct": 3.0}


def load_arms(results_dir: Path, seeds: int, arms=("gnn", "mpoff"), expect_tasks: Optional[int] = None) -> Dict[str, Dict[int, dict]]:
    out: Dict[str, Dict[int, dict]] = {}
    for arm in arms:
        out[arm] = {}
        for s in range(1, seeds + 1):
            p = results_dir / f"{arm}_s{s}.summary.json"
            if not p.exists():
                raise RuntimeError(f"missing summary {p} -- the gate is not complete, refusing to read")
            d = _stats(p)
            if expect_tasks is not None and d["num_tasks"] != expect_tasks:
                raise RuntimeError(f"{p}: num_tasks {d['num_tasks']} != {expect_tasks} -- an incomplete run is not averaged")
            out[arm][s] = d
    return out


def improvements(a: Dict[int, dict], b: Dict[int, dict], seeds: int) -> List[float]:
    """100 * (b - a) / b per seed: positive = a lower total_rtt than b."""
    return [100.0 * (b[s]["total_rtt"] - a[s]["total_rtt"]) / b[s]["total_rtt"] for s in range(1, seeds + 1)]


def paired(a, b, seeds):
    imp = improvements(a, b, seeds)
    return {"median_improvement_pct": st.median(imp), "mean_improvement_pct": st.mean(imp),
            "a_better_seeds": sum(1 for x in imp if x > 0), "n": len(imp),
            "p_exact_wilcoxon": wilcoxon_exact([b[s]["total_rtt"] - a[s]["total_rtt"] for s in range(1, seeds + 1)]),
            "per_seed_pct": imp,
            "a_median_total_rtt": st.median(x["total_rtt"] for x in a.values()),
            "b_median_total_rtt": st.median(x["total_rtt"] for x in b.values())}


def read_l1(warm, t1b, seeds):
    r = paired(warm, t1b, seeds); m, p = r["median_improvement_pct"], r["p_exact_wilcoxon"]
    if p is not None and p < ALPHA and m >= BARS["l1_median_pct"]:
        r["reading"] = "WARM-HELPS"
    elif p is not None and p < ALPHA and m <= -BARS["l1_median_pct"]:
        r["reading"] = "WARM-HURTS"
    else:
        r["reading"] = "NO-EFFECT"
    return r


def read_l2(gnn, mpoff, seeds):
    r = paired(gnn, mpoff, seeds); m, p, k = r["median_improvement_pct"], r["p_exact_wilcoxon"], r["a_better_seeds"]
    if p is not None and p < ALPHA and m >= BARS["l2_median_pct"] and k >= BARS["l2_seeds"]:
        r["reading"] = "GNN-NEEDED-LIVE"
    elif p is not None and p < ALPHA and m <= -BARS["l2_median_pct"] and (seeds - k) >= BARS["l2_seeds"]:
        r["reading"] = "POINTWISE-BETTER"
    else:
        r["reading"] = "TIE"
    return r


def read_l3(gnn, knative_total: float, seeds):
    imp = [100.0 * (knative_total - gnn[s]["total_rtt"]) / knative_total for s in range(1, seeds + 1)]
    k = sum(1 for x in imp if x > 0); m = st.median(imp)
    return {"median_improvement_pct": m, "gnn_better_seeds": k, "n": seeds, "per_seed_pct": imp,
            "knative_total_rtt": knative_total,
            "p_sign_exact": wilcoxon_exact([knative_total - gnn[s]["total_rtt"] for s in range(1, seeds + 1)]),
            "reading": "BEATS-KNATIVE" if (k >= BARS["l3_seeds"] and m >= BARS["l3_median_pct"]) else "NOT-BEATS-KNATIVE"}


def side(arms, seeds, key):
    return {arm: st.median(arms[arm][s][key] for s in range(1, seeds + 1)) for arm in arms}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--warm-capped", type=Path, required=True)
    ap.add_argument("--warm-uncapped", type=Path, required=True)
    ap.add_argument("--t1b-capped", type=Path, required=True)
    ap.add_argument("--t1b-uncapped", type=Path, required=True)
    ap.add_argument("--knative-check", type=Path, required=True, help="dir with the re-run knative_network / knative_network_batch summaries")
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--expect-tasks", type=int, default=450729)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    kn_landed = _stats(args.t1b_capped / "knative_network.summary.json")
    kn_check = _stats(args.knative_check / "knative_network.summary.json")
    knb_landed = _stats(args.t1b_capped / "knative_network_batch.summary.json")
    knb_check = _stats(args.knative_check / "knative_network_batch.summary.json")
    rel = abs(kn_check["total_rtt"] - kn_landed["total_rtt"]) / kn_landed["total_rtt"]
    relb = abs(knb_check["total_rtt"] - knb_landed["total_rtt"]) / knb_landed["total_rtt"]
    knative_reproduced = rel < 1e-9 and relb < 1e-9
    knative_total = kn_check["total_rtt"]  # the re-run under the current code is the comparator either way
    for d in (kn_check, knb_check):
        if d["num_tasks"] != args.expect_tasks:
            raise RuntimeError(f"knative check run has num_tasks {d['num_tasks']} != {args.expect_tasks}")

    out = {"lineage": "peer_affinity_warm_v1", "stage": "W1 live gate", "bars": BARS, "alpha": ALPHA, "seeds": args.seeds,
           "knative": {"landed_total_rtt": kn_landed["total_rtt"], "rerun_total_rtt": kn_check["total_rtt"],
                       "landed_batch_total_rtt": knb_landed["total_rtt"], "rerun_batch_total_rtt": knb_check["total_rtt"],
                       "rel_diff": rel, "rel_diff_batch": relb, "reproduced": knative_reproduced}}
    for cfg, warm_dir, t1b_dir in (("capped", args.warm_capped, args.t1b_capped), ("uncapped", args.warm_uncapped, args.t1b_uncapped)):
        warm = load_arms(warm_dir, args.seeds, expect_tasks=args.expect_tasks)
        t1b = load_arms(t1b_dir, args.seeds, expect_tasks=args.expect_tasks)
        block = {
            "L1_warm_gnn_vs_t1b_gnn": read_l1(warm["gnn"], t1b["gnn"], args.seeds),
            "L1b_warm_mpoff_vs_t1b_mpoff": read_l1(warm["mpoff"], t1b["mpoff"], args.seeds),
            "L2_warm_gnn_vs_warm_mpoff": read_l2(warm["gnn"], warm["mpoff"], args.seeds),
            "L2_t1b_gnn_vs_t1b_mpoff_reference": read_l2(t1b["gnn"], t1b["mpoff"], args.seeds),
            "L3_warm_gnn_vs_knative": read_l3(warm["gnn"], knative_total, args.seeds),
            "L3b_warm_mpoff_vs_knative": read_l3(warm["mpoff"], knative_total, args.seeds),
            "L3_t1b_gnn_vs_knative_reference": read_l3(t1b["gnn"], knative_total, args.seeds),
            "side": {"median_end_time": {"warm": side(warm, args.seeds, "end_time"), "t1b": side(t1b, args.seeds, "end_time"),
                                         "knative": kn_check["end_time"]},
                     "median_peer_exchange": {"warm": side(warm, args.seeds, "peer_exchange"), "t1b": side(t1b, args.seeds, "peer_exchange"),
                                              "knative": kn_check["peer_exchange"]},
                     "counters_warm_gnn_s1": warm["gnn"][1]["counters"]},
        }
        block["headline"] = ("WINNING-GNN" if (block["L2_warm_gnn_vs_warm_mpoff"]["reading"] == "GNN-NEEDED-LIVE"
                                               and block["L3_warm_gnn_vs_knative"]["reading"] == "BEATS-KNATIVE") else "NO-WINNING-GNN")
        out[cfg] = block
    out["headline_primary_capped"] = out["capped"]["headline"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    for cfg in ("capped", "uncapped"):
        b = out[cfg]
        print(f"[{cfg}] L1 {b['L1_warm_gnn_vs_t1b_gnn']['reading']} ({b['L1_warm_gnn_vs_t1b_gnn']['median_improvement_pct']:+.2f} %, "
              f"p={b['L1_warm_gnn_vs_t1b_gnn']['p_exact_wilcoxon']}, {b['L1_warm_gnn_vs_t1b_gnn']['a_better_seeds']}/{args.seeds}) | "
              f"L2 {b['L2_warm_gnn_vs_warm_mpoff']['reading']} ({b['L2_warm_gnn_vs_warm_mpoff']['median_improvement_pct']:+.2f} %, "
              f"p={b['L2_warm_gnn_vs_warm_mpoff']['p_exact_wilcoxon']}, {b['L2_warm_gnn_vs_warm_mpoff']['a_better_seeds']}/{args.seeds}) | "
              f"L3 {b['L3_warm_gnn_vs_knative']['reading']} ({b['L3_warm_gnn_vs_knative']['median_improvement_pct']:+.2f} %, "
              f"{b['L3_warm_gnn_vs_knative']['gnn_better_seeds']}/{args.seeds}) | headline {b['headline']}")
    print(f"[knative] reproduced={knative_reproduced} rel_diff={rel:.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
