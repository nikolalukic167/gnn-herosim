#!/usr/bin/env python3
"""Direct additivity test for a route_b separable control — no decoder, cap or surrogate.

WHY THIS EXISTS. S0, the screen's separability gate, asks whether an additive min-marginal
surrogate recovers the CONSTRAINED optimum. That is an indirect test: it reads additivity
through a decoder and a capacity cap, so a failure has at least three candidate causes
(non-additive cost, a decode bug, or the cap path) and the gate cannot tell you which. On
the amended H2 that ambiguity cost a session — see
docs/lineages/route_b_env_pivot_v1.md, "the amended H2 is generated, and it is VOID".

This measures additivity DIRECTLY. The co-sim sweep is exhaustive, so
`placements/placements.jsonl` holds every plan with its true rtt. If

    cost(plan) = sum_t c_t(p_t)      exactly

then regressing rtt on one-hot (task, platform) indicators over the FULL sweep fits
perfectly: R^2 = 1, residual 0. Nothing else is in the path — no decoder, no cap, no
surrogate, no tie-break. A nonzero residual is a statement about the physics.

WHAT IT IMPORTS FROM THE SCORER: nothing. Same discipline as
verify_route_b_scorer_agreement.py — this must be able to contradict the scorer.

READING RULE. Report the MEDIAN per-dataset R^2 and residual, split on the
replica_configs arm (`n_rows`, the unconstrained sweep size). Route_b's whole defect
history is statistics that held on one arm and broke on its neighbour, so a pooled number
without its arm split is not a reading — see the route-b-preflight skill, step 3. The
minimum R^2 is reported too and is expected to be low on a handful of datasets even in a
genuinely additive corpus; it is the median that carries the verdict.

CALIBRATION (measured 2026-08-28, whole corpora, both arms):

    control corpus     R^2 median (per arm)   residual, median % of mean rtt   S0
    H0 ctrl            0.999991 / 0.999757    0.159% / 0.721%                  PASS
    H1 ctrl            0.999975 / 0.999856    0.193% / 0.999%                  PASS
    H2 ctrl            0.7826   / 0.8161      12.822% / 13.814%                FAIL

The controls that pass S0 fit to a fraction of a percent; the one that fails does not fit
at all. That contrast is what makes this a usable instrument rather than a number without
a scale.

--co-residency additionally breaks the residual down by the maximum number of tasks
sharing one node, which is what separates "residual physics coupling" from "something
else". --pairwise fits two nested models on top of the additive one (a same-node pair
indicator, then a same-(node, platform) pair indicator) to say whether any coupling found
is pairwise or higher-order. NOTE the same-slot term is vacuous on any corpus whose sweep
requires globally distinct replicas: no two tasks can share a slot, so the column is
always empty. It is kept because a future grid may relax uniqueness, and a silently
absent term would then be indistinguishable from a zero one.

Usage:
  measure_route_b_additivity.py --corpus simulation_data/gnn_datasets_route_b_pivot_h2_ctrl
  measure_route_b_additivity.py --corpus <c> --co-residency --pairwise --out report.json
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

Plan = Dict[int, Tuple[int, int]]


def load_sweep(ds_dir: Path) -> Tuple[List[Plan], np.ndarray]:
    """Every plan in the dataset's exhaustive sweep, with its true rtt."""
    jsonl = ds_dir / "placements" / "placements.jsonl"
    if not jsonl.exists():
        raise RuntimeError(
            f"{ds_dir}: placements/placements.jsonl missing — additivity is measured over "
            "the FULL sweep and cannot be computed from best.json alone (see "
            "docs/notes/placements_jsonl_required.md)"
        )
    plans: List[Plan] = []
    rtts: List[float] = []
    with open(jsonl) as fh:
        for line in fh:
            rec = json.loads(line)
            plans.append({int(k): tuple(v) for k, v in rec["placement_plan"].items()})
            rtts.append(rec["rtt"])
    if not plans:
        raise RuntimeError(f"{ds_dir}: sweep is empty")
    return plans, np.asarray(rtts, dtype=np.float64)


def _design(plans: List[Plan], mode: str) -> np.ndarray:
    """One-hot design. mode: 'additive' | 'pair_node' | 'pair_slot' (each nests the prior).

    Per-task blocks each sum to 1, so the design is rank-deficient against the intercept;
    lstsq's minimum-norm solution is used and R^2 is unaffected by which representative it
    picks.
    """
    cols: Dict[tuple, int] = {}
    rows: List[set] = []
    for plan in plans:
        row = {cols.setdefault(("main", t, s), len(cols)) for t, s in plan.items()}
        if mode in ("pair_node", "pair_slot"):
            for a, b in itertools.combinations(sorted(plan), 2):
                if plan[a][0] == plan[b][0]:
                    row.add(cols.setdefault(("pairnode", a, b), len(cols)))
        if mode == "pair_slot":
            for a, b in itertools.combinations(sorted(plan), 2):
                if plan[a] == plan[b]:
                    row.add(cols.setdefault(("pairslot", a, b), len(cols)))
        rows.append(row)

    X = np.zeros((len(plans), len(cols) + 1), dtype=np.float64)
    X[:, -1] = 1.0
    for i, row in enumerate(rows):
        for j in row:
            X[i, j] = 1.0
    return X


def fit(plans: List[Plan], rtts: np.ndarray, mode: str = "additive"):
    """Returns (r2, residual vector, residual RMS as % of mean rtt, n_params)."""
    X = _design(plans, mode)
    beta, *_ = np.linalg.lstsq(X, rtts, rcond=None)
    resid = rtts - X @ beta
    ss_tot = float(((rtts - rtts.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else float("nan")
    rms_pct = 100.0 * float(np.sqrt((resid ** 2).mean())) / rtts.mean()
    return r2, resid, rms_pct, X.shape[1]


def max_coresidency(plan: Plan) -> int:
    return max(collections.Counter(slot[0] for slot in plan.values()).values())


def measure_corpus(corpus: Path, want_cores: bool, want_pairs: bool) -> dict:
    per_arm: Dict[int, List[dict]] = collections.defaultdict(list)
    cores_pool: Dict[int, List[float]] = collections.defaultdict(list)

    for ds_dir in sorted(corpus.glob("ds_*")):
        if not (ds_dir / "placements" / "placements.jsonl").exists():
            continue
        plans, rtts = load_sweep(ds_dir)
        r2, resid, rms_pct, _ = fit(plans, rtts)
        entry = {"dataset": ds_dir.name, "n_rows": len(plans),
                 "r2": r2, "resid_pct_of_mean_rtt": rms_pct}
        if want_pairs:
            r2_pn, _, rms_pn, _ = fit(plans, rtts, "pair_node")
            r2_ps, _, rms_ps, _ = fit(plans, rtts, "pair_slot")
            entry["pair_node"] = {"r2": r2_pn, "resid_pct": rms_pn}
            entry["pair_slot"] = {"r2": r2_ps, "resid_pct": rms_ps}
        if want_cores:
            for plan, r in zip(plans, resid):
                cores_pool[max_coresidency(plan)].append(100.0 * r / rtts.mean())
        per_arm[len(plans)].append(entry)

    if not per_arm:
        raise RuntimeError(f"{corpus}: no datasets with a sweep — nothing to measure")

    out = {"corpus": str(corpus), "by_arm": {}, "n_datasets": sum(map(len, per_arm.values()))}
    for arm, entries in sorted(per_arm.items()):
        r2s = np.array([e["r2"] for e in entries])
        rms = np.array([e["resid_pct_of_mean_rtt"] for e in entries])
        block = {
            "n_datasets": len(entries),
            "r2_median": float(np.median(r2s)), "r2_min": float(r2s.min()),
            "r2_max": float(r2s.max()),
            "resid_pct_median": float(np.median(rms)), "resid_pct_max": float(rms.max()),
        }
        if want_pairs:
            pn = np.array([e["pair_node"]["r2"] for e in entries])
            ps = np.array([e["pair_slot"]["r2"] for e in entries])
            block["pair_node_r2_median"] = float(np.median(pn))
            block["pair_slot_r2_median"] = float(np.median(ps))
        out["by_arm"][str(arm)] = block
    if want_cores:
        out["coresidency"] = {
            str(c): {"n_plans": len(v),
                     "resid_pct_rms": float(np.sqrt((np.asarray(v) ** 2).mean())),
                     "resid_pct_mean": float(np.asarray(v).mean())}
            for c, v in sorted(cores_pool.items())
        }
    return out


def render(res: dict) -> None:
    print(f"\ncorpus: {res['corpus']}   ({res['n_datasets']} datasets fitted)")
    print(f"{'arm n_rows':>11s} {'n_ds':>5s} {'R2 median':>11s} {'R2 min':>10s} "
          f"{'resid% med':>11s} {'resid% max':>11s}")
    for arm, b in res["by_arm"].items():
        print(f"{arm:>11s} {b['n_datasets']:>5d} {b['r2_median']:>11.6f} "
              f"{b['r2_min']:>10.6f} {b['resid_pct_median']:>10.3f}% "
              f"{b['resid_pct_max']:>10.3f}%")
        if "pair_node_r2_median" in b:
            print(f"{'':>11s} {'':>5s} +pair-same-node R2 median "
                  f"{b['pair_node_r2_median']:.6f}   "
                  f"+pair-same-slot {b['pair_slot_r2_median']:.6f} "
                  f"(vacuous under globally-distinct replicas)")
    if "coresidency" in res:
        print("\n  residual by max tasks sharing one node (% of each dataset's mean rtt):")
        print(f"  {'tasks/node':>10s} {'n_plans':>10s} {'RMS%':>9s} {'mean%':>9s}")
        for c, b in res["coresidency"].items():
            print(f"  {c:>10s} {b['n_plans']:>10d} {b['resid_pct_rms']:>8.3f}% "
                  f"{b['resid_pct_mean']:>8.3f}%")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", action="append", required=True,
                    help="dataset collection dir (repeatable)")
    ap.add_argument("--co-residency", action="store_true",
                    help="break the residual down by max tasks sharing one node")
    ap.add_argument("--pairwise", action="store_true",
                    help="also fit same-node and same-slot pair terms (is it pairwise?)")
    ap.add_argument("--out", default=None, help="write the full JSON report here")
    args = ap.parse_args()

    results = []
    for c in args.corpus:
        res = measure_corpus(Path(c), args.co_residency, args.pairwise)
        render(res)
        results.append(res)

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
