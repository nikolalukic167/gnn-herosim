#!/usr/bin/env python3
"""Audit whether --spread-plans-only additive fits are saturated (params ≈ observations).

The spread-plans isolation control in separability_diagnostic.py restricts each sweep to
plans placing every task on a distinct node, then fits the additive model
rtt(plan) ≈ μ + Σ_t f_t(plan[t]).  A spread-plans additive R² of exactly 1.00000 is the
signature the control exists to produce — but it is also the signature of a fit with as
many free parameters as observations.  This audit separates the two:

1. rows/params ratio per dataset — free params = 1 + Σ_t (|levels_t| − 1); a ratio near 1
   means the fit is (near-)saturated and its R² carries no information.
2. held-out R² — fit on a random half of the spread plans (seed 0), score on the other
   half.  Overfitting cannot produce held-out R² = 1.0 exactly; additive physics can.

Verdict rule used when this was first run (2026-08-24, program_verdict_v1): the control
is suspect if the median ratio < 2; it survives iff median held-out R² ≥ 0.999, fails if
< 0.99.  Datasets with < 8 spread plans are skipped (a split of that size resolves
nothing) and datasets whose held-out R² < 0.999 AND ratio ≤ ~2.3 should be reported as
unresolvable at their own n, not as counter-evidence.

Measured 2026-08-24 (first 150 datasets per collection):
  gnn_datasets_4tasks_1060_warmth_v2   ratio median 97.3, held-out R² = 1.0 on 150/150
  gnn_datasets_4tasks_sparse_warmth_v2 ratio median 14.7, held-out R² = 1.0 on 137/144
                                       (7 failures, all ratio ≤ 2.29 / ≤ 16 rows)
  netc_multihop_v1_mh_off              ratio median 7.8 (min 2.0), held-out 1.0 on 48/48

Usage:
  pipenv run python3 scripts_cosim/audit_spread_fit_saturation.py <collection_dir_name>...
  (names are resolved under simulation_data/; pass --limit to cap datasets per collection)
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]


def load_spread(ds: Path):
    rows = []
    with open(ds / "placements" / "placements.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            plan = r["placement_plan"]
            nodes = [plan[k][0] for k in sorted(plan, key=int)]
            if len(set(nodes)) == len(nodes):
                rows.append(
                    ([tuple(plan[k]) for k in sorted(plan, key=int)], float(r["rtt"]))
                )
    return rows


def design(rows):
    n_tasks = len(rows[0][0])
    cats = [sorted({r[0][t] for r in rows}) for t in range(n_tasks)]
    # One-hot with intercept; drop the first level per task for identifiability.
    p = 1 + sum(len(c) - 1 for c in cats)
    X = np.zeros((len(rows), p))
    X[:, 0] = 1.0
    col = 1
    idx = {}
    for t, c in enumerate(cats):
        for lvl in c[1:]:
            idx[(t, lvl)] = col
            col += 1
    y = np.zeros(len(rows))
    for i, (plan, rtt) in enumerate(rows):
        y[i] = rtt
        for t, pl in enumerate(plan):
            j = idx.get((t, pl))
            if j is not None:
                X[i, j] = 1.0
    return X, y, p


def r2(y, pred):
    ss = ((y - y.mean()) ** 2).sum()
    return 1.0 if ss == 0 else 1 - ((y - pred) ** 2).sum() / ss


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("collections", nargs="+")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--heldout-detail-below", type=float, default=0.999,
                    help="List every dataset whose held-out R² falls below this.")
    args = ap.parse_args()

    exit_code = 0
    for name in args.collections:
        root = BASE_DIR / "simulation_data" / name
        if not root.is_dir():
            print(f"ERROR: no such collection dir: {root}", file=sys.stderr)
            exit_code = 1
            continue
        ratios, ho, bad, skipped = [], [], [], 0
        dss = sorted(d for d in root.iterdir() if d.name.startswith("ds_"))[: args.limit]
        for ds in dss:
            try:
                rows = load_spread(ds)
            except FileNotFoundError:
                skipped += 1
                continue
            if len(rows) < 4:
                skipped += 1
                continue
            X, y, p = design(rows)
            ratios.append(len(rows) / p)
            if len(rows) < 8:
                skipped += 1
                continue
            rng = random.Random(0)
            order = list(range(len(rows)))
            rng.shuffle(order)
            half = len(rows) // 2
            tr, te = order[:half], order[half:]
            beta, _, _, _ = np.linalg.lstsq(X[tr], y[tr], rcond=None)
            r = r2(y[te], X[te] @ beta)
            ho.append(r)
            if r < args.heldout_detail_below:
                bad.append((ds.name, len(rows), p, len(rows) / p, r))

        ra, hoa = np.array(ratios), np.array(ho)
        q = lambda a, x: float(np.quantile(a, x)) if len(a) else float("nan")
        print(f"{name}: n_ratio={len(ra)} n_heldout={len(hoa)} skipped={skipped}")
        print(f"  rows/params ratio: median={q(ra, .5):.2f} p10={q(ra, .1):.2f} "
              f"min={ra.min() if len(ra) else float('nan'):.2f}")
        print(f"  held-out R2:       median={q(hoa, .5):.5f} p10={q(hoa, .1):.5f} "
              f"min={hoa.min() if len(hoa) else float('nan'):.5f}")
        for b in bad:
            print("    below threshold: %s rows=%d params=%d ratio=%.2f heldoutR2=%.4f" % b)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
