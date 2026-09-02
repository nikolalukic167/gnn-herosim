#!/usr/bin/env python3
"""The registered Phase 3 verdict: paired CL-minus-Frozen, over training seeds.

objective_pivot_v1 Phase 3 registration + Amendment D. Reads the gate's
`evaluate_policy.py` output and applies the criterion exactly as signed, with no
discretion left in this file:

  * the replication unit is the **training seed**, not the cell and not the episode
    (Amendment D1: the earlier sizing rule sized on evaluation noise, which is the wrong
    random variable);
  * for each seed, the paired quantity is the CL arm's mean total RTT across the gate
    cells minus its Frozen init's, as a fraction of the Frozen init's — so positive means
    closed-loop training helped;
  * the test is a one-sided exact Wilcoxon signed-rank over those per-seed differences;
  * **KILL: if the paired improvement is <= 0 at the registered n, P1 freezes as
    measured-negative.** No re-runs with tweaked hyperparameters. That is the registered
    consequence and this script prints it rather than leaving it to interpretation.

Knative and the frozen arms are reported alongside as context. They are not the primary
statistic and cannot change the verdict.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

MIN_SEEDS = 16  # Amendment D2 floor


def exact_wilcoxon_greater(diffs: List[float]) -> Tuple[float, float]:
    """One-sided exact Wilcoxon signed-rank, H1: median > 0. Returns (W+, p).

    Exact rather than normal-approximate: at n = 16 the approximation is not trustworthy
    at the tail, and this program's other paired tests are exact, so the numbers must be
    comparable to them.
    """
    nz = [d for d in diffs if d != 0.0]
    n = len(nz)
    if n == 0:
        return 0.0, 1.0
    order = sorted(range(n), key=lambda i: abs(nz[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(nz[order[j + 1]]) == abs(nz[order[i]]):
            j += 1
        avg = (i + j) / 2.0 + 1.0  # average rank over the tie block
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    w_plus = sum(r for d, r in zip(nz, ranks) if d > 0)
    # Enumerate all 2^n sign assignments of the observed ranks.
    if n > 22:
        raise SystemExit(f"FAIL LOUD: exact enumeration refused at n={n}; use a permutation test")
    count = 0
    total = 0
    for signs in product((0, 1), repeat=n):
        total += 1
        if sum(r for s, r in zip(signs, ranks) if s) >= w_plus - 1e-12:
            count += 1
    return w_plus, count / total


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gate-json", type=Path, required=True,
                    help="Output of evaluate_policy.py on the GATE cells.")
    ap.add_argument("--cl-prefix", required=True,
                    help="Label prefix of the closed-loop arm, e.g. 'cl_gnn_s' (seed appended).")
    ap.add_argument("--frozen-label", required=True,
                    help="Label of this arm's frozen init, e.g. 'frozen_gnn'.")
    ap.add_argument("--knative-label", default="knative")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--min-seeds", type=int, default=MIN_SEEDS)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    payload = json.loads(args.gate_json.read_text())
    rows = payload["rows"]
    cells = payload["cells"]

    def mean_for(label: str) -> Optional[float]:
        vals = [r["total_rtt"] for r in rows if r["label"] == label]
        if not vals:
            return None
        if len(vals) != len(cells):
            raise SystemExit(
                f"FAIL LOUD: arm {label!r} has {len(vals)} cells but the gate declares "
                f"{len(cells)}. A partially-evaluated arm cannot be paired."
            )
        return st.mean(vals)

    frozen = mean_for(args.frozen_label)
    if frozen is None:
        raise SystemExit(f"FAIL LOUD: no rows for frozen arm {args.frozen_label!r}")

    seeds: List[Tuple[str, float]] = []
    for label in sorted({r["label"] for r in rows}):
        if label.startswith(args.cl_prefix):
            seeds.append((label, mean_for(label)))
    if len(seeds) < args.min_seeds:
        raise SystemExit(
            f"FAIL LOUD: {len(seeds)} closed-loop seeds found, Amendment D2 sets a floor "
            f"of {args.min_seeds}. Reporting a verdict below the floor is exactly the "
            f"single-draw failure the floor exists to prevent."
        )

    # Positive = the closed-loop policy is FASTER than the frozen init it started from.
    diffs = [(frozen - m) / frozen for _, m in seeds]
    w_plus, p = exact_wilcoxon_greater(diffs)
    mean_d, median_d = st.mean(diffs), st.median(diffs)
    n_pos = sum(1 for d in diffs if d > 0)

    fired = (median_d > 0) and (p < args.alpha)
    verdict = "POSITIVE" if fired else (
        "MEASURED-NEGATIVE (P1 frozen; kill criterion)" if median_d <= 0
        else "NOT ESTABLISHED at the registered alpha"
    )

    summary: Dict[str, Any] = {
        "gate_json": str(args.gate_json),
        "cells": cells,
        "n_seeds": len(seeds),
        "frozen_label": args.frozen_label,
        "frozen_mean_total_rtt": frozen,
        "per_seed_rel_improvement": {lbl: (frozen - m) / frozen for lbl, m in seeds},
        "mean_rel_improvement": mean_d,
        "median_rel_improvement": median_d,
        "n_seeds_better_than_frozen": n_pos,
        "wilcoxon_w_plus": w_plus,
        "wilcoxon_p_one_sided": p,
        "alpha": args.alpha,
        "verdict": verdict,
    }
    kn = mean_for(args.knative_label)
    if kn is not None:
        summary["knative_mean_total_rtt"] = kn
        summary["frozen_vs_knative_rel"] = (kn - frozen) / kn
        summary["cl_median_vs_knative_rel"] = (
            kn - st.median([m for _, m in seeds])) / kn

    print("\n=== PHASE 3 GATE — registered readout ===")
    print(f"  cells            : {', '.join(cells)}")
    print(f"  seeds            : {len(seeds)} (floor {args.min_seeds})")
    print(f"  frozen mean RTT  : {frozen:,.1f}")
    if kn is not None:
        print(f"  knative mean RTT : {kn:,.1f}  (frozen is {summary['frozen_vs_knative_rel']:+.2%} vs it)")
    print(f"  paired improvement over frozen: mean {mean_d:+.4%}, median {median_d:+.4%}")
    print(f"  seeds better than frozen: {n_pos}/{len(seeds)}")
    print(f"  exact Wilcoxon one-sided p = {p:.6f}  (alpha {args.alpha})")
    print(f"  VERDICT: {verdict}")
    if median_d <= 0:
        print("\n  The kill criterion applies as registered: P1 freezes as measured-negative.")
        print("  No re-runs with tweaked hyperparameters. This closes the last open path")
        print("  to the latency claim, and is itself an answer.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
