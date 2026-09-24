#!/usr/bin/env python3
"""serving_stability_v1 -- S1 (early advantage) and S2 (stability).

Registered in docs/lineages/serving_stability_v1.md on 2026-09-15. Every bar below is a
module constant and was committed before the arms it reads had finished.

The claim: the learned arms beat reactive Knative while queues are shallow and lose once
their own queues run away, because reactive's min(queue) rule is self-stabilising by
construction and the learned arms have no such mechanism.

S1 EARLY ADVANTAGE (the surprising one, and the one most likely to evaporate).
   Per arm and seed, mean queue time over deciles 1-2 minus reactive's on the same cell.
   Bar: median over seeds < 0, in >= S1_MIN_SEEDS of the seeds read, on >= S1_MIN_CELLS of
   3 cells, for the gnn arm => EARLY-ADVANTAGE-REAL.
   Registered expectation: UNCERTAIN. It rests on two checkpoints on one cell.

S2 STABILITY.
   Per arm and seed, (max over deciles of mean queue) / (min over deciles).
   Bar: reactive's ratio <= S2_REACTIVE_MAX_RATIO on a cell AND each learned arm's median
   ratio >= S2_ARM_MULTIPLE x reactive's, on >= S2_MIN_CELLS of 3 => ARMS-DO-NOT-STABILISE.

BURNED, and excluded: the scoping read used gnn seeds 8 and 14 and the reactive arm on
cell_s7901's trace-and-cell. Those three are dropped from C1 and C1 therefore contributes a
reduced seed count; C2 and C3 are untouched. A bar cannot be applied to the data that
suggested it.

DECLARED IN ADVANCE: gnn seed 3's checkpoint deterministically livelocks the simulator
(drainable_objective_v1 Phase C, reproduced byte-for-byte), so it is absent from every cell
and the gnn arm is read at n = 15 -- or n = 13 on C1, where the burn also removes 8 and 14.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# --------------------------------------------------------------------------- BARS
S1_MIN_SEEDS = 12
S1_MIN_CELLS = 2
S2_REACTIVE_MAX_RATIO = 2.5
S2_ARM_MULTIPLE = 2.0
S2_MIN_CELLS = 2
EARLY_DECILES = 2                 # "while queues are shallow" = the first fifth of the trace
MIN_SEEDS_PER_CELL_ARM = 12       # below this a (cell, arm) cell is VOID, not read
S1_REGISTERED_EXPECTATION = "UNCERTAIN"

CELLS = ("cell_s7901_f4000_pg16", "cell_s9001_f4000_pg16", "cell_s9002_f4000_pg16")
ARMS = ("gnn", "mpoff")

# The scoping read that motivated this lineage used these, on this cell only.
BURNED: Dict[str, Tuple[Tuple[str, int], ...]] = {
    "cell_s7901_f4000_pg16": (("gnn", 8), ("gnn", 14)),
}
# gnn seed 3 livelocks on every cell; declared, not discovered.
KNOWN_UNSERVABLE: Tuple[Tuple[str, int], ...] = (("gnn", 3),)


class StabilityReadError(RuntimeError):
    """Fail loud: a quietly dropped arm is a quietly different experiment."""


def early_queue(summary: Dict[str, Any], n: int = EARLY_DECILES) -> float:
    rows = (summary.get("decile_summary") or {}).get("deciles") or []
    if len(rows) < n:
        raise StabilityReadError(f"{summary.get('arm')}: {len(rows)} deciles, need {n}")
    vals = [r["mean_queue_s"] for r in rows[:n]]
    if any(v is None for v in vals):
        raise StabilityReadError(f"{summary.get('arm')}: an early decile has no mean queue")
    weights = [r["n"] for r in rows[:n]]
    return sum(v * w for v, w in zip(vals, weights)) / sum(weights)


def stability_ratio(summary: Dict[str, Any]) -> Optional[float]:
    return (summary.get("decile_summary") or {}).get("queue_max_over_min")


def load(results: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for path in sorted(results.glob("*.summary.json")):
        doc = json.loads(path.read_text())
        arm = doc.get("arm")
        if not arm:
            raise StabilityReadError(f"{path}: no arm name")
        out[arm] = doc
    if not out:
        raise StabilityReadError(f"{results}: no summaries")
    return out


def seeds_for(arms: Dict[str, Dict[str, Any]], cell: str, arm: str) -> List[int]:
    burned = set(BURNED.get(cell, ()))
    keep: List[int] = []
    for seed in range(1, 17):
        if (arm, seed) in burned or (arm, seed) in KNOWN_UNSERVABLE:
            continue
        if f"{cell}__{arm}_s{seed}" in arms:
            keep.append(seed)
    return keep


def read(arms: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "lineage": "serving_stability_v1",
        "bars": {"s1_min_seeds": S1_MIN_SEEDS, "s1_min_cells": S1_MIN_CELLS,
                 "early_deciles": EARLY_DECILES,
                 "s2_reactive_max_ratio": S2_REACTIVE_MAX_RATIO,
                 "s2_arm_multiple": S2_ARM_MULTIPLE, "s2_min_cells": S2_MIN_CELLS},
        "excluded": {"burned": {c: [list(x) for x in v] for c, v in BURNED.items()},
                     "known_unservable": [list(x) for x in KNOWN_UNSERVABLE]},
        "S1": {"per_cell": {}}, "S2": {"per_cell": {}},
    }
    s1_cells: Dict[str, Dict[str, bool]] = {}
    s2_cells: Dict[str, Dict[str, bool]] = {}

    for cell in CELLS:
        ref = arms.get(f"{cell}__reactive")
        if ref is None:
            raise StabilityReadError(f"{cell}: no reactive arm -- nothing to compare against")
        ref_early = early_queue(ref)
        ref_ratio = stability_ratio(ref)
        s1_row: Dict[str, Any] = {"reactive_early_queue_s": ref_early}
        s2_row: Dict[str, Any] = {"reactive_ratio": ref_ratio,
                                  "reactive_clears": ref_ratio is not None
                                  and ref_ratio <= S2_REACTIVE_MAX_RATIO}
        s1_cells[cell] = {}
        s2_cells[cell] = {}
        for arm in ARMS:
            seeds = seeds_for(arms, cell, arm)
            if len(seeds) < MIN_SEEDS_PER_CELL_ARM:
                s1_row[arm] = {"verdict": "VOID-TOO-FEW-SEEDS", "n": len(seeds)}
                s2_row[arm] = {"verdict": "VOID-TOO-FEW-SEEDS", "n": len(seeds)}
                s1_cells[cell][arm] = False
                s2_cells[cell][arm] = False
                continue
            deltas = [early_queue(arms[f"{cell}__{arm}_s{s}"]) - ref_early for s in seeds]
            better = sum(1 for d in deltas if d < 0)
            fires = st.median(deltas) < 0 and better >= S1_MIN_SEEDS
            s1_row[arm] = {"n": len(seeds), "median_delta_s": st.median(deltas),
                           "seeds_ahead": better, "min_delta_s": min(deltas),
                           "max_delta_s": max(deltas), "fires": fires}
            s1_cells[cell][arm] = fires

            ratios = [r for r in (stability_ratio(arms[f"{cell}__{arm}_s{s}"]) for s in seeds)
                      if r is not None]
            if not ratios:
                s2_row[arm] = {"verdict": "VOID-NO-RATIOS", "n": 0}
                s2_cells[cell][arm] = False
                continue
            med = st.median(ratios)
            arm_fires = bool(s2_row["reactive_clears"] and ref_ratio
                             and med >= S2_ARM_MULTIPLE * ref_ratio)
            s2_row[arm] = {"n": len(ratios), "median_ratio": med, "min_ratio": min(ratios),
                           "max_ratio": max(ratios), "fires": arm_fires}
            s2_cells[cell][arm] = arm_fires
        out["S1"]["per_cell"][cell] = s1_row
        out["S2"]["per_cell"][cell] = s2_row

    gnn_cells = sum(1 for c in CELLS if s1_cells[c].get("gnn"))
    out["S1"]["gnn_cells_firing"] = gnn_cells
    out["S1"]["fires"] = gnn_cells >= S1_MIN_CELLS
    out["S1"]["verdict"] = ("EARLY-ADVANTAGE-REAL" if out["S1"]["fires"]
                            else "NO-EARLY-ADVANTAGE")
    out["S1"]["registered_expectation"] = S1_REGISTERED_EXPECTATION
    out["S1"]["mpoff_cells_firing"] = sum(1 for c in CELLS if s1_cells[c].get("mpoff"))

    s2_both = sum(1 for c in CELLS if all(s2_cells[c].get(a) for a in ARMS))
    out["S2"]["cells_firing"] = s2_both
    out["S2"]["fires"] = s2_both >= S2_MIN_CELLS
    out["S2"]["verdict"] = ("ARMS-DO-NOT-STABILISE" if out["S2"]["fires"]
                            else "STABILITY-NOT-SEPARATED")
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    result = read(load(args.results))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))

    print(f"\n[S1] early advantage over deciles 1-{EARLY_DECILES}: median delta < 0 and "
          f">= {S1_MIN_SEEDS} seeds ahead, on >= {S1_MIN_CELLS} of {len(CELLS)} cells")
    for cell, row in result["S1"]["per_cell"].items():
        print(f"     {cell}  reactive early queue {row['reactive_early_queue_s']:.3f} s")
        for arm in ARMS:
            a = row[arm]
            if "verdict" in a:
                print(f"       {arm:6s} {a['verdict']} (n={a['n']})")
            else:
                print(f"       {arm:6s} n={a['n']:2d}  median delta "
                      f"{a['median_delta_s']:+8.3f} s  ahead {a['seeds_ahead']}/{a['n']}"
                      f"  range {a['min_delta_s']:+.2f}..{a['max_delta_s']:+.2f}"
                      f"  -> {'FIRES' if a['fires'] else 'no'}")
    print(f"[S1] {result['S1']['verdict']}  (gnn cells firing "
          f"{result['S1']['gnn_cells_firing']}/{len(CELLS)}, mpoff "
          f"{result['S1']['mpoff_cells_firing']}/{len(CELLS)}; registered expectation "
          f"{S1_REGISTERED_EXPECTATION})")

    print(f"\n[S2] stability: reactive max/min decile queue <= {S2_REACTIVE_MAX_RATIO} and "
          f"each arm >= {S2_ARM_MULTIPLE}x reactive, on >= {S2_MIN_CELLS} cells")
    for cell, row in result["S2"]["per_cell"].items():
        rr = row["reactive_ratio"]
        print(f"     {cell}  reactive {rr if rr is None else f'{rr:.2f}'}"
              f" -> {'clears' if row['reactive_clears'] else 'FAILS'}")
        for arm in ARMS:
            a = row[arm]
            if "verdict" in a:
                print(f"       {arm:6s} {a['verdict']}")
            else:
                print(f"       {arm:6s} n={a['n']:2d}  median {a['median_ratio']:6.2f}"
                      f"  range {a['min_ratio']:.2f}..{a['max_ratio']:.2f}"
                      f"  -> {'FIRES' if a['fires'] else 'no'}")
    print(f"[S2] {result['S2']['verdict']}  ({result['S2']['cells_firing']}/{len(CELLS)} cells)")
    print(f"[wrote] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
