#!/usr/bin/env python3
"""route_a scaling probe: does the parent->child transfer term create real coupling?

The go/no-go before spending an n>=200 corpus. `program_verdict_v1` closed the supervised
route by theorem — with per-task costs separable and placements freely chosen, the
componentwise minimiser is optimal under any monotone aggregation — and five physics
mechanisms have already died proving it. Route A's claim is that a child's input read
priced by the distance from its PARENT's node is a pairwise term over two jointly-decided
placements, and therefore not separable.

This measures exactly that claim, and nothing else:

  **additive-argmin regret** — fit the best per-task-independent model of cost that the
  sweep permits (each task's marginal best placement, taken independently), build the plan
  it implies, and ask how much worse that plan is than the true optimum. Zero regret means
  the target IS separable and route A has failed; the pointwise model already wins.

Reported on **spread plans only** (every task on a distinct node) as the primary view, for
the reason `separability_diagnostic.py` established: on that subset node-occupancy excess
is identically zero, so the one-integer repair that killed the previous four mechanisms is
a constant and cannot explain anything left over.

The lever is `stateSize`. Unlike link bandwidth — where the additive and interaction terms
both scale as 1/bandwidth and the ratio is INVARIANT — the coupled transfer scales with
stateSize while queue work does not. So the ratio should MOVE across the probe's arms. A
flat curve is itself the answer.

Usage:
  score_route_a_scaling_probe.py --corpus simulation_data/gnn_datasets_route_a_probe_153600 ...
"""

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---- Pre-registered go/no-go. Fix before looking at any number. -------------------
#
# PROCEED to a full corpus only if some arm clears BOTH:
#   * spread-plan additive-argmin regret  > MIN_REGRET_PCT, and
#   * the regret RISES with stateSize (the two-line prediction actually holding).
# Otherwise route A dies here, cheaply, which is the point of running a probe.
MIN_REGRET_PCT = 5.0

Plan = Dict[int, Tuple[int, int]]


def load_sweep(dataset_dir: Path) -> Optional[List[Tuple[Plan, float, Optional[float]]]]:
    path = dataset_dir / "placements" / "placements.jsonl"
    if not path.is_file():
        # A dataset the generator skipped or failed on. Counted and reported rather than
        # aborting — but never silently: a probe scored on an unknown subset of its corpus
        # is exactly the kind of number this repo has had to retire before.
        return None
    rows: List[Tuple[Plan, float, Optional[float]]] = []
    with open(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            plan = {int(k): tuple(v) for k, v in row["placement_plan"].items()}
            rows.append((plan, float(row["rtt"]), row.get("makespan")))
    if not rows:
        raise SystemExit(f"FAIL LOUD: {path} is empty")
    return rows


def spread_only(rows):
    """Every task on a distinct node — the subset where node-occupancy excess is 0."""
    out = []
    for plan, rtt, makespan in rows:
        nodes = [node for node, _platform in plan.values()]
        if len(set(nodes)) == len(nodes):
            out.append((plan, rtt, makespan))
    return out


def additive_argmin_regret(rows, objective: str) -> Optional[float]:
    """Relative regret of the best per-task-independent plan, vs the true optimum.

    The additive model each task can see: the minimum observed cost over all plans in which
    that task took a given placement. Taking each task's argmin independently is exactly
    what a pointwise model does. If the objective is separable that plan IS the optimum and
    this returns 0.
    """
    idx = 1 if objective == "rtt" else 2
    scored = [(plan, value) for plan, *rest in ((p, r, m) for p, r, m in rows)
              for value in [rest[idx - 1]] if value is not None]
    if not scored:
        return None

    marginal: Dict[int, Dict[Tuple[int, int], float]] = {}
    for plan, value in scored:
        for task_id, placement in plan.items():
            slot = marginal.setdefault(task_id, {})
            if placement not in slot or value < slot[placement]:
                slot[placement] = value

    additive_plan = {
        task_id: min(options.items(), key=lambda kv: (kv[1], kv[0]))[0]
        for task_id, options in marginal.items()
    }

    lookup = {tuple(sorted(plan.items())): value for plan, value in scored}
    key = tuple(sorted(additive_plan.items()))
    if key not in lookup:
        # The componentwise minimiser is not itself a feasible plan (collisions, or the
        # sweep does not enumerate it). That is a coupling signal in its own right, but a
        # different one; report it rather than silently scoring something else.
        return None
    best = min(value for _plan, value in scored)
    if best <= 0:
        return None
    return 100.0 * (lookup[key] - best) / best


def score_corpus(corpus: Path, objective: str) -> Dict[str, object]:
    datasets = sorted(d for d in corpus.glob("ds_*") if d.is_dir())
    if not datasets:
        raise SystemExit(f"FAIL LOUD: no ds_* under {corpus}")

    full_regrets, spread_regrets, infeasible, missing = [], [], 0, 0
    for dataset in datasets:
        rows = load_sweep(dataset)
        if rows is None:
            missing += 1
            continue
        r_full = additive_argmin_regret(rows, objective)
        if r_full is None:
            infeasible += 1
        else:
            full_regrets.append(r_full)
        spread = spread_only(rows)
        if spread:
            r_spread = additive_argmin_regret(spread, objective)
            if r_spread is not None:
                spread_regrets.append(r_spread)

    def summary(values):
        if not values:
            return {"n": 0, "mean": None, "median": None, "max": None, "nonzero_frac": None}
        return {
            "n": len(values),
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "max": max(values),
            "nonzero_frac": sum(1 for v in values if v > 1e-9) / len(values),
        }

    return {
        "corpus": str(corpus),
        "objective": objective,
        "n_datasets": len(datasets),
        "n_missing_sweeps": missing,
        "componentwise_plan_infeasible": infeasible,
        "full": summary(full_regrets),
        "spread": summary(spread_regrets),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, nargs="+", required=True)
    ap.add_argument("--objective", choices=("rtt", "makespan"), default="rtt")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    reports = [score_corpus(c, args.objective) for c in args.corpus]

    print(f"=== route_a scaling probe ({args.objective}) ===")
    print(f"{'corpus':52s} {'n':>3s} {'spread regret %':>16s} {'nonzero':>8s} {'max %':>8s}")
    for rep in reports:
        s = rep["spread"]
        mean = "n/a" if s["mean"] is None else f"{s['mean']:.3f}"
        nz = "n/a" if s["nonzero_frac"] is None else f"{s['nonzero_frac']:.2f}"
        mx = "n/a" if s["max"] is None else f"{s['max']:.3f}"
        flag = f"  !! {rep['n_missing_sweeps']} dataset(s) had no sweep" if rep["n_missing_sweeps"] else ""
        print(f"{Path(rep['corpus']).name[:52]:52s} {rep['n_datasets']:>3d} {mean:>16s} {nz:>8s} {mx:>8s}{flag}")

    print(f"\n{'corpus':52s} {'full-sweep regret %':>20s}")
    for rep in reports:
        f = rep["full"]
        mean = "n/a" if f["mean"] is None else f"{f['mean']:.3f}"
        print(f"{Path(rep['corpus']).name[:52]:52s} {mean:>20s}"
              + (f"   ({rep['componentwise_plan_infeasible']} infeasible)"
                 if rep["componentwise_plan_infeasible"] else ""))

    means = [r["spread"]["mean"] for r in reports if r["spread"]["mean"] is not None]
    best = max(means) if means else 0.0
    rising = len(means) >= 2 and means[-1] > means[0] + 1e-9

    print(f"\n=== GO / NO-GO (pre-registered: regret > {MIN_REGRET_PCT}% AND rising with stateSize) ===")
    print(f"  best spread-plan regret : {best:.3f}%")
    print(f"  rises with stateSize    : {rising}")
    if best > MIN_REGRET_PCT and rising:
        verdict = "GO"
        print("\nGO — the transfer term creates coupling a pointwise model cannot express,\n"
              "and it responds to the lever. Proceed to pre-registration and a full corpus.")
    else:
        verdict = "NO-GO"
        print("\nNO-GO — route A does not clear its own pre-registered bar on this physics.\n"
              "This is the cheap, intended failure point: nothing after the probe is spent.\n"
              "Report the table; do not soften the threshold after the fact.")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(
            {"verdict": verdict, "min_regret_pct": MIN_REGRET_PCT,
             "best_spread_regret_pct": best, "rising": rising, "reports": reports},
            indent=1, sort_keys=True) + "\n")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
