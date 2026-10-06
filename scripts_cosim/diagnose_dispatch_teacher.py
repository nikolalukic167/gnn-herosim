"""Measure whether equal-quality dispatch teachers imply stable total-rank labels."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from scripts_cosim.audit_radical_physics import load_problem
from scripts_cosim.radical_physics_screen import save
from src.placement.radical.dispatch import DispatchNative, priorities
from src.placement.radical.dispatch_live import replay


def searched(engine, problem, assignment, seed, steps):
    q, assignment, dims, rank = engine.checked_priority(
        problem, assignment, priorities(problem, assignment, "srpt")
    )
    cost = engine.dispatch_lib.dispatch_search(
        q, assignment, rank, *dims, steps, seed
    )
    if not np.isfinite(cost):
        raise RuntimeError("nonterminating teacher search")
    return float(cost), rank


def start_order(problem, assignment, rank):
    starts = [event for event in replay(problem, assignment, rank)["events"]
              if event["event"] == "start"]
    return [(event["job"], event["operation"]) for event in starts]


def inversion_fraction(left, right):
    position = {value: index for index, value in enumerate(right)}
    mapped = [position[value] for value in left]
    inversions = sum(mapped[i] > mapped[j] for i in range(len(mapped))
                     for j in range(i + 1, len(mapped)))
    pairs = len(mapped) * (len(mapped) - 1) // 2
    return inversions / pairs


def diagnose(corpus, output, seeds, steps):
    if output.exists():
        raise ValueError("preserve earlier diagnostic")
    engine = DispatchNative(output.parent / "teacher_diagnostic_build")
    rows = []
    for cell in sorted(corpus.glob("ds_131*")):
        problem = load_problem(cell / "problem.json")
        best = json.loads((cell / "best.json").read_text())
        assignment = np.asarray(best["placement_plan"], dtype=np.int64)
        alternatives = []
        for seed in seeds:
            cost, rank = searched(engine, problem, assignment, seed, steps)
            alternatives.append({"seed": seed, "cost": cost, "rank": rank,
                                 "order": start_order(problem, assignment, rank)})
        for left, right in itertools.combinations(alternatives, 2):
            rank_mae = np.abs(left["rank"] - right["rank"]).mean() / (assignment.size - 1)
            rows.append({
                "problem_seed": int(problem["seed"]),
                "left_seed": left["seed"],
                "right_seed": right["seed"],
                "relative_cost_gap": abs(left["cost"] - right["cost"]) /
                                     min(left["cost"], right["cost"]),
                "normalized_rank_mae": float(rank_mae),
                "start_order_inversion_fraction": inversion_fraction(
                    left["order"], right["order"]),
                "identical_start_order": left["order"] == right["order"],
            })
    def quantiles(key):
        values = np.asarray([row[key] for row in rows], dtype=float)
        return {"median": float(np.median(values)), "p25": float(np.quantile(values, .25)),
                "p75": float(np.quantile(values, .75)), "max": float(values.max())}
    result = {
        "status": "PASS", "contract": "mixed_dispatch_teacher_stability_v1",
        "problems": len({row["problem_seed"] for row in rows}), "pairs": len(rows),
        "search_seeds": seeds, "steps": steps,
        "relative_cost_gap": quantiles("relative_cost_gap"),
        "normalized_rank_mae": quantiles("normalized_rank_mae"),
        "start_order_inversion_fraction": quantiles("start_order_inversion_fraction"),
        "identical_start_order_fraction": float(np.mean([row["identical_start_order"] for row in rows])),
        "near_equal": {},
        "rows": rows,
    }
    for threshold in (.01, .02, .05):
        subset = [row for row in rows if row["relative_cost_gap"] <= threshold]
        result["near_equal"][str(threshold)] = {
            "pairs": len(subset),
            "normalized_rank_mae_median": float(np.median(
                [row["normalized_rank_mae"] for row in subset])) if subset else None,
            "start_order_inversion_median": float(np.median(
                [row["start_order_inversion_fraction"] for row in subset])) if subset else None,
            "identical_start_order_fraction": float(np.mean(
                [row["identical_start_order"] for row in subset])) if subset else None,
        }
    save(output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[77, 78, 79, 80])
    parser.add_argument("--steps", type=int, default=8192)
    args = parser.parse_args()
    diagnose(args.corpus, args.out, args.seeds, args.steps)
