"""Fresh-data qualification for ready-set state/action dispatch learning."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts_cosim.radical_physics_screen import save, sha
from src.placement.radical.dispatch import DispatchNative, priorities
from src.placement.radical.dispatch_state import trace_priority
from src.placement.radical.environment import initial, pack, problem, serialized

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ["experiments/mixed_dispatch_state_v1_protocol.json",
           "scripts_cosim/screen_dispatch_state.py",
           "src/placement/radical/dispatch_state.py",
           "src/placement/radical/dispatch.py", "src/placement/radical/dispatch.cpp",
           "src/placement/radical/mixed.cpp", "src/placement/radical/kernel.cpp",
           "src/placement/radical/environment.py"]


def searched(engine, physical, assignment, seed, steps=8192):
    q, assignment, dims, rank = engine.checked_priority(
        physical, assignment, priorities(physical, assignment, "srpt"))
    cost = engine.dispatch_lib.dispatch_search(q, assignment, rank, *dims, steps, seed)
    return float(cost), rank


def run(output):
    if output.exists():
        raise ValueError("preserve earlier screen")
    output.mkdir(parents=True)
    protocol = json.loads((ROOT / "experiments/mixed_dispatch_state_v1_protocol.json").read_text())
    sources = {path: sha(ROOT / path) for path in SOURCES}
    engine = DispatchNative(output / "build")
    low, high = protocol["validation_seeds"]
    rows = []
    identities = set()
    for seed in range(low, high + 1):
        physical = problem(seed)
        identity = hashlib.sha256(pack(physical).tobytes()).hexdigest()
        if identity in identities:
            raise ValueError("duplicate infrastructure")
        identities.add(identity)
        assignment = engine.search(physical, initial(physical, "affinity"), 64)[1]
        rule_cost, _ = engine.search_priority(
            physical, assignment, priorities(physical, assignment, "srpt"), 128)
        candidates = [searched(engine, physical, assignment, teacher_seed)
                      for teacher_seed in (77, 78, 79, 80)]
        teacher_cost, teacher_rank = min(candidates, key=lambda value: value[0])
        realized_rank, states = trace_priority(physical, assignment, teacher_rank)
        realized_cost, _ = engine.score_priority(physical, assignment, realized_rank)
        if realized_cost != teacher_cost:
            raise ValueError("state trace does not reproduce teacher")
        sizes = [int(state["feasible"].sum()) for state in states]
        cell = output / f"ds_{seed}"
        cell.mkdir()
        save(cell / "problem.json", serialized(physical))
        save(cell / "result.json", {"assignment": assignment.tolist(),
             "teacher_priority": teacher_rank.tolist(), "realized_priority": realized_rank.tolist(),
             "teacher_cost": teacher_cost, "rule_cost": rule_cost, "candidate_sizes": sizes})
        rows.append({"seed": seed, "identity": identity, "teacher_cost": teacher_cost,
                     "rule_cost": rule_cost, "gain": (rule_cost - teacher_cost) / rule_cost,
                     "states": len(states), "multi_choice_states": sum(size > 1 for size in sizes),
                     "mean_candidates": float(np.mean(sizes)),
                     "problem_sha256": sha(cell / "problem.json"),
                     "result_sha256": sha(cell / "result.json")})
    gains = np.asarray([row["gain"] for row in rows])
    total_states = sum(row["states"] for row in rows)
    multi = sum(row["multi_choice_states"] for row in rows)
    result = {"status": "PASS" if gains.mean() >= .05 and multi / total_states >= .25 else "NO_GO",
              "contract": "mixed_dispatch_ready_set_v1", "sources": sources,
              "unique_inputs": len(identities), "mean_teacher_gain": float(gains.mean()),
              "median_teacher_gain": float(np.median(gains)),
              "multi_choice_state_fraction": multi / total_states,
              "mean_candidates": float(np.mean([row["mean_candidates"] for row in rows])),
              "exact_teacher_replays": len(rows), "rows": rows}
    save(output / "read.json", result)
    print(json.dumps({key: value for key, value in result.items() if key not in ("rows", "sources")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args().out)
