"""Generate exact counterfactual feasible-action value data."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts_cosim.radical_physics_screen import save, sha
from src.placement.radical.dispatch import DispatchNative, priorities
from src.placement.radical.dispatch_value import trajectory_values
from src.placement.radical.environment import initial, pack, problem, serialized
from src.policy.dispatch_priority.model import features
from src.policy.dispatch_value.model import CONTRACT

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ["experiments/mixed_dispatch_value_v1_protocol.json",
           "scripts_cosim/generate_dispatch_value_data.py",
           "src/placement/radical/dispatch_value.py", "src/placement/radical/dispatch_state.py",
           "src/policy/dispatch_value/model.py", "src/policy/dispatch_state/model.py",
           "src/policy/dispatch_priority/model.py", "src/policy/mixed/model.py",
           "src/placement/radical/dispatch.py", "src/placement/radical/dispatch.cpp",
           "src/placement/radical/mixed.cpp", "src/placement/radical/kernel.cpp",
           "src/placement/radical/environment.py"]


def searched(engine, physical, assignment, seed):
    q, assignment, dims, rank = engine.checked_priority(
        physical, assignment, priorities(physical, assignment, "srpt"))
    cost = engine.dispatch_lib.dispatch_search(q, assignment, rank, *dims, 8192, seed)
    return float(cost), rank


def generate(output):
    if output.exists():
        raise ValueError("preserve earlier corpus")
    output.mkdir(parents=True)
    protocol = json.loads((ROOT / "experiments/mixed_dispatch_value_v1_protocol.json").read_text())
    sources = {path: sha(ROOT / path) for path in SOURCES}
    engine = DispatchNative(output / "build")
    cases, files, identities = [], {}, set()
    for split in ("train", "validation", "test"):
        buffers = {key: [] for key in ("seeds", "x", "xh", "adj", "assignment", "dynamic",
                                        "feasible", "action_cost", "teacher_cost", "rule_cost")}
        low, high = protocol[f"{split}_seeds"]
        for seed in range(low, high + 1):
            physical = problem(seed)
            identity = hashlib.sha256(pack(physical).tobytes()).hexdigest()
            if identity in identities:
                raise ValueError("duplicate physical input")
            identities.add(identity)
            assignment = engine.search(physical, initial(physical, "affinity"), 64)[1]
            rule_cost, rule_rank = engine.search_priority(
                physical, assignment, priorities(physical, assignment, "srpt"), 128)
            teacher_cost, teacher_rank = min(
                (searched(engine, physical, assignment, search_seed)
                 for search_seed in (77, 78, 79, 80)), key=lambda value: value[0])
            trajectory = [trajectory_values(engine, physical, assignment, rank)
                          for rank in (teacher_rank, rule_rank)]
            if engine.score_priority(physical, assignment, trajectory[0][0])[0] != teacher_cost:
                raise ValueError("teacher value trajectory mismatch")
            if engine.score_priority(physical, assignment, trajectory[1][0])[0] != rule_cost:
                raise ValueError("rule value trajectory mismatch")
            states = [state for _, rows, _ in trajectory for state in rows]
            action_cost = np.concatenate([values for _, _, values in trajectory])
            x, adjacency = features(physical, assignment)
            xh, _ = features(physical, assignment, True)
            cell = output / f"ds_{seed}"
            cell.mkdir()
            save(cell / "problem.json", serialized(physical))
            save(cell / "best.json", {"objective": teacher_cost,
                 "placement_plan": assignment.tolist(), "priority": trajectory[0][0].tolist(),
                 "global_optimality_certified": False})
            cases.append({"seed": seed, "split": split, "identity": identity,
                          "problem_sha256": sha(cell / "problem.json"),
                          "best_sha256": sha(cell / "best.json")})
            values = {"seeds": seed, "x": x, "xh": xh, "adj": adjacency,
                      "assignment": assignment,
                      "dynamic": np.stack([state["dynamic"] for state in states]),
                      "feasible": np.stack([state["feasible"] for state in states]),
                      "action_cost": action_cost, "teacher_cost": teacher_cost,
                      "rule_cost": rule_cost}
            for key, value in values.items():
                buffers[key].append(value)
            if seed % 8 == 7:
                print("DATA", split, seed, flush=True)
        path = output / f"{split}.npz"
        np.savez_compressed(path, **{key: np.asarray(value) for key, value in buffers.items()})
        files[path.name] = sha(path)
    save(output / "METADATA.json", {"contract": CONTRACT, "physics": "mixed_execution_v1",
         "dispatch": "mixed_ready_action_value_v1", "sources": sources,
         "files": files, "cases": cases})
    save(output / "VALIDATION.json", {"status": "PASS", "unique_inputs": len(identities),
         "counterfactual_states": len(identities) * 96,
         "metadata_sha256": sha(output / "METADATA.json")})
    print("VALIDATED", len(cases), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    generate(parser.parse_args().out)
