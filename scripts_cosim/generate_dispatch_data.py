"""Generate and independently validate the fixed-placement priority corpus."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts_cosim.audit_radical_physics import load_problem
from scripts_cosim.radical_physics_screen import save, sha
from src.placement.radical.dispatch import DispatchNative
from src.placement.radical.environment import initial, pack, problem, serialized
from src.policy.dispatch_priority.model import CONTRACT, features

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = [
    "scripts_cosim/generate_dispatch_data.py", "src/policy/dispatch_priority/model.py",
    "src/policy/mixed/model.py", "src/placement/radical/dispatch.py",
    "src/placement/radical/dispatch.cpp", "src/placement/radical/mixed.py",
    "src/placement/radical/mixed.cpp", "src/placement/radical/kernel.cpp",
    "src/placement/radical/environment.py",
]


def generate(output):
    if output.exists():
        raise ValueError("preserve earlier corpus")
    output.mkdir(parents=True)
    protocol = json.loads((ROOT / "experiments/mixed_dispatch_learning_v1_protocol.json").read_text())
    engine = DispatchNative(output / "build")
    sources = {path: sha(ROOT / path) for path in SOURCE_PATHS}
    save(output / "protocol_before_run.json", {"protocol": protocol, "sources": sources, "native": engine.provenance})
    prior = set()
    for path in output.parent.glob("*/source_validation.json"):
        for row in json.loads(path.read_text()).get("inputs", {}).values():
            if isinstance(row, dict) and "identity" in row:
                prior.add(row["identity"])
    cases = []
    inputs = {}
    for split in ("train", "validation", "test"):
        low, high = protocol[f"{split}_seeds"]
        for seed in range(low, high + 1):
            physical = problem(seed)
            identity = hashlib.sha256(pack(physical).tobytes()).hexdigest()
            if identity in prior:
                raise ValueError("reused physical problem")
            prior.add(identity)
            cell = output / f"ds_{seed}"
            (cell / "placements").mkdir(parents=True)
            path = cell / "problem.json"
            save(path, serialized(physical))
            record = {"seed": seed, "split": split, "identity": identity, "sha256": sha(path)}
            cases.append(record)
            inputs[str(seed)] = record
    save(output / "source_validation.json", {"status": "PASS", "inputs": inputs, "prior_identities_checked": len(prior) - len(cases)})
    files = {}
    for split in ("train", "validation", "test"):
        buffers = {key: [] for key in ("seeds", "x", "xh", "adj", "assignment", "target", "teacher_cost", "rule_cost")}
        for record in [row for row in cases if row["split"] == split]:
            seed = record["seed"]
            cell = output / f"ds_{seed}"
            physical = load_problem(cell / "problem.json")
            if sha(cell / "problem.json") != record["sha256"] or any(sha(ROOT / path) != digest for path, digest in sources.items()):
                raise ValueError("source changed during generation")
            _, assignment = engine.search(physical, initial(physical, "affinity"), 64)
            rule_cost, rule_priority = engine.search_priority(physical, assignment,
                __import__("src.placement.radical.dispatch", fromlist=["priorities"]).priorities(physical, assignment, "srpt"), 128)
            teachers = []
            for start in ("spt", "srpt"):
                start_priority = __import__("src.placement.radical.dispatch", fromlist=["priorities"]).priorities(physical, assignment, start)
                teachers.append((start, *engine.search_priority(physical, assignment, start_priority, 8192)))
            chosen = min([("rule", rule_cost, rule_priority), *teachers], key=lambda row: row[1])
            x, adjacency = features(physical, assignment)
            xh, _ = features(physical, assignment, True)
            target = chosen[2].reshape(-1).astype(np.float32) / (assignment.size - 1)
            placements = cell / "placements" / "placements.jsonl"
            with placements.open("w") as stream:
                stream.write(json.dumps({"arm": "srpt128", "placement_plan": assignment.tolist(),
                                         "priority": rule_priority.tolist(), "objective": rule_cost}) + "\n")
                for name, cost, priority in teachers:
                    stream.write(json.dumps({"arm": f"teacher_{name}8192", "placement_plan": assignment.tolist(),
                                             "priority": priority.tolist(), "objective": cost}) + "\n")
            save(cell / "best.json", {"teacher_arm": chosen[0], "objective": chosen[1],
                 "placement_plan": assignment.tolist(), "priority": chosen[2].tolist(), "global_optimality_certified": False})
            record.update({"placements_sha256": sha(placements), "best_sha256": sha(cell / "best.json")})
            values = {"seeds": seed, "x": x, "xh": xh, "adj": adjacency, "assignment": assignment,
                      "target": target, "teacher_cost": chosen[1], "rule_cost": rule_cost}
            for key, value in values.items():
                buffers[key].append(value)
            if seed % 8 == 7:
                print("DATA", split, seed, flush=True)
        path = output / f"{split}.npz"
        np.savez_compressed(path, **{key: np.array(value) for key, value in buffers.items()})
        files[path.name] = sha(path)
    save(output / "METADATA.json", {"contract": CONTRACT, "physics": "mixed_execution_v1",
         "dispatch": "mixed_ready_priority_v1", "sources": sources, "files": files, "cases": cases})
    validate(output)


def validate(root):
    metadata = json.loads((root / "METADATA.json").read_text())
    engine = DispatchNative(root / "validation_build")
    if any(sha(ROOT / path) != digest for path, digest in metadata["sources"].items()):
        raise ValueError("data source changed")
    seen = set()
    for split in ("train", "validation", "test"):
        if sha(root / f"{split}.npz") != metadata["files"][f"{split}.npz"]:
            raise ValueError("cache changed")
        with np.load(root / f"{split}.npz") as cache:
            for index, seed in enumerate(cache["seeds"]):
                record = next(row for row in metadata["cases"] if row["seed"] == seed)
                physical = load_problem(root / f"ds_{seed}" / "problem.json")
                identity = hashlib.sha256(pack(physical).tobytes()).hexdigest()
                if identity in seen or identity != record["identity"] or record["split"] != split:
                    raise ValueError("identity/split failure")
                seen.add(identity)
                cell = root / f"ds_{seed}"
                if sha(cell / "problem.json") != record["sha256"] or sha(cell / "placements/placements.jsonl") != record["placements_sha256"] or sha(cell / "best.json") != record["best_sha256"]:
                    raise ValueError("dataset artifact changed")
                assignment = cache["assignment"][index]
                x, adjacency = features(physical, assignment)
                xh, _ = features(physical, assignment, True)
                if not np.array_equal(x, cache["x"][index]) or not np.array_equal(xh, cache["xh"][index]) or not np.array_equal(adjacency, cache["adj"][index]):
                    raise ValueError("feature mismatch")
                best = json.loads((cell / "best.json").read_text())
                priority = np.array(best["priority"])
                if engine.score_priority(physical, assignment, priority)[0] != best["objective"] or best["objective"] != cache["teacher_cost"][index]:
                    raise ValueError("teacher mismatch")
                expected = priority.reshape(-1) / (assignment.size - 1)
                if not np.array_equal(expected.astype(np.float32), cache["target"][index]):
                    raise ValueError("target mismatch")
                rows = [json.loads(line) for line in (cell / "placements/placements.jsonl").read_text().splitlines()]
                if len(rows) != 3 or min(row["objective"] for row in rows) != best["objective"]:
                    raise ValueError("incomplete teacher arms")
                for row in rows:
                    if engine.score_priority(physical, np.array(row["placement_plan"]), np.array(row["priority"]))[0] != row["objective"]:
                        raise ValueError("retained plan mismatch")
    save(root / "VALIDATION.json", {"status": "PASS", "unique_inputs": len(seen),
         "retained_final_plans": len(seen) * 3, "metadata_sha256": sha(root / "METADATA.json")})
    print("VALIDATED", len(seen), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    generate(parser.parse_args().out)
