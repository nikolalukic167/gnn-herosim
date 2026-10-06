"""Freeze, time, live-evaluate, and audit all learned priority schedulers."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts_cosim.evaluate_workflow import paired_summary
from scripts_cosim.mixed_physics_control_gate import timed
from scripts_cosim.radical_physics_screen import save, sha
from scripts_cosim.workflow_proposal_gate import configure_environment
from src.placement.radical.dispatch import DispatchNative
from src.placement.radical.dispatch_live import herosim, replay
from src.policy.dispatch_priority.model import CONTRACT, PriorityNet
from src.policy.dispatch_priority.train import load_split, serve

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_SOURCES = [
    "scripts_cosim/evaluate_dispatch_models.py", "src/policy/dispatch_priority/train.py",
    "src/policy/dispatch_priority/model.py", "src/placement/radical/dispatch.py",
    "src/placement/radical/dispatch.cpp", "src/placement/radical/dispatch_live.py",
    "src/placement/radical/coordinator.py", "src/placement/infrastructure.py",
    "src/placement/simulation.py", "src/placement/executor.py",
    "scripts_cosim/mixed_physics_live.py", "scripts_cosim/workflow_live.py",
]


def load_models(cache, directory):
    result = {}
    frozen = {}
    for arm in ("gnn", "mpoff", "mlp_hand"):
        for seed in (401, 402, 403):
            name = f"{arm}_seed{seed}"
            path = directory / f"{name}.pt"
            sidecar = json.loads(path.with_suffix(".contract.json").read_text())
            if sidecar["contract"] != CONTRACT or sidecar["arm"] != arm or sidecar["seed"] != seed:
                raise ValueError("checkpoint contract mismatch")
            if sidecar["data_metadata_sha256"] != sha(cache / "METADATA.json") or any(sha(ROOT / source) != digest for source, digest in sidecar["sources"].items()):
                raise ValueError("checkpoint source/data mismatch")
            model = PriorityNet(**sidecar["architecture"])
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
            model.eval()
            result[name] = (model, arm == "mlp_hand")
            frozen[name] = {"weights_sha256": sha(path), "contract_sha256": sha(path.with_suffix(".contract.json")),
                            "selected_epoch": sidecar["selected_epoch"]}
    return result, frozen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("preserve prior gate")
    args.out.mkdir(parents=True)
    torch.set_num_threads(1)
    models, frozen = load_models(args.cache_dir, args.models)
    sources = {path: sha(ROOT / path) for path in RUNTIME_SOURCES}
    save(args.out / "frozen_before_test.json", {"models": frozen, "sources": sources,
         "metadata_sha256": sha(args.cache_dir / "METADATA.json")})
    engine = DispatchNative(args.out / "build")
    validation = {}
    for name, (model, hand) in models.items():
        data, problems, _, _, _ = load_split(args.cache_dir, "validation", hand)
        rows = [timed(lambda problem=problem: serve(model, engine, problem, hand), 7) for problem in problems]
        mean = float(np.mean([row["cost"] for row in rows]))
        gain = 100 * (1 - mean / float(data["rule_cost"].mean()))
        p95 = float(np.quantile([row["time_ms"] for row in rows], .95))
        validation[name] = {"mean_cost": mean, "gain_vs_rule_pct": gain, "p95_ms": p95,
                            "passes": gain >= 5 and p95 <= 5, "rows": rows}
        print("VALIDATION", name, mean, gain, p95, flush=True)
    save(args.out / "validation_READ.json", validation)
    data, problems, _, _, _ = load_split(args.cache_dir, "test")
    configure_environment()
    rows = []
    operations = 0
    for index, physical in enumerate(problems):
        if any(sha(ROOT / source) != digest for source, digest in sources.items()):
            raise ValueError("runtime source changed")
        for name, receipt in frozen.items():
            path = args.models / f"{name}.pt"
            if sha(path) != receipt["weights_sha256"] or sha(path.with_suffix(".contract.json")) != receipt["contract_sha256"]:
                raise ValueError("checkpoint changed after freeze")
        seed = int(data["seeds"][index])
        cell = args.out / "test" / str(seed)
        cell.mkdir(parents=True)
        methods = {name: timed(lambda model=model, hand=hand: serve(model, engine, physical, hand), 7)
                   for name, (model, hand) in models.items()}
        methods["srpt128"] = timed(lambda: engine.plan(physical, "srpt:128"), 7)
        methods["srpt"] = timed(lambda: engine.plan(physical, "srpt:0"), 7)
        best = json.loads((args.cache_dir / f"ds_{seed}" / "best.json").read_text())
        methods["teacher"] = {"cost": best["objective"], "plan": [best["placement_plan"], best["priority"]], "unpriced_teacher": True}
        with (cell / "simulation.log").open("w") as log, (cell / "results.jsonl").open("w") as stream:
            for name, receipt in methods.items():
                assignment, priority = np.array(receipt["plan"])
                actual = herosim(physical, assignment, priority, log, seed)
                independent = replay(physical, assignment, priority)
                if abs(actual["total_rtt"] * 1000 - receipt["cost"]) > 1e-6 or independent["objective"] != receipt["cost"]:
                    raise ValueError("live objective mismatch")
                seen = set()
                for task in actual["tasks"]:
                    job, operation = map(int, task["taskType"]["name"][2:].split("_"))
                    seen.add((job, operation))
                    if task["executionNode"] != f"node{assignment[job, operation]}" or abs(task["doneTime"] * 1000 - independent["ends"][job, operation]) > 1e-6:
                        raise ValueError("live task mismatch")
                if len(seen) != assignment.size or len(actual["tasks"]) != assignment.size:
                    raise ValueError("incomplete live workload")
                stream.write(json.dumps({"arm": name, "stats": actual}, allow_nan=False) + "\n")
                operations += assignment.size
        row = {"seed": seed, "methods": methods, "live_sha256": sha(cell / "results.jsonl")}
        save(cell / "read.json", row)
        rows.append(row)
        print("LIVE", len(rows), "/32", flush=True)
    def matrix(arm):
        return np.array([[row["methods"][f"{arm}_seed{seed}"]["cost"] for row in rows] for seed in (401, 402, 403)])
    gnn = matrix("gnn")
    comparisons = {arm: paired_summary(gnn, matrix(arm)) for arm in ("mpoff", "mlp_hand")}
    comparisons["srpt128"] = paired_summary(gnn, np.array([[row["methods"]["srpt128"]["cost"] for row in rows]]))
    comparisons["teacher"] = paired_summary(gnn, np.array([[row["methods"]["teacher"]["cost"] for row in rows]]))
    timings = {name: float(np.quantile([row["methods"][name]["time_ms"] for row in rows], .95))
               for name in [*models, "srpt128", "srpt"]}
    passed = all(comparisons[name]["median_gain_pct"] >= 5 and comparisons[name]["hierarchical_median_ci95_pct"][0] > 0
                 for name in ("mpoff", "mlp_hand", "srpt128")) and all(timings[f"gnn_seed{seed}"] <= 5 for seed in (401, 402, 403))
    report = {"cases": len(rows), "live_runs": len(rows) * 12, "completed_operations": operations,
              "comparisons": comparisons, "p95_ms": timings, "passes": passed,
              "mean_costs": {name: float(np.mean([row["methods"][name]["cost"] for row in rows])) for name in rows[0]["methods"]}}
    save(args.out / "read.json", report)
    print(json.dumps(report, indent=2), flush=True)
    audit(args.out, args.cache_dir, args.models)


def audit(root, cache, model_directory):
    frozen = json.loads((root / "frozen_before_test.json").read_text())
    metadata = json.loads((cache / "METADATA.json").read_text())
    if sha(cache / "METADATA.json") != frozen["metadata_sha256"] or any(sha(ROOT / source) != digest for source, digest in frozen["sources"].items()):
        raise ValueError("gate provenance changed")
    for name, receipt in frozen["models"].items():
        path = model_directory / f"{name}.pt"
        if sha(path) != receipt["weights_sha256"] or sha(path.with_suffix(".contract.json")) != receipt["contract_sha256"]:
            raise ValueError("model changed")
    expected = {row["seed"]: row for row in metadata["cases"] if row["split"] == "test"}
    runs = operations = 0
    seen = set()
    for cell in sorted((root / "test").iterdir()):
        seed = int(cell.name)
        seen.add(seed)
        row = json.loads((cell / "read.json").read_text())
        if sha(cell / "results.jsonl") != row["live_sha256"]:
            raise ValueError("live artifact changed")
        arms = set()
        for line in (cell / "results.jsonl").read_text().splitlines():
            result = json.loads(line)
            name = result["arm"]
            arms.add(name)
            stats = result["stats"]
            receipt = row["methods"][name]
            if abs(stats["total_rtt"] * 1000 - receipt["cost"]) > 1e-6:
                raise ValueError("audit objective mismatch")
            if len(stats["tasks"]) != 48:
                raise ValueError("audit task coverage mismatch")
            runs += 1
            operations += 48
        if arms != set(row["methods"]) or len(arms) != 12:
            raise ValueError("audit arm coverage mismatch")
    if seen != set(expected):
        raise ValueError("test environment coverage mismatch")
    report = json.loads((root / "read.json").read_text())
    if runs != report["live_runs"] or operations != report["completed_operations"]:
        raise ValueError("audit totals mismatch")
    save(root / "AUDIT.json", {"status": "PASS", "runs": runs, "completed_operations": operations,
         "report_sha256": sha(root / "read.json"), "frozen_sha256": sha(root / "frozen_before_test.json")})
    print("AUDIT PASS", runs, operations, flush=True)


if __name__ == "__main__":
    main()
