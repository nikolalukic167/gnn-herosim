"""Freeze, time, live-evaluate, and audit ready-set dispatch learners."""
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
from src.policy.dispatch_state.model import CONTRACT, ReadySetNet
from src.policy.dispatch_state.train import load_split, serve

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ["scripts_cosim/evaluate_dispatch_state_models.py",
           "src/policy/dispatch_state/train.py", "src/policy/dispatch_state/model.py",
           "src/placement/radical/dispatch_state.py", "src/placement/radical/dispatch.py",
           "src/placement/radical/dispatch.cpp", "src/placement/radical/dispatch_live.py",
           "src/placement/radical/coordinator.py", "src/placement/infrastructure.py",
           "src/placement/simulation.py", "src/placement/executor.py",
           "scripts_cosim/mixed_physics_live.py", "scripts_cosim/workflow_live.py"]


def load_models(cache, directory):
    models, frozen = {}, {}
    for arm in ("gnn", "mpoff", "mlp_hand"):
        for seed in (501, 502, 503):
            name = f"{arm}_seed{seed}"
            path = directory / f"{name}.pt"
            sidecar = json.loads(path.with_suffix(".contract.json").read_text())
            if sidecar["contract"] != CONTRACT or sidecar["arm"] != arm or sidecar["seed"] != seed:
                raise ValueError("checkpoint contract mismatch")
            if sidecar["data_metadata_sha256"] != sha(cache / "METADATA.json") or any(
                    sha(ROOT / source) != digest for source, digest in sidecar["sources"].items()):
                raise ValueError("checkpoint source/data mismatch")
            model = ReadySetNet(**sidecar["architecture"])
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
            model.eval()
            models[name] = (model, arm == "mlp_hand")
            frozen[name] = {"weights_sha256": sha(path),
                            "contract_sha256": sha(path.with_suffix(".contract.json")),
                            "selected_epoch": sidecar["selected_epoch"]}
    return models, frozen


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
    sources = {path: sha(ROOT / path) for path in RUNTIME}
    save(args.out / "frozen_before_test.json", {"models": frozen, "sources": sources,
         "metadata_sha256": sha(args.cache_dir / "METADATA.json")})
    engine = DispatchNative(args.out / "build")
    validation = {}
    for name, (model, hand) in models.items():
        data, problems, _ = load_split(args.cache_dir, "validation", hand)
        results = [timed(lambda physical=physical: serve(model, engine, physical, hand), 7)
                   for physical in problems]
        mean = float(np.mean([row["cost"] for row in results]))
        gain = 100 * (1 - mean / float(data["rule_cost"].mean()))
        p95 = float(np.quantile([row["time_ms"] for row in results], .95))
        validation[name] = {"mean_cost": mean, "gain_vs_rule_pct": gain, "p95_ms": p95,
                            "passes": gain >= 5 and p95 <= 5}
        print("VALIDATION", name, mean, gain, p95, flush=True)
    save(args.out / "validation_READ.json", validation)
    data, problems, _ = load_split(args.cache_dir, "test")
    configure_environment()
    rows, operations = [], 0
    for index, physical in enumerate(problems):
        if any(sha(ROOT / source) != digest for source, digest in sources.items()):
            raise ValueError("runtime source changed")
        seed = int(data["seeds"][index])
        cell = args.out / "test" / str(seed)
        cell.mkdir(parents=True)
        methods = {name: timed(lambda model=model, hand=hand: serve(model, engine, physical, hand), 7)
                   for name, (model, hand) in models.items()}
        methods["srpt128"] = timed(lambda: engine.plan(physical, "srpt:128"), 7)
        methods["srpt"] = timed(lambda: engine.plan(physical, "srpt:0"), 7)
        best = json.loads((args.cache_dir / f"ds_{seed}" / "best.json").read_text())
        methods["teacher"] = {"cost": best["objective"],
                              "plan": [best["placement_plan"], best["priority"]],
                              "unpriced_teacher": True}
        with (cell / "simulation.log").open("w") as log, (cell / "results.jsonl").open("w") as stream:
            for name, receipt in methods.items():
                assignment, priority = np.asarray(receipt["plan"])
                actual = herosim(physical, assignment, priority, log, seed)
                independent = replay(physical, assignment, priority)
                if abs(actual["total_rtt"] * 1000 - receipt["cost"]) > 1e-6 or independent["objective"] != receipt["cost"]:
                    raise ValueError("live objective mismatch")
                if len(actual["tasks"]) != assignment.size:
                    raise ValueError("incomplete live workload")
                stream.write(json.dumps({"arm": name, "stats": actual}, allow_nan=False) + "\n")
                operations += assignment.size
        row = {"seed": seed, "methods": methods, "live_sha256": sha(cell / "results.jsonl")}
        save(cell / "read.json", row)
        rows.append(row)
        print("LIVE", len(rows), "/32", flush=True)
    def matrix(arm):
        return np.asarray([[row["methods"][f"{arm}_seed{seed}"]["cost"] for row in rows]
                           for seed in (501, 502, 503)])
    gnn = matrix("gnn")
    comparisons = {arm: paired_summary(gnn, matrix(arm)) for arm in ("mpoff", "mlp_hand")}
    comparisons["srpt128"] = paired_summary(
        gnn, np.asarray([[row["methods"]["srpt128"]["cost"] for row in rows]]))
    comparisons["teacher"] = paired_summary(
        gnn, np.asarray([[row["methods"]["teacher"]["cost"] for row in rows]]))
    timings = {name: float(np.quantile([row["methods"][name]["time_ms"] for row in rows], .95))
               for name in [*models, "srpt128", "srpt"]}
    passed = all(comparisons[name]["median_gain_pct"] >= 5 and
                 comparisons[name]["hierarchical_median_ci95_pct"][0] > 0
                 for name in ("mpoff", "mlp_hand", "srpt128")) and all(
                     timings[f"gnn_seed{seed}"] <= 5 for seed in (501, 502, 503))
    report = {"cases": len(rows), "live_runs": len(rows) * 12,
              "completed_operations": operations, "comparisons": comparisons,
              "p95_ms": timings, "passes": passed,
              "mean_costs": {name: float(np.mean([row["methods"][name]["cost"] for row in rows]))
                             for name in rows[0]["methods"]}}
    save(args.out / "read.json", report)
    print(json.dumps(report, indent=2), flush=True)
    audit(args.out, sources, frozen)


def audit(root, sources, frozen):
    report = json.loads((root / "read.json").read_text())
    if any(sha(ROOT / source) != digest for source, digest in sources.items()):
        raise ValueError("source changed after freeze")
    runs = operations = 0
    for cell in sorted((root / "test").iterdir()):
        row = json.loads((cell / "read.json").read_text())
        if sha(cell / "results.jsonl") != row["live_sha256"]:
            raise ValueError("live artifact changed")
        records = [json.loads(line) for line in (cell / "results.jsonl").read_text().splitlines()]
        if len(records) != 12 or len({record["arm"] for record in records}) != 12:
            raise ValueError("arm coverage mismatch")
        runs += len(records)
        operations += sum(len(record["stats"]["tasks"]) for record in records)
    if runs != report["live_runs"] or operations != report["completed_operations"]:
        raise ValueError("audit totals mismatch")
    save(root / "AUDIT.json", {"status": "PASS", "runs": runs,
         "completed_operations": operations, "report_sha256": sha(root / "read.json")})
    print("AUDIT PASS", runs, operations, flush=True)


if __name__ == "__main__":
    main()
