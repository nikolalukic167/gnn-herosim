"""Matched priority-ranking pilot for fixed-placement mixed execution."""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from scripts_cosim.audit_radical_physics import load_problem
from scripts_cosim.radical_physics_screen import save, sha
from src.placement.radical.dispatch import DispatchNative
from src.policy.dispatch_priority.model import CONTRACT, PriorityNet, features
from src.policy.workflow.train import seed_everything

ROOT = Path(__file__).resolve().parents[3]


def load_split(root, split, hand=False):
    metadata = json.loads((root / "METADATA.json").read_text())
    if metadata["contract"] != CONTRACT or sha(root / f"{split}.npz") != metadata["files"][f"{split}.npz"]:
        raise ValueError("cache contract/fingerprint mismatch")
    if any(sha(ROOT / path) != digest for path, digest in metadata["sources"].items()):
        raise ValueError("data source changed")
    with np.load(root / f"{split}.npz") as stored:
        data = {key: stored[key].copy() for key in stored.files}
    records = {row["seed"]: row for row in metadata["cases"] if row["split"] == split}
    problems = []
    for seed in data["seeds"]:
        path = root / f"ds_{seed}" / "problem.json"
        if sha(path) != records[int(seed)]["sha256"]:
            raise ValueError("physical input changed")
        problems.append(load_problem(path))
    x_key = "xh" if hand else "x"
    return data, problems, torch.tensor(data[x_key]), torch.tensor(data["adj"]), torch.tensor(data["target"])


def ranks(scores):
    order = np.argsort(scores, axis=-1, kind="stable")
    result = np.empty_like(order, dtype=np.int64)
    np.put_along_axis(result, order, np.arange(scores.shape[-1])[None], axis=-1)
    return result


def predict(model, x, adjacency):
    model.eval()
    with torch.inference_mode():
        values = torch.cat([model(x[i:i + 16], adjacency[i:i + 16]) for i in range(0, len(x), 16)])
    return ranks(values.numpy())


def costs(engine, problems, assignments, priorities):
    return np.array([engine.score_priority(problem, assignment, priority.reshape(assignment.shape))[0]
                     for problem, assignment, priority in zip(problems, assignments, priorities)])


def train_epoch(model, optimizer, x, adjacency, target, generator):
    model.train()
    order = torch.randperm(len(x), generator=generator)
    total = 0.0
    for ids in order.split(16):
        loss = nn.functional.smooth_l1_loss(model(x[ids], adjacency[ids]), target[ids])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        total += float(loss.detach()) * len(ids)
    return total / len(x)


def serve(model, engine, problem, hand=False):
    _, assignment = engine.search(problem, __import__("src.placement.radical.environment", fromlist=["initial"]).initial(problem, "affinity"), 64)
    x, adjacency = features(problem, assignment, hand)
    model.eval()
    with torch.inference_mode():
        score = model(torch.from_numpy(x)[None], torch.from_numpy(adjacency)[None]).numpy()
    priority = ranks(score)[0].reshape(assignment.shape)
    cost, _ = engine.score_priority(problem, assignment, priority)
    return cost, np.stack([assignment, priority])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=["gnn", "mpoff", "mlp_hand"], required=True)
    parser.add_argument("--seed", type=int, default=401)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wandb-project", default="herosim-dispatch-priority")
    args = parser.parse_args()
    torch.set_num_threads(1)
    seed_everything(args.seed)
    hand = args.arm == "mlp_hand"
    train, _, x, adjacency, target = load_split(args.cache_dir, "train", hand)
    validation, problems, vx, va, vt = load_split(args.cache_dir, "validation", hand)
    if set(train["seeds"]) & set(validation["seeds"]):
        raise ValueError("overlapping splits")
    architecture = {"feature_dim": x.shape[-1], "hidden": 16, "layers": 2, "mp": args.arm == "gnn"}
    model = PriorityNet(**architecture)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(args.seed)
    engine = DispatchNative(args.output_dir / "build")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / f"{args.arm}_seed{args.seed}.pt"
    if checkpoint.exists():
        raise ValueError("preserve prior checkpoint")
    source_paths = ["src/policy/dispatch_priority/train.py", "src/policy/dispatch_priority/model.py", "run_experiment.py"]
    sources = {path: sha(ROOT / path) for path in source_paths}
    import wandb
    run = wandb.init(project=args.wandb_project, name=f"dispatch-{args.arm}-{args.seed}", mode="offline",
                     config={"arm": args.arm, "seed": args.seed, "contract": CONTRACT,
                             "architecture": architecture, "epochs": args.epochs})
    if run is None:
        raise RuntimeError("missing W&B run")
    history = []
    best = float("inf")
    selected = -1
    started = time.monotonic()
    try:
        for epoch in range(args.epochs + 1):
            loss = None if epoch == 0 else train_epoch(model, optimizer, x, adjacency, target, generator)
            prediction = predict(model, vx, va)
            served = costs(engine, problems, validation["assignment"], prediction)
            metric = float(served.mean())
            row = {"epoch": epoch, "validation_mean_served_rtt_ms": metric,
                   "validation_teacher_rtt_ms": float(validation["teacher_cost"].mean()),
                   "validation_rule_rtt_ms": float(validation["rule_cost"].mean()),
                   "validation_rank_mae": float(np.mean(np.abs(prediction / 47 - validation["target"]))) }
            if loss is not None:
                row["train_loss"] = loss
            history.append(row)
            run.log(row, step=epoch)
            if metric < best:
                best, selected = metric, epoch
                torch.save(model.state_dict(), checkpoint)
                save(checkpoint.with_suffix(".contract.json"), {
                    "contract": CONTRACT, "physics": "mixed_execution_v1", "dispatch": "mixed_ready_priority_v1",
                    "arm": args.arm, "seed": args.seed, "architecture": architecture,
                    "selected_epoch": epoch, "selection": "minimum_validation_mean_direct_priority_cost",
                    "data_metadata_sha256": sha(args.cache_dir / "METADATA.json"), "sources": sources,
                    "wandb_mode": "offline", "wandb_run_dir": run.dir,
                    "torch_seeded": True, "deterministic_algorithms": True,
                })
            if epoch % 5 == 0:
                print(args.arm, args.seed, row, flush=True)
        save(checkpoint.with_suffix(".training.json"), {"history": history, "selected_epoch": selected,
             "selected_metric": best, "wall_s": time.monotonic() - started, "wandb_run_dir": run.dir})
        run.summary.update({"selected_epoch": selected, "selected_validation_rtt_ms": best})
    finally:
        run.finish()


if __name__ == "__main__":
    main()
