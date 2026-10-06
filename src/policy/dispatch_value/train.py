"""Outcome-aware feasible-action learner with a NumPy dynamic serving head."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from scripts_cosim.audit_radical_physics import load_problem
from scripts_cosim.radical_physics_screen import save, sha
from src.placement.radical.dispatch import DispatchNative
from src.placement.radical.dispatch_state import materialize
from src.placement.radical.environment import initial
from src.policy.dispatch_priority.model import features
from src.policy.dispatch_value.model import CONTRACT, ReadySetNet
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
        if sha(path) != records[int(seed)]["problem_sha256"]:
            raise ValueError("physical input changed")
        problems.append(load_problem(path))
    key = "xh" if hand else "x"
    tensors = [torch.tensor(data[name]) for name in (key, "adj", "dynamic", "feasible",
                                                      "action_cost", "rule_cost")]
    return data, problems, tensors


def listwise_loss(logits, feasible, action_cost, rule_cost):
    finite_cost = torch.nan_to_num(action_cost, nan=float("inf"))
    minimum = finite_cost.min(-1, keepdim=True).values
    temperature = rule_cost[:, None, None] * .01
    target_logits = -(finite_cost - minimum) / temperature
    target_logits = target_logits.masked_fill(~feasible, -1e9)
    target = torch.softmax(target_logits, dim=-1)
    return -(target * torch.log_softmax(logits, dim=-1)).sum(-1).mean()


def train_epoch(model, optimizer, tensors, generator):
    x, adjacency, dynamic, feasible, action_cost, rule_cost = tensors
    order = torch.randperm(len(x), generator=generator)
    total = 0.0
    model.train()
    for ids in order.split(4):
        logits = model(x[ids], adjacency[ids], dynamic[ids], feasible[ids])
        loss = listwise_loss(logits, feasible[ids], action_cost[ids], rule_cost[ids])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        total += float(loss.detach()) * len(ids)
    return total / len(x)


def action_accuracy(model, tensors):
    x, adjacency, dynamic, feasible, action_cost, _ = tensors
    model.eval()
    with torch.inference_mode():
        prediction = torch.cat([model(x[i:i + 8], adjacency[i:i + 8], dynamic[i:i + 8],
                                      feasible[i:i + 8]).argmax(-1)
                                for i in range(0, len(x), 8)])
    best = torch.nan_to_num(action_cost, nan=float("inf")).argmin(-1)
    multi = feasible.sum(-1) > 1
    return float((prediction[multi] == best[multi]).float().mean())


def numpy_head(model, encoded):
    first, second = model.head[0], model.head[2]
    w1 = first.weight.detach().numpy()
    b1 = first.bias.detach().numpy()
    w2 = second.weight.detach().numpy()
    b2 = second.bias.detach().numpy()
    def choose(dynamic, feasible):
        joined = np.concatenate([encoded, dynamic], axis=-1)
        hidden = np.maximum(joined @ w1.T + b1, 0)
        logits = (hidden @ w2.T + b2).reshape(-1)
        logits[~feasible] = -np.inf
        return int(np.argmax(logits))
    return choose


def serve(model, engine, physical, hand=False):
    assignment = engine.search(physical, initial(physical, "affinity"), 64)[1]
    x, adjacency = features(physical, assignment, hand)
    model.eval()
    with torch.inference_mode():
        encoded = model.encode(torch.from_numpy(x)[None], torch.from_numpy(adjacency)[None])[0].numpy()
    priority, _ = materialize(physical, assignment, numpy_head(model, encoded))
    cost = engine.score_priority(physical, assignment, priority)[0]
    return cost, np.stack([assignment, priority])


def validation_cost(model, engine, problems, hand):
    return float(np.mean([serve(model, engine, physical, hand)[0] for physical in problems]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=["gnn", "mpoff", "mlp_hand"], required=True)
    parser.add_argument("--seed", type=int, default=601)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wandb-project", default="herosim-dispatch-value")
    args = parser.parse_args()
    torch.set_num_threads(1)
    seed_everything(args.seed)
    hand = args.arm == "mlp_hand"
    train, _, train_tensors = load_split(args.cache_dir, "train", hand)
    validation, problems, validation_tensors = load_split(args.cache_dir, "validation", hand)
    architecture = {"feature_dim": train_tensors[0].shape[-1], "dynamic_dim": 13,
                    "hidden": 16, "layers": 2, "mp": args.arm == "gnn"}
    model = ReadySetNet(**architecture)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / f"{args.arm}_seed{args.seed}.pt"
    if checkpoint.exists():
        raise ValueError("preserve prior checkpoint")
    engine = DispatchNative(args.output_dir / "build")
    source_paths = ["src/policy/dispatch_value/train.py", "src/policy/dispatch_value/model.py",
                    "src/placement/radical/dispatch_value.py",
                    "src/placement/radical/dispatch_state.py", "run_experiment.py"]
    sources = {path: sha(ROOT / path) for path in source_paths}
    import wandb
    run = wandb.init(project=args.wandb_project, name=f"dispatch-value-{args.arm}-{args.seed}",
                     mode="offline", config={"arm": args.arm, "seed": args.seed,
                     "contract": CONTRACT, "architecture": architecture, "epochs": args.epochs})
    if run is None:
        raise RuntimeError("missing W&B run")
    history, best, selected = [], float("inf"), -1
    started = time.monotonic()
    try:
        for epoch in range(args.epochs + 1):
            loss = None if epoch == 0 else train_epoch(model, optimizer, train_tensors, generator)
            accuracy = action_accuracy(model, validation_tensors)
            metric = validation_cost(model, engine, problems, hand) if epoch % 5 == 0 else None
            row = {"epoch": epoch, "validation_best_action_accuracy": accuracy,
                   "validation_teacher_rtt_ms": float(validation["teacher_cost"].mean()),
                   "validation_rule_rtt_ms": float(validation["rule_cost"].mean())}
            if loss is not None:
                row["train_loss"] = loss
            if metric is not None:
                row["validation_mean_served_rtt_ms"] = metric
                if metric < best:
                    best, selected = metric, epoch
                    torch.save(model.state_dict(), checkpoint)
                    save(checkpoint.with_suffix(".contract.json"), {
                        "contract": CONTRACT, "physics": "mixed_execution_v1",
                        "dispatch": "mixed_ready_action_value_v1", "arm": args.arm,
                        "seed": args.seed, "architecture": architecture,
                        "selected_epoch": epoch,
                        "selection": "minimum_validation_mean_action_value_policy_cost",
                        "data_metadata_sha256": sha(args.cache_dir / "METADATA.json"),
                        "sources": sources, "wandb_mode": "offline", "wandb_run_dir": run.dir})
            history.append(row)
            run.log(row, step=epoch)
            if epoch % 5 == 0:
                print(args.arm, args.seed, row, flush=True)
        save(checkpoint.with_suffix(".training.json"), {"history": history,
             "selected_epoch": selected, "selected_metric": best,
             "wall_s": time.monotonic() - started, "wandb_run_dir": run.dir})
        run.summary.update({"selected_epoch": selected, "selected_validation_rtt_ms": best})
    finally:
        run.finish()


if __name__ == "__main__":
    main()
