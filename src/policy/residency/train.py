"""Config-driven matched training for measured-residency placement."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from src.placement.residency import CONTRACT, evaluate
from src.policy.residency.model import ResidencyNet, decode_many


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def train_epoch(model, optimizer, x, q, generator, batch_size=128):
    order = torch.randperm(len(x), generator=generator)
    total = 0.
    model.train()
    for indices in order.split(batch_size):
        batch = x[indices]
        step = batch[:, :, 0, 9].argmax(-1)
        logits = model(batch)[torch.arange(len(indices)), step]
        target = torch.softmax(-q[indices], -1)
        loss = -(target * logits.log_softmax(-1)).sum(-1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        total += float(loss.detach()) * len(indices)
    return total / len(x)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=["gnn", "mpoff", "hand_mlp"], required=True)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--wandb-project", default="herosim-residency-placement")
    parser.add_argument("--wandb-mode", choices=["offline"], default="offline")
    args = parser.parse_args()
    torch.set_num_threads(1)
    seed_everything(args.seed)
    metadata = json.loads((args.cache_dir / "METADATA.json").read_text())
    if metadata["contract"] != CONTRACT:
        raise ValueError("wrong dataset physics")
    for name in ["train.npz", "train.json", "validation.json"]:
        if sha(args.cache_dir / name) != metadata["files"][name]:
            raise ValueError("dataset fingerprint mismatch")
    if set(metadata["ids"]["train"]) & set(metadata["ids"]["validation"]):
        raise ValueError("train/validation leakage")
    with np.load(args.cache_dir / "train.npz") as data:
        x, q = torch.from_numpy(data["x"]), torch.from_numpy(data["q"])
    if not torch.isfinite(x).all() or not torch.isfinite(q).all():
        raise ValueError("non-finite training data")
    validation = json.loads((args.cache_dir / "validation.json").read_text())
    architecture = {"arm": args.arm, "hidden": 48, "layers": 2}
    model = ResidencyNet(**architecture)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    generator = torch.Generator().manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / f"{args.arm}_seed{args.seed}.pt"
    if checkpoint.exists():
        raise ValueError("checkpoint already exists")
    import wandb
    run = wandb.init(project=args.wandb_project, name=os.environ.get("WANDB_RUN_NAME"),
                     mode=args.wandb_mode, config={"arm": args.arm, "seed": args.seed,
                     "contract": CONTRACT, "architecture": architecture, "chance_ce": float(np.log(3)),
                     "cache_metadata_sha256": sha(args.cache_dir / "METADATA.json")})
    if run is None:
        raise RuntimeError("W&B initialization failed")
    history, best = [], float("inf")
    started = time.perf_counter()
    try:
        for epoch in range(args.epochs + 1):
            loss = None if epoch == 0 else train_epoch(model, optimizer, x, q, generator)
            plans = decode_many(model, validation)
            costs = np.asarray([evaluate(c, p)[0] for c, p in zip(validation, plans)])
            reference = np.asarray([c["optimal_cost"] for c in validation])
            metric = float(np.mean(costs / reference - 1))
            row = {"epoch": epoch, "validation_regret": metric, "train_soft_ce": loss,
                   "chance_ce": float(np.log(3))}
            history.append(row)
            run.log(row, step=epoch)
            if metric < best:
                best = metric
                torch.save(model.state_dict(), checkpoint)
                sidecar = {"contract": CONTRACT, "architecture": architecture, "seed": args.seed,
                           "torch_seeded": True, "deterministic_algorithms": True,
                           "selected_epoch": epoch, "validation_regret": metric,
                           "cache_metadata_sha256": sha(args.cache_dir / "METADATA.json"),
                           "sources": {str(p.relative_to(Path(__file__).resolve().parents[3])): sha(p)
                                       for p in [Path(__file__), Path(__file__).with_name("model.py"),
                                                 Path(__file__).resolve().parents[2] / "placement/residency.py"]},
                           "wandb_mode": "offline", "wandb_run_dir": run.dir}
                checkpoint.with_suffix(".contract.json").write_text(json.dumps(sidecar, indent=2) + "\n")
            if epoch % 5 == 0:
                print(args.arm, args.seed, row, flush=True)
        checkpoint.with_suffix(".training.json").write_text(json.dumps({"history": history,
                     "wall_s": time.perf_counter() - started, "best": best}, indent=2) + "\n")
    finally:
        run.finish()


if __name__ == "__main__":
    main()
