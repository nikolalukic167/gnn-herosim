"""Qualify bounded action-value overrides of the strong srpt:128 schedule."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts_cosim.radical_physics_screen import save, sha
from src.placement.radical.dispatch import DispatchNative
from src.placement.radical.dispatch_state import materialize
from src.policy.dispatch_priority.model import features
from src.policy.dispatch_value.model import CONTRACT, ReadySetNet
from src.policy.dispatch_value.train import load_split

ROOT = Path(__file__).resolve().parents[1]


def load_models(cache, directory):
    result = {}
    for arm in ("gnn", "mpoff", "mlp_hand"):
        for seed in (601, 602, 603):
            name = f"{arm}_seed{seed}"
            path = directory / f"{name}.pt"
            sidecar = json.loads(path.with_suffix(".contract.json").read_text())
            if sidecar["contract"] != CONTRACT or sidecar["data_metadata_sha256"] != sha(cache / "METADATA.json"):
                raise ValueError("checkpoint contract mismatch")
            if any(sha(ROOT / source) != digest for source, digest in sidecar["sources"].items()):
                raise ValueError("checkpoint source mismatch")
            model = ReadySetNet(**sidecar["architecture"])
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
            model.eval()
            result[name] = (model, arm == "mlp_hand")
    return result


def scorer(model, encoded):
    first, second = model.head[0], model.head[2]
    w1, b1 = first.weight.detach().numpy(), first.bias.detach().numpy()
    w2, b2 = second.weight.detach().numpy(), second.bias.detach().numpy()
    def scores(dynamic, feasible):
        joined = np.concatenate([encoded, dynamic], axis=-1)
        hidden = np.maximum(joined @ w1.T + b1, 0)
        logits = (hidden @ w2.T + b2).reshape(-1)
        logits[~feasible] = -np.inf
        return logits
    return scores


def hybrid(problem, assignment, base_priority, scores, threshold, cap):
    base = np.asarray(base_priority).reshape(-1)
    overrides = 0
    def choose(dynamic, feasible):
        nonlocal overrides
        candidates = np.flatnonzero(feasible)
        fallback = int(candidates[np.argmin(base[candidates])])
        logits = scores(dynamic, feasible)
        proposed = int(np.argmax(logits))
        margin = float(logits[proposed] - logits[fallback])
        if proposed != fallback and overrides < cap and margin >= threshold:
            overrides += 1
            return proposed
        return fallback
    priority, _ = materialize(problem, assignment, choose)
    return priority, overrides


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("preserve earlier screen")
    torch.set_num_threads(1)
    engine = DispatchNative(args.out.parent / "residual_build")
    loaded = load_models(args.cache_dir, args.models)
    data, problems, _ = load_split(args.cache_dir, "validation")
    thresholds = [0.0, .02, .05, .1, .2, .5, 1.0, 2.0]
    caps = [1, 2, 4, 8, 16, 48]
    base_rows = [engine.plan(problem, "srpt:128") for problem in problems]
    base_mean = float(np.mean([row[0] for row in base_rows]))
    results = {}
    for name, (model, hand) in loaded.items():
        grid = {(threshold, cap): [] for threshold in thresholds for cap in caps}
        override_grid = {(threshold, cap): [] for threshold in thresholds for cap in caps}
        for problem, (_, plan) in zip(problems, base_rows):
            assignment, base_priority = np.asarray(plan)
            x, adjacency = features(problem, assignment, hand)
            with torch.inference_mode():
                encoded = model.encode(torch.from_numpy(x)[None],
                                       torch.from_numpy(adjacency)[None])[0].numpy()
            scores = scorer(model, encoded)
            for setting in grid:
                priority, overrides = hybrid(problem, assignment, base_priority, scores, *setting)
                grid[setting].append(engine.score_priority(problem, assignment, priority)[0])
                override_grid[setting].append(overrides)
        choices = []
        for setting, costs in grid.items():
            threshold, cap = setting
            mean = float(np.mean(costs))
            choices.append({"threshold": threshold, "cap": cap, "mean_cost": mean,
                            "gain_vs_srpt128_pct": 100 * (1 - mean / base_mean),
                            "mean_overrides": float(np.mean(override_grid[setting]))})
        results[name] = min(choices, key=lambda row: (row["mean_cost"], row["cap"], -row["threshold"]))
        print(name, results[name], flush=True)
    arm_best = {arm: min((row for name, row in results.items() if name.startswith(arm)),
                         key=lambda row: row["mean_cost"])
                for arm in ("gnn", "mpoff", "mlp_hand")}
    gnn = arm_best["gnn"]
    qualifies = (gnn["gain_vs_srpt128_pct"] >= 2 and
                 all(100 * (1 - gnn["mean_cost"] / arm_best[arm]["mean_cost"]) >= 1
                     for arm in ("mpoff", "mlp_hand")))
    report = {"status": "ADVANCE" if qualifies else "NO_GO", "base_mean": base_mean,
              "models": results, "arm_best": arm_best,
              "qualification": "GNN >=2% over srpt128 and >=1% over each residual learned control"}
    save(args.out, report)
    print(json.dumps({"status": report["status"], "base_mean": base_mean,
                      "arm_best": arm_best}, indent=2), flush=True)


if __name__ == "__main__":
    main()
