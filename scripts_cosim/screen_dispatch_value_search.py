"""Screen exact-accepted model proposals over the srpt:128 schedule."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts_cosim.radical_physics_screen import save
from scripts_cosim.screen_dispatch_value_residual import load_models, scorer
from src.placement.radical.dispatch import DispatchNative
from src.placement.radical.dispatch_state import trace_priority
from src.policy.dispatch_priority.model import features
from src.policy.dispatch_value.train import load_split


def proposed_rank(order, step, candidate, size):
    prefix = order[:step]
    used = set(prefix)
    remaining = [index for index in order if index not in used and index != candidate]
    proposal = [*prefix, int(candidate), *remaining]
    rank = np.empty(size, dtype=np.int64)
    rank[np.asarray(proposal)] = np.arange(size)
    return rank


def improve(engine, problem, assignment, initial_priority, scores, rounds, width):
    priority = np.asarray(initial_priority).copy()
    cost = engine.score_priority(problem, assignment, priority)[0]
    evaluations = accepted = 0
    for _ in range(rounds):
        realized, states = trace_priority(problem, assignment, priority)
        order = [state["target"] for state in states]
        proposals = []
        for step, state in enumerate(states):
            logits = scores(state["dynamic"], state["feasible"])
            selected = int(np.argmax(logits))
            current = order[step]
            if selected == current:
                continue
            margin = float(logits[selected] - logits[current])
            candidate = proposed_rank(order, step, selected, assignment.size)
            proposals.append((margin, candidate.reshape(assignment.shape)))
        best_cost, best_priority = cost, priority
        seen = set()
        for _, candidate in sorted(proposals, key=lambda row: row[0], reverse=True):
            key = candidate.tobytes()
            if key in seen:
                continue
            seen.add(key)
            candidate_cost = engine.score_priority(problem, assignment, candidate)[0]
            evaluations += 1
            if candidate_cost < best_cost:
                best_cost, best_priority = candidate_cost, candidate
            if len(seen) == width:
                break
        if best_cost >= cost:
            break
        cost, priority = best_cost, best_priority.copy()
        accepted += 1
    return cost, priority, evaluations, accepted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("preserve earlier screen")
    torch.set_num_threads(1)
    engine = DispatchNative(args.out.parent / "proposal_search_build")
    learned = load_models(args.cache_dir, args.models)
    _, problems, _ = load_split(args.cache_dir, "validation")
    settings = [(1, 4), (2, 4), (4, 4), (2, 8), (4, 8)]
    baseline = [engine.plan(problem, "srpt:128") for problem in problems]
    base_mean = float(np.mean([row[0] for row in baseline]))
    result = {}
    for name, (model, hand) in learned.items():
        rows = {setting: [] for setting in settings}
        for problem, (_, plan) in zip(problems, baseline):
            assignment, priority = np.asarray(plan)
            x, adjacency = features(problem, assignment, hand)
            with torch.inference_mode():
                encoded = model.encode(torch.from_numpy(x)[None],
                                       torch.from_numpy(adjacency)[None])[0].numpy()
            scores = scorer(model, encoded)
            for setting in settings:
                cost, _, evaluations, accepted = improve(
                    engine, problem, assignment, priority, scores, *setting)
                rows[setting].append({"cost": cost, "evaluations": evaluations,
                                      "accepted": accepted})
        choices = []
        for (rounds, width), values in rows.items():
            mean = float(np.mean([row["cost"] for row in values]))
            choices.append({"rounds": rounds, "width": width, "mean_cost": mean,
                            "gain_vs_srpt128_pct": 100 * (1 - mean / base_mean),
                            "mean_evaluations": float(np.mean([row["evaluations"] for row in values])),
                            "mean_accepted": float(np.mean([row["accepted"] for row in values]))})
        result[name] = min(choices, key=lambda row: (row["mean_cost"], row["mean_evaluations"]))
        print(name, result[name], flush=True)
    arm_best = {arm: min((row for name, row in result.items() if name.startswith(arm)),
                         key=lambda row: row["mean_cost"])
                for arm in ("gnn", "mpoff", "mlp_hand")}
    gnn = arm_best["gnn"]
    qualifies = (gnn["gain_vs_srpt128_pct"] >= 5 and
                 all(100 * (1 - gnn["mean_cost"] / arm_best[arm]["mean_cost"]) >= 1
                     for arm in ("mpoff", "mlp_hand")))
    report = {"status": "ADVANCE" if qualifies else "NO_GO", "base_mean": base_mean,
              "models": result, "arm_best": arm_best,
              "qualification": "GNN >=5% over srpt128 and >=1% over each matched proposal control"}
    save(args.out, report)
    print(json.dumps({"status": report["status"], "base_mean": base_mean,
                      "arm_best": arm_best}, indent=2), flush=True)


if __name__ == "__main__":
    main()
