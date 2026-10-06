"""Frozen-model rank and one-budgeted-refinement diagnostic; no new primary arm."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from scripts_cosim.radical_physics_screen import save, sha
from src.placement.radical.dispatch import DispatchNative
from src.policy.dispatch_priority.model import PriorityNet
from src.policy.dispatch_priority.train import load_split, serve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    args = parser.parse_args()
    frozen = json.loads((args.gate / "frozen_before_test.json").read_text())
    engine = DispatchNative(args.gate / "diagnostic_build")
    result = {}
    for arm in ("gnn", "mpoff", "mlp_hand"):
        data, problems, _, _, target = load_split(args.cache_dir, "test", arm == "mlp_hand")
        for seed in (401, 402, 403):
            name = f"{arm}_seed{seed}"
            path = args.models / f"{name}.pt"
            if sha(path) != frozen["models"][name]["weights_sha256"]:
                raise ValueError("frozen weights changed")
            sidecar = json.loads(path.with_suffix(".contract.json").read_text())
            model = PriorityNet(**sidecar["architecture"])
            model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
            model.eval()
            raw = []
            refined = []
            timings = []
            rank_mae = []
            for index, physical in enumerate(problems):
                started = time.perf_counter()
                cost, plan = serve(model, engine, physical, arm == "mlp_hand")
                refined_cost, _ = engine.search_priority(physical, plan[0], plan[1], 64)
                timings.append(1000 * (time.perf_counter() - started))
                raw.append(cost)
                refined.append(refined_cost)
                rank_mae.append(float(np.abs(plan[1].reshape(-1) / 47 - target[index].numpy()).mean()))
            result[name] = {"raw_mean_cost": float(np.mean(raw)), "refine64_mean_cost": float(np.mean(refined)),
                            "refine64_p95_ms": float(np.quantile(timings, .95)), "teacher_rank_mae": float(np.mean(rank_mae)),
                            "gain_vs_srpt128_pct": float(100 * (1 - np.mean(refined) / data["rule_cost"].mean()))}
    save(args.gate / "decoder_diagnostic.json", {"status": "PASS", "models": result,
         "scope": "posthoc frozen-weight diagnostic; predicted priority plus 64 swaps, no live-policy or primary claim",
         "source_sha256": sha(Path(__file__))})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
