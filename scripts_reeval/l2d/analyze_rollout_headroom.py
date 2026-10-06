"""Read the frozen one-step L2D rollout headroom screen."""
from __future__ import annotations

import argparse
import json

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--six", required=True)
    p.add_argument("--ten", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=400)
    a = p.parse_args()
    rng = np.random.default_rng(0)
    sizes = []
    for path, n in ((a.six, 6), (a.ten, 10)):
        data = json.load(open(path))
        if (data["n_j"], data["n_m"], data["seed"], data["n_test"]) != (n, n, a.seed, 16):
            raise ValueError(f"wrong frozen input contract: {path}")
        records = data["records"]
        if len(records) != 16 or sorted(r["case"] for r in records) != list(range(16)):
            raise ValueError(f"incomplete cases: {path}")
        hand = np.array([r["best_rule_makespan"] for r in records])
        rollout = np.array([min(v["makespan"] for v in r["rollout"].values()) for r in records])
        gains = 100 * (hand - rollout) / hand
        sampled = gains[rng.integers(0, 16, size=(100_000, 16))]
        ci = [float(v) for v in np.quantile(np.median(sampled, axis=1), [0.025, 0.975])]
        sizes.append({"size": n, "test_data_sha256": data["test_data_sha256"],
                      "hand_mean_makespan": float(hand.mean()),
                      "rollout_mean_makespan": float(rollout.mean()),
                      "paired_gains_percent": [float(v) for v in gains],
                      "median_gain_percent": float(np.median(gains)),
                      "mean_gain_percent": float(gains.mean()),
                      "positive_cases": int((gains > 0).sum()),
                      "bootstrap_median_95_ci": ci,
                      "median_portfolio_wall_ms": float(np.median([r["portfolio_wall_ms"] for r in records])),
                      "median_rollout_wall_ms": float(np.median([
                          sum(v["wall_ms"] for v in r["rollout"].values()) for r in records]))})
    result = {"status": "GO" if all(s["median_gain_percent"] >= 5 and s["positive_cases"] >= 12
                                    for s in sizes) else "NO_GO",
              "screen_seed": a.seed, "n_instances_per_size": 16,
              "bootstrap_seed": 0, "bootstrap_resamples": 100_000,
              "sizes": sizes}
    with open(a.out, "w") as fh:
        json.dump(result, fh, indent=1)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
