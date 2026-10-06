"""Read the frozen L2D learning-efficiency gate from per-run test JSONs."""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


def confidence_interval(values, rng, n_resamples=100_000):
    indices = rng.integers(0, len(values), size=(n_resamples, len(values)))
    means = values[indices].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    by_key = {}
    hashes = {}
    files = sorted(glob.glob(os.path.join(a.runs, "l2d_*", "fixed_budget_test_seed302.json")))
    for path in files:
        records = json.load(open(path))
        if len(records) != 2 or {r["update"] for r in records} != {1000, 3000}:
            raise ValueError(f"missing milestone: {path}")
        for r in records:
            size = r["n_j"]
            key = (size, r["arm"], r["study_seed"], r["update"])
            if key in by_key or r["n_j"] != r["n_m"] or r["test_seed"] != 302 or r["n_test"] != 100:
                raise ValueError(f"duplicate or invalid record: {path} {key}")
            if len(r["makespans"]) != 100 or not np.isclose(np.mean(r["makespans"]), r["mean_makespan"]):
                raise ValueError(f"invalid makespans: {path}")
            if r["train_instances"] != r["update"] * 4:
                raise ValueError(f"invalid training budget: {path}")
            hashes.setdefault(size, set()).add(r["test_data_sha256"])
            by_key[key] = r
    expected = {(size, arm, seed, update)
                for size in (6, 10) for arm in ("gnn", "mpoff", "mlp_t1")
                for seed in range(8) for update in (1000, 3000)}
    if set(by_key) != expected or len(files) != 48 or any(len(hashes[size]) != 1 for size in (6, 10)):
        raise ValueError(f"incomplete or mismatched gate: got {len(files)} files, {len(by_key)} records, hashes={hashes}")
    rng = np.random.default_rng(0)
    comparisons = []
    for size in (6, 10):
        for control in ("mpoff", "mlp_t1"):
            for g_update, c_update in ((1000, 3000), (1000, 1000), (3000, 3000)):
                g = np.array([by_key[(size, "gnn", s, g_update)]["mean_makespan"] for s in range(8)])
                c = np.array([by_key[(size, control, s, c_update)]["mean_makespan"] for s in range(8)])
                gains = 100 * (c - g) / c
                comparisons.append({"size": size, "control": control, "gnn_update": g_update,
                                    "control_update": c_update, "gnn_mean_makespan": float(g.mean()),
                                    "control_mean_makespan": float(c.mean()),
                                    "paired_improvement_percent": [float(v) for v in gains],
                                    "mean_improvement_percent": float(gains.mean()),
                                    "bootstrap_95_ci": confidence_interval(gains, rng),
                                    "positive_seeds": int((gains > 0).sum())})
    primary = [r for r in comparisons if r["gnn_update"] == 1000 and r["control_update"] == 3000]
    output = {"status": "PASS" if all(r["bootstrap_95_ci"][0] > 0 for r in primary) else "FAIL",
              "n_runs": len(files), "n_records": len(by_key),
              "bootstrap_seed": 0, "bootstrap_resamples": 100_000,
              "test_data_sha256_by_size": {str(k): next(iter(v)) for k, v in hashes.items()},
              "comparisons": comparisons}
    with open(a.out, "w") as fh:
        json.dump(output, fh, indent=1)
    print(json.dumps(output, indent=1))


if __name__ == "__main__":
    main()
