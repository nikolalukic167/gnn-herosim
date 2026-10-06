"""Read the registered seed-303 midpoint follow-up without changing the primary gate."""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from analyze_fixed_budget import confidence_interval


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    by_key = {}
    hashes = {}
    files = sorted(glob.glob(os.path.join(a.base, "runs", "l2d_*", "fixed_budget_test_seed303.json")))
    files += sorted(glob.glob(os.path.join(a.base, "followup_runs", "l2d_*", "fixed_budget_test_seed303.json")))
    for path in files:
        records = json.load(open(path))
        if len(records) != 1:
            raise ValueError(f"expected one fixed checkpoint: {path}")
        r = records[0]
        size, arm, seed, update = r["n_j"], r["arm"], r["study_seed"], r["update"]
        expected_update = 2000 if arm == "gnn" else 3000
        key = (size, arm, seed)
        if (key in by_key or size not in (6, 10) or r["n_m"] != size or
                seed not in range(8) or update != expected_update or
                r["train_instances"] != update * 4 or r["test_seed"] != 303 or
                r["n_test"] != 100 or len(r["makespans"]) != 100 or
                not np.isclose(np.mean(r["makespans"]), r["mean_makespan"])):
            raise ValueError(f"invalid record: {path}")
        hashes.setdefault(size, set()).add(r["test_data_sha256"])
        by_key[key] = r
    expected = {(size, arm, seed) for size in (6, 10)
                for arm in ("gnn", "mpoff", "mlp_t1") for seed in range(8)}
    if set(by_key) != expected or len(files) != 48 or any(len(hashes[size]) != 1 for size in (6, 10)):
        raise ValueError(f"incomplete gate: {len(files)} files, {len(by_key)} records, hashes={hashes}")
    rng = np.random.default_rng(0)
    comparisons = []
    for size in (6, 10):
        for control in ("mpoff", "mlp_t1"):
            g = np.array([by_key[(size, "gnn", s)]["mean_makespan"] for s in range(8)])
            c = np.array([by_key[(size, control, s)]["mean_makespan"] for s in range(8)])
            gains = 100 * (c - g) / c
            comparisons.append({"size": size, "control": control,
                                "gnn_mean_makespan": float(g.mean()),
                                "control_mean_makespan": float(c.mean()),
                                "paired_improvement_percent": [float(v) for v in gains],
                                "mean_improvement_percent": float(gains.mean()),
                                "bootstrap_95_ci": confidence_interval(gains, rng),
                                "positive_seeds": int((gains > 0).sum())})
    result = {"status": "PASS" if all(c["bootstrap_95_ci"][0] > 0 for c in comparisons) else "FAIL",
              "n_runs": len(files), "bootstrap_seed": 0, "bootstrap_resamples": 100_000,
              "test_data_sha256_by_size": {str(k): next(iter(v)) for k, v in hashes.items()},
              "comparisons": comparisons}
    with open(a.out, "w") as fh:
        json.dump(result, fh, indent=1)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
