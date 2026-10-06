"""Evaluate fixed-update L2D checkpoints on fresh, deterministic instances."""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np
import torch

from l2d_bridge import load_l2d, upstream_commit
from train_arm import greedy_eval, make_policy


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--test_seed", type=int, default=302)
    p.add_argument("--n_test", type=int, default=100)
    p.add_argument("--updates", default="1000,3000")
    p.add_argument("--threads", type=int, default=2)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    updates = [int(x) for x in a.updates.split(",")]
    if not updates or len(set(updates)) != len(updates) or any(x <= 0 for x in updates):
        raise ValueError("updates must be distinct positive integers")
    results = []
    for update in updates:
        ckpt = os.path.join(a.run_dir, f"update_{update}.pth")
        with open(ckpt + ".contract.json") as fh:
            meta = json.load(fh)
        if meta["checkpoint"] != "fixed_update" or meta["selected_update"] != update:
            raise ValueError(f"invalid checkpoint contract: {ckpt}")
        if meta["upstream_commit"] != upstream_commit():
            raise ValueError(f"upstream commit mismatch: {ckpt}")
        L = load_l2d(meta["n_j"], meta["n_m"], "cpu")
        np.random.seed(a.test_seed)
        data = [L.uni_instance_gen(n_j=meta["n_j"], n_m=meta["n_m"],
                                   low=L.configs.low, high=L.configs.high)
                for _ in range(a.n_test)]
        data_digest = hashlib.sha256()
        for instance in data:
            for array in instance:
                data_digest.update(np.ascontiguousarray(array).tobytes())
        pol = make_policy(L, meta["arm"], meta["n_j"], meta["n_m"], torch.device("cpu"))
        pol.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        pol.eval()
        values = greedy_eval(L, meta["arm"], pol, data, meta["n_j"], meta["n_m"],
                             torch.device("cpu"))
        results.append({"arm": meta["arm"], "study_seed": meta["study_seed"],
                        "n_j": meta["n_j"], "n_m": meta["n_m"],
                        "update": update, "train_instances": update * meta["num_envs"],
                        "test_seed": a.test_seed, "n_test": a.n_test,
                        "checkpoint_sha256": sha256_file(ckpt),
                        "contract_sha256": sha256_file(ckpt + ".contract.json"),
                        "test_data_sha256": data_digest.hexdigest(),
                        "mean_makespan": float(values.mean()),
                        "makespans": [float(x) for x in values]})
    out = os.path.join(a.run_dir, f"fixed_budget_test_seed{a.test_seed}.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=1)
    print(out, [(r["update"], r["mean_makespan"]) for r in results], flush=True)


if __name__ == "__main__":
    main()
