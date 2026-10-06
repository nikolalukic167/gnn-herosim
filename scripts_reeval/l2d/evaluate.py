"""literature_reeval_v1 -- greedy evaluation of trained arms, upstream SavedNetwork, and PDRs.

Splits:
  val    -- upstream DataGen/generatedData{n_j}_{n_m}_Seed200.npy: used by every arm for checkpoint
            selection AND by the paper as its test set (Table 1). Reported, never primary.
  test   -- 100 fresh instances, np seed 300, generated once with upstream uni_instance_gen and
            frozen at simulation_data/literature_reeval_v1/l2d/test_{n_j}x{n_m}_seed300.npy. PRIMARY.
  tai    -- Taillard benchmark of the same size when upstream ships it (15x15 only in this study).

Writes one JSON per (subject, split) under --out with per-instance makespans.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from l2d_bridge import load_l2d, upstream_commit  # noqa: E402
from arm_features import ARMS, INPUT_DIM  # noqa: E402
from pdr import PDRS, run_pdr  # noqa: E402
from train_arm import greedy_eval, make_policy  # noqa: E402

TEST_SEED = 300
N_TEST = 100


def test_set_path(data_dir, n_j, n_m):
    return os.path.join(data_dir, f"test_{n_j}x{n_m}_seed{TEST_SEED}.npy")


def ensure_test_set(L, data_dir, n_j, n_m):
    p = test_set_path(data_dir, n_j, n_m)
    if not os.path.exists(p):
        np.random.seed(TEST_SEED)
        data = np.array([L.uni_instance_gen(n_j=n_j, n_m=n_m, low=L.configs.low, high=L.configs.high)
                         for _ in range(N_TEST)])
        np.save(p, data)
        print(f"generated {p}")
    d = np.load(p)
    return [(d[i][0], d[i][1]) for i in range(d.shape[0])]


def load_split(L, data_dir, n_j, n_m, split):
    if split == "val":
        d = np.load(os.path.join(L.root, f"DataGen/generatedData{n_j}_{n_m}_Seed200.npy"))
    elif split == "test":
        return ensure_test_set(L, data_dir, n_j, n_m)
    elif split == "tai":
        p = os.path.join(L.root, f"BenchDataNmpy/tai{n_j}x{n_m}.npy")
        if not os.path.exists(p):
            return None
        d = np.load(p)
    else:
        raise ValueError(split)
    return [(d[i][0], d[i][1]) for i in range(d.shape[0])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_j", type=int, required=True)
    ap.add_argument("--n_m", type=int, required=True)
    ap.add_argument("--runs", required=True, help="dir holding l2d_{n_j}x{n_m}_{arm}_s{seed}/ run dirs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--data_dir", default="simulation_data/literature_reeval_v1/l2d")
    ap.add_argument("--splits", default="val,test,tai")
    ap.add_argument("--checkpoints", default="best_val,final")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    device = torch.device("cpu")
    L = load_l2d(a.n_j, a.n_m, "cpu")
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(a.data_dir, exist_ok=True)

    splits = {s: load_split(L, a.data_dir, a.n_j, a.n_m, s) for s in a.splits.split(",")}
    splits = {k: v for k, v in splits.items() if v is not None}

    def write(name, split, makespans, extra):
        rec = {"subject": name, "split": split, "n_j": a.n_j, "n_m": a.n_m, "n_instances": len(makespans),
               "mean_makespan": float(np.mean(makespans)), "makespans": [float(x) for x in makespans],
               "upstream_commit": upstream_commit(), **extra}
        with open(os.path.join(a.out, f"{name}__{split}.json"), "w") as fh:
            json.dump(rec, fh)
        print(f"{name:32s} {split:5s} mean {rec['mean_makespan']:9.2f}")

    # PDRs
    env = L.SJSSP(n_j=a.n_j, n_m=a.n_m)
    for rule in PDRS:
        for split, data in splits.items():
            write(f"pdr_{rule}", split, [run_pdr(rule, env, d) for d in data], {"kind": "pdr"})

    # upstream SavedNetwork (reproduction anchor; trained by the authors, arm == gnn)
    sp = os.path.join(L.root, f"SavedNetwork/{a.n_j}_{a.n_m}_1_99.pth")
    if os.path.exists(sp):
        pol = make_policy(L, "gnn", a.n_j, a.n_m, device)
        pol.load_state_dict(torch.load(sp, map_location="cpu"))
        for split, data in splits.items():
            write("upstream_saved_gnn", split, greedy_eval(L, "gnn", pol, data, a.n_j, a.n_m, device),
                  {"kind": "upstream_checkpoint", "path": sp})

    # our arms
    for run_dir in sorted(glob.glob(os.path.join(a.runs, f"l2d_{a.n_j}x{a.n_m}_*_s*"))):
        for ck in a.checkpoints.split(","):
            pth = os.path.join(run_dir, f"{ck}.pth")
            if not os.path.exists(pth):
                print(f"skip {pth} (missing)")
                continue
            with open(pth + ".contract.json") as fh:
                meta = json.load(fh)
            arm = meta["arm"]
            assert arm in ARMS and meta["input_dim"] == INPUT_DIM[arm], meta
            pol = make_policy(L, arm, a.n_j, a.n_m, device)
            pol.load_state_dict(torch.load(pth, map_location="cpu"))
            name = f"{os.path.basename(run_dir)}__{ck}"
            for split, data in splits.items():
                write(name, split, greedy_eval(L, arm, pol, data, a.n_j, a.n_m, device),
                      {"kind": "arm", "arm": arm, "seed": meta["study_seed"], "checkpoint": ck, "lr": meta["lr"],
                       "selected_update": meta.get("selected_update")})


if __name__ == "__main__":
    main()
