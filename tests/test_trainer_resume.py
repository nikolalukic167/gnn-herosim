"""NEAR_RTT_RESUME (train_near_rtt.py): a run killed after epoch 0 and resumed ends with weights bit-identical to the straight run.

The real Arm 2 path (re-encode auto, group anchor, random-prefix K = 2, rows-only score) on a mini cache, through run_experiment.py as
subprocesses (the trainer trains at import). Skipped unless RESUME_TEST_CACHE names a graphs.pkl cache directory (the Arm 1 mini cache on
datalab). Also checks that the flag off, and the flag on without a kill, give the same final weights, and, when RESUME_TEST_OLD_WT names a
worktree at the pre-resume code, that those equal the pre-resume trainer's weights.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
CACHE = os.environ.get("RESUME_TEST_CACHE", "")
PHYS = {"HEROSIM_TRANSFER_MODEL": "pipelined", "HEROSIM_REPLICA_RELEASE": "1", "HEROSIM_SCALEOUT": "kpa", "HEROSIM_SHARED_AUTOSCALER": "0",
        "GATE_FIXED_POLICY_TIME_SCALE": "1.0", "R1A_SLATE": "declared_pruning_v1_uncapped", "WANDB_MODE": "offline",
        "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
pytestmark = pytest.mark.skipif(not CACHE or not (Path(CACHE) / "graphs.pkl").exists(), reason="RESUME_TEST_CACHE not set to a cache dir")


def _env(extra=None, wt=REPO):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("NEAR_RTT_", "GNN_", "PARTIAL_STATE", "HEROSIM_"))}
    env.update(PHYS)
    env.update({"PYTHONPATH": f"{wt}:{wt}/src/notebooks", "PYTHONUNBUFFERED": "1", "WANDB_ENTITY": "nikolalukic167-tu-wien"})
    env.update(extra or {})
    return env


def _config(name, tmp_path, wt=REPO):
    cfg = yaml.safe_load(open(REPO / "experiments/uncut_arm2_gnn_eng.yaml"))
    cfg["cache_dir"] = CACHE
    cfg["env"].pop("NEAR_RTT_SPLIT_ARTIFACT", None)
    cfg["args"].update({"epochs": 2, "min-epochs": 2})
    cfg["wandb"].update({"run_name": name, "project": "gnn-uncut-arm2-smoke"})
    path = wt / "experiments" / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def _run(name, tmp_path, extra=None, wt=REPO, kill_after_epoch0=False):
    cfg = _config(name, tmp_path, wt)
    cmd = [sys.executable, "-u", "run_experiment.py", str(cfg), "--seed", "7"]
    log = open(tmp_path / f"{name}.log", "ab")
    proc = subprocess.Popen(cmd, cwd=wt, env=_env(extra, wt), stdout=log, stderr=subprocess.STDOUT)
    state = wt / "models" / f"{name}-seed7.resume.pt"
    try:
        if kill_after_epoch0:
            while proc.poll() is None:
                if state.exists() and torch.load(state, map_location="cpu", weights_only=False)["epoch"] == 0:
                    proc.send_signal(signal.SIGKILL)
                    break
                time.sleep(0.2)
        proc.wait(timeout=1800)
    finally:
        cfg.unlink(missing_ok=True)
    return proc.returncode


def _final(name, wt=REPO):
    return torch.load(wt / "models" / f"{name}-seed7-final.pt", map_location="cpu")


def _equal(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def _cleanup(*names, wt=REPO):
    for n in names:
        for f in (wt / "models").glob(f"{n}-seed7*"):
            f.unlink()


def test_kill_after_epoch0_and_resume_is_bit_identical_to_the_straight_run(tmp_path):
    tag = f"resume-test-{os.getpid()}"
    names = [f"{tag}-straight", f"{tag}-off", f"{tag}-killed"]
    _cleanup(*names)
    try:
        on = {"NEAR_RTT_RESUME": "1", "NEAR_RTT_RESUME_PIN": "t"}
        assert _run(names[0], tmp_path, on) == 0
        assert _run(names[1], tmp_path) == 0  # flag off
        assert not (REPO / "models" / f"{names[1]}-seed7.resume.pt").exists()
        _run(names[2], tmp_path, on, kill_after_epoch0=True)
        assert not (REPO / "models" / f"{names[2]}-seed7-final.pt").exists(), "the kill came too late: the run had already finished"
        assert _run(names[2], tmp_path, on) == 0
        straight, off, resumed = (_final(n) for n in names)
        assert _equal(straight, off), "flag on (no kill) differs from flag off"
        assert _equal(straight, resumed), "the resumed run differs from the straight run"
        best = [torch.load(REPO / "models" / f"{n}-seed7.pt", map_location="cpu") for n in (names[0], names[2])]
        assert _equal(best[0], best[1])
        old = os.environ.get("RESUME_TEST_OLD_WT")
        if old:
            oldwt = Path(old)
            oname = f"{tag}-old"
            _cleanup(oname, wt=oldwt)
            try:
                assert _run(oname, tmp_path, wt=oldwt) == 0
                assert _equal(off, _final(oname, oldwt)), "flag off differs from the pre-resume trainer"
            finally:
                _cleanup(oname, wt=oldwt)
    finally:
        _cleanup(*names)


def test_resume_refuses_a_different_pin(tmp_path):
    name = f"resume-test-pin-{os.getpid()}"
    _cleanup(name)
    try:
        assert _run(name, tmp_path, {"NEAR_RTT_RESUME": "1", "NEAR_RTT_RESUME_PIN": "a"}) == 0
        assert _run(name, tmp_path, {"NEAR_RTT_RESUME": "1", "NEAR_RTT_RESUME_PIN": "b"}) != 0
    finally:
        _cleanup(name)


# ---- the expected move-regret lever (uncut_groups_s0_v1): resume at epoch k plus continuation = the uninterrupted run, with lambda > 0 on a mini cache --------------------------------

def _arm3_files(tmp_path):
    """A pinned split for the mini cache (train / val / test by position), a synthetic move-regret table on the train graphs and a mask-only file on the val graphs."""
    import pickle

    import numpy as np

    sys.path.insert(0, str(REPO / "src" / "notebooks"))
    from non_unique_lib.training_contract import SPLIT_ARTIFACT_SCHEMA, canonical_parent_id

    graphs = pickle.load(open(Path(CACHE) / "graphs.pkl", "rb"))
    ids = [str(g.dataset_id) for g in graphs]
    parents = [canonical_parent_id(getattr(g, "parent_dataset_id", None) or i) for g, i in zip(graphs, ids)]
    assert len(set(parents)) == len(parents) and len(graphs) >= 10
    n = len(graphs)
    n_tr, n_va = int(0.7 * n), int(0.2 * n)
    split = {"schema": SPLIT_ARTIFACT_SCHEMA, "train": parents[:n_tr], "val": parents[n_tr:n_tr + n_va], "test": parents[n_tr + n_va:]}
    assert split["test"] and split["val"]
    split_path = tmp_path / "mini_split.json"
    split_path.write_text(__import__("json").dumps(split))
    rng = np.random.RandomState(0)

    def rows(g, with_d):
        d, u = [], []
        for t in range(int(g.n_tasks)):
            c = len(g.task_logit_to_placement[t])
            d.append([float(x) for x in rng.uniform(0.0, 10.0, size=c)])
            u.append([int(x) for x in (rng.uniform(size=c) < 0.2)])
        return d, u

    train = {}
    for g, i in zip(graphs[:n_tr], ids[:n_tr]):
        d, u = rows(g, True)
        train[i] = {"d": d, "u": u, "S0": 1.0, "n": int(g.n_tasks)}
    val = {}
    for g, i in zip(graphs[n_tr:n_tr + n_va], ids[n_tr:n_tr + n_va]):
        _d, u = rows(g, False)
        val[i] = {"u": u, "n": int(g.n_tasks)}
    table = tmp_path / "mini_move_regret.pkl"
    pickle.dump({"header": {"alpha_key": "inf", "d_max": 10.0}, "groups": train}, open(table, "wb"))
    mask = tmp_path / "mini_unscored_val.pkl"
    pickle.dump({"header": {"alpha_key": "inf", "d_max": 10.0, "mask_only": True}, "groups": val}, open(mask, "wb"))
    return split_path, table, mask


def _arm3_run(name, tmp_path, files, extra=None, kill_after_epoch0=False, lam=None):
    split_path, table, mask = files
    cfg = yaml.safe_load(open(REPO / "experiments/uncut_arm3_gnn_eng_l2p0.yaml"))
    cfg["cache_dir"] = CACHE
    cfg["env"].update({"NEAR_RTT_SPLIT_ARTIFACT": str(split_path), "NEAR_RTT_MOVE_REGRET_PATH": str(table), "NEAR_RTT_UNSCORED_MASK_VAL_PATH": str(mask)})
    if lam is not None:
        cfg["env"]["NEAR_RTT_REGRET_LAMBDA"] = lam
    cfg["args"].update({"epochs": 3, "min-epochs": 3, "patience": 3, "torch-threads": 1, "num-dataloader-workers": 0})
    cfg["wandb"].update({"run_name": name, "project": "gnn-uncut-arm3-smoke"})
    path = REPO / "experiments" / f"{name}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    log = open(tmp_path / f"{name}.log", "ab")
    proc = subprocess.Popen([sys.executable, "-u", "run_experiment.py", str(path), "--seed", "7"], cwd=REPO,
                            env=_env({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", **(extra or {})}), stdout=log, stderr=subprocess.STDOUT)
    state = REPO / "models" / f"{name}-seed7.resume.pt"
    try:
        if kill_after_epoch0:
            while proc.poll() is None:
                if state.exists() and torch.load(state, map_location="cpu", weights_only=False)["epoch"] == 0:
                    proc.send_signal(signal.SIGKILL)
                    break
                time.sleep(0.2)
        proc.wait(timeout=3600)
    finally:
        path.unlink(missing_ok=True)
    return proc.returncode


def test_the_regret_lever_resumes_bit_identically_and_records_the_resume_point(tmp_path):
    import json

    files = _arm3_files(tmp_path)
    tag = f"resume-arm3-{os.getpid()}"
    names = [f"{tag}-straight", f"{tag}-killed"]
    _cleanup(*names)
    try:
        on = {"NEAR_RTT_RESUME": "1", "NEAR_RTT_RESUME_PIN": "t"}
        assert _arm3_run(names[0], tmp_path, files, on) == 0
        _arm3_run(names[1], tmp_path, files, on, kill_after_epoch0=True)
        assert not (REPO / "models" / f"{names[1]}-seed7-final.pt").exists(), "the kill came too late: the run had already finished"
        assert _arm3_run(names[1], tmp_path, files, on) == 0
        straight, resumed = _final(names[0]), _final(names[1])
        assert _equal(straight, resumed), "the resumed regret run differs from the straight run"
        best = [torch.load(REPO / "models" / f"{n}-seed7.pt", map_location="cpu") for n in names]
        assert _equal(best[0], best[1]), "the selected checkpoint differs"
        side = [json.load(open(REPO / "models" / f"{n}-seed7-final.contract.json")) for n in names]
        assert side[0]["resume_points"] == [] and side[1]["resume_points"] == [0]
        assert side[1]["regret_lambda"] == 2.0 and side[1]["move_regret"]["sha256"] == side[0]["move_regret"]["sha256"]
        # a resume under another lambda is refused (the fingerprint carries lambda and the table's sha256)
        assert _arm3_run(names[1], tmp_path, files, on, lam="0.5") != 0
    finally:
        _cleanup(*names)
