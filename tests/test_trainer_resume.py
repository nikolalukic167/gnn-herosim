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
