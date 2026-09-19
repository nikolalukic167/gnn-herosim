"""Tests for the P1 script's pure parts (log parsing, the shared denominator)."""
import json
import pickle
import tempfile
from pathlib import Path

import pytest

from scripts_cosim.partial_state_v3_p1 import mean_test_opt_rtt, parse_task_log

LOG = """[INTERP] /home/x/python3 2.5.1+cu121
[CODE] abc123 on feat/closed-loop-p1
[TASK 19/32] arm=mpoff seed=4 cfg=experiments/partial_state_v3_mpoff.yaml -> models/partial-state-v3-mpoff-lr2e3-seed4.pt
wandb: Syncing run partial-state-v3-mpoff
wandb: 🚀 View run partial-state-v3-mpoff at: https://wandb.ai/nikolalukic167-tu-wien/gnn-peer-affinity-v1/runs/e5xbplxf
[run end] patience: 131/300 epochs, last improvement at epoch 70
"""


def test_parse_task_log_reads_arm_seed_and_run_id():
    assert parse_task_log(LOG) == ("mpoff", 4, "e5xbplxf")


def test_parse_task_log_is_none_safe():
    assert parse_task_log("nothing here\n") == (None, None, None)
    assert parse_task_log("[TASK 0/32] arm=gnn seed=1 cfg=x -> y\n") == ("gnn", 1, None)


def test_mean_test_opt_rtt_uses_the_split_test_ids_only():
    d = Path(tempfile.mkdtemp())
    (d / "cache").mkdir()
    pickle.dump({"a/ds_0": 10.0, "a/ds_1": 30.0, "b/ds_0": 1000.0}, open(d / "cache" / "optimal_rtt.pkl", "wb"))
    json.dump({"test": ["a/ds_0", "a/ds_1"]}, open(d / "split.json", "w"))
    assert mean_test_opt_rtt(d / "cache", d / "split.json") == pytest.approx(20.0)
    json.dump({"test": ["a/ds_0", "c/ds_9"]}, open(d / "split.json", "w"))
    with pytest.raises(ValueError, match="absent"):
        mean_test_opt_rtt(d / "cache", d / "split.json")
