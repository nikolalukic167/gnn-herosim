#!/usr/bin/env python3
from __future__ import annotations

"""Launch near-RTT v2 CE-only training on atomic-21 cache."""

import os
import sys
import runpy
from pathlib import Path


def main() -> None:
    os.environ.setdefault("NEAR_RTT_LOSS_VARIANT", "near-rtt-v2-trash-exp")
    os.environ.setdefault("NEAR_RTT_SIDECAR_NAME", "valid_combos_near_rtt_v2_capped.pkl")
    os.environ.setdefault("NEAR_RTT_MARGIN_MODE", "exp")
    os.environ.setdefault("NEAR_RTT_MARGIN_CAP", "8.0")
    os.environ.setdefault("NEAR_RTT_MARGIN_EXP_SCALE", "0.75")
    os.environ.setdefault("NEAR_RTT_MARGIN_EXP_CLIP", "4.0")
    os.environ.setdefault("NEAR_RTT_TRASH_DELTA", "5.0")
    os.environ.setdefault("NEAR_RTT_TRASH_WEIGHT", "1.0")
    os.environ.setdefault("NEAR_RTT_FAR_WEIGHT", "0.75")
    os.environ.setdefault("NEAR_RTT_UNMAPPED_PENALTY", "8.0")
    os.environ.setdefault("WANDB_RUN_NAME", "near-rtt-v2-atomic21-ce-only-v2")
    os.environ.setdefault(
        "WANDB_TAGS",
        "near-rtt,loss-v2,ce-only,atomic21,is-cold,raw-queue",
    )
    os.environ.setdefault("NEAR_RTT_REQUIRE_CACHE_VERSION", "7.1-atomic21")

    _REPO_ROOT = Path(__file__).resolve().parents[2]
    _CACHE_DIR = (
        _REPO_ROOT
        / "simulation_data"
        / "artifacts"
        / "run_queue_big"
        / "graphs_cache_gnn_datasets_4tasks_atomic21_seq"
    )

    trainer_path = Path(__file__).with_name("train_near_rtt.py")
    argv = [
        str(trainer_path),
        "--cache-dir",
        str(_CACHE_DIR),
        "--regret-loss-weight",
        "0",
        "--ce-loss-weight",
        "1",
        "--epochs",
        "100",
        "--wandb-project",
        "gnn-near-rtt-jun2026",
    ]
    sys.argv = argv
    runpy.run_path(str(trainer_path), run_name="__main__")


if __name__ == "__main__":
    main()
