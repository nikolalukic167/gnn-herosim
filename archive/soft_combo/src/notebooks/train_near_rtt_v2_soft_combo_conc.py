#!/usr/bin/env python3
from __future__ import annotations

"""Launch anchored near-RTT v2 soft-combo training with queue concentration regularization."""

import os
import sys
import runpy
from pathlib import Path


def main() -> None:
    os.environ.setdefault("NEAR_RTT_LOSS_VARIANT", "near-rtt-v2-trash-exp")
    os.environ.setdefault("NEAR_RTT_SIDECAR_NAME", "valid_combos_near_rtt_v2_capped.pkl")
    os.environ.setdefault("NEAR_RTT_TRAIN_OBJECTIVE", "soft_combo_conc")
    os.environ.setdefault("NEAR_RTT_SOFT_COMBO_TAU", "0.25")
    os.environ.setdefault("NEAR_RTT_SOFT_COMBO_MAX_COMBOS", "4096")
    os.environ.setdefault("NEAR_RTT_CONC_GAMMA", "0.02")
    os.environ.setdefault("NEAR_RTT_CONC_CAP", "1.5")
    os.environ.setdefault("WANDB_RUN_NAME", "near-rtt-v2-clean-1230-soft-combo-conc")
    os.environ.setdefault(
        "WANDB_TAGS",
        "near-rtt,loss-v2,soft-combo,anchored-ce,concentration,clean-1230,scheduler-adaptive",
    )

    trainer_path = Path(__file__).with_name("train_near_rtt.py")
    sys.argv = [
        str(trainer_path),
        "--regret-loss-weight",
        "0",
        "--ce-loss-weight",
        "1",
        "--wandb-project",
        "gnn-near-rtt-jun2026",
    ]
    runpy.run_path(str(trainer_path), run_name="__main__")


if __name__ == "__main__":
    main()
