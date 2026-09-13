#!/usr/bin/env python3
from __future__ import annotations

"""Launch near-RTT training with the loss/sidecar v2 defaults."""

import os
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
    os.environ.setdefault("WANDB_RUN_NAME", "near-rtt-v2-clean-1230")
    os.environ.setdefault(
        "WANDB_TAGS",
        "near-rtt,loss-v2,trash-sidecar,clean-1230,scheduler-adaptive",
    )

    trainer_path = Path(__file__).with_name("train_near_rtt.py")
    runpy.run_path(str(trainer_path), run_name="__main__")


if __name__ == "__main__":
    main()
