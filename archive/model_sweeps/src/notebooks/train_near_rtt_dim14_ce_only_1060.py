#!/usr/bin/env python3
from __future__ import annotations

"""CE-only near-RTT training on the legacy 1060 dim-14 cache (not warmth_v2).

Same cache/split as train_near_rtt_ce_reduced_features.py ablation; full 14-d features.
"""

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
    os.environ.setdefault("WANDB_RUN_NAME", "near-rtt-dim14-ce-only-1060")
    os.environ.setdefault(
        "WANDB_TAGS",
        "near-rtt,ce-only,dim14,1060,legacy-cache,ablation-pair",
    )

    repo_root = Path(__file__).resolve().parents[2]
    cache_dir = Path(
        os.environ.get(
            "NEAR_RTT_CACHE_DIR",
            str(
                repo_root
                / "simulation_data"
                / "artifacts"
                / "run_queue_big"
                / "graphs_cache_gnn_datasets_4tasks_1060"
            ),
        )
    )

    trainer_path = Path(__file__).with_name("train_near_rtt.py")
    argv = [
        str(trainer_path),
        "--cache-dir",
        str(cache_dir),
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
