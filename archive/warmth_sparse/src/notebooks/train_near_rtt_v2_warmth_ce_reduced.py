#!/usr/bin/env python3
from __future__ import annotations

"""CE-only reduced-feature ablation on warmth_v2 cache (task=3, platform=6, edge=2)."""

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
    os.environ.setdefault("WANDB_RUN_NAME", "near-rtt-v2-warmth-ce-reduced")
    os.environ.setdefault(
        "WANDB_TAGS",
        "near-rtt,ce-only,reduced-features,warmth-v2,node-disk-v2,ablation",
    )

    _repo = Path(__file__).resolve().parents[2]
    _cache = Path(
        os.environ.get(
            "NEAR_RTT_CACHE_DIR",
            str(_repo / "simulation_data" / "graphs_cache_gnn_datasets_4tasks_1060_warmth_v2"),
        )
    )

    trainer_path = Path(__file__).with_name("train_near_rtt_ce_reduced_features.py")
    sys.argv = [
        str(trainer_path),
        "--cache-dir",
        str(_cache),
        "--regret-loss-weight",
        "0",
        "--ce-loss-weight",
        "1",
        "--epochs",
        "100",
        "--wandb-project",
        "gnn-near-rtt-warmth-v2-jun2026",
    ]
    runpy.run_path(str(trainer_path), run_name="__main__")


if __name__ == "__main__":
    main()
