#!/usr/bin/env python3
from __future__ import annotations

"""CE-only GNN on B1 seq cache (src_norm, dim13 disk hit, sandbox is_warm)."""

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
    os.environ.setdefault("NEAR_RTT_REQUIRE_CACHE_VERSION", "7.1-atomic21")
    os.environ.setdefault("WANDB_RUN_NAME", "near-rtt-v2-warmth-features-v1-ce-only")
    os.environ.setdefault(
        "WANDB_TAGS",
        "near-rtt,ce-only,dim14,b1-features,warmth-v2,sparse-v2,seq-cache,node-disk-v2",
    )

    repo = Path(__file__).resolve().parents[2]
    cache = Path(
        os.environ.get(
            "NEAR_RTT_CACHE_DIR",
            str(repo / "simulation_data" / "graphs_cache_warmth_v2_features_v1"),
        )
    )

    trainer_path = Path(__file__).with_name("train_near_rtt.py")
    sys.argv = [
        str(trainer_path),
        "--cache-dir",
        str(cache),
        "--regret-loss-weight",
        "0",
        "--ce-loss-weight",
        "1",
        "--epochs",
        os.environ.get("NEAR_RTT_TRAIN_EPOCHS", "100"),
        "--num-dataloader-workers",
        os.environ.get("NEAR_RTT_DATALOADER_WORKERS", "0"),
        "--wandb-project",
        "gnn-near-rtt-warmth-v2-jun2026",
    ]
    runpy.run_path(str(trainer_path), run_name="__main__")


if __name__ == "__main__":
    main()
