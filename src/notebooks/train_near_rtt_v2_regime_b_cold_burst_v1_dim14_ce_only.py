#!/usr/bin/env python3
from __future__ import annotations

"""Train dim14 CE-only from scratch on Regime B cold-burst cache (450 ds)."""

import os
import sys
import runpy
from pathlib import Path


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    cache = Path(
        os.environ.get(
            "NEAR_RTT_CACHE_DIR",
            str(repo / "simulation_data" / "graphs_cache_regime_b_cold_burst_v1"),
        )
    )
    if not cache.is_dir():
        raise FileNotFoundError(f"Cache dir missing: {cache}")

    os.environ.pop("TRAIN_INIT_CHECKPOINT", None)
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
    os.environ.setdefault(
        "WANDB_RUN_NAME", "near-rtt-v2-regime-b-cold-burst-v1-dim14-ce-only"
    )
    os.environ.setdefault(
        "WANDB_TAGS",
        "near-rtt,ce-only,dim14,regime-b,cold-burst-v1,from-scratch,platform-reuse-v1",
    )

    trainer = Path(__file__).with_name("train_near_rtt.py")
    sys.argv = [
        str(trainer),
        "--cache-dir",
        str(cache),
        "--regret-loss-weight",
        "0",
        "--ce-loss-weight",
        "1",
        "--epochs",
        os.environ.get("NEAR_RTT_TRAIN_EPOCHS", "100"),
        "--wandb-project",
        os.environ.get("WANDB_PROJECT", "gnn-near-rtt-regime-b-aug2026"),
    ]
    runpy.run_path(str(trainer), run_name="__main__")


if __name__ == "__main__":
    main()
