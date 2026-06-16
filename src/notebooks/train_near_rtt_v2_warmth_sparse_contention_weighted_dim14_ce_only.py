#!/usr/bin/env python3
from __future__ import annotations

"""Train dim14 CE-only on weighted merged warmth+sparse+contention_v2 cache."""

import os
import sys
import runpy
from pathlib import Path


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    cache = Path(
        os.environ.get(
            "NEAR_RTT_CACHE_DIR",
            str(repo / "simulation_data" / "graphs_cache_warmth_sparse_contention_v2_weighted"),
        )
    )
    if not cache.is_dir():
        raise FileNotFoundError(f"Cache dir missing: {cache}")

    os.environ.pop("TRAIN_INIT_CHECKPOINT", None)
    os.environ.setdefault("NEAR_RTT_LOSS_VARIANT", "near-rtt-v2-trash-exp")
    os.environ.setdefault("NEAR_RTT_SIDECAR_NAME", "valid_combos_near_rtt_v2_capped.pkl")
    os.environ.setdefault("WANDB_RUN_NAME", "near-rtt-v2-warmth-sparse-contention-weighted-dim14-ce-only")
    os.environ.setdefault(
        "WANDB_TAGS",
        "near-rtt,ce-only,dim14,warmth-v2,sparse-v2,contention-v2,weighted-merge",
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
        os.environ.get("WANDB_PROJECT", "gnn-near-rtt-contention-v2-jun2026"),
    ]
    runpy.run_path(str(trainer), run_name="__main__")


if __name__ == "__main__":
    main()
