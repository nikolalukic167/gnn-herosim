#!/usr/bin/env python3
from __future__ import annotations

"""dim22 MLP from contention_v2 batch cache (feature parity with GNN)."""

import os
import sys
import runpy
from pathlib import Path


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    batch_cache = Path(
        os.environ.get(
            "NEAR_RTT_CACHE_DIR",
            str(repo / "simulation_data" / "graphs_cache_contention_v2"),
        )
    )
    if not batch_cache.is_dir():
        raise FileNotFoundError(f"Batch cache dir missing: {batch_cache}")

    model = Path(
        os.environ.get(
            "MLP_MODEL_PATH",
            str(repo / "models" / "tabular" / "batch_edge_mlp_contention_v2_dim22_batchcache.pt"),
        )
    )

    trainer_path = repo / "src" / "policy" / "tabular" / "train_mlp_dim22_from_batch.py"
    if not trainer_path.exists():
        raise FileNotFoundError(f"Batch dim22 trainer not found: {trainer_path}")

    sys.argv = [
        str(trainer_path),
        "--cache-dir",
        str(batch_cache),
        "--output",
        str(model),
        "--epochs",
        os.environ.get("MLP_EPOCHS", "100"),
        "--patience",
        os.environ.get("MLP_PATIENCE", "10"),
        "--hidden-dim",
        "64",
        "--lr",
        "1e-3",
        "--random-state",
        "42",
        "--test-size",
        "0.2",
    ]
    runpy.run_path(str(trainer_path), run_name="__main__")


if __name__ == "__main__":
    main()
