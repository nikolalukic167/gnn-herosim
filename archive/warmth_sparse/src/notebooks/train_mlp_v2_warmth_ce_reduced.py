#!/usr/bin/env python3
from __future__ import annotations

"""CE-reduced MLP ablation on warmth_v2 seq cache (task=3, platform=6, edge=2)."""

import os
import sys
import runpy
from pathlib import Path


def main() -> None:
    _repo = Path(__file__).resolve().parents[2]
    _seq_cache = Path(
        os.environ.get(
            "MLP_SEQ_CACHE_DIR",
            str(_repo / "simulation_data" / "graphs_cache_gnn_datasets_4tasks_1060_warmth_v2_seq"),
        )
    )
    _model = Path(
        os.environ.get(
            "MLP_MODEL_PATH",
            str(_repo / "models" / "tabular" / "batch_edge_mlp_warmth_ce_reduced.pt"),
        )
    )

    trainer_path = _repo / "src" / "policy" / "tabular" / "train_mlp_ce_reduced.py"
    if not trainer_path.exists():
        raise FileNotFoundError(f"Reduced MLP trainer not found: {trainer_path}")

    sys.argv = [
        str(trainer_path),
        "--cache-dir",
        str(_seq_cache),
        "--project-root",
        str(_repo),
        "--output",
        str(_model),
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
