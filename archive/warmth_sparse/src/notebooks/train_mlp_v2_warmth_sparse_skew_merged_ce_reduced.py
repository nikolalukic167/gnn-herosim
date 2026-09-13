#!/usr/bin/env python3
from __future__ import annotations

"""CE-reduced MLP from scratch on merged warmth_v2 + sparse_warmth_v2 + skew_warmth_v2 seq cache."""

import os
import sys
import runpy
from pathlib import Path


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    seq_cache = Path(
        os.environ.get(
            "MLP_SEQ_CACHE_DIR",
            str(repo / "simulation_data" / "graphs_cache_warmth_v2_sparse_skew_merged_seq"),
        )
    )
    if not seq_cache.is_dir():
        raise FileNotFoundError(f"Seq cache dir missing: {seq_cache}")

    model = Path(
        os.environ.get(
            "MLP_MODEL_PATH",
            str(repo / "models" / "tabular" / "batch_edge_mlp_warmth_sparse_skew_merged_ce_reduced.pt"),
        )
    )

    trainer_path = repo / "src" / "policy" / "tabular" / "train_mlp_ce_reduced.py"
    if not trainer_path.exists():
        raise FileNotFoundError(f"Reduced MLP trainer not found: {trainer_path}")

    sys.argv = [
        str(trainer_path),
        "--cache-dir",
        str(seq_cache),
        "--project-root",
        str(repo),
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
