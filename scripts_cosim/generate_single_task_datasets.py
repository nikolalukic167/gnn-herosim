#!/usr/bin/env python3
"""
Generate marginal-oracle 1-task co-sim datasets for Regime B (XGBoost per-arrival).

Thin wrapper around generate_gnn_datasets_fast.py with --num-tasks 1 and default
output under gnn_datasets_1task.

Usage:
  pipenv run python3 scripts_cosim/generate_single_task_datasets.py --quiet --max-datasets 100
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = PROJECT_ROOT / "scripts_cosim" / "generate_gnn_datasets_fast.py"

def main() -> None:
    if not SCRIPT.exists():
        raise FileNotFoundError(f"Missing generator script: {SCRIPT}")
    user_args = sys.argv[1:] or ["--quiet", "--max-datasets", "10"]
    sys.argv = [
        str(SCRIPT),
        "--num-tasks",
        "1",
        "--output-subdir",
        "gnn_datasets_1task",
        "--progress-log-name",
        "progress_1tasks.txt",
        *user_args,
    ]
    runpy.run_path(str(SCRIPT), run_name="__main__")


if __name__ == "__main__":
    main()
