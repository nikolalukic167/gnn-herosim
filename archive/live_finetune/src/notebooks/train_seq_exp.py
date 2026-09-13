#!/usr/bin/env python3
"""
Experiment runner: filtered high-queue cache + stratified negatives.

What this does vs train_seq.py:
  - Cache: graphs_cache_gnn_datasets_4tasks_seq_filtered
    Built from the 883 datasets with avg_queue_time >= 0.5s (23.8% of corpus).
    These are the datasets where queue load dominates RTT and prefix-optimal
    diverges most from shortest-queue — the regime the model currently misses.
  - Stratified negatives (TRAIN_STRATIFIED_NEGATIVES=1):
    Hard-negative pool sampled 40% near-optimal / 40% moderate / 20% catastrophic
    instead of 88% catastrophic.  Teaches the model to rank close alternatives,
    not only to avoid disastrous joint placements.
  - W&B run tagged exp-filtered-strat for easy comparison against baseline.

Usage:
  cd src/notebooks
  pipenv run python3 train_seq_exp.py --epochs 30 --num-dataloader-workers 0
  pipenv run python3 train_seq_exp.py --epochs 30 --num-dataloader-workers 0 --regret-loss-weight 0.7
"""

import os
import sys
import runpy
from pathlib import Path

# ── experiment config ─────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]  # repo root

FILTERED_CACHE = (
    _ROOT
    / "simulation_data"
    / "artifacts"
    / "run_queue_big"
    / "graphs_cache_gnn_datasets_4tasks_seq_filtered"
)

os.environ["TRAIN_STRATIFIED_NEGATIVES"] = "1"
os.environ["WANDB_TAGS"] = "exp-filtered-strat,6.3-filtered,no-seqblend"

# ── inject --cache-dir into argv if not already provided ─────────────────────
if "--cache-dir" not in sys.argv:
    sys.argv += ["--cache-dir", str(FILTERED_CACHE)]

print("=" * 70)
print("EXPERIMENT: filtered cache + stratified negatives")
print(f"  Cache dir : {FILTERED_CACHE}")
print(f"  Stratified: {os.environ['TRAIN_STRATIFIED_NEGATIVES']}")
print(f"  Extra argv: {sys.argv[1:]}")
print("=" * 70)

# ── run train_seq.py in the same process (shares all imports/model code) ─────
runpy.run_path(str(_HERE / "train_seq.py"), run_name="__main__")
