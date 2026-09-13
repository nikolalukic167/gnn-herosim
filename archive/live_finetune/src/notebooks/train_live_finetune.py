#!/usr/bin/env python3
"""Fine-tune sequential GNN on live-labeled snapshot cache."""

import os
import sys
import runpy
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]

LIVE_CACHE = (
    _ROOT
    / "simulation_data"
    / "artifacts"
    / "run_queue_big"
    / "graphs_cache_gnn_datasets_live_150_150_seq"
)

os.environ.setdefault(
    "TRAIN_INIT_CHECKPOINT",
    str(_ROOT / "src" / "notebooks" / "models" / "brisk-cosmos-41.pt"),
)
os.environ.setdefault("WANDB_TAGS", "live-finetune,150-150,real-autoscale")
os.environ["TRAIN_STRATIFIED_NEGATIVES"] = "1"

if "--cache-dir" not in sys.argv:
    sys.argv += ["--cache-dir", str(LIVE_CACHE)]

print("=" * 70)
print("LIVE FINE-TUNE: real autoscale snapshots + co-sim labels")
print(f"  Cache dir : {LIVE_CACHE}")
print(f"  Init ckpt : {os.environ['TRAIN_INIT_CHECKPOINT']}")
print(f"  Extra argv: {sys.argv[1:]}")
print("=" * 70)

runpy.run_path(str(_HERE / "train_seq.py"), run_name="__main__")
