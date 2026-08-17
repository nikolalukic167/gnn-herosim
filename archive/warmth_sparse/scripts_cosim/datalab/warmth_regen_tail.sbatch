#!/bin/bash
#SBATCH --job-name=warmth-tail
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/warmth-tail-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/warmth-tail-%j.err
#SBATCH --partition=GPU-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=24:00:00

# Finish warmth_v2 tail: ds_00488..ds_00499 (12 datasets, full BF + placements.jsonl).

set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

export OUTPUT_SUBDIR=gnn_datasets_4tasks_1060_warmth_v2
export WARMTH_PHYSICS=node_disk_v2
export GRID=warmth_v2
export GNN_CAPTURE_DATASET_STATE=0
export COSIM_SUPPRESS_SIM_PRINTS=1
export START_FROM=488
export MAX_DATASETS=12
export SHARD_LABEL=tail488
export ONLY_MISSING_JSONL=0

echo "=== Clear recovered-only JSONL on tail (no best.json) — full BF will rewrite ==="
python3 - <<'PY'
from pathlib import Path
base = Path("simulation_data/gnn_datasets_4tasks_1060_warmth_v2")
for i in range(488, 500):
    ds = base / f"ds_{i:05d}"
    best = ds / "best.json"
    jsonl = ds / "placements" / "placements.jsonl"
    if not best.exists() and jsonl.exists():
        jsonl.unlink()
        print(f"removed stale jsonl: {ds.name}")
PY

bash scripts_cosim/datalab/run_warmth_regen_range.sh

echo "=== Tail done: best.json count ==="
find "simulation_data/${OUTPUT_SUBDIR}" -name best.json ! -path '*/.bf_scratch/*' 2>/dev/null | wc -l
echo "=== JSONL count ==="
find "simulation_data/${OUTPUT_SUBDIR}" -path '*/placements/placements.jsonl' -size +0c 2>/dev/null | wc -l

echo "=== Validate ds_00488..ds_00499 have best.json + non-empty placements.jsonl ==="
python3 - <<'PY'
from pathlib import Path
import sys

base = Path("simulation_data/gnn_datasets_4tasks_1060_warmth_v2")
bad = []
for i in range(488, 500):
    ds = base / f"ds_{i:05d}"
    best = ds / "best.json"
    jsonl = ds / "placements" / "placements.jsonl"
    if not best.exists():
        bad.append(f"{ds.name}: missing best.json")
    elif not jsonl.exists() or jsonl.stat().st_size == 0:
        bad.append(f"{ds.name}: missing placements.jsonl")
if bad:
    print("TAIL VALIDATION FAILED:", file=sys.stderr)
    for line in bad:
        print(f"  {line}", file=sys.stderr)
    sys.exit(1)
print("tail488 OK: 12/12 best.json + placements.jsonl")
PY
