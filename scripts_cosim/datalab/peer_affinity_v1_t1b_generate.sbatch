#!/bin/bash
# peer_affinity_v1 T1b -- corpus expansion (docs/lineages/peer_affinity_v1.md).
# 500 fresh seeds (7200-7699) of the SAME cell the T1 training corpus used (c3, 200 MB, k=10),
# written to gnn_datasets_peer_affinity_v1_c3_x200_train2. Generation env is replicated from the
# T1 corpus EXACTLY (incl. GNN_CAPTURE_DATASET_STATE unset -> SSC rewritten afterwards by
# refresh_optimal_full_stats.py --rewrite-ssc, as the T1 corpus was) so the merged corpus is
# homogeneous. 20 shards x 25 datasets.
#
#SBATCH --job-name=pa-t1b-gen
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/pa-t1b-gen-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/pa-t1b-gen-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-19

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim

if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1

# Physics + sweep env: identical to the T1 training corpus.
export HEROSIM_PEER_EXCHANGE=1
export HEROSIM_RETAIN_TASK_TIMES=1
export HEROSIM_RETAIN_PEER_STATS=1
export MAX_PLACEMENT_COMBINATIONS_SKIP=2000000
export GNN_CAPTURE_DATASET_STATE=0
export COSIM_SUPPRESS_SIM_PRINTS=1

GRID=peer_affinity_screen_c3_x200_train2
OUT=gnn_datasets_peer_affinity_v1_c3_x200_train2
TOTAL=500
SHARDS=20
PER=$(( (TOTAL + SHARDS - 1) / SHARDS ))
START=$(( SLURM_ARRAY_TASK_ID * PER ))
COUNT=$PER
if (( START >= TOTAL )); then echo "nothing to do"; exit 0; fi
if (( START + COUNT > TOTAL )); then COUNT=$(( TOTAL - START )); fi
WORKERS=$(( ${SLURM_CPUS_PER_TASK:-32} - 1 ))

echo "[TASK ${SLURM_ARRAY_TASK_ID}] grid=$GRID out=$OUT range=$START..$((START+COUNT-1)) workers=$WORKERS"
python3 -c "import sys,torch;print('[INTERP]',sys.executable)"
mkdir -p logs "simulation_data/${OUT}"

python3 -u scripts_cosim/generate_gnn_datasets_fast.py \
  --quiet \
  --grid "$GRID" \
  --num-tasks 10 \
  --max-datasets "$COUNT" \
  --start-from "$START" \
  --resume \
  --allow-non-unique-replicas \
  --warmth-physics node_disk_v2 \
  --output-subdir "$OUT" \
  --progress-log-name "progress_${OUT}_array${SLURM_ARRAY_TASK_ID}.txt" \
  --workers "$WORKERS"

echo "=== task ${SLURM_ARRAY_TASK_ID} done ==="
