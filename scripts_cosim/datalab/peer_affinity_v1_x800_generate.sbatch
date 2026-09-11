#!/bin/bash
# peer_affinity_v1 -- denser-graph rung corpora (docs/lineages/peer_affinity_v1.md, 2026-09-11).
# Same cell as T1/T1b (c3, k = 10, alpha ladder unchanged) with the peer graph changed:
#   p3: THREE partners per task at 800 MB   (paper-screen GO cell x800_a1.5_k10c3_p3)
#   p2: two partners per task at 800 MB     (matched control: payload moved, partners not)
# Generation env replicated from the T1b corpus exactly. Parameterised by env so one script
# builds all four corpora:
#   sbatch --export=ALL,GRID=peer_affinity_screen_c3_x800_p3_train,OUT=gnn_datasets_peer_affinity_v1_c3_x800_p3_train,TOTAL=500 --array=0-19 <this>
#   sbatch --export=ALL,GRID=peer_affinity_screen_c3_x800_p3_r2,OUT=gnn_datasets_peer_affinity_v1_c3_x800_p3_r2,TOTAL=34,SHARDS=2 --array=0-1 <this>
#   (same for _p2_)
#
#SBATCH --job-name=pa-x800-gen
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/pa-x800-gen-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/pa-x800-gen-%A_%a.err
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

# Physics + sweep env: identical to the T1/T1b training corpora.
export HEROSIM_PEER_EXCHANGE=1
export HEROSIM_RETAIN_TASK_TIMES=1
export HEROSIM_RETAIN_PEER_STATS=1
export MAX_PLACEMENT_COMBINATIONS_SKIP=2000000
export GNN_CAPTURE_DATASET_STATE=0
export COSIM_SUPPRESS_SIM_PRINTS=1

GRID="${GRID:-peer_affinity_screen_c3_x800_p3_train}"
OUT="${OUT:-gnn_datasets_peer_affinity_v1_c3_x800_p3_train}"
TOTAL="${TOTAL:-500}"
SHARDS="${SHARDS:-20}"
PER=$(( (TOTAL + SHARDS - 1) / SHARDS ))
START=$(( SLURM_ARRAY_TASK_ID * PER ))
COUNT=$PER
if (( START >= TOTAL )); then echo "nothing to do"; exit 0; fi
if (( START + COUNT > TOTAL )); then COUNT=$(( TOTAL - START )); fi
WORKERS=$(( ${SLURM_CPUS_PER_TASK:-32} - 1 ))
echo "[TASK ${SLURM_ARRAY_TASK_ID}] grid=$GRID out=$OUT range=$START..$((START+COUNT-1)) workers=$WORKERS"
python3 -c "import sys,torch;print('[INTERP]',sys.executable)"
echo "[CODE] $(git describe --dirty --always) on $(git rev-parse --abbrev-ref HEAD)"
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
