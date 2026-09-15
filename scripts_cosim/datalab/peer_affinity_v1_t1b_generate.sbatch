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

# drainable_objective_v1 (2026-09-14): optional measured backlog clock. Unset (the default)
# leaves `seed_virtual_warmup` charging a seeded queued task `execution + comm`, which is
# how every existing corpus was generated and keeps this script byte-identical for them.
# Set, it charges what a queued task actually costs under HEROSIM_PEER_EXCHANGE=1 -- the
# peer transfers and source->platform latency that formula omits. A corpus on one clock
# must never be merged with a corpus on the other, which is why the live-clock presets have
# their own seed ranges and output dirs.
if [[ -n "${DRAIN_TABLE:-}" ]]; then
  [[ -f "$DRAIN_TABLE" ]] || { echo "FAIL LOUD: DRAIN_TABLE=$DRAIN_TABLE does not exist" >&2; exit 1; }
  export HEROSIM_BACKLOG_DRAIN_TABLE="$DRAIN_TABLE"
  echo "[CLOCK] measured: $DRAIN_TABLE ($(md5sum "$DRAIN_TABLE" | cut -c1-12))"
else
  echo "[CLOCK] default: seed_virtual_warmup exec+comm"
fi

# Overridable so the SAME script extends the corpus instead of being forked per experiment
# (CLAUDE.md rule 2). Defaults reproduce the T1b expansion exactly. `FIRST` is where this
# submission starts in the preset's deterministic enumeration; index -> seed is fixed by the
# preset and the loop order, so extending with FIRST=500 TOTAL=1500 continues into the
# declared-but-unused seeds and never re-maps an existing ds_ index
# (drainable_debug_v1 Phase 3, 2026-09-14).
GRID="${GRID:-peer_affinity_screen_c3_x200_train2}"
OUT="${OUT:-gnn_datasets_peer_affinity_v1_c3_x200_train2}"
FIRST="${FIRST:-0}"
TOTAL="${TOTAL:-500}"
SHARDS="${SHARDS:-20}"
SPAN=$(( TOTAL - FIRST ))
(( SPAN > 0 )) || { echo "FAIL LOUD: TOTAL=$TOTAL must exceed FIRST=$FIRST" >&2; exit 1; }
PER=$(( (SPAN + SHARDS - 1) / SHARDS ))
START=$(( FIRST + SLURM_ARRAY_TASK_ID * PER ))
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
