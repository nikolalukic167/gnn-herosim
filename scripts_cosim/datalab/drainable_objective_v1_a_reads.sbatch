#!/bin/bash
# drainable_objective_v1 -- Phase A reads (docs/lineages/drainable_objective_v1.md).
# No training, no simulation: three reads over artifacts drainable_debug_v1 already
# produced. Bars are constants at the top of each read tool and were committed before
# this job ran.
#
#   A1  clock gap        the D1 live-audit captures vs seed_virtual_warmup's formula.
#                        Emits the measured drain table Phase B generates its corpus with.
#   A2  label reversal   the 151 D1 co-sim states, every sweep row relabelled at
#                        V in {0.5, 1, 2}; does the shaped label prefer the reactive plan?
#                        Carries the rank-stability control inherited from the
#                        horizon-return stop.
#   A3  externality      D2's six retained raw arms (4.4 GB of per-task records): does the
#                        closed form track the queueing that actually happened?
#
# A3 is the memory-hungry one -- 780 MB of JSON per arm, parsed one arm at a time.
#
#SBATCH --job-name=dobj-a-reads
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/dobj-a-reads-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/dobj-a-reads-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=6:00:00
#SBATCH --array=0-2

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim

if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=""
export PYTHONHASHSEED=0

echo "[INTERP] $(python3 -c 'import sys;print(sys.executable)')"
echo "[CODE] $(git describe --dirty --always) on $(git rev-parse --abbrev-ref HEAD)"

OUT=simulation_data/drainable_objective_v1
mkdir -p "$OUT" logs

D1=simulation_data/drainable_debug_v1/d1
D2=simulation_data/peer_affinity_live_gate/results/dbg_D2_f4000_pg16
# The gate rung's realised arrival rate: drainable_f4000_n50000.json delivers 50,000
# events at 0.46 tasks/s (drainable_regime_v1 S0). The label's lambda is a property of
# the deployment it is for, so it is stated here and echoed into every result.
RATE="${RATE:-0.46}"
echo "[KNOBS] RATE=$RATE  D1=$D1  D2=$D2"

T=${SLURM_ARRAY_TASK_ID}

if [[ $T -eq 0 ]]; then
  echo "=== A1: the backlog clock ==="
  for f in knb gnn mpoff; do
    [[ -f "$D1/snapshots/$f.jsonl" ]] || { echo "FAIL LOUD: missing $D1/snapshots/$f.jsonl" >&2; exit 1; }
  done
  python3 scripts_cosim/drainable_objective_v1_clock_read.py \
    --snapshots "$D1/snapshots/knb.jsonl" \
    --snapshots "$D1/snapshots/gnn.jsonl" \
    --snapshots "$D1/snapshots/mpoff.jsonl" \
    --corpus simulation_data/gnn_datasets_peer_affinity_v1_c3_x200_train \
    --corpus simulation_data/gnn_datasets_peer_affinity_v1_c3_x200_train2 \
    --task-types data/nofs-ids/task-types.json \
    --out "$OUT/a1_clock_read.json" \
    --drain-table "$OUT/drain_table.json"

elif [[ $T -eq 1 ]]; then
  echo "=== A2: does the shaped label prefer what wins the stream? ==="
  DECODED=simulation_data/drainable_debug_v1/d1_decoded.json
  [[ -f "$DECODED" ]] || { echo "FAIL LOUD: missing $DECODED (D1's decode)" >&2; exit 1; }
  for src in knb gnn mpoff; do
    CORPUS="simulation_data/gnn_datasets_drainable_debug_v1_d1_${src}"
    [[ -d "$CORPUS" ]] || { echo "FAIL LOUD: missing $CORPUS" >&2; exit 1; }
    for ARM in gnn mpoff; do
      echo "--- source $src, arm $ARM ---"
      python3 scripts_cosim/drainable_objective_v1_label_reversal_read.py \
        --corpus "$CORPUS" \
        --decoded "$DECODED" \
        --arm "$ARM" \
        --arrival-rate "$RATE" \
        --task-types data/nofs-ids/task-types.json \
        --out "$OUT/a2_${src}_${ARM}.json"
    done
  done

elif [[ $T -eq 2 ]]; then
  echo "=== A3: does the closed form track the realized externality? ==="
  ARGS=()
  for arm in knative_network knative_network_batch gnn_s1 gnn_s2 mpoff_s1 mpoff_s2; do
    [[ -f "$D2/$arm.json" ]] || { echo "FAIL LOUD: missing raw result $D2/$arm.json -- D2 kept these with KEEP_RAW=1; without them this read cannot run" >&2; exit 1; }
    ARGS+=(--raw "$arm=$D2/$arm.json")
  done
  python3 scripts_cosim/drainable_objective_v1_externality_read.py \
    "${ARGS[@]}" --out "$OUT/a3_externality.json"
fi

echo "[DONE] task $T"
