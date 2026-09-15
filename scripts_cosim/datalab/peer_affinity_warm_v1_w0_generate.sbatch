#!/bin/bash
# peer_affinity_warm_v1 -- W0 screen: brute-force-label live-audit snapshots from the gate cell
# (docs/lineages/peer_affinity_warm_v1.md). Diagnostic only: cell_s7901 is the gate topology,
# so nothing cut from it ever enters a training corpus.
#
# One array task = one shard file of snapshots (made locally by split_warm_snapshots.py):
#   simulation_data/peer_affinity_warm_v1/w0/<SOURCE>_shard<k>.jsonl
# Datasets land in simulation_data/gnn_datasets_peer_affinity_warm_v1_w0_<SOURCE>/ds_<k*PER+i>.
#   sbatch --export=ALL,SOURCE=knb,SHARDS=10,PER=5 --array=0-9 <this>
#   sbatch --export=ALL,SOURCE=gnn,SHARDS=10,PER=5 --array=0-9 <this>
#
#SBATCH --job-name=pa-warm-w0
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/pa-warm-w0-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/pa-warm-w0-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-9

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim

if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1

# Physics + sweep env: identical to every peer_affinity training corpus (t1b / x800 generate).
export HEROSIM_PEER_EXCHANGE=1
export HEROSIM_RETAIN_TASK_TIMES=1
export HEROSIM_RETAIN_PEER_STATS=1
export MAX_PLACEMENT_COMBINATIONS_SKIP=2000000
export GNN_CAPTURE_DATASET_STATE=0
export COSIM_SUPPRESS_SIM_PRINTS=1
# The autoscaler's 1 s tick is a serving artefact; on a 6e5 s drain horizon it made one
# plan cost ~20 s / ~30 MB and OOM-killed every 64 GB task of the first W0 arrays
# (761329/761343). Label-invariant (bit-identical on the prototype sweeps).
export COSIM_AUTOSCALER_RECONCILE_INTERVAL=1e12

SOURCE="${SOURCE:?set SOURCE=knb or SOURCE=gnn}"
PER="${PER:-5}"
TARGET="${TARGET:-20000}"
W0=simulation_data/peer_affinity_warm_v1/w0
SHARD="$W0/${SOURCE}_shard${SLURM_ARRAY_TASK_ID}.jsonl"
TRACE=simulation_data/peer_affinity_live_gate/workloads/workload-150-150-peer_p2_x200.json
CELL=simulation_data/peer_affinity_live_gate/configs/cell_s7901.json
OUT="simulation_data/gnn_datasets_peer_affinity_warm_v1_w0_${SOURCE}"
[[ -f "$SHARD" ]] || { echo "FAIL LOUD: shard missing: $SHARD (rsync simulation_data/peer_affinity_warm_v1/w0)" >&2; exit 1; }
[[ -f "$TRACE" && -f "$CELL" ]] || { echo "FAIL LOUD: trace or cell config missing" >&2; exit 1; }
WORKERS=$(( ${SLURM_CPUS_PER_TASK:-32} - 1 ))
echo "[TASK ${SLURM_ARRAY_TASK_ID}] source=$SOURCE shard=$SHARD out=$OUT start=$(( SLURM_ARRAY_TASK_ID * PER )) target=$TARGET workers=$WORKERS"
python3 -c "import sys,torch;print('[INTERP]',sys.executable)"
echo "[CODE] $(git describe --dirty --always) on $(git rev-parse --abbrev-ref HEAD)"
mkdir -p logs "$OUT"

python3 -u scripts_cosim/make_warm_corpus.py \
  --snapshots "$SHARD" --trace "$TRACE" --cell-config "$CELL" \
  --output-dir "$OUT" --source-tag "${SOURCE}_s7901" \
  --start-index $(( SLURM_ARRAY_TASK_ID * PER )) --limit "$PER" \
  --target-combos "$TARGET" --workers "$WORKERS" --quiet
echo "=== task ${SLURM_ARRAY_TASK_ID} done ==="
