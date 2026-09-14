#!/bin/bash
# drainable_debug_v1 -- D1, step 2: brute-force-label the captured live batch states.
# One array task = one shard of one source's snapshots.
#   SOURCE=knb  sbatch --export=ALL --array=0-9 <this>     (per source)
#
# Same pipeline and the same physics env as peer_affinity_warm_v1's W0 generate, with one
# addition: --force-candidates-from-plans keeps the replicas the live shortest-queue rule
# would use inside the candidate subset, so that plan is present in the enumerated sweep and
# D1 can score it on every dataset instead of only where the draw happened to include it.
#
#SBATCH --job-name=dbg-d1-gen
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/dbg-d1-gen-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/dbg-d1-gen-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=08:00:00

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$PWD" PYTHONUNBUFFERED=1

# Physics + sweep env: identical to every peer_affinity corpus and to the W0/W1 warm cuts.
export HEROSIM_PEER_EXCHANGE=1
export HEROSIM_RETAIN_TASK_TIMES=1
export HEROSIM_RETAIN_PEER_STATS=1
export MAX_PLACEMENT_COMBINATIONS_SKIP=2000000
export GNN_CAPTURE_DATASET_STATE=0
export COSIM_SUPPRESS_SIM_PRINTS=1
# The autoscaler's 1 s tick is a serving artefact and OOM-kills a long-horizon sweep;
# label-invariant (bit-identical on the W0 prototype sweeps).
export COSIM_AUTOSCALER_RECONCILE_INTERVAL=1e12

SOURCE="${SOURCE:?set SOURCE=knb|gnn|mpoff}"
PER="${PER:-12}"
TARGET="${TARGET:-20000}"
D1=simulation_data/drainable_debug_v1/d1
SHARD="$D1/shards/${SOURCE}_shard${SLURM_ARRAY_TASK_ID}.jsonl"
TRACE="${TRACE:-simulation_data/peer_affinity_live_gate/workloads/drainable_f4000_n50000.json}"
CELL="${CELL:-simulation_data/peer_affinity_live_gate/configs/cell_s7901_f4000_pg16.json}"
OUT="simulation_data/gnn_datasets_drainable_debug_v1_d1_${SOURCE}"
[[ -f "$SHARD" ]] || { echo "FAIL LOUD: shard missing: $SHARD" >&2; exit 1; }
[[ -f "$TRACE" && -f "$CELL" ]] || { echo "FAIL LOUD: trace or cell config missing" >&2; exit 1; }
WORKERS=$(( ${SLURM_CPUS_PER_TASK:-32} - 1 ))
echo "[TASK ${SLURM_ARRAY_TASK_ID}] source=$SOURCE shard=$SHARD out=$OUT start=$(( SLURM_ARRAY_TASK_ID * PER )) workers=$WORKERS"
echo "[CODE] $(git describe --dirty --always) on $(git rev-parse --abbrev-ref HEAD)"
mkdir -p logs "$OUT"

python3 -u scripts_cosim/make_warm_corpus.py \
  --snapshots "$SHARD" --trace "$TRACE" --cell-config "$CELL" \
  --output-dir "$OUT" --source-tag "${SOURCE}_f4000" \
  --start-index $(( SLURM_ARRAY_TASK_ID * PER )) --limit "$PER" \
  --target-combos "$TARGET" --workers "$WORKERS" \
  --force-candidates-from-plans --quiet
echo "=== task ${SLURM_ARRAY_TASK_ID} done ==="
