#!/usr/bin/env bash
# Submit a one-cell-per-task live-sim array on datalab CPU-amd.
# Run FROM the datalab login node (transfer script SSHes this).
#
# Required env: SWEEP_DIR, JOBS_TSV, JOB_NAME
# Optional: ARRAY, SLURM_TIME, CPUS, MEM
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/nikola.lukic/gnn-herosim}"
cd "$PROJECT_ROOT"

SWEEP_DIR="${SWEEP_DIR:?SWEEP_DIR required}"
JOBS_TSV="${JOBS_TSV:?JOBS_TSV required}"
JOB_NAME="${JOB_NAME:?JOB_NAME required}"
SLURM_TIME="${SLURM_TIME:-06:00:00}"
CPUS="${CPUS:-8}"
MEM="${MEM:-32G}"
PARTITION="${PARTITION:-CPU-amd}"

[[ -f "$JOBS_TSV" ]] || { echo "ERROR: missing $JOBS_TSV" >&2; exit 1; }
njobs=$(grep -cve '^[[:space:]]*$' "$JOBS_TSV" || true)
(( njobs > 0 )) || { echo "ERROR: empty jobs file $JOBS_TSV" >&2; exit 1; }
ARRAY_SPEC="${ARRAY:-0-$((njobs - 1))}"

mkdir -p logs "$SWEEP_DIR/results"
sed -i 's/\r$//' "$JOBS_TSV" scripts_cosim/datalab/live_cpu_amd.sbatch \
  scripts_cosim/datalab/run_sealed_holdout_one.sh \
  scripts_cosim/datalab/submit_live_cpu_amd.sh

echo "=== sbatch --test-only ${PARTITION} cpus=${CPUS} mem=${MEM} time=${SLURM_TIME} ==="
sbatch --test-only \
  --partition="$PARTITION" \
  --cpus-per-task="$CPUS" \
  --mem="$MEM" \
  --time="$SLURM_TIME" \
  --wrap=/bin/true

scancel -u "${USER}" --name="$JOB_NAME" 2>/dev/null || true

job=$(sbatch \
  --job-name="$JOB_NAME" \
  --output="${PROJECT_ROOT}/logs/${JOB_NAME}-%A_%a.out" \
  --error="${PROJECT_ROOT}/logs/${JOB_NAME}-%A_%a.err" \
  --partition="$PARTITION" \
  --cpus-per-task="$CPUS" \
  --mem="$MEM" \
  --time="$SLURM_TIME" \
  --array="$ARRAY_SPEC" \
  --export=ALL,SWEEP_DIR,JOBS_TSV,HEROSIM_WARMTH_PHYSICS,HEROSIM_REQUIRE_EXPLICIT_PHYSICS,QUEUE_FEATURE_CONTRACT,GNN_MODEL,MLP_MODEL,WORKLOAD,TIMEOUT \
  scripts_cosim/datalab/live_cpu_amd.sbatch | awk '{print $NF}')

echo "Submitted ${JOB_NAME} job=${job} array=${ARRAY_SPEC} n=${njobs}"
echo "PARTITION=${PARTITION} CPUS=${CPUS} MEM=${MEM} TIME=${SLURM_TIME}"
echo "SWEEP_DIR=${SWEEP_DIR}"
echo "JOBS_TSV=${JOBS_TSV}"
echo "Monitor: squeue -u \$USER -n ${JOB_NAME}"
