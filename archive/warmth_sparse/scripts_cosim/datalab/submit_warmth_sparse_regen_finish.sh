#!/usr/bin/env bash
# Submit single-job sparse_warmth_v2 finisher (--resume ds_0..350, 48 gaps).
set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

PARTITION="${PARTITION:-GPU-a100s}"
CPUS="${CPUS:-128}"

sed -i 's/\r$//' \
  scripts_cosim/datalab/warmth_sparse_regen_finish.sbatch \
  scripts_cosim/datalab/run_warmth_regen_range.sh

echo "=== Cancel pending sparse-finish jobs ==="
for j in $(squeue -u "$USER" -h -o "%i %j" | awk '/sparse-finish/ {print $1}'); do
  echo "scancel ${j}"
  scancel "${j}" || true
done
sleep 1

done=0
for i in $(seq 0 350); do
  [ -f "simulation_data/gnn_datasets_4tasks_sparse_warmth_v2/ds_$(printf '%05d' "$i")/best.json" ] && done=$((done + 1))
done
echo "best.json before submit: ${done}/351"

echo
echo "=== Partition check ==="
sinfo -p "${PARTITION}" -o "%P %C"

echo
echo "Test-only (${PARTITION}, ${CPUS} CPUs):"
sbatch --test-only \
  --partition="${PARTITION}" \
  --cpus-per-task="${CPUS}" \
  scripts_cosim/datalab/warmth_sparse_regen_finish.sbatch

job_id=$(sbatch \
  --job-name=sparse-finish \
  --partition="${PARTITION}" \
  --cpus-per-task="${CPUS}" \
  scripts_cosim/datalab/warmth_sparse_regen_finish.sbatch | awk '{print $NF}')

echo "Submitted sparse finisher: ${job_id} (${CPUS} CPUs, --resume ds_0..350)"
echo "Monitor: squeue -u \$USER · tail -f logs/sparse-finish-${job_id}.out"
