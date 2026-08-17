#!/usr/bin/env bash
# Cancel pending/running warmth tail jobs; submit one array=0-0 finisher with --resume.
set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

PARTITION="${PARTITION:-GPU-a100}"
CPUS="${CPUS:-64}"

sed -i 's/\r$//' \
  scripts_cosim/datalab/run_warmth_regen_range.sh \
  scripts_cosim/datalab/warmth_regen_finish.sbatch

echo "=== Cancel warmth regen jobs (pending + running) ==="
for j in $(squeue -u "$USER" -h -o "%i %j" | awk '/warmth/ {print $1}'); do
  echo "scancel ${j}"
  scancel "${j}" || true
done
sleep 2

echo
echo "=== best.json before submit ==="
find "simulation_data/gnn_datasets_4tasks_1060_warmth_v2" -name best.json 2>/dev/null | wc -l

echo
echo "=== Submit single finisher (array 0-0, --resume ds_00000..499) ==="
sbatch --test-only --partition="${PARTITION}" --cpus-per-task="${CPUS}" \
  scripts_cosim/datalab/warmth_regen_finish.sbatch
job_id=$(sbatch --partition="${PARTITION}" --cpus-per-task="${CPUS}" \
  scripts_cosim/datalab/warmth_regen_finish.sbatch | awk '{print $NF}')
echo "Submitted job ${job_id} (${job_id}_0)"

echo
squeue -u "$USER" | grep -E "warmth|JOBID" || true
