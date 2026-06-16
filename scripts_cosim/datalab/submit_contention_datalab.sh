#!/usr/bin/env bash
# Sync contention scripts to datalab and submit v2 finisher + v3 generation.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
REMOTE="datalab:/home/nikola.lukic/gnn-herosim"

echo "=== rsync contention scripts + grid to datalab ==="
rsync -az \
  scripts_cosim/generate_gnn_datasets_fast.py \
  scripts_cosim/datalab/run_contention_regen_shard.sh \
  scripts_cosim/datalab/contention_v2_finish.sbatch \
  scripts_cosim/datalab/contention_v3_regen.sbatch \
  "${REMOTE}/scripts_cosim/" \
  "${REMOTE}/scripts_cosim/datalab/"

ssh datalab 'cd /home/nikola.lukic/gnn-herosim && mkdir -p logs simulation_data/gnn_datasets_4tasks_contention_v3 && chmod +x scripts_cosim/datalab/run_contention_regen_shard.sh'

echo "=== submit contention_v2 finisher (22 missing jsonl) ==="
JOB_V2=$(ssh datalab 'cd /home/nikola.lukic/gnn-herosim && sbatch scripts_cosim/datalab/contention_v2_finish.sbatch' | awk '{print $NF}')
echo "contention_v2_finish job: ${JOB_V2}"

echo "=== submit contention_v3 full grid (900 ds) ==="
JOB_V3=$(ssh datalab 'cd /home/nikola.lukic/gnn-herosim && sbatch scripts_cosim/datalab/contention_v3_regen.sbatch' | awk '{print $NF}')
echo "contention_v3_regen job: ${JOB_V3}"

echo "Done. Monitor: ssh datalab squeue -u nikola.lukic"
