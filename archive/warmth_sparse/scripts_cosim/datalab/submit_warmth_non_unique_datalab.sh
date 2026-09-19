#!/usr/bin/env bash
# Sync non-unique backfill scripts to datalab and submit warmth_v2 + sparse_warmth_v2 arrays.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
REMOTE="datalab:/home/nikola.lukic/gnn-herosim"

PARTITION="${PARTITION:-GPU-a100s}"
CPUS="${CPUS:-32}"
TIME="${TIME:-72:00:00}"

for f in \
  scripts_cosim/generate_non_unique_placements_fast.py \
  scripts_cosim/datalab/run_warmth_non_unique_shard.sh \
  scripts_cosim/datalab/warmth_non_unique_warmth.sbatch \
  scripts_cosim/datalab/warmth_non_unique_sparse.sbatch; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing $f" >&2
    exit 1
  fi
done

sed -i 's/\r$//' \
  scripts_cosim/generate_non_unique_placements_fast.py \
  scripts_cosim/datalab/run_warmth_non_unique_shard.sh \
  scripts_cosim/datalab/warmth_non_unique_warmth.sbatch \
  scripts_cosim/datalab/warmth_non_unique_sparse.sbatch

echo "=== rsync non-unique backfill scripts to datalab ==="
rsync -az \
  scripts_cosim/generate_non_unique_placements_fast.py \
  "${REMOTE}/scripts_cosim/"
rsync -az \
  scripts_cosim/datalab/run_warmth_non_unique_shard.sh \
  scripts_cosim/datalab/warmth_non_unique_warmth.sbatch \
  scripts_cosim/datalab/warmth_non_unique_sparse.sbatch \
  "${REMOTE}/scripts_cosim/datalab/"

ssh datalab 'cd /home/nikola.lukic/gnn-herosim && chmod +x scripts_cosim/datalab/run_warmth_non_unique_shard.sh && mkdir -p logs'

echo "=== JSONL counts before submit ==="
ssh datalab 'cd /home/nikola.lukic/gnn-herosim && echo -n "warmth jsonl: " && find simulation_data/gnn_datasets_4tasks_1060_warmth_v2 -name placements.jsonl 2>/dev/null | wc -l && echo -n "sparse jsonl: " && find simulation_data/gnn_datasets_4tasks_sparse_warmth_v2 -name placements.jsonl 2>/dev/null | wc -l'

echo
echo "=== test-only schedule (${PARTITION}) ==="
ssh datalab "cd /home/nikola.lukic/gnn-herosim && sbatch --test-only --partition=${PARTITION} --cpus-per-task=${CPUS} --time=${TIME} --array=0-3 scripts_cosim/datalab/warmth_non_unique_warmth.sbatch"

warmth_job=$(ssh datalab "cd /home/nikola.lukic/gnn-herosim && sbatch --partition=${PARTITION} --cpus-per-task=${CPUS} --time=${TIME} --array=0-3 scripts_cosim/datalab/warmth_non_unique_warmth.sbatch" | awk '{print $NF}')
echo "Submitted warmth non-unique array: ${warmth_job} (500 ds, 4 shards)"

sparse_job=$(ssh datalab "cd /home/nikola.lukic/gnn-herosim && sbatch --partition=${PARTITION} --cpus-per-task=${CPUS} --time=${TIME} --array=0-3 scripts_cosim/datalab/warmth_non_unique_sparse.sbatch" | awk '{print $NF}')
echo "Submitted sparse non-unique array: ${sparse_job} (351 ds, 4 shards)"

echo "Monitor: ssh datalab squeue -u nikola.lukic"
echo "Logs: logs/non_unique_gnn_datasets_4tasks_*_array*.log"
