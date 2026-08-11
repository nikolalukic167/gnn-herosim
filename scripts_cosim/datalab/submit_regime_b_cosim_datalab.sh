#!/usr/bin/env bash
# Sync Regime B co-sim generator + submit CPU-amd array on datalab (450 ds).
#
#   bash scripts_cosim/datalab/submit_regime_b_cosim_datalab.sh
#   ARRAY=0-0 TOTAL_DATASETS=2 bash ...   # tiny pilot
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
REMOTE_HOST="${REMOTE_HOST:-datalab}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/nikola.lukic/gnn-herosim}"
REMOTE="${REMOTE_HOST}:${REMOTE_ROOT}"
ARRAY="${ARRAY:-0-9}"
PARTITION="${PARTITION:-CPU-amd}"
CPUS="${CPUS:-64}"
MEM="${MEM:-96G}"
TIME="${TIME:-24:00:00}"

echo "=== rsync Regime B co-sim sources -> ${REMOTE} ==="
rsync -az \
  scripts_cosim/generate_gnn_datasets_fast.py \
  scripts_cosim/regime_b_problem_spec.py \
  scripts_cosim/regime_b_metrics.py \
  scripts_cosim/calibrate_regime_b.py \
  scripts_cosim/build_regime_b_live_stub.py \
  "${REMOTE}/scripts_cosim/"

rsync -az \
  scripts_cosim/datalab/run_contention_regen_shard.sh \
  scripts_cosim/datalab/regime_b_cold_burst_v1_cosim.sbatch \
  scripts_cosim/datalab/submit_regime_b_cosim_datalab.sh \
  "${REMOTE}/scripts_cosim/datalab/"

ssh "${REMOTE_HOST}" bash -s <<EOF
set -euo pipefail
cd ${REMOTE_ROOT}
mkdir -p logs simulation_data/gnn_datasets_4tasks_regime_b_cold_burst_v1 \\
  simulation_data/regime_b_cold_burst_v1/live_stub
chmod +x scripts_cosim/datalab/run_contention_regen_shard.sh \\
  scripts_cosim/datalab/submit_regime_b_cosim_datalab.sh
# Heal CRLF if any
sed -i 's/\r\$//' scripts_cosim/datalab/run_contention_regen_shard.sh \\
  scripts_cosim/datalab/regime_b_cold_burst_v1_cosim.sbatch \\
  scripts_cosim/datalab/submit_regime_b_cosim_datalab.sh
# Live stub (cheap)
if command -v micromamba >/dev/null 2>&1; then
  eval "\$(micromamba shell hook --shell bash)"
  micromamba activate gnn
fi
python3 scripts_cosim/build_regime_b_live_stub.py
python3 scripts_cosim/calibrate_regime_b.py --mode target
EOF

echo "=== submit Regime B co-sim on ${PARTITION} array=${ARRAY} ==="
JOB=$(ssh "${REMOTE_HOST}" bash -s <<EOF
set -euo pipefail
cd ${REMOTE_ROOT}
sbatch --partition=${PARTITION} --cpus-per-task=${CPUS} --mem=${MEM} --time=${TIME} \\
  --array=${ARRAY} \\
  scripts_cosim/datalab/regime_b_cold_burst_v1_cosim.sbatch | awk '{print \$NF}'
EOF
)
echo "Submitted job: ${JOB}"
echo "Monitor: ssh ${REMOTE_HOST} squeue -u \$USER -j ${JOB}"
echo "Health:  ssh ${REMOTE_HOST} 'bash ${REMOTE_ROOT}/scripts_cosim/datalab/cosim_health_report.sh simulation_data/gnn_datasets_4tasks_regime_b_cold_burst_v1'"
