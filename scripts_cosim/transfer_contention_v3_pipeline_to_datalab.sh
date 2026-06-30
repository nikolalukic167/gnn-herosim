#!/usr/bin/env bash
# Sync contention_v3 pipeline scripts to datalab and submit SLURM chain:
#   recache → (GNN train ∥ MLP train) → live gate array → compare
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SSH_KEY="${SSH_KEY:-$HOME/.ssh/nikolalukic167}"
REMOTE="${REMOTE:-nikola.lukic@cluster.datalab.tuwien.ac.at}"
REPO="/home/nikola.lukic/gnn-herosim"
RSYNC_SSH="ssh -i ${SSH_KEY} -o BatchMode=yes"

export SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/contention_v3_live_gate_20260620}"
export TIMEOUT="${TIMEOUT:-18000}"
FORCE="${FORCE:-0}"

echo "=== rsync contention_v3 pipeline to datalab ==="
rsync -avP -e "$RSYNC_SSH" \
  src/notebooks/train_near_rtt_v2_contention_v3_dim14_ce_only.py \
  src/notebooks/train_mlp_contention_v3_dim22_batchcache.py \
  "${REMOTE}:${REPO}/src/notebooks/"

rsync -avP -e "$RSYNC_SSH" \
  scripts_cosim/datalab/run_contention_v3_recache.sh \
  scripts_cosim/datalab/run_contention_v3_gnn_train.sh \
  scripts_cosim/datalab/run_contention_v3_mlp_train.sh \
  scripts_cosim/datalab/run_contention_v3_live_gate_one.sh \
  scripts_cosim/datalab/run_contention_v3_compare.sh \
  scripts_cosim/datalab/contention_v3_recache.sbatch \
  scripts_cosim/datalab/contention_v3_gnn_train.sbatch \
  scripts_cosim/datalab/contention_v3_mlp_train.sbatch \
  scripts_cosim/datalab/contention_v3_live_gate_gpu.sbatch \
  scripts_cosim/datalab/contention_v3_compare.sbatch \
  "${REMOTE}:${REPO}/scripts_cosim/datalab/"

rsync -avP -e "$RSYNC_SSH" \
  scripts_cosim/important/compare_contention_v2_live_gate.py \
  "${REMOTE}:${REPO}/scripts_cosim/important/"

rsync -avP -e "$RSYNC_SSH" \
  src/policy/tabular/train_mlp_dim22_from_batch.py \
  src/policy/tabular/constants.py \
  src/policy/tabular/reduced_features.py \
  src/policy/tabular/mlp_model.py \
  "${REMOTE}:${REPO}/src/policy/tabular/" 2>/dev/null || true

ssh -i "$SSH_KEY" -o BatchMode=yes "$REMOTE" "cd ${REPO} && \
  sed -i 's/\r$//' scripts_cosim/datalab/run_contention_v3_*.sh scripts_cosim/datalab/contention_v3_*.sbatch && \
  chmod +x scripts_cosim/datalab/run_contention_v3_*.sh && \
  mkdir -p logs logs/contention_v3_pipeline models models/tabular src/notebooks/models \
    simulation_data/normal_sim_sweeps/contention_v3_live_gate_20260620/results"

echo "=== submit SLURM pipeline (SWEEP_DIR=${SWEEP_DIR} TIMEOUT=${TIMEOUT}) ==="
ssh -i "$SSH_KEY" -o BatchMode=yes "$REMOTE" "cd ${REPO} && \
  export SWEEP_DIR='${SWEEP_DIR}' TIMEOUT='${TIMEOUT}' FORCE_RECACHE='${FORCE}' FORCE_RETRAIN='${FORCE}' FORCE_RERUN='${FORCE}'; \
  scancel -u nikola.lukic --name=cv3-recache 2>/dev/null || true; \
  scancel -u nikola.lukic --name=cv3-gnn 2>/dev/null || true; \
  scancel -u nikola.lukic --name=cv3-mlp 2>/dev/null || true; \
  scancel -u nikola.lukic --name=cv3-gate 2>/dev/null || true; \
  scancel -u nikola.lukic --name=cv3-cmp 2>/dev/null || true; \
  J1=\$(sbatch --parsable scripts_cosim/datalab/contention_v3_recache.sbatch); \
  J2=\$(sbatch --parsable --dependency=afterok:\${J1} scripts_cosim/datalab/contention_v3_gnn_train.sbatch); \
  J3=\$(sbatch --parsable --dependency=afterok:\${J1} scripts_cosim/datalab/contention_v3_mlp_train.sbatch); \
  J4=\$(sbatch --parsable --dependency=afterok:\${J2}:\${J3} --export=ALL,SWEEP_DIR='${SWEEP_DIR}',TIMEOUT='${TIMEOUT}' scripts_cosim/datalab/contention_v3_live_gate_gpu.sbatch); \
  J5=\$(sbatch --parsable --dependency=afterok:\${J4} --export=ALL,SWEEP_DIR='${SWEEP_DIR}' scripts_cosim/datalab/contention_v3_compare.sbatch); \
  echo \"recache=\${J1} gnn=\${J2} mlp=\${J3} live_gate=\${J4} compare=\${J5}\"; \
  echo \"Monitor: squeue -u nikola.lukic\"; \
  echo \"Logs: logs/cv3-* logs/contention_v3_* logs/contention_v3_pipeline/\""

echo "=== submitted ==="
