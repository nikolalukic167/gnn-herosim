#!/usr/bin/env bash
# Rsync HEAD live-sim code + 873/v5.5 ckpts to datalab and submit two CPU-amd arrays:
#   1) remaining coupled-trio MLP+GNN (30 cells)
#   2) sealed-holdout re-baseline (60 cells, new dir, post-25732cf physics)
#
# Knative trio cells stay local (almost done). Do not start the local re-baseline.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DATALAB_USER="${DATALAB_USER:-nikola.lukic}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/nikolalukic167}"
DEST_HOST="${DEST_HOST:-cluster.datalab.tuwien.ac.at}"
REMOTE="${DATALAB_USER}@${DEST_HOST}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/nikola.lukic/gnn-herosim}"
SSH=(ssh -i "$SSH_KEY" -o BatchMode=yes -o StrictHostKeyChecking=accept-new)
RSYNC=(rsync -azP -e "ssh -i ${SSH_KEY} -o BatchMode=yes")

TRIO_SWEEP="simulation_data/normal_sim_sweeps/contention_v2_873_v5.5_coupled_trio_20260813"
REBASE_SWEEP="simulation_data/normal_sim_sweeps/contention_v2_873_v5.5_sealed_holdout_rebaseline_20260813"
ORIGINAL_SEALED="simulation_data/normal_sim_sweeps/contention_v2_873_v5.5_sealed_holdout_20260806"
KN_CFG="simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs"
SKEW_CFG="simulation_data/normal_sim_sweeps/atomic21_skew_configs/05_sparse_40_40_p25_degree_skew.json"

GNN_MODEL="models/near-rtt-v2-contention-v2-873-v5.5-dim14-ce-only.pt"
MLP_MODEL="models/tabular/batch_edge_mlp_contention_v2_873_v5.5_dim22_batchcache.pt"
WORKLOAD="data/nofs-ids/traces/workload-125-225.json"

[[ -f "$SSH_KEY" ]] || { echo "ERROR: SSH key missing: $SSH_KEY" >&2; exit 1; }
[[ -f "$GNN_MODEL" ]] || { echo "ERROR: missing $GNN_MODEL" >&2; exit 1; }
[[ -f "$MLP_MODEL" ]] || { echo "ERROR: missing $MLP_MODEL" >&2; exit 1; }
[[ -f "$WORKLOAD" ]] || { echo "ERROR: missing $WORKLOAD" >&2; exit 1; }

JOBS_DIR="scripts_cosim/datalab/jobs"
mkdir -p "$JOBS_DIR" logs

write_trio_jobs() {
  local out="$JOBS_DIR/coupled_trio_mlp_gnn.tsv"
  : >"$out"
  local name path seed pol
  local configs=(
    "sparse_p25|${KN_CFG}/05_sparse_40_40_p25.json"
    "sparse_p35|${KN_CFG}/00_balanced_30_30_p35.json"
    "sparse_p25_skew|${SKEW_CFG}"
  )
  for pol in mlp gnn; do
    for entry in "${configs[@]}"; do
      name="${entry%%|*}"
      path="${entry#*|}"
      for seed in 42 43 44 45 46; do
        printf '%s\t%s\t%s\t%s\n' "$pol" "$name" "$path" "$seed" >>"$out"
      done
    done
  done
  echo "wrote $out ($(wc -l <"$out") jobs)"
}

write_rebaseline_jobs() {
  local out="$JOBS_DIR/sealed_holdout_rebaseline.tsv"
  : >"$out"
  local name path seed pol
  local configs=(
    "balanced_p50|${ORIGINAL_SEALED}/configs/01_balanced_40_40_p50.json"
    "balanced_p60|${ORIGINAL_SEALED}/configs/02_balanced_50_50_p60.json"
    "client_heavy_p50|${ORIGINAL_SEALED}/configs/03_client_heavy_50_35_p50.json"
    "server_heavy_p50|${ORIGINAL_SEALED}/configs/04_server_heavy_35_50_p50.json"
  )
  for pol in knative mlp gnn; do
    for entry in "${configs[@]}"; do
      name="${entry%%|*}"
      path="${entry#*|}"
      for seed in 42 43 44 45 46; do
        printf '%s\t%s\t%s\t%s\n' "$pol" "$name" "$path" "$seed" >>"$out"
      done
    done
  done
  echo "wrote $out ($(wc -l <"$out") jobs)"
}

write_trio_jobs
write_rebaseline_jobs
sed -i 's/\r$//' "$JOBS_DIR"/*.tsv

echo "=== 1) remote mkdir ==="
"${SSH[@]}" "$REMOTE" "mkdir -p \
  ${REMOTE_ROOT}/src \
  ${REMOTE_ROOT}/models/tabular \
  ${REMOTE_ROOT}/logs \
  ${REMOTE_ROOT}/scripts_cosim/datalab/jobs \
  ${REMOTE_ROOT}/scripts_cosim/important \
  ${REMOTE_ROOT}/${TRIO_SWEEP}/results \
  ${REMOTE_ROOT}/${TRIO_SWEEP}/configs \
  ${REMOTE_ROOT}/${REBASE_SWEEP}/results \
  ${REMOTE_ROOT}/${REBASE_SWEEP}/configs \
  ${REMOTE_ROOT}/${ORIGINAL_SEALED}/configs \
  ${REMOTE_ROOT}/${KN_CFG} \
  ${REMOTE_ROOT}/simulation_data/normal_sim_sweeps/atomic21_skew_configs \
  ${REMOTE_ROOT}/data/nofs-ids/traces"

echo "=== 2) rsync live-sim code (HEAD physics + provenance) ==="
"${RSYNC[@]}" --delete --exclude '__pycache__' --exclude '*.pyc' \
  src/ "${REMOTE}:${REMOTE_ROOT}/src/"
"${RSYNC[@]}" \
  scripts_cosim/run_simulation.py \
  scripts_cosim/sweep_metrics.py \
  "${REMOTE}:${REMOTE_ROOT}/scripts_cosim/"
"${RSYNC[@]}" \
  scripts_cosim/datalab/run_sealed_holdout_one.sh \
  scripts_cosim/datalab/live_cpu_amd.sbatch \
  scripts_cosim/datalab/submit_live_cpu_amd.sh \
  scripts_cosim/datalab/transfer_and_submit_rq3_cpu_amd.sh \
  "${REMOTE}:${REMOTE_ROOT}/scripts_cosim/datalab/"
"${RSYNC[@]}" \
  scripts_cosim/datalab/jobs/coupled_trio_mlp_gnn.tsv \
  scripts_cosim/datalab/jobs/sealed_holdout_rebaseline.tsv \
  "${REMOTE}:${REMOTE_ROOT}/scripts_cosim/datalab/jobs/"
"${RSYNC[@]}" \
  scripts_cosim/important/compare_sealed_live_holdout.py \
  "${REMOTE}:${REMOTE_ROOT}/scripts_cosim/important/"

echo "=== 3) rsync models + workload + configs ==="
"${RSYNC[@]}" \
  "$GNN_MODEL" "${GNN_MODEL}.meta.json" \
  "${REMOTE}:${REMOTE_ROOT}/models/"
"${RSYNC[@]}" \
  "$MLP_MODEL" "${MLP_MODEL}.meta.json" \
  "${REMOTE}:${REMOTE_ROOT}/models/tabular/"
"${RSYNC[@]}" "$WORKLOAD" "${REMOTE}:${REMOTE_ROOT}/data/nofs-ids/traces/"
"${RSYNC[@]}" \
  "${KN_CFG}/05_sparse_40_40_p25.json" \
  "${KN_CFG}/00_balanced_30_30_p35.json" \
  "${REMOTE}:${REMOTE_ROOT}/${KN_CFG}/"
"${RSYNC[@]}" \
  "$SKEW_CFG" \
  "${REMOTE}:${REMOTE_ROOT}/simulation_data/normal_sim_sweeps/atomic21_skew_configs/"
"${RSYNC[@]}" \
  "${ORIGINAL_SEALED}/configs/" \
  "${REMOTE}:${REMOTE_ROOT}/${ORIGINAL_SEALED}/configs/"
"${RSYNC[@]}" \
  "${ORIGINAL_SEALED}/configs/" \
  "${REMOTE}:${REMOTE_ROOT}/${REBASE_SWEEP}/configs/"

echo "=== 4) CRLF heal + submit both arrays ==="
"${SSH[@]}" "$REMOTE" bash -s <<EOF
set -euo pipefail
cd ${REMOTE_ROOT}
chmod +x scripts_cosim/datalab/run_sealed_holdout_one.sh \
  scripts_cosim/datalab/submit_live_cpu_amd.sh \
  scripts_cosim/datalab/live_cpu_amd.sbatch
sed -i 's/\r\$//' scripts_cosim/datalab/*.sh scripts_cosim/datalab/*.sbatch \
  scripts_cosim/datalab/jobs/*.tsv

export HEROSIM_WARMTH_PHYSICS=node_disk_v2
export HEROSIM_REQUIRE_EXPLICIT_PHYSICS=1
export GNN_MODEL=${GNN_MODEL}
export MLP_MODEL=${MLP_MODEL}
export WORKLOAD=${WORKLOAD}
export TIMEOUT=18000

echo "--- trio MLP+GNN ---"
SWEEP_DIR=${TRIO_SWEEP} \
JOBS_TSV=scripts_cosim/datalab/jobs/coupled_trio_mlp_gnn.tsv \
JOB_NAME=cv2-trio-cpu \
  bash scripts_cosim/datalab/submit_live_cpu_amd.sh

echo "--- sealed re-baseline ---"
SWEEP_DIR=${REBASE_SWEEP} \
JOBS_TSV=scripts_cosim/datalab/jobs/sealed_holdout_rebaseline.tsv \
JOB_NAME=cv2-rebase-cpu \
  bash scripts_cosim/datalab/submit_live_cpu_amd.sh
EOF

echo "=== submitted ==="
echo "Monitor: ssh -i ${SSH_KEY} ${REMOTE} 'squeue -u ${DATALAB_USER} -n cv2-trio-cpu,cv2-rebase-cpu'"
