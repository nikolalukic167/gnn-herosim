#!/usr/bin/env bash
# Push sweep_bipartite_coordination_v1 to datalab (24 GPU jobs).
set -euo pipefail

ROOT="/root/projects/my-herosim"
cd "$ROOT"

SSH_KEY="${SSH_KEY:-$HOME/.ssh/nikolalukic167}"
REMOTE="${REMOTE:-nikola.lukic@cluster.datalab.tuwien.ac.at}"
REPO="${REPO:-/home/nikola.lukic/gnn-herosim}"
RSYNC_SSH="ssh -i ${SSH_KEY}"

SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/sweep_bipartite_coordination_v1}"
WORKLOAD="${WORKLOAD:-data/nofs-ids/traces/workload-125-225.json}"
SKEW_BASE="simulation_data/normal_sim_sweeps/atomic21_skew_configs/default_20_20_degree_skew.json"

for f in \
  models/near-rtt-v2-dim14-ce-only.pt \
  models/tabular/batch_edge_mlp.pt \
  "$WORKLOAD" \
  "$SKEW_BASE"; do
  [[ -f "$f" ]] || { echo "ERROR: missing $f" >&2; exit 1; }
done

bash scripts_cosim/important/prepare_bipartite_coordination_sweep.sh

job_count=$(($(wc -l < "${SWEEP_DIR}/configs/jobs_dim22.tsv") - 1))
if [[ "$job_count" -ne 18 ]]; then
  echo "ERROR: expected 18 jobs (k=4,6,8 × 3 seeks × 2 policies), got ${job_count}" >&2
  exit 1
fi

ssh -i "$SSH_KEY" "$REMOTE" "mkdir -p \
  ${REPO}/models/tabular \
  ${REPO}/data/nofs-ids/traces \
  ${REPO}/${SWEEP_DIR}/configs \
  ${REPO}/${SWEEP_DIR}/results \
  ${REPO}/simulation_data/normal_sim_sweeps/atomic21_skew_configs \
  ${REPO}/scripts_cosim/datalab \
  ${REPO}/scripts_cosim/important \
  ${REPO}/src/policy/gnn \
  ${REPO}/src/policy/tabular \
  ${REPO}/src \
  ${REPO}/logs"

rsync -avP -e "$RSYNC_SSH" models/near-rtt-v2-dim14-ce-only.pt "${REMOTE}:${REPO}/models/"
rsync -avP -e "$RSYNC_SSH" models/tabular/batch_edge_mlp.pt "${REMOTE}:${REPO}/models/tabular/"
rsync -avP -e "$RSYNC_SSH" "$WORKLOAD" "${REMOTE}:${REPO}/data/nofs-ids/traces/"
rsync -avP -e "$RSYNC_SSH" "$SKEW_BASE" "${REMOTE}:${REPO}/simulation_data/normal_sim_sweeps/atomic21_skew_configs/"
rsync -avP -e "$RSYNC_SSH" "${SWEEP_DIR}/configs/" "${REMOTE}:${REPO}/${SWEEP_DIR}/configs/"
rsync -avP -e "$RSYNC_SSH" scripts_cosim/datalab/ "${REMOTE}:${REPO}/scripts_cosim/datalab/"
rsync -avP -e "$RSYNC_SSH" scripts_cosim/important/prepare_bipartite_coordination_sweep.sh \
  scripts_cosim/important/generate_tiered_hub_configs.py \
  scripts_cosim/important/compare_bipartite_coordination_sweep.py \
  "${REMOTE}:${REPO}/scripts_cosim/important/"
rsync -avP -e "$RSYNC_SSH" scripts_cosim/run_simulation.py "${REMOTE}:${REPO}/scripts_cosim/"
rsync -avP -e "$RSYNC_SSH" src/executesimulation.py src/generate_infrastructure.py "${REMOTE}:${REPO}/src/"
rsync -avP -e "$RSYNC_SSH" \
  src/policy/gnn/scheduler.py \
  "${REMOTE}:${REPO}/src/policy/gnn/"
rsync -avP -e "$RSYNC_SSH" \
  src/policy/tabular/feature_builder.py \
  src/policy/tabular/mlp_scheduler.py \
  src/policy/tabular/mlp_model.py \
  src/policy/tabular/constants.py \
  src/policy/tabular/scheduler.py \
  "${REMOTE}:${REPO}/src/policy/tabular/"

if [[ -d "${SWEEP_DIR}/results" ]] && [[ -n "$(ls -A "${SWEEP_DIR}/results" 2>/dev/null)" ]]; then
  rsync -avP -e "$RSYNC_SSH" "${SWEEP_DIR}/results/" "${REMOTE}:${REPO}/${SWEEP_DIR}/results/"
fi

echo "[+] Transfer complete (${job_count} GPU jobs)."
echo "Submit on datalab:"
echo "  SWEEP_DIR=${SWEEP_DIR} WORKLOAD=${WORKLOAD} ssh -i ${SSH_KEY} ${REMOTE} \\"
echo "    'cd ${REPO} && sed -i \"s/\\r\$//\" scripts_cosim/datalab/*.sh scripts_cosim/datalab/*.sbatch && bash scripts_cosim/datalab/submit_bipartite_coordination.sh'"
