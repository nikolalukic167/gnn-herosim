#!/usr/bin/env bash
# Push strategic-merge + weighted-merge live gate artifacts to datalab and submit SLURM.
set -euo pipefail

ROOT="/root/projects/my-herosim"
cd "$ROOT"

SSH_KEY="${SSH_KEY:-$HOME/.ssh/nikolalukic167}"
REMOTE="${REMOTE:-nikola.lukic@cluster.datalab.tuwien.ac.at}"
REPO="${REPO:-/home/nikola.lukic/gnn-herosim}"
RSYNC_SSH="ssh -i ${SSH_KEY}"

WORKLOAD="data/nofs-ids/traces/workload-125-225.json"
WSSM_SWEEP="simulation_data/normal_sim_sweeps/strategic_merge_wss_live_gate_20260616"
CONT_SWEEP="simulation_data/normal_sim_sweeps/strategic_merge_contention_live_gate_20260616"
WT_SWEEP="simulation_data/normal_sim_sweeps/merged_contention_weighted_live_gate_20260616_105500"
BIPARTITE_CFG="simulation_data/normal_sim_sweeps/sweep_bipartite_coordination_v1/configs"
KN_CFG="simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs"
SKEW_CFG="simulation_data/normal_sim_sweeps/atomic21_skew_configs"

MODELS=(
  models/near-rtt-v2-strategic-merge-wss-cont-v2-dim14-ce-only.pt
  models/tabular/batch_edge_mlp_strategic_merge_wss_cont_v2_dim22_batchcache.pt
  models/near-rtt-v2-warmth-sparse-contention-weighted-dim14-ce-only.pt
  models/tabular/batch_edge_mlp_warmth_sparse_contention_v2_weighted_dim22_batchcache.pt
)

for f in "$WORKLOAD" "${MODELS[@]}"; do
  [[ -f "$f" ]] || { echo "ERROR: missing $f" >&2; exit 1; }
done

ssh -i "$SSH_KEY" "$REMOTE" "mkdir -p \
  ${REPO}/models/tabular \
  ${REPO}/data/nofs-ids/traces \
  ${REPO}/${WSSM_SWEEP}/results \
  ${REPO}/${CONT_SWEEP}/results \
  ${REPO}/${WT_SWEEP}/results \
  ${REPO}/${BIPARTITE_CFG} \
  ${REPO}/${KN_CFG} \
  ${REPO}/${SKEW_CFG} \
  ${REPO}/scripts_cosim/datalab \
  ${REPO}/scripts_cosim/important \
  ${REPO}/scripts_cosim \
  ${REPO}/src/policy/gnn \
  ${REPO}/src/policy/tabular \
  ${REPO}/src \
  ${REPO}/logs"

for m in "${MODELS[@]}"; do
  rsync -avP -e "$RSYNC_SSH" "$m" "${REMOTE}:${REPO}/$(dirname "$m")/"
done

rsync -avP -e "$RSYNC_SSH" "$WORKLOAD" "${REMOTE}:${REPO}/data/nofs-ids/traces/"
rsync -avP -e "$RSYNC_SSH" "${BIPARTITE_CFG}/" "${REMOTE}:${REPO}/${BIPARTITE_CFG}/"
rsync -avP -e "$RSYNC_SSH" "${KN_CFG}/" "${REMOTE}:${REPO}/${KN_CFG}/"
rsync -avP -e "$RSYNC_SSH" "${SKEW_CFG}/" "${REMOTE}:${REPO}/${SKEW_CFG}/"

for sweep in "$WSSM_SWEEP" "$CONT_SWEEP" "$WT_SWEEP"; do
  if [[ -d "${sweep}/results" ]] && [[ -n "$(ls -A "${sweep}/results" 2>/dev/null)" ]]; then
    rsync -avP -e "$RSYNC_SSH" "${sweep}/results/" "${REMOTE}:${REPO}/${sweep}/results/"
  fi
done

rsync -avP -e "$RSYNC_SSH" scripts_cosim/datalab/ "${REMOTE}:${REPO}/scripts_cosim/datalab/"
rsync -avP -e "$RSYNC_SSH" scripts_cosim/run_simulation.py "${REMOTE}:${REPO}/scripts_cosim/"
rsync -avP -e "$RSYNC_SSH" \
  scripts_cosim/important/compare_wssm_expanded_live_gate.py \
  scripts_cosim/important/compare_contention_v2_live_gate.py \
  scripts_cosim/important/compare_merged_contention_live_gate.py \
  "${REMOTE}:${REPO}/scripts_cosim/important/"
rsync -avP -e "$RSYNC_SSH" \
  src/executesimulation.py \
  src/generate_infrastructure.py \
  "${REMOTE}:${REPO}/src/"
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

echo "[+] Transfer complete."
echo "[+] To rsync results only (no SLURM): bash scripts_cosim/transfer_strategic_merge_live_gates_results_to_datalab.sh"
echo "[+] To submit SLURM: bash scripts_cosim/datalab/submit_strategic_merge_live_gates.sh"
