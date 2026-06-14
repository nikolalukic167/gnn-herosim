#!/usr/bin/env bash
# Push sparse-merged ce-reduced hub9 sweep (27 jobs) to datalab and submit.
set -euo pipefail

ROOT="/root/projects/my-herosim"
cd "$ROOT"

SSH_KEY="${SSH_KEY:-$HOME/.ssh/nikolalukic167}"
REMOTE="${REMOTE:-nikola.lukic@cluster.datalab.tuwien.ac.at}"
REPO="${REPO:-/home/nikola.lukic/gnn-herosim}"
RSYNC_SSH="ssh -i ${SSH_KEY}"

SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/warmth_sparse_merged_ce_reduced_hub9_20260611}"
GNN_MODEL="models/near-rtt-v2-warmth-sparse-merged-ce-reduced.pt"
MLP_MODEL="models/tabular/batch_edge_mlp_warmth_sparse_merged_ce_reduced.pt"
WORKLOAD="${WORKLOAD:-data/nofs-ids/traces/workload-125-225.json}"
SKEW_BASE="simulation_data/normal_sim_sweeps/atomic21_skew_configs/default_20_20_degree_skew.json"

for f in "$GNN_MODEL" "$MLP_MODEL" "$WORKLOAD" "$SKEW_BASE"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing $f" >&2
    exit 1
  fi
done

bash scripts_cosim/important/prepare_warmth_sparse_merged_hub9_sweep.sh

job_count=$(($(wc -l < "${SWEEP_DIR}/configs/jobs_smcr_hub9.tsv") - 1))
if [[ "$job_count" -ne 27 ]]; then
  echo "ERROR: expected 27 jobs, got ${job_count}" >&2
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
  ${REPO}/src/placement \
  ${REPO}/src/policy/gnn \
  ${REPO}/src/policy/tabular \
  ${REPO}/src \
  ${REPO}/logs"

echo "[1/4] rsync models..."
rsync -avP -e "$RSYNC_SSH" "$GNN_MODEL" "${REMOTE}:${REPO}/models/"
rsync -avP -e "$RSYNC_SSH" "$MLP_MODEL" "${REMOTE}:${REPO}/models/tabular/"
rsync -avP -e "$RSYNC_SSH" "${MLP_MODEL}.meta.json" "${REMOTE}:${REPO}/models/tabular/"

echo "[2/4] rsync workload + hub configs..."
rsync -avP -e "$RSYNC_SSH" "$WORKLOAD" "${REMOTE}:${REPO}/data/nofs-ids/traces/"
rsync -avP -e "$RSYNC_SSH" "$SKEW_BASE" "${REMOTE}:${REPO}/simulation_data/normal_sim_sweeps/atomic21_skew_configs/"
rsync -avP -e "$RSYNC_SSH" "${SWEEP_DIR}/configs/" "${REMOTE}:${REPO}/${SWEEP_DIR}/configs/"

echo "[3/4] rsync inference code + datalab scripts..."
rsync -avP -e "$RSYNC_SSH" \
  scripts_cosim/datalab/run_warmth_sparse_merged_ce_reduced_hub9_one.sh \
  scripts_cosim/datalab/warmth_sparse_merged_ce_reduced_hub9_gpu.sbatch \
  scripts_cosim/datalab/submit_warmth_sparse_merged_ce_reduced_hub9.sh \
  "${REMOTE}:${REPO}/scripts_cosim/datalab/"
rsync -avP -e "$RSYNC_SSH" \
  scripts_cosim/important/prepare_warmth_sparse_merged_hub9_sweep.sh \
  scripts_cosim/important/generate_tiered_hub_configs.py \
  "${REMOTE}:${REPO}/scripts_cosim/important/"
rsync -avP -e "$RSYNC_SSH" scripts_cosim/run_simulation.py "${REMOTE}:${REPO}/scripts_cosim/"
rsync -avP -e "$RSYNC_SSH" \
  src/executesimulation.py \
  src/generate_infrastructure.py \
  "${REMOTE}:${REPO}/src/"
rsync -avP -e "$RSYNC_SSH" \
  src/placement/warmth.py \
  src/placement/simulation.py \
  "${REMOTE}:${REPO}/src/placement/"
rsync -avP -e "$RSYNC_SSH" \
  src/policy/tabular/feature_builder.py \
  src/policy/tabular/mlp_scheduler.py \
  src/policy/tabular/mlp_model.py \
  src/policy/tabular/reduced_features.py \
  src/policy/tabular/constants.py \
  "${REMOTE}:${REPO}/src/policy/tabular/"
rsync -avP -e "$RSYNC_SSH" \
  src/policy/gnn/seq_decode.py \
  src/policy/gnn/gnn_model.py \
  src/policy/gnn/scheduler.py \
  "${REMOTE}:${REPO}/src/policy/gnn/"

ssh -i "$SSH_KEY" "$REMOTE" "chmod +x \
  ${REPO}/scripts_cosim/datalab/run_warmth_sparse_merged_ce_reduced_hub9_one.sh \
  ${REPO}/scripts_cosim/datalab/submit_warmth_sparse_merged_ce_reduced_hub9.sh \
  ${REPO}/scripts_cosim/important/prepare_warmth_sparse_merged_hub9_sweep.sh && \
  sed -i 's/\r$//' ${REPO}/scripts_cosim/datalab/*.sh ${REPO}/scripts_cosim/datalab/*.sbatch"

echo "[4/4] submit on datalab..."
submit_out=$(ssh -i "$SSH_KEY" "$REMOTE" "cd ${REPO} && bash scripts_cosim/datalab/submit_warmth_sparse_merged_ce_reduced_hub9.sh")
echo "$submit_out"

job_id=$(echo "$submit_out" | sed -n 's/^JOB_ID=//p')
partition=$(echo "$submit_out" | sed -n 's/^PARTITION=//p')

if [[ -z "$job_id" ]]; then
  echo "ERROR: submit did not return JOB_ID" >&2
  exit 1
fi

echo ""
echo "Waiting 60s then checking queue..."
sleep 60

ssh -i "$SSH_KEY" "$REMOTE" "squeue -u nikola.lukic -j ${job_id} -o '%.10i %.12P %.16j %.2t %.10M %R' ; echo '---' ; squeue -u nikola.lukic -n wsmcr-hub9-${partition#GPU-} -o '%.10i %.2t %.10M %R' 2>/dev/null || true"

echo ""
echo "Transfer + submit complete."
echo "Sweep: ${SWEEP_DIR}"
echo "Job: ${job_id} on ${partition:-unknown}"
