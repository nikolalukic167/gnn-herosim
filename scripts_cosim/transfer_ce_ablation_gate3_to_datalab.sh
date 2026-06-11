#!/usr/bin/env bash
# Push legacy-1060 CE ablation artifacts + training cache to datalab.
set -euo pipefail

ROOT="/root/projects/my-herosim"
cd "$ROOT"

SSH_KEY="${SSH_KEY:-$HOME/.ssh/nikolalukic167}"
REMOTE="${REMOTE:-nikola.lukic@cluster.datalab.tuwien.ac.at}"
REPO="${REPO:-/home/nikola.lukic/gnn-herosim}"
RSYNC_SSH="ssh -i ${SSH_KEY}"

WORKLOAD="data/nofs-ids/traces/workload-100-100.json"
CFG_SRC="simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs"
BASE_CONFIG="simulation_data/space_with_network.json"
SWEEP_DIR="simulation_data/normal_sim_sweeps/ce_ablation_gate3_20260611"

MODELS=(
  models/near-rtt-ce-reduced-features.pt
  models/near-rtt-v2-dim14-1060.pt
)

for f in "$WORKLOAD" "$BASE_CONFIG"; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing $f" >&2
    exit 1
  fi
done
for m in "${MODELS[@]}"; do
  if [[ ! -f "$m" ]]; then
    echo "ERROR: missing $m" >&2
    exit 1
  fi
done
ssh -i "$SSH_KEY" "$REMOTE" "mkdir -p ${REPO}/models ${REPO}/data/nofs-ids/traces \
  ${REPO}/${CFG_SRC} ${REPO}/${SWEEP_DIR}/results \
  ${REPO}/scripts_cosim/datalab ${REPO}/logs \
  ${REPO}/src/policy/gnn ${REPO}/src/policy/tabular"

echo "[1/4] rsync inference code (ce_reduced layout)..."
rsync -avP -e "$RSYNC_SSH" src/executesimulation.py "${REMOTE}:${REPO}/src/"
rsync -avP -e "$RSYNC_SSH" src/policy/gnn/gnn_model.py src/policy/gnn/seq_decode.py \
  "${REMOTE}:${REPO}/src/policy/gnn/"
rsync -avP -e "$RSYNC_SSH" src/policy/tabular/feature_builder.py \
  "${REMOTE}:${REPO}/src/policy/tabular/"

echo "[2/4] rsync datalab scripts..."
rsync -avP -e "$RSYNC_SSH" \
  scripts_cosim/datalab/run_ce_ablation_gate3_one.sh \
  scripts_cosim/datalab/ce_ablation_gate3_gpu.sbatch \
  scripts_cosim/datalab/submit_ce_ablation_gate3.sh \
  "${REMOTE}:${REPO}/scripts_cosim/datalab/"

echo "[3/4] rsync models..."
for m in "${MODELS[@]}"; do
  rsync -avP -e "$RSYNC_SSH" "$m" "${REMOTE}:${REPO}/models/"
done

echo "[4/4] rsync workload + configs..."
rsync -avP -e "$RSYNC_SSH" "$WORKLOAD" "${REMOTE}:${REPO}/data/nofs-ids/traces/"
rsync -avP -e "$RSYNC_SSH" "$BASE_CONFIG" "${REMOTE}:${REPO}/simulation_data/"
rsync -avP -e "$RSYNC_SSH" "${CFG_SRC}/01_balanced_40_40_p50.json" \
  "${CFG_SRC}/05_sparse_40_40_p25.json" \
  "${REMOTE}:${REPO}/${CFG_SRC}/"

echo "[+] Transfer complete (sweep-only; no training cache)."
echo "    Submit: ssh -i ${SSH_KEY} ${REMOTE} 'cd ${REPO} && bash scripts_cosim/datalab/submit_ce_ablation_gate3.sh'"
