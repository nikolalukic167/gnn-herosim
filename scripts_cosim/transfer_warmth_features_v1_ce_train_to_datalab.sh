#!/usr/bin/env bash
# Push CE-only train pipeline to datalab and submit (cache already on datalab from recache).
set -euo pipefail

ROOT="/root/projects/my-herosim"
cd "$ROOT"

SSH_KEY="${SSH_KEY:-$HOME/.ssh/nikolalukic167}"
REMOTE="${REMOTE:-nikola.lukic@cluster.datalab.tuwien.ac.at}"
REPO="${REPO:-/home/nikola.lukic/gnn-herosim}"
RSYNC_SSH="ssh -i ${SSH_KEY}"

ssh -i "$SSH_KEY" "$REMOTE" "mkdir -p \
  ${REPO}/logs ${REPO}/models ${REPO}/src/notebooks/models \
  ${REPO}/src/notebooks/non_unique_lib ${REPO}/src/policy/tabular \
  ${REPO}/src/policy/gnn ${REPO}/scripts_cosim/datalab"

rsync -avP -e "$RSYNC_SSH" \
  src/notebooks/train_near_rtt_v2_warmth_features_v1_ce_only.py \
  src/notebooks/train_near_rtt.py \
  "${REMOTE}:${REPO}/src/notebooks/"

rsync -avP -e "$RSYNC_SSH" \
  src/notebooks/non_unique_lib/ \
  "${REMOTE}:${REPO}/src/notebooks/non_unique_lib/"

rsync -avP -e "$RSYNC_SSH" \
  src/policy/tabular/constants.py \
  "${REMOTE}:${REPO}/src/policy/tabular/"
rsync -avP -e "$RSYNC_SSH" \
  src/policy/gnn/gnn_model.py \
  "${REMOTE}:${REPO}/src/policy/gnn/"

rsync -avP -e "$RSYNC_SSH" \
  scripts_cosim/datalab/run_warmth_features_v1_ce_train.sh \
  scripts_cosim/datalab/warmth_features_v1_ce_train.sbatch \
  scripts_cosim/datalab/submit_warmth_features_v1_ce_train.sh \
  "${REMOTE}:${REPO}/scripts_cosim/datalab/"

ssh -i "$SSH_KEY" "$REMOTE" \
  "cd ${REPO} && sed -i 's/\r$//' scripts_cosim/datalab/run_warmth_features_v1_ce_train.sh scripts_cosim/datalab/submit_warmth_features_v1_ce_train.sh && \
  bash scripts_cosim/datalab/submit_warmth_features_v1_ce_train.sh"

echo "features_v1 CE-only train submitted on datalab"
