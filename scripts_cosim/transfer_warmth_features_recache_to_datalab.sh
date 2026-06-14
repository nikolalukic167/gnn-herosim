#!/usr/bin/env bash
# Push B1 seq recache pipeline to datalab and submit SLURM job.
set -euo pipefail

ROOT="/root/projects/my-herosim"
cd "$ROOT"

SSH_KEY="${SSH_KEY:-$HOME/.ssh/nikolalukic167}"
REMOTE="${REMOTE:-nikola.lukic@cluster.datalab.tuwien.ac.at}"
REPO="${REPO:-/home/nikola.lukic/gnn-herosim}"
RSYNC_SSH="ssh -i ${SSH_KEY}"

for f in \
  src/notebooks/prepare_graphs_cache_seq.py \
  scripts_cosim/datalab/run_warmth_features_recache.sh \
  scripts_cosim/datalab/warmth_features_recache.sbatch \
  scripts_cosim/datalab/submit_warmth_features_recache.sh; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing $f" >&2
    exit 1
  fi
done

ssh -i "$SSH_KEY" "$REMOTE" "mkdir -p \
  ${REPO}/logs \
  ${REPO}/simulation_data/graphs_cache_warmth_v2_features_v1 \
  ${REPO}/src/notebooks/non_unique_lib \
  ${REPO}/scripts_cosim/datalab"

rsync -avP -e "$RSYNC_SSH" \
  src/notebooks/prepare_graphs_cache_seq.py \
  "${REMOTE}:${REPO}/src/notebooks/"

rsync -avP -e "$RSYNC_SSH" \
  src/notebooks/non_unique_lib/ \
  "${REMOTE}:${REPO}/src/notebooks/non_unique_lib/"

rsync -avP -e "$RSYNC_SSH" \
  scripts_cosim/datalab/run_warmth_features_recache.sh \
  scripts_cosim/datalab/warmth_features_recache.sbatch \
  scripts_cosim/datalab/submit_warmth_features_recache.sh \
  "${REMOTE}:${REPO}/scripts_cosim/datalab/"

ssh -i "$SSH_KEY" "$REMOTE" \
  "cd ${REPO} && sed -i 's/\r$//' scripts_cosim/datalab/run_warmth_features_recache.sh scripts_cosim/datalab/submit_warmth_features_recache.sh && \
  bash scripts_cosim/datalab/submit_warmth_features_recache.sh"

echo "features_v1 recache submitted on datalab"
