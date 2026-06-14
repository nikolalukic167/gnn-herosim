#!/usr/bin/env bash
# Pull sparse_warmth_v2 co-sim datasets from datalab to mitrix.
set -euo pipefail

ROOT="/root/projects/my-herosim"
cd "$ROOT"

SSH_KEY="${SSH_KEY:-$HOME/.ssh/nikolalukic167}"
REMOTE="${REMOTE:-nikola.lukic@cluster.datalab.tuwien.ac.at}"
REPO="${REPO:-/home/nikola.lukic/gnn-herosim}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-gnn_datasets_4tasks_sparse_warmth_v2}"

mkdir -p "simulation_data/${OUTPUT_SUBDIR}" logs

echo "[+] Rsync ${OUTPUT_SUBDIR} from datalab..."
rsync -avP -e "ssh -i ${SSH_KEY}" \
  "${REMOTE}:${REPO}/simulation_data/${OUTPUT_SUBDIR}/" \
  "simulation_data/${OUTPUT_SUBDIR}/"

done=$(find "simulation_data/${OUTPUT_SUBDIR}" -name best.json 2>/dev/null | wc -l)
echo "[+] Pull complete: ${done} best.json in simulation_data/${OUTPUT_SUBDIR}/"
