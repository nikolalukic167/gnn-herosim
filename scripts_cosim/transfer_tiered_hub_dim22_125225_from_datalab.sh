#!/usr/bin/env bash
# Pull tiered-hub dim22 125-225 sweep results from datalab.
set -euo pipefail

ROOT="/root/projects/my-herosim"
cd "$ROOT"

SSH_KEY="${SSH_KEY:-$HOME/.ssh/nikolalukic167}"
REMOTE="${REMOTE:-nikola.lukic@cluster.datalab.tuwien.ac.at}"
REPO="${REPO:-/home/nikola.lukic/gnn-herosim}"
SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/tiered_hub_gnn_mlp_125225_20260610}"
LOCAL_DIR="${ROOT}/${SWEEP_DIR}"

mkdir -p "$LOCAL_DIR"

rsync -avP -e "ssh -i ${SSH_KEY}" \
  "${REMOTE}:${REPO}/${SWEEP_DIR}/" \
  "$LOCAL_DIR/"

echo "[+] Pulled to ${LOCAL_DIR}"
echo "Quick RTT check:"
for f in "$LOCAL_DIR"/results/*_gnn_dim22.json "$LOCAL_DIR"/results/*_mlp_dim22.json; do
  [[ -f "$f" ]] || continue
  pipenv run python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(f\"{sys.argv[1].split('/')[-1]}: {d['total_rtt']}\")" "$f"
done
