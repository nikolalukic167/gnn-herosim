#!/usr/bin/env bash
# Pull all 4 experiment group results back from datalab to mitrix.
# Run from repo root after SLURM jobs complete.
set -euo pipefail

REMOTE="${REMOTE:-nikola.lukic@cluster.datalab.tuwien.ac.at}"
REPO="${REPO:-/home/nikola.lukic/gnn-herosim}"
SSH_KEY="${SSH_KEY:-}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

rsync_from() {
  local src="$1" dst="$2"
  mkdir -p "${ROOT}/${dst}"
  if [[ -n "$SSH_KEY" ]]; then
    rsync -avz --progress -e "ssh -i $SSH_KEY" "${REMOTE}:${REPO}/${src}" "${ROOT}/${dst}/"
  else
    rsync -avz --progress "${REMOTE}:${REPO}/${src}" "${ROOT}/${dst}/"
  fi
}

echo "=== Pulling mega compare results from datalab ==="

rsync_from "simulation_data/normal_sim_sweeps/mega_compare_all7_20260614/results/" \
           "simulation_data/normal_sim_sweeps/mega_compare_all7_20260614/results"

rsync_from "simulation_data/normal_sim_sweeps/bipartite_v2_skew_merged_20260614/results/" \
           "simulation_data/normal_sim_sweeps/bipartite_v2_skew_merged_20260614/results"

rsync_from "simulation_data/normal_sim_sweeps/skew3_full_gate_20260614/results/" \
           "simulation_data/normal_sim_sweeps/skew3_full_gate_20260614/results"

rsync_from "simulation_data/normal_sim_sweeps/skew4_new_models_20260614/results/" \
           "simulation_data/normal_sim_sweeps/skew4_new_models_20260614/results"

echo ""
echo "=== Transfer done. Run analysis: ==="
echo "  pipenv run python3 scripts_cosim/important/compare_mega_matrix.py"
