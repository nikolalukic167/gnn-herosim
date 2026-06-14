#!/bin/bash
# Transfer all artifacts needed for the mega experiment matrix to datalab.
# Run from repo root on mitrix before submitting SLURM jobs.
set -euo pipefail

REMOTE="${REMOTE:-nikola.lukic@cluster.datalab.tuwien.ac.at}"
REPO="${REPO:-/home/nikola.lukic/gnn-herosim}"
SSH_KEY="${SSH_KEY:-}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

rsync_to() {
  local src="$1" dst="$2"
  if [[ -n "$SSH_KEY" ]]; then
    rsync -avz --progress -e "ssh -i $SSH_KEY" "$src" "${REMOTE}:${REPO}/${dst}"
  else
    rsync -avz --progress "$src" "${REMOTE}:${REPO}/${dst}"
  fi
}

echo "=== Transferring mega compare artifacts to datalab ==="
echo "Remote: ${REMOTE}:${REPO}"

# ── New GNN models ──────────────────────────────────────────────────────────
echo "→ GNN models"
rsync_to "${ROOT}/models/near-rtt-v2-warmth-dim14-ce-only.pt"                        models/
rsync_to "${ROOT}/models/near-rtt-v2-warmth-sparse-merged-dim14-ce-only.pt"          models/
rsync_to "${ROOT}/models/near-rtt-v2-warmth-sparse-skew-merged-dim14-ce-only.pt"     models/

# ── New MLP models ──────────────────────────────────────────────────────────
echo "→ MLP models"
rsync_to "${ROOT}/models/tabular/batch_edge_mlp_warmth_v2.pt"                                      models/tabular/
rsync_to "${ROOT}/models/tabular/batch_edge_mlp_warmth_v2.pt.meta.json"                            models/tabular/
rsync_to "${ROOT}/models/tabular/batch_edge_mlp_warmth_sparse_merged.pt"                           models/tabular/
rsync_to "${ROOT}/models/tabular/batch_edge_mlp_warmth_sparse_merged.pt.meta.json"                 models/tabular/
rsync_to "${ROOT}/models/tabular/batch_edge_mlp_warmth_sparse_skew_merged_ce_reduced.pt"           models/tabular/
rsync_to "${ROOT}/models/tabular/batch_edge_mlp_warmth_sparse_skew_merged_ce_reduced.pt.meta.json" models/tabular/

# ── Workloads ───────────────────────────────────────────────────────────────
echo "→ Workloads"
rsync_to "${ROOT}/data/nofs-ids/traces/workload-100-100.json" data/nofs-ids/traces/
rsync_to "${ROOT}/data/nofs-ids/traces/workload-125-225.json" data/nofs-ids/traces/

# ── Config files ────────────────────────────────────────────────────────────
echo "→ Config files (standard 7-config + skew + bipartite)"
rsync_to "${ROOT}/simulation_data/space_with_network.json" simulation_data/
rsync_to "${ROOT}/simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs/" \
         simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs/
rsync_to "${ROOT}/simulation_data/normal_sim_sweeps/atomic21_skew_configs/" \
         simulation_data/normal_sim_sweeps/atomic21_skew_configs/
rsync_to "${ROOT}/simulation_data/normal_sim_sweeps/sweep_bipartite_coordination_v1/configs/" \
         simulation_data/normal_sim_sweeps/sweep_bipartite_coordination_v1/configs/

# ── SLURM runner scripts ─────────────────────────────────────────────────────
echo "→ Runner scripts"
rsync_to "${ROOT}/scripts_cosim/datalab/mega_compare_all7.sbatch"          scripts_cosim/datalab/
rsync_to "${ROOT}/scripts_cosim/datalab/run_mega_compare_all7_one.sh"      scripts_cosim/datalab/
rsync_to "${ROOT}/scripts_cosim/datalab/bipartite_skew_merged_gpu.sbatch"  scripts_cosim/datalab/
rsync_to "${ROOT}/scripts_cosim/datalab/run_bipartite_skew_merged_one.sh"  scripts_cosim/datalab/
rsync_to "${ROOT}/scripts_cosim/datalab/skew3_full_gate_gpu.sbatch"        scripts_cosim/datalab/
rsync_to "${ROOT}/scripts_cosim/datalab/skew3_full_gate_knative.sbatch"    scripts_cosim/datalab/
rsync_to "${ROOT}/scripts_cosim/datalab/run_skew3_full_gate_one.sh"        scripts_cosim/datalab/
rsync_to "${ROOT}/scripts_cosim/datalab/skew4_new_models_gpu.sbatch"       scripts_cosim/datalab/
rsync_to "${ROOT}/scripts_cosim/datalab/skew4_new_models_knative.sbatch"   scripts_cosim/datalab/
rsync_to "${ROOT}/scripts_cosim/datalab/run_skew4_new_models_one.sh"       scripts_cosim/datalab/

# ── Core simulation runner ────────────────────────────────────────────────────
echo "→ Simulation runner"
rsync_to "${ROOT}/scripts_cosim/run_simulation.py" scripts_cosim/

echo ""
echo "=== Transfer complete. On datalab, fix CRLF and submit: ==="
cat <<'EOF'
  cd /home/nikola.lukic/gnn-herosim
  git pull   # or: git fetch && git checkout <branch>
  sed -i 's/\r$//' scripts_cosim/datalab/mega_compare_all7.sbatch \
                    scripts_cosim/datalab/run_mega_compare_all7_one.sh \
                    scripts_cosim/datalab/bipartite_skew_merged_gpu.sbatch \
                    scripts_cosim/datalab/run_bipartite_skew_merged_one.sh \
                    scripts_cosim/datalab/skew3_full_gate_gpu.sbatch \
                    scripts_cosim/datalab/skew3_full_gate_knative.sbatch \
                    scripts_cosim/datalab/run_skew3_full_gate_one.sh \
                    scripts_cosim/datalab/skew4_new_models_gpu.sbatch \
                    scripts_cosim/datalab/skew4_new_models_knative.sbatch \
                    scripts_cosim/datalab/run_skew4_new_models_one.sh
  chmod +x scripts_cosim/datalab/run_mega_compare_all7_one.sh \
           scripts_cosim/datalab/run_bipartite_skew_merged_one.sh \
           scripts_cosim/datalab/run_skew3_full_gate_one.sh \
           scripts_cosim/datalab/run_skew4_new_models_one.sh

  # Group 1 (42 jobs, GPU-l40s, 100-100):
  sbatch scripts_cosim/datalab/mega_compare_all7.sbatch

  # Group 2 (18 jobs, GPU-a40, 125-225, node_disk_v2):
  sbatch scripts_cosim/datalab/bipartite_skew_merged_gpu.sbatch

  # Group 3 (6 GPU + 3 Knative, node_disk_v2, 100-100):
  sbatch scripts_cosim/datalab/skew3_full_gate_gpu.sbatch
  sbatch scripts_cosim/datalab/skew3_full_gate_knative.sbatch

  # Group 4 (8 GPU + 4 Knative, node_disk_v2, 125-225):
  sbatch scripts_cosim/datalab/skew4_new_models_gpu.sbatch
  sbatch scripts_cosim/datalab/skew4_new_models_knative.sbatch

  squeue -u nikola.lukic
EOF
