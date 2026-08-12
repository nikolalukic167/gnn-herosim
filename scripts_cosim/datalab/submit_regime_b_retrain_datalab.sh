#!/usr/bin/env bash
# Sync Regime B retrain sources → datalab, submit recache → GNN → MLP → zeroshot.
#
#   bash scripts_cosim/datalab/submit_regime_b_retrain_datalab.sh
#
# Dependency chain via sbatch --dependency=afterok.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
REMOTE_HOST="${REMOTE_HOST:-datalab}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/nikola.lukic/gnn-herosim}"
REMOTE="${REMOTE_HOST}:${REMOTE_ROOT}"

echo "=== rsync Regime B pull-obs retrain sources -> ${REMOTE} ==="
rsync -az \
  src/notebooks/train_near_rtt_v2_regime_b_cold_burst_v1_dim16_ce_only.py \
  src/notebooks/train_mlp_regime_b_cold_burst_v1_dim24_batchcache.py \
  src/notebooks/prepare_graphs_cache.py \
  src/notebooks/train_near_rtt.py \
  "${REMOTE}/src/notebooks/"

rsync -az \
  scripts_cosim/regime_b_problem_spec.py \
  scripts_cosim/regime_b_metrics.py \
  scripts_cosim/build_regime_b_live_stub.py \
  scripts_cosim/run_regime_b_live_stub_baselines.py \
  scripts_cosim/calibrate_regime_b.py \
  scripts_cosim/validate_training_cache_contract.py \
  scripts_cosim/refresh_optimal_full_stats.py \
  scripts_cosim/test_pull_observables.py \
  "${REMOTE}/scripts_cosim/"

rsync -az \
  scripts_cosim/datalab/run_regime_b_recache.sh \
  scripts_cosim/datalab/run_regime_b_gnn_train.sh \
  scripts_cosim/datalab/run_regime_b_mlp_train.sh \
  scripts_cosim/datalab/regime_b_recache.sbatch \
  scripts_cosim/datalab/regime_b_gnn_train.sbatch \
  scripts_cosim/datalab/regime_b_mlp_train.sbatch \
  scripts_cosim/datalab/regime_b_oracle_split_zeroshot.sbatch \
  scripts_cosim/datalab/submit_regime_b_retrain_datalab.sh \
  "${REMOTE}/scripts_cosim/datalab/"

rsync -az \
  src/placement/warmth.py \
  "${REMOTE}/src/placement/"

rsync -az \
  src/policy/tabular/train_mlp_dim22_from_batch.py \
  src/policy/tabular/reduced_features.py \
  src/policy/tabular/feature_builder.py \
  src/policy/tabular/mlp_scheduler.py \
  "${REMOTE}/src/policy/tabular/"

rsync -az \
  src/executesimulation.py \
  "${REMOTE}/src/"

ssh "${REMOTE_HOST}" bash -s <<EOF
set -euo pipefail
cd ${REMOTE_ROOT}
mkdir -p logs logs/regime_b_pipeline models models/tabular \\
  simulation_data/graphs_cache_regime_b_cold_burst_v1 \\
  simulation_data/regime_b_cold_burst_v1/live_stub_oracle_split_v1 \\
  simulation_data/normal_sim_sweeps/regime_b_cold_burst_v1_oracle_split_v1_pullobs_zeroshot
chmod +x scripts_cosim/datalab/run_regime_b_*.sh \\
  scripts_cosim/datalab/submit_regime_b_retrain_datalab.sh
sed -i 's/\r\$//' scripts_cosim/datalab/run_regime_b_*.sh \\
  scripts_cosim/datalab/regime_b_*.sbatch \\
  scripts_cosim/datalab/submit_regime_b_retrain_datalab.sh
EOF

echo "=== submit dependency chain: recache → gnn → mlp → zeroshot ==="
JOBS=$(ssh "${REMOTE_HOST}" bash -s <<'EOF'
set -euo pipefail
cd /home/nikola.lukic/gnn-herosim
export FORCE_RECACHE=1
RECACHE=$(sbatch --parsable scripts_cosim/datalab/regime_b_recache.sbatch)
GNN=$(sbatch --parsable --dependency=afterok:${RECACHE} scripts_cosim/datalab/regime_b_gnn_train.sbatch)
MLP=$(sbatch --parsable --dependency=afterok:${GNN} scripts_cosim/datalab/regime_b_mlp_train.sbatch)
ZS=$(sbatch --parsable --dependency=afterok:${MLP} scripts_cosim/datalab/regime_b_oracle_split_zeroshot.sbatch)
echo "RECACHE=${RECACHE}"
echo "GNN=${GNN}"
echo "MLP=${MLP}"
echo "ZEROSHOT=${ZS}"
EOF
)
echo "$JOBS"
echo "Monitor: ssh ${REMOTE_HOST} squeue -u \$USER"
echo "When done, pull:"
echo "  rsync -az ${REMOTE}/models/near-rtt-v2-regime-b-cold-burst-v1-dim16-ce-only.pt* models/"
echo "  rsync -az ${REMOTE}/models/tabular/batch_edge_mlp_regime_b_cold_burst_v1_dim24_batchcache.pt* models/tabular/"
echo "  rsync -az ${REMOTE}/simulation_data/normal_sim_sweeps/regime_b_cold_burst_v1_oracle_split_v1_pullobs_zeroshot/ \\"
echo "    simulation_data/normal_sim_sweeps/regime_b_cold_burst_v1_oracle_split_v1_pullobs_zeroshot/"
