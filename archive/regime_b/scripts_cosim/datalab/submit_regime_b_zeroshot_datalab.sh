#!/usr/bin/env bash
# Sync Regime B zero-shot runner + verify 873/v5.5 ckpts match mitrix, then submit.
#
# Follows sealed-holdout transfer pattern (rsync models if missing/mismatched).
#   bash scripts_cosim/datalab/submit_regime_b_zeroshot_datalab.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
REMOTE_HOST="${REMOTE_HOST:-datalab}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/nikola.lukic/gnn-herosim}"
REMOTE="${REMOTE_HOST}:${REMOTE_ROOT}"
PARTITION="${PARTITION:-GPU-l40s,GPU-a40,GPU-a100s}"
GNN_REL="models/near-rtt-v2-contention-v2-873-v5.5-dim14-ce-only.pt"
MLP_REL="models/tabular/batch_edge_mlp_contention_v2_873_v5.5_dim22_batchcache.pt"
EXPECT_GNN="3efed4728536a3b9931cf37510b90232"
EXPECT_MLP="aa40dc51eb60b8c88993248378433f72"

[[ -f "$GNN_REL" ]] || { echo "ERROR: missing local $GNN_REL" >&2; exit 1; }
[[ -f "$MLP_REL" ]] || { echo "ERROR: missing local $MLP_REL" >&2; exit 1; }
local_gnn=$(md5sum "$GNN_REL" | awk '{print $1}')
local_mlp=$(md5sum "$MLP_REL" | awk '{print $1}')
[[ "$local_gnn" == "$EXPECT_GNN" ]] || {
  echo "ERROR: local GNN md5=$local_gnn != $EXPECT_GNN" >&2; exit 1
}
[[ "$local_mlp" == "$EXPECT_MLP" ]] || {
  echo "ERROR: local MLP md5=$local_mlp != $EXPECT_MLP" >&2; exit 1
}

echo "=== rsync scripts ==="
rsync -az \
  scripts_cosim/regime_b_problem_spec.py \
  scripts_cosim/regime_b_metrics.py \
  scripts_cosim/calibrate_regime_b.py \
  scripts_cosim/build_regime_b_live_stub.py \
  scripts_cosim/run_regime_b_live_stub_baselines.py \
  "${REMOTE}/scripts_cosim/"
rsync -az \
  scripts_cosim/datalab/regime_b_zeroshot.sbatch \
  scripts_cosim/datalab/submit_regime_b_zeroshot_datalab.sh \
  "${REMOTE}/scripts_cosim/datalab/"

echo "=== ensure identical 873/v5.5 ckpts on datalab ==="
need_rsync=0
if ! ssh "${REMOTE_HOST}" bash -s <<EOF
set -euo pipefail
cd ${REMOTE_ROOT}
mkdir -p models/tabular logs simulation_data/regime_b_cold_burst_v1/live_stub \\
  simulation_data/normal_sim_sweeps/regime_b_cold_burst_v1_zeroshot/results
[[ -f ${GNN_REL} ]] || exit 2
[[ -f ${MLP_REL} ]] || exit 2
g=\$(md5sum ${GNN_REL} | awk '{print \$1}')
m=\$(md5sum ${MLP_REL} | awk '{print \$1}')
echo "remote GNN md5=\$g"
echo "remote MLP md5=\$m"
[[ "\$g" == "${EXPECT_GNN}" ]] || exit 3
[[ "\$m" == "${EXPECT_MLP}" ]] || exit 3
exit 0
EOF
then
  need_rsync=1
fi
if [[ "$need_rsync" == "1" ]]; then
  echo "=== rsync models (md5 mismatch or missing) ==="
  rsync -az "$GNN_REL" "${GNN_REL}.meta.json" "${REMOTE}/models/"
  rsync -az "$MLP_REL" "${MLP_REL}.meta.json" "${REMOTE}/models/tabular/"
fi

ssh "${REMOTE_HOST}" bash -s <<EOF
set -euo pipefail
cd ${REMOTE_ROOT}
g=\$(md5sum ${GNN_REL} | awk '{print \$1}')
m=\$(md5sum ${MLP_REL} | awk '{print \$1}')
[[ "\$g" == "${EXPECT_GNN}" ]] || { echo "FAIL GNN md5=\$g"; exit 1; }
[[ "\$m" == "${EXPECT_MLP}" ]] || { echo "FAIL MLP md5=\$m"; exit 1; }
echo "OK identical ckpts: GNN=\$g MLP=\$m"
sed -i 's/\r\$//' scripts_cosim/datalab/regime_b_zeroshot.sbatch \\
  scripts_cosim/run_regime_b_live_stub_baselines.py
chmod +x scripts_cosim/datalab/submit_regime_b_zeroshot_datalab.sh
EOF

echo "=== submit zero-shot on ${PARTITION} ==="
JOB=$(ssh "${REMOTE_HOST}" bash -s <<EOF
set -euo pipefail
cd ${REMOTE_ROOT}
sbatch --partition=${PARTITION} scripts_cosim/datalab/regime_b_zeroshot.sbatch | awk '{print \$NF}'
EOF
)
echo "Submitted zero-shot job: ${JOB}"
echo "Monitor: ssh ${REMOTE_HOST} squeue -j ${JOB}"
echo "Pull later:"
echo "  rsync -az ${REMOTE}/simulation_data/normal_sim_sweeps/regime_b_cold_burst_v1_zeroshot/ \\"
echo "    simulation_data/normal_sim_sweeps/regime_b_cold_burst_v1_zeroshot/"
