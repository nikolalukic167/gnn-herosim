#!/usr/bin/env bash
# Pull the siv1 5-seed trio results back from datalab, reuse the Knative arm, and seal the
# sweep with manifest.json + compare.json.
#
# Knative reads neither checkpoint, so its 15 cells from
# contention_v2_873_v5.5_coupled_trio_20260813 are valid here verbatim: same configs, seeds,
# workload and node_disk_v2 physics. They are symlinked rather than copied (~120MB each) and
# the manifest records where they came from.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DATALAB_USER="${DATALAB_USER:-nikola.lukic}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/nikolalukic167}"
DEST_HOST="${DEST_HOST:-cluster.datalab.tuwien.ac.at}"
REMOTE="${DATALAB_USER}@${DEST_HOST}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/nikola.lukic/gnn-herosim}"

SWEEP="${SWEEP:-simulation_data/normal_sim_sweeps/contention_v2_873_v5.7_siv1_coupled_trio_20260813}"
KN_SWEEP="${KN_SWEEP:-simulation_data/normal_sim_sweeps/contention_v2_873_v5.5_coupled_trio_20260813}"
KN_CFG="simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs"
SKEW_CFG="simulation_data/normal_sim_sweeps/atomic21_skew_configs/05_sparse_40_40_p25_degree_skew.json"
GNN_MODEL="models/near-rtt-v2-contention-v2-873-v5.7-siv1-dim14-ce-only.pt"
MLP_MODEL="models/tabular/batch_edge_mlp_contention_v2_873_v5.7_siv1_dim22_batchcache.pt"
WORKLOAD="data/nofs-ids/traces/workload-125-225.json"
SEEDS="${SEEDS:-42,43,44,45,46}"
CONFIGS="sparse_p25,sparse_p35,sparse_p25_skew"

mkdir -p "${SWEEP}/results" "${SWEEP}/configs" logs

echo "=== 1) rsync ML results back ==="
rsync -azP -e "ssh -i ${SSH_KEY} -o BatchMode=yes" \
  "${REMOTE}:${REMOTE_ROOT}/${SWEEP}/results/" "${SWEEP}/results/" \
  | tail -5

echo "=== 2) link the Knative arm ==="
for cfg in sparse_p25 sparse_p35 sparse_p25_skew; do
  for seed in ${SEEDS//,/ }; do
    src="${ROOT}/${KN_SWEEP}/results/${cfg}_s${seed}_knative.json"
    [[ -f "$src" ]] || { echo "ERROR: missing Knative cell $src" >&2; exit 1; }
    ln -sfn "$src" "${SWEEP}/results/${cfg}_s${seed}_knative.json"
  done
done
echo "linked $(ls -1 ${SWEEP}/results/*_knative.json | wc -l) Knative cells"

echo "=== 3) copy configs so the sweep is self-describing ==="
cp -f "${KN_CFG}/05_sparse_40_40_p25.json" "${SWEEP}/configs/"
cp -f "${KN_CFG}/00_balanced_30_30_p35.json" "${SWEEP}/configs/"
cp -f "$SKEW_CFG" "${SWEEP}/configs/"

echo "=== 4) completeness check (expect 45 result JSONs) ==="
n=$(ls -1 "${SWEEP}/results"/*.json 2>/dev/null | grep -v decode_stats | wc -l)
echo "  result JSONs: ${n}/45"
(( n == 45 )) || { echo "ERROR: incomplete sweep (${n}/45)" >&2; exit 1; }

echo "=== 5) manifest ==="
pipenv run python3 scripts_cosim/important/write_sweep_manifest.py \
  --sweep-dir "$SWEEP" \
  --kind multi_seed_live_gate_coupled_trio_scale_invariant_v1 \
  --note "Coupled trio with scale_invariant_v1 queue features (uncapped dim7 p90, log1p dim13) after the queue-feature OOD post-mortem. Same labels/splits/hparams as 873/v5.5; only dim7/dim13 scaling and the resulting weights differ. Knative arm symlinked from ${KN_SWEEP} (policy-independent: identical configs, seeds, workload and node_disk_v2 physics)." \
  --physics node_disk_v2 \
  --workload "$WORKLOAD" \
  --seeds "$SEEDS" \
  --configs "$CONFIGS" \
  --gnn-model "$GNN_MODEL" \
  --mlp-model "$MLP_MODEL" \
  --extra-json "$(cat <<JSON
{
  "queue_feature_contract": "scale_invariant_v1",
  "cache": "simulation_data/graphs_cache_contention_v2_873_v5.7_siv1_dim14",
  "cache_version": "5.7",
  "decode_mode": "argmax",
  "inference_feature_layout": "dim22",
  "knative_arm_source": "${KN_SWEEP}/results (symlinked, policy-independent)",
  "baseline_for_comparison": "${KN_SWEEP} (legacy_v0 features, same 873 labels)",
  "seed42_smoke": "simulation_data/normal_sim_sweeps/contention_v2_873_v5.7_siv1_coupled_trio_smoke_20260813",
  "offline_retrain": {
    "gnn_val_acc": 0.626, "gnn_test_acc": 0.557,
    "mlp_val_edge_acc": 0.916, "mlp_test_edge_acc": 0.873
  }
}
JSON
)" \
  ${FORCE_MANIFEST:+--force}

echo "=== 6) compare ==="
pipenv run python3 scripts_cosim/important/compare_sealed_live_holdout.py \
  --sweep-dir "$SWEEP" \
  --report "${SWEEP}/compare.json"

echo "=== sealed: ${SWEEP} ==="
