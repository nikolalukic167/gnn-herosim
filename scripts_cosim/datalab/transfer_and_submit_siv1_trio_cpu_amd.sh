#!/usr/bin/env bash
# Rsync HEAD live-sim code + the scale_invariant_v1 ckpts to datalab and submit the
# 5-seed coupled-trio gate (MLP+GNN x 3 configs x seeds 42-46 = 30 CPU-amd cells).
#
# Why only MLP+GNN: Knative does not read either checkpoint, so its 15 trio cells from
# contention_v2_873_v5.5_coupled_trio_20260813 are reusable verbatim (same configs, seeds,
# workload and node_disk_v2 physics). They are linked in locally after transfer-back.
#
# Promotes the seed-42 smoke, where MLP siv1 beat Knative on all three cells
# (total_rtt 0.747x, p99 0.40-0.60x) while GNN siv1 regressed below GNN v5.5.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DATALAB_USER="${DATALAB_USER:-nikola.lukic}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/nikolalukic167}"
DEST_HOST="${DEST_HOST:-cluster.datalab.tuwien.ac.at}"
REMOTE="${DATALAB_USER}@${DEST_HOST}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/nikola.lukic/gnn-herosim}"
SSH=(ssh -i "$SSH_KEY" -o BatchMode=yes -o StrictHostKeyChecking=accept-new)
RSYNC=(rsync -azP -e "ssh -i ${SSH_KEY} -o BatchMode=yes")

SWEEP="${SWEEP:-simulation_data/normal_sim_sweeps/contention_v2_873_v5.7_siv1_coupled_trio_20260813}"
KN_CFG="simulation_data/normal_sim_sweeps/knative_network_20260606_192413/configs"
SKEW_CFG="simulation_data/normal_sim_sweeps/atomic21_skew_configs/05_sparse_40_40_p25_degree_skew.json"

GNN_MODEL="models/near-rtt-v2-contention-v2-873-v5.7-siv1-dim14-ce-only.pt"
GNN_SIDECAR="models/near-rtt-v2-contention-v2-873-v5.7-siv1-dim14-ce-only.contract.json"
MLP_MODEL="models/tabular/batch_edge_mlp_contention_v2_873_v5.7_siv1_dim22_batchcache.pt"
WORKLOAD="data/nofs-ids/traces/workload-125-225.json"
QUEUE_FEATURE_CONTRACT="${QUEUE_FEATURE_CONTRACT:-scale_invariant_v1}"

[[ -f "$SSH_KEY" ]] || { echo "ERROR: SSH key missing: $SSH_KEY" >&2; exit 1; }
for f in "$GNN_MODEL" "$GNN_SIDECAR" "$MLP_MODEL" "${MLP_MODEL}.meta.json" "$WORKLOAD" "$SKEW_CFG"; do
  [[ -f "$f" ]] || { echo "ERROR: missing $f" >&2; exit 1; }
done

# Both checkpoints must actually declare the contract we are about to serve, or the sweep is
# a train/serve mismatch that no downstream metric would reveal.
python3 - "$GNN_SIDECAR" "$MLP_MODEL" "$QUEUE_FEATURE_CONTRACT" <<'PY'
import json, sys
sidecar, mlp, want = sys.argv[1], sys.argv[2], sys.argv[3]
got = json.loads(open(sidecar).read()).get("queue_feature_contract")
if got != want:
    raise SystemExit(f"ERROR: {sidecar} declares {got!r}, expected {want!r}")
try:
    import torch
except ModuleNotFoundError:
    print(f"  [warn] torch unavailable locally; skipped MLP ckpt contract check")
else:
    ck = torch.load(mlp, map_location="cpu", weights_only=False)
    got = ck.get("queue_feature_contract")
    if got != want:
        raise SystemExit(f"ERROR: {mlp} declares {got!r}, expected {want!r}")
print(f"  contract verified on both checkpoints: {want}")
PY

JOBS_DIR="scripts_cosim/datalab/jobs"
JOBS_TSV="${JOBS_DIR}/siv1_coupled_trio_mlp_gnn.tsv"
mkdir -p "$JOBS_DIR" logs

: >"$JOBS_TSV"
configs=(
  "sparse_p25|${KN_CFG}/05_sparse_40_40_p25.json"
  "sparse_p35|${KN_CFG}/00_balanced_30_30_p35.json"
  "sparse_p25_skew|${SKEW_CFG}"
)
for pol in mlp gnn; do
  for entry in "${configs[@]}"; do
    name="${entry%%|*}"
    path="${entry#*|}"
    for seed in 42 43 44 45 46; do
      printf '%s\t%s\t%s\t%s\n' "$pol" "$name" "$path" "$seed" >>"$JOBS_TSV"
    done
  done
done
sed -i 's/\r$//' "$JOBS_TSV"
N_JOBS=$(wc -l <"$JOBS_TSV")
echo "wrote $JOBS_TSV (${N_JOBS} cells)"

echo "=== 1) remote mkdir ==="
"${SSH[@]}" "$REMOTE" "mkdir -p \
  ${REMOTE_ROOT}/src \
  ${REMOTE_ROOT}/models/tabular \
  ${REMOTE_ROOT}/logs \
  ${REMOTE_ROOT}/scripts_cosim/datalab/jobs \
  ${REMOTE_ROOT}/${SWEEP}/results \
  ${REMOTE_ROOT}/${SWEEP}/configs \
  ${REMOTE_ROOT}/${KN_CFG} \
  ${REMOTE_ROOT}/simulation_data/normal_sim_sweeps/atomic21_skew_configs \
  ${REMOTE_ROOT}/data/nofs-ids/traces"

echo "=== 2) rsync live-sim code (queue_features contracts + provenance fix) ==="
"${RSYNC[@]}" --delete --exclude '__pycache__' --exclude '*.pyc' \
  src/ "${REMOTE}:${REMOTE_ROOT}/src/"
"${RSYNC[@]}" \
  scripts_cosim/run_simulation.py \
  scripts_cosim/sweep_metrics.py \
  "${REMOTE}:${REMOTE_ROOT}/scripts_cosim/"
"${RSYNC[@]}" \
  scripts_cosim/datalab/run_sealed_holdout_one.sh \
  scripts_cosim/datalab/live_cpu_amd.sbatch \
  scripts_cosim/datalab/submit_live_cpu_amd.sh \
  "${REMOTE}:${REMOTE_ROOT}/scripts_cosim/datalab/"
"${RSYNC[@]}" "$JOBS_TSV" "${REMOTE}:${REMOTE_ROOT}/${JOBS_DIR}/"

echo "=== 3) rsync siv1 models (sidecar included) + workload + configs ==="
"${RSYNC[@]}" "$GNN_MODEL" "$GNN_SIDECAR" "${REMOTE}:${REMOTE_ROOT}/models/"
"${RSYNC[@]}" "$MLP_MODEL" "${MLP_MODEL}.meta.json" \
  "${REMOTE}:${REMOTE_ROOT}/models/tabular/"
"${RSYNC[@]}" "$WORKLOAD" "${REMOTE}:${REMOTE_ROOT}/data/nofs-ids/traces/"
"${RSYNC[@]}" \
  "${KN_CFG}/05_sparse_40_40_p25.json" \
  "${KN_CFG}/00_balanced_30_30_p35.json" \
  "${REMOTE}:${REMOTE_ROOT}/${KN_CFG}/"
"${RSYNC[@]}" "$SKEW_CFG" \
  "${REMOTE}:${REMOTE_ROOT}/simulation_data/normal_sim_sweeps/atomic21_skew_configs/"

echo "=== 4) CRLF heal + submit ==="
"${SSH[@]}" "$REMOTE" bash -s <<EOF
set -euo pipefail
cd ${REMOTE_ROOT}
chmod +x scripts_cosim/datalab/run_sealed_holdout_one.sh \
  scripts_cosim/datalab/submit_live_cpu_amd.sh \
  scripts_cosim/datalab/live_cpu_amd.sbatch
sed -i 's/\r\$//' scripts_cosim/datalab/*.sh scripts_cosim/datalab/*.sbatch \
  scripts_cosim/datalab/jobs/*.tsv

[[ -f ${GNN_SIDECAR} ]] || { echo "ERROR: sidecar did not transfer" >&2; exit 1; }

export HEROSIM_WARMTH_PHYSICS=node_disk_v2
export HEROSIM_REQUIRE_EXPLICIT_PHYSICS=1
export QUEUE_FEATURE_CONTRACT=${QUEUE_FEATURE_CONTRACT}
export GNN_MODEL=${GNN_MODEL}
export MLP_MODEL=${MLP_MODEL}
export WORKLOAD=${WORKLOAD}
export TIMEOUT=18000

SWEEP_DIR=${SWEEP} \
JOBS_TSV=${JOBS_TSV} \
JOB_NAME=siv1-trio-cpu \
ARRAY=0-$((N_JOBS - 1)) \
  bash scripts_cosim/datalab/submit_live_cpu_amd.sh
EOF

echo "=== submitted ==="
echo "Monitor: ssh -i ${SSH_KEY} ${REMOTE} 'squeue -u ${DATALAB_USER} -n siv1-trio-cpu'"
