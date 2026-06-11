#!/usr/bin/env bash
# Push warmth co-sim code + local pilot progress to datalab for second-half regen.
set -euo pipefail

ROOT="/root/projects/my-herosim"
cd "$ROOT"

SSH_KEY="${SSH_KEY:-$HOME/.ssh/nikolalukic167}"
REMOTE="${REMOTE:-nikola.lukic@cluster.datalab.tuwien.ac.at}"
REPO="${REPO:-/home/nikola.lukic/gnn-herosim}"
RSYNC_SSH="ssh -i ${SSH_KEY}"

OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-gnn_datasets_4tasks_1060_warmth_v2}"
LOCAL_OUT="simulation_data/${OUTPUT_SUBDIR}"

for f in \
  simulation_data/space_with_network.json \
  simulation_data/sample_simple.json \
  simulation_data/lhs_samples_simple.npy \
  simulation_data/lhs_samples_simple_mapping.pkl \
  data/nofs-ids/traces/workload-10.json; do
  if [[ ! -e "$f" ]]; then
    echo "ERROR: missing $f" >&2
    exit 1
  fi
done

ssh -i "$SSH_KEY" "$REMOTE" "mkdir -p \
  ${REPO}/logs \
  ${REPO}/simulation_data \
  ${REPO}/${LOCAL_OUT} \
  ${REPO}/scripts_cosim/datalab \
  ${REPO}/src/placement \
  ${REPO}/src/policy"

# Warmth + co-sim code (not yet on remote main)
rsync -avP -e "$RSYNC_SSH" \
  scripts_cosim/generate_gnn_datasets_fast.py \
  scripts_cosim/datalab/warmth_pilot50_second_half.sbatch \
  scripts_cosim/datalab/run_warmth_pilot50_second_half.sh \
  "${REMOTE}:${REPO}/scripts_cosim/"

rsync -avP -e "$RSYNC_SSH" \
  scripts_cosim/datalab/warmth_pilot50_second_half.sbatch \
  scripts_cosim/datalab/run_warmth_pilot50_second_half.sh \
  "${REMOTE}:${REPO}/scripts_cosim/datalab/"

rsync -avP -e "$RSYNC_SSH" \
  src/executecosimulation.py \
  src/generate_infrastructure.py \
  "${REMOTE}:${REPO}/src/"

rsync -avP -e "$RSYNC_SSH" src/placement/ "${REMOTE}:${REPO}/src/placement/"
rsync -avP -e "$RSYNC_SSH" src/policy/ "${REMOTE}:${REPO}/src/policy/"

rsync -avP -e "$RSYNC_SSH" \
  simulation_data/space_with_network.json \
  simulation_data/sample_simple.json \
  simulation_data/lhs_samples_simple.npy \
  simulation_data/lhs_samples_simple_mapping.pkl \
  "${REMOTE}:${REPO}/simulation_data/"

# Sim inputs (templates dir needed for workload generation)
rsync -avP -e "$RSYNC_SSH" \
  data/nofs-ids/traces/workload-10.json \
  "${REMOTE}:${REPO}/data/nofs-ids/traces/"
if [[ -d data/nofs-ids/traces/gnn_templates ]]; then
  rsync -avP -e "$RSYNC_SSH" \
    data/nofs-ids/traces/gnn_templates/ \
    "${REMOTE}:${REPO}/data/nofs-ids/traces/gnn_templates/"
fi

# Completed local datasets so datalab --resume skips overlap (ds_00025+)
if [[ -d "${LOCAL_OUT}" ]]; then
  rsync -avP -e "$RSYNC_SSH" \
    "${LOCAL_OUT}/" \
    "${REMOTE}:${REPO}/${LOCAL_OUT}/"
fi

if [[ -f "logs/progress_${OUTPUT_SUBDIR}.txt" ]]; then
  rsync -avP -e "$RSYNC_SSH" \
    "logs/progress_${OUTPUT_SUBDIR}.txt" \
    "${REMOTE}:${REPO}/logs/"
fi

echo "[+] Transfer complete."
echo "    Remote out: ${REPO}/${LOCAL_OUT}"
echo "    Submit:"
echo "      ssh -i ${SSH_KEY} ${REMOTE} 'cd ${REPO} && sed -i \"s/\\r\$//\" scripts_cosim/datalab/run_warmth_pilot50_second_half.sh scripts_cosim/datalab/warmth_pilot50_second_half.sbatch && sbatch scripts_cosim/datalab/warmth_pilot50_second_half.sbatch'"
