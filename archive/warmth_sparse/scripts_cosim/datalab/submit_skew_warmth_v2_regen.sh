#!/usr/bin/env bash
# Submit 4-shard SLURM array for skew_warmth_v2 co-sim regen (288 ds).
# Auto-picks CPU partition with immediate schedule (48 CPUs/shard so 4 arrays fit).
set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

CPUS="${CPUS:-48}"
TIME="${TIME:-24:00:00}"
CANCEL_STALE="${CANCEL_STALE:-1}"

for f in \
  scripts_cosim/datalab/skew_warmth_v2_regen.sbatch \
  scripts_cosim/datalab/run_warmth_regen_shard.sh \
  scripts_cosim/generate_gnn_datasets_fast.py; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing $f" >&2
    exit 1
  fi
done

sed -i 's/\r$//' \
  scripts_cosim/datalab/skew_warmth_v2_regen.sbatch \
  scripts_cosim/datalab/run_warmth_regen_shard.sh

if [[ "${CANCEL_STALE}" == "1" ]]; then
  echo "=== Cancel stale skew-warmth jobs ==="
  for j in $(squeue -u "$USER" -h -o "%i %j" | awk '/skew-warmth/ {print $1}'); do
    echo "scancel ${j}"
    scancel "${j}" || true
  done
  sleep 2
fi

echo "=== best.json before submit ==="
find "simulation_data/gnn_datasets_4tasks_skew_warmth_v2" -name best.json 2>/dev/null | wc -l

echo "=== Remove stale shared gnn_templates (parallel race fix) ==="
rm -rf data/nofs-ids/traces/gnn_templates data/nofs-ids/traces/gnn_templates_* 2>/dev/null || true

pick_partition() {
  local part="$1"
  local cpus="$2"
  local tag="$3"
  echo "--- test-only ${part} (${cpus} CPUs × 4 array tasks) ---"
  if out=$(sbatch --test-only \
    --partition="$part" \
    --cpus-per-task="$cpus" \
    --time="${TIME}" \
    --array=0-3 \
    scripts_cosim/datalab/skew_warmth_v2_regen.sbatch 2>&1); then
    echo "$out"
    if echo "$out" | grep -qiE 'PENDING|ReqNodeNotAvail|QOSMax|AssocMaxJobsLimit|Resources|NOT be executed'; then
      return 1
    fi
    SELECTED_PARTITION="$part"
    SELECTED_CPUS="$cpus"
    SELECTED_TAG="$tag"
    return 0
  fi
  echo "$out" >&2
  return 1
}

SELECTED_PARTITION=""
SELECTED_CPUS=""
SELECTED_TAG=""

echo "User queue before submit:"
squeue -u "$(whoami)" -o "%.10i %.12P %.16j %.2t %R" || true
echo
echo "Partition landscape (GPU-a100s):"
sinfo -p GPU-a100s -o "%P %a %C" 2>/dev/null || true

# 48 CPUs/shard → 4×48=192 CPUs; fits alongside other jobs (range264 pattern).
if ! pick_partition "GPU-a100s" "${CPUS}" "a100s48"; then
  echo "GPU-a100s@${CPUS} not immediately schedulable; trying GPU-a100@${CPUS}..."
  if ! pick_partition "GPU-a100" "${CPUS}" "a10048"; then
    echo "Trying GPU-a100s @ 64 CPUs..."
    if ! pick_partition "GPU-a100s" "64" "a100s64"; then
      echo "Trying GPU-a100 @ 64 CPUs..."
      if ! pick_partition "GPU-a100" "64" "a10064"; then
        echo "ERROR: no partition can schedule 4-array skew_warmth_v2 now" >&2
        exit 1
      fi
    fi
  fi
fi

echo ""
echo "Selected: ${SELECTED_PARTITION} (${SELECTED_CPUS} CPUs/task)"

job_id=$(sbatch \
  --job-name="skew-warmth-${SELECTED_TAG}" \
  --output="/home/nikola.lukic/gnn-herosim/logs/skew-warmth-${SELECTED_TAG}-%A_%a.out" \
  --error="/home/nikola.lukic/gnn-herosim/logs/skew-warmth-${SELECTED_TAG}-%A_%a.err" \
  --partition="${SELECTED_PARTITION}" \
  --cpus-per-task="${SELECTED_CPUS}" \
  --time="${TIME}" \
  --array=0-3 \
  scripts_cosim/datalab/skew_warmth_v2_regen.sbatch | awk '{print $NF}')

echo ""
echo "Submitted skew_warmth_v2 regen array: ${job_id} (288 ds, ${SELECTED_CPUS} CPUs/task, 4 shards)"
echo "Monitor: squeue -u \$USER · tail -f logs/skew-warmth-${SELECTED_TAG}-${job_id}_*.out"
echo "Progress: find simulation_data/gnn_datasets_4tasks_skew_warmth_v2 -name best.json | wc -l"
echo "JOB_ID=${job_id}"
echo "PARTITION=${SELECTED_PARTITION}"
echo "CPUS=${SELECTED_CPUS}"

echo ""
echo "=== Queue after submit ==="
squeue -u "$(whoami)" -o "%.18i %.9P %.14j %.2t %.10M %C %R" | grep -E "skew-warmth|JOBID" || true
