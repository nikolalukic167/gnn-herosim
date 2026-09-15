#!/bin/bash
# offline_live_transfer_v1 -- R2 step 2: score the four registered surrogate candidates
# against live latency and apply the promotion rule.
#
#   sbatch scripts_cosim/datalab/offline_live_transfer_v1_r2_read.sbatch
#
#SBATCH --job-name=olt-r2read
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/olt-r2read-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/olt-r2read-%j.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim

if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1

python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"
echo "[CODE] $(git describe --dirty --always) on $(git rev-parse --abbrev-ref HEAD)"
echo "[BARS] $(grep -E '^(R2_POOLED_Z_MIN_ABS_RHO|R2_WITHIN_MIN_ABS_RHO|R2_WITHIN_MIN_FAMILIES|R2_ALPHA|R2_HOLM_N|R2_MAX_PROMOTED|R2_QUEUE_SCALE|R2_MIN_COMMON_DATASETS) ' scripts_cosim/offline_live_transfer_v1_surrogate_read.py | tr '\n' ' ')"

BASE=simulation_data/offline_live_transfer_v1
DEC="$BASE/r2_decoded"
SCALED="$BASE/r2_decoded_scaled"

n_dec=$(ls "$DEC"/*.json 2>/dev/null | wc -l)
[[ "$n_dec" -eq 48 ]] || { echo "FAIL LOUD: $n_dec decoded files in $DEC, expected 48" >&2; exit 1; }

# range_sensitivity needs the scaled pass. Without a COMPLETE one the candidate is
# NOT-COMPUTED rather than computed on a subset -- and Holm still divides by four.
n_sc=$(ls "$SCALED"/*.json 2>/dev/null | wc -l)
if [[ "$n_sc" -eq 48 ]]; then
  SCALED_ARG=(--scaled-dir "$SCALED")
  echo "[inputs] scaled pass complete ($n_sc files): range_sensitivity will be computed"
else
  SCALED_ARG=()
  echo "[inputs] scaled pass has $n_sc/48 files: range_sensitivity NOT-COMPUTED"
fi

${HEROSIM_PY} scripts_cosim/offline_live_transfer_v1_surrogate_read.py \
  --decoded-dir "$DEC" "${SCALED_ARG[@]}" \
  --gate-root simulation_data/peer_affinity_live_gate/results \
  --sim-root simulation_data \
  --out "$BASE/r2_surrogates.json"

echo "=== R2 read done: $BASE/r2_surrogates.json ==="
