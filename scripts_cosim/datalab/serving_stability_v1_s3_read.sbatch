#!/bin/bash
# serving_stability_v1 -- S3 read: the live gate that closes the lineage.
#
#   sbatch scripts_cosim/datalab/serving_stability_v1_s3_read.sbatch
#
#SBATCH --job-name=ss-s3read
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/ss-s3read-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/ss-s3read-%j.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$PWD" PYTHONUNBUFFERED=1

echo "[CODE] $(git describe --dirty --always) on $(git rev-parse --abbrev-ref HEAD)"
echo "[BARS] $(grep -E '^(S3_MIN_SEEDS|S3_MIN_CELLS|S3_ALPHA|S3B_MIN_BIND_PCT|S3A_MAX_HANGS|S3_K) ' scripts_cosim/serving_stability_v1_s3_read.py | tr '\n' ' ')"

R=simulation_data/peer_affinity_live_gate/results
echo "[inputs] guarded $(ls $R/ss_s3/*.summary.json 2>/dev/null | wc -l) / unguarded $(ls $R/ss_s1s2/*.summary.json 2>/dev/null | wc -l)"

${HEROSIM_PY} scripts_cosim/serving_stability_v1_s3_read.py \
  --guarded "$R/ss_s3" --unguarded "$R/ss_s1s2" \
  --out simulation_data/serving_stability_v1/s3.json

echo "=== S3 read done ==="
