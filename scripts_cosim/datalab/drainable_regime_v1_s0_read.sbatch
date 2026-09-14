#!/bin/bash
# drainable_regime_v1 -- S0, step 3: the registered read. Re-runs nothing; applies the bars.
#   sbatch --dependency=afterok:<screen job> scripts_cosim/datalab/drainable_regime_v1_s0_read.sbatch
#
#SBATCH --job-name=drain-s0-read
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/drain-s0-read-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/drain-s0-read-%j.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$PWD" PYTHONUNBUFFERED=1
echo "[CODE] $(git describe --dirty --always) on $(git rev-parse --abbrev-ref HEAD)"

S0="${DRAIN_S0_DIR:-simulation_data/drainable_regime_v1/s0}"
n=$(ls "$S0"/results/*.summary.json 2>/dev/null | wc -l)
WANT="${DRAIN_EXPECT_ARMS:-12}"
[[ "$n" -eq "$WANT" ]] || { echo "FAIL LOUD: expected $WANT arm summaries under $S0/results, found $n" >&2; exit 1; }

python3 scripts_cosim/drainable_regime_s0_read.py \
  --results-dir "$S0/results" --snapshots-dir "$S0/snapshots" \
  --factors ${DRAIN_FACTORS:-300 500 700 1000 2000 4000} --output "$S0/s0_read.json"
echo "=== S0 read done ==="
