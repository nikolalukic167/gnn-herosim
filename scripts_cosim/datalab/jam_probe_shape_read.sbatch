#!/bin/bash
# JAM PROBE -- scoping shape read. No bars; orders work, closes nothing.
#
#   sbatch scripts_cosim/datalab/jam_probe_shape_read.sbatch
#
#SBATCH --job-name=jam-shape
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/jam-shape-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/jam-shape-%j.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=01:00:00

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

CAP=simulation_data/peer_affinity_live_gate/results/olt_r3_capture
REF="$CAP/reactive_knative_network.json"
for f in "$REF" "$CAP/best_gnn_s14.json" "$CAP/worst_gnn_s8.json"; do
  [[ -f "$f" ]] || { echo "FAIL LOUD: capture missing: $f" >&2; exit 1; }
done

${HEROSIM_PY} scripts_cosim/jam_probe_shape_read.py \
  --reference "$REF" \
  --arm "gnn_best_s14=$CAP/best_gnn_s14.json" \
  --arm "gnn_worst_s8=$CAP/worst_gnn_s8.json" \
  --out simulation_data/jam_probe/shape.json

echo "=== jam probe shape read done ==="
