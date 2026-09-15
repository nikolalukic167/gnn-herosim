#!/bin/bash
# queue_range_v1 -- Q0..Q4 over the gate's summaries. Reactive comes from serving_stability_v1's
# own results dir and is NOT re-run: same trace, same cells, same physics, and re-running it
# would burn compute to reproduce a number already in the record.
#
#SBATCH --job-name=qr-read
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/qr-read-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/qr-read-%j.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1

echo "[CODE] $(git describe --dirty --always) on $(git rev-parse --abbrev-ref HEAD)"
GATE=simulation_data/peer_affinity_live_gate
mkdir -p simulation_data/queue_range_v1
python3 scripts_cosim/queue_range_v1_read.py \
  --results "$GATE/results/qr_v1" \
  --reactive-results "$GATE/results/ss_s1s2" \
  --out simulation_data/queue_range_v1/gate.json
echo "=== queue_range_v1 read done ==="
