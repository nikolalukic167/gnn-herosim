#!/bin/bash
# env_lever_v1 -- mint the levered workloads and cells (one task, CPU node, ~1 min).
# docs/lineages/{burst_groups_v1,payload_scale_v1,backbone_sparsity_v1}.md
#
#   sbatch scripts_cosim/datalab/env_lever_v1_mint.sbatch
#
#SBATCH --job-name=el1-mint
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/el1-mint-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/el1-mint-%j.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PWD"
echo "[CODE] $(git describe --dirty --always) on $(git rev-parse --abbrev-ref HEAD)"
# cell levers are minted for every candidate topology: the lever's own screen decides which
# four become the study
python3 scripts_cosim/env_lever_v1_mint.py --gate simulation_data/peer_affinity_live_gate \
  --topologies 9001,9002,9003,9005,9101,9102,9103,9104,9105,9106,9107,9108
python3 scripts_cosim/env_lever_v1_mint.py --gate simulation_data/peer_affinity_live_gate \
  --topologies 9001,9002,9003,9005,9101,9102,9103,9104,9105,9106,9107,9108 --check
echo "=== mint done ==="
