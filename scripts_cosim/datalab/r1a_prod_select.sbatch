#!/bin/bash
# r1_attribution_v1: per-arm configuration selection on validation topologies only, and staging of the chosen checkpoints for the gate.
#   sbatch --dependency=afterok:<training array> --export=ALL,WT=<pinned worktree>,INPUTS=<gate inputs dir>,OUT=<selection.json> r1a_prod_select.sbatch
# Another lineage (scale_160_v1) through the environment: SPLIT (the split artifact), PREFIX (run-name prefix, scale-160-v1), ARMS ("gnn_eng gnn_eng_physmp").
#SBATCH --job-name=r1a-select
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/r1a-select-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail
WT=${WT:?}; INPUTS=${INPUTS:?}; OUT=${OUT:?}
cd "$WT"
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export PYTHONPATH="$WT"
[[ ! -e "$OUT" ]] || { echo "FAIL LOUD: $OUT exists"; exit 1; }
python3 scripts_cosim/r1a_select_checkpoints.py --models-dir /home/nikola.lukic/gnn-herosim/models --inputs-dir "$INPUTS" \
  --split "${SPLIT:-experiments/r1_attribution_v1_split.json}" --out "$OUT" ${PREFIX:+--prefix "$PREFIX"} ${ARMS:+--arms $ARMS}
