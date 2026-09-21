#!/bin/bash
#SBATCH --job-name=roll-phaseA
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/roll-phaseA-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/roll-phaseA-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=05:00:00
#SBATCH --array=0-23
# rollout_imitation_v1 Phase A: label the multi-candidate decisions on each of the 24 training
# topologies (seeds 9201-9224), forced-rollout at N in {20,50,100}. The pooled read
# (rollout_imitation_v1_phaseA_read.py) fires R0 (rank stability). See docs/lineages/rollout_imitation_v1.md.
set -euo pipefail
cd /home/nikola.lukic/gnn-herosim
eval "$(micromamba shell hook --shell bash)" && micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH=.

SEED=$(( 9201 + SLURM_ARRAY_TASK_ID ))
G=simulation_data/peer_affinity_live_gate
OUT=results/rollout_imitation_v1/labels/s${SEED}.json
mkdir -p "$(dirname "$OUT")"
if [[ -e "$OUT" ]]; then echo "[s$SEED] exists, skip"; exit 0; fi

echo "[s$SEED] labelling training topology cc40s${SEED} (unburst f700)"
python3 scripts_cosim/rollout_imitation_v1_label.py --seed "$SEED" \
  --config  "$G/configs/cc40s${SEED}.json" \
  --workload "$G/workloads/drainable_f700_n50000.json" \
  --out "$OUT" --max-decisions 40
echo "[s$SEED] done -> $OUT"
