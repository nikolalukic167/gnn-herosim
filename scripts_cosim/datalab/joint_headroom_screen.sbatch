#!/bin/bash
# Read-only descriptive screen on production E train corpus (coordinator task 2026-10-09). Writes only to OUTDIR (/share).
#SBATCH --job-name=jointhr
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/jointhr-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
set -euo pipefail
eval "$(micromamba shell hook --shell bash)" && micromamba activate gnn
cd /home/nikola.lukic/gnn-herosim
OUTDIR=/share/nikola.lukic/workload_fix_v1/headroom_E; mkdir -p $OUTDIR
python3 scripts_cosim/joint_headroom_screen.py $OUTDIR/headroom.jsonl /share/nikola.lukic/workload_fix_v1/corpus_prod/gnn_datasets_wf1_train_E8894a08 --workers 16
