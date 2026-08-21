#!/bin/bash
# topology_transfer_v1 Phase 4 pre-registered gate (LINEAGES.md: topology_transfer_v1).
# 5 seeds (42-46) as array tasks 0-4; each trains pointwise/gnn_base/gnn_node on the
# topology_size split (train 20/28/40, held-out 60/80), tier_launch, 120 epochs.
#SBATCH --job-name=topo-xfer-v1-p4
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/topo-xfer-v1-p4-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/topo-xfer-v1-p4-%A_%a.err
# Same partition list as mp_arm_gnn_train.sbatch / full_corpus_siv1_gnn_train.sbatch.
# Do NOT add GPU-rtx6000: the gnn env's torch has no kernel image for those cards.
#SBATCH --partition=GPU-a40,GPU-a100s,GPU-l40s,GPU-a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-4

set -euo pipefail

cd /home/nikola.lukic/gnn-herosim
bash scripts_cosim/datalab/run_topo_transfer_v1_phase4_gate.sh
