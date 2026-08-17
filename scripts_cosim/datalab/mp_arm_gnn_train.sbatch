#!/bin/bash
# Submit with the arm selected explicitly, e.g.
#   sbatch --export=ALL,ARM=residual            scripts_cosim/datalab/mp_arm_gnn_train.sbatch
#   sbatch --export=ALL,ARM=residual_node_edges scripts_cosim/datalab/mp_arm_gnn_train.sbatch
# The two arms are independent and are meant to run concurrently.
#SBATCH --job-name=mp-arm
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/mp-arm-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/mp-arm-%j.err
# Partition list matches full_corpus_siv1_gnn_train.sbatch. Do NOT add GPU-rtx6000: the
# gnn env's torch 2.5.1+cu121 has no kernel image for those cards and both arms died at
# init with "CUDA error: no kernel image is available for execution on the device".
#SBATCH --partition=GPU-a40,GPU-a100s,GPU-l40s,GPU-a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00

set -euo pipefail
ulimit -n 65536 2>/dev/null || true

cd /home/nikola.lukic/gnn-herosim
ARM="${ARM:?set ARM via --export=ALL,ARM=...}" bash scripts_cosim/datalab/run_mp_arm_gnn_train.sh
