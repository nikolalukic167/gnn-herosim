#!/bin/bash
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --exclude=os-cpu-slurm-21,os-cpu-slurm-15,os-cpu-slurm-13,os-cpu-slurm-14,os-cpu-slurm-16,os-cpu-slurm-10,os-cpu-slurm-17
#SBATCH --output=/share/nikola.lukic/uncut_groups/ils_cal/logs/%x-%j.out
# one exclusive node, one arm (ARM), SER = series dir; WT / EXPECT_HEAD = the pinned worktree and its commit
set -euo pipefail
eval "$(micromamba shell hook --shell bash)" && micromamba activate gnn
export HEROSIM_PY=python3
bash /share/nikola.lukic/uncut_groups/ils_cal/cot_ils_steps.sh "${ARM:?}" "${SER:?}"
