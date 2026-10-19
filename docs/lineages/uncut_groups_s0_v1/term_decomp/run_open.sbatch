#!/bin/bash
#SBATCH --partition=CPU-amd
#SBATCH --exclude=os-cpu-slurm-13,os-cpu-slurm-14,os-cpu-slurm-15,os-cpu-slurm-16,os-cpu-slurm-17,os-cpu-slurm-18,os-cpu-slurm-19,os-cpu-slurm-21,os-cpu-slurm-22
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/share/nikola.lukic/uncut_groups/term_decomp/%x-%j.out
eval "$(micromamba shell hook --shell bash)" && micromamba activate gnn
python3 /share/nikola.lukic/uncut_groups/term_decomp/term_opening.py $SET /share/nikola.lukic/uncut_groups/term_decomp
