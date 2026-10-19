#!/bin/bash
#SBATCH -J s5-s36-read
#SBATCH -p CPU-amd
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH -o /share/nikola.lukic/uncut_groups/size64_s36/read-%j.out
eval "$(micromamba shell hook --shell bash)" && micromamba activate gnn
python3 /share/nikola.lukic/uncut_groups/size64_s36/size64_s36_read.py /share/nikola.lukic/uncut_groups/size64_s36/reads
md5sum /share/nikola.lukic/uncut_groups/size64_s36/reads/size64_s36_read.json /share/nikola.lukic/uncut_groups/size64_s36/reads/size64_s36_read.txt
