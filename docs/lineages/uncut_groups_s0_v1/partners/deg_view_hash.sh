#!/bin/bash
#SBATCH --job-name=deg-view-hash
#SBATCH --partition=CPU-amd
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --exclude=os-cpu-slurm-21,os-cpu-slurm-15
#SBATCH --output=/share/nikola.lukic/uncut_groups/partners/reads/deg-view-hash-%j.out
set -euo pipefail
P=/share/nikola.lukic/uncut_groups/partners; cd $P
echo "ls -lR md5 $(cd deg_view && ls -lR --time-style=+ . | md5sum | cut -c1-32)"
find -L deg_view -type f | sort > reads/deg_view_files.txt
echo "files $(wc -l < reads/deg_view_files.txt)"
xargs md5sum < reads/deg_view_files.txt > reads/deg_view_md5sums.txt
echo "content md5 (find -L deg_view -type f | sort | xargs md5sum | md5sum) $(md5sum < reads/deg_view_md5sums.txt | cut -c1-32)"
for L in k1 k2 k4 k8; do for e in cells_reactive cells_learned cells_bar inputs_group; do echo "$L/$e -> $(readlink -f deg_view/$L/$e)"; done; done
