#!/bin/bash
# Submits the 12 live_link_features_v1 development trainings (4 arms x seeds 1-3), 3 per node on os-cpu-slurm-6/7/8/9 (round robin), after the corpus, split, labels and cache are merged and filed.
# Run on the login node from WT (sbatch only). Needs WT, EXPECT_HEAD, EXPECT_CPU, SPLIT_SHA, LABEL_SHA, CACHE_SHA (one shared corpus: the same three for every arm). Set DRY=1 to print the commands.
set -euo pipefail
NODES=(os-cpu-slurm-6 os-cpu-slurm-7 os-cpu-slurm-8 os-cpu-slurm-9)
i=0
for arm in gnn twin mlp mlp_wide; do for s in 1 2 3; do
  cmd=(sbatch --nodelist="${NODES[$((i % 4))]}" --job-name="link-$arm-s$s" --export="ALL,CFGN=uncut_link_$arm,S=$s,LABEL_SHA=${LABEL_SHA:?set LABEL_SHA},CACHE_SHA=${CACHE_SHA:?set CACHE_SHA},SPLIT_SHA=${SPLIT_SHA:?set SPLIT_SHA},EXPECT_CPU=${EXPECT_CPU:?set EXPECT_CPU}" "$(dirname "$0")/uncut_link_train.sbatch")
  [[ "${DRY:-0}" == 1 ]] && echo "${cmd[@]}" || "${cmd[@]}"
  i=$((i+1))
done; done
