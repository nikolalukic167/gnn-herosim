#!/bin/bash
# Submits the 12 full-training jobs, 3 per node on 4 nodes (round robin), after the registration is merged. Run on the login node from WT (sbatch only).
# Needs WT, EXPECT_HEAD, EXPECT_CPU, SPLIT_SHA, LABEL_SHA, CACHE_SHA (one shared corpus: the same three for both arms). Set DRY=1 to print the commands.
set -euo pipefail
# the shared corpus (one for both arms): split_union.json, label_override_union.json, cache_v6/cache/metadata.json
SPLIT_SHA=${SPLIT_SHA:-18e33008abb4bd0443e1bd79e02bb47f38998ac77e05b920bfe213afff2cd232}
LABEL_SHA=${LABEL_SHA:-67559c5bbe756996bee2e6e6ed41730fae87abff54e77210d503ecb5a03027ce}
CACHE_SHA=${CACHE_SHA:-5d10bd957951df1dcb0ebed046f14d02022b66d4804569598386ef54aa0fd03f}
NODES=(os-cpu-slurm-6 os-cpu-slurm-7 os-cpu-slurm-8 os-cpu-slurm-9)   # none of the excluded nodes 10,13-17,21
i=0
for arm in gnn twin; do for s in 1 2 3 4 5 6; do
  cmd=(sbatch --nodelist="${NODES[$((i % 4))]}" --job-name="dagger-full-$arm-s$s" --export="ALL,CFGN=uncut_dagger_full_$arm,S=$s,LABEL_SHA=$LABEL_SHA,CACHE_SHA=$CACHE_SHA,SPLIT_SHA=$SPLIT_SHA,EXPECT_CPU=${EXPECT_CPU:?set EXPECT_CPU}" "$(dirname "$0")/uncut_dagger_full_train.sbatch")
  [[ "${DRY:-0}" == 1 ]] && echo "${cmd[@]}" || "${cmd[@]}"
  i=$((i+1))
done; done
