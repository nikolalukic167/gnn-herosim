#!/bin/bash
# Submits the 12 live_link_features_v1 development trainings (4 arms x seeds 1-3), 3 per node on os-cpu-slurm-6/7/8/9 (round robin), after the corpus, split, labels and cache are merged and filed.
# Run on the login node from WT (sbatch only). Needs WT, EXPECT_HEAD, EXPECT_CPU, SPLIT_SHA, LABEL_SHA, CACHE_SHA (one shared corpus: the same three for every arm). Set DRY=1 to print the commands.
set -euo pipefail
# The 4 nodes with the most idle CPUs (each job takes 12 CPUs, three per node, so a node needs >= 36 idle), never 21 or 15; NODES="a b c d" overrides. Re-read at every launch: other sessions' jobs move.
if [[ -z "${NODES:-}" ]]; then
  NODES=$(sinfo -N -p CPU-amd -h -o "%N %C" | awk '{split($2,c,"/"); print c[2], $1}' | grep -vE ' os-cpu-slurm-(21|15)$' | sort -rn | awk '$1 >= 36 {print $2}' | head -4 | tr '\n' ' ')
fi
read -r -a NODES <<< "$NODES"
(( ${#NODES[@]} == 4 )) || { echo "FAIL LOUD: found ${#NODES[@]} nodes with >= 36 idle CPUs (${NODES[*]:-none}); need 4"; exit 1; }
echo "nodes: ${NODES[*]}"
i=0
for arm in gnn twin mlp mlp_wide; do for s in 1 2 3; do
  cmd=(sbatch --nodelist="${NODES[$((i % 4))]}" --job-name="link-$arm-s$s" --export="ALL,CFGN=uncut_link_$arm,S=$s,LABEL_SHA=${LABEL_SHA:?set LABEL_SHA},CACHE_SHA=${CACHE_SHA:?set CACHE_SHA},SPLIT_SHA=${SPLIT_SHA:?set SPLIT_SHA},EXPECT_CPU=${EXPECT_CPU:?set EXPECT_CPU}" "$(dirname "$0")/uncut_link_train.sbatch")
  [[ "${DRY:-0}" == 1 ]] && echo "${cmd[@]}" || "${cmd[@]}"
  i=$((i+1))
done; done
