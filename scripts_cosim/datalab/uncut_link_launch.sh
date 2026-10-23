#!/bin/bash
# Submits live_link_features_v1 training jobs, one per "CFGN:seed" in $JOBS, on the login node from WT (sbatch only). Examples:
#   JOBS="uncut_link_gnn:1 uncut_link_twin:1 uncut_link_mlp:1 uncut_link_mlp_wide:1" ...                      the as-built 20-epoch seed-1 arms (CPU)
#   JOBS="uncut_link_gnn_e100:1 uncut_link_twin_e100:1 uncut_link_mlp_e100:1 uncut_link_mlp_wide_e100:1" ...   their 100-epoch A2/A3 runs
#   JOBS="uncut_linkconv_gnn:1 uncut_linkconv_twin:1 uncut_linkconv_gnn_rep:1" ...                            variant (b), its GPU twin and the repeat (GPU)
# Needs WT, EXPECT_HEAD, SPLIT_SHA, LABEL_SHA, CACHE_SHA (one shared corpus: the same three for every arm), EXPECT_CPU for CPU jobs. DRY=1 prints the commands.
# CPU jobs go to the nodes with the most idle CPUs (each takes 12; re-read from sinfo at every launch, never 21 or 15; EXCLUDE_NODES="os-cpu-slurm-6 ..." adds more, e.g. while
# other trainings fill them); GPU jobs (CFGN containing "linkconv") go to GPU-a40 / GPU-l40s. The 44-job queue cap is counted with the batch before anything is submitted.
set -euo pipefail
: "${JOBS:?set JOBS}" "${WT:?}" "${EXPECT_HEAD:?}" "${SPLIT_SHA:?}" "${LABEL_SHA:?}" "${CACHE_SHA:?}"
read -r -a SPECS <<< "$JOBS"
if [[ "${DRY:-0}" != 1 ]]; then
  QUEUED=$(squeue -u "$USER" -h | wc -l)
  (( QUEUED + ${#SPECS[@]} <= 44 )) || { echo "FAIL LOUD: queue $QUEUED + ${#SPECS[@]} new jobs exceeds the 44-job cap"; exit 1; }
fi
declare -A IDLE
while read -r node idle; do IDLE[$node]=$idle; done < <(sinfo -N -p CPU-amd -h -o "%N %C" | awk '{split($2,c,"/"); print $1, c[2]}')
for n in os-cpu-slurm-21 os-cpu-slurm-15 ${EXCLUDE_NODES:-}; do unset 'IDLE[$n]' || true; done
DIR=$(dirname "$0")
for spec in "${SPECS[@]}"; do
  cfgn=${spec%%:*}; seed=${spec##*:}
  extra=()
  if [[ "$cfgn" == *linkconv* ]]; then
    extra=(--partition=GPU-a40,GPU-l40s --gres=gpu:1 --cpus-per-task=8 --mem=64G "--time=${GPU_TIME:-24:00:00}")
  else
    best=""; bi=-1
    for n in "${!IDLE[@]}"; do (( ${IDLE[$n]} > bi )) && { best=$n; bi=${IDLE[$n]}; }; done
    (( bi >= 12 )) || { echo "FAIL LOUD: no CPU node with 12 idle CPUs left for $spec"; exit 1; }
    IDLE[$best]=$(( bi - 12 )); extra=(--nodelist="$best")
  fi
  cmd=(sbatch "${extra[@]}" --job-name="link-${cfgn#uncut_}-s$seed" --export="ALL,CFGN=$cfgn,S=$seed,LABEL_SHA=$LABEL_SHA,CACHE_SHA=$CACHE_SHA,SPLIT_SHA=$SPLIT_SHA,EXPECT_CPU=${EXPECT_CPU:-gpu-venue}" "$DIR/uncut_link_train.sbatch")
  [[ "${DRY:-0}" == 1 ]] && echo "${cmd[@]}" || "${cmd[@]}"
done
