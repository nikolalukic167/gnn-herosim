#!/usr/bin/env bash
# literature_reeval_v1 -- launch the registered L2D grid locally (CPU, 2 threads per run).
# Usage: scripts_reeval/l2d/run_study.sh <n_j> <n_m> <n_seeds> [parallel] [max_updates]
set -euo pipefail
cd "$(dirname "$0")/../.."
N_J=$1; N_M=$2; N_SEEDS=$3; PAR=${4:-14}; MAXU=${5:-10000}
OUT=simulation_data/literature_reeval_v1/l2d/runs
LOGS=simulation_data/literature_reeval_v1/l2d/logs; mkdir -p "$LOGS" "$OUT"
export PIPENV_IGNORE_VIRTUALENVS=1 VIRTUAL_ENV= PYTHONPATH="$PWD" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
jobs=()
for s in $(seq 0 $((N_SEEDS-1))); do for arm in gnn mpoff mlp_t1 mlp_t1x; do jobs+=("$arm $s"); done; done
printf '%s\n' "${jobs[@]}" | xargs -P "$PAR" -L 1 bash -c '
  arm=$0; s=$1; name=l2d_'"$N_J"'x'"$N_M"'_${arm}_s${s}
  if [ -f '"$OUT"'/$name/final.pth ]; then echo "$name done, skip"; exit 0; fi
  pipenv run python3 scripts_reeval/l2d/train_arm.py --arm $arm --n_j '"$N_J"' --n_m '"$N_M"' --seed $s \
      --max_updates '"$MAXU"' --out '"$OUT"' --threads 2 > '"$LOGS"'/$name.log 2>&1 && echo "$name ok" || echo "$name FAILED rc=$?"'
