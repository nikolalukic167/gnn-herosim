#!/bin/bash
# Audit 2026-09-13 -- is the live candidate set inside the training corpus's support?
#
# A 3k local smoke says no: the live decoder is offered `xavierGpu` candidates, a platform
# type present in ZERO of the 516 training datasets, carrying a demand 8x anything the cache
# contains (1.739 vs 0.213). Because node_caps = alpha x max single candidate demand on that
# node, one such candidate raises a node's cap ~8x and the capacity mask stops binding there
# for the whole batch -- the decoder's only concentration control before the platform cap.
#
# This array measures it on the REAL production decode traces (serving_gap_v1/traces_h2:
# 3 corpora x 2 arms x 16 seeds, every sampled batch's graph and decoded plan), and reports
# the statistic that decides whether it is an ARM asymmetry or a shared handicap: the share
# of decoded placements landing on a platform type outside the checkpoint's own corpus.
#
#SBATCH --job-name=pa-candsup
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/pa-candsup-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/pa-candsup-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-2

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim

if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=""
python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"
echo "[CODE] $(git describe --dirty --always) on $(git rev-parse --abbrev-ref HEAD)"

case ${SLURM_ARRAY_TASK_ID} in
  0) CORPUS=x200p2; CACHE=simulation_data/graphs_cache_peer_affinity_v1_t1b ;;
  1) CORPUS=x800p2; CACHE=simulation_data/graphs_cache_peer_affinity_v1_x800_p2 ;;
  2) CORPUS=x800p3; CACHE=simulation_data/graphs_cache_peer_affinity_v1_x800_p3 ;;
  *) echo "FAIL LOUD: unknown task ${SLURM_ARRAY_TASK_ID}" >&2; exit 1 ;;
esac
TRACES=simulation_data/serving_gap_v1/traces_h2/$CORPUS  # h1 is SLIM (no graph); h2 carries the served graph
OUT=simulation_data/peer_affinity_candidate_support_${CORPUS}.json
[[ -d "$CACHE" ]] || { echo "FAIL LOUD: cache missing: $CACHE" >&2; exit 1; }
[[ -d "$TRACES" ]] || { echo "FAIL LOUD: traces missing: $TRACES" >&2; exit 1; }
echo "[TASK ${SLURM_ARRAY_TASK_ID}] corpus=$CORPUS -> $OUT"

${HEROSIM_PY:-pipenv run python3} scripts_cosim/candidate_support_probe.py \
  --cache-dir "$CACHE" --traces "$TRACES" --corpus "$CORPUS" --out "$OUT"
echo "[DONE] $(date)"
