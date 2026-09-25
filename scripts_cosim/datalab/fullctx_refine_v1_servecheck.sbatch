#!/bin/bash
# backlog_corpus_v1 P1 -- train/serve parity on this lineage's held-out cells, served with the in-flight
# capture fix (docs/lineages/backlog_corpus_v1.md, Amendment 1). Otherwise load_repr_v1's check verbatim: the live serving
# path on held-out co-sim datasets must equal the cache graph field by field, including backlog_s and
# service_s, decode the offline report's plan and reproduce its sweep row (peer_affinity_live_serve_check.py).
# fullctx_refine_v1: task 0 = fc1load seed 1. Runs after fullctx_refine_v1_eval.sbatch.
#   sbatch --array=0 fullctx_refine_v1_servecheck.sbatch
#SBATCH --job-name=fc1-serve
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/fc1-serve-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/fc1-serve-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=08:00:00

set -euo pipefail
WT=/home/nikola.lukic/gnn-herosim-wt/backlog_corpus
cd "$WT"
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$WT"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
echo "[CODE] $(git describe --dirty --always) at $WT"
T=${SLURM_ARRAY_TASK_ID:?FAIL LOUD: SLURM_ARRAY_TASK_ID unset}
ARM=fc1load
export PARTIAL_STATE_CONTRACT=partial_state_v4 PARTIAL_STATE_LOAD_SECONDS=1 PARTIAL_STATE_PEER_MASS=1
export GNN_PREFIX_ALPHA_KEY=inf
unset GNN_DISABLE_MESSAGE_PASSING
if [[ $ARM == bc1mpoff ]]; then export GNN_DISABLE_MESSAGE_PASSING=1; fi
export HEROSIM_INFLIGHT_CAPTURE=service_end_v1
CKPT=models/fullctx-refine-v1-${ARM}-lr2e3-seed1.pt
REPORT=simulation_data/backlog_corpus_v1/fc1_eval_${ARM}_s1.json
OUT=simulation_data/backlog_corpus_v1/fc1_servecheck_${ARM}_s1.json
[[ -f "$REPORT" ]] || { echo "FAIL LOUD: offline report $REPORT missing (run the eval first)" >&2; exit 1; }
python3 scripts_cosim/peer_affinity_live_serve_check.py --checkpoint "$CKPT" \
  --cache-dir simulation_data/graphs_cache_backlog_corpus_v1_psv4_inf \
  --split-artifact experiments/backlog_corpus_v1_split.json --split test \
  --report "$REPORT" --limit 60 --workers 14 --output "$OUT"
echo "[done] $OUT"
