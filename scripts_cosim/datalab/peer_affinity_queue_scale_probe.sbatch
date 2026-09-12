#!/bin/bash
# Audit 2026-09-12 -- does the LIVE queue magnitude erase the offline message-passing edge?
#
# Measured first: legacy_v0's dim-7 divisor is min(max(1, p90 over ALL platforms), 100), and
# that p90 is 0 in 516/516 training datasets (3.7 % of platforms ever busy, server-only
# replicas), so the divisor is 1.0 everywhere in training and the "normalized" queue column
# is the RAW depth, 0-42. On the production trace the depth behind a chosen platform is
# ~1.3e4, i.e. k ~ 300x the training range (docs/gates/gate-tools.md, 2026-09-12).
#
# scripts_cosim/queue_scale_probe.py presents the same held-out co-sim datasets with the two
# queue columns multiplied by k and re-measures each arm's decode regret against the SAME
# brute-force optimum. On x200 p2 (run locally) it reproduces the registered T1b contrast at
# k=1 (+5.14 pp, p=0.001, 13/16 -- the published number to the decimal) and then flips sign
# by k=10: -2.56 pp, reaching -3.46 pp (p=0.025) at k=30. Degradation from k=1 is +16.3 pp
# for `gnn` against +6.6 pp for `mpoff` -- 2.5x, which is what the architecture predicts: with
# MP off a task embedding never touches a platform feature, so the pointwise twin meets the
# corrupted column once, at its own candidate's row, while the GIN mixes every reachable
# platform embedding into every task embedding and PeerConv then spreads it across peers.
#
# This array runs the same probe on the two x800 corpora, whose live contrasts are the ones
# that actually reversed (-4.42 % and -10.67 %). Prediction if the mechanism is the cause:
# the crossover k is LOWER on the denser graph (more peer edges = more exposure).
#
#SBATCH --job-name=pa-qscale
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/pa-qscale-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/pa-qscale-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --array=0-3

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

# the offline read's environment, verbatim (scripts_cosim/peer_affinity_t1_read.py)
export PARTIAL_STATE_CONTRACT=partial_state_v2
export EVAL_DECODE_REPLICA_REUSE=1
export EVAL_DECODE_RELAX=1

case ${SLURM_ARRAY_TASK_ID} in
  0) RUNG=p2; ARM=gnn ;;
  1) RUNG=p2; ARM=mpoff ;;
  2) RUNG=p3; ARM=gnn ;;
  3) RUNG=p3; ARM=mpoff ;;
  *) echo "FAIL LOUD: unknown task ${SLURM_ARRAY_TASK_ID}" >&2; exit 1 ;;
esac
# The loader verifies this against the sidecar and raises on a mismatch, so it must be
# exported per arm -- an unset flag resolves to MP ON and would score mpoff through GIN
# weights it never trained (route_b 2026-09-03, 5.7x error).
if [[ $ARM == mpoff ]]; then export GNN_DISABLE_MESSAGE_PASSING=1; else export GNN_DISABLE_MESSAGE_PASSING=0; fi

CACHE=simulation_data/graphs_cache_peer_affinity_v1_x800_${RUNG}
SPLIT=experiments/peer_affinity_v1_x800_${RUNG}_split.json
TAG=peer-affinity-v1-x800${RUNG}
OUT=simulation_data/peer_affinity_queue_scale_probe_x800${RUNG}_${ARM}.json
mkdir -p logs
[[ -d "$CACHE" ]] || { echo "FAIL LOUD: cache missing: $CACHE" >&2; exit 1; }
[[ -f "$SPLIT" ]] || { echo "FAIL LOUD: split artifact missing: $SPLIT" >&2; exit 1; }
[[ -f "models/${TAG}-${ARM}-lr2e3-seed1.pt" ]] || { echo "FAIL LOUD: checkpoints missing for $TAG/$ARM" >&2; exit 1; }
echo "[TASK ${SLURM_ARRAY_TASK_ID}] rung=$RUNG arm=$ARM mp_off=${GNN_DISABLE_MESSAGE_PASSING} -> $OUT"
if [[ -f "$OUT" ]]; then echo "[TASK] output exists, not re-running: $OUT"; exit 0; fi

${HEROSIM_PY:-pipenv run python3} scripts_cosim/queue_scale_probe.py \
  --cache-dir "$CACHE" --split-artifact "$SPLIT" --tag "$TAG" --lr lr2e3 \
  --arms "$ARM" --seeds 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 \
  --scales 1.0 3.0 10.0 30.0 100.0 300.0 --alpha-key 2.5 --split test \
  --out "$OUT"
echo "[DONE] $(date)"
