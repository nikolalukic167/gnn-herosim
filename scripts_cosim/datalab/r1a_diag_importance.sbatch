#!/bin/bash
# r1_attribution_v1 diagnostic (3a), read-only and VALIDATION split only: permutation importance per feature group for gnn_eng, twin_eng and mlp_same
# (one checkpoint each, a mid-training snapshot unless training has finished), on the production cache. One job, three checkpoints in sequence.
#   sbatch --export=ALL,WT=<worktree with r1a_permutation_importance.py>,CACHE=<cache dir>,SPLIT=<split.json>,OUT=<new dir>[,CKPTS="a.pt b.pt c.pt",LIMIT=N,REPEATS=3] r1a_diag_importance.sbatch
#SBATCH --job-name=r1a-dimp
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/r1a-dimp-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
set -uo pipefail
WT=${WT:?}; CACHE=${CACHE:?}; SPLIT=${SPLIT:?}; OUT=${OUT:?}; LIMIT=${LIMIT:-0}; REPEATS=${REPEATS:-3}
M=/home/nikola.lukic/gnn-herosim/models
CKPTS=${CKPTS:-"$M/r1-attribution-v1-gnn-eng-g1-seed1.pt $M/r1-attribution-v1-twin-eng-g1-seed1.pt $M/r1-attribution-v1-mlp-same-g1-seed1.pt"}
cd "$WT"
for v in $(compgen -e | grep -E '^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|NEAR_RTT_|PARTIAL_STATE)'); do unset "$v"; done
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT:$WT/src/notebooks" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""
export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa HEROSIM_SHARED_AUTOSCALER=0 GATE_FIXED_POLICY_TIME_SCALE=1.0
export GNN_DECODE_MODE=masked_topo GNN_PREFIX_ALPHA_KEY=inf GNN_SERVE_CANDIDATE_SLATE=declared_pruning_v1
echo "HEAD $(git rev-parse HEAD) node $(hostname)"
[[ ! -e "$OUT" ]] || { echo "FAIL LOUD: $OUT exists"; exit 1; }
mkdir -p "$OUT"
for C in $CKPTS; do
  # a checkpoint snapshot: the file may be rewritten by a running training, so work on a private copy
  B=$(basename "$C" .pt); cp "$C" "$OUT/$B.pt"; cp "${C%.pt}.contract.json" "$OUT/$B.contract.json"
  echo "=== $B (copied $(date +%T))"
  python3 scripts_cosim/r1a_permutation_importance.py --ckpt "$OUT/$B.pt" --cache-dir "$CACHE" --split "$SPLIT" --out "$OUT/$B.importance.json" \
    --repeats "$REPEATS" --limit "$LIMIT" 2>&1 | grep -v Warn
done
