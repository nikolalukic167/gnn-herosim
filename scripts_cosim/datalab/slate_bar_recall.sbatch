#!/bin/bash
# slate_bar_v1 recall audit (docs/lineages/slate_bar_v1.md, S6 section): scripts_cosim/slate_bar_v1_recall.py sample | run. CPU only.
#   sbatch --export=ALL,WT=<pinned worktree>,EXPECT_HEAD=<sha>,MODE=sample|run,ROOT=<out dir>[,LIMIT=<n>,REPEAT=<n>] slate_bar_recall.sbatch
#   sample writes ROOT/jobs.jsonl; run reads it and writes ROOT/rows.jsonl (ROOT/smoke_rows.jsonl when LIMIT is set)
#   submit with --output=/share/nikola.lukic/slate_bar/logs/sb-recall-%j.out (/home is near quota)
#SBATCH --job-name=sb-recall
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/sb-recall-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
set -euo pipefail
WT=${WT:?FAIL LOUD: set WT}; EXPECT_HEAD=${EXPECT_HEAD:?}; MODE=${MODE:?}; ROOT=${ROOT:?}
CORPUS=${CORPUS:-/share/nikola.lukic/workload_fix_v1/accel_replica_corpus}
SPLIT=${SPLIT:-/share/nikola.lukic/long_horizon/inputs_0605a5d2/inputs/accel_replica_v1_split.json}
STATES=${STATES:-/share/nikola.lukic/big_groups/states/top5}
cd "$WT"
for v in $(compgen -e | grep -E '^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|NEAR_RTT_|PARTIAL_STATE|WF1_|DECLARED_SLATE)'); do unset "$v"; done
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa HEROSIM_SHARED_AUTOSCALER=0 GATE_FIXED_POLICY_TIME_SCALE=1.0
export HEROSIM_PEER_EXCHANGE=1 HEROSIM_SERVER_ONLY_REPLICAS=1 HEROSIM_WARMTH_PHYSICS=node_disk_v2 PYTHONHASHSEED=0 HEROSIM_GNN_DEVICE=cpu
export GNN_DECODE_MODE=masked_topo GNN_BATCH_BY_PEER_GROUP=1 HEROSIM_SNAPSHOT_FIDELITY=1
export HEROSIM_RAW_DIR=/tmp/sbr_${SLURM_JOB_ID:-$$}; mkdir -p "$HEROSIM_RAW_DIR"; trap 'rm -rf "$HEROSIM_RAW_DIR"' EXIT
python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"
HEAD=$(git rev-parse HEAD)
echo "HEAD $HEAD node $(hostname) job=${SLURM_JOB_ID:-} mode=$MODE root=$ROOT"
[[ "$HEAD" == "$EXPECT_HEAD"* ]] || { echo "FAIL LOUD: worktree HEAD $HEAD is not $EXPECT_HEAD"; exit 1; }
mkdir -p "$ROOT"
case "$MODE" in
  sample)
    [[ ! -e "$ROOT/jobs.jsonl" ]] || { echo "FAIL LOUD: $ROOT/jobs.jsonl exists"; exit 1; }
    python3 scripts_cosim/slate_bar_v1_recall.py sample "$ROOT/jobs.jsonl" --corpus "$CORPUS" --split "$SPLIT" --states "$STATES" --workers 16
    head -1 "$ROOT/jobs.jsonl" | cut -c1-2000;;
  run)
    OUT="$ROOT/rows.jsonl"; EXTRA="--repeat ${REPEAT:-20}"
    [[ -z "${LIMIT:-}" ]] || { OUT="$ROOT/smoke_rows.jsonl"; EXTRA="--repeat ${REPEAT:-0} --limit $LIMIT"; }
    [[ ! -e "$OUT" ]] || { echo "FAIL LOUD: $OUT exists"; exit 1; }
    python3 scripts_cosim/slate_bar_v1_recall.py run "$OUT" "$ROOT/jobs.jsonl" --workers "${WORKERS:-15}" $EXTRA
    python3 scripts_cosim/slate_bar_v1_recall.py read "$OUT" --json "${OUT%.jsonl}.read.json" | head -1;;
  *) echo "FAIL LOUD: MODE=$MODE"; exit 1;;
esac
