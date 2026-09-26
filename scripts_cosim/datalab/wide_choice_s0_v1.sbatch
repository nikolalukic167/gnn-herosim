#!/bin/bash
# wide_choice_s0_v1 (docs/lineages/wide_choice_s0_v1.md): wide-slate held-out groups from the
# backlog_corpus_v1 held-out snapshot shards (cc40s9223/9224 x w0/w1, backlog rungs 0 and 3 only),
# exhaustively swept; then CD, the 1-pass greedy and xs1load s1-4 (one-pass / self-refine / GNN-seeded
# CD) replayed through the engine; then the W read. Sweeps are gzipped afterwards (/home budget).
# Runs in the worktree ~/gnn-herosim-wt/wide_choice (simulation_data inputs and models are symlinks).
#   sbatch --export=ALL,MODE=smoke,SMOKE=1 wide_choice_s0_v1.sbatch        (2 groups, tier A, shard 0, replay, read)
#   G=$(sbatch --parsable --export=ALL,MODE=gen --array=0-7 wide_choice_s0_v1.sbatch)   (tasks 0-3 tier A, 4-7 tier B)
#   sbatch --export=ALL,MODE=read --dependency=afterok:$G wide_choice_s0_v1.sbatch
#SBATCH --job-name=wc-s0
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/wc-s0-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/wc-s0-%j.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=240G
#SBATCH --time=08:00:00
set -euo pipefail
WT=/home/nikola.lukic/gnn-herosim-wt/wide_choice
cd "$WT"
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$WT"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"
echo "[CODE] $(git describe --dirty --always) at $WT"
[[ -z "$(git status --porcelain -- src scripts_cosim experiments | grep -v '^??' || true)" ]] || { echo "FAIL LOUD: worktree dirty" >&2; exit 1; }
[[ -L models ]] || { echo "FAIL LOUD: models/ must be the symlink into ~/gnn-herosim/models" >&2; exit 1; }
DRAIN=simulation_data/drainable_objective_v1/drain_table.json
[[ -f "$DRAIN" ]] || { echo "FAIL LOUD: drain table $DRAIN missing" >&2; exit 1; }
# the corpus physics, verbatim from backlog_corpus_v1_corpus.sbatch
export HEROSIM_PEER_EXCHANGE=1 HEROSIM_RETAIN_TASK_TIMES=1 HEROSIM_RETAIN_PEER_STATS=1
export MAX_PLACEMENT_COMBINATIONS_SKIP=2000000 GNN_CAPTURE_DATASET_STATE=0 COSIM_SUPPRESS_SIM_PRINTS=1
export COSIM_AUTOSCALER_RECONCILE_INTERVAL=1e12
export HEROSIM_BACKLOG_DRAIN_TABLE="$DRAIN"

CKPTS=()
for s in 1 2 3 4; do CKPTS+=("models/exchange-seconds-v1-xs1load-lr2e3-seed$s.pt"); done
OUTROOT=simulation_data/wide_choice_s0_v1${SMOKE:+/smoke}
G=simulation_data/peer_affinity_live_gate
WORKERS=$(( SLURM_CPUS_PER_TASK - 1 ))
SHARDS=("cc40s9223 w0" "cc40s9223 w1" "cc40s9224 w0" "cc40s9224 w1")
declare -A TARGET=([A]=100000 [B]=250000) LIMIT=([A]=50 [B]=15)
[[ -n "${SMOKE:-}" ]] && LIMIT=([A]=2 [B]=1)

gen() {  # gen <tier> <shard index>
  local NAME=$1 i=$2 CELL WINDOW
  read -r CELL WINDOW <<< "${SHARDS[$i]}"
  local OUT=$OUTROOT/gnn_datasets_wide_$NAME
  local SHARD=simulation_data/joint_burst_v2/shards/${CELL}_${WINDOW}_shard0.jsonl
  local TRACE=$G/workloads/burst_drainable_${WINDOW}_n50000.json CFG=$G/configs/${CELL}.json
  [[ $WINDOW == w0 ]] && TRACE=$G/workloads/burst_drainable_f4000_n50000.json
  [[ -s "$SHARD" && -f "$CFG" && -f "$TRACE" ]] || { echo "FAIL LOUD: inputs missing for $CELL $WINDOW" >&2; exit 1; }
  mkdir -p "$OUT/shard$i"
  echo "=== tier $NAME target ${TARGET[$NAME]}: $CELL $WINDOW -> $OUT/shard$i (limit ${LIMIT[$NAME]})"
  local t0=$SECONDS
  # one output dir per array task: warm_manifest.jsonl is appended per dataset, never shared across tasks
  python3 scripts_cosim/make_warm_corpus.py --snapshots "$SHARD" --trace "$TRACE" --cell-config "$CFG" \
    --output-dir "$OUT/shard$i" --source-tag "wc${NAME}_${CELL}_${WINDOW}" --start-index $(( i * 1000 )) --limit "${LIMIT[$NAME]}" \
    --target-combos "${TARGET[$NAME]}" --workers "$WORKERS" --no-cap-filter --min-choice-fraction 0.8 --quiet \
    --synthetic-backlog-rungs 0 3 --synthetic-backlog-busy-prob 0.5 --synthetic-backlog-task-seconds 4.5 \
    --synthetic-backlog-seed 7101
  local n; n=$(grep -c '"status": "success"' "$OUT/shard$i/warm_manifest.jsonl" || true)
  echo "=== $n groups in $(( SECONDS - t0 )) s"; du -sh "$OUT/shard$i"
  [[ "$n" -ge 1 ]] || { echo "FAIL LOUD: no group generated for $CELL $WINDOW" >&2; exit 1; }
}

read_all() {
  local ARGS=() NAME OUT
  for NAME in "$@"; do
    OUT=$OUTROOT/gnn_datasets_wide_$NAME
    cat "$OUT"/shard*/warm_manifest.jsonl > "$OUT/warm_manifest_all.jsonl"
    echo "=== replay tier $NAME"
    local t0=$SECONDS
    python3 scripts_cosim/wide_choice_s0_v1_replay.py --datasets-dir "$OUT"/shard* --checkpoints "${CKPTS[@]}" \
      --workers "$WORKERS" --output "$OUTROOT/replay_$NAME.json"
    echo "=== replay in $(( SECONDS - t0 )) s"
    ARGS+=(--tier "$NAME" "$OUTROOT/replay_$NAME.json" "$OUT/warm_manifest_all.jsonl")
  done
  python3 scripts_cosim/wide_choice_s0_v1_read.py "${ARGS[@]}" --out "$OUTROOT/wide_choice_s0_v1_read.json" | tail -90
  echo "=== gzip sweeps"
  find "$OUTROOT" -name placements.jsonl -exec gzip -f {} +
  du -sh "$OUTROOT"
}

case "${MODE:-}" in
  smoke) gen A 0; read_all A ;;
  gen)   T=${SLURM_ARRAY_TASK_ID:?FAIL LOUD: array task id}; NAMES=(A B); gen "${NAMES[$(( T / 4 ))]}" $(( T % 4 )) ;;
  read)  read_all A B ;;
  *) echo "FAIL LOUD: MODE must be smoke, gen or read" >&2; exit 1 ;;
esac
echo "=== done"
