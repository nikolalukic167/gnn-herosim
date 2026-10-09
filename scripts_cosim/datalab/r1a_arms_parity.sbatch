#!/bin/bash
# r1_attribution_v1: train/serve feature parity for each learned arm, under each HEROSIM_INFLIGHT_CAPTURE setting.
# wf1_fidelity_parity.py replays each dataset's snapshot under the live scheduler and compares the graph the arm's checkpoint is served
# with against the cache graph, attribute by attribute. The cache was built from snapshots captured with the capture mode UNSET (legacy).
#   sbatch --export=ALL,WT=<pinned worktree>,DS=<corpus dir with ds_*>,CACHE=<v5 cache>,OUT=<new dir> r1a_arms_parity.sbatch
#SBATCH --job-name=r1a-par
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/r1a-par-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
set -uo pipefail
WT=${WT:?}; DS=${DS:?}; CACHE=${CACHE:?}; OUT=${OUT:?}
ARMS=${ARMS:-"gnn_eng twin_eng mlp_same gnn_raw twin_raw gnn_eng_physmp set_transformer"}
MODES=${MODES:-"legacy service_end_v1"}
cd "$WT"
for v in $(compgen -e | grep -E '^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|NEAR_RTT_|PARTIAL_STATE)'); do unset "$v"; done
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT:$WT/src/notebooks" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=""
echo "HEAD $(git rev-parse HEAD)"
[[ ! -e "$OUT" ]] || { echo "FAIL LOUD: $OUT exists"; exit 1; }
mkdir -p "$OUT"
export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa HEROSIM_SHARED_AUTOSCALER=0 GATE_FIXED_POLICY_TIME_SCALE=1.0
export HEROSIM_PEER_EXCHANGE=1 HEROSIM_SERVER_ONLY_REPLICAS=1 HEROSIM_WARMTH_PHYSICS=node_disk_v2 PYTHONHASHSEED=0 HEROSIM_SNAPSHOT_FIDELITY=1
export SIM_FORCE_FULL_STATS=1 DAG_PRIMARY_ALPHA_KEY=inf PARTIAL_STATE_CONTRACT=partial_state_v5 PARTIAL_STATE_LOAD_SECONDS=1 PARTIAL_STATE_PEER_MASS=1 PARTIAL_STATE_EXCHANGE_SECONDS=1
export NEAR_RTT_LABEL_BACKLOG_CLOCK=seeded GNN_PREFIX_ALPHA_KEY=inf GNN_SERVE_CANDIDATE_SLATE=declared_pruning_v1
for MODE in $MODES; do
  for ARM in $ARMS; do
    RUN=r1-attribution-v1-${ARM//_/-}-smoke-seed1
    if [[ "$MODE" == legacy ]]; then unset HEROSIM_INFLIGHT_CAPTURE; else export HEROSIM_INFLIGHT_CAPTURE=$MODE; fi
    python3 scripts_cosim/wf1_fidelity_parity.py --datasets "$DS" --cache-dir "$CACHE" --checkpoint "models/$RUN.pt" --workers 6 \
      --out "$OUT/${MODE}__${ARM}.json" > "$OUT/${MODE}__${ARM}.log" 2>&1
    echo "[$MODE $ARM] rc=$? $(tail -2 "$OUT/${MODE}__${ARM}.log" | tr '\n' ' ' | cut -c1-300)"
  done
done
