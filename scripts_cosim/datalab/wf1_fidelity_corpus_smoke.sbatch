#!/bin/bash
# r1_attribution_v1 preparation: smoke of the fidelity-mode corpus (make_warm_corpus --fidelity): the 6 smoke datasets rebuilt
# from the fidelity captures of wf1_fidelity_cost.sbatch, every candidate plan replayed as i11_replay replays a live decision
# (queued tasks are state, label = sum over the batch of done - scheduled). Then: acceptance checks, SSC rewrite, v5 cache.
#   sbatch --export=ALL,WT=<worktree>,CORP=<corpus_check_b dir> wf1_fidelity_corpus_smoke.sbatch
#SBATCH --job-name=wf1fidc
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/wf1fidc-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=04:00:00
set -euo pipefail
WT=${WT:?FAIL LOUD: set WT}
CORP=${CORP:?FAIL LOUD: set CORP}
TOPOS=${TOPOS:-"9601 9607"}
OUT=${OUT:-$CORP/fidelity_corpus}
cd "$WT"
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT:$WT/src/notebooks" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
echo "HEAD $(git rev-parse HEAD) node $(hostname)"
[[ ! -e "$OUT" ]] || { echo "FAIL LOUD: $OUT exists"; exit 1; }
mkdir -p "$OUT"
export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa HEROSIM_SHARED_AUTOSCALER=0 GATE_FIXED_POLICY_TIME_SCALE=1.0
export HEROSIM_PEER_EXCHANGE=1 HEROSIM_SERVER_ONLY_REPLICAS=1 HEROSIM_WARMTH_PHYSICS=node_disk_v2 PYTHONHASHSEED=0 HEROSIM_SNAPSHOT_FIDELITY=1
export MAX_PLACEMENT_COMBINATIONS_SKIP=2000000 GNN_CAPTURE_DATASET_STATE=0 COSIM_SUPPRESS_SIM_PRINTS=1 HEROSIM_RETAIN_TASK_TIMES=1 HEROSIM_RETAIN_PEER_STATS=1
export HEROSIM_TRACE_DATASET_ERRORS=1 COSIM_PLACEMENT_TIMEOUT_S=120
echo "=== [1/4] make_warm_corpus --fidelity"
CORPUS="$OUT/gnn_datasets_wf1_fidelity"; i=0
for T in $TOPOS; do
  python3 scripts_cosim/make_warm_corpus.py --snapshots "$CORP/fidelity_cost/snapshots/cc40s${T}_g0.jsonl" --trace "$CORP/inputs/wf1_hi/wl/grounded_g0_n50000.json" \
    --cell-config "$CORP/inputs/wf1_hi/cfg/cc40s${T}.json" --output-dir "$CORPUS" --source-tag "wf1fid_cc40s${T}_g0" --start-index $(( i * 100 )) --limit 3 \
    --target-combos 20000 --workers 30 --no-cap-filter --min-choice-fraction 0.2 --quiet --variable-group-min 2 --fidelity
  i=$(( i + 1 ))
done
echo "datasets: $(ls -d "$CORPUS"/ds_* | wc -l), with placements.jsonl: $(ls "$CORPUS"/ds_*/placements/placements.jsonl | wc -l)"
echo "=== [2/4] acceptance checks (sweep complete, sidecar, independent re-simulation)"
python3 scripts_cosim/wf1_fidelity_corpus_check.py --datasets "$CORPUS" --sample 25
echo "=== [3/4] SSC rewrite"
python3 scripts_cosim/refresh_optimal_full_stats.py --base-dir "$CORPUS" --rewrite-ssc | tail -2
echo "=== [4/4] v5 cache under the R1 environment (default label)"
export DAG_PRIMARY_ALPHA_KEY=inf PARTIAL_STATE_CONTRACT=partial_state_v5 NEAR_RTT_LABEL_BACKLOG_CLOCK=seeded
python3 -u src/notebooks/prepare_graphs_cache.py --base-dirs "$CORPUS" --cache-dir "$OUT/cache_v5" --dag-partial-state --platform-feature-dim 14 2>&1 | grep -v "it/s\]" | tail -12 | cut -c1-260
python3 - "$OUT/cache_v5" <<'PY'
import json, pickle, sys
c = sys.argv[1]
m = json.load(open(c + "/metadata.json"))
g = pickle.load(open(c + "/graphs.pkl", "rb"))
print("cache:", len(g), "graphs; task", tuple(g[0].task_features.shape), "platform", tuple(g[0].platform_features.shape),
      "contract", m.get("partial_state_contract"), "physics_env", m.get("physics_env"))
PY
