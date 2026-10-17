#!/bin/bash
# partial_state_v6 recache of the Arm 1 corpus: the existing Arm 1 cache view (stored states, SSC files unchanged) built into a partial_state_v6 cache, i.e. the v5 columns byte for byte plus the
# two in-flight platform columns (placement/inflight_features.py). The view is only READ; nothing under it changes.
#   sbatch --export=ALL,WT=<pin>,EXPECT_HEAD=<sha>,VIEW=<old cache_run/view>,OUT=<new dir>[,LIMIT=<n datasets per split, smoke>] uncut_arm1_cache_v6.sbatch
#SBATCH --job-name=arm1-cache-v6
#SBATCH --output=/share/nikola.lukic/uncut_groups/inflight/logs/%x-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=04:00:00
set -euo pipefail
WT=${WT:?}; OUT=${OUT:?}; VIEW=${VIEW:?}; EXPECT_HEAD=${EXPECT_HEAD:?}; LIMIT=${LIMIT:-0}
OLDCACHE=${OLDCACHE:-$(dirname "$VIEW")/cache}
cd "$WT"
for v in $(compgen -e | grep -E '^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|NEAR_RTT_|PARTIAL_STATE)'); do unset "$v"; done
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"; micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT:$WT/src/notebooks" PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
HEAD=$(git rev-parse HEAD); echo "HEAD $HEAD node $(hostname)"
[[ "$HEAD" == "$EXPECT_HEAD"* ]] || { echo "FAIL LOUD: worktree HEAD $HEAD is not $EXPECT_HEAD"; exit 1; }
[[ ! -e "$OUT" ]] || { echo "FAIL LOUD: $OUT exists"; exit 1; }
mkdir -p "$OUT"
export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa HEROSIM_SHARED_AUTOSCALER=0 GATE_FIXED_POLICY_TIME_SCALE=1.0
# the Arm 1 recipe's other flags, as the v5 cache was built (uncut_arm1_cache.sbatch): only the contract differs
export DAG_PRIMARY_ALPHA_KEY=inf PARTIAL_STATE_CONTRACT=partial_state_v6 NEAR_RTT_LABEL_BACKLOG_CLOCK=seeded
if [[ "$LIMIT" != "0" ]]; then
  # smoke: the first LIMIT datasets of each split, linked into a private view (the dataset ids are the old ones, so the old graphs can be compared id for id)
  mkdir -p "$OUT/view"
  for B in "$VIEW"/*/; do n=$(basename "$B"); mkdir "$OUT/view/$n"; for d in $(ls "$B" | head -n "$LIMIT"); do ln -s "$(readlink -f "$B/$d")" "$OUT/view/$n/$d"; done; done
  BASE="$OUT/view"
else
  BASE="$VIEW"
fi
T0=$(date +%s)
python3 -u src/notebooks/prepare_graphs_cache.py --base-dirs "$BASE"/*/ --cache-dir "$OUT/cache" --dag-partial-state --platform-feature-dim 14 > "$OUT/cache.log" 2>&1 || { tail -30 "$OUT/cache.log"; echo "FAIL LOUD: cache build failed"; exit 1; }
echo "cache build wall seconds: $(( $(date +%s) - T0 ))"
grep -vE "it/s\]|^\s*$" "$OUT/cache.log" | tail -12 | cut -c1-200
python3 scripts_cosim/uncut_inflight_cache_check.py --old "$OLDCACHE" --new "$OUT/cache" --view "$BASE" --out "$OUT/inflight_check.json"
echo "=== done; cache in $OUT/cache"
