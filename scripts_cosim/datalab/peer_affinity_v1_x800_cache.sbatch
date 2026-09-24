#!/bin/bash
# peer_affinity_v1 -- denser-graph rung: build the training cache + split artifact for one x800
# corpus family ON DATALAB (the corpora live here; local disk holds no more). Recipe = T1b's
# build (scratch t1b_build.sh, node "Training registration T1b") with the set-aside step made
# explicit:
#   1. SSC rewrite (refresh_optimal_full_stats --rewrite-ssc) on train and held-out
#   2. alpha pre-scan: datasets with no feasible sweep row at the tightest rung (2.0) go to
#      <corpus>_setaside/ -- T1b did this by hand for 4/350; the count is printed and recorded
#   3. prepare_graphs_cache --dag-partial-state --platform-feature-dim 14 (alpha key 2.5, v2)
#   4. near-RTT sidecar (valid_combos_near_rtt_capped.pkl) -- GNN tasks die without it
#   5. split artifact with the r2 corpus as the held-out block (rsync back + commit, then train)
#   RUNG=p3 (3 partners, 800 MB)  |  RUNG=p2 (matched control: 2 partners, 800 MB)
#   sbatch --export=ALL,RUNG=p3 scripts_cosim/datalab/peer_affinity_v1_x800_cache.sbatch
#
#SBATCH --job-name=pa-x800-cache
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/pa-x800-cache-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/pa-x800-cache-%j.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PWD:$PWD/src/notebooks"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export DAG_PRIMARY_ALPHA_KEY=2.5 PARTIAL_STATE_CONTRACT=partial_state_v2
python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"
echo "[CODE] $(git describe --dirty --always) on $(git rev-parse --abbrev-ref HEAD)"

RUNG="${RUNG:?set RUNG=p3 or RUNG=p2}"
TRAIN=simulation_data/gnn_datasets_peer_affinity_v1_c3_x800_${RUNG}_train
TEST=simulation_data/gnn_datasets_peer_affinity_v1_c3_x800_${RUNG}_r2
CACHE=simulation_data/graphs_cache_peer_affinity_v1_x800_${RUNG}
SPLIT=experiments/peer_affinity_v1_x800_${RUNG}_split.json
[[ -f "$SPLIT" ]] && { echo "FAIL LOUD: $SPLIT exists -- a split is frozen once runs depend on it"; exit 1; }
for d in "$TRAIN" "$TEST"; do
  n=$(find "$d" -mindepth 3 -maxdepth 3 -path '*/placements/placements.jsonl' -size +0 | wc -l)
  echo "[corpus] $d: $n complete sweeps"
  bad=$(find "$d" -mindepth 1 -maxdepth 1 -type d -name 'ds_*' | while read -r ds; do [[ -s "$ds/placements/placements.jsonl" && -s "$ds/best.json" ]] || echo "$ds"; done | wc -l)
  [[ "$bad" == "0" ]] || { echo "FAIL LOUD: $bad dataset dirs under $d lack best.json or a non-empty sweep -- generation not finished"; exit 1; }
done

echo "=== [1/5] SSC rewrite"
python3 scripts_cosim/refresh_optimal_full_stats.py --base-dir "$TRAIN" --rewrite-ssc | tail -2
python3 scripts_cosim/refresh_optimal_full_stats.py --base-dir "$TEST" --rewrite-ssc | tail -2
for d in "$TRAIN" "$TEST"; do
  [[ $(ls "$d"/ds_*/system_state_captured_unique.json | wc -l) -eq $(ls -d "$d"/ds_* | wc -l) ]] || { echo "FAIL LOUD: SSC missing under $d"; exit 1; }
done

echo "=== [2/5] alpha pre-scan (tightest rung 2.0)"
python3 scripts_cosim/peer_affinity_alpha_prescan.py "$TRAIN" "$TEST" --alpha 2.0 --set-aside-dir "simulation_data/gnn_datasets_peer_affinity_v1_c3_x800_${RUNG}_setaside" | tail -12

echo "=== [3/5] cache -> $CACHE"
rm -rf "$CACHE"
python3 -u src/notebooks/prepare_graphs_cache.py --base-dirs "$TRAIN" "$TEST" --cache-dir "$CACHE" \
  --dag-partial-state --platform-feature-dim 14 2>&1 | grep -v "it/s\]" | tail -40
python3 - "$CACHE" <<'PY'
import json, sys
m = json.load(open(sys.argv[1] + "/metadata.json"))
need = {"partial_state_contract": "partial_state_v2", "dag_primary_alpha_key": "2.5", "peer_exchange_block": True, "platform_feature_dim": 14}
for k, v in need.items():
    assert m.get(k) == v, f"FAIL LOUD: cache {k}={m.get(k)!r} != {v!r}"
print("[cache] contract ok, num_datasets =", m.get("num_datasets"), "rtt rows =", m.get("num_rtt_combo_rows"))
PY

echo "=== [4/5] near-RTT sidecar"
python3 -u src/notebooks/prepare_graphs_cache_near_rtt.py --cache-dir "$CACHE" --skip-cache-build 2>&1 | grep -v "it/s\]" | tail -4
[[ -f "$CACHE/valid_combos_near_rtt_capped.pkl" ]] || { echo "FAIL LOUD: no near-RTT sidecar"; exit 1; }

echo "=== [5/5] split artifact"
python3 scripts_cosim/make_peer_affinity_t1_split.py --cache-dir "$CACHE" \
  --heldout-corpus "gnn_datasets_peer_affinity_v1_c3_x800_${RUNG}_r2" --output "$SPLIT"
sha256sum "$SPLIT"
du -sh "$CACHE"
echo "=== x800 $RUNG cache done: rsync $SPLIT back, commit, then submit the train sbatch"
