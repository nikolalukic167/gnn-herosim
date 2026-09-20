#!/bin/bash
# joint_burst_v2 -- the cache on the UNCAPPED burst corpus: v1 recipe, primary alpha rung inf, no alpha pre-scan (the loaded states are the point).
# docs/lineages/joint_burst_v2.md. One task, ~30 min.
#
#SBATCH --job-name=jb2-cache
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/jb1-cache-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/jb1-cache-%j.err
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
export DAG_PRIMARY_ALPHA_KEY=inf PARTIAL_STATE_CONTRACT=partial_state_v3
echo "[CODE] $(git describe --dirty --always) on $(git rev-parse --abbrev-ref HEAD)"
TRAIN=simulation_data/gnn_datasets_joint_burst_v2_train
TEST=simulation_data/gnn_datasets_joint_burst_v2_heldout
CACHE=simulation_data/graphs_cache_joint_burst_v2_psv3_inf
SPLIT=experiments/joint_burst_v2_split.json
DRAIN="${DRAIN_TABLE:-simulation_data/drainable_objective_v1/drain_table.json}"
RATE="${RATE:-0.46}"
[[ -f "$SPLIT" ]] && { echo "FAIL LOUD: $SPLIT exists -- a split is frozen once runs depend on it"; exit 1; }
[[ -f "$DRAIN" ]] || { echo "FAIL LOUD: drain table $DRAIN missing"; exit 1; }
for d in "$TRAIN" "$TEST"; do
  read -r n bad <<< "$(python3 - "$d" <<'PY'
import json, os, sys
d = sys.argv[1]; ok = 0; bad = []
for ds in sorted(x for x in os.listdir(d) if x.startswith("ds_")):
    try:
        m = json.load(open(os.path.join(d, ds, "placement_metadata.json")))
        rows = sum(1 for _ in open(os.path.join(d, ds, "placements", "placements.jsonl")))
        good = m.get("sweep_complete") is True and rows == m["num_placements"] and os.path.getsize(os.path.join(d, ds, "best.json")) > 0
    except Exception:
        good = False
    (ok := ok + 1) if good else bad.append(ds)
print(ok, len(bad)); print("incomplete:", " ".join(bad) or "none", file=sys.stderr)
PY
)"
  echo "[corpus] $d: $n complete sweeps, $bad incomplete"
  [[ "$bad" == "0" ]] || { echo "FAIL LOUD: $bad dataset dirs under $d are incomplete"; exit 1; }
done
echo "=== [1/6] SSC rewrite"
python3 scripts_cosim/refresh_optimal_full_stats.py --base-dir "$TRAIN" --rewrite-ssc | tail -2
python3 scripts_cosim/refresh_optimal_full_stats.py --base-dir "$TEST" --rewrite-ssc | tail -2
echo "=== [2/6] alpha pre-scan SKIPPED: joint_burst_v2 keeps the cap-infeasible (loaded) states on purpose; they carry a label at alpha=inf"
echo "=== [3/6] cache -> $CACHE"
rm -rf "$CACHE"
python3 -u src/notebooks/prepare_graphs_cache.py --base-dirs "$TRAIN" "$TEST" --cache-dir "$CACHE" \
  --dag-partial-state --platform-feature-dim 14 \
  --label-objective "rtt_drift:1" --label-arrival-rate "$RATE" --label-drain-table "$DRAIN" 2>&1 | grep -v "it/s\]" | tail -30
python3 - "$CACHE" <<'PY'
import json, sys
m = json.load(open(sys.argv[1] + "/metadata.json"))
for k, v in (("partial_state_contract", "partial_state_v3"), ("dag_primary_alpha_key", "inf"), ("peer_exchange_block", True), ("platform_feature_dim", 14)):
    assert m.get(k) == v, f"FAIL LOUD: cache {k}={m.get(k)!r} != {v!r}"
got = m.get("label_objective") or "rtt"
assert got.startswith("rtt_drift:1") and "clock=measured" in got, f"FAIL LOUD: label_objective={got!r}"
print("[cache] contract ok, label =", got, "| datasets =", m.get("num_datasets"), "| rtt rows =", m.get("num_rtt_combo_rows"))
PY
echo "=== [4/6] near-RTT sidecar"
python3 -u src/notebooks/prepare_graphs_cache_near_rtt.py --cache-dir "$CACHE" --skip-cache-build 2>&1 | grep -v "it/s\]" | tail -4
[[ -f "$CACHE/valid_combos_near_rtt_capped.pkl" ]] || { echo "FAIL LOUD: no near-RTT sidecar"; exit 1; }
echo "=== [5/6] candidate-not-in-sweep guard"
mkdir -p simulation_data/joint_burst_v2
python3 -u scripts_cosim/peer_only_v1_candidate_check.py --cache "$CACHE" --out simulation_data/joint_burst_v2/candidate_check.json
echo "=== [6/6] split artifact (held-out corpus = the 3 set-aside cells)"
python3 scripts_cosim/make_peer_affinity_t1_split.py --cache-dir "$CACHE" \
  --heldout-corpus "gnn_datasets_joint_burst_v2_heldout" --output "$SPLIT"
sha256sum "$SPLIT"; du -sh "$CACHE"
echo "=== cache done: rsync $SPLIT back, commit, then train ==="
