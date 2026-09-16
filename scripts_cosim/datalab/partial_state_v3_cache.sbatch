#!/bin/bash
# partial_state_v3 -- the v3 cache: the SAME 516 T1b parents, the SAME split, the SAME V = 1
# label as graphs_cache_drainable_objective_v1_v1, built under PARTIAL_STATE_CONTRACT=
# partial_state_v3. docs/lineages/partial_state_v3.md, bar P0.
#
# Recipe = drainable_objective_v1_cache.sbatch task 0 (V = 1), verbatim except the contract
# and the cache name. The cache stores the partial-state INGREDIENTS (node_rank, caps,
# routes, peer table), not the columns -- the columns are computed under the contract at
# train and serve time -- so this cache is expected to differ from the v2 cache only in
# metadata.json. P0 measures that on all 516 datasets rather than assuming it.
#
#   sbatch scripts_cosim/datalab/partial_state_v3_cache.sbatch
#
#SBATCH --job-name=psv3-cache
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/psv3-cache-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/psv3-cache-%j.err
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
export DAG_PRIMARY_ALPHA_KEY=2.5 PARTIAL_STATE_CONTRACT=partial_state_v3
python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"
echo "[CODE] $(git describe --dirty --always) on $(git rev-parse --abbrev-ref HEAD)"
python3 -c "from src.policy.tabular.reduced_features import partial_state_feature_dim as d; assert d()==22, d(); print('[contract] partial_state_v3 -> 22 columns')"

V=1
TRAIN=simulation_data/gnn_datasets_peer_affinity_v1_c3_x200_train
TRAIN2=simulation_data/gnn_datasets_peer_affinity_v1_c3_x200_train2
TEST=simulation_data/gnn_datasets_peer_affinity_v1_c3_x200_r2
PARENTS=experiments/drainable_objective_v1_t1b_parents.json
CACHE=simulation_data/graphs_cache_drainable_objective_v1_v1_psv3
V2CACHE=simulation_data/graphs_cache_drainable_objective_v1_v1
SPLIT=experiments/peer_affinity_v1_t1b_split.json
DRAIN="${DRAIN_TABLE:-simulation_data/drainable_objective_v1/drain_table.json}"
RATE="${RATE:-0.46}"

echo "[cache] V=$V -> $CACHE (v2 twin: $V2CACHE)"
[[ -f "$DRAIN" ]] || { echo "FAIL LOUD: missing $DRAIN" >&2; exit 1; }
[[ -f "$SPLIT" ]] || { echo "FAIL LOUD: $SPLIT missing" >&2; exit 1; }
[[ -f "$PARENTS" ]] || { echo "FAIL LOUD: $PARENTS missing" >&2; exit 1; }
[[ -f "$V2CACHE/metadata.json" ]] || { echo "FAIL LOUD: v2 cache $V2CACHE missing -- P0 has nothing to compare against" >&2; exit 1; }
for d in "$TRAIN" "$TRAIN2" "$TEST"; do [[ -d "$d" ]] || { echo "FAIL LOUD: corpus missing: $d" >&2; exit 1; }; done

echo "=== [1/4] cache -> $CACHE"
rm -rf "$CACHE"
python3 -u src/notebooks/prepare_graphs_cache.py --base-dirs "$TRAIN" "$TRAIN2" "$TEST" --cache-dir "$CACHE" \
  --dag-partial-state --platform-feature-dim 14 \
  --oversample-manifest "$PARENTS" \
  --label-objective "rtt_drift:$V" \
  --label-arrival-rate "$RATE" \
  --label-drain-table "$DRAIN" 2>&1 | grep -v "it/s\]" | tail -40

python3 - "$CACHE" "$V2CACHE" <<'PY'
import json, sys
m = json.load(open(sys.argv[1] + "/metadata.json")); m2 = json.load(open(sys.argv[2] + "/metadata.json"))
need = {"partial_state_contract": "partial_state_v3", "dag_primary_alpha_key": "2.5",
        "peer_exchange_block": True, "platform_feature_dim": 14}
for k, val in need.items():
    assert m.get(k) == val, f"FAIL LOUD: cache {k}={m.get(k)!r} != {val!r}"
if m.get("num_datasets") != 516:
    raise SystemExit(f"FAIL LOUD: cache holds {m.get('num_datasets')} datasets, the corpus is 516")
# Same corpus, same label, same clock as the v2 twin: only the contract may differ.
for k in ("num_datasets", "label_objective", "dag_primary_alpha_key", "platform_feature_dim", "num_rtt_combo_rows"):
    if m.get(k) != m2.get(k):
        raise SystemExit(f"FAIL LOUD: v3 cache {k}={m.get(k)!r} but v2 twin has {m2.get(k)!r}")
print("[cache] contract =", m["partial_state_contract"], "| label =", m["label_objective"],
      "| datasets =", m["num_datasets"], "| rtt rows =", m.get("num_rtt_combo_rows"))
PY

echo "=== [2/4] near-RTT sidecar"
python3 -u src/notebooks/prepare_graphs_cache_near_rtt.py --cache-dir "$CACHE" --skip-cache-build 2>&1 | grep -v "it/s\]" | tail -4
[[ -f "$CACHE/valid_combos_near_rtt_capped.pkl" ]] || { echo "FAIL LOUD: no near-RTT sidecar" >&2; exit 1; }

echo "=== [3/4] split: REUSED from T1b, asserted to cover this cache"
python3 - "$CACHE" "$SPLIT" <<'SPLITCHK'
import json, pickle, sys
cache, split = sys.argv[1], sys.argv[2]
ids = list(pickle.load(open(cache + "/dataset_ids.pkl", "rb")))
sp = json.load(open(split))
have = set(ids) | {i.split("/")[-1] for i in ids}
missing = [p for key in ("train", "val", "test") for p in (sp.get(key) or [])
           if p not in have and p.split("/")[-1] not in have]
if missing:
    raise SystemExit(f"FAIL LOUD: {len(missing)} split parents absent from the cache, e.g. {missing[:3]}")
print("[split] T1b split covers this cache: %d datasets" % len(ids))
SPLITCHK

echo "=== [4/4] P0: the instrument bar, on all 516 datasets"
python3 -u scripts_cosim/partial_state_v3_p0.py --v2-cache "$V2CACHE" --v3-cache "$CACHE" \
  --out simulation_data/partial_state_v3/p0.json
du -sh "$CACHE"
echo "=== v3 cache done ==="
