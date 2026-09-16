#!/bin/bash
# peer_only_v1 -- Phase B corpus: the 1,670-dataset cache under partial_state_v3, on datalab.
# docs/lineages/peer_only_v1.md, bar B0.
#
# gnn_datasets_peer_affinity_v1_c3_x200_train2 on datalab is the 1,500-dataset in-place
# extension; its 1,154 post-T1b datasets never received the T1b preparation, which is why
# no cache over the full corpus has ever been built (job 769544 and three dobj-cache jobs
# died on the missing SSC files). This job applies that preparation, in the T1b/W1 recipe's
# order, then builds ONE cache over train (136) + train2 (1,500 minus alpha set-asides) +
# r2 (34), no parent manifest, same V = 1 measured-clock label and alpha 2.5 as the psv3
# cache, and ends with B0 (blocking): the 516 T1b parents must carry identical partial-state
# ingredients to the psv3 cache and the 34-dataset test split must be present.
#
#   1. refresh_optimal_full_stats.py --rewrite-ssc on train2 (rebuilds SSC from
#      optimal_result.json only; no simulation)
#   2. peer_affinity_alpha_prescan.py --alpha 2.0 with a set-aside dir, then an ASSERTION
#      that none of the 516 T1b parents was moved (a moved parent would rewrite the corpus
#      under every psv3 checkpoint; the job fails loud and names it)
#   3. cache -> graphs_cache_peer_only_v1_1670_psv3
#   4. near-RTT sidecar
#   5. B0
#
#   sbatch scripts_cosim/datalab/peer_only_v1_corpus.sbatch
#
#SBATCH --job-name=po-corpus
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/po-corpus-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/po-corpus-%j.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=12:00:00

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

TRAIN=simulation_data/gnn_datasets_peer_affinity_v1_c3_x200_train
TRAIN2=simulation_data/gnn_datasets_peer_affinity_v1_c3_x200_train2
TEST=simulation_data/gnn_datasets_peer_affinity_v1_c3_x200_r2
PARENTS=experiments/drainable_objective_v1_t1b_parents.json
SMALL=simulation_data/graphs_cache_drainable_objective_v1_v1_psv3
CACHE=simulation_data/graphs_cache_peer_only_v1_1670_psv3
SETASIDE=simulation_data/gnn_datasets_peer_affinity_v1_c3_x200_train2_setaside_peer_only_v1
SPLIT=experiments/peer_affinity_v1_t1b_split.json
DRAIN="${DRAIN_TABLE:-simulation_data/drainable_objective_v1/drain_table.json}"
RATE="${RATE:-0.46}"
for p in "$TRAIN" "$TRAIN2" "$TEST" "$PARENTS" "$SMALL/metadata.json" "$SPLIT" "$DRAIN"; do
  [[ -e "$p" ]] || { echo "FAIL LOUD: missing $p" >&2; exit 1; }
done
echo "[corpus] train2 before: $(ls -d $TRAIN2/ds_* | wc -l) datasets, $(ls $TRAIN2/ds_*/system_state_captured_unique.json 2>/dev/null | wc -l) with SSC"

echo "=== [1/5] SSC rewrite on train2 (from optimal_result.json; no simulation)"
python3 scripts_cosim/refresh_optimal_full_stats.py --base-dir "$TRAIN2" --rewrite-ssc | tail -3
n_ssc=$(ls $TRAIN2/ds_*/system_state_captured_unique.json 2>/dev/null | wc -l)
n_ds=$(ls -d $TRAIN2/ds_* | wc -l)
echo "[corpus] train2 after: $n_ds datasets, $n_ssc with SSC"
[[ "$n_ssc" == "$n_ds" ]] || { echo "FAIL LOUD: $((n_ds - n_ssc)) train2 datasets still lack SSC after --rewrite-ssc" >&2; exit 1; }
# train and r2 already carry theirs (136/136, 34/34) -- asserted, not assumed
[[ "$(ls $TRAIN/ds_*/system_state_captured_unique.json | wc -l)" == "$(ls -d $TRAIN/ds_* | wc -l)" ]] || { echo "FAIL LOUD: train lacks SSC" >&2; exit 1; }
[[ "$(ls $TEST/ds_*/system_state_captured_unique.json | wc -l)" == "$(ls -d $TEST/ds_* | wc -l)" ]] || { echo "FAIL LOUD: r2 lacks SSC" >&2; exit 1; }

echo "=== [2/5] alpha pre-scan (tightest rung 2.0) on train2 only; train and r2 were scanned by T1b"
python3 scripts_cosim/peer_affinity_alpha_prescan.py "$TRAIN2" --alpha 2.0 --set-aside-dir "$SETASIDE" | tail -8
python3 - "$PARENTS" "$TRAIN" "$TRAIN2" "$TEST" <<'PY'
import json, os, sys
p = json.load(open(sys.argv[1]))
ids = [k for k in (p.get("weights") or {}).keys()] if isinstance(p.get("weights"), dict) else [k for k in p if not k.startswith("_")]
missing = [i for i in ids if not os.path.isdir(os.path.join("simulation_data", i))]
if missing:
    raise SystemExit(f"FAIL LOUD: the pre-scan moved {len(missing)} of the 516 T1b parents, e.g. {missing[:3]}; "
                     "restore them from the set-aside dir before anything else runs")
print(f"[corpus] all {len(ids)} T1b parents still in place")
PY
echo "[corpus] train2 now: $(ls -d $TRAIN2/ds_* | wc -l) datasets; set aside: $(find $SETASIDE -maxdepth 2 -name 'ds_*' 2>/dev/null | wc -l)"

echo "=== [3/5] cache -> $CACHE"
rm -rf "$CACHE"
python3 -u src/notebooks/prepare_graphs_cache.py --base-dirs "$TRAIN" "$TRAIN2" "$TEST" --cache-dir "$CACHE" \
  --dag-partial-state --platform-feature-dim 14 \
  --label-objective "rtt_drift:1" \
  --label-arrival-rate "$RATE" \
  --label-drain-table "$DRAIN" 2>&1 | grep -v "it/s\]" | tail -40
python3 - "$CACHE" "$SMALL" <<'PY'
import json, sys
m = json.load(open(sys.argv[1] + "/metadata.json")); s = json.load(open(sys.argv[2] + "/metadata.json"))
for k in ("partial_state_contract", "label_objective", "dag_primary_alpha_key", "platform_feature_dim", "peer_exchange_block"):
    if m.get(k) != s.get(k):
        raise SystemExit(f"FAIL LOUD: big cache {k}={m.get(k)!r} but psv3 cache has {s.get(k)!r}")
n = int(m.get("num_datasets") or 0)
if n < 1500:
    raise SystemExit(f"FAIL LOUD: big cache holds {n} datasets, B0 needs >= 1500")
print("[cache] contract =", m["partial_state_contract"], "| label =", m["label_objective"], "| datasets =", n,
      "| rtt rows =", m.get("num_rtt_combo_rows"))
PY

echo "=== [4/5] near-RTT sidecar"
python3 -u src/notebooks/prepare_graphs_cache_near_rtt.py --cache-dir "$CACHE" --skip-cache-build 2>&1 | grep -v "it/s\]" | tail -4
[[ -f "$CACHE/valid_combos_near_rtt_capped.pkl" ]] || { echo "FAIL LOUD: no near-RTT sidecar" >&2; exit 1; }

echo "=== [5/5] B0"
mkdir -p simulation_data/peer_only_v1
python3 -u scripts_cosim/peer_only_v1_b0.py --big-cache "$CACHE" --small-cache "$SMALL" --split "$SPLIT" \
  --out simulation_data/peer_only_v1/b0.json
du -sh "$CACHE"
echo "=== corpus done ==="
