#!/bin/bash
# drainable_objective_v1 -- Phase B: one cache per V over the SAME live-clock corpus.
# Recipe = T1b's build, as peer_affinity_warm_v1_w1_cache.sbatch records it; the only
# addition is --label-objective, which changes what the targets mean and nothing else.
#
#   array task 0 -> V = 0     (the CONTROL: live clock, one-step label)
#              1 -> V = 0.5
#              2 -> V = 1     (the treatment the live gate reads)
#              3 -> V = 2
#
# Four caches over one corpus is the point: the corpus, the split seeds, the features and
# the decoder are byte-identical across the four, so a contrast between two of them is a
# contrast about the label and nothing else.
#
#   sbatch --array=0-3 scripts_cosim/datalab/drainable_objective_v1_cache.sbatch
#
#SBATCH --job-name=dobj-cache
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/dobj-cache-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/dobj-cache-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --array=0-3

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

VS=(0 0.5 1 2)
TAGS=(v0 v05 v1 v2)
T=${SLURM_ARRAY_TASK_ID}
V=${VS[$T]}
TAG=${TAGS[$T]}

TRAIN=simulation_data/gnn_datasets_peer_affinity_v1_c3_x200_lc
TEST=simulation_data/gnn_datasets_peer_affinity_v1_c3_x200_lc_r2
CACHE=simulation_data/graphs_cache_drainable_objective_v1_${TAG}
SPLIT=experiments/drainable_objective_v1_${TAG}_split.json
DRAIN="${DRAIN_TABLE:-simulation_data/drainable_objective_v1/drain_table.json}"
# The label's lambda: the gate rung's realised arrival rate (drainable_f4000_n50000.json,
# 0.46 tasks/s). A property of the deployment the label is for, stated here and recorded
# in the cache's own metadata by the builder.
RATE="${RATE:-0.46}"

echo "[TASK $T] V=$V tag=$TAG cache=$CACHE"
[[ -f "$DRAIN" ]] || { echo "FAIL LOUD: missing $DRAIN -- run A1 first; a live-clock corpus labelled on the exec+comm clock is the null read this lineage already made once" >&2; exit 1; }
echo "[CLOCK] $DRAIN ($(md5sum "$DRAIN" | cut -c1-12))  RATE=$RATE"
[[ -f "$SPLIT" ]] && { echo "FAIL LOUD: $SPLIT exists -- a split is frozen once runs depend on it" >&2; exit 1; }

for d in "$TRAIN" "$TEST"; do
  [[ -d "$d" ]] || { echo "FAIL LOUD: corpus missing: $d" >&2; exit 1; }
  # Complete = placement_metadata.json says sweep_complete AND rows == plans. A non-empty
  # placements.jsonl is not evidence (peer_affinity_warm_v1 W0 attempt 1).
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
  [[ "$bad" == "0" ]] || { echo "FAIL LOUD: $bad dataset dirs under $d are incomplete" >&2; exit 1; }
done

# Steps 1 and 2 touch the corpus, not the cache, so only one array task may do them.
# Task 0 runs them; the others wait for the marker rather than racing (datalab pitfall #7).
MARK=simulation_data/drainable_objective_v1/.lc_corpus_prepared
if [[ $T -eq 0 ]]; then
  echo "=== [1/5] SSC rewrite"
  python3 scripts_cosim/refresh_optimal_full_stats.py --base-dir "$TRAIN" --rewrite-ssc | tail -2
  python3 scripts_cosim/refresh_optimal_full_stats.py --base-dir "$TEST" --rewrite-ssc | tail -2
  for d in "$TRAIN" "$TEST"; do
    [[ $(ls "$d"/ds_*/system_state_captured_unique.json | wc -l) -eq $(ls -d "$d"/ds_* | wc -l) ]] || { echo "FAIL LOUD: SSC missing under $d" >&2; exit 1; }
  done
  echo "=== [2/5] alpha pre-scan (tightest rung 2.0)"
  python3 scripts_cosim/peer_affinity_alpha_prescan.py "$TRAIN" "$TEST" --alpha 2.0 \
    --set-aside-dir simulation_data/gnn_datasets_peer_affinity_v1_c3_x200_lc_setaside | tail -12
  mkdir -p "$(dirname "$MARK")" && touch "$MARK"
else
  for _ in $(seq 1 240); do [[ -f "$MARK" ]] && break; sleep 30; done
  [[ -f "$MARK" ]] || { echo "FAIL LOUD: task 0 never produced $MARK after 2 h" >&2; exit 1; }
fi

echo "=== [3/5] cache -> $CACHE (label V=$V)"
rm -rf "$CACHE"
python3 -u src/notebooks/prepare_graphs_cache.py --base-dirs "$TRAIN" "$TEST" --cache-dir "$CACHE" \
  --dag-partial-state --platform-feature-dim 14 \
  --label-objective "$( [[ "$V" == "0" ]] && echo rtt || echo "rtt_drift:$V" )" \
  --label-arrival-rate "$RATE" \
  --label-drain-table "$DRAIN" 2>&1 | grep -v "it/s\]" | tail -40

python3 - "$CACHE" "$V" <<'PY'
import json, sys
m = json.load(open(sys.argv[1] + "/metadata.json")); v = sys.argv[2]
need = {"partial_state_contract": "partial_state_v2", "dag_primary_alpha_key": "2.5",
        "peer_exchange_block": True, "platform_feature_dim": 14}
for k, val in need.items():
    assert m.get(k) == val, f"FAIL LOUD: cache {k}={m.get(k)!r} != {val!r}"
got = m.get("label_objective")
want_prefix = "rtt" if v == "0" else f"rtt_drift:{float(v):g}"
assert got and got.startswith(want_prefix), f"FAIL LOUD: cache label_objective={got!r}, expected {want_prefix}*"
if v != "0":
    assert "clock=measured" in got, f"FAIL LOUD: cache was labelled on the exec+comm clock: {got!r}"
print("[cache] contract ok, label =", got, "| datasets =", m.get("num_datasets"),
      "| rtt rows =", m.get("num_rtt_combo_rows"))
PY

echo "=== [4/5] near-RTT sidecar"
python3 -u src/notebooks/prepare_graphs_cache_near_rtt.py --cache-dir "$CACHE" --skip-cache-build 2>&1 | grep -v "it/s\]" | tail -4
[[ -f "$CACHE/valid_combos_near_rtt_capped.pkl" ]] || { echo "FAIL LOUD: no near-RTT sidecar" >&2; exit 1; }

echo "=== [5/5] split artifact"
python3 scripts_cosim/make_peer_affinity_t1_split.py --cache-dir "$CACHE" \
  --heldout-corpus "gnn_datasets_peer_affinity_v1_c3_x200_lc_r2" --output "$SPLIT"
sha256sum "$SPLIT"
du -sh "$CACHE"
echo "=== V=$V cache done: rsync $SPLIT back and commit before training ==="
