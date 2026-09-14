#!/bin/bash
# drainable_objective_v1 -- Phase B: one cache per V over the SAME live-clock corpus.
# Recipe = T1b's build, as peer_affinity_warm_v1_w1_cache.sbatch records it; the only
# addition is --label-objective, which changes what the targets mean and nothing else.
#
# Amendment 1 (2026-09-14): one cache, over the EXISTING T1b corpus and split.
#
# The registered design regenerated a corpus on the measured backlog clock and built four
# caches over it. Measured before any of that ran: repricing the backlog by A1's 784x moves
# the median sweep RTT ~1 % and leaves the ARGMIN PLAN -- the training target -- unchanged,
# because the best plan avoids backlog under either clock. So the live-clock corpus IS the
# corpus, and V = 0 is the T1b cache that already exists with checkpoints already gated at
# this cell. Only the V = 1 cache is new.
#
# The contrast is then same corpus, same split, same seeds, same lr, only the label.
#
#   array task 0 -> V = 1     (the treatment)
#              1 -> V = 0.5   (offline dose-response only)
#              2 -> V = 2     (offline dose-response only)
#
#   sbatch --array=0-2 scripts_cosim/datalab/drainable_objective_v1_cache.sbatch
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
#SBATCH --array=0-2

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

VS=(1 0.5 2)
TAGS=(v1 v05 v2)
T=${SLURM_ARRAY_TASK_ID}
V=${VS[$T]}
TAG=${TAGS[$T]}

# The T1b corpus, unchanged. Its own split is REUSED verbatim, so V = 1 and the T1b
# control share train/val/test parent-for-parent -- a split that moved between the arms
# would confound the one contrast this lineage exists to make.
TRAIN=simulation_data/gnn_datasets_peer_affinity_v1_c3_x200_train
TRAIN2=simulation_data/gnn_datasets_peer_affinity_v1_c3_x200_train2
TEST=simulation_data/gnn_datasets_peer_affinity_v1_c3_x200_r2
# `_train2` was EXTENDED IN PLACE from 346 to 1500 datasets by a later generation job, so
# "the T1b corpus" cannot be reconstructed from directory contents any more. This manifest
# is the exact 516 parents T1b trained on, derived from its split; every weight is 1, so
# --oversample-manifest acts as a pure filter. Without it this cache would silently be a
# 1,670-dataset corpus and the contrast against T1b would be about corpus size, which is
# the single largest measured lever in this program.
PARENTS=experiments/drainable_objective_v1_t1b_parents.json
CACHE=simulation_data/graphs_cache_drainable_objective_v1_${TAG}
SPLIT=experiments/peer_affinity_v1_t1b_split.json
DRAIN="${DRAIN_TABLE:-simulation_data/drainable_objective_v1/drain_table.json}"
# The label's lambda: the gate rung's realised arrival rate (drainable_f4000_n50000.json,
# 0.46 tasks/s). A property of the deployment the label is for, stated here and recorded
# in the cache's own metadata by the builder.
RATE="${RATE:-0.46}"

echo "[TASK $T] V=$V tag=$TAG cache=$CACHE"
[[ -f "$DRAIN" ]] || { echo "FAIL LOUD: missing $DRAIN -- run A1 first; a live-clock corpus labelled on the exec+comm clock is the null read this lineage already made once" >&2; exit 1; }
echo "[CLOCK] $DRAIN ($(md5sum "$DRAIN" | cut -c1-12))  RATE=$RATE"
[[ -f "$SPLIT" ]] || { echo "FAIL LOUD: $SPLIT missing -- V = 1 must reuse T1b's split verbatim" >&2; exit 1; }
[[ -f "$PARENTS" ]] || { echo "FAIL LOUD: $PARENTS missing -- without it this cache is a different corpus from T1b's" >&2; exit 1; }

for d in "$TRAIN" "$TRAIN2" "$TEST"; do
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

# Steps 1 and 2 of the T1b recipe (SSC rewrite, alpha pre-scan) are NOT re-run: this IS the
# T1b corpus and they were run on it when T1b was built. Re-running them would rewrite the
# corpus underneath the T1b checkpoints that are this lineage's control arm.
echo "=== [1-2/5] skipped: T1b corpus already prepared (SSC + alpha pre-scan)"

echo "=== [3/5] cache -> $CACHE (label V=$V)"
rm -rf "$CACHE"
python3 -u src/notebooks/prepare_graphs_cache.py --base-dirs "$TRAIN" "$TRAIN2" "$TEST" --cache-dir "$CACHE" \
  --dag-partial-state --platform-feature-dim 14 \
  --oversample-manifest "$PARENTS" \
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
n = m.get("num_datasets")
if n != 516:
    raise SystemExit(f"FAIL LOUD: cache holds {n} datasets, T1b's corpus is 516 -- the "
                     "parent filter did not apply and this contrast would be about corpus "
                     "size, not about the label")
print("[cache] contract ok, label =", got, "| datasets =", n,
      "| rtt rows =", m.get("num_rtt_combo_rows"))
PY

echo "=== [4/5] near-RTT sidecar"
python3 -u src/notebooks/prepare_graphs_cache_near_rtt.py --cache-dir "$CACHE" --skip-cache-build 2>&1 | grep -v "it/s\]" | tail -4
[[ -f "$CACHE/valid_combos_near_rtt_capped.pkl" ]] || { echo "FAIL LOUD: no near-RTT sidecar" >&2; exit 1; }

echo "=== [5/5] split: REUSED from T1b, asserted to cover this cache"
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
print("[split] reused T1b split covers this cache: %d datasets, %d/%d/%d train/val/test"
      % (len(ids), len(sp.get("train") or []), len(sp.get("val") or []), len(sp.get("test") or [])))
SPLITCHK
sha256sum "$SPLIT"
du -sh "$CACHE"
echo "=== V=$V cache done ==="
