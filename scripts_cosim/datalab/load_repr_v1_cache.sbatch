#!/bin/bash
# load_repr_v1 -- the jb2 cache rebuilt under partial_state_v4 (docs/lineages/load_repr_v1.md).
# joint_burst_v2_cache.sbatch with the contract swapped. The corpus is not touched (its SSC rewrite
# ran for jb2) and the frozen jb2 split is checked, never rebuilt. Runs in the worktree
# ~/gnn-herosim-wt/load_repr; simulation_data/* and models are symlinks into ~/gnn-herosim.
#SBATCH --job-name=lr1-cache
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/lr1-cache-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/lr1-cache-%j.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00

set -euo pipefail
WT=/home/nikola.lukic/gnn-herosim-wt/load_repr
cd "$WT"
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$WT:$WT/src/notebooks"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export DAG_PRIMARY_ALPHA_KEY=inf PARTIAL_STATE_CONTRACT=partial_state_v4
echo "[CODE] $(git describe --dirty --always) at $WT"
[[ -z "$(git status --porcelain -- src scripts_cosim experiments run_experiment.py)" ]] || { echo "FAIL LOUD: worktree dirty" >&2; exit 1; }
TRAIN=simulation_data/gnn_datasets_joint_burst_v2_train
TEST=simulation_data/gnn_datasets_joint_burst_v2_heldout
CACHE=simulation_data/graphs_cache_joint_burst_v2_psv4_inf
SPLIT=experiments/joint_burst_v2_split.json
DRAIN=simulation_data/drainable_objective_v1/drain_table.json
RATE=0.46
[[ -L "$CACHE" ]] || { echo "FAIL LOUD: $CACHE must be the symlink into ~/gnn-herosim/simulation_data" >&2; exit 1; }
[[ "$(sha256sum "$SPLIT" | cut -d' ' -f1)" == e54f33645bbdcf48e93ce58056b4d0bd6d2bccd4972a37d52aef30411d921d1d ]] || { echo "FAIL LOUD: split sha mismatch" >&2; exit 1; }
[[ -f "$DRAIN" ]] || { echo "FAIL LOUD: drain table $DRAIN missing" >&2; exit 1; }
[[ -z "$(ls -A "$CACHE")" ]] || { echo "FAIL LOUD: $CACHE is not empty; refusing to overwrite" >&2; exit 1; }
echo "=== [1/3] cache -> $CACHE"
python3 -u src/notebooks/prepare_graphs_cache.py --base-dirs "$TRAIN" "$TEST" --cache-dir "$CACHE/" \
  --dag-partial-state --platform-feature-dim 14 \
  --label-objective "rtt_drift:1" --label-arrival-rate "$RATE" --label-drain-table "$DRAIN" 2>&1 | grep -v "it/s\]" | tail -30
python3 - "$CACHE" <<'PY'
import json, pickle, sys
m = json.load(open(sys.argv[1] + "/metadata.json"))
for k, v in (("partial_state_contract", "partial_state_v4"), ("dag_primary_alpha_key", "inf"), ("peer_exchange_block", True), ("platform_feature_dim", 14)):
    assert m.get(k) == v, f"FAIL LOUD: cache {k}={m.get(k)!r} != {v!r}"
got = m.get("label_objective") or "rtt"
assert got.startswith("rtt_drift:1") and "clock=measured" in got, f"FAIL LOUD: label_objective={got!r}"
graphs = pickle.load(open(sys.argv[1] + "/graphs.pkl", "rb"))
n = nz = 0
for g in graphs:
    psc = g.partial_state_ctx
    assert psc.get("backlog_s") and psc.get("service_s"), "FAIL LOUD: a graph lacks the v4 load block"
    n += len(psc["backlog_s"]); nz += sum(1 for v in psc["backlog_s"].values() if v > 0)
print("[cache] v4 ok, label =", got, "| graphs =", len(graphs), "| replicas", n, "with backlog > 0:", nz)
PY
echo "=== [2/3] near-RTT sidecar"
python3 -u src/notebooks/prepare_graphs_cache_near_rtt.py --cache-dir "$CACHE" --skip-cache-build 2>&1 | grep -v "it/s\]" | tail -4
[[ -f "$CACHE/valid_combos_near_rtt_capped.pkl" ]] || { echo "FAIL LOUD: no near-RTT sidecar"; exit 1; }
echo "=== [3/3] candidate-not-in-sweep guard"
python3 -u scripts_cosim/peer_only_v1_candidate_check.py --cache "$CACHE" --out simulation_data/load_repr_v1/candidate_check.json
# the frozen split's coverage of this cache is asserted by the trainer (assert_split_artifact_covers)
du -sh "$CACHE/"
echo "=== cache done ==="
