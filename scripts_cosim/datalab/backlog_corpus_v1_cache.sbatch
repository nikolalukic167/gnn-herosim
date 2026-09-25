#!/bin/bash
# backlog_corpus_v1 -- the cache on the synthetic-backlog corpus (docs/lineages/backlog_corpus_v1.md).
# The load_repr_v1 cache recipe (partial_state_v4, alpha inf, rtt_drift:1 at 0.46/s on the measured
# drain table) with the label's backlog term on the SEEDED clock, so B_p is the backlog the co-sim
# replayed. Then the near-RTT sidecar, the candidate guard, a cell-grouped split, and the audit
# against the joint_burst_v2 twins.
#SBATCH --job-name=bc1-cache
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/bc1-cache-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/bc1-cache-%j.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=10:00:00

set -euo pipefail
WT=/home/nikola.lukic/gnn-herosim-wt/backlog_corpus
cd "$WT"
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$WT:$WT/src/notebooks"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export DAG_PRIMARY_ALPHA_KEY=inf PARTIAL_STATE_CONTRACT=partial_state_v4
export NEAR_RTT_LABEL_BACKLOG_CLOCK=seeded
echo "[CODE] $(git describe --dirty --always) at $WT"
[[ -z "$(git status --porcelain -- src scripts_cosim | grep -v '^??' || true)" ]] || { echo "FAIL LOUD: worktree dirty" >&2; exit 1; }
TRAIN=simulation_data/gnn_datasets_backlog_corpus_v1_train
TEST=simulation_data/gnn_datasets_backlog_corpus_v1_heldout
CACHE=simulation_data/graphs_cache_backlog_corpus_v1_psv4_inf
SPLIT=experiments/backlog_corpus_v1_split.json
DRAIN=simulation_data/drainable_objective_v1/drain_table.json
RATE=0.46
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
echo "=== [2/6] cache -> $CACHE"
rm -rf "$CACHE"
python3 -u src/notebooks/prepare_graphs_cache.py --base-dirs "$TRAIN" "$TEST" --cache-dir "$CACHE" \
  --dag-partial-state --platform-feature-dim 14 \
  --label-objective "rtt_drift:1" --label-arrival-rate "$RATE" --label-drain-table "$DRAIN" 2>&1 | grep -v "it/s\]" | tail -30
python3 - "$CACHE" <<'PY'
import json, pickle, sys
m = json.load(open(sys.argv[1] + "/metadata.json"))
for k, v in (("partial_state_contract", "partial_state_v4"), ("dag_primary_alpha_key", "inf"), ("peer_exchange_block", True), ("platform_feature_dim", 14)):
    assert m.get(k) == v, f"FAIL LOUD: cache {k}={m.get(k)!r} != {v!r}"
got = m.get("label_objective") or "rtt"
assert got.startswith("rtt_drift:1") and "clock=measured" in got and got.endswith(",backlog=seeded"), f"FAIL LOUD: label_objective={got!r}"
graphs = pickle.load(open(sys.argv[1] + "/graphs.pkl", "rb"))
n = nz = 0
for g in graphs:
    psc = g.partial_state_ctx
    assert psc.get("backlog_s") and psc.get("service_s"), "FAIL LOUD: a graph lacks the v4 load block"
    n += len(psc["backlog_s"]); nz += sum(1 for v in psc["backlog_s"].values() if v > 0)
print("[cache] v4 ok, label =", got, "| graphs =", len(graphs), f"| candidate replicas {n}, backlog > 0 on {nz} ({nz / max(1, n):.1%})")
assert nz / max(1, n) > 0.2, "FAIL LOUD: the synthetic backlog did not reach the v4 columns"
PY
echo "=== [3/6] near-RTT sidecar"
python3 -u src/notebooks/prepare_graphs_cache_near_rtt.py --cache-dir "$CACHE" --skip-cache-build 2>&1 | grep -v "it/s\]" | tail -4
[[ -f "$CACHE/valid_combos_near_rtt_capped.pkl" ]] || { echo "FAIL LOUD: no near-RTT sidecar"; exit 1; }
echo "=== [4/6] candidate-not-in-sweep guard"
mkdir -p simulation_data/backlog_corpus_v1
python3 -u scripts_cosim/peer_only_v1_candidate_check.py --cache "$CACHE" --out simulation_data/backlog_corpus_v1/candidate_check.json
echo "=== [5/6] split artifact: held-out cells 9223/9224 = test, val = whole training cells"
python3 scripts_cosim/make_peer_affinity_t1_split.py --cache-dir "$CACHE" \
  --heldout-corpus "gnn_datasets_backlog_corpus_v1_heldout" --val-group-by cell --output "$SPLIT"
sha256sum "$SPLIT"
echo "=== [6/6] audit against the joint_burst_v2 twins"
python3 scripts_cosim/backlog_corpus_v1_audit.py --corpus "$TRAIN" "$TEST" \
  --twins simulation_data/gnn_datasets_joint_burst_v2_train simulation_data/gnn_datasets_joint_burst_v2_heldout \
  --out simulation_data/backlog_corpus_v1/audit.json
du -sh "$CACHE" "$TRAIN" "$TEST"
echo "=== cache done: rsync $SPLIT back and commit it before any training ==="
