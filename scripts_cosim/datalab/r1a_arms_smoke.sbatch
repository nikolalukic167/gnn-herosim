#!/bin/bash
# r1_attribution_v1: smoke of the seven learned-arm configs on S6's smoke corpus (workload_fix_v1/smoke_hardening, 2a7c85c9 lineage).
#  1. copy the smoke datasets, rewrite SSC, build the v5 cache under the R1 environment, make a split artifact for it;
#  2. for every arm: train the _smoke config at seed 1 TWICE through run_experiment.py and require bit-identical weights;
#  3. sidecheck each sidecar, load each checkpoint through the serving loader (declared pruning on) and decode three batches.
# Reads the smoke corpus; writes only under OUT, the pinned worktree's simulation_data/<smoke cache> and experiments/<smoke split>.
#   sbatch --export=ALL,WT=/home/nikola.lukic/gnn-herosim-wt/<pinned>,OUT=<new dir> r1a_arms_smoke.sbatch
#SBATCH --job-name=r1a-arms
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/r1a-arms-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
set -euo pipefail
ulimit -n 65536 2>/dev/null || true
WT=${WT:?FAIL LOUD: set WT (pinned worktree)}
OUT=${OUT:?FAIL LOUD: set OUT}
SRC=${SRC:-/home/nikola.lukic/gnn-herosim/simulation_data/workload_fix_v1/smoke_hardening}
ARMS=${ARMS:-"gnn_eng twin_eng mlp_same gnn_raw twin_raw gnn_eng_physmp set_transformer"}
cd "$WT"
for v in $(compgen -e | grep -E '^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|NEAR_RTT_|PARTIAL_STATE)'); do unset "$v"; done
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT:$WT/src/notebooks" PYTHONUNBUFFERED=1 WANDB_ENTITY=nikolalukic167-tu-wien
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=""
python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"
echo "[CODE] $(git describe --dirty --always) at $WT"
[[ -z "$(git status --porcelain -- src scripts_cosim experiments run_experiment.py)" ]] || { echo "FAIL LOUD: worktree dirty" >&2; exit 1; }
[[ -L models ]] || { echo "FAIL LOUD: models/ must be the symlink into ~/gnn-herosim/models" >&2; exit 1; }
[[ ! -e "$OUT" ]] || { echo "FAIL LOUD: $OUT exists"; exit 1; }
CACHE=simulation_data/graphs_cache_r1_attribution_v1_smoke_psv5
SPLIT=experiments/r1_attribution_v1_smoke_split.json
[[ ! -e "$CACHE" && ! -e "$SPLIT" ]] || { echo "FAIL LOUD: $CACHE or $SPLIT exists in the worktree"; exit 1; }
mkdir -p "$OUT/corpus" "$OUT/run1" "$OUT/run2"
export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa HEROSIM_SHARED_AUTOSCALER=0 GATE_FIXED_POLICY_TIME_SCALE=1.0
export HEROSIM_PEER_EXCHANGE=1 HEROSIM_SERVER_ONLY_REPLICAS=1 HEROSIM_WARMTH_PHYSICS=node_disk_v2 PYTHONHASHSEED=0

echo "=== [1/3] corpus copy, SSC rewrite, v5 cache, split"
for d in "$SRC"/gnn_datasets_wf1_train "$SRC"/gnn_datasets_wf1_heldout; do cp -r "$d"/ds_* "$OUT/corpus/"; done
echo "datasets: $(ls -d "$OUT"/corpus/ds_* | wc -l)"
export DAG_PRIMARY_ALPHA_KEY=inf PARTIAL_STATE_CONTRACT=partial_state_v5 NEAR_RTT_LABEL_BACKLOG_CLOCK=seeded
python3 scripts_cosim/refresh_optimal_full_stats.py --base-dir "$OUT/corpus" --rewrite-ssc | tail -1
python3 -u src/notebooks/prepare_graphs_cache.py --base-dirs "$OUT/corpus" --cache-dir "$CACHE" --dag-partial-state --platform-feature-dim 14 2>&1 | grep -v "it/s\]" | tail -3
python3 scripts_cosim/make_split_artifact.py --cache-dir "$CACHE" --output "$SPLIT"
python3 - "$CACHE" <<'PY'
import json, pickle, sys
m = json.load(open(sys.argv[1] + "/metadata.json")); g = pickle.load(open(sys.argv[1] + "/graphs.pkl", "rb"))
print("cache:", len(g), "graphs; contract", m.get("partial_state_contract"), "slate", m.get("candidate_slate"), "physics", m.get("physics_env"))
assert m.get("partial_state_contract") == "partial_state_v5" and m.get("candidate_slate") == "declared_pruning_v1", "FAIL LOUD: cache is not the v5 declared-pruning cache"
PY
unset PARTIAL_STATE_CONTRACT   # the configs set what the trainer needs

echo "=== [2/3] train each _smoke config twice at seed 1; weights must be bit-identical"
for ARM in $ARMS; do
  CFG=experiments/r1_attribution_v1_${ARM}_smoke.yaml
  RUN=$(python3 -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['wandb']['run_name'])" "$CFG")-seed1
  for X in 1 2; do
    rm -f "models/$RUN.pt" "models/$RUN.contract.json" "models/$RUN-final.pt" "models/$RUN-final.contract.json"
    python3 -u run_experiment.py "$CFG" --seed 1 > "$OUT/run$X/$ARM.log" 2>&1 || { tail -30 "$OUT/run$X/$ARM.log"; echo "FAIL LOUD: $ARM run $X failed"; exit 1; }
    for f in "$RUN.pt" "$RUN.contract.json" "$RUN-final.pt" "$RUN-final.contract.json"; do
      [[ -f "models/$f" ]] || { echo "FAIL LOUD: $ARM run $X produced no models/$f"; exit 1; }
      cp "models/$f" "$OUT/run$X/${ARM}__$f"
    done
  done
  python3 - "$OUT" "$ARM" "$RUN" <<'PY'
import json, sys, torch
out, arm, run = sys.argv[1:4]
for suffix in ("", "-final"):
    a = torch.load(f"{out}/run1/{arm}__{run}{suffix}.pt", map_location="cpu"); b = torch.load(f"{out}/run2/{arm}__{run}{suffix}.pt", map_location="cpu")
    assert set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a), f"FAIL LOUD: {arm}{suffix}: two runs at one seed differ"
sa = json.load(open(f"{out}/run1/{arm}__{run}.contract.json")); sb = json.load(open(f"{out}/run2/{arm}__{run}.contract.json"))
for k in ("cache_dir", "train_seed", "arm_kind", "split_artifact"):
    assert sa.get(k) == sb.get(k), f"FAIL LOUD: {arm}: sidecar {k} differs between runs"
print(f"[determinism] {arm}: {len(a)} tensors bit-identical across two runs (best and final); params {sum(v.numel() for v in a.values()):,}")
PY
done

echo "=== [3/3] sidecheck, serving loader, decode"
export GNN_SERVE_CANDIDATE_SLATE=declared_pruning_v1 GNN_DECODE_MODE=masked_topo GNN_PREFIX_ALPHA_KEY=inf
for ARM in $ARMS; do
  RUN=$(python3 -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['wandb']['run_name'])" experiments/r1_attribution_v1_${ARM}_smoke.yaml)-seed1
  python3 scripts_cosim/joint_burst_v2_sidecheck.py "models/$RUN.contract.json" "ra_$ARM" "$SPLIT" inf
  python3 - "models/$RUN.pt" "$CACHE" "$ARM" <<'PY'
import os, pickle, sys
import torch
from pathlib import Path
from src.policy.gnn.prefix_serving import decode_prefix_conditioned, load_prefix_conditioned_gnn
ck, cache, arm = sys.argv[1:4]
model, options, sidecar = load_prefix_conditioned_gnn(Path(ck), device=torch.device("cpu"))
assert sidecar["arm_kind"] == {"mlp_same": "mlp_same", "set_transformer": "set_transformer"}.get(arm, "gnn")
graphs = pickle.load(open(cache + "/graphs.pkl", "rb"))[:3]
plans = [decode_prefix_conditioned(model, g, options) for g in graphs]
assert all(len(p) == int(g.n_tasks) for p, g in zip(plans, graphs))
print(f"[serve] {arm}: loaded as {type(model).__name__}, decoded {len(plans)} batches, plan sizes {[len(p) for p in plans]}")
PY
done
echo "=== all ${ARMS// /,} passed; outputs in $OUT"
