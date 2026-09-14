#!/bin/bash
# drainable_objective_v1 -- Phase B training. Same recipe as peer_affinity_v1's T1b GNN
# arms (that script is the template); the only differences are the cache, the split and
# the checkpoint names, which carry the label the arm was fitted to.
#
# 8 configs (V in {0, 0.5, 1, 2} x {gnn, mpoff}) with an ASYMMETRIC seed budget, as
# registered: 16 seeds at V = 0 and V = 1 (the gated arms), 8 at V = 0.5 and V = 2
# (offline dose-response only, never gated, never quoted live). 48 + 48 = 96 tasks:
#
#   tasks  0- 15  v0  gnn   seeds 1-16      tasks 48- 55  v05 gnn   seeds 1-8
#   tasks 16- 31  v0  mpoff seeds 1-16      tasks 56- 63  v05 mpoff seeds 1-8
#   tasks 32- 47  v1  gnn   seeds 1-16      tasks 64- 71  v2  gnn   seeds 1-8
#   tasks 72- 87  v1  mpoff seeds 1-16   -- see the table built below; the shell is the
#                                           single source of truth, this comment is a map.
#
# lr is 2e-3 for every arm and is NOT re-tuned: re-selecting it now would be choosing a
# hyperparameter with knowledge of the outcome. Carried as a stated limitation, the same
# way objective_pivot_v1 Phase 3 carried lr = 1e-4.
#
#   sbatch scripts_cosim/datalab/drainable_objective_v1_train.sbatch
#
#SBATCH --job-name=dobj-train
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/dobj-train-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/dobj-train-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-95

set -euo pipefail
# datalab-pitfalls #11: without this every GNN task dies in the first epochs with
# `RuntimeError: received 0 items of ancdata`.
ulimit -n 65536 2>/dev/null || true
cd /home/nikola.lukic/gnn-herosim

if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export WANDB_ENTITY=nikolalukic167-tu-wien
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=""

python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__,'cuda',torch.cuda.is_available())"
echo "[CODE] $(git describe --dirty --always) on $(git rev-parse --abbrev-ref HEAD)"

# (tag, arm, n_seeds) in task order. Built into a flat task table so the mapping is
# computed once, here, rather than asserted in a comment that can drift from it.
SPECS=("v0 gnn 16" "v0 mpoff 16" "v1 gnn 16" "v1 mpoff 16"
       "v05 gnn 8" "v05 mpoff 8" "v2 gnn 8" "v2 mpoff 8")
TAGS=(); ARMS=(); SEEDS=()
for spec in "${SPECS[@]}"; do
  read -r tag arm n <<< "$spec"
  for s in $(seq 1 "$n"); do TAGS+=("$tag"); ARMS+=("$arm"); SEEDS+=("$s"); done
done
NTASKS=${#TAGS[@]}

T=${SLURM_ARRAY_TASK_ID}
if (( T >= NTASKS )); then echo "[TASK $T] beyond the $NTASKS declared arms, nothing to do"; exit 0; fi
TAG=${TAGS[$T]}; ARM=${ARMS[$T]}; SEED=${SEEDS[$T]}

CFG=experiments/drainable_objective_v1_${TAG}_${ARM}.yaml
CK="models/drainable-objective-v1-${TAG}-${ARM}-lr2e3-seed${SEED}.pt"
CACHE=simulation_data/graphs_cache_drainable_objective_v1_${TAG}
SPLIT=experiments/drainable_objective_v1_${TAG}_split.json
echo "[TASK $T/$NTASKS] label=$TAG arm=$ARM seed=$SEED cfg=$CFG -> $CK"

[[ -f "$CFG" ]] || { echo "FAIL LOUD: $CFG missing (git pull?)" >&2; exit 1; }
[[ -f "$CACHE/metadata.json" ]] || { echo "FAIL LOUD: $CACHE has no metadata.json" >&2; exit 1; }
[[ -f "$CACHE/valid_combos_near_rtt_capped.pkl" ]] || { echo "FAIL LOUD: $CACHE lacks valid_combos_near_rtt_capped.pkl" >&2; exit 1; }
[[ -f "$CACHE/rtt_chunk_0.pkl" ]] || { echo "FAIL LOUD: $CACHE lacks rtt_chunk_*.pkl -- NEAR_RTT_VAL_EXACT_REGRET needs the full sweep" >&2; exit 1; }
[[ -f "$SPLIT" ]] || { echo "FAIL LOUD: split artifact $SPLIT missing" >&2; exit 1; }

# The cache must carry the label this arm's name claims. An arm trained on the wrong cache
# would be indistinguishable afterwards from one trained on the right one -- the checkpoint
# name is not evidence, the cache's own metadata is.
python3 - "$CACHE" "$TAG" <<'EOF2'
import json, sys
m = json.load(open(sys.argv[1] + "/metadata.json")); tag = sys.argv[2]
for k, v in (("partial_state_contract", "partial_state_v2"), ("dag_primary_alpha_key", "2.5"),
             ("peer_exchange_block", True), ("platform_feature_dim", 14)):
    if m.get(k) != v:
        raise SystemExit(f"FAIL LOUD: cache metadata {k}={m.get(k)!r}, this recipe requires {v!r}")
got = m.get("label_objective") or "rtt"
want = {"v0": "rtt", "v05": "rtt_drift:0.5", "v1": "rtt_drift:1", "v2": "rtt_drift:2"}[tag]
if not got.startswith(want):
    raise SystemExit(f"FAIL LOUD: cache label_objective={got!r} but this arm is {tag} ({want})")
if tag != "v0" and "clock=measured" not in got:
    raise SystemExit(f"FAIL LOUD: cache was labelled on the exec+comm clock: {got!r}")
print("[cache] contract ok, label =", got)
EOF2
mkdir -p logs models

if [[ -f "$CK" ]]; then echo "[skip] $CK exists"; exit 0; fi

python3 -u run_experiment.py "$CFG" --seed "$SEED"

# The trainer names its checkpoint from wandb.run_name + -seedN; rename to this script's
# convention only if they differ, and fail loud if nothing was written at all.
PRODUCED="models/drainable-objective-v1-${TAG}-${ARM}-seed${SEED}.pt"
if [[ -f "$PRODUCED" && ! -f "$CK" ]]; then
  mv "$PRODUCED" "$CK"
  [[ -f "${PRODUCED%.pt}.contract.json" ]] && mv "${PRODUCED%.pt}.contract.json" "${CK%.pt}.contract.json"
fi
[[ -f "$CK" ]] || { echo "FAIL LOUD: $CFG seed $SEED produced no $CK (looked for $PRODUCED too)" >&2; exit 1; }
SC="${CK%.pt}.contract.json"
[[ -f "$SC" ]] || { echo "FAIL LOUD: no sidecar $SC" >&2; exit 1; }
python3 - "$SC" "$ARM" "$SPLIT" "$TAG" <<'EOF2'
import hashlib, json, sys
sc, arm, split, tag = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3], sys.argv[4]
checks = {"disable_message_passing": arm == "mpoff", "mp_peer_edges": True,
          "partial_state_contract": "partial_state_v2", "dag_alpha_key": "2.5",
          "decode_replica_reuse": True, "decode_relax_on_stuck": True, "peer_mass": True}
for k, want in checks.items():
    if sc.get(k) != want:
        raise SystemExit(f"FAIL LOUD: sidecar {k}={sc.get(k)!r}, arm {arm} expects {want!r}")
# drainable_objective_v1: the label must reach the checkpoint, or a served arm cannot say
# what it was fitted to and the gate's arms become unattributable six weeks later.
got = sc.get("label_objective") or "rtt"
want = {"v0": "rtt", "v05": "rtt_drift:0.5", "v1": "rtt_drift:1", "v2": "rtt_drift:2"}[tag]
if not got.startswith(want):
    raise SystemExit(f"FAIL LOUD: sidecar label_objective={got!r} but this arm is {tag} ({want})")
sa = sc.get("split_artifact") or {}
sha = hashlib.sha256(open(split, "rb").read()).hexdigest()
if sa.get("sha256") != sha:
    raise SystemExit(f"FAIL LOUD: sidecar split sha {sa.get('sha256')} != {split} {sha}")
print("[sidecar] ok, label =", got)
EOF2
echo "=== task $T done: $CK ==="
