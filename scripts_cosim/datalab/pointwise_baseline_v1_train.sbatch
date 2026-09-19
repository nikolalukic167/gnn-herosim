#!/bin/bash
# pointwise_baseline_v1 -- training the REAL pointwise model class.
# docs/lineages/pointwise_baseline_v1.md.
#
# Task table (SPECS below is the single source of truth; 16 seeds per (corpus, lr)):
#     0-15   516  lr5e4        48-63   1670 lr5e4
#    16-31   516  lr1e3        64-79   1670 lr1e3
#    32-47   516  lr2e3        80-95   1670 lr2e3
#
# The table carries ALL THREE learning rates at full width on purpose, so that task indices
# never move once the selection (J0) picks one. Only the blocks actually needed are submitted:
#
#   J0 selection (3 seeds per arm, 18 tasks, offline only -- no live arm runs):
#     sbatch --array=0-2,16-18,32-34,48-50,64-66,80-82 scripts_cosim/datalab/pointwise_baseline_v1_train.sbatch
#   J1 the SELECTED lr at full width, e.g. if lr1e3 wins:
#     sbatch --array=19-31 ... ; sbatch --array=67-79 ...
#
# A task whose checkpoint already exists exits 0, so re-submitting a block that overlaps the
# selection seeds is safe and costs nothing.
#
# WHY the lr is selected rather than inherited: `literature_reeval_v1` found a published
# method's reported tie was caused by the authors' own recipe under-training it. Handing this
# programme's baseline the GNN's learning rate untested would be the same error, in the
# direction that flatters the GNN. Selection is on HELD-OUT OFFLINE score only; the live gate
# only ever sees the winner.
#
#SBATCH --job-name=pb-train
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/pb-train-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/pb-train-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --array=0-95

set -euo pipefail
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
python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"
echo "[CODE] $(git describe --dirty --always) on $(git rev-parse --abbrev-ref HEAD)"

SPECS=("516 lr5e4" "516 lr1e3" "516 lr2e3" "1670 lr5e4" "1670 lr1e3" "1670 lr2e3")
SEEDS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
CORP=(); LRS=(); SD=()
for spec in "${SPECS[@]}"; do
  read -r corpus lr <<< "$spec"
  for s in "${SEEDS[@]}"; do CORP+=("$corpus"); LRS+=("$lr"); SD+=("$s"); done
done
NTASKS=${#CORP[@]}
T=${SLURM_ARRAY_TASK_ID}
if (( T >= NTASKS )); then echo "[TASK $T] beyond the $NTASKS declared arms"; exit 0; fi
CORPUS=${CORP[$T]}; LR=${LRS[$T]}; SEED=${SD[$T]}

if [[ $CORPUS == 516 ]]; then
  CACHE=simulation_data/graphs_cache_drainable_objective_v1_v1_psv3
  SPLIT=experiments/peer_affinity_v1_t1b_split.json
  WANT_N=516; WANT_SPLIT="386 96 34"
else
  CACHE=simulation_data/graphs_cache_peer_only_v1_1670_psv3
  SPLIT=experiments/peer_only_v1_1670_split.json
  WANT_N=1670; WANT_SPLIT=""     # asserted against the artifact's own totals below
fi
CFG=experiments/pointwise_baseline_v1_${CORPUS}_mlp_${LR}.yaml
CK="models/tabular/pointwise-baseline-v1-${CORPUS}-mlp-${LR}_seed${SEED}.pt"
echo "[TASK $T/$NTASKS] corpus=$CORPUS lr=$LR seed=$SEED cfg=$CFG -> $CK"

[[ -f "$CFG" ]] || { echo "FAIL LOUD: $CFG missing (git pull?)" >&2; exit 1; }
[[ -f "$CACHE/metadata.json" ]] || { echo "FAIL LOUD: $CACHE has no metadata.json" >&2; exit 1; }
[[ -f "$SPLIT" ]] || { echo "FAIL LOUD: split artifact $SPLIT missing" >&2; exit 1; }
python3 - "$CACHE" "$SPLIT" "$WANT_N" "$WANT_SPLIT" <<'EOF2'
import json, sys
cache, split, want_n, want_split = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
m = json.load(open(cache + "/metadata.json"))
for k, v in (("partial_state_contract", "partial_state_v3"), ("dag_primary_alpha_key", "2.5"),
             ("peer_exchange_block", True), ("platform_feature_dim", 14)):
    if m.get(k) != v:
        raise SystemExit(f"FAIL LOUD: cache metadata {k}={m.get(k)!r}, this recipe requires {v!r}")
# A corpus is its SPLIT, not its directory. This arm's entire purpose is to be corpus-matched
# to arms already on disk, so the size is asserted rather than assumed.
if int(m.get("num_datasets") or -1) != want_n:
    raise SystemExit(f"FAIL LOUD: cache declares num_datasets={m.get('num_datasets')!r}, "
                     f"this arm is the {want_n}-dataset one")
s = json.load(open(split))
n = [len(s[k]) for k in ("train", "val", "test")]
if want_split and [int(x) for x in want_split.split()] != n:
    raise SystemExit(f"FAIL LOUD: split artifact is {n}, expected {want_split}")
if sum(n) != want_n:
    raise SystemExit(f"FAIL LOUD: split totals {sum(n)} != cache num_datasets {want_n}")
print(f"[cache] {want_n} datasets, contract ok, split {n}")
EOF2
mkdir -p logs models/tabular
if [[ -f "$CK" ]]; then echo "[skip] $CK exists"; exit 0; fi

python3 -u run_experiment.py "$CFG" --seed "$SEED"

[[ -f "$CK" ]] || { echo "FAIL LOUD: $CFG seed $SEED produced no $CK" >&2; exit 1; }
# An MLP checkpoint has NO .contract.json -- its contract lives INSIDE the .pt (a blanket
# sidecar requirement silently blocks every MLP arm, docs/lessons.md). So pin the keys the
# trainer actually writes there, which is a real check and not a waived one.
python3 - "$CK" "$SPLIT" <<'EOF2'
import hashlib, json, sys
import torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
# dim47crk, NOT dim63crk: the v3 block is 22 columns, so the MLP layout is 25 + 22 = 47.
# The names are distinct on purpose -- `inference_feature_layout` is what stops v2 weights
# meeting v3 features -- so pinning the wrong one here is exactly the mistake worth failing on.
want = {"partial_state": True, "partial_state_contract": "partial_state_v3",
        "peer_mass": True, "inference_feature_layout": "dim47crk"}
for k, v in want.items():
    if ck.get(k) != v:
        raise SystemExit(f"FAIL LOUD: checkpoint {k}={ck.get(k)!r}, expected {v!r}")
sha = hashlib.sha256(open(sys.argv[2], "rb").read()).hexdigest()
got = (ck.get("split_artifact") or {}).get("sha256") if isinstance(ck.get("split_artifact"), dict) else None
if got is not None and got != sha:
    raise SystemExit(f"FAIL LOUD: checkpoint split sha {got[:12]} != {sys.argv[2]} {sha[:12]}")
print("[contract] ok:", {k: ck.get(k) for k in
      ("inference_feature_layout", "partial_state_contract", "peer_mass",
       "queue_feature_contract")})
EOF2
echo "=== task $T done: $CK ==="
