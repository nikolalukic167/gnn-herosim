#!/bin/bash
# partial_state_v3 -- training: the drainable_objective_v1 V = 1 recipe under the size-free
# partial-state contract. docs/lineages/partial_state_v3.md.
#
# Template: drainable_objective_v1_train.sbatch. Same corpus (516 parents), same split, same
# label, same lr (2e-3, NOT re-tuned), same 16 seeds per arm. Differences, all declared in
# experiments/partial_state_v3_{gnn,mpoff}.yaml: the psv3 cache, PARTIAL_STATE_CONTRACT=
# partial_state_v3 (dim 38 -> 22), and --patience 60 --min-epochs 100.
#
# 2 arms x 16 seeds = 32 tasks. `mpoff` is the pointwise control; no GNN-vs-pointwise claim.
#
#   sbatch scripts_cosim/datalab/partial_state_v3_train.sbatch
#
#SBATCH --job-name=psv3-train
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/psv3-train-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/psv3-train-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-31

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

SPECS=("gnn 16" "mpoff 16")
ARMS=(); SEEDS=()
for spec in "${SPECS[@]}"; do
  read -r arm n <<< "$spec"
  for s in $(seq 1 "$n"); do ARMS+=("$arm"); SEEDS+=("$s"); done
done
NTASKS=${#ARMS[@]}

T=${SLURM_ARRAY_TASK_ID}
if (( T >= NTASKS )); then echo "[TASK $T] beyond the $NTASKS declared arms, nothing to do"; exit 0; fi
ARM=${ARMS[$T]}; SEED=${SEEDS[$T]}

CFG=experiments/partial_state_v3_${ARM}.yaml
CK="models/partial-state-v3-${ARM}-lr2e3-seed${SEED}.pt"
CACHE=simulation_data/graphs_cache_drainable_objective_v1_v1_psv3
SPLIT=experiments/peer_affinity_v1_t1b_split.json
P0=simulation_data/partial_state_v3/p0.json
echo "[TASK $T/$NTASKS] arm=$ARM seed=$SEED cfg=$CFG -> $CK"

[[ -f "$CFG" ]] || { echo "FAIL LOUD: $CFG missing (git pull?)" >&2; exit 1; }
[[ -f "$CACHE/metadata.json" ]] || { echo "FAIL LOUD: $CACHE has no metadata.json" >&2; exit 1; }
[[ -f "$CACHE/valid_combos_near_rtt_capped.pkl" ]] || { echo "FAIL LOUD: $CACHE lacks valid_combos_near_rtt_capped.pkl" >&2; exit 1; }
[[ -f "$CACHE/rtt_chunk_0.pkl" ]] || { echo "FAIL LOUD: $CACHE lacks rtt_chunk_*.pkl" >&2; exit 1; }
[[ -f "$SPLIT" ]] || { echo "FAIL LOUD: split artifact $SPLIT missing" >&2; exit 1; }
# P0 is BLOCKING: a cache that is not a pure representation change trains nothing.
[[ -f "$P0" ]] || { echo "FAIL LOUD: $P0 missing -- run partial_state_v3_cache.sbatch (P0) first" >&2; exit 1; }
python3 - "$P0" <<'EOF2'
import json, sys
p0 = json.load(open(sys.argv[1]))
if p0.get("verdict") != "INSTRUMENT-PASS":
    raise SystemExit(f"FAIL LOUD: P0 verdict is {p0.get('verdict')!r}; nothing is trained on a cache that fails the instrument bar")
print("[P0] INSTRUMENT-PASS on", p0.get("n_datasets"), "datasets")
EOF2

python3 - "$CACHE" <<'EOF2'
import json, sys
m = json.load(open(sys.argv[1] + "/metadata.json"))
for k, v in (("partial_state_contract", "partial_state_v3"), ("dag_primary_alpha_key", "2.5"),
             ("peer_exchange_block", True), ("platform_feature_dim", 14), ("num_datasets", 516)):
    if m.get(k) != v:
        raise SystemExit(f"FAIL LOUD: cache metadata {k}={m.get(k)!r}, this recipe requires {v!r}")
got = m.get("label_objective") or "rtt"
if not got.startswith("rtt_drift:1") or "clock=measured" not in got:
    raise SystemExit(f"FAIL LOUD: cache label_objective={got!r}, expected the V = 1 measured-clock label")
print("[cache] contract ok, label =", got)
EOF2
mkdir -p logs models

if [[ -f "$CK" ]]; then echo "[skip] $CK exists"; exit 0; fi

python3 -u run_experiment.py "$CFG" --seed "$SEED"

PRODUCED="models/partial-state-v3-${ARM}-seed${SEED}.pt"
if [[ -f "$PRODUCED" && ! -f "$CK" ]]; then
  mv "$PRODUCED" "$CK"
  [[ -f "${PRODUCED%.pt}.contract.json" ]] && mv "${PRODUCED%.pt}.contract.json" "${CK%.pt}.contract.json"
  [[ -f "${PRODUCED%.pt}-final.pt" ]] && mv "${PRODUCED%.pt}-final.pt" "${CK%.pt}-final.pt"
  [[ -f "${PRODUCED%.pt}-final.contract.json" ]] && mv "${PRODUCED%.pt}-final.contract.json" "${CK%.pt}-final.contract.json"
fi
[[ -f "$CK" ]] || { echo "FAIL LOUD: $CFG seed $SEED produced no $CK (looked for $PRODUCED too)" >&2; exit 1; }
SC="${CK%.pt}.contract.json"
[[ -f "$SC" ]] || { echo "FAIL LOUD: no sidecar $SC" >&2; exit 1; }
python3 - "$SC" "$ARM" "$SPLIT" <<'EOF2'
import hashlib, json, sys
sc, arm, split = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3]
checks = {"disable_message_passing": arm == "mpoff", "mp_peer_edges": True,
          "partial_state_contract": "partial_state_v3", "partial_state_feature_dim": 22,
          "dag_alpha_key": "2.5", "decode_replica_reuse": True, "decode_relax_on_stuck": True,
          "peer_mass": True}
for k, want in checks.items():
    if sc.get(k) != want:
        raise SystemExit(f"FAIL LOUD: sidecar {k}={sc.get(k)!r}, arm {arm} expects {want!r}")
es = sc.get("early_stopping") or {}
if (es.get("patience"), es.get("min_epochs")) != (60, 100):
    raise SystemExit(f"FAIL LOUD: sidecar early_stopping={es!r}, the recipe is patience 60 / floor 100")
got = sc.get("label_objective") or "rtt"
if not got.startswith("rtt_drift:1"):
    raise SystemExit(f"FAIL LOUD: sidecar label_objective={got!r}, expected the V = 1 label")
sa = sc.get("split_artifact") or {}
sha = hashlib.sha256(open(split, "rb").read()).hexdigest()
if sa.get("sha256") != sha:
    raise SystemExit(f"FAIL LOUD: sidecar split sha {sa.get('sha256')} != {split} {sha}")
print("[sidecar] ok: contract v3, dim 22, patience 60/100, label =", got)
EOF2
echo "=== task $T done: $CK ==="
