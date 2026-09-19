#!/bin/bash
# peer_only_v1 -- training. docs/lineages/peer_only_v1.md.
#
# Task table (the shell SPECS below is the single source of truth):
#   Phase A   tasks  0-15   peeronly on the 516-dataset psv3 cache      (experiments/peer_only_v1_peeronly.yaml)
#   Phase B   tasks 16-31   gnn      on the 1,670-dataset cache         (experiments/peer_only_v1_1670_gnn.yaml)
#             tasks 32-47   mpoff    on the 1,670-dataset cache         (experiments/peer_only_v1_1670_mpoff.yaml)
#             tasks 48-63   peeronly on the 1,670-dataset cache         (experiments/peer_only_v1_1670_peeronly.yaml)
#   C4        tasks 64-79   gnnres   on the 1,670-dataset cache         (experiments/peer_only_v1_1670_gnnres.yaml)
#                           AMENDMENT 9: residual bipartite MP, the repair C3 names
#
# Same recipe as partial_state_v3 (T1b, lr 2e-3 not re-tuned, CE-only, teacher-forced
# masked_topo, T1b split, patience 60 / floor 100). Phase B tasks refuse to start without
# a passing B0 artefact. Checkpoints: models/peer-only-v1-{516,1670}-{arm}-lr2e3-seed{n}.pt
#
#   sbatch --array=0-15  scripts_cosim/datalab/peer_only_v1_train.sbatch   # Phase A
#   sbatch --array=16-63 scripts_cosim/datalab/peer_only_v1_train.sbatch   # Phase B, after po-corpus
#
#SBATCH --job-name=po-train
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/po-train-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/po-train-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-79

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
python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__,'cuda',torch.cuda.is_available())"
echo "[CODE] $(git describe --dirty --always) on $(git rev-parse --abbrev-ref HEAD)"

# C4 -- AMENDMENT 9 (2026-09-18), tasks 64-79. gnnres: the gnn recipe with RESIDUAL
# bipartite message passing (x = x0 + mp_gate * h instead of x = h), the one repair C3
# licenses. APPENDED so tasks 0-63 keep their arms and their checkpoints.
SPECS=("516 peeronly 16" "1670 gnn 16" "1670 mpoff 16" "1670 peeronly 16" "1670 gnnres 16")
CORPORA=(); ARMS=(); SEEDS=()
for spec in "${SPECS[@]}"; do
  read -r corpus arm n <<< "$spec"
  for s in $(seq 1 "$n"); do CORPORA+=("$corpus"); ARMS+=("$arm"); SEEDS+=("$s"); done
done
NTASKS=${#ARMS[@]}
T=${SLURM_ARRAY_TASK_ID}
if (( T >= NTASKS )); then echo "[TASK $T] beyond the $NTASKS declared arms"; exit 0; fi
CORPUS=${CORPORA[$T]}; ARM=${ARMS[$T]}; SEED=${SEEDS[$T]}

if [[ $CORPUS == 516 ]]; then
  CFG=experiments/peer_only_v1_${ARM}.yaml
  CACHE=simulation_data/graphs_cache_drainable_objective_v1_v1_psv3
  GUARD=simulation_data/partial_state_v3/p0.json; GUARD_KEY=verdict
else
  CFG=experiments/peer_only_v1_1670_${ARM}.yaml
  CACHE=simulation_data/graphs_cache_peer_only_v1_1670_psv3
  GUARD=simulation_data/peer_only_v1/b0.json; GUARD_KEY=verdict
fi
CK="models/peer-only-v1-${CORPUS}-${ARM}-lr2e3-seed${SEED}.pt"
# The split artifact must enumerate EVERY parent of the cache (assert_split_artifact_covers
# refuses a subset -- job 782116 died on that in 9 s). The 1,657 artifact carries T1b's test
# and val verbatim and only extends train, so selection and held-out are identical across
# the two corpora.
if [[ $CORPUS == 516 ]]; then SPLIT=experiments/peer_affinity_v1_t1b_split.json
else SPLIT=experiments/peer_only_v1_1670_split.json; fi
echo "[TASK $T/$NTASKS] corpus=$CORPUS arm=$ARM seed=$SEED cfg=$CFG cache=$CACHE -> $CK"

[[ -f "$CFG" ]] || { echo "FAIL LOUD: $CFG missing (git pull?)" >&2; exit 1; }
[[ -f "$CACHE/metadata.json" ]] || { echo "FAIL LOUD: $CACHE has no metadata.json" >&2; exit 1; }
[[ -f "$CACHE/valid_combos_near_rtt_capped.pkl" ]] || { echo "FAIL LOUD: $CACHE lacks the near-RTT sidecar" >&2; exit 1; }
[[ -f "$CACHE/rtt_chunk_0.pkl" ]] || { echo "FAIL LOUD: $CACHE lacks rtt_chunk_*.pkl" >&2; exit 1; }
[[ -f "$SPLIT" ]] || { echo "FAIL LOUD: split artifact $SPLIT missing" >&2; exit 1; }
[[ -f "$GUARD" ]] || { echo "FAIL LOUD: $GUARD missing -- the instrument bar for this cache has not been read" >&2; exit 1; }
python3 - "$GUARD" <<'EOF2'
import json, sys
g = json.load(open(sys.argv[1]))
if g.get("verdict") != "INSTRUMENT-PASS":
    raise SystemExit(f"FAIL LOUD: {sys.argv[1]} verdict is {g.get('verdict')!r}; nothing trains on a cache that failed its instrument bar")
print("[guard] INSTRUMENT-PASS from", sys.argv[1])
EOF2
python3 - "$CACHE" <<'EOF2'
import json, sys
m = json.load(open(sys.argv[1] + "/metadata.json"))
for k, v in (("partial_state_contract", "partial_state_v3"), ("dag_primary_alpha_key", "2.5"),
             ("peer_exchange_block", True), ("platform_feature_dim", 14)):
    if m.get(k) != v:
        raise SystemExit(f"FAIL LOUD: cache metadata {k}={m.get(k)!r}, this recipe requires {v!r}")
got = m.get("label_objective") or "rtt"
if not got.startswith("rtt_drift:1") or "clock=measured" not in got:
    raise SystemExit(f"FAIL LOUD: cache label_objective={got!r}")
print("[cache] contract ok, label =", got, "| datasets =", m.get("num_datasets"))
EOF2
mkdir -p logs models
if [[ -f "$CK" ]]; then echo "[skip] $CK exists"; exit 0; fi

python3 -u run_experiment.py "$CFG" --seed "$SEED"

RUN_NAME=$(python3 -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['wandb']['run_name'])" "$CFG")
PRODUCED="models/${RUN_NAME}-seed${SEED}.pt"
if [[ -f "$PRODUCED" && ! -f "$CK" ]]; then
  mv "$PRODUCED" "$CK"
  [[ -f "${PRODUCED%.pt}.contract.json" ]] && mv "${PRODUCED%.pt}.contract.json" "${CK%.pt}.contract.json"
  [[ -f "${PRODUCED%.pt}-final.pt" ]] && mv "${PRODUCED%.pt}-final.pt" "${CK%.pt}-final.pt"
  [[ -f "${PRODUCED%.pt}-final.contract.json" ]] && mv "${PRODUCED%.pt}-final.contract.json" "${CK%.pt}-final.contract.json"
fi
[[ -f "$CK" ]] || { echo "FAIL LOUD: $CFG seed $SEED produced no $CK (looked for $PRODUCED)" >&2; exit 1; }
SC="${CK%.pt}.contract.json"
[[ -f "$SC" ]] || { echo "FAIL LOUD: no sidecar $SC" >&2; exit 1; }
python3 - "$SC" "$ARM" "$SPLIT" <<'EOF2'
import hashlib, json, sys
sc, arm, split = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3]
checks = {"disable_message_passing": arm == "mpoff", "mp_peer_edges": True,
          "mp_platform_edges": arm != "peeronly",
          # C4: mp_residual is the ONE thing separating gnnres from gnn. Pin it on BOTH
          # arms -- checking only the new one would let a silently-residual gnn through.
          "mp_residual": arm == "gnnres",
          "partial_state_contract": "partial_state_v3", "partial_state_feature_dim": 22,
          "dag_alpha_key": "2.5", "decode_replica_reuse": True, "decode_relax_on_stuck": True, "peer_mass": True}
for k, want in checks.items():
    if sc.get(k) != want:
        raise SystemExit(f"FAIL LOUD: sidecar {k}={sc.get(k)!r}, arm {arm} expects {want!r}")
es = sc.get("early_stopping") or {}
if (es.get("patience"), es.get("min_epochs")) != (60, 100):
    raise SystemExit(f"FAIL LOUD: sidecar early_stopping={es!r}")
sa = sc.get("split_artifact") or {}
sha = hashlib.sha256(open(split, "rb").read()).hexdigest()
if sa.get("sha256") != sha:
    raise SystemExit(f"FAIL LOUD: sidecar split sha {sa.get('sha256')} != {split} {sha}")
print("[sidecar] ok:", {k: sc.get(k) for k in ("disable_message_passing", "mp_peer_edges", "mp_platform_edges")})
EOF2
echo "=== task $T done: $CK ==="
