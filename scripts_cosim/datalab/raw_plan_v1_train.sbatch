#!/bin/bash
# raw_plan_v1 (docs/lineages/raw_plan_v1.md) -- ARM=rawgnn|rawmlp, seeds 1-4 (tasks 0-3); SMOKE=1 runs the 2-epoch
# smoke config instead (checkpoint named -smoke, never served). peak_controls_v1_train.sbatch but for the arm and the
# raw-plan sidecar checks. Worktree ~/gnn-herosim-wt/grounded; cache, datasets and models are symlinks.
#   sbatch --array=0-3 --export=ALL,ARM=rawgnn raw_plan_v1_train.sbatch
#SBATCH --job-name=rp1-train
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/rp1-train-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/rp1-train-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00

set -euo pipefail
ulimit -n 65536 2>/dev/null || true
WT=/home/nikola.lukic/gnn-herosim-wt/grounded
cd "$WT"
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$WT"
export PYTHONUNBUFFERED=1
export WANDB_ENTITY=nikolalukic167-tu-wien
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 CUDA_VISIBLE_DEVICES=""
python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"
echo "[CODE] $(git describe --dirty --always) at $WT"
[[ -z "$(git status --porcelain -- src scripts_cosim experiments run_experiment.py)" ]] || { echo "FAIL LOUD: worktree dirty" >&2; exit 1; }
[[ -L models ]] || { echo "FAIL LOUD: models/ must be the symlink into ~/gnn-herosim/models" >&2; exit 1; }

T=${SLURM_ARRAY_TASK_ID:?FAIL LOUD: SLURM_ARRAY_TASK_ID unset}
ARM=${ARM:?FAIL LOUD: ARM unset (rawgnn|rawmlp)}; SEED=$(( T + 1 ))
[[ $ARM == rawgnn || $ARM == rawmlp ]] || { echo "FAIL LOUD: ARM=$ARM" >&2; exit 1; }
SUF=""; [[ "${SMOKE:-0}" == 1 ]] && SUF="_smoke"
CFG=experiments/raw_plan_v1_${ARM}${SUF}.yaml
CACHE=simulation_data/graphs_cache_backlog_corpus_v1_psv4_inf
SPLIT=experiments/backlog_corpus_v1_split.json
CK="models/raw-plan-v1-${ARM}${SUF//_/-}-lr2e3-seed${SEED}.pt"
echo "[TASK $T] arm=$ARM seed=$SEED cfg=$CFG -> $CK"
[[ -f "$CFG" ]] || { echo "FAIL LOUD: $CFG missing" >&2; exit 1; }
[[ -f "$CACHE/metadata.json" && -f "$CACHE/valid_combos_near_rtt_capped.pkl" ]] || { echo "FAIL LOUD: $CACHE incomplete" >&2; exit 1; }
[[ "$(sha256sum "$SPLIT" | cut -d' ' -f1)" == 9c459b16ebf5e96dad7486f87cd681a647fe2e3121e16316dd0ae5df4fad5463 ]] || { echo "FAIL LOUD: split sha mismatch" >&2; exit 1; }
python3 - "$CACHE" <<'EOF2'
import json, sys
m = json.load(open(sys.argv[1] + "/metadata.json"))
for k, v in (("partial_state_contract", "partial_state_v4"), ("dag_primary_alpha_key", "inf"), ("peer_exchange_block", True), ("platform_feature_dim", 14)):
    if m.get(k) != v:
        raise SystemExit(f"FAIL LOUD: cache metadata {k}={m.get(k)!r}, this recipe requires {v!r}")
print("[cache] v4 ok | datasets =", m.get("num_datasets"))
EOF2
mkdir -p logs
if [[ -f "$CK" ]]; then echo "FAIL LOUD: $CK already exists; refusing to overwrite" >&2; exit 1; fi
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
python3 scripts_cosim/joint_burst_v2_sidecheck.py "$SC" "$ARM" "$SPLIT" inf
python3 - "$SC" "$ARM" "$CK" <<'EOF2'
import json, sys
import torch
sc, arm, ck = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3]
checks = {"decode_replica_reuse": True, "decode_relax_on_stuck": True, "plan_raw": True,
          "partial_state_feature_dim": 2, "disable_message_passing": arm == "rawmlp",
          "mp_bipartite_edge_conv": arm == "rawgnn", "mp_peer_edges": True, "full_context_ce_weight": 1.0}
for k, want in checks.items():
    if sc.get(k) != want:
        raise SystemExit(f"FAIL LOUD: sidecar {k}={sc.get(k)!r}, arm {arm} expects {want!r}")
es = sc.get("early_stopping") or {}
if "smoke" not in ck and (es.get("patience"), es.get("min_epochs")) != (60, 100):
    raise SystemExit(f"FAIL LOUD: sidecar early_stopping={es!r}")
sd = torch.load(ck, map_location="cpu")
if (arm == "rawgnn") != any(k.startswith("bip_convs.") for k in sd):
    raise SystemExit(f"FAIL LOUD: bip_convs.* presence does not match arm {arm}")
w = sd["edge_scorer.fc1.weight"].shape[1]
if w != 64 + 64 + 5 + 2:
    raise SystemExit(f"FAIL LOUD: scorer input width {w}, raw plan expects 135")
print(f"[sidecar] ok: arm={arm} contract={sc.get('partial_state_contract')} load_seconds={sc.get('load_seconds')}")
EOF2
echo "=== task $T done: $CK ==="
