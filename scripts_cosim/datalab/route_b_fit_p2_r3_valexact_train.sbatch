#!/bin/bash
# route_b fit-ceiling Phase 2 RETRAIN (2026-09-07): rung-3 GNN MP-ON + MP-OFF only, with
# NEAR_RTT_VAL_EXACT_REGRET=1 so checkpoint selection uses the full enumerated sweep instead
# of the ~66%-censored capped sidecar (docs/gates/gate-tools.md, 2026-09-07 training-chain
# audit entry). The MLP arm needs no retrain -- it was already val-selected on an uncensored
# metric. Evaluated with the existing scripts_cosim/eval_route_b_stage2_arm.py against the
# SAME rung-3 cache/split as the original registration; nothing about the corpus changes.
#
#   task = 8*arm + (seed-1)   arm 0 gnn / 1 mpoff, seed 1..8
#
#SBATCH --job-name=rb-p2-r3-valexact
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/rb-p2-r3-valexact-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/rb-p2-r3-valexact-%A_%a.err
#SBATCH --partition=GPU-a40,GPU-a100s,GPU-l40s,GPU-a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-15

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
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8

python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__,'cuda',torch.cuda.is_available())"
echo "[CODE] $(git describe --dirty --always) on $(git rev-parse --abbrev-ref HEAD)"

T=${SLURM_ARRAY_TASK_ID}
ARMI=$(( T / 8 )); SEED=$(( T % 8 + 1 ))
case $ARMI in
  0) ARM=gnn;   CFG=experiments/route_b_fit_p2_r3_gnn_valexact.yaml;   CK="models/route-b-fit-p2-r3-gnn-valexact-seed${SEED}.pt" ;;
  1) ARM=mpoff; CFG=experiments/route_b_fit_p2_r3_mpoff_valexact.yaml; CK="models/route-b-fit-p2-r3-mpoff-valexact-seed${SEED}.pt" ;;
esac
CACHE=simulation_data/graphs_cache_route_b_fit_p2_r3
SPLIT=experiments/route_b_fit_p2_split_r3.json
echo "[TASK $T] arm=$ARM seed=$SEED cfg=$CFG -> $CK"

[[ -f "$CACHE/metadata.json" ]] || { echo "FAIL LOUD: $CACHE has no metadata.json" >&2; exit 1; }
[[ -f "$CACHE/valid_combos_near_rtt_capped.pkl" ]] || { echo "FAIL LOUD: $CACHE lacks valid_combos_near_rtt_capped.pkl (build locally, rsync; never let concurrent tasks race on it)" >&2; exit 1; }
[[ -f "$CACHE/rtt_chunk_0.pkl" ]] || { echo "FAIL LOUD: $CACHE lacks rtt_chunk_*.pkl -- NEAR_RTT_VAL_EXACT_REGRET needs the full sweep" >&2; exit 1; }
[[ -f "$SPLIT" ]] || { echo "FAIL LOUD: split artifact $SPLIT missing" >&2; exit 1; }
mkdir -p logs models

# We want the VAL-SELECTED checkpoint this time, not -final -- that is the whole point of
# the retrain, so skip-if-exists keys on it rather than the -final sibling.
if [[ -f "$CK" ]]; then echo "[skip] $CK exists"; exit 0; fi

python3 -u run_experiment.py "$CFG" --seed "$SEED"

[[ -f "$CK" ]] || { echo "FAIL LOUD: $CFG seed $SEED produced no $CK" >&2; exit 1; }
SC="${CK%.pt}.contract.json"
[[ -f "$SC" ]] || { echo "FAIL LOUD: no sidecar $SC" >&2; exit 1; }
python3 - "$SC" "$ARM" "$SPLIT" <<'EOF'
import hashlib, json, sys
sc, arm, split = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3]
want = arm == "mpoff"
got = sc.get("disable_message_passing")
if got != want:
    raise SystemExit(f"FAIL LOUD: sidecar disable_message_passing={got}, arm {arm} expects {want}")
sa = sc.get("split_artifact") or {}
sha = hashlib.sha256(open(split, "rb").read()).hexdigest()
if sa.get("sha256") not in (None, sha):
    raise SystemExit(f"FAIL LOUD: sidecar split sha {sa.get('sha256')} != {split} {sha}")
print(f"[sidecar] disable_message_passing={got} split_sha_ok={sa.get('sha256') == sha}")
EOF
echo "=== task $T done: $CK ==="
