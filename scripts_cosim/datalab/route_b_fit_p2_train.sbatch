#!/bin/bash
# route_b fit-ceiling Phase 2 (2026-09-06): LEARNING CURVE on the Arm S DAG corpus — the
# three Phase 1 arms at three training-set sizes against ONE fixed held-out block. Registered
# before any rung was trained: docs/lineages/route_b_v1.md, "Phase 2" entry.
#
#   task = 24*(rung-1) + 8*arm + (seed-1)      rung 1..3, arm 0 gnn / 1 mpoff / 2 mlp, seed 1..8
#   rung 1: 204 train parents  (graphs_cache_route_b_fit_p2_r1)
#   rung 2: 612 train parents  (graphs_cache_route_b_fit_p2_r2)
#   rung 3: 1020 train parents (graphs_cache_route_b_fit_p2_r3)
#
# Training only. Evaluation (scripts_cosim/route_b_fit_p2_eval.sh) needs placements.jsonl
# from the dataset dirs, which live locally; checkpoints + sidecars are rsynced back.
# Rung caches were built locally by scripts_cosim/route_b_fit_p2_build.sh and rsynced here
# WITH their valid_combos_near_rtt_capped.pkl — 24 tasks lazily building that sidecar on a
# shared cache would race (datalab-pitfalls #7), so this script refuses to start without it.
#
#SBATCH --job-name=rb-fit-p2
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/rb-fit-p2-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/rb-fit-p2-%A_%a.err
#SBATCH --partition=GPU-a40,GPU-a100s,GPU-l40s,GPU-a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-71

set -euo pipefail
# datalab-pitfalls #11: without this every GNN task dies in the first epochs with
# `RuntimeError: received 0 items of ancdata` (job 740198, 2026-09-06).
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
RUNG=$(( T / 24 + 1 )); REM=$(( T % 24 )); ARMI=$(( REM / 8 )); SEED=$(( REM % 8 + 1 ))
case $ARMI in
  0) ARM=gnn;   CFG=experiments/route_b_fit_p2_r${RUNG}_gnn.yaml;   CK="models/route-b-fit-p2-r${RUNG}-gnn-seed${SEED}-final.pt" ;;
  1) ARM=mpoff; CFG=experiments/route_b_fit_p2_r${RUNG}_mpoff.yaml; CK="models/route-b-fit-p2-r${RUNG}-mpoff-seed${SEED}-final.pt" ;;
  2) ARM=mlp;   CFG=experiments/route_b_fit_p2_r${RUNG}_mlp.yaml;   CK="models/tabular/route_b_fit_p2_r${RUNG}_mlp_seed${SEED}.pt" ;;
esac
CACHE=simulation_data/graphs_cache_route_b_fit_p2_r${RUNG}
SPLIT=experiments/route_b_fit_p2_split_r${RUNG}.json
echo "[TASK $T] rung=$RUNG arm=$ARM seed=$SEED cfg=$CFG -> $CK"

[[ -f "$CACHE/metadata.json" ]] || { echo "FAIL LOUD: $CACHE has no metadata.json" >&2; exit 1; }
[[ -f "$CACHE/valid_combos_near_rtt_capped.pkl" ]] || { echo "FAIL LOUD: $CACHE lacks valid_combos_near_rtt_capped.pkl (build locally, rsync; never let 24 tasks race on it)" >&2; exit 1; }
[[ -f "$SPLIT" ]] || { echo "FAIL LOUD: split artifact $SPLIT missing" >&2; exit 1; }
mkdir -p logs models/tabular

if [[ -f "$CK" ]]; then echo "[skip] $CK exists"; exit 0; fi

python3 -u run_experiment.py "$CFG" --seed "$SEED"

[[ -f "$CK" ]] || { echo "FAIL LOUD: $CFG seed $SEED produced no $CK" >&2; exit 1; }
if [[ $ARM != mlp ]]; then
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
fi
echo "=== task $T done: $CK ==="
