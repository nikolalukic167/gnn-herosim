#!/bin/bash
# route_b fit-ceiling Phase 1 (2026-09-06): seeds for the three arms of the 2026-09-03
# probe, so the held-out ordering (MP-OFF < MLP+prefix < GNN) can be read as a paired
# statistic instead of one draw. Exploration, not a registration — see the Phase 1 entry
# in docs/lineages/route_b_v1.md.
#
#   task  0- 6  GNN MP-ON  lr 2e-3  seeds 2-8   (seed 1 exists: route-b-fit-a1-e300-lr2e3-seed1)
#   task  7-14  GNN MP-OFF lr 2e-3  seeds 1-8   (new arm: the probe's MP-OFF ran at 5e-4)
#   task 15-21  MLP+prefix          seeds 2-8   (seed 1 exists: route_b_fit_a2_long_seed1)
#
# Training only. Evaluation (eval_route_b_stage2_arm.py) needs placements.jsonl from the
# 204 arm_s dataset dirs, which live locally; checkpoints + sidecars are rsynced back.
#
#SBATCH --job-name=rb-fit-p1
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/rb-fit-p1-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/rb-fit-p1-%A_%a.err
#SBATCH --partition=GPU-a40,GPU-a100s,GPU-l40s,GPU-a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-21

set -euo pipefail
# train_near_rtt.py's multi-worker DataLoader shares tensors over file descriptors; at the
# node default of 1024 every GNN task dies in the first epochs with
# `RuntimeError: received 0 items of ancdata` (job 740198, 15/15 GNN tasks, 2026-09-06).
# Every other GPU training sbatch in this directory carries this line for that reason.
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

CACHE=simulation_data/graphs_cache_route_b_pilot_s_dag
[[ -f "$CACHE/metadata.json" ]] || { echo "FAIL LOUD: $CACHE has no metadata.json" >&2; exit 1; }
[[ -f experiments/route_b_stage2_split_v1.json ]] || { echo "FAIL LOUD: split artifact missing" >&2; exit 1; }
mkdir -p logs models/tabular

T=${SLURM_ARRAY_TASK_ID}
if   (( T <= 6 ));  then ARM=gnn;   SEED=$((T + 2));  CFG=experiments/route_b_fit_a1_e300_lr2e3.yaml;  CK="models/route-b-fit-a1-e300-lr2e3-seed${SEED}-final.pt"
elif (( T <= 14 )); then ARM=mpoff; SEED=$((T - 6));  CFG=experiments/route_b_fit_p1_mpoff_lr2e3.yaml; CK="models/route-b-fit-p1-mpoff-lr2e3-seed${SEED}-final.pt"
else                     ARM=mlp;   SEED=$((T - 13)); CFG=experiments/route_b_fit_a2_long.yaml;        CK="models/tabular/route_b_fit_a2_long_seed${SEED}.pt"
fi
echo "[TASK $T] arm=$ARM seed=$SEED cfg=$CFG -> $CK"

if [[ -f "$CK" ]]; then echo "[skip] $CK exists"; exit 0; fi

python3 -u run_experiment.py "$CFG" --seed "$SEED"

[[ -f "$CK" ]] || { echo "FAIL LOUD: $CFG seed $SEED produced no $CK" >&2; exit 1; }
if [[ $ARM != mlp ]]; then
  SC="${CK%.pt}.contract.json"
  [[ -f "$SC" ]] || { echo "FAIL LOUD: no sidecar $SC" >&2; exit 1; }
  python3 - "$SC" "$ARM" <<'EOF'
import json, sys
sc, arm = json.load(open(sys.argv[1])), sys.argv[2]
want = arm == "mpoff"
got = sc.get("disable_message_passing")
if got != want:
    raise SystemExit(f"FAIL LOUD: sidecar disable_message_passing={got}, arm {arm} expects {want}")
print(f"[sidecar] disable_message_passing={got} OK")
EOF
fi
echo "=== task $T done: $CK ==="
