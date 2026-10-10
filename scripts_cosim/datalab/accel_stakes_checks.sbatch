#!/bin/bash
# NEAR_RTT_CE_GRAPH_WEIGHT checks: determinism test, off-path bit-identity vs the base commit, flag-on reproducibility.
#   sbatch --export=ALL,NEW=<patched pinned worktree>,BASE=<base pinned worktree>,OUT=<dir> accel_stakes_checks.sbatch
#SBATCH --job-name=accel-stakes-chk
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/accel-stakes-chk-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
set -uo pipefail
NEW=${NEW:?}; BASE=${BASE:?}; OUT=${OUT:?}; mkdir -p "$OUT"
for v in $(compgen -e | grep -E '^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|NEAR_RTT_|PARTIAL_STATE)'); do unset "$v"; done
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"; micromamba activate gnn
export HEROSIM_PY=python3 PYTHONUNBUFFERED=1 WANDB_MODE=disabled OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
fail=0
cd "$NEW"; export PYTHONPATH="$NEW"
echo "== determinism test (patched tree $(git rev-parse --short HEAD))"; python3 -m pytest tests/test_trainer_determinism.py -q 2>&1 | tail -5 || fail=1
python3 scripts_cosim/accel_stakes_checks.py "$NEW" "$OUT/new_off.pt" || fail=1
python3 scripts_cosim/accel_stakes_checks.py "$NEW" "$OUT/new_on_a.pt" stakes || fail=1
python3 scripts_cosim/accel_stakes_checks.py "$NEW" "$OUT/new_on_b.pt" stakes || fail=1
cd "$BASE"; export PYTHONPATH="$BASE"
echo "== base tree $(git rev-parse --short HEAD)"; python3 "$NEW/scripts_cosim/accel_stakes_checks.py" "$BASE" "$OUT/base.pt" || fail=1
cd "$NEW"
python3 scripts_cosim/accel_stakes_checks.py compare "$OUT/base.pt" "$OUT/new_off.pt" || fail=1      # off-path: must be identical
python3 scripts_cosim/accel_stakes_checks.py compare "$OUT/new_on_a.pt" "$OUT/new_on_b.pt" || fail=1  # flag on twice: must be identical
python3 scripts_cosim/accel_stakes_checks.py compare "$OUT/new_off.pt" "$OUT/new_on_a.pt" && { echo "FAIL: flag on changed nothing"; fail=1; } || echo "flag on != flag off (expected)"
grep -h "ce-graph-weight" "$HOME"/gnn-herosim/logs/accel-stakes-chk-${SLURM_JOB_ID}.out | head -3
echo "=== fail=$fail"; exit $fail
