#!/bin/bash
# NEAR_RTT_CE_GRAPH_WEIGHT checks on the real accel cache: base vs patched-off bit-identity, flag-on run twice (determinism), flag-on differs from off.
#   sbatch --export=ALL,NEW=<patched pinned worktree>,BASE=<base pinned worktree>,OUT=<dir> accel_stakes_checks.sbatch
# Both worktrees need a real models/ dir, experiments/accel_replica_v1_split.json and simulation_data/graphs_cache_accel_replica_v1_psv5_inf.
#SBATCH --job-name=accel-stakes-chk
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/accel-stakes-chk-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --cpus-per-task=36
#SBATCH --mem=150G
#SBATCH --time=02:00:00
set -uo pipefail
NEW=${NEW:?}; BASE=${BASE:?}; OUT=${OUT:?}; mkdir -p "$OUT"
for v in $(compgen -e | grep -E '^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|NEAR_RTT_|PARTIAL_STATE)'); do unset "$v"; done
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"; micromamba activate gnn
export HEROSIM_PY=python3 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=""
export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa HEROSIM_SHARED_AUTOSCALER=0 GATE_FIXED_POLICY_TIME_SCALE=1.0
S="$NEW/scripts_cosim/accel_stakes_checks.py"; fail=0
( cd "$BASE" && python3 "$S" run "$BASE" "$OUT/base.pt" > "$OUT/base.log" 2>&1 ) & pb=$!
( cd "$NEW" && for t in off on_a on_b; do python3 "$S" run "$NEW" "$OUT/new_$t.pt" $([[ $t == off ]] || echo stakes) > "$OUT/new_$t.log" 2>&1 || exit 1; done ) & pn=$!
wait $pb || { echo "FAIL: base run"; fail=1; }; wait $pn || { echo "FAIL: patched runs"; fail=1; }
grep -h "ce-graph-weight" "$OUT"/new_on_a.log | head -2
python3 "$S" compare "$OUT/base.pt" "$OUT/new_off.pt" || fail=1
python3 "$S" compare "$OUT/new_on_a.pt" "$OUT/new_on_b.pt" || fail=1
python3 "$S" compare "$OUT/new_off.pt" "$OUT/new_on_a.pt" && { echo "FAIL: flag on changed nothing"; fail=1; } || echo "flag on != flag off (expected)"
echo "=== fail=$fail"; exit $fail
