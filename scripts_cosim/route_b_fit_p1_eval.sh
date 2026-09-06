#!/bin/bash
# route_b fit-ceiling Phase 1 — evaluate every arm/seed checkpoint offline (local: the
# evaluator reads placements.jsonl from the 204 arm_s dataset dirs, which live here).
# Writes simulation_data/route_b_fit_p1/eval_{arm}_seed{N}.json (val-selected) and
# eval_{arm}final_seed{N}.json (last epoch; GNN arms only — the MLP trainer keeps best-val).
# Then: scripts_cosim/analyze_route_b_fit_p1.py --arm gnn=... --arm mpoff=... --arm mlp=...
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${HEROSIM_PY:-pipenv run python3}"
export PIPENV_IGNORE_VIRTUALENVS=1 VIRTUAL_ENV= PYTHONPATH="$PWD"
OUT=simulation_data/route_b_fit_p1; mkdir -p "$OUT"
CACHE=simulation_data/graphs_cache_route_b_pilot_s_dag
SPLIT=experiments/route_b_stage2_split_v1.json

run_eval() {  # arm ckpt report [mp_off]
  local arm=$1 ck=$2 rep=$3 mpoff=${4:-0}
  [[ -f "$ck" ]] || { echo "FAIL LOUD: missing $ck" >&2; exit 1; }
  if [[ -f "$rep" ]]; then echo "[skip] $rep"; return; fi
  if [[ $mpoff == 1 ]]; then export GNN_DISABLE_MESSAGE_PASSING=1; else unset GNN_DISABLE_MESSAGE_PASSING; fi
  $PY scripts_cosim/eval_route_b_stage2_arm.py --checkpoint "$ck" --cache-dir "$CACHE" \
      --split-artifact "$SPLIT" --report "$rep" > "$OUT/$(basename "$rep" .json).log" 2>&1 \
    || { echo "FAIL LOUD: eval failed for $ck (see $OUT/$(basename "$rep" .json).log)" >&2; exit 1; }
  echo "[ok] $rep: $(tail -1 "$OUT/$(basename "$rep" .json).log")"
}

for s in 1 2 3 4 5 6 7 8; do
  run_eval gnn   models/route-b-fit-a1-e300-lr2e3-seed${s}.pt        "$OUT/eval_gnn_seed${s}.json"
  run_eval gnn   models/route-b-fit-a1-e300-lr2e3-seed${s}-final.pt  "$OUT/eval_gnnfinal_seed${s}.json"
  run_eval mpoff models/route-b-fit-p1-mpoff-lr2e3-seed${s}.pt        "$OUT/eval_mpoff_seed${s}.json"      1
  run_eval mpoff models/route-b-fit-p1-mpoff-lr2e3-seed${s}-final.pt  "$OUT/eval_mpofffinal_seed${s}.json" 1
  run_eval mlp   models/tabular/route_b_fit_a2_long_seed${s}.pt       "$OUT/eval_mlp_seed${s}.json"
done
echo "=== all reports in $OUT ==="
