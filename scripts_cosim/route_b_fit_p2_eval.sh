#!/bin/bash
# route_b fit-ceiling Phase 2 — evaluate every rung/arm/seed checkpoint offline (local: the
# evaluator reads placements.jsonl from the dataset dirs, which live here; ~8 s per 204
# datasets). GNN arms are scored at their LAST epoch (`-final.pt`), as Phase 1 was; the MLP
# trainer writes one best-val checkpoint. Then the registered reader per rung:
#   scripts_cosim/analyze_route_b_fit_p1.py --arm gnn=... --arm mpoff=... --arm mlp=...
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${HEROSIM_PY:-pipenv run python3}"
export PIPENV_IGNORE_VIRTUALENVS=1 VIRTUAL_ENV= PYTHONPATH="$PWD"
OUT=simulation_data/route_b_fit_p2; mkdir -p "$OUT"
SEEDS="${SEEDS:-1 2 3 4 5 6 7 8}"
RUNGS="${RUNGS:-1 2 3}"

run_eval() {  # rung arm ckpt [mp_off]
  local r=$1 arm=$2 ck=$3 mpoff=${4:-0}
  local rep="$OUT/eval_r${r}_${arm}_seed${s}.json"
  [[ -f "$ck" ]] || { echo "FAIL LOUD: missing $ck" >&2; exit 1; }
  if [[ -f "$rep" ]]; then echo "[skip] $rep"; return; fi
  if [[ $mpoff == 1 ]]; then export GNN_DISABLE_MESSAGE_PASSING=1; else unset GNN_DISABLE_MESSAGE_PASSING; fi
  $PY scripts_cosim/eval_route_b_stage2_arm.py --checkpoint "$ck" \
      --cache-dir "simulation_data/graphs_cache_route_b_fit_p2_r${r}" \
      --split-artifact "experiments/route_b_fit_p2_split_r${r}.json" --report "$rep" \
      > "$OUT/$(basename "$rep" .json).log" 2>&1 \
    || { echo "FAIL LOUD: eval failed for $ck (see $OUT/$(basename "$rep" .json).log)" >&2; exit 1; }
  echo "[ok] $rep: $(tail -1 "$OUT/$(basename "$rep" .json).log")"
}

for r in $RUNGS; do
  for s in $SEEDS; do
    run_eval "$r" gnn   "models/route-b-fit-p2-r${r}-gnn-seed${s}-final.pt"
    run_eval "$r" mpoff "models/route-b-fit-p2-r${r}-mpoff-seed${s}-final.pt" 1
    run_eval "$r" mlp   "models/tabular/route_b_fit_p2_r${r}_mlp_seed${s}.pt"
  done
  $PY scripts_cosim/analyze_route_b_fit_p1.py \
      --arm "gnn=$OUT/eval_r${r}_gnn_seed*.json" \
      --arm "mpoff=$OUT/eval_r${r}_mpoff_seed*.json" \
      --arm "mlp=$OUT/eval_r${r}_mlp_seed*.json" \
      --out "simulation_data/route_b_fit_p2_r${r}_verdict.json" | tee "$OUT/verdict_r${r}.txt"
done
echo "=== all reports in $OUT ==="
