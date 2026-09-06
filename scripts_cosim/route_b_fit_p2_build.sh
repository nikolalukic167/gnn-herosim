#!/bin/bash
# route_b fit-ceiling Phase 2 (2026-09-06): build the three learning-curve rung caches and
# their split artifacts. Runs LOCALLY (the 204-graph precedent built in 7.65 s; the 8-hour
# recache precedent was a 2,816-dataset merged corpus). Cache flags are exactly those of the
# frozen `graphs_cache_route_b_pilot_s_dag` (metadata.json: platform_feature_dim 14,
# queue_feature_contract legacy_v0, dag_partial_state true) — rebuilding that cache with
# these flags reproduced graphs.pkl/optimal_rtt.pkl byte-identical on 2026-09-06.
#
#   rung 1: arm_s (204 train)                         + holdout  -> 456 graphs
#   rung 2: arm_s + x_train_a (612 train)             + holdout  -> 864 graphs
#   rung 3: arm_s + x_train_a + x_train_b (1020 train) + holdout -> 1272 graphs
#   holdout: seeds 5001-5017 test (204), 5018-5021 val (48), FIXED across rungs
#
# Then: rsync the three cache dirs to datalab, commit experiments/route_b_fit_p2_split_r*.json,
# sbatch scripts_cosim/datalab/route_b_fit_p2_train.sbatch.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${HEROSIM_PY:-pipenv run python3}"
export PIPENV_IGNORE_VIRTUALENVS=1 VIRTUAL_ENV= PYTHONPATH="$PWD"
SD=simulation_data
ARM_S=$SD/gnn_datasets_dag4_route_b_pilot_v1_arm_s
HOLD=$SD/gnn_datasets_dag4_route_b_pilot_v1_x_holdout
TA=$SD/gnn_datasets_dag4_route_b_pilot_v1_x_train_a
TB=$SD/gnn_datasets_dag4_route_b_pilot_v1_x_train_b

count_jsonl() { ls "$1"/ds_*/placements/placements.jsonl 2>/dev/null | wc -l; }
for d in "$ARM_S:204" "$HOLD:252" "$TA:408" "$TB:408"; do
  dir=${d%%:*}; want=${d##*:}; got=$(count_jsonl "$dir")
  [[ "$got" == "$want" ]] || { echo "FAIL LOUD: $dir has $got placements.jsonl, expected $want" >&2; exit 1; }
done

build() {  # rung base_dirs...
  local r=$1; shift
  local cache=$SD/graphs_cache_route_b_fit_p2_r$r
  if [[ -f "$cache/metadata.json" ]]; then echo "[skip] $cache exists"; else
    $PY src/notebooks/prepare_graphs_cache.py --base-dirs "$@" --cache-dir "$cache" \
        --platform-feature-dim 14 --queue-feature-contract legacy_v0 --dag-partial-state \
      || { echo "FAIL LOUD: cache build r$r failed" >&2; exit 1; }
  fi
  # The trainer's val/selection sidecar. Built HERE, once, with the builder's defaults
  # (caps 256/384/256/192/0, deltas 0.05/0.30/1.00/5.00 — identical to the frozen
  # cache's valid_combos_near_rtt_capped_meta.json), never lazily by 24 concurrent
  # SLURM tasks on the rsynced cache (datalab-pitfalls #7).
  if [[ -f "$cache/valid_combos_near_rtt_capped.pkl" ]]; then echo "[skip] sidecar r$r exists"; else
    (cd src/notebooks && $PY build_capped_near_rtt_sidecar.py --cache-dir "../../$cache" --seed 42) \
      || { echo "FAIL LOUD: sidecar build r$r failed" >&2; exit 1; }
  fi
  local split=experiments/route_b_fit_p2_split_r$r.json
  if [[ -f "$split" ]]; then echo "[skip] $split exists"; else
    $PY scripts_cosim/make_split_artifact_by_block.py --cache-dir "$cache" \
        --holdout-dir gnn_datasets_dag4_route_b_pilot_v1_x_holdout \
        --test-seeds 5001-5017 --val-seeds 5018-5021 --output "$split"
  fi
}
build 1 "$ARM_S" "$HOLD"
build 2 "$ARM_S" "$TA" "$HOLD"
build 3 "$ARM_S" "$TA" "$TB" "$HOLD"

for r in 1 2 3; do
  $PY - "$SD/graphs_cache_route_b_fit_p2_r$r" "experiments/route_b_fit_p2_split_r$r.json" <<'EOF'
import json, sys
m = json.load(open(sys.argv[1] + "/metadata.json")); s = json.load(open(sys.argv[2]))
print(f"[r{sys.argv[1][-1]}] graphs={m['num_graphs']} train={len(s['train'])} val={len(s['val'])} test={len(s['test'])}")
EOF
done
echo "=== Phase 2 caches + splits built ==="
