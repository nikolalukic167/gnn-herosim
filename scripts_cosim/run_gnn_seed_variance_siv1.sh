#!/usr/bin/env bash
# Separate "scale_invariant_v1 features hurt the GNN" from "seed 42 was a bad draw".
#
# The seed-42 smoke showed GNN siv1 live-regressing below GNN v5.5 while the MLP improved
# sharply, but the siv1 GNN is also a fresh checkpoint that scored worse offline
# (test 55.7% vs 60.3%, greedy regret 0.911s vs 0.848s). Weight-init variance and the
# feature change are therefore confounded.
#
# Design: hold the canonical-parent split fixed (its random_state stays 42 inside the
# trainer) and vary only NEAR_RTT_TRAIN_SEED, on both caches. Seed 42 already exists for
# both, so two extra seeds per contract gives n=3 each.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SIV1_CACHE="simulation_data/graphs_cache_contention_v2_873_v5.7_siv1_dim14"
LEGACY_CACHE="simulation_data/graphs_cache_contention_v2_873_v5.5"
PROJECT="${WANDB_PROJECT:-gnn-queue-scale-invariant-aug2026}"
SEEDS="${SEEDS:-43 44}"

for d in "$SIV1_CACHE" "$LEGACY_CACHE"; do
  [[ -d "$d" ]] || { echo "ERROR: missing cache $d" >&2; exit 1; }
done
mkdir -p logs

run_one() {
  local tag="$1" cache="$2" seed="$3"
  local name="near-rtt-v2-cv2-873-${tag}-dim14-ce-only-seed${seed}"
  local log="logs/train_gnn_${tag}_seed${seed}.log"
  if [[ -f "models/${name}.pt" ]]; then
    echo "SKIP ${name} (exists)"
    return 0
  fi
  echo "=== ${name} (cache=${cache}) ==="
  NEAR_RTT_CACHE_DIR="$cache" \
  NEAR_RTT_TRAIN_SEED="$seed" \
  WANDB_PROJECT="$PROJECT" \
  WANDB_RUN_NAME="$name" \
  WANDB_TAGS="near-rtt,ce-only,dim14,contention-v2,seed-variance,${tag}" \
    pipenv run python3 src/notebooks/train_near_rtt_v2_contention_v2_dim14_ce_only.py \
    >"$log" 2>&1
  echo "done -> models/${name}.pt (log ${log})"
}

for seed in $SEEDS; do
  run_one siv1 "$SIV1_CACHE" "$seed"
done
for seed in $SEEDS; do
  run_one legacy "$LEGACY_CACHE" "$seed"
done

echo "=== all seed-variance runs complete ==="
