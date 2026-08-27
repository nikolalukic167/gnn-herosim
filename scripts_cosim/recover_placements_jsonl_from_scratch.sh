#!/usr/bin/env bash
# Copy ds_*/.bf_scratch/placements.jsonl → placements/placements.jsonl when scratch
# survived a crashed BF run. Does not recreate missing scratch — re-run BF for those.
# See docs/notes/placements_jsonl_required.md
set -euo pipefail

BASE_DIR="${1:-simulation_data/gnn_datasets_4tasks_1060_warmth_v2}"
cd "$(dirname "$0")/.."

copied=0
skipped=0
for scratch in "$BASE_DIR"/ds_*/.bf_scratch/placements.jsonl; do
  [[ -f "$scratch" ]] || continue
  ds_dir="$(dirname "$(dirname "$scratch")")"
  pub="$ds_dir/placements/placements.jsonl"
  if [[ -f "$pub" && "$(stat -c%s "$pub")" -gt 0 ]]; then
    skipped=$((skipped + 1))
    continue
  fi
  mkdir -p "$ds_dir/placements"
  cp -f "$scratch" "$pub"
  echo "copied $(basename "$ds_dir")"
  copied=$((copied + 1))
done

echo "=== Done: copied=$copied already_had_jsonl=$skipped ==="
