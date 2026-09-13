#!/usr/bin/env bash
# Submit warmth_v2 tail finisher + JSONL BF repair (parallel, non-conflicting).
set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

PARTITION="${PARTITION:-GPU-a100}"
CPUS="${CPUS:-64}"

for f in \
  scripts_cosim/datalab/run_warmth_regen_range.sh \
  scripts_cosim/datalab/warmth_regen_tail.sbatch \
  scripts_cosim/datalab/warmth_jsonl_bf_repair.sbatch \
  scripts_cosim/recover_placements_jsonl_from_scratch.sh \
  scripts_cosim/generate_gnn_datasets_fast.py; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing $f" >&2
    exit 1
  fi
done

sed -i 's/\r$//' \
  scripts_cosim/datalab/run_warmth_regen_range.sh \
  scripts_cosim/datalab/warmth_regen_tail.sbatch \
  scripts_cosim/datalab/warmth_jsonl_bf_repair.sbatch \
  scripts_cosim/recover_placements_jsonl_from_scratch.sh

BASE="simulation_data/gnn_datasets_4tasks_1060_warmth_v2"
echo "=== Before submit ==="
echo -n "best.json: "
find "${BASE}" -name best.json 2>/dev/null | wc -l
echo -n "placements.jsonl (non-empty): "
find "${BASE}" -path '*/placements/placements.jsonl' -size +0c 2>/dev/null | wc -l
echo "Missing JSONL (has best):"
python3 - <<'PY'
from pathlib import Path
base = Path("simulation_data/gnn_datasets_4tasks_1060_warmth_v2")
missing = []
for ds in sorted(base.glob("ds_*")):
    best = ds / "best.json"
    jsonl = ds / "placements" / "placements.jsonl"
    if best.exists() and (not jsonl.exists() or jsonl.stat().st_size == 0):
        missing.append(ds.name)
print("  " + ", ".join(missing) if missing else "  (none)")
print(f"  count={len(missing)}")
PY

echo
echo "=== Remove stale shared gnn_templates (parallel race fix) ==="
rm -rf data/nofs-ids/traces/gnn_templates data/nofs-ids/traces/gnn_templates_* 2>/dev/null || true

pick_partition() {
  local part="$1"
  echo "--- test-only ${part} ---"
  if out=$(sbatch --test-only --partition="$part" --cpus-per-task="${CPUS}" \
    scripts_cosim/datalab/warmth_regen_tail.sbatch 2>&1); then
    echo "$out"
    if echo "$out" | grep -qiE 'PENDING|ReqNodeNotAvail|QOSMax|AssocMaxJobsLimit|Resources'; then
      return 1
    fi
    SELECTED_PARTITION="$part"
    return 0
  fi
  echo "$out" >&2
  return 1
}

SELECTED_PARTITION=""
if ! pick_partition "${PARTITION}"; then
  if ! pick_partition "GPU-a100s"; then
    echo "ERROR: no partition can schedule warmth finish jobs now" >&2
    exit 1
  fi
fi
echo "Using partition: ${SELECTED_PARTITION}"

tail_id=$(sbatch --partition="${SELECTED_PARTITION}" --cpus-per-task="${CPUS}" \
  scripts_cosim/datalab/warmth_regen_tail.sbatch | awk '{print $NF}')
jsonl_id=$(sbatch --partition="${SELECTED_PARTITION}" --cpus-per-task="${CPUS}" \
  scripts_cosim/datalab/warmth_jsonl_bf_repair.sbatch | awk '{print $NF}')

echo ""
echo "Submitted warmth tail finisher: ${tail_id} (ds_00488..499)"
echo "Submitted JSONL BF repair:    ${jsonl_id} (--only-missing-jsonl, scan 500)"
echo "TAIL_JOB_ID=${tail_id}"
echo "JSONL_JOB_ID=${jsonl_id}"

echo ""
squeue -u "$(whoami)" -o "%.10i %.12P %.16j %.2t %R" | grep -E "warmth|JOBID" || true
