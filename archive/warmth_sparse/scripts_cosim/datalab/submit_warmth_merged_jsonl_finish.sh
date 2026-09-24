#!/usr/bin/env bash
# Submit merged warmth+sparse JSONL BF finish (training corpus backfill).
set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

PARTITION="${PARTITION:-GPU-a100}"
CPUS="${CPUS:-64}"

for f in \
  scripts_cosim/datalab/run_warmth_merged_jsonl_finish.sh \
  scripts_cosim/datalab/warmth_merged_jsonl_finish.sbatch \
  scripts_cosim/recover_placements_jsonl_from_scratch.sh \
  scripts_cosim/generate_gnn_datasets_fast.py; do
  if [[ ! -f "$f" ]]; then
    echo "ERROR: missing $f" >&2
    exit 1
  fi
done

sed -i 's/\r$//' \
  scripts_cosim/datalab/run_warmth_merged_jsonl_finish.sh \
  scripts_cosim/datalab/warmth_merged_jsonl_finish.sbatch \
  scripts_cosim/recover_placements_jsonl_from_scratch.sh

echo "=== Merged corpus audit (before) ==="
python3 - <<'PY'
from pathlib import Path

def audit(label, base):
    base = Path(base)
    both = sum(
        1 for ds in base.glob("ds_*")
        if (ds / "best.json").exists()
        and (ds / "placements/placements.jsonl").exists()
        and (ds / "placements/placements.jsonl").stat().st_size > 0
    )
    total = len(list(base.glob("ds_*")))
    print(f"{label}: {both}/{total} complete (best+jsonl)")

audit("warmth_v2", "simulation_data/gnn_datasets_4tasks_1060_warmth_v2")
audit("sparse_v2", "simulation_data/gnn_datasets_4tasks_sparse_warmth_v2")
PY

echo
echo "=== Cancel stale merged-jsonl jobs ==="
for j in $(squeue -u "$USER" -h -o "%i %j" | awk '/merged-jsonl/ {print $1}'); do
  echo "scancel ${j}"
  scancel "${j}" || true
done
sleep 2

rm -rf data/nofs-ids/traces/gnn_templates data/nofs-ids/traces/gnn_templates_* 2>/dev/null || true

job_id=$(sbatch --partition="${PARTITION}" --cpus-per-task="${CPUS}" \
  scripts_cosim/datalab/warmth_merged_jsonl_finish.sbatch | awk '{print $NF}')

echo ""
echo "Submitted merged JSONL finish: ${job_id}"
echo "MERGED_JSONL_JOB_ID=${job_id}"
squeue -u "$USER" | grep -E "merged-jsonl|JOBID" || true
