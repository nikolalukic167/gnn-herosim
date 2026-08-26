#!/bin/bash
#SBATCH --job-name=rc8t-score
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/route-c-8task-score-%A.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/route-c-8task-score-%A.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# route_c_link_transfer_v1 SCREEN — score the 8-task contingency rung (regen job
# 714713; the first attempt, 714278, died on a home-quota exhaustion that poisoned
# every shard including the "COMPLETED" ones — placement_errors.log carried
# Errno 122 tracebacks, so the whole 24-dataset corpus was regenerated from scratch).
#
# Order per the registration: completeness gate, then the registered manipulation
# check (rung VALID iff median per-dataset sum(link_wait)/sum(rtt) >= 0.10), then the
# scorer. The scorer is cheap to run in the same job even if the rung reads INVALID;
# its report is only *read* in full if the rung is VALID.
#
# ALPHAS 4.0,12.0 — 4.0 is the §9d equal-tightness primary for 8 tasks (double the
# 4-task TIGHT_ALPHA, reasoned in route_b_8task_score.sbatch; do not "fix" it back);
# 12.0 is the near-unconstrained cross-check. --include-per-dataset is MANDATORY:
# route_b_coefficient_transfer.py reads report["per_dataset"][str(alpha)].
#
# Compute node, not login: streams ~20 GB of placements.jsonl.

set -euo pipefail
PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PROJECT_ROOT"

CORPUS=simulation_data/gnn_datasets_route_c_link_screen_8task

# --- Completeness gate (fail loud before reading anything) ---
n_ds=$(ls -d "$CORPUS"/ds_* 2>/dev/null | wc -l)
n_jsonl=$(find "$CORPUS" -name placements.jsonl -size +0 | wc -l)
n_sweep_ok=$(python3 - <<'EOF'
import json, pathlib
ok = 0
for d in sorted(pathlib.Path("simulation_data/gnn_datasets_route_c_link_screen_8task").glob("ds_*")):
    p = d / "placement_metadata.json"
    if not p.exists():
        print(f"MISSING metadata: {p}")
    elif not json.loads(p.read_text()).get("sweep_complete", False):
        print(f"NOT sweep_complete: {p}")
    else:
        ok += 1
print(ok)
EOF
)
n_sweep_ok=$(echo "$n_sweep_ok" | tail -1)
n_errlogs=$(find "$CORPUS" -name placement_errors.log -size +0 | wc -l)
echo "completeness: datasets=$n_ds/24 jsonl=$n_jsonl/24 sweep_complete=$n_sweep_ok/24 nonempty_error_logs=$n_errlogs"
if [ "$n_ds" -ne 24 ] || [ "$n_jsonl" -ne 24 ] || [ "$n_sweep_ok" -ne 24 ] || [ "$n_errlogs" -ne 0 ]; then
  echo "COMPLETENESS GATE FAILED — do not score this corpus" >&2
  exit 1
fi

# --- Registered manipulation check ---
python3 -u scripts_cosim/route_c_screen_manipulation_check.py \
  --corpus "$CORPUS" \
  --out simulation_data/route_c_screen_8task_manipulation.json

# --- Scorer (report read in full only if the rung is VALID) ---
python3 -u scripts_cosim/score_route_b_contention.py \
  --corpus "$CORPUS" \
  --objective rtt --alphas 4.0,12.0 --include-per-dataset \
  --out simulation_data/route_c_screen_8task_rtt.json
