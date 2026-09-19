#!/bin/bash
# link_mp_v1 — final unattended stage: verify -> extract -> score.
# Registration: docs/lineages/link_mp_v1.md (Amendment 2 appended this stage).
#
# Runs at HEAD in the MAIN checkout on purpose: extraction and scoring are analysis
# tools, and the scorer's own PIN constants (score_link_mp_v1.py) protect arm identity —
# any arm whose recorded provenance is not the pinned commit VOIDs regardless of where
# this stage runs.
#
# Submit with:  --dependency=afterok:<linkmp-gate array job id>
#SBATCH --job-name=linkmp-score
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/linkmp-score-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/linkmp-score-%j.err
#SBATCH --partition=CPU-amd
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export PYTHONPATH="$PWD" PYTHONUNBUFFERED=1

echo "=== Step 1: every lg result must parse END-TO-END (prefix validity is not document validity) ==="
python3 - <<'PY'
import json, glob, sys
bad, n = [], 0
for p in glob.glob("simulation_data/normal_sim_sweeps/*_lg*/results/*_s0_gnn.json"):
    n += 1
    try:
        json.load(open(p))
    except Exception as e:
        bad.append((p, f"{type(e).__name__}: {str(e)[:60]}"))
print(f"parsed end-to-end: {n - len(bad)}/{n}")
if n < 960:
    raise SystemExit(f"FAIL LOUD: only {n} lg results on disk, expected 960 — do not score")
if bad:
    for p, why in bad[:10]:
        print(f"  BAD {why}\n      {p}")
    raise SystemExit(f"FAIL LOUD: {len(bad)} truncated/unparseable results — delete and re-run them")
PY

echo "=== Step 2: rebuild gate_stats_summary.json (all arms, gz-transparent) ==="
python3 -u scripts_cosim/important/extract_gate_stats_summary.py \
  --out simulation_data/gate_stats_summary.json

echo "=== Step 3: registered scorer ==="
python3 -u scripts_cosim/important/score_link_mp_v1.py \
  --summary simulation_data/gate_stats_summary.json \
  --json-out simulation_data/link_mp_v1_verdict.json
echo "=== score stage complete — verdict in simulation_data/link_mp_v1_verdict.json ==="
