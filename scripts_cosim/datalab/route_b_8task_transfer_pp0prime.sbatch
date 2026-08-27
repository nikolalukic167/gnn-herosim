#!/bin/bash
#SBATCH --job-name=rb8t-transfer-pp0p
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/route-b-8task-transfer-pp0prime-%A.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/route-b-8task-transfer-pp0prime-%A.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00

# PP0' (docs/lineages/route_b_v1/stage2-preregistration.md §10, corrected registration) — step 1 of 2 on
# the 8-task corpus: re-run route_b_coefficient_transfer.py with the PP0' report
# extension (per-dataset krank fractions + tie bands, additive fields only), then
# assert BIT-IDENTITY of every pre-existing key against the frozen §9d artifact
# simulation_data/route_b_8task_coefficient_transfer.json. A changed pre-existing
# value is a hard failure — the frozen §9d numbers are the reference, and this job
# exists to attach per-dataset detail to them, not to move them.
#
# Step 2 (route_b_8task_krank_verify.sbatch) is submitted with
# --dependency=afterok:<this job id>.
#
# Budget: the original 8-task transfer ran 8-13 h single-process (see
# route_b_8task_transfer.sbatch); the extension adds only report assembly.

set -euo pipefail
PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PROJECT_ROOT"

python3 -u scripts_cosim/route_b_coefficient_transfer.py \
  --corpus simulation_data/gnn_datasets_dag4_route_b_pilot_v1_8task \
  --report simulation_data/route_b_8task_rtt.json --alpha 4.0 \
  --out simulation_data/route_b_8task_coefficient_transfer_pp0prime.json

python3 - <<'EOF'
import json, sys

frozen = json.load(open("simulation_data/route_b_8task_coefficient_transfer.json"))
new = json.load(open("simulation_data/route_b_8task_coefficient_transfer_pp0prime.json"))

removed, changed, added = [], [], []
def walk(a, b, path):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in a:
            if k not in b: removed.append(path + [k])
            else: walk(a[k], b[k], path + [k])
        for k in b:
            if k not in a: added.append(path + [k])
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b): changed.append((path, f"len {len(a)} vs {len(b)}")); return
        for i, (x, y) in enumerate(zip(a, b)): walk(x, y, path + [i])
    else:
        if a != b: changed.append((path, (a, b)))

walk(frozen, new, [])
if removed or changed:
    print("!! PP0' bit-identity FAILED against the frozen §9d artifact")
    for p in removed[:20]: print("  removed:", p)
    for p, v in changed[:20]: print("  changed:", p, v)
    sys.exit(1)
groups = sorted({tuple(p[:2]) for p in added})
allowed = {"krank_exploratory", "krank_pooled_exploratory"}
bad = [g for g in groups if g[0] not in allowed]
if bad:
    print("!! PP0' added keys outside the krank blocks:", bad)
    sys.exit(1)
print(f"OK: frozen §9d artifact reproduced bit-identically; "
      f"{len(added)} leaf keys added, all inside {sorted(allowed)}")
EOF
