#!/bin/bash
# raw_plan_v2 Phase C screen witness (diagnostic; changes no rule). The Phase C screen admitted 0 of 48 because
# reactive / batched runs hung, against 8 of 48 for fresh_topo_burst_v1's 9425-9472 extension. This reruns the
# admission screen, with Phase C's settings (this code, this venue, --no-scope, 600 s), on four topologies that
# fresh_topo_burst_v1 admitted: 9434 9435 9444 9446. If they hang here, the hangs come from code or venue, not from
# the new topologies.
#   sbatch --export=ALL,WT=<grounded worktree> raw_plan_v2_screen_witness.sbatch
#SBATCH --job-name=rp2-wit
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/rp2-wit-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=120G
#SBATCH --time=02:00:00
set -euo pipefail
WT=${WT:-/home/nikola.lukic/gnn-herosim-wt/grounded}
FRESH=/home/nikola.lukic/gnn-herosim/simulation_data/cd_gap_v1/gates/fresh
O=/home/nikola.lukic/gnn-herosim/simulation_data/raw_plan_v2/confirm/witness_screen
cd "$WT"
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT"
echo "HEAD $(git rev-parse HEAD)  node $(hostname)"
[[ -z "$(git status --porcelain -- src scripts_cosim experiments run_experiment.py)" ]] || { echo "FAIL LOUD: worktree dirty"; exit 1; }
rm -rf "$O"; mkdir -p "$O"
python3 - "$FRESH" "$O" <<'PY'
import sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "scripts_cosim")
import fresh_topo_burst_v1_gate as G
G.NO_SCOPE = True
fresh, out = sys.argv[1], sys.argv[2]
todo = [G.task(t, w, k) for k in ("reactive", "batched") for t in (9434, 9435, 9444, 9446) for w in G.WINDOWS]
with ThreadPoolExecutor(max_workers=32) as pool:
    for line in pool.map(lambda t: G.run_one(t, fresh, out, "4G", 600), todo):
        print(line, flush=True)
PY
echo "=== witness done ==="
