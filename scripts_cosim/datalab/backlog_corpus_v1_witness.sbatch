#!/bin/bash
# backlog_corpus_v1 venue/code witness (docs/lineages/backlog_corpus_v1.md, Amendment 2): at this branch's code
# the default (legacy-capture) path must reproduce, to the digit, CD s0 (cd_gap_v1 witness, = the fresh gate)
# and v4load s1 (load_repr_v1 phase fix, 807153) on cc40s9119 w1. v4load_se s1 is the smoke of the capture
# arm: it must run clean and record HEROSIM_INFLIGHT_CAPTURE=service_end_v1.
#   sbatch backlog_corpus_v1_witness.sbatch
#SBATCH --job-name=bc1-witness
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/bc1-witness-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/bc1-witness-%j.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail
WT=/home/nikola.lukic/gnn-herosim-wt/backlog_corpus
G=/home/nikola.lukic/gnn-herosim/simulation_data/cd_gap_v1/gates
OUT=/home/nikola.lukic/gnn-herosim/simulation_data/backlog_corpus_v1/gates/witness
cd "$WT"
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$WT"
python3 -c "import sys,torch;print(sys.executable, torch.__version__)"
echo "HEAD $(git -C "$WT" rev-parse HEAD)"
DIRTY="$(git -C "$WT" status --porcelain -- src scripts_cosim run_experiment.py)"
if [ -n "$DIRTY" ]; then echo "FAIL LOUD: worktree dirty: $DIRTY"; exit 1; fi
mkdir -p "$OUT"
python3 - "$WT" "$G" "$OUT" <<'PY'
import json, sys
from concurrent.futures import ThreadPoolExecutor
wt, g, out = sys.argv[1:4]
sys.path.insert(0, wt + "/scripts_cosim")
import fresh_topo_burst_v1_gate as G
G.NO_SCOPE = True
ts = [G.task(9119, "w1", "cd", 0), G.task(9119, "w1", "v4load", 1), G.task(9119, "w1", "v4load_se", 1)]
with ThreadPoolExecutor(3) as p:
    for line in p.map(lambda t: G.run_one(t, g + "/fresh", out, "3000M", 1800), ts):
        print(line, flush=True)
ref = {"cd": g + "/witness", "v4load": g + "/fix"}
bad = []
for t in ts:
    name = G.arm_name(t)
    s = json.load(open(f"{out}/{name}.summary.json"))
    line = f"{name} total_rtt={s['total_rtt']!r} capture={s['env'].get('HEROSIM_INFLIGHT_CAPTURE')!r} code={s['code']}"
    if t["kind"] in ref:
        r = json.load(open(f"{ref[t['kind']]}/{name}.summary.json"))
        line += f" | reference {r['total_rtt']!r}"
        if s["total_rtt"] != r["total_rtt"]:
            bad.append(name)
    print(line, flush=True)
if bad:
    raise SystemExit(f"FAIL LOUD: default path not bit-identical at this code: {bad}")
print("[witness] default path reproduces CD and v4load to the digit", flush=True)
PY
