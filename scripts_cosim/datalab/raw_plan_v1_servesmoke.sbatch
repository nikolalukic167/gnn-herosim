#!/bin/bash
# raw_plan_v1 serving smoke: the 2-epoch smoke checkpoints of rawgnn / rawmlp served through the gate's own run_one
# (same env, sidecheck and instrument checks) on one x2 cell, 9420 g0x20. Proves the live raw-plan path before the
# real checkpoints exist; its summaries are never read as results.
#   sbatch --export=ALL,WT=<grounded worktree> raw_plan_v1_servesmoke.sbatch
#SBATCH --job-name=rp1-serve
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/rp1-serve-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
set -euo pipefail
WT=${WT:-/home/nikola.lukic/gnn-herosim-wt/grounded}
IN=/home/nikola.lukic/gnn-herosim/simulation_data/backlog_corpus_v1/gates/inputs
S=/home/nikola.lukic/gnn-herosim/simulation_data/backlog_corpus_v1/gates/rawsmoke
cd "$WT"
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT"
[[ -z "$(git status --porcelain -- src scripts_cosim experiments run_experiment.py)" ]] || { echo "FAIL LOUD: worktree dirty"; exit 1; }
rm -rf "$S"; mkdir -p "$S/in/grounded_x20/models" "$S/out"
for d in cfg wl; do ln -s "$IN/grounded_x20/$d" "$S/in/grounded_x20/$d"; done
ln -s "$IN/backlog_corpus_v1_split.json" "$S/in/grounded_x20/backlog_corpus_v1_split.json"
ln -s "$IN/backlog_corpus_v1_split.json" "$S/in/backlog_corpus_v1_split.json"
for A in rawgnn rawmlp; do
  for ext in .pt .contract.json; do
    cp "$WT/models/raw-plan-v1-$A-smoke-lr2e3-seed1$ext" "$S/in/grounded_x20/models/raw-plan-v1-$A-lr2e3-seed1$ext"
  done
done
python3 - "$S" <<'PY'
import sys
sys.path.insert(0, "scripts_cosim")
import fresh_topo_burst_v1_gate as G
G.NO_SCOPE = True
S = sys.argv[1]
for kind in ("rawgnn_selfref", "rawmlp_selfref"):
    print(G.run_one(G.task(9420, "g0x20", kind, 1), f"{S}/in", f"{S}/out", "4G", 2700), flush=True)
PY
ls "$S/out"
