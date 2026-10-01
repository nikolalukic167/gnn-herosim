#!/bin/bash
# raw_plan_v2 serving smoke + train/serve parity (docs/lineages/raw_plan_v2.md, Steps 3-4).
# The four 2-epoch smoke checkpoints are served through the gate's own run_one on 9420 g0x20, with the raw-plan dump
# instrument on. Then scripts_cosim/raw_plan_v2_parity.py checks, per arm:
#   cache  - serving vs trainer model on cached graphs, decode + self-refine committed sets, bitwise;
#   replay - the live dump re-scored by the trainer model, bitwise.
# Its summaries are never read as results.
#   sbatch --export=ALL,WT=<grounded worktree> raw_plan_v2_servesmoke.sbatch
#SBATCH --job-name=rp2-serve
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/rp2-serve-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
set -euo pipefail
WT=${WT:-/home/nikola.lukic/gnn-herosim-wt/grounded}
IN=/home/nikola.lukic/gnn-herosim/simulation_data/backlog_corpus_v1/gates/inputs
S=/home/nikola.lukic/gnn-herosim/simulation_data/backlog_corpus_v1/gates/rp2smoke
CACHE="$WT/simulation_data/graphs_cache_backlog_corpus_v1_psv4_inf"
ARMS=(rawE rawS rawES rawStwin)
cd "$WT"
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT" OMP_NUM_THREADS=1
echo "HEAD $(git rev-parse HEAD)"
[[ -z "$(git status --porcelain -- src scripts_cosim experiments run_experiment.py)" ]] || { echo "FAIL LOUD: worktree dirty"; exit 1; }
rm -rf "$S"; mkdir -p "$S/in/grounded_x20/models" "$S/out" "$S/dump"
for d in cfg wl; do ln -s "$IN/grounded_x20/$d" "$S/in/grounded_x20/$d"; done
ln -s "$IN/backlog_corpus_v1_split.json" "$S/in/grounded_x20/backlog_corpus_v1_split.json"
ln -s "$IN/backlog_corpus_v1_split.json" "$S/in/backlog_corpus_v1_split.json"
for A in "${ARMS[@]}"; do
  for ext in .pt .contract.json; do
    cp "$WT/models/raw-plan-v2-$A-smoke-lr2e3-seed1$ext" "$S/in/grounded_x20/models/raw-plan-v2-$A-lr2e3-seed1$ext"
  done
done

for A in "${ARMS[@]}"; do
  GNN_PLAN_RAW_DUMP="$S/dump/$A" GNN_PLAN_RAW_DUMP_N=150 python3 - "$S" "$A" <<'PY'
import sys
sys.path.insert(0, "scripts_cosim")
import fresh_topo_burst_v1_gate as G
G.NO_SCOPE = True
S, arm = sys.argv[1], sys.argv[2]
print(G.run_one(G.task(9420, "g0x20", f"{arm}_selfref", 1), f"{S}/in", f"{S}/out", "6G", 2700), flush=True)
PY
done
ls "$S/out"

for A in "${ARMS[@]}"; do
  CK="$S/in/grounded_x20/models/raw-plan-v2-$A-lr2e3-seed1.pt"
  CFG="experiments/raw_plan_v2_${A}_smoke.yaml"
  ( unset GNN_DISABLE_MESSAGE_PASSING; python3 scripts_cosim/raw_plan_v2_parity.py cache --ckpt "$CK" --config "$CFG" --cache "$CACHE" --n-graphs 20 )
  ( unset GNN_DISABLE_MESSAGE_PASSING; python3 scripts_cosim/raw_plan_v2_parity.py replay --ckpt "$CK" --config "$CFG" --dump "$S/dump/$A" )
done
echo "=== rp2 servesmoke + parity: all arms ok ==="
