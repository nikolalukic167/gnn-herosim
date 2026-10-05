#!/bin/bash
# rule_baselines_v1 gate (docs/lineages/rule_baselines_v1.md): six hand rules (random, least-loaded, locality-first,
# Decima's tuned weighted fair, one-pass greedy, self-predict) on small_batch_confirm_v1's 19 topologies at x2/x3/x5.
# lf1gnn / lf1mlp are read from local_features_v1's gate directory; the reader runs at the end.
#SBATCH --job-name=rb1-gate
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/rb1-gate-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=24:00:00
set -euo pipefail
WT=${WT:-/home/nikola.lukic/gnn-herosim-wt/live_headroom}
TIMEOUT=${TIMEOUT:-2700}
C=/home/nikola.lukic/gnn-herosim/simulation_data/small_batch_confirm_v1
LF=/home/nikola.lukic/gnn-herosim/simulation_data/local_features_v1
RB=/home/nikola.lukic/gnn-herosim/simulation_data/rule_baselines_v1
IN="$C/inputs"
P=${PAR:-$(( ${SLURM_CPUS_PER_TASK:-64} - 2 ))}
cd "$WT"
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT" HEROSIM_DECIMA_ALPHA=1.0
python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"
echo "HEAD $(git rev-parse HEAD)  node $(hostname)  P=$P  kinds=${RB1_KINDS_RUN:-all}"
[[ -z "$(git status --porcelain -- src scripts_cosim experiments run_experiment.py)" ]] || { echo "FAIL LOUD: worktree dirty"; exit 1; }
cmp "$WT/docs/lineages/small_batch_confirm_v1/selected.json" "$IN/selected.json"
mkdir -p "$RB/gate"
set +e
python3 scripts_cosim/fresh_topo_burst_v1_gate.py rb1 --inputs "$IN" --out "$RB/gate" --selection "$IN/selected.json" \
  --parallel "$P" --mem 4G --timeout "$TIMEOUT" --no-scope | grep -v "^\[skip"
rc=${PIPESTATUS[0]}
set -e
echo "=== rb1: $(ls "$RB"/gate/*.summary.json 2>/dev/null | wc -l)/1368 summaries, $(ls "$RB"/gate/*.failed.json 2>/dev/null | wc -l) failed ==="
find "$RB/gate" -name "*.log" -size +50M -exec gzip -1 {} \;
python3 scripts_cosim/rule_baselines_v1_read.py --gate "$RB/gate" --lf1 "$LF/gate" --selection "$IN/selected.json" \
  --out "$RB/rb1_read.json"
echo "driver rc=$rc"
