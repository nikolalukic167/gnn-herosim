#!/bin/bash
# local_features_v1 Amendment 2(a) retry: every run that timed out at 2700 s is rerun once at TIMEOUT (default 8100 s).
#   lf1 arms: in the gate dir, each *.failed.json is renamed *.failed-t2700.json (kept) and lf1conf reruns those tasks.
#   references: the rules part of small_batch_confirm_v1's gate (CD, cdextr, reactive, xs1load s1-4) is re-run into
#   $LF/ref_retry, seeded with symlinks to that gate's existing summaries so only its missing runs execute; that
#   lineage's own directory is not touched.
#SBATCH --job-name=lf1-retry
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/lf1-retry-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=24:00:00
set -euo pipefail
WT=${WT:-/home/nikola.lukic/gnn-herosim-wt/live_headroom}
TIMEOUT=${TIMEOUT:-8100}
C=/home/nikola.lukic/gnn-herosim/simulation_data/small_batch_confirm_v1
LF=/home/nikola.lukic/gnn-herosim/simulation_data/local_features_v1
IN="$C/inputs"
R="$LF/ref_retry"
cd "$WT"
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT"
python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"
echo "HEAD $(git rev-parse HEAD)  node $(hostname)  TIMEOUT=$TIMEOUT"
[[ -z "$(git status --porcelain -- src scripts_cosim experiments run_experiment.py)" ]] || { echo "FAIL LOUD: worktree dirty"; exit 1; }

mkdir -p "$R"
n=0
for f in "$C"/gate/*__{cd,cdextr,reactive}_s0.summary.json "$C"/gate/*__xs1load_selfref_s[1-4].summary.json; do
  [ -e "$R/$(basename "$f")" ] || { ln -s "$f" "$R/"; n=$((n + 1)); }
done
echo "[ref_retry] linked $n reference summaries"
for f in "$LF"/gate/*.failed.json; do
  [ -e "$f" ] || continue
  mv "$f" "${f%.failed.json}.failed-t2700.json"
done
echo "[gate] $(ls "$LF"/gate/*.failed-t2700.json 2>/dev/null | wc -l) lf1 runs to rerun"

set +e
SBCONF_KINDS=rules python3 scripts_cosim/fresh_topo_burst_v1_gate.py sbconf --inputs "$IN" --out "$R" \
  --selection "$IN/selected.json" --parallel 30 --mem 4G --timeout "$TIMEOUT" --no-scope &
LF1_KINDS_RUN=lf1gnn,lf1twin,lf1mlp python3 scripts_cosim/fresh_topo_burst_v1_gate.py lf1conf --inputs "$IN" --out "$LF/gate" \
  --selection "$IN/selected.json" --parallel 30 --mem 4G --timeout "$TIMEOUT" --no-scope &
wait %1; rc1=$?
wait %2; rc2=$?
set -e
echo "=== still failed: refs $(ls "$R"/*.failed.json 2>/dev/null | wc -l), lf1 $(ls "$LF"/gate/*.failed.json 2>/dev/null | wc -l) ==="
echo "driver rc=$rc1/$rc2"
