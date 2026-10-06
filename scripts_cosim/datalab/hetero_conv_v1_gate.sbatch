#!/bin/bash
# hetero_conv_v1 gate: lf1het (8 seeds) on small_batch_confirm_v1's 19 topologies at x2/x3/x5 (docs/lineages/hetero_conv_v1.md).
# lf1gnn / lf1twin / lf1mlp come from local_features_v1's gate, CD and the other references from small_batch_confirm_v1's;
# the reader runs at the end.
#SBATCH --job-name=het-gate
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/het-gate-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=24:00:00
set -euo pipefail
WT=${WT:-/home/nikola.lukic/gnn-herosim-wt/live_headroom}
TIMEOUT=${TIMEOUT:-2700}
D=/home/nikola.lukic/gnn-herosim/simulation_data
C=$D/small_batch_confirm_v1
AG=$D/hetero_conv_v1
IN="$C/inputs"
P=${PAR:-$(( ${SLURM_CPUS_PER_TASK:-64} - 2 ))}
KINDS=lf1het
cd "$WT"
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT"
export HEROSIM_RAW_DIR=/tmp/herosim_raw_${SLURM_JOB_ID:-$$}  # node-local: 62 x ~250 MB raw results stay off the home quota
mkdir -p "$HEROSIM_RAW_DIR"; trap 'rm -rf "$HEROSIM_RAW_DIR"' EXIT
python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"
echo "HEAD $(git rev-parse HEAD)  node $(hostname)  P=$P  kinds=$KINDS"
[[ -z "$(git status --porcelain -- src scripts_cosim experiments run_experiment.py)" ]] || { echo "FAIL LOUD: worktree dirty"; exit 1; }
cmp "$WT/docs/lineages/small_batch_confirm_v1/selected.json" "$IN/selected.json"
for arm in ${KINDS//,/ }; do
  for s in 1 2 3 4 5 6 7 8; do
    for ext in .pt .contract.json; do
      f="$WT/models/hetero-conv-v1-$arm-lr2e3-seed$s$ext"
      [ -f "$f" ] || { echo "FAIL LOUD: missing $f"; exit 1; }
      cp -n "$f" "$IN/models/"; cmp "$f" "$IN/models/$(basename "$f")" || { echo "FAIL LOUD: models copy of $f differs"; exit 1; }
    done
  done
done
mkdir -p "$AG/gate"
set +e
python3 scripts_cosim/fresh_topo_burst_v1_gate.py het1 --inputs "$IN" --out "$AG/gate" \
  --selection "$IN/selected.json" --parallel "$P" --mem 4G --timeout "$TIMEOUT" --no-scope | grep -v "^\[skip"
rc=${PIPESTATUS[0]}
set -e
echo "=== het1: $(ls "$AG"/gate/*.summary.json 2>/dev/null | wc -l)/1824 summaries, $(ls "$AG"/gate/*.failed.json 2>/dev/null | wc -l) failed ==="
find "$AG/gate" -name "*.log" -size +50M -exec gzip -1 {} \;
python3 scripts_cosim/hetero_conv_v1_read.py --gate "$AG/gate" --refs "$C/gate" "$D/local_features_v1/ref_retry" "$D/local_features_v1/gate" \
  --failed "$D/local_features_v1/retry_records" "$AG/retry_records" --selection "$IN/selected.json" --out "$AG/het_read.json"
echo "driver rc=$rc"
