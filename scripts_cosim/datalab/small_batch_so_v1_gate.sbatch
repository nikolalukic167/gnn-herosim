#!/bin/bash
# small_batch_so_v1 gate: so1load / so1mpoff (seeds 1-4) on the single-origin server-only cells of client_local_v1
# (inputs_so_server: small_batch_confirm_v1's 19 topologies x g0-g3 x {x2, x3, x5}). CD, the other rules and the
# zero-shot sb1load / sb1mpoff are read from client_local_v1/gate_so_server; the reader runs at the end.
#SBATCH --job-name=so1-gate
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/so1-gate-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=24:00:00
set -euo pipefail
WT=${WT:-/home/nikola.lukic/gnn-herosim-wt/sbso}
TIMEOUT=${TIMEOUT:-2700}
D=/home/nikola.lukic/gnn-herosim/simulation_data
IN=$D/client_local_v1/inputs_so_server
OUT=$D/small_batch_so_v1/gate
SPLIT=experiments/small_batch_so_v1_split.json
P=${PAR:-$(( ${SLURM_CPUS_PER_TASK:-64} - 2 ))}
KINDS=${SO1_KINDS_RUN:-so1load,so1mpoff}  # Amendment 1: so1lfgnn,so1lfmlp
cd "$WT"
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT"
export HEROSIM_RAW_DIR=/tmp/herosim_raw_${SLURM_JOB_ID:-$$}  # node-local: 62 x ~250 MB raw results stay off the home quota
mkdir -p "$HEROSIM_RAW_DIR"; trap 'rm -rf "$HEROSIM_RAW_DIR"' EXIT
python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"
echo "HEAD $(git rev-parse HEAD)  node $(hostname)  P=$P"
[[ -z "$(git status --porcelain -- src scripts_cosim experiments run_experiment.py)" ]] || { echo "FAIL LOUD: worktree dirty"; exit 1; }
for R in x20 x30 x50; do
  [[ -f "$IN/grounded_$R/.built" ]] || { echo "FAIL LOUD: $IN/grounded_$R not built"; exit 1; }
  cp -n "$SPLIT" "$IN/grounded_$R/"; cmp "$SPLIT" "$IN/grounded_$R/$(basename "$SPLIT")"
  M=$(readlink -f "$IN/grounded_$R/models")
  for arm in ${KINDS//,/ }; do
    for s in 1 2 3 4; do
      for ext in .pt .contract.json; do
        f="$WT/models/small-batch-so-v1-$arm-lr2e3-seed$s$ext"
        [ -f "$f" ] || { echo "FAIL LOUD: missing $f"; exit 1; }
        cp -n "$f" "$M/"; cmp "$f" "$M/$(basename "$f")" || { echo "FAIL LOUD: models copy of $f differs"; exit 1; }
      done
    done
  done
done
mkdir -p "$OUT"
set +e
SO1_KINDS_RUN=$KINDS python3 scripts_cosim/fresh_topo_burst_v1_gate.py so1 --inputs "$IN" --out "$OUT" \
  --selection "$IN/selected.json" --parallel "$P" --mem 4G --timeout "$TIMEOUT" --no-scope | grep -v "^\[skip"
rc=${PIPESTATUS[0]}
set -e
echo "=== so1 ($KINDS): $(ls "$OUT"/*.summary.json 2>/dev/null | wc -l) summaries, $(ls "$OUT"/*.failed.json 2>/dev/null | wc -l) failed ==="
find "$OUT" -name "*.log" -size +50M -exec gzip -1 {} \;
python3 scripts_cosim/small_batch_so_v1_read.py --gate "$OUT" --refs "$D/client_local_v1/gate_so_server" \
  --failed "$D/small_batch_so_v1/retry_records" --selection "$IN/selected.json" --out "$D/small_batch_so_v1/so1_read.json"
echo "driver rc=$rc"
