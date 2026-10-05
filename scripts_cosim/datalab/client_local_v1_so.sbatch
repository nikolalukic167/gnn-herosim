#!/bin/bash
# client_local_v1 part B (single origin): the grounded workloads with every peer group's tasks on one client (the
# client of its earliest task; client_local_v1_single_origin.py), on the same 19 topologies and x2/x3/x5 windows.
# Two input trees share the rewritten workloads: inputs_so_server (the server-only cells of small_batch_confirm_v1)
# and inputs_so_client (the client-enabled cells of client_local_v1).
#   BUILD=1                     build both trees and exit
#   SO_VARIANT=server|client    which tree to run (server: the five server-only rules; client: all seven)
#SBATCH --job-name=cl1-so
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/cl1-so-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=24:00:00
set -euo pipefail
WT=${WT:-/home/nikola.lukic/gnn-herosim-wt/single_origin}
TIMEOUT=${TIMEOUT:-2700}
D=/home/nikola.lukic/gnn-herosim/simulation_data
SRC=$D/small_batch_confirm_v1/inputs
CL=$D/client_local_v1
P=${PAR:-$(( ${SLURM_CPUS_PER_TASK:-64} - 2 ))}
cd "$WT"
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT"
export HEROSIM_RAW_DIR=/tmp/herosim_raw_${SLURM_JOB_ID:-$$}  # node-local: 62 x ~250 MB raw results stay off the home quota
mkdir -p "$HEROSIM_RAW_DIR"; trap 'rm -rf "$HEROSIM_RAW_DIR"' EXIT
python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"
echo "HEAD $(git rev-parse HEAD)  node $(hostname)  P=$P  BUILD=${BUILD:-0} variant=${SO_VARIANT:-}"
[[ -z "$(git status --porcelain -- src scripts_cosim experiments run_experiment.py)" ]] || { echo "FAIL LOUD: worktree dirty"; exit 1; }

if [[ ${BUILD:-0} == 1 ]]; then
  for R in x20 x30 x50; do [[ -f "$CL/inputs/grounded_$R/.built" ]] || { echo "FAIL LOUD: client_local_v1 inputs $R not built"; exit 1; }; done
  for V in server client; do
    IN=$CL/inputs_so_$V; mkdir -p "$IN"
    cp -n "$SRC/selected.json" "$IN/selected.json"; cmp "$SRC/selected.json" "$IN/selected.json"
    for R in x20 x30 x50; do
      T="$IN/grounded_$R"
      [[ -f "$T/.built" ]] && continue
      rm -rf "$T"; mkdir -p "$T"
      C=$([[ $V == server ]] && echo "$SRC/grounded_$R" || echo "$CL/inputs/grounded_$R")
      ln -s "$(readlink -f "$C/cfg")" "$T/cfg"
      for f in models backlog_corpus_v1_split.json joint_burst_v2_split.json small_batch_v1_split.json; do ln -s "$(readlink -f "$SRC/grounded_$R/$f")" "$T/$f"; done
      W=$CL/wl_so/grounded_$R
      [[ -d $W ]] || python3 scripts_cosim/client_local_v1_single_origin.py "$SRC/grounded_$R/wl" "$W"
      ln -s "$W" "$T/wl"
      touch "$T/.built"
    done
  done
  echo "=== single-origin inputs built ==="
  exit 0
fi

case "${SO_VARIANT:-}" in
  server) KINDS=reactive,cd,locality,selfpredict,batched ;;
  client) KINDS=reactive,cd,locality,selfpredict,batched,offload,localfirst ;;
  *) echo "FAIL LOUD: SO_VARIANT must be server or client"; exit 1 ;;
esac
IN=$CL/inputs_so_$SO_VARIANT; OUT=$CL/gate_so_$SO_VARIANT
for R in x20 x30 x50; do [[ -f "$IN/grounded_$R/.built" ]] || { echo "FAIL LOUD: $IN/grounded_$R not built (BUILD=1 first)"; exit 1; }; done
mkdir -p "$OUT"
set +e
CL1_KINDS_RUN=$KINDS python3 scripts_cosim/fresh_topo_burst_v1_gate.py cl1 --inputs "$IN" --out "$OUT" \
  --selection "$IN/selected.json" --parallel "$P" --mem 4G --timeout "$TIMEOUT" --no-scope | grep -v "^\[skip"
rc=${PIPESTATUS[0]}
set -e
echo "=== so-$SO_VARIANT: $(ls "$OUT"/*.summary.json 2>/dev/null | wc -l) summaries, $(ls "$OUT"/*.failed.json 2>/dev/null | wc -l) failed ==="
find "$OUT" -name "*.log" -size +50M -exec gzip -1 {} \;
echo "driver rc=$rc"
