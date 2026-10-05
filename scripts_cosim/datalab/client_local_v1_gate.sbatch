#!/bin/bash
# client_local_v1 feasibility gate: may running a call on its own client pay? small_batch_confirm_v1's 19 topologies and
# grounded x2/x3/x5 windows, with every cell changed in three ways only: each client hosts one replica per function type
# (replicas.<type>.per_client = 1), the backbone routes every node pair so a partner exchange from a client has a path
# (network.backbone.exchange_routes = all_pairs), and "client_local_v1": true (the gate then serves
# HEROSIM_SERVER_ONLY_REPLICAS=0). Rules only: Knative, CD, locality-first, self-predict, one-pass greedy,
# always-offload, local-first. The server-only runs of the same rules on the same cells are the comparison.
#   BUILD=1            build $CL/inputs and exit
#   CL1_KINDS_RUN=...  a subset of the kinds (default all)
#SBATCH --job-name=cl1-gate
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/cl1-gate-%j.out
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
SRC=$D/small_batch_confirm_v1/inputs
CL=$D/client_local_v1
IN=$CL/inputs
P=${PAR:-$(( ${SLURM_CPUS_PER_TASK:-64} - 2 ))}
cd "$WT"
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT"
python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"
echo "HEAD $(git rev-parse HEAD)  node $(hostname)  P=$P  BUILD=${BUILD:-0} kinds=${CL1_KINDS_RUN:-all}"
[[ -z "$(git status --porcelain -- src scripts_cosim experiments run_experiment.py)" ]] || { echo "FAIL LOUD: worktree dirty"; exit 1; }

if [[ ${BUILD:-0} == 1 ]]; then
  mkdir -p "$IN"
  cp -n "$SRC/selected.json" "$IN/selected.json"; cmp "$SRC/selected.json" "$IN/selected.json"
  for R in x20 x30 x50; do
    S="$SRC/grounded_$R"; T="$IN/grounded_$R"
    [[ -f "$S/.built" ]] || { echo "FAIL LOUD: $S not built"; exit 1; }
    [[ -f "$T/.built" ]] && continue
    rm -rf "$T"; mkdir -p "$T/cfg"
    for f in wl models backlog_corpus_v1_split.json joint_burst_v2_split.json small_batch_v1_split.json; do ln -s "$(readlink -f "$S/$f")" "$T/$f"; done
    python3 - "$S/cfg" "$T/cfg" <<'PY'
import json, os, sys
src, dst = sys.argv[1:]
n = 0
for name in sorted(os.listdir(src)):
    cell = json.load(open(os.path.join(src, name)))
    for t, r in cell["replicas"].items():
        assert r.get("per_client", 0) == 0, (name, t)
        r["per_client"] = 1
    bb = cell["network"]["backbone"]
    assert "exchange_routes" not in bb, name
    bb["exchange_routes"] = "all_pairs"
    cell["client_local_v1"] = True
    json.dump(cell, open(os.path.join(dst, name), "w"), indent=1)
    n += 1
print(f"[cells] {n} cells written to {dst}")
PY
    touch "$T/.built"
  done
  echo "=== inputs built ==="
  exit 0
fi

for R in x20 x30 x50; do [[ -f "$IN/grounded_$R/.built" ]] || { echo "FAIL LOUD: $R not built (BUILD=1 first)"; exit 1; }; done
mkdir -p "$CL/gate"
set +e
python3 scripts_cosim/fresh_topo_burst_v1_gate.py cl1 --inputs "$IN" --out "$CL/gate" --selection "$IN/selected.json" \
  --parallel "$P" --mem 4G --timeout "$TIMEOUT" --no-scope | grep -v "^\[skip"
rc=${PIPESTATUS[0]}
set -e
echo "=== cl1: $(ls "$CL"/gate/*.summary.json 2>/dev/null | wc -l) summaries total, $(ls "$CL"/gate/*.failed.json 2>/dev/null | wc -l) failed ==="
find "$CL/gate" -name "*.log" -size +50M -exec gzip -1 {} \;
python3 scripts_cosim/client_local_v1_read.py --gate "$CL/gate" --out "$CL/cl1_read.json"
echo "driver rc=$rc"
