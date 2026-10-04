#!/bin/bash
# scale_sweep_v1 gate (docs/lineages/scale_sweep_v1.md): the engineered arms and the rules at x3 / x5 on the 12 development
# topologies, in three conditions, each its own inputs dir $S/inputs/grounded_<rung><cond>:
#   m2 / m4  the grounded windows re-minted with --merge-k 2 / 4 (same libraries, bases and seeds as g0-g3), so a
#            decision batch holds ~2x / ~4x the tasks; cells take scheduler.batch_size 16 (the masked_topo maximum)
#   s12      the grounded windows unchanged on cells with 12 servers instead of 6 (cluster_scale_v1_cells.mint, servers
#            the only field that moves)
#   BUILD=1                 build every condition and exit
#   SCALE_CONDS_RUN=<cond>  run phase `scale` for one condition: m2 | m4 | s12
#SBATCH --job-name=scale
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/scale-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=24:00:00
set -euo pipefail
WT=${WT:-/home/nikola.lukic/gnn-herosim-wt/live_headroom}
TIMEOUT=${TIMEOUT:-2700}
S=/home/nikola.lukic/gnn-herosim/simulation_data/scale_sweep_v1
DEVIN=/home/nikola.lukic/gnn-herosim/simulation_data/backlog_corpus_v1/gates/inputs
BASEWL=/home/nikola.lukic/gnn-herosim/simulation_data/peer_affinity_live_gate/workloads
IN="$S/inputs"
P=${PAR:-$(( ${SLURM_CPUS_PER_TASK:-64} - 2 ))}
cd "$WT"
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT"
python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"
echo "HEAD $(git rev-parse HEAD)  node $(hostname)  P=$P  BUILD=${BUILD:-0} cond=${SCALE_CONDS_RUN:-}"
[[ -z "$(git status --porcelain -- src scripts_cosim experiments run_experiment.py)" ]] || { echo "FAIL LOUD: worktree dirty"; exit 1; }
TOPOS=$(python3 -c "import json,sys;print(' '.join(map(str,json.load(open(sys.argv[1]))['topologies'])))" "$DEVIN/selected.json")

if [[ ${BUILD:-0} == 1 ]]; then
  mkdir -p "$IN"
  cp -n "$DEVIN/selected.json" "$IN/selected.json"; cmp "$DEVIN/selected.json" "$IN/selected.json"
  LIB="$WT/docs/lineages/grounded_workload_v1"
  for K in 2 4; do
    W="$IN/wl_m$K"; mkdir -p "$W"; i=0
    for pair in 0:drainable_f4000_n50000 50:drainable_w1_n50000 100:drainable_w2_n50000 130:drainable_w3_n50000; do
      L=${pair%%:*}; BW=${pair##*:}
      [ -f "$W/grounded_g${i}_n50000.json" ] || python3 scripts_cosim/grounded_workload_v1_mint.py --merge-k $K \
        --lib "$LIB/fanout_lib_$L.json.gz" --base "$BASEWL/$BW.json" --seed $((7400 + i)) --out "$W/grounded_g${i}_n50000.json" \
        | python3 -c "import json,sys;m=json.load(sys.stdin);print('[mint] k=$K g$i mean group',round(m['mean_group_size'],2),'rate',round(m['rate_per_s'],4))"
      i=$((i + 1))
    done
  done
  python3 - "$DEVIN/cfg" "$IN" $TOPOS <<'PY'
import copy, json, os, sys
sys.path.insert(0, os.getcwd())
from scripts_cosim.cluster_scale_v1_cells import mint
src, out, topos = sys.argv[1], sys.argv[2], [int(t) for t in sys.argv[3:]]
for d in ("cfg_b16", "cfg_s12"):
    os.makedirs(os.path.join(out, d), exist_ok=True)
for t in topos:
    cell = json.load(open(f"{src}/cc40s{t}.json"))
    assert int(cell["scheduler"]["batch_size"]) == 10 and int(cell["nodes"]["server_nodes"]["count"]) == 6, t
    b16 = copy.deepcopy(cell)
    b16["scheduler"]["batch_size"] = 16
    json.dump(b16, open(f"{out}/cfg_b16/cc40s{t}.json", "w"), indent=1)
    s12 = mint(cell, 12, int(cell["network"]["topology"]["seed"]), vary="servers")
    json.dump(s12, open(f"{out}/cfg_s12/cc40s{t}.json", "w"), indent=1)
print(f"[cells] {len(topos)} cells: batch_size 16 copies and 12-server mints")
PY
  for C in m2 m4 s12; do
    SRC="$IN/src_$C"
    if [[ ! -L "$SRC/wl" ]]; then
      rm -rf "$SRC"; mkdir -p "$SRC"
      case $C in m2) ln -s "$IN/cfg_b16" "$SRC/cfg"; ln -s "$IN/wl_m2" "$SRC/wl";;
                 m4) ln -s "$IN/cfg_b16" "$SRC/cfg"; ln -s "$IN/wl_m4" "$SRC/wl";;
                 s12) ln -s "$IN/cfg_s12" "$SRC/cfg"; ln -s "$DEVIN/grounded/wl" "$SRC/wl";; esac
      ln -s "$DEVIN/models" "$SRC/models"; ln -s "$DEVIN/joint_burst_v2_split.json" "$SRC/joint_burst_v2_split.json"
    fi
    for pair in x30:0.3333333333333333 x50:0.2; do
      R=${pair%%:*}; F=${pair##*:}; D="$IN/grounded_$R$C"
      if [[ ! -f "$D/.built" ]]; then
        rm -rf "$D"; mkdir -p "$D"
        python3 scripts_cosim/cd_gap_v1_build_b.py --src "$SRC" --out "$D" --factor "$F" --topologies $TOPOS
        for f in backlog_corpus_v1_split.json small_batch_v1_split.json; do ln -s "$DEVIN/$f" "$D/$f"; done
        [ "$(ls "$D"/wl/grounded_g*_n50000.json | wc -l)" = 4 ] || { echo "FAIL LOUD: $R$C build incomplete"; exit 1; }
        touch "$D/.built"
      fi
    done
  done
  echo "=== inputs built ==="
  exit 0
fi

C=${SCALE_CONDS_RUN:?FAIL LOUD: SCALE_CONDS_RUN unset (m2 | m4 | s12)}
for R in x30 x50; do [[ -f "$IN/grounded_$R$C/.built" ]] || { echo "FAIL LOUD: $R$C not built (BUILD=1 first)"; exit 1; }; done
set +e
python3 scripts_cosim/fresh_topo_burst_v1_gate.py scale --inputs "$IN" --out "$S/gate" --selection "$IN/selected.json" \
  --parallel "$P" --mem 4G --timeout "$TIMEOUT" --no-scope
rc=$?
set -e
echo "=== scale $C: $(ls "$S/gate"/*"$C"__*.summary.json 2>/dev/null | wc -l)/1728 summaries ==="
echo "driver rc=$rc"
exit $rc
