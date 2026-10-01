#!/bin/bash
# raw_plan_v2 Phase C gate (docs/lineages/raw_plan_v2.md) on the unseen topologies that raw_plan_v2_mint_screen.sbatch
# admitted. It builds a separate inputs tree, $C/inputs, so the development cells are never touched:
#   cfg from the Phase C mint; the grounded g0-g3 windows linked from the development tree (topology-independent,
#   sha-checked); grounded_x20/x30/x50 built with cd_gap_v1_build_b.py exactly as peak_load_v2 built them.
# Then phase rp2conf runs: the dev-selected arm, its twin, the plain MLP (4 seeds each), CD, cdextr and reactive.
#   sbatch --export=ALL,WT=<grounded worktree>,RP2_SELECTED=rawS,TIMEOUT=2700 raw_plan_v2_confirm.sbatch
#SBATCH --job-name=rp2-conf
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/rp2-conf-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=24:00:00
set -euo pipefail
WT=${WT:-/home/nikola.lukic/gnn-herosim-wt/grounded}
TIMEOUT=${TIMEOUT:-2700}
SEL=${RP2_SELECTED:?FAIL LOUD: RP2_SELECTED unset (the selected arm of the dev read)}
[[ $SEL == rawE || $SEL == rawS || $SEL == rawES ]] || { echo "FAIL LOUD: RP2_SELECTED=$SEL"; exit 1; }
TWIN=rawStwin; [[ $SEL == rawE ]] && TWIN=""
C=/home/nikola.lukic/gnn-herosim/simulation_data/raw_plan_v2/confirm
DEVIN=/home/nikola.lukic/gnn-herosim/simulation_data/backlog_corpus_v1/gates/inputs
IN="$C/inputs"
P=${PAR:-$(( ${SLURM_CPUS_PER_TASK:-64} - 2 ))}
cd "$WT"
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT" RP2_SELECTED="$SEL"
python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"
echo "HEAD $(git rev-parse HEAD)  node $(hostname)  P=$P  selected=$SEL twin=${TWIN:-rawmlp}"
[[ -z "$(git status --porcelain -- src scripts_cosim experiments run_experiment.py)" ]] || { echo "FAIL LOUD: worktree dirty"; exit 1; }
python3 - "$C/selected.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
assert r.get("verdict") == "DESIGN-READY" and len(r.get("topologies") or []) == 12, r.get("verdict")
assert all(9473 <= t <= 9568 for t in r["topologies"]), r["topologies"]
print("[selection] Phase C study:", r["topologies"])
PY

mkdir -p "$IN/models" "$IN/grounded"
[[ -e "$IN/cfg" ]] || ln -s "$C/cfg" "$IN/cfg"
[[ -e "$IN/grounded/wl" ]] || ln -s "$DEVIN/grounded/wl" "$IN/grounded/wl"
(cd "$IN/grounded/wl" && sha256sum -c "$WT/docs/lineages/grounded_workload_v1/workloads.sha256") \
  || { echo "FAIL LOUD: grounded workloads differ from the local mint"; exit 1; }
cp -n "$C/selected.json" "$IN/selected.json"
cmp "$C/selected.json" "$IN/selected.json" || { echo "FAIL LOUD: stale selection copy"; exit 1; }
for f in backlog_corpus_v1_split.json joint_burst_v2_split.json; do
  cp -n "$DEVIN/$f" "$IN/$f"; cmp "$DEVIN/$f" "$IN/$f" || { echo "FAIL LOUD: stale $f"; exit 1; }
done
for s in 1 2 3 4; do
  FILES=("$WT/models/raw-plan-v1-rawmlp-lr2e3-seed$s" "$WT/models/raw-plan-v2-$SEL-lr2e3-seed$s")
  [[ -n $TWIN ]] && FILES+=("$WT/models/raw-plan-v2-$TWIN-lr2e3-seed$s")
  for f in "${FILES[@]}"; do
    for ext in .pt .contract.json; do
      [ -f "$f$ext" ] || { echo "FAIL LOUD: missing $f$ext"; exit 1; }
      cp -n "$f$ext" "$IN/models/"
      cmp "$f$ext" "$IN/models/$(basename "$f$ext")" || { echo "FAIL LOUD: models copy of $f$ext differs"; exit 1; }
    done
  done
done

SRC="$IN/grounded_src"
if [[ ! -L "$SRC/wl" ]]; then
  rm -rf "$SRC"; mkdir -p "$SRC"
  ln -s "$IN/cfg" "$SRC/cfg"; ln -s "$IN/grounded/wl" "$SRC/wl"; ln -s "$IN/models" "$SRC/models"
  ln -s "$IN/joint_burst_v2_split.json" "$SRC/joint_burst_v2_split.json"
fi
TOPOS=$(python3 -c "import json,sys;print(' '.join(map(str,json.load(open(sys.argv[1]))['topologies'])))" "$IN/selected.json")
for pair in x20:0.5 x30:0.3333333333333333 x50:0.2; do
  R=${pair%%:*}; F=${pair##*:}; D="$IN/grounded_$R"
  if [[ ! -f "$D/.built" ]]; then
    rm -rf "$D"; mkdir -p "$D"
    python3 scripts_cosim/cd_gap_v1_build_b.py --src "$SRC" --out "$D" --factor "$F" --topologies $TOPOS
    ln -s "$IN/backlog_corpus_v1_split.json" "$D/backlog_corpus_v1_split.json"
    [ "$(ls "$D"/wl/grounded_g*_n50000.json | wc -l)" = 4 ] || { echo "FAIL LOUD: grounded $R build incomplete"; exit 1; }
    touch "$D/.built"
  fi
done

set +e
python3 scripts_cosim/fresh_topo_burst_v1_gate.py rp2conf --inputs "$IN" --out "$C/gate" --selection "$IN/selected.json" \
  --parallel "$P" --mem 4G --timeout "$TIMEOUT" --no-scope
rc=$?
set -e
N=$(ls "$C/gate"/*.summary.json 2>/dev/null | wc -l)
echo "=== rp2conf: $N/2160 summaries ==="
echo "driver rc=$rc"
exit $rc
