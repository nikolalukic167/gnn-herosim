#!/bin/bash
# small_batch_confirm_v1 gate (docs/lineages/small_batch_confirm_v1.md) on the 19 unseen topologies that raw_plan_v2's
# Phase C screen admitted (raw_plan_v2_mint_screen.sbatch; same admission rule). Its own inputs tree, $C/inputs:
#   cfg from the raw_plan_v2 Phase C mint (96 cells, 19 used); the grounded g0-g3 windows linked from the development
#   tree (sha-checked); grounded_x20/x30/x50 built with cd_gap_v1_build_b.py for the 19 cells, as peak_load_v2 did.
#   BUILD=1                  build the inputs tree and exit (run once, before the parts)
#   SBCONF_KINDS=<part>      run one part of phase sbconf: sb1load | sb1mpoff | rules (cd, cdextr, reactive, xs1load)
#   sbatch --export=ALL,WT=<grounded worktree>,BUILD=1 small_batch_confirm_v1_gate.sbatch
#   sbatch --export=ALL,WT=<grounded worktree>,SBCONF_KINDS=sb1load,PAR=48 small_batch_confirm_v1_gate.sbatch
#SBATCH --job-name=sbconf
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/sbconf-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=24:00:00
set -euo pipefail
WT=${WT:-/home/nikola.lukic/gnn-herosim-wt/grounded}
TIMEOUT=${TIMEOUT:-2700}
C=/home/nikola.lukic/gnn-herosim/simulation_data/small_batch_confirm_v1
RC=/home/nikola.lukic/gnn-herosim/simulation_data/raw_plan_v2/confirm
DEVIN=/home/nikola.lukic/gnn-herosim/simulation_data/backlog_corpus_v1/gates/inputs
SBIN=/home/nikola.lukic/gnn-herosim-wt/small_batch/experiments/small_batch_v1_split.json
IN="$C/inputs"
P=${PAR:-$(( ${SLURM_CPUS_PER_TASK:-64} - 2 ))}
cd "$WT"
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT"
python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"
echo "HEAD $(git rev-parse HEAD)  node $(hostname)  P=$P  BUILD=${BUILD:-0} part=${SBCONF_KINDS:-}"
[[ -z "$(git status --porcelain -- src scripts_cosim experiments run_experiment.py)" ]] || { echo "FAIL LOUD: worktree dirty"; exit 1; }
SEL="$WT/docs/lineages/small_batch_confirm_v1/selected.json"
python3 - "$SEL" "$RC/selected.json" <<'PY'
import json, sys
r, rc = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
assert r.get("verdict") == "DESIGN-READY" and len(r["topologies"]) == 19, r.get("verdict")
assert r["topologies"] == rc["qualified"], "FAIL LOUD: selection is not raw_plan_v2's admitted pool"
print("[selection] small_batch_confirm_v1 study:", r["topologies"])
PY

if [[ ${BUILD:-0} == 1 ]]; then
  mkdir -p "$IN/models" "$IN/grounded"
  [[ -e "$IN/cfg" ]] || ln -s "$RC/cfg" "$IN/cfg"
  [[ -e "$IN/grounded/wl" ]] || ln -s "$DEVIN/grounded/wl" "$IN/grounded/wl"
  (cd "$IN/grounded/wl" && sha256sum -c "$WT/docs/lineages/grounded_workload_v1/workloads.sha256") \
    || { echo "FAIL LOUD: grounded workloads differ from the local mint"; exit 1; }
  cp -n "$SEL" "$IN/selected.json"; cmp "$SEL" "$IN/selected.json" || { echo "FAIL LOUD: stale selection copy"; exit 1; }
  for f in backlog_corpus_v1_split.json joint_burst_v2_split.json; do
    cp -n "$DEVIN/$f" "$IN/$f"; cmp "$DEVIN/$f" "$IN/$f" || { echo "FAIL LOUD: stale $f"; exit 1; }
  done
  cp -n "$SBIN" "$IN/small_batch_v1_split.json"; cmp "$SBIN" "$IN/small_batch_v1_split.json" || { echo "FAIL LOUD: stale small split"; exit 1; }
  [[ "$(sha256sum "$IN/small_batch_v1_split.json" | cut -d' ' -f1)" == a580315e5b8cfeb9135a57460399c2d575b7874c13a5340cb3a72e5489382fba ]] \
    || { echo "FAIL LOUD: small split sha mismatch"; exit 1; }
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
      for f in backlog_corpus_v1_split.json small_batch_v1_split.json; do ln -s "$IN/$f" "$D/$f"; done
      [ "$(ls "$D"/wl/grounded_g*_n50000.json | wc -l)" = 4 ] || { echo "FAIL LOUD: grounded $R build incomplete"; exit 1; }
      touch "$D/.built"
    fi
  done
  echo "=== inputs built ==="
  exit 0
fi

PART=${SBCONF_KINDS:?FAIL LOUD: SBCONF_KINDS unset (sb1load | sb1mpoff | rules)}
for R in x20 x30 x50; do [[ -f "$IN/grounded_$R/.built" ]] || { echo "FAIL LOUD: inputs not built (BUILD=1 first)"; exit 1; }; done
case $PART in
  sb1load|sb1mpoff) STEMS=(); for s in 1 2 3 4 5 6 7 8; do STEMS+=("small-batch-v1-$PART-lr2e3-seed$s"); done;;
  rules) STEMS=(); for s in 1 2 3 4; do STEMS+=("exchange-seconds-v1-xs1load-lr2e3-seed$s"); done;;
  *) echo "FAIL LOUD: part $PART"; exit 1;;
esac
for st in "${STEMS[@]}"; do
  for ext in .pt .contract.json; do
    f="$WT/models/$st$ext"
    [ -f "$f" ] || { echo "FAIL LOUD: missing $f"; exit 1; }
    cp -n "$f" "$IN/models/"
    cmp "$f" "$IN/models/$st$ext" || { echo "FAIL LOUD: models copy of $f differs"; exit 1; }
  done
done

set +e
python3 scripts_cosim/fresh_topo_burst_v1_gate.py sbconf --inputs "$IN" --out "$C/gate" --selection "$IN/selected.json" \
  --parallel "$P" --mem 4G --timeout "$TIMEOUT" --no-scope
rc=$?
set -e
echo "=== sbconf $PART: $(ls "$C/gate"/*.summary.json 2>/dev/null | wc -l)/5244 summaries in the gate dir ==="
echo "driver rc=$rc"
exit $rc
