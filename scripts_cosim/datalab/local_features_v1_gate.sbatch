#!/bin/bash
# local_features_v1 gate (docs/lineages/local_features_v1.md) on small_batch_confirm_v1's 19 topologies and inputs tree.
# The references (CD, cdextr, sb1load, sb1mpoff, xs1load) are read from that lineage's gate directory.
#   SMOKE=1                 train/serve parity on cached graphs for seed 1 of each arm; one live run per arm, with
#                           lf1gnn's live decisions dumped and replayed bitwise through the trainer's model
#   LF1_KINDS_RUN=<arm>     the full phase for one arm (lf1gnn | lf1twin | lf1mlp), 8 seeds
#SBATCH --job-name=lf1-gate
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/lf1-gate-%j.out
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
CACHE=/home/nikola.lukic/gnn-herosim-wt/small_batch/simulation_data/graphs_cache_small_batch_v1_psv4_inf
IN="$C/inputs"
P=${PAR:-$(( ${SLURM_CPUS_PER_TASK:-64} - 2 ))}
cd "$WT"
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT"
python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"
echo "HEAD $(git rev-parse HEAD)  node $(hostname)  P=$P  SMOKE=${SMOKE:-0} arms=${LF1_KINDS_RUN:-}"
[[ -z "$(git status --porcelain -- src scripts_cosim experiments run_experiment.py)" ]] || { echo "FAIL LOUD: worktree dirty"; exit 1; }
for R in x20 x30 x50; do [[ -f "$IN/grounded_$R/.built" ]] || { echo "FAIL LOUD: small_batch_confirm_v1 inputs missing"; exit 1; }; done
cmp "$WT/docs/lineages/small_batch_confirm_v1/selected.json" "$IN/selected.json"
mkdir -p "$LF"

copy_models() {  # <arm> <seeds...>
  local arm=$1; shift
  for s in "$@"; do
    for ext in .pt .contract.json; do
      f="$WT/models/local-features-v1-$arm-lr2e3-seed$s$ext"
      [ -f "$f" ] || { echo "FAIL LOUD: missing $f"; exit 1; }
      cp -n "$f" "$IN/models/"; cmp "$f" "$IN/models/$(basename "$f")" || { echo "FAIL LOUD: models copy of $f differs"; exit 1; }
    done
  done
}

if [[ ${SMOKE:-0} == 1 ]]; then
  for arm in lf1gnn lf1twin lf1mlp; do
    copy_models $arm 1
    python3 scripts_cosim/raw_plan_v2_parity.py cache --ckpt "$WT/models/local-features-v1-$arm-lr2e3-seed1.pt" \
      --config "experiments/local_features_v1_$arm.yaml" --cache "$CACHE" --n-graphs 20
  done
  DUMP="$LF/smoke_dump_lf1gnn"; rm -rf "${DUMP:?}"
  for arm in lf1gnn lf1twin lf1mlp; do
    if [[ $arm == lf1gnn ]]; then export GNN_PLAN_RAW_DUMP="$DUMP" GNN_PLAN_RAW_DUMP_N=200; else unset GNN_PLAN_RAW_DUMP GNN_PLAN_RAW_DUMP_N; fi
    LF1_SMOKE=$arm python3 scripts_cosim/fresh_topo_burst_v1_gate.py lf1conf --inputs "$IN" --out "$LF/smoke" \
      --selection "$IN/selected.json" --parallel 1 --mem 4G --timeout "$TIMEOUT" --no-scope
  done
  unset GNN_PLAN_RAW_DUMP GNN_PLAN_RAW_DUMP_N
  # the live runs score on one BLAS thread; an 8-thread replay differs in the 7th digit (4.8e-7, 2026-10-04)
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 scripts_cosim/raw_plan_v2_parity.py replay --ckpt "$WT/models/local-features-v1-lf1gnn-lr2e3-seed1.pt" \
    --config experiments/local_features_v1_lf1gnn.yaml --dump "$DUMP"
  N=$(ls "$LF/smoke"/*.summary.json 2>/dev/null | wc -l)
  [[ $N == 3 ]] || { echo "FAIL LOUD: smoke wrote $N/3 summaries"; exit 1; }
  echo "=== smoke ok ==="
  exit 0
fi

ARM=${LF1_KINDS_RUN:?FAIL LOUD: LF1_KINDS_RUN unset (lf1gnn | lf1twin | lf1mlp)}
copy_models "$ARM" 1 2 3 4 5 6 7 8
set +e
python3 scripts_cosim/fresh_topo_burst_v1_gate.py lf1conf --inputs "$IN" --out "$LF/gate" --selection "$IN/selected.json" \
  --parallel "$P" --mem 4G --timeout "$TIMEOUT" --no-scope
rc=$?
set -e
echo "=== lf1conf $ARM: $(ls "$LF/gate"/*"__${ARM}_selfref"*.summary.json 2>/dev/null | wc -l)/1824 summaries ==="
echo "driver rc=$rc"
exit $rc
