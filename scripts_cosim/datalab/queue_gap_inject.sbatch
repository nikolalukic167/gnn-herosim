#!/bin/bash
# Information-injection test (docs/lineages/raw_plan_v2.md): hand exchange / in-batch-load terms subtracted from the
# model's own logits at decode, 5x5 grid, raw GNN, raw MLP and the engineered-context GNN as the reference, 4 seeds.
# The zero setting must reproduce the diagnostic's refine-0 regrets.
#   sbatch --export=ALL,WT=<grounded worktree>[,STAGE=2] queue_gap_inject.sbatch
#SBATCH --job-name=qinj
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/qinj-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=240G
#SBATCH --time=03:00:00
set -euo pipefail
WT=${WT:-/home/nikola.lukic/gnn-herosim-wt/grounded}
O=/home/nikola.lukic/gnn-herosim/simulation_data/raw_plan_v2/qinj
cd "$WT"
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
echo "HEAD $(git rev-parse HEAD)  node $(hostname)"
[[ -z "$(git status --porcelain -- src scripts_cosim experiments run_experiment.py)" ]] || { echo "FAIL LOUD: worktree dirty"; exit 1; }
mkdir -p "$O"
cat > "$O/run_one.sh" <<'RUN'
#!/bin/bash
case "$1" in
  rawgnn) STEM=raw-plan-v1-rawgnn;; rawmlp) STEM=raw-plan-v1-rawmlp;; psignn) STEM=exchange-seconds-v1-xs1load;;
  *) echo "FAIL LOUD: arm $1" >&2; exit 1;;
esac
F="$QINJ_OUT/$1_s$2.json"
[[ $QINJ_STAGE == 2 ]] && F="$QINJ_OUT/$1_s$2_st2.json"
[[ -f "$F" ]] || python3 scripts_cosim/queue_gap_inject_eval.py --ckpt "models/$STEM-lr2e3-seed$2.pt" --out "$F" --stage "$QINJ_STAGE" 2>&1 \
  | grep -vi warn | grep "a=0.0 b=0.0\|Error\|FAIL\|Traceback" || true
[[ -f "$F" ]] || { echo "FAIL LOUD: no $F" >&2; exit 1; }
RUN
chmod +x "$O/run_one.sh"; export QINJ_OUT="$O" QINJ_STAGE="${STAGE:-1}"
for arm in rawgnn rawmlp psignn; do for s in 1 2 3 4; do echo "$arm $s"; done; done | xargs -P 12 -L 1 "$O/run_one.sh"
echo "=== qinj done: $(ls "$O"/*_s*.json | wc -l) result files ==="
