#!/bin/bash
# Queue-gap diagnostic (docs/lineages/raw_plan_v2.md): the served decoder on every VAL graph for the v1 raw GNN / raw
# MLP and the engineered-context GNN / MP-OFF twin, 4 seeds each, self-refine 0 and 3. Seed 1 of the v1 raw GNN runs
# first on its own: its refine-0 regret must reproduce the trainer's best val regret (19.10 s).
#   sbatch --export=ALL,WT=<grounded worktree> queue_gap_diag.sbatch
#SBATCH --job-name=qgap
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/qgap-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=240G
#SBATCH --time=04:00:00
set -euo pipefail
WT=${WT:-/home/nikola.lukic/gnn-herosim-wt/grounded}
O=/home/nikola.lukic/gnn-herosim/simulation_data/raw_plan_v2/qgap
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
# run_one.sh <arm> <seed> <refine>
case "$1" in
  rawgnn) STEM=raw-plan-v1-rawgnn;; rawmlp) STEM=raw-plan-v1-rawmlp;;
  psignn) STEM=exchange-seconds-v1-xs1load;; psimlp) STEM=peak-controls-v1-xs1mpoff;;
  *) echo "FAIL LOUD: arm $1" >&2; exit 1;;
esac
F="$QGAP_OUT/$1_s$2_r$3.json"
[[ -f "$F" ]] || python3 scripts_cosim/queue_gap_diag_eval.py --ckpt "models/$STEM-lr2e3-seed$2.pt" --refine "$3" --out "$F" 2>&1 \
  | grep -vi warn | grep "^\[diag\]\|Error\|FAIL" || true
[[ -f "$F" ]] || { echo "FAIL LOUD: no $F" >&2; exit 1; }
RUN
chmod +x "$O/run_one.sh"; export QGAP_OUT="$O"
"$O/run_one.sh" rawgnn 1 0
for arm in rawgnn rawmlp psignn psimlp; do for s in 1 2 3 4; do for r in 0 3; do echo "$arm $s $r"; done; done; done \
  | xargs -P 10 -L 1 "$O/run_one.sh"
echo "=== qgap done: $(ls "$O"/*.json | wc -l)/32 ==="
