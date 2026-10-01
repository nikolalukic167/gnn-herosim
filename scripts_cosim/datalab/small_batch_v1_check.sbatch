#!/bin/bash
# small_batch_v1 check A (docs/lineages/small_batch_v1.md): the existing 10-task-trained checkpoints (rawgnn, rawmlp,
# xs1load, xs1mpoff; 4 seeds; self-refine 0 and 3) decoded on the small-batch VAL, plus the CD-replica and independent-argmin
# rules on both the small-batch and the 10-task VAL. The 10-task checkpoints' regrets on the 10-task VAL come from the
# queue-gap diagnostic (raw_plan_v2/qgap), run once already.
#   sbatch --export=ALL,WT=/home/nikola.lukic/gnn-herosim-wt/small_batch small_batch_v1_check.sbatch
#SBATCH --job-name=sb1-check
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/sb1-check-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=240G
#SBATCH --time=04:00:00
set -euo pipefail
WT=${WT:-/home/nikola.lukic/gnn-herosim-wt/small_batch}
GR=/home/nikola.lukic/gnn-herosim-wt/grounded
O=/home/nikola.lukic/gnn-herosim/simulation_data/small_batch_v1/check
cd "$WT"
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
echo "HEAD $(git rev-parse HEAD)  node $(hostname)"
[[ -z "$(git status --porcelain -- src scripts_cosim experiments run_experiment.py)" ]] || { echo "FAIL LOUD: worktree dirty"; exit 1; }
CACHE=simulation_data/graphs_cache_small_batch_v1_psv4_inf; SPLIT=experiments/small_batch_v1_split.json
[[ -f "$CACHE/metadata.json" && -f "$SPLIT" ]] || { echo "FAIL LOUD: cache or split missing"; exit 1; }
mkdir -p "$O/qgap"
export CACHE SPLIT QGAP_OUT="$O/qgap"
cat > "$O/run_one.sh" <<'RUN'
#!/bin/bash
case "$1" in
  rawgnn) STEM=raw-plan-v1-rawgnn;; rawmlp) STEM=raw-plan-v1-rawmlp;;
  psignn) STEM=exchange-seconds-v1-xs1load;; psimlp) STEM=peak-controls-v1-xs1mpoff;;
  *) echo "FAIL LOUD: arm $1" >&2; exit 1;;
esac
F="$QGAP_OUT/$1_s$2_r$3.json"
[[ -f "$F" ]] || python3 scripts_cosim/queue_gap_diag_eval.py --ckpt "models/$STEM-lr2e3-seed$2.pt" --refine "$3" --out "$F" \
  --cache "$CACHE" --split "$SPLIT" 2>&1 | grep -vi warn | grep "^\[diag\]\|Error\|FAIL" || true
[[ -f "$F" ]] || { echo "FAIL LOUD: no $F" >&2; exit 1; }
RUN
chmod +x "$O/run_one.sh"
python3 scripts_cosim/small_batch_rules_eval.py --cache "$CACHE" --split "$SPLIT" --out "$O/rules_small.json" &
python3 scripts_cosim/small_batch_rules_eval.py --cache "$GR/simulation_data/graphs_cache_backlog_corpus_v1_psv4_inf" \
  --split "$GR/experiments/backlog_corpus_v1_split.json" --out "$O/rules_old.json" &
for arm in rawgnn rawmlp psignn psimlp; do for s in 1 2 3 4; do for r in 0 3; do echo "$arm $s $r"; done; done; done \
  | xargs -P 10 -L 1 "$O/run_one.sh"
wait
echo "=== check done: $(ls "$O"/qgap/*.json | wc -l)/32 diag files ==="
python3 scripts_cosim/small_batch_check_read.py --old-diag /home/nikola.lukic/gnn-herosim/simulation_data/raw_plan_v2/qgap \
  --new-diag "$O/qgap" --old-rules "$O/rules_old.json" --new-rules "$O/rules_small.json" \
  --old-root "$GR/simulation_data" --new-root "$WT/simulation_data" --cells 9207 9208 9209 --out "$O/check_read.json"
