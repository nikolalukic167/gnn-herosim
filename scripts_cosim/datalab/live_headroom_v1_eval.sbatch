#!/bin/bash
# live_headroom_v1 -- the 10-task-trained checkpoints (rawgnn, rawmlp, xs1load, xs1mpoff; 4 seeds; self-refine 0 and 3) and the CD /
# independent-argmin rules decoded on the live-snapshot cache, then the reader (docs/lineages/live_headroom_v1.md).
#   sbatch --export=ALL,WT=/home/nikola.lukic/gnn-herosim-wt/live_headroom scripts_cosim/datalab/live_headroom_v1_eval.sbatch
#SBATCH --job-name=lh1-eval
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/lh1-eval-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=240G
#SBATCH --time=06:00:00
set -euo pipefail
WT=${WT:-/home/nikola.lukic/gnn-herosim-wt/live_headroom}
O=/home/nikola.lukic/gnn-herosim/simulation_data/live_headroom_v1/eval
cd "$WT"
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
echo "HEAD $(git rev-parse HEAD)  node $(hostname)"
[[ -z "$(git status --porcelain -- src scripts_cosim experiments run_experiment.py | grep -v '^??' || true)" ]] || { echo "FAIL LOUD: worktree dirty"; exit 1; }
CACHE=simulation_data/graphs_cache_live_headroom_v1_psv4_inf; SPLIT=experiments/live_headroom_v1_split.json
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
python3 scripts_cosim/small_batch_rules_eval.py --cache "$CACHE" --split "$SPLIT" --out "$O/rules.json" &
for arm in rawgnn rawmlp psignn psimlp; do for s in 1 2 3 4; do for r in 0 3; do echo "$arm $s $r"; done; done; done \
  | xargs -P 10 -L 1 "$O/run_one.sh"
wait
echo "=== eval done: $(ls "$O"/qgap/*.json | wc -l)/32 diag files ==="
python3 scripts_cosim/live_headroom_read.py --diag "$O/qgap" --rules "$O/rules.json" --cache "$CACHE" \
  --root "$WT/simulation_data" --out "$O/live_headroom_read.json"
