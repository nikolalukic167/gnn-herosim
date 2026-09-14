#!/bin/bash
# peer_affinity_warm_v1 -- W1 offline read, part 2: the registered T1 read over the warm
# reports (peer_affinity_warm_v1_w1_score.sbatch wrote them). Re-runs nothing; aggregates.
#   sbatch --dependency=afterok:<score job> scripts_cosim/datalab/peer_affinity_warm_v1_w1_read.sbatch
#
#SBATCH --job-name=pa-warm-w1-read
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/pa-warm-w1-read-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/pa-warm-w1-read-%j.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$PWD" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=""
echo "[CODE] $(git describe --dirty --always) on $(git rev-parse --abbrev-ref HEAD)"

# The read tool's convergence check globs logs/pa-t1-gnn-*_{task}.out with
# task = cfg_index*seeds + (seed-1), cfg_index 2 = gnn lr2e3, 5 = mpoff lr2e3. The W1 train
# array (762848) is task = 16*arm + (seed-1) under a different name; link them into a scratch
# logs dir so the slope check reads the real curves instead of silently returning None.
TRAIN_JOB="${TRAIN_JOB:-762848}"
LOGS=simulation_data/peer_affinity_warm_read_logs
rm -rf "$LOGS"; mkdir -p "$LOGS"
for t in $(seq 0 31); do
  src="logs/pa-warm-w1-train-${TRAIN_JOB}_${t}.out"
  [[ -f "$src" ]] || { echo "FAIL LOUD: train log missing: $src" >&2; exit 1; }
  if [[ $t -lt 16 ]]; then dst=$(( 2 * 16 + t )); else dst=$(( 5 * 16 + t - 16 )); fi
  ln -s "$PWD/$src" "$LOGS/pa-t1-gnn-${TRAIN_JOB}_${dst}.out"
done

REPORTS=simulation_data/peer_affinity_warm_reports
n=$(ls "$REPORTS"/peer-affinity-v1-warm-*.json 2>/dev/null | wc -l)
[[ "$n" -eq 64 ]] || { echo "FAIL LOUD: expected 64 reports in $REPORTS, found $n" >&2; exit 1; }

python3 scripts_cosim/peer_affinity_t1_read.py \
  --tag warm --seeds 16 \
  --models-dir models \
  --cache-dir simulation_data/graphs_cache_peer_affinity_v1_warm \
  --split experiments/peer_affinity_v1_warm_split.json \
  --reports-dir "$REPORTS" \
  --logs-dir "$LOGS" \
  --out simulation_data/peer_affinity_warm_read.json
python3 - <<'PY'
import json
r = json.load(open("simulation_data/peer_affinity_warm_read.json"))
print("chosen_lr", r["chosen_lr"])
for k, c in r["contrasts"].items():
    if "mpoff" in k and k.startswith("gnn"):
        print(k, {x: c[x] for x in ("n_seeds", "median_delta_pp", "mean_delta_pp", "p_exact_wilcoxon", "reading", "selector")})
PY
echo "=== warm W1 offline read done ==="
