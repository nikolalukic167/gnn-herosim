#!/bin/bash
#SBATCH --job-name=siv1-gate
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/siv1-gate-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/siv1-gate-%A_%a.err
#SBATCH --partition=GPU-a40,GPU-a100s,GPU-l40s,GPU-a100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --array=0-14

# siv1_full_corpus — the lineage's first live gate on a real workload trace.
# 15 array tasks = 3 policies x 5 parity-verified topology cells.
#
# Cells are minted and infra-parity-verified by make_full_corpus_siv1_gate_cells.py.
# Task 0 mints them if absent; the others wait for the config to appear so five
# concurrent tasks do not race to generate the same files.
#
# NOTE: no --seed is passed to the simulator anywhere in this gate. The live CLI's
# --seed OVERRIDES the config's topology seed (executesimulation.py:842-854), which
# would silently change the infrastructure per array task.

set -euo pipefail

PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

# NOTE: --shell bash, not --bash. Login nodes accept the old --bash spelling but compute
# nodes run a newer micromamba that rejects it, so this fails ONLY under sbatch (job
# 707292: all 15 array tasks dead in ~1s). Every other script in this directory uses this
# form; a login-node sanity check cannot catch the difference.
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn

# REQUIRED. `pipenv run python3` resolves its own venv and shells straight past the
# activation above, so without this every runner script below would silently execute in
# an undeclared venv (torch 2.12.0+cu130) instead of the `gnn` env this job just
# activated. That is what happened to every cluster gate through 2026-08-21; see
# PARITY.md rule 1. The runners read ${HEROSIM_PY:-pipenv run python3}, so locally
# (where this is unset) nothing changes.
export HEROSIM_PY=python3

# Placement tie-breaks are now sorted deterministically by (node.id, platform.id), so this
# is defense-in-depth against any unaudited set/dict-hash dependency, not the primary fix.
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export SWEEP_DIR="${SWEEP_DIR:-simulation_data/normal_sim_sweeps/full_corpus_siv1_live_gate_20260820}"
export WORKLOAD="${WORKLOAD:-data/nofs-ids/traces/workload-125-225.json}"
export GNN_MODEL="${GNN_MODEL:-models/near-rtt-v2-full-corpus-siv1-dim14-ce-only.pt}"
export MLP_MODEL="${MLP_MODEL:-models/tabular/batch_edge_mlp_full_corpus_siv1_dim22_batchcache.pt}"
export HEROSIM_WARMTH_PHYSICS=node_disk_v2
export GNN_DECODE_MODE=argmax
export GNN_BATCH_SIZE=4
export GNN_BATCH_TIMEOUT=0.002
export TIMEOUT="${TIMEOUT:-18000}"
export PYTHONPATH="$PROJECT_ROOT"
export PYTHONUNBUFFERED=1

CFG_DIR="${SWEEP_DIR}/configs"
CELLS=(cell01_p25_s9001 cell02_p35_s9002 cell03_p15_s9003 cell04_p50_s9004 cell05_p20_s9005)
policies=(knative mlp gnn)

policy="${policies[$(( SLURM_ARRAY_TASK_ID / 5 ))]}"
name="${CELLS[$(( SLURM_ARRAY_TASK_ID % 5 ))]}"

# --- cells: mint once (task 0), everyone else waits ------------------------------
if [[ "${SLURM_ARRAY_TASK_ID}" == "0" ]]; then
  if [[ ! -f "${CFG_DIR}/${CELLS[4]}.json" ]]; then
    echo "[gate] minting parity-verified cells"
    python3 scripts_cosim/important/make_full_corpus_siv1_gate_cells.py --sweep-dir "$SWEEP_DIR"
  fi
else
  for _ in $(seq 1 120); do
    [[ -f "${CFG_DIR}/${name}.json" ]] && break
    sleep 10
  done
fi
[[ -f "${CFG_DIR}/${name}.json" ]] || { echo "ERROR: cell config never appeared: ${CFG_DIR}/${name}.json" >&2; exit 1; }

# --- preflight: this cell must match the training corpus before simulating -------
python3 scripts_cosim/verify_live_infra_parity.py \
  --dataset "${SWEEP_DIR}/cell_infrastructure/${name}" -v

echo "=== siv1 live gate: ${policy} / ${name} ==="
bash scripts_cosim/important/run_full_corpus_siv1_live_gate.sh "$policy" "$name"
