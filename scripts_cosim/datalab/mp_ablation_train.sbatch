#!/bin/bash
# mp_ablation_v1 — train MESSAGE-PASSING-OFF draws, seeds 1-16, AT THE PINNED COMMIT
# c08aa7e, so they pair seed-for-seed with the MP-ON arms gnndraws1-16.
#
# Registration: docs/lineages/objective_pivot_v1/phase1-registration-draft.md (signed off
# 2026-08-28). This extends gnn_draw_study_v1's 8 seeded draws to 16; the frozen
# comparison arms were produced at commit c08aa7e in a clean tree (verified from job
# 712389's run_provenance), and HEAD has since gained ~5,000 route_b lines in
# behavior-relevant files — so the new draws train in a dedicated worktree pinned to that
# exact commit, making them exchangeable with seeds 1-8 BY CONSTRUCTION.
#
# This file lives at HEAD in the MAIN repo and only cd's into the pinned worktree; putting
# it inside the worktree would dirty the worktree's CODE_PATHS (scripts_cosim/ is one) and
# void every arm under the registration's own rule. Everything below the pin assertion is
# copied from scripts_cosim/datalab/gnn_draw_study_train.sbatch AS OF c08aa7e, with
# exactly three changes: the cd target, the seed arithmetic (9-16), and the job naming.
# See that file's comments for the rationale of each guard — they all still apply.
#
# Submit with:  --dependency=afterok:<objp1-determinism job id>
#SBATCH --job-name=mpabl-train
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/mpabl-train-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/mpabl-train-%A_%a.err
#SBATCH --partition=GPU-a40,GPU-l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-15

set -euo pipefail

PIN_DIR=/home/nikola.lukic/gnn-herosim-pin-c08aa7e
PIN_COMMIT=c08aa7ee140fd51e3d384f97df3f31b126df96ab
cd "$PIN_DIR"

if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PWD:$PWD/src/notebooks"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
mkdir -p logs models

# The registration's binding condition: exact commit, clean CODE_PATHS. Fail before any
# GPU time is spent, not after.
HEAD_NOW=$(git rev-parse HEAD)
[[ "$HEAD_NOW" == "$PIN_COMMIT" ]] || {
  echo "FAIL LOUD: pinned worktree at $HEAD_NOW, registration requires $PIN_COMMIT" >&2; exit 1; }
DIRTY=$(git status --porcelain -- src scripts_cosim experiments run_experiment.py)
[[ -z "$DIRTY" ]] || {
  echo "FAIL LOUD: pinned worktree dirty under CODE_PATHS:" >&2; echo "$DIRTY" >&2; exit 1; }
echo "[PIN] $PIN_DIR at $PIN_COMMIT, CODE_PATHS clean"

python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__,'cuda',torch.cuda.is_available())"

SEED=$(( SLURM_ARRAY_TASK_ID + 1 ))          # seeds 1..16
[[ "$SEED" -ge 1 && "$SEED" -le 16 ]] || { echo "FAIL LOUD: bad seed $SEED (array must be 0-15)" >&2; exit 1; }

# The deployed config: the cache the deployed checkpoint was fitted on. simulation_data/
# and models/ are symlinks into the main checkout, so checkpoints and caches are shared
# with the seed 1-8 arms — only the CODE differs, and it differs by exactly nothing.
export CACHE_DIR="simulation_data/graphs_cache_full_corpus_siv1_dim14"
export WANDB_RUN_NAME="gnn-draw-mpoff-s${SEED}"
export NEAR_RTT_TRAIN_SEED="$SEED"
OUT="models/${WANDB_RUN_NAME}.pt"

[[ ! -f "$OUT" ]] || { echo "FAIL LOUD: $OUT already exists — refusing to overwrite a draw" >&2; exit 1; }

# Same corpus guard as the seed 1-8 array: these must be draws of the deployed config.
DEPLOYED_CONTRACT="models/near-rtt-v2-full-corpus-siv1-dim14-ce-only.contract.json"
export MIN_GRAPHS
MIN_GRAPHS=$(python3 - "$DEPLOYED_CONTRACT" "$CACHE_DIR" <<'PY'
import json, pickle, sys
contract_path, cache_dir = sys.argv[1], sys.argv[2]
want = int(json.load(open(contract_path))["corpus"]["n_datasets"])
have = len(pickle.load(open(f"{cache_dir}/graphs.pkl", "rb")))
if have != want:
    raise SystemExit(
        f"FAIL LOUD: cache has {have} graphs, deployed checkpoint was fitted on {want}. "
        f"These would not be draws of the deployed config."
    )
print(want)
PY
) || { echo "FAIL LOUD: corpus does not match the deployed checkpoint" >&2; exit 1; }
echo "[CORPUS] ${MIN_GRAPHS} graphs, matches deployed checkpoint contract"

[[ -n "${WANDB_RUN_NAME}" ]] || { echo "FAIL LOUD: WANDB_RUN_NAME must be set" >&2; exit 1; }
[[ -f "${CACHE_DIR}/graphs.pkl" ]] || { echo "FAIL LOUD: missing ${CACHE_DIR}/graphs.pkl" >&2; exit 1; }

# mp_ablation_v1: message passing OFF is the ONLY intended difference from the
# gnndraws1-16 arms. The GIN module is still constructed (identical parameter set and
# init at a fixed seed), only its forward call is skipped -- see gnn_model.py:427.
export GNN_DISABLE_MESSAGE_PASSING=1
[[ "${GNN_DISABLE_MESSAGE_PASSING}" == "1" ]] || { echo "FAIL LOUD: ablation lever not set" >&2; exit 1; }
# Everything else must still be unset, or this is not a clean one-factor ablation.
for v in NEAR_RTT_MP_RESIDUAL GNN_MP_NODE_EDGES; do
  if [[ -n "${!v:-}" ]]; then
    echo "FAIL LOUD: $v is set to '${!v}'; arms must differ only by message passing" >&2
    exit 1
  fi
done

# Single-process data loading — reproducibility first, and the cluster ulimit second
# (see the seed 1-8 sbatch for the full story; job 712372 died on 4 workers).
export NEAR_RTT_DATALOADER_WORKERS=0

echo "=== task=${SLURM_ARRAY_TASK_ID} seed=${SEED} cache=${CACHE_DIR} -> ${OUT} (pinned ${PIN_COMMIT:0:12}) ==="
bash scripts_cosim/datalab/run_full_corpus_siv1_gnn_train.sh

[[ -f "$OUT" ]] || { echo "FAIL LOUD: checkpoint not written: $OUT" >&2; exit 1; }

# Validity gate: the contract must name THIS seed and confirm the run was reproducible.
python3 - "$OUT" "$SEED" <<'PY'
import json, sys
from pathlib import Path
out, want_seed = sys.argv[1], int(sys.argv[2])
sidecar = Path(out).with_suffix(".contract.json")
if not sidecar.is_file():
    raise SystemExit(f"FAIL LOUD: {sidecar} missing — a checkpoint without a contract is not evidence")
c = json.load(open(sidecar))
if int(c.get("train_seed", -1)) != want_seed:
    raise SystemExit(f"FAIL LOUD: {out} contract train_seed={c.get('train_seed')} != {want_seed}")
if not c.get("deterministic_algorithms"):
    raise SystemExit(f"FAIL LOUD: {out} trained without deterministic algorithms; this draw is not reproducible")
print(f"[OK] {out} seed={want_seed} layout={c.get('inference_feature_layout')} "
      f"queue={c.get('queue_feature_contract')} mp_residual={c.get('mp_residual')} "
      f"mp_node_edges={c.get('mp_node_edges')}")
PY
echo "=== task=${SLURM_ARRAY_TASK_ID} seed=${SEED} COMPLETE ==="
