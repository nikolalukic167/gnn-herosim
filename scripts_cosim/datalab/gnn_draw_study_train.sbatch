#!/bin/bash
# gnn_draw_study_v1 — train 8 seeded GNN draws of the DEPLOYED config.
#
# The question. p5b_draw_study measured the MLP's live reliability as a training-draw
# lottery: collapse counts swing 0 -> 26 of 30 cells on the seed alone, and 4 of its 8
# draws collapse. Against that, both GNN arms came in at 0/30. That reads decisive and is
# not: it is 2-3 GNN draws against a distribution in which a draw collapses about half the
# time, so p ~ 0.125. Writing "the GNN never collapses and the MLP does" on that evidence
# would repeat exactly the error p5b_draw_study was run to find.
#
# 8 clean draws is p ~ 0.004 under the same null, which is why the array is 0-7 and not
# 0-2. One config, not four: the claim under test is about the checkpoint that is actually
# deployed, and spreading the same budget over two caches would halve the power on the only
# question being asked.
#
# What makes a draw a draw. NEAR_RTT_TRAIN_SEED seeds init and shuffling ONLY. The
# canonical-parent split is separately fixed at random_state=42 inside the trainer and must
# stay that way, or this measures data-split variance instead of weight-init variance.
# Nothing else may vary across arms - in particular no NEAR_RTT_MP_RESIDUAL /
# GNN_MP_NODE_EDGES, which would make the arms different architectures wearing seed labels.
#
# Reproducibility of a draw is a precondition, not a bonus: as of 2026-08-25 the trainer
# also sets torch.use_deterministic_algorithms, and the checkpoint contract records both
# train_seed and deterministic_algorithms so the gate can assert which draw it is serving.
#SBATCH --job-name=gnn-draw-train
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/gnn-draw-train-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/gnn-draw-train-%A_%a.err
#SBATCH --partition=GPU-a40,GPU-l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-7

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim

if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PWD:$PWD/src/notebooks"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
mkdir -p logs models

python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__,'cuda',torch.cuda.is_available())"

SEED=$(( SLURM_ARRAY_TASK_ID + 1 ))          # seeds 1..8
[[ "$SEED" -ge 1 && "$SEED" -le 8 ]] || { echo "FAIL LOUD: bad seed $SEED (array must be 0-7)" >&2; exit 1; }

# The deployed config: the cache the deployed checkpoint was fitted on.
export CACHE_DIR="simulation_data/graphs_cache_full_corpus_siv1_dim14"
export WANDB_RUN_NAME="gnn-draw-s${SEED}"
export NEAR_RTT_TRAIN_SEED="$SEED"
OUT="models/${WANDB_RUN_NAME}.pt"

# run_full_corpus_siv1_gnn_train.sh guards on MIN_GRAPHS, default 2700 — a stale floor
# taken from the 2,816-dataset compatibility group, not from what actually survives
# caching. The deployed checkpoint's own contract records n_datasets=2651, so the default
# rejects the very corpus it was fitted on. Replace the magic number with the thing it was
# trying to approximate: these draws must be trained on the SAME corpus as the deployed
# checkpoint, or they are not draws of the deployed config and the study compares nothing.
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

# train_near_rtt.py saves to models/{wandb.run.name}.pt, so an unset WANDB_RUN_NAME means
# wandb invents a name and the checkpoint lands somewhere nobody is looking for it.
[[ -n "${WANDB_RUN_NAME}" ]] || { echo "FAIL LOUD: WANDB_RUN_NAME must be set" >&2; exit 1; }
[[ -f "${CACHE_DIR}/graphs.pkl" ]] || { echo "FAIL LOUD: missing ${CACHE_DIR}/graphs.pkl" >&2; exit 1; }

# Not a pure seed study if any of these are set — fail rather than silently produce arms
# that differ by architecture.
for v in NEAR_RTT_MP_RESIDUAL GNN_MP_NODE_EDGES GNN_DISABLE_MESSAGE_PASSING; do
  if [[ -n "${!v:-}" ]]; then
    echo "FAIL LOUD: $v is set to '${!v}'; draws must differ only by seed" >&2
    exit 1
  fi
done

# Single-process data loading. Two reasons, and the second is the better one:
#   1. The cluster's ulimit -n is 1024, and 4 workers sharing tensors by file descriptor
#      exhausts it — job 712372 died 6 batches into epoch 0 on all 8 tasks with EOFError
#      in rebuild_storage_fd.
#   2. This is a reproducibility study. Worker processes introduce an ordering the seed
#      does not control, which is exactly the class of variance the study exists to rule
#      out. __getitem__ only attaches lightweight metadata, so the workers bought little.
export NEAR_RTT_DATALOADER_WORKERS=0

echo "=== task=${SLURM_ARRAY_TASK_ID} seed=${SEED} cache=${CACHE_DIR} -> ${OUT} ==="
bash scripts_cosim/datalab/run_full_corpus_siv1_gnn_train.sh

[[ -f "$OUT" ]] || { echo "FAIL LOUD: checkpoint not written: $OUT" >&2; exit 1; }

# Validity gate: the contract must name THIS seed and confirm the run was reproducible.
# Without both, "8 seeded draws" is 8 files whose names happen to differ.
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
