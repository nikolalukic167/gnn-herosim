#!/bin/bash
# link_mp_v1 — train all three arm families, seeds 1-16, AT THE PINNED COMMIT.
# Registration: docs/lineages/link_mp_v1.md.
#
# 48 tasks = 3 families x 16 seeds:
#   family 0  lgctrl   plain MP-ON, cache OFF                 (attribution control)
#   family 1  lgmpoff  MP-OFF, cache OFF                      (the pointwise bar)
#   family 2  lgon     MP-ON + core_v1 network entities, cache CORE   (the treatment)
#
# All three train on the SAME corpus through the SAME wrapper with the SAME
# hyperparameters; each family differs from lgctrl by exactly one factor. At a fixed seed
# the shared modules (encoders, GIN, scorer) initialize bit-identically across families —
# the net encoders are constructed AFTER them (gnn_model.py __init__ order), so lgon's
# extra draws never perturb the shared init.
#
# Submit with:  --dependency=afterok:<linkmp-recache job id>
#SBATCH --job-name=linkmp-train
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/linkmp-train-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/linkmp-train-%A_%a.err
#SBATCH --partition=GPU-a40,GPU-l40s
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-47

set -euo pipefail

PIN_DIR=/home/nikola.lukic/gnn-herosim-pin-8aef27a
PIN_COMMIT=8aef27a98fd636000008468d75a52d645f999969
cd "$PIN_DIR"

if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PWD:$PWD/src/notebooks"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
mkdir -p logs models

HEAD_NOW=$(git rev-parse HEAD)
[[ "$HEAD_NOW" == "$PIN_COMMIT" ]] || {
  echo "FAIL LOUD: pinned worktree at $HEAD_NOW, registration requires $PIN_COMMIT" >&2; exit 1; }
DIRTY=$(git status --porcelain -- src scripts_cosim experiments run_experiment.py)
[[ -z "$DIRTY" ]] || {
  echo "FAIL LOUD: pinned worktree dirty under CODE_PATHS:" >&2; echo "$DIRTY" >&2; exit 1; }
echo "[PIN] $PIN_DIR at $PIN_COMMIT, CODE_PATHS clean"

python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__,'cuda',torch.cuda.is_available())"

I="$SLURM_ARRAY_TASK_ID"
FAMS=(lgctrl lgmpoff lgon)
FAM="${FAMS[$(( I / 16 ))]}"
SEED=$(( I % 16 + 1 ))
[[ "$SEED" -ge 1 && "$SEED" -le 16 ]] || { echo "FAIL LOUD: bad seed $SEED" >&2; exit 1; }

CACHE_OFF="simulation_data/graphs_cache_link_mp_v1_dim14"
CACHE_CORE="simulation_data/graphs_cache_link_mp_v1_core_v1_dim14"

# One factor per family; everything else must be UNSET or this is not a clean ablation.
case "$FAM" in
  lgctrl)  export CACHE_DIR="$CACHE_OFF" ;;
  lgmpoff) export CACHE_DIR="$CACHE_OFF"
           export GNN_DISABLE_MESSAGE_PASSING=1 ;;
  lgon)    export CACHE_DIR="$CACHE_CORE"
           export NEAR_RTT_MP_NETWORK_ENTITIES=1
           export NETWORK_GRAPH_CONTRACT=core_v1 ;;
  *) echo "FAIL LOUD: bad family $FAM" >&2; exit 1 ;;
esac
for v in NEAR_RTT_MP_RESIDUAL NEAR_RTT_MP_NODE_EDGES GNN_MP_NODE_EDGES \
         NEAR_RTT_MP_DAG_EDGES NEAR_RTT_TASK_TYPE_ONEHOT NEAR_RTT_PARTIAL_STATE_EDGES; do
  if [[ -n "${!v:-}" ]]; then
    echo "FAIL LOUD: $v is set to '${!v}'; families must differ only by their one factor" >&2
    exit 1
  fi
done

export WANDB_RUN_NAME="gnn-linkmp-${FAM}-s${SEED}"
export NEAR_RTT_TRAIN_SEED="$SEED"
OUT="models/${WANDB_RUN_NAME}.pt"
[[ ! -f "$OUT" ]] || { echo "FAIL LOUD: $OUT already exists — refusing to overwrite a draw" >&2; exit 1; }

# The two caches must still pair up at train time, and MIN_GRAPHS pins the corpus size the
# wrapper then re-checks.
export MIN_GRAPHS
MIN_GRAPHS=$(python3 - "$CACHE_OFF" "$CACHE_CORE" <<'PY'
import pickle, sys
off = len(pickle.load(open(f"{sys.argv[1]}/graphs.pkl", "rb")))
core = len(pickle.load(open(f"{sys.argv[2]}/graphs.pkl", "rb")))
if off != core:
    raise SystemExit(f"FAIL LOUD: cache sizes differ: off={off} core={core}")
if off < 1500:
    raise SystemExit(f"FAIL LOUD: only {off} graphs (< 1500)")
print(off)
PY
) || { echo "FAIL LOUD: cache pairing check failed" >&2; exit 1; }
echo "[CORPUS] ${MIN_GRAPHS} graphs, both caches equal"

# Single-process data loading — reproducibility, and the cluster ulimit (job 712372).
export NEAR_RTT_DATALOADER_WORKERS=0

echo "=== task=${I} fam=${FAM} seed=${SEED} cache=${CACHE_DIR} -> ${OUT} (pinned ${PIN_COMMIT:0:12}) ==="
bash scripts_cosim/datalab/run_full_corpus_siv1_gnn_train.sh

[[ -f "$OUT" ]] || { echo "FAIL LOUD: checkpoint not written: $OUT" >&2; exit 1; }

# Validity gate: the sidecar must name this seed, be reproducible, and record exactly the
# graph its family trained on.
python3 - "$OUT" "$SEED" "$FAM" <<'PY'
import json, sys
from pathlib import Path
out, want_seed, fam = sys.argv[1], int(sys.argv[2]), sys.argv[3]
sidecar = Path(out).with_suffix(".contract.json")
if not sidecar.is_file():
    raise SystemExit(f"FAIL LOUD: {sidecar} missing — a checkpoint without a contract is not evidence")
c = json.load(open(sidecar))
if int(c.get("train_seed", -1)) != want_seed:
    raise SystemExit(f"FAIL LOUD: {out} contract train_seed={c.get('train_seed')} != {want_seed}")
if not c.get("deterministic_algorithms"):
    raise SystemExit(f"FAIL LOUD: {out} trained without deterministic algorithms; not a reproducible draw")
ngc = c.get("network_graph_contract")
if fam == "lgon" and ngc != "core_v1":
    raise SystemExit(f"FAIL LOUD: lgon sidecar records network_graph_contract={ngc!r}, expected 'core_v1'")
if fam != "lgon" and ngc not in (None, "off"):
    raise SystemExit(f"FAIL LOUD: {fam} sidecar records network_graph_contract={ngc!r}; the control leaked the treatment")
print(f"[OK] {out} fam={fam} seed={want_seed} layout={c.get('inference_feature_layout')} "
      f"queue={c.get('queue_feature_contract')} network_graph={ngc}")
PY
echo "=== task=${I} fam=${FAM} seed=${SEED} COMPLETE ==="
