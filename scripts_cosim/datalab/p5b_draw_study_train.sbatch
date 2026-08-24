#!/bin/bash
# P5b follow-up — the draw study. Resolves the confound P5b closed on.
#
# P5b found the candidate-relative feature moving collapse 7/30 -> 2/30 on one cache and
# 7/30 -> 17/30 on the other. Both arms passed --random-state 42, so "cache" and "training
# draw" were confounded. Then it turned out to be worse than that: nothing seeded torch, so
# --random-state pinned only the parent split and the batch order and the model's weight
# INIT was drawn from OS entropy. Every MLP checkpoint in this repo before 2026-08-24 is an
# unreproducible draw -- including the two the 2026-08-23 subsection describes as "an A/B on
# training data alone".
#
# torch.manual_seed is now wired to --random-state, so seeds mean something. This trains a
# full 4x4 grid so the variance can finally be attributed:
#
#     {dim14, dim14_tempfix} x {dim22, dim25cr} x seeds {1,2,3,4} = 16 checkpoints
#
# The dim22 arms are NOT optional padding: if a fixed cache+layout swings its own collapse
# count across seeds by as much as the feature appeared to move it, then the feature effect
# was never an effect and the whole reliability record is a statement about draws.
#
# Existing s42 checkpoints are kept as-is and treated as an unseeded 5th draw; they cannot
# be reproduced by any seed now, which is exactly the point.
#SBATCH --job-name=p5b-draw-train
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/p5b-draw-train-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/p5b-draw-train-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --array=0-15

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PWD:$PWD/src/notebooks"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
mkdir -p logs models/tabular/draw_study

python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"

I="$SLURM_ARRAY_TASK_ID"
SEEDS=(1 2 3 4)
SEED="${SEEDS[$(( I % 4 ))]}"
COMBO=$(( I / 4 ))          # 0..3 = {cache} x {layout}

case "$COMBO" in
  0) CACHE=simulation_data/graphs_cache_full_corpus_siv1_dim14;         TAG=dim14;   LAYOUT=dim22   ;;
  1) CACHE=simulation_data/graphs_cache_full_corpus_siv1_dim14;         TAG=dim14;   LAYOUT=dim25cr ;;
  2) CACHE=simulation_data/graphs_cache_full_corpus_siv1_dim14_tempfix; TAG=tempfix; LAYOUT=dim22   ;;
  3) CACHE=simulation_data/graphs_cache_full_corpus_siv1_dim14_tempfix; TAG=tempfix; LAYOUT=dim25cr ;;
  *) echo "FAIL LOUD: bad combo $COMBO (array must be 0-15)" >&2; exit 1 ;;
esac

CR_FLAG=""
[[ "$LAYOUT" == "dim25cr" ]] && CR_FLAG="--candidate-relative-queue"

OUT="models/tabular/draw_study/mlp_${TAG}_${LAYOUT}_s${SEED}.pt"
[[ -f "${CACHE}/graphs.pkl" ]] || { echo "FAIL LOUD: missing ${CACHE}/graphs.pkl" >&2; exit 1; }

echo "=== task=${I} cache=${TAG} layout=${LAYOUT} seed=${SEED} -> ${OUT} ==="
python3 -u src/policy/tabular/train_mlp_dim22_from_batch.py \
  --cache-dir "$CACHE" \
  --output "$OUT" \
  $CR_FLAG \
  --epochs 100 --patience 10 --hidden-dim 64 --lr 1e-3 --random-state "$SEED" --test-size 0.2 \
  --wandb-project gnn-full-corpus-siv1-aug2026 \
  --wandb-run-name "p5b-draw-${TAG}-${LAYOUT}-s${SEED}" \
  --wandb-entity nikolalukic167-tu-wien

[[ -f "$OUT" ]] || { echo "FAIL LOUD: checkpoint not written: $OUT" >&2; exit 1; }

python3 - "$OUT" "$LAYOUT" <<'PY'
import json, sys
out, want = sys.argv[1], sys.argv[2]
m = json.load(open(out + ".meta.json"))
if m["inference_feature_layout"] != want:
    raise SystemExit(f"FAIL LOUD: {out} declares {m['inference_feature_layout']!r}, wanted {want!r}")
if not m.get("torch_seeded"):
    raise SystemExit(f"FAIL LOUD: {out} was not torch-seeded — the draw study needs reproducible inits")
cr = m.get("candidate_relative_ablation_argmax_change")
print(f"[OK] {out} layout={want} seed={m['random_state']} test_acc={m['test_edge_accuracy']:.4f} "
      f"cr_ablation={cr if cr is None else format(cr, '.3f')}")
# Validity gate 2 still applies to every candidate-relative arm.
if want == "dim25cr" and (cr is None or cr < 0.05):
    raise SystemExit(f"FAIL LOUD: validity gate 2 — CR columns move only {cr} of argmaxes")
PY
echo "=== task=${I} COMPLETE ==="
