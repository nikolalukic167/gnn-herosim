#!/bin/bash
# P5b (program_verdict_v1): the candidate-relative MLP retrains — one per cache, so each
# has a byte-identical baseline to be paired against on the 30 gate cells:
#
#   dim25cr on graphs_cache_full_corpus_siv1_dim14          -> pairs with the `mlp` arm
#   dim25cr on graphs_cache_full_corpus_siv1_dim14_tempfix  -> pairs with `mlptempfix`
#
# Mirrors fc_siv1_mlp_tempfix.sbatch / run_full_corpus_siv1_mlp_train.sh exactly except
# for the cache/output/run-name and --candidate-relative-queue. Hyperparameters are copied
# verbatim (hidden 64, lr 1e-3, seed 42, test-size 0.2, epochs 100, patience 10): the
# paired gate is an A/B on the FEATURE SET, so nothing else may move.
#
# The trainer prints and records validity gate 2 (does the model actually use the new
# columns?). Read it before submitting the array — a null from an inert feature is
# evidence about nothing. Locally, at 2 epochs on the dim14 cache, it was 29.1%.
#SBATCH --job-name=fc-siv1-mlp-candrel
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/fc-siv1-mlp-candrel-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/fc-siv1-mlp-candrel-%j.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PWD:$PWD/src/notebooks"
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16
mkdir -p logs models/tabular

python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"

train_one () {
  local cache="$1" out="$2" runname="$3"
  [[ -f "${cache}/graphs.pkl" ]] || { echo "FAIL LOUD: missing ${cache}/graphs.pkl" >&2; exit 1; }
  echo "=== train ${runname}  cache=${cache} ==="
  python3 -u src/policy/tabular/train_mlp_dim22_from_batch.py \
    --cache-dir "$cache" \
    --output "$out" \
    --candidate-relative-queue \
    --epochs 100 --patience 10 --hidden-dim 64 --lr 1e-3 --random-state 42 --test-size 0.2 \
    --wandb-project gnn-full-corpus-siv1-aug2026 \
    --wandb-run-name "$runname" \
    --wandb-entity nikolalukic167-tu-wien
  [[ -f "$out" ]] || { echo "FAIL LOUD: checkpoint not written: $out" >&2; exit 1; }

  # Fail here rather than after 30 gate runs: a checkpoint that does not declare dim25cr
  # would be served under a guessed layout, and one whose CR columns move no decisions
  # makes the whole control vacuous.
  python3 - "$out" <<'PY'
import json, sys
meta = json.load(open(sys.argv[1] + ".meta.json"))
layout = meta["inference_feature_layout"]
change = meta["candidate_relative_ablation_argmax_change"]
if layout != "dim25cr" or not meta.get("candidate_relative"):
    raise SystemExit(f"FAIL LOUD: {sys.argv[1]} declares layout={layout!r} candidate_relative="
                     f"{meta.get('candidate_relative')!r}; expected dim25cr / True")
print(f"[VALIDITY] {sys.argv[1]}: layout={layout} test_acc={meta['test_edge_accuracy']:.4f} "
      f"cr_ablation_argmax_change={change:.1%} (pre-registered gate: >= 5%)")
if change is None or change < 0.05:
    raise SystemExit("FAIL LOUD: validity gate 2 — the candidate-relative columns move "
                     f"only {change:.1%} of held-out argmaxes. The feature is inert; fix "
                     "the representation and retrain. DO NOT gate this checkpoint.")
PY
}

train_one simulation_data/graphs_cache_full_corpus_siv1_dim14 \
  models/tabular/batch_edge_mlp_full_corpus_siv1_dim25cr_batchcache.pt \
  batch-edge-mlp-full-corpus-siv1-dim25cr

train_one simulation_data/graphs_cache_full_corpus_siv1_dim14_tempfix \
  models/tabular/batch_edge_mlp_full_corpus_siv1_dim25cr_batchcache_tempfix.pt \
  batch-edge-mlp-full-corpus-siv1-dim25cr-tempfix

ls -la models/tabular/batch_edge_mlp_full_corpus_siv1_dim25cr_batchcache*.pt*
echo "=== P5b candrel trains complete — both validity gates passed ==="
