#!/bin/bash
# r1_attribution_v1 PRODUCTION corpus, step 1 of 3 -- PREPARED, NOT SUBMITTED (rungs come from S4).
# Builds the WF1 inputs (W2 payloads, W4 mix, single origin, 1 s window, then W3 access classes) for the TRAINING topologies at the
# given rungs, into $ROOT/inputs. One job, run once.
#   RUNGS="lo=0.2666,hi=11.6139"-style list of tag=multiplier from S4 (no default: fail loud)
#   TOPOS="9201 9202 ..."  training topologies (the 19 test topologies are excluded by the caller)
#   CFGDIR                 directory of cc40s<seed>.json cell configs for those topologies
#   ROOT                   output root, must not exist
#   sbatch --export=ALL,WT=...,ROOT=...,CFGDIR=...,TOPOS="...",RUNGS="lo=...,hi=..." wf1_corpus_prod_inputs.sbatch
#SBATCH --job-name=wf1p-in
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/wf1p-in-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
set -euo pipefail
WT=${WT:?FAIL LOUD: set WT (pinned worktree)}
ROOT=${ROOT:?FAIL LOUD: set ROOT}
CFGDIR=${CFGDIR:?FAIL LOUD: set CFGDIR}
TOPOS=${TOPOS:?FAIL LOUD: set TOPOS}
RUNGS=${RUNGS:?FAIL LOUD: set RUNGS (tag=multiplier,... from S4; no default)}
X1=/home/nikola.lukic/gnn-herosim/simulation_data/small_batch_confirm_v1/inputs/grounded/wl
cd "$WT"
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT"
echo "HEAD $(git rev-parse HEAD)"
[[ ! -e "$ROOT" ]] || { echo "FAIL LOUD: $ROOT exists"; exit 1; }
mkdir -p "$ROOT"
TAGS=""
for spec in ${RUNGS//,/ }; do
  tag=${spec%%=*}; m=${spec##*=}
  factor=$(python3 -c "print(repr(1.0/$m))")
  python3 scripts_cosim/workload_fix_v1_build.py --grounded-wl "$X1" --cfg-dir "$CFGDIR" --topologies $TOPOS --rung "$tag=$factor" \
    --payload-sampler wf1_v1 --task-mix wf1_v1 --batch-timeout-fixed 1 --out "$ROOT/_b_$tag"
  mkdir -p "$ROOT/inputs_w24"; mv "$ROOT/_b_$tag/wf1_$tag" "$ROOT/inputs_w24/wf1_$tag"; cp "$ROOT/_b_$tag/manifest.json" "$ROOT/inputs_w24/manifest_$tag.json"
  TAGS="$TAGS $tag"
done
python3 scripts_cosim/workload_fix_v1_w3_build.py --w2-inputs "$ROOT/inputs_w24" --tags $TAGS --out "$ROOT/inputs" --diff-out "$ROOT/w3_vs_w2_diff.json"
echo "inputs ready: $ROOT/inputs (rungs:$TAGS)"
