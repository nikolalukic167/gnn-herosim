#!/bin/bash
# r1_attribution_v1 PRODUCTION corpus, step 1 of 3 -- submit only on the coordinator's word.
# Builds the WF1 inputs (W2 payloads, W4 mix, single origin, 1 s window, then W3 access classes) for the corpus topologies (train and held-out) at the
# three load rungs (light x1.2584, moderate x5.9402, heavy x11.6139), into $ROOT/inputs. One job, run once.
#   RUNGS="light=1.2584,moderate=5.9402,heavy=11.6139"  tag=multiplier list (no default: fail loud); export it in the shell, not via --export
#   TOPOS="9201 9202 ..."  corpus topologies (the 19 test topologies are excluded by the caller)
#   CFGDIR                 directory of cc40s<seed>.json cell configs for those topologies (not with SCALE)
#   SCALE="160:24:0.6"     scale_160_v1: mint each topology's cfg from BASECFG with these clients:servers:p (scale_probe_cfg.py, into
#                          $ROOT/cfg_src) and spread the single-origin windows over the clients after W3 (scale_probe_remap_clients.py).
#                          RUNGS are then the final multipliers at this size (factor = 1/m); no servers/6 scaling is applied here.
#   ROOT                   output root, must not exist
#   sbatch --export=ALL,WT=...,ROOT=...,CFGDIR=...,TOPOS="...",RUNGS="light=...,moderate=...,heavy=..." wf1_corpus_prod_inputs.sbatch
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
SCALE=${SCALE:-}
if [[ -n "$SCALE" ]]; then
  [[ -z "${CFGDIR:-}" ]] || { echo "FAIL LOUD: SCALE mints its own cfgs; unset CFGDIR"; exit 1; }
  BASECFG=${BASECFG:-/home/nikola.lukic/gnn-herosim/simulation_data/workload_fix_v1/corpus_prod_prov/cfg_src/cc40s9101.json}
  IFS=: read -r NC NS P <<< "$SCALE"; [[ -n "$NC" && -n "$NS" && -n "$P" ]] || { echo "FAIL LOUD: SCALE=$SCALE, want clients:servers:p"; exit 1; }
else
  CFGDIR=${CFGDIR:?FAIL LOUD: set CFGDIR}
fi
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
# the root may be on /share: prove a compute node can write there before building anything
echo ok > "$ROOT/.write_test" && [[ "$(cat "$ROOT/.write_test")" == ok ]] && rm -f "$ROOT/.write_test" || { echo "FAIL LOUD: cannot write under $ROOT"; exit 1; }
df -P -BG "$ROOT" | tail -1
if [[ -n "$SCALE" ]]; then
  CFGDIR="$ROOT/cfg_src"
  for t in $TOPOS; do python3 scripts_cosim/scale_probe_cfg.py "$BASECFG" "$CFGDIR" --clients "$NC" --servers "$NS" --p "$P" --seed "$t" >/dev/null; done
  echo "minted $(ls "$CFGDIR" | wc -l) cfgs at ${NC}c x ${NS}s p=$P from $BASECFG"
fi
# the train / held-out split of TOPOS, written beside the inputs (HELDOUT_TOPOS: r1_attribution_v1 'held-out 9297-9320' intersected with TOPOS)
if [[ -n "${HELDOUT_TOPOS:-}" ]]; then
  python3 - "$ROOT/split.json" "$TOPOS" "$HELDOUT_TOPOS" <<'PY'
import json, sys
out, topos, held = sys.argv[1], sys.argv[2].split(), sys.argv[3].split()
assert set(held) <= set(topos), sorted(set(held) - set(topos))
import os
json.dump({"source": os.environ.get("SPLIT_SOURCE", "r1_attribution_v1: held-out 9297-9320 intersected with corpus_prod_prov/feasible_topologies.txt"),
           "heldout": sorted(held, key=int), "train": sorted(set(topos) - set(held), key=int)}, open(out, "w"), indent=1)
PY
fi
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
if [[ -n "$SCALE" ]]; then
  for tag in $TAGS; do for f in "$ROOT/inputs/wf1_$tag/wl/"*.json; do
    python3 scripts_cosim/scale_probe_remap_clients.py "$f" "$f.tmp" --clients "$NC" >/dev/null && mv "$f.tmp" "$f"
  done; done
  echo "remapped the windows over $NC clients"
fi
echo "inputs ready: $ROOT/inputs (rungs:$TAGS)"
