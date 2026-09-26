#!/bin/bash
# backlog_corpus_v1 live gate, phase bc1 (docs/lineages/backlog_corpus_v1.md): bc1load x4, bc1mpoff x4 and
# v4load_se x4 on the fresh_topo_burst_v1 study, as cd_gap_v1's phase_fix.sbatch (--no-scope, 2.5 GB, 1800 s).
# CD, self-predict and v4load (legacy capture) are the existing runs, licensed by the witness job.
# Inputs: $B/inputs = cfg/, wl/ and selected.json from the fresh inputs, both split files, and models/ holding
# the v4load and bc1 checkpoints with their sidecars (md5-checked against their sources below).
#   sbatch backlog_corpus_v1_gate.sbatch                                   (phase bc1)
#   sbatch --export=ALL,PHASE=bc1selfref backlog_corpus_v1_gate.sbatch    (Amendment 3)
#   sbatch --export=ALL,PHASE=fc1 backlog_corpus_v1_gate.sbatch           (fullctx_refine_v1)
#   sbatch --export=ALL,PHASE=xs1 backlog_corpus_v1_gate.sbatch           (exchange_seconds_v1)
#   sbatch --export=ALL,PHASE=xs1cd backlog_corpus_v1_gate.sbatch         (seeded_cd_xs1_v1)
#SBATCH --job-name=bc1-gate
#SBATCH --output=/home/nikola.lukic/gnn-herosim/simulation_data/backlog_corpus_v1/gates/%x-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/simulation_data/backlog_corpus_v1/gates/%x-%j.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=24:00:00

set -euo pipefail
WT=/home/nikola.lukic/gnn-herosim-wt/backlog_corpus
FRESH=/home/nikola.lukic/gnn-herosim/simulation_data/cd_gap_v1/gates/fresh
B=/home/nikola.lukic/gnn-herosim/simulation_data/backlog_corpus_v1/gates
IN="$B/inputs"
PHASE=${PHASE:-bc1}
OUT="$B/$PHASE"
P=${PAR:-$(( ${SLURM_CPUS_PER_TASK:-64} - 2 ))}
cd "$WT"
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$WT"
python3 -c "import sys,torch;print(sys.executable, torch.__version__)"
echo "HEAD $(git -C "$WT" rev-parse HEAD)  node $(hostname)  P=$P"
DIRTY="$(git -C "$WT" status --porcelain -- src scripts_cosim experiments run_experiment.py)"
if [ -n "$DIRTY" ]; then echo "FAIL LOUD: worktree dirty: $DIRTY"; exit 1; fi
grep -q "default path reproduces CD and v4load to the digit" /home/nikola.lukic/gnn-herosim/logs/bc1-witness-*.out \
  || { echo "FAIL LOUD: no passing witness (backlog_corpus_v1_witness.sbatch)"; exit 1; }

mkdir -p "$IN/models" "$OUT"
for d in cfg wl; do [ -e "$IN/$d" ] || ln -s "$FRESH/$d" "$IN/$d"; done
cp -n "$FRESH/selected.json" "$FRESH/joint_burst_v2_split.json" "$IN/"
cp -n "$WT/experiments/backlog_corpus_v1_split.json" "$IN/"
cmp "$WT/experiments/backlog_corpus_v1_split.json" "$IN/backlog_corpus_v1_split.json" || { echo "FAIL LOUD: stale split copy"; exit 1; }
for s in 1 2 3 4; do
  EXTRA=()
  if [[ $PHASE == fc1 ]]; then EXTRA=("$WT/models/fullctx-refine-v1-fc1load-lr2e3-seed$s"); fi
  if [[ $PHASE == xs1 || $PHASE == xs1cd ]]; then EXTRA=("$WT/models/exchange-seconds-v1-xs1load-lr2e3-seed$s"); fi
  for f in "$FRESH/models/load-repr-v1-v4load-gnnedge0-lr2e3-seed$s" \
           "$WT/models/backlog-corpus-v1-bc1load-lr2e3-seed$s" "$WT/models/backlog-corpus-v1-bc1mpoff-lr2e3-seed$s" \
           "${EXTRA[@]}"; do
    for ext in .pt .contract.json; do
      [ -f "$f$ext" ] || { echo "FAIL LOUD: missing $f$ext"; exit 1; }
      cp -n "$f$ext" "$IN/models/"
      cmp "$f$ext" "$IN/models/$(basename "$f$ext")" || { echo "FAIL LOUD: $IN/models copy of $f$ext differs"; exit 1; }
    done
  done
done

# Logs are not evidence (summaries come from the raw JSON); keep them within the /home quota.
(
  while true; do
    find "$OUT" -maxdepth 1 -name '*.log' -size +100M -exec truncate -s 0 {} + 2>/dev/null || true
    for f in "$OUT"/*.log; do
      [ -e "$f" ] || continue
      b="${f%.log}"
      if { [ -e "$b.summary.json" ] || [ -e "$b.failed.json" ]; } && [ "$(stat -c %s "$f")" -gt 262144 ]; then
        tail -c 262144 "$f" > "$f.t" && mv "$f.t" "$f" || true
      fi
    done
    sleep 30
  done
) &
WD=$!
trap 'kill $WD 2>/dev/null || true' EXIT

rc=0
python3 scripts_cosim/fresh_topo_burst_v1_gate.py "$PHASE" --inputs "$IN" --out "$OUT" \
  --selection "$IN/selected.json" --parallel "$P" --mem 2500M --timeout 1800 --no-scope || rc=$?
kill $WD 2>/dev/null || true
echo "driver rc=$rc"
exit $rc
