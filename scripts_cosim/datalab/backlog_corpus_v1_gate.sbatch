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
#   sbatch --export=ALL,PHASE=capacity,WT=<capacity worktree> backlog_corpus_v1_gate.sbatch   (capacity_sweep_v1)
#   sbatch --export=ALL,PHASE=ladder,WT=<burst_ladder worktree>,TIMEOUT=2700 backlog_corpus_v1_gate.sbatch  (burst_ladder_v1)
#   sbatch --export=ALL,PHASE=ladderjit,WT=<burst_ladder worktree>,TIMEOUT=2700 backlog_corpus_v1_gate.sbatch  (burst_ladder_v1 Amendment 1)
#   sbatch --export=ALL,PHASE=guard,WT=<replica_guard worktree>,TIMEOUT=2700 backlog_corpus_v1_gate.sbatch  (replica_guard_v1; needs the x10 draws built)
#   sbatch --export=ALL,PHASE=decimatune,WT=<decima worktree> backlog_corpus_v1_gate.sbatch            (decima_rule_v1 tuning)
#   sbatch --export=ALL,PHASE=decima,WT=<decima worktree>,HEROSIM_DECIMA_ALPHA=<tuned> backlog_corpus_v1_gate.sbatch
#   sbatch --export=ALL,PHASE=x11confirm backlog_corpus_v1_gate.sbatch       (x11_confirm_v1)
#   sbatch --export=ALL,PHASE=grounded,WT=<grounded worktree> backlog_corpus_v1_gate.sbatch   (grounded_workload_v1)
#   sbatch --export=ALL,PHASE=x15fill,WT=<grounded worktree>,TIMEOUT=2700 backlog_corpus_v1_gate.sbatch      (peak_load_v1 a)
#   sbatch --export=ALL,PHASE=groundedx15,WT=<grounded worktree>,TIMEOUT=2700 backlog_corpus_v1_gate.sbatch  (peak_load_v1 b)
#   sbatch --export=ALL,PHASE=groundedladder,WT=<grounded worktree>,TIMEOUT=2700 backlog_corpus_v1_gate.sbatch  (peak_load_v2)
#   sbatch --export=ALL,PHASE=peakctl,WT=<grounded worktree>,TIMEOUT=2700 backlog_corpus_v1_gate.sbatch   (peak_controls_v1 CD+externality)
#   sbatch --export=ALL,PHASE=peakmlp,WT=<grounded worktree>,TIMEOUT=2700 backlog_corpus_v1_gate.sbatch   (peak_controls_v1 MP-OFF twin)
#   sbatch --export=ALL,PHASE=rawplan,WT=<grounded worktree>,TIMEOUT=2700 backlog_corpus_v1_gate.sbatch   (raw_plan_v1; or PHASE=rawgnn / rawmlp)
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
WT=${WT:-/home/nikola.lukic/gnn-herosim-wt/backlog_corpus}
TIMEOUT=${TIMEOUT:-1800}
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
  if [[ $PHASE == xs1 || $PHASE == xs1cd || $PHASE == ladder || $PHASE == jitsmoke || $PHASE == ladderjit || $PHASE == capacity || $PHASE == guard || $PHASE == x11confirm || $PHASE == grounded || $PHASE == x15fill || $PHASE == groundedx15 || $PHASE == groundedladder || $PHASE == peakctl || $PHASE == peakmlp || $PHASE == rawplan || $PHASE == rawgnn || $PHASE == rawmlp ]]; then EXTRA=("$WT/models/exchange-seconds-v1-xs1load-lr2e3-seed$s"); fi
  if [[ $PHASE == rawplan || $PHASE == rawgnn ]]; then EXTRA+=("$WT/models/raw-plan-v1-rawgnn-lr2e3-seed$s"); fi
  if [[ $PHASE == rawplan || $PHASE == rawmlp ]]; then EXTRA+=("$WT/models/raw-plan-v1-rawmlp-lr2e3-seed$s"); fi
  if [[ $PHASE == peakmlp ]]; then EXTRA+=("$WT/models/peak-controls-v1-xs1mpoff-lr2e3-seed$s"); fi
  if [[ $PHASE == grounded || $PHASE == groundedx15 ]]; then EXTRA+=("$FRESH/models/joint-burst-v2-gnnedge0-lr2e3-seed$s" "$FRESH/models/joint-burst-v2-mpoff-lr2e3-seed$s"); fi
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

# burst_ladder_v1: w0 at intensity x1 / x1.5 / x2 (timestamps and policy time constants x1, x2/3, x1/2),
# built once into $IN/ladder/x<rung> from $IN; the existing inputs are never rewritten.
if [[ $PHASE == ladder ]]; then
  TOPOS=$(python3 -c "import json,sys;print(' '.join(map(str,json.load(open(sys.argv[1]))['topologies'])))" "$IN/selected.json")
  for pair in x10:1.0 x15:0.6666666666666666 x20:0.5; do
    R=${pair%%:*}; F=${pair##*:}; D="$IN/ladder/$R"
    if [[ ! -f "$D/.built" ]]; then
      rm -rf "$D"; mkdir -p "$D"
      python3 scripts_cosim/cd_gap_v1_build_b.py --src "$IN" --out "$D" --factor "$F" --topologies $TOPOS
      ln -s "$IN/backlog_corpus_v1_split.json" "$D/backlog_corpus_v1_split.json"
      find "$D/wl" -name '*.json' ! -name 'burst_drainable_f4000_n50000.json' -delete
      touch "$D/.built"
    fi
  done
fi

# burst_ladder_v1 Amendment 1: perturbed draws d1-d4 of the x1 and x1.5 rungs (burst_ladder_jitter.py), built once.
if [[ $PHASE == jitsmoke || $PHASE == ladderjit ]]; then
  for R in x10 x15; do
    [[ -f "$IN/ladder/$R/.built" ]] || { echo "FAIL LOUD: rung $R not built (run PHASE=ladder first)"; exit 1; }
    for k in 1 2 3 4; do
      D="$IN/ladder/${R}d$k"
      if [[ ! -f "$D/.built" ]]; then
        rm -rf "$D"
        python3 scripts_cosim/burst_ladder_jitter.py --src "$IN/ladder/$R" --out "$D" --seed "$k"
      fi
    done
  done
fi

# capacity_sweep_v1: rungs x1.1 / x1.2 / x1.3 built like the ladder (factor 1/intensity), then the same 4 draws;
# the x1 draws must already exist (burst_ladder_v1 Amendment 1). Built once; existing inputs never rewritten.
if [[ $PHASE == capacity ]]; then
  TOPOS=$(python3 -c "import json,sys;print(' '.join(map(str,json.load(open(sys.argv[1]))['topologies'])))" "$IN/selected.json")
  for k in 1 2 3 4; do [[ -f "$IN/ladder/x10d$k/.built" ]] || { echo "FAIL LOUD: x10d$k not built (burst_ladder_v1 ladderjit)"; exit 1; }; done
  for pair in x11:0.9090909090909091 x12:0.8333333333333334 x13:0.7692307692307692; do
    R=${pair%%:*}; F=${pair##*:}; D="$IN/ladder/$R"
    if [[ ! -f "$D/.built" ]]; then
      rm -rf "$D"; mkdir -p "$D"
      python3 scripts_cosim/cd_gap_v1_build_b.py --src "$IN" --out "$D" --factor "$F" --topologies $TOPOS
      ln -s "$IN/backlog_corpus_v1_split.json" "$D/backlog_corpus_v1_split.json"
      find "$D/wl" -name '*.json' ! -name 'burst_drainable_f4000_n50000.json' -delete
      touch "$D/.built"
    fi
    for k in 1 2 3 4; do
      DD="$IN/ladder/${R}d$k"
      if [[ ! -f "$DD/.built" ]]; then
        rm -rf "$DD"
        python3 scripts_cosim/burst_ladder_jitter.py --src "$D" --out "$DD" --seed "$k"
      fi
    done
  done
fi

# x11_confirm_v1 reuses the x1.1 draws exactly as capacity_sweep_v1 built them.
if [[ $PHASE == x11confirm ]]; then
  for k in 1 2 3 4; do
    [[ -f "$IN/ladder/x11d$k/.built" ]] || { echo "FAIL LOUD: draw x11d$k not built (capacity_sweep_v1)"; exit 1; }
  done
fi

# grounded_workload_v1: the four Alibaba-grounded windows, minted here from the committed fan-out libraries and the
# study's pre-burst windows; every file must match the sha256 minted locally (docs/lineages/grounded_workload_v1/).
# Its witness reruns xs1load_selfref seed 1 on two x1.1 draws, so those draws must exist.
if [[ $PHASE == grounded || $PHASE == groundedx15 || $PHASE == groundedladder ]]; then
  export HEROSIM_DECIMA_ALPHA=1.0
  if [[ $PHASE == grounded ]]; then
    [[ -f "$IN/ladder/x11d1/.built" ]] || { echo "FAIL LOUD: draw x11d1 not built (capacity_sweep_v1)"; exit 1; }
  fi
  G="$IN/grounded/wl"; BASEWL=/home/nikola.lukic/gnn-herosim/simulation_data/peer_affinity_live_gate/workloads
  LIB="$WT/docs/lineages/grounded_workload_v1"
  mkdir -p "$G"
  i=0
  for pair in 0:drainable_f4000_n50000 50:drainable_w1_n50000 100:drainable_w2_n50000 130:drainable_w3_n50000; do
    K=${pair%%:*}; BW=${pair##*:}
    [ -f "$G/grounded_g${i}_n50000.json" ] || python3 scripts_cosim/grounded_workload_v1_mint.py \
      --lib "$LIB/fanout_lib_$K.json.gz" --base "$BASEWL/$BW.json" --seed $((7400 + i)) --out "$G/grounded_g${i}_n50000.json" >/dev/null
    i=$((i + 1))
  done
  (cd "$G" && sha256sum -c "$LIB/workloads.sha256") || { echo "FAIL LOUD: grounded workloads differ from the local mint"; exit 1; }
fi

# peak_load_v1 (b): the grounded windows at x1.5, built with the ladder protocol (cd_gap_v1_build_b.py, factor 2/3 on
# timestamps and policy time constants) from a source dir that pairs the study cells with the grounded workloads.
if [[ $PHASE == groundedx15 ]]; then
  D="$IN/grounded_x15"
  if [[ ! -f "$D/.built" ]]; then
    SRC="$IN/grounded_src"; rm -rf "$SRC" "$D"; mkdir -p "$SRC" "$D"
    ln -s "$IN/cfg" "$SRC/cfg"; ln -s "$IN/grounded/wl" "$SRC/wl"; ln -s "$IN/models" "$SRC/models"
    ln -s "$IN/joint_burst_v2_split.json" "$SRC/joint_burst_v2_split.json"
    TOPOS=$(python3 -c "import json,sys;print(' '.join(map(str,json.load(open(sys.argv[1]))['topologies'])))" "$IN/selected.json")
    python3 scripts_cosim/cd_gap_v1_build_b.py --src "$SRC" --out "$D" --factor 0.6666666666666666 --topologies $TOPOS
    ln -s "$IN/backlog_corpus_v1_split.json" "$D/backlog_corpus_v1_split.json"
    [ "$(ls "$D"/wl/grounded_g*_n50000.json | wc -l)" = 4 ] || { echo "FAIL LOUD: grounded x1.5 build incomplete"; exit 1; }
    touch "$D/.built"
  fi
fi

# peak_load_v2: the grounded windows at x2 / x3 / x5, built exactly as (b)'s x1.5.
if [[ $PHASE == groundedladder ]]; then
  SRC="$IN/grounded_src"
  [ -L "$SRC/wl" ] || { echo "FAIL LOUD: $SRC missing (build groundedx15 first)"; exit 1; }
  TOPOS=$(python3 -c "import json,sys;print(' '.join(map(str,json.load(open(sys.argv[1]))['topologies'])))" "$IN/selected.json")
  for pair in x20:0.5 x30:0.3333333333333333 x50:0.2; do
    R=${pair%%:*}; F=${pair##*:}; D="$IN/grounded_$R"
    if [[ ! -f "$D/.built" ]]; then
      rm -rf "$D"; mkdir -p "$D"
      python3 scripts_cosim/cd_gap_v1_build_b.py --src "$SRC" --out "$D" --factor "$F" --topologies $TOPOS
      ln -s "$IN/backlog_corpus_v1_split.json" "$D/backlog_corpus_v1_split.json"
      [ "$(ls "$D"/wl/grounded_g*_n50000.json | wc -l)" = 4 ] || { echo "FAIL LOUD: grounded $R build incomplete"; exit 1; }
      touch "$D/.built"
    fi
  done
fi

# peak_controls_v1 reuses the x2 / x3 / x5 cells exactly as peak_load_v2 built them.
if [[ $PHASE == peakctl || $PHASE == peakmlp || $PHASE == rawplan || $PHASE == rawgnn || $PHASE == rawmlp ]]; then
  for R in x20 x30 x50; do
    [[ -f "$IN/grounded_$R/.built" ]] || { echo "FAIL LOUD: grounded_$R not built (peak_load_v2 groundedladder)"; exit 1; }
  done
fi

# peak_load_v1 (a) reuses the x1.5 draws exactly as burst_ladder_v1 Amendment 1 built them.
if [[ $PHASE == x15fill ]]; then
  export HEROSIM_DECIMA_ALPHA=1.0
  for k in 1 2 3 4; do
    [[ -f "$IN/ladder/x15d$k/.built" ]] || { echo "FAIL LOUD: draw x15d$k not built (burst_ladder_v1 ladderjit)"; exit 1; }
  done
fi

# replica_guard_v1 reuses the x1 draws exactly as burst_ladder_v1 Amendment 1 built them.
if [[ $PHASE == guard ]]; then
  for k in 1 2 3 4; do
    [[ -f "$IN/ladder/x10d$k/.built" ]] || { echo "FAIL LOUD: draw x10d$k not built (burst_ladder_v1 ladderjit)"; exit 1; }
  done
fi

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
  --selection "$IN/selected.json" --parallel "$P" --mem 2500M --timeout "$TIMEOUT" --no-scope || rc=$?
kill $WD 2>/dev/null || true
echo "driver rc=$rc"
exit $rc
