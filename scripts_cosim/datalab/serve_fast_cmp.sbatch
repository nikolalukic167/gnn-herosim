#!/bin/bash
# rp/gnn-serve-fast: the same Arm 1b k = 1 cells (2 topologies x both rungs x seeds 1, 2, window g0) run under several flag configurations on ONE node.
#   MODE=identity  every cell runs under the identity probe (one pickle per cell per config) -> serve_fast_compare.py, serve_fast_sim_identical.py
#   MODE=timing    no probe; each config is run twice, in mirrored order, so node load is balanced across configs
# CONFIGS is a space-separated list over A (all flags off) B (BUILD) C (COLUMNS) D (ROWS) F (BUILD+COLUMNS) E (all on). Diagnostic only: never a gate cell, never reused.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/share/nikola.lukic/uncut_arm2/servefast/logs/%x-%j.out
set -uo pipefail
P=/share/nikola.lukic/uncut_arm2/servefast
: "${WT:?}" "${EXPECT_HEAD:?}" "${MODE:?}" "${CONFIGS:?}"
echo "[sf] host=$(hostname) job=$SLURM_JOB_ID mode=$MODE configs=$CONFIGS model=$(lscpu | sed -n 's/^Model name:[[:space:]]*//p') tpc=$(lscpu | sed -n 's/^Thread(s) per core:[[:space:]]*//p') smt=$(cat /sys/devices/system/cpu/smt/control) nproc=$(nproc) cpuset=$(taskset -pc $$ | sed 's/.*: //')"
squeue -w "$(hostname)" -h -o "%i %u %P %C" | sed 's/^/[sf] on node: /'
export HEROSIM_GNN_DEVICE=cpu
export R1A_EXPECT_SMT=$(cat /sys/devices/system/cpu/smt/control) R1A_EXPECT_CPU_MODEL="$(lscpu | sed -n 's/^Model name:[[:space:]]*//p')"
export IN=$P/in SELECT=/share/nikola.lukic/uncut_groups/arm1_selected.json
export R1A_ARMS=ra_gnn_eng_arm1b_nosplit_k100 R1A_SEEDS=1,2 R1A_TOPOS=19537,19538 R1A_WINDOWS=g0 WF1_RUNGS=moderate,heavy PAR=8 CELL_MEM=16G LIMIT=14400 R1A_SPLIT=uncut_arm1_split.json R1A_SLATE=declared_pruning_v1_uncapped
n=0
for c in $CONFIGS; do
  n=$((n+1)); tag="${MODE}_${n}_${c}"
  unset SERVEFAST_BUILD SERVEFAST_COLUMNS SERVEFAST_ROWS SERVEFAST_PROBE_OUT
  case "$c" in
    A) ;;
    B) export SERVEFAST_BUILD=1 ;;
    C) export SERVEFAST_COLUMNS=1 ;;
    D) export SERVEFAST_ROWS=1 ;;
    F) export SERVEFAST_BUILD=1 SERVEFAST_COLUMNS=1 ;;
    E) export SERVEFAST_BUILD=1 SERVEFAST_COLUMNS=1 SERVEFAST_ROWS=1 ;;
    *) echo "FAIL LOUD: config $c"; exit 1 ;;
  esac
  [[ "$MODE" == identity ]] && { mkdir -p "$P/probe/$tag"; export SERVEFAST_PROBE_OUT="$P/probe/$tag/p"; }
  export OUT=$P/out/$tag; mkdir -p "$OUT"
  t0=$(date +%s)
  echo "[sf] config $c ($tag) flags: BUILD=${SERVEFAST_BUILD:-0} COLUMNS=${SERVEFAST_COLUMNS:-0} ROWS=${SERVEFAST_ROWS:-0}"
  bash "$WT/scripts_cosim/datalab/r1a_gate.sbatch" 2>&1 | grep -E "done|failed|summaries|FAIL|rror" | cut -c1-200
  echo "[sf] config $c wall $(( $(date +%s)-t0 )) s"
done
