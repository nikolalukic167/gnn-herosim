#!/bin/bash
#SBATCH --job-name=s6-rho
#SBATCH --partition=CPU-amd
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --exclude=os-cpu-slurm-[21,15,13,14,16,10,17]
#SBATCH --output=/share/nikola.lukic/uncut_groups/anomaly/fairshare/logs/s6-rho-%x-%j.out
# usage: sbatch --export=ALL,SIZE=<24|64|128>,PIN=<sha> rho_run.sh
# Offered link utilisation from the filed switch-off bar cells (SCIENTIST's A1 ruling 3): re-run each cell (g0, both rungs, all 12 topologies of the size, bar
# s0, switch off) at PIN through a copy of r1a_gate.sbatch with HEROSIM_AUDIT_TRACE on, assert the re-run equals the filed cell (rho_extract.py), keep only the
# per-link usage. One cell at a time (PAR=1; the trace path is per cell).
set -euo pipefail
SIZE=${SIZE:?}; PIN=${PIN:?}
U=/share/nikola.lukic/uncut_groups; R=$U/anomaly/fairshare/rho/$SIZE; mkdir -p $R $U/anomaly/fairshare/logs
case $SIZE in
  24) IN=$U/comb/in_comball; SEL=$U/arm1_selected.json; TOP=$(seq 19537 19548); FILED=$U/comb/cells_bar_g8 ;;
  64) IN=$U/size64/in_size64; SEL=$U/size64b_selected.json; TOP=$(seq 30129 30140); FILED=$U/size64/cells_bar ;;
  128) IN=$U/size128/in_size128; SEL=$U/size128_selected.json; TOP=$(seq 30241 30252); FILED=$U/size128/cells_bar ;;
  *) echo "FAIL LOUD: size"; exit 2 ;;
esac
WTP=$HOME/gnn-herosim-wt/s6_fs_${PIN:0:8}
[[ $(git -C $WTP rev-parse HEAD) == $PIN* ]] || { echo "FAIL LOUD: worktree"; exit 2; }
HERE=$HOME/s6_scripts/rho
sed 's#^python3 scripts_cosim/fresh_topo_burst_v1_gate.py r1a#export HEROSIM_AUDIT_TRACE="$TRACE_OUT"\n&#' $WTP/scripts_cosim/datalab/r1a_gate.sbatch > /tmp/rho_gate_$$.sbatch
grep -q 'HEROSIM_AUDIT_TRACE="$TRACE_OUT"' /tmp/rho_gate_$$.sbatch || { echo "FAIL LOUD: patch"; exit 2; }
eval "$(micromamba shell hook --shell bash)" && micromamba activate gnn
for RUNG in moderate heavy; do
  for T in $TOP; do
    OUTC=$R/cell_${RUNG}_$T; [ -e $R/${RUNG}_$T.json ] && continue
    rm -rf $OUTC; export TRACE_OUT=/tmp/rho_trace_${SLURM_JOB_ID:-$$}.jsonl; rm -f $TRACE_OUT
    ( export WT=$WTP EXPECT_HEAD=${PIN:0:8} IN=$IN OUT=$OUTC SELECT=$SEL R1A_ARMS=cd_ib0_inflightfull_k100 R1A_SEEDS=0 R1A_TOPOS=$T R1A_WINDOWS=g0 WF1_RUNGS=$RUNG PAR=1 CELL_MEM=16G LIMIT=14400 R1A_SPLIT=uncut_arm1_split.json R1A_SLATE=declared_pruning_v1_uncapped
      unset R1A_EXCHANGE_TCP R1A_EXCHANGE_FAIRSHARE R1A_EXCHANGE_FAIRSHARE_PARTNERS
      bash /tmp/rho_gate_$$.sbatch ) > $R/log_${RUNG}_$T.txt 2>&1 || { echo "FAIL cell $RUNG $T"; tail -5 $R/log_${RUNG}_$T.txt; continue; }
    N=cc40s${T}__g0${RUNG}__cd_ib0_inflightfull_k100_s0
    python3 $HERE/rho_extract.py $TRACE_OUT $OUTC/$N.summary.json $FILED/$N.summary.json $R/${RUNG}_$T.json || echo "FAIL extract $RUNG $T"
    rm -f $TRACE_OUT; rm -rf $OUTC
  done
done
echo "[rho] size $SIZE done: $(ls $R/*.json | wc -l) cells"
