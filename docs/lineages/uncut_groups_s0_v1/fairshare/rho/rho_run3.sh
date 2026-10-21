#!/bin/bash
#SBATCH --job-name=s6-rho3
#SBATCH --partition=CPU-amd
#SBATCH --cpus-per-task=32
#SBATCH --mem=220G
#SBATCH --time=12:00:00
#SBATCH --exclude=os-cpu-slurm-[21,15,13,14,16,10,17]
#SBATCH --output=/share/nikola.lukic/uncut_groups/anomaly/fairshare/logs/s6-rho3-%x-%j.out
# usage: sbatch --export=ALL,SIZE=<24|64|128>,PIN=<sha>,P=<parallel cells> rho_run3.sh
# Fair-share diagnostic (expert + SCIENTIST): offered link utilisation under the bar's, FL gnn s1's and FL MLP s1's placements. Each filed switch-off cell (moderate and
# heavy, windows g0 first then g1-g3, the 12 topologies of the size) is re-run at PIN through a copy of r1a_gate.sbatch with HEROSIM_AUDIT_TRACE on and checked equal to
# its filed cell by rho_extract2.py; only per-link usage is kept. P cells run in parallel (own trace file each); window-major order so g0 finishes first.
set -uo pipefail
SIZE=${SIZE:?}; PIN=${PIN:?}; P=${P:-12}
U=/share/nikola.lukic/uncut_groups; R=$U/anomaly/fairshare/rho3/$SIZE; mkdir -p $R $U/anomaly/fairshare/logs
case $SIZE in
  24) INB=$U/comb/in_comball; INM=$U/comb/in_comball; SEL=$U/arm1_selected.json; TOP="$(seq 19537 19548)"; FB=$U/comb/cells_bar_g8; FG=$U/comb/cells_comb_fl; FM=$U/comb/cells_comb ;;
  64) INB=$U/size64/in_size64; INM=$U/size64/in_size64; SEL=$U/size64b_selected.json; TOP="$(seq 30129 30140)"; FB=$U/size64/cells_bar; FG=$U/size64/cells_learned; FM=$U/size64/cells_desc ;;
  128) INB=$U/size128/in_size128; INM=$U/size128/in_mlp128; SEL=$U/size128_selected.json; TOP="$(seq 30241 30252)"; FB=$U/size128/cells_bar; FG=$U/size128/cells_learned; FM=$U/size128/mlp_cells ;;
  *) echo "FAIL LOUD: size"; exit 2 ;;
esac
WTP=$HOME/gnn-herosim-wt/s6_fs_${PIN:0:8}
[[ $(git -C $WTP rev-parse HEAD) == $PIN* ]] || { echo "FAIL LOUD: worktree"; exit 2; }
HERE=$HOME/s6_scripts/rho
GATE=/tmp/rho3_gate_$SLURM_JOB_ID.sbatch
sed 's#^python3 scripts_cosim/fresh_topo_burst_v1_gate.py r1a#export HEROSIM_AUDIT_TRACE="$TRACE_OUT"\n&#' $WTP/scripts_cosim/datalab/r1a_gate.sbatch > $GATE
grep -q 'HEROSIM_AUDIT_TRACE="$TRACE_OUT"' $GATE || { echo "FAIL LOUD: patch"; exit 2; }
# P must not exceed the job's physical cores (SMT is off on these nodes and a -c N job gets N/2 usable cores: nproc is the authority, SCIENTIST's condition)
NPROC=$(nproc); TPC=$(lscpu | awk -F: '/Thread\(s\) per core/{gsub(/ /,"",$2);print $2}'); [ "$P" -le "$NPROC" ] || P=$NPROC
echo "node $(hostname) job $SLURM_JOB_ID cpus-per-task ${SLURM_CPUS_PER_TASK:-?} nproc $NPROC threads-per-core $TPC cpu-model $(lscpu | awk -F: '/Model name/{gsub(/^ +/,"",$2);print $2}') cpuset $(cat /sys/fs/cgroup/cpuset.cpus.effective 2>/dev/null) parallel-cells $P" > $R/node.txt
cat $R/node.txt
export WTP PIN SIZE R SEL INB INM FB FG FM HERE GATE
cat > /tmp/rho3_worker_$SLURM_JOB_ID.sh <<'W'
#!/bin/bash
# args: arm win rung topo
ARM=$1; W=$2; RUNG=$3; T=$4
case $ARM in bar) KIND=cd_ib0_inflightfull_k100; SEED=0; IN=$INB; FILED=$FB ;; gnn) KIND=ra_gnn_eng_combfl_nosplit_k100; SEED=1; IN=$INM; [ $SIZE = 128 ] && IN=$INB; FILED=$FG ;; mlp) KIND=ra_mlp_same_combfl_nosplit_k100; SEED=1; IN=$INM; FILED=$FM ;; esac
OUTJ=$R/$ARM/${W}_${RUNG}_$T.json; [ -e $OUTJ ] && exit 0
mkdir -p $R/$ARM; TAG=${ARM}_${W}_${RUNG}_$T; OUTC=$R/$ARM/cell_$TAG; TRACE_OUT=/tmp/rho3_${SLURM_JOB_ID}_$TAG.jsonl; rm -rf $OUTC $TRACE_OUT
# the gate's EXIT trap removes /tmp/herosim_raw_${SLURM_JOB_ID:-$$}: a job-wide id would let one finishing cell delete the raw output of the cells still running
( export TRACE_OUT WT=$WTP EXPECT_HEAD=${PIN:0:8} IN OUT=$OUTC SELECT=$SEL R1A_ARMS=$KIND R1A_SEEDS=$SEED R1A_TOPOS=$T R1A_WINDOWS=$W WF1_RUNGS=$RUNG PAR=1 CELL_MEM=16G LIMIT=14400 R1A_SPLIT=uncut_arm1_split.json R1A_SLATE=declared_pruning_v1_uncapped
  unset R1A_EXCHANGE_TCP R1A_EXCHANGE_FAIRSHARE R1A_EXCHANGE_FAIRSHARE_PARTNERS SLURM_JOB_ID SLURM_ARRAY_TASK_ID
  bash $GATE ) > $R/$ARM/log_$TAG.txt 2>&1 || { echo "FAIL cell $TAG"; tail -3 $R/$ARM/log_$TAG.txt; rm -f $TRACE_OUT; exit 0; }
N=cc40s${T}__${W}${RUNG}__${KIND}_s$SEED
if python3 $HERE/rho_extract2.py $TRACE_OUT $OUTC/$N.summary.json $FILED/$N.summary.json $OUTJ; then rm -rf $OUTC $TRACE_OUT; else echo "FAIL extract $TAG (cell dir and trace kept)"; fi
W
chmod +x /tmp/rho3_worker_$SLURM_JOB_ID.sh
eval "$(micromamba shell hook --shell bash)" && micromamba activate gnn
for W in g0 g1 g2 g3; do for ARM in gnn mlp bar; do for RUNG in moderate heavy; do for T in $TOP; do echo "$ARM $W $RUNG $T"; done; done; done; done > /tmp/rho3_list_$SLURM_JOB_ID.txt
wc -l /tmp/rho3_list_$SLURM_JOB_ID.txt
xargs -P $P -L 1 /tmp/rho3_worker_$SLURM_JOB_ID.sh < /tmp/rho3_list_$SLURM_JOB_ID.txt
echo "[rho3] size $SIZE done: $(ls $R/*/*.json 2>/dev/null | wc -l) cells"
