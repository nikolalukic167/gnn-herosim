#!/bin/bash
# live_link_features_v1 confirmation cells (A6: static variant, 40-epoch seeds 1-3) on the 12 sealed ids: 30285-30287, 30289-30296 (input root llf_conf) and 30297 (llf_conf_r1).
# Option D (TCP=1, FAIRSHARE=1 serial, default cellular uplink), cap200 inputs, nosplit k100, g0+g1, moderate+heavy. The recipe of S6's fs_g.sh (r1a_gate.sbatch at a pinned worktree) with one IN dir per root.
#   PIN=<full sha> NODES="os-cpu-slurm-6 ..." [DRY=1] [GROUPS_ONLY="static refs bar"] [ROOTS_ONLY="main r1"] [RUNGS_ONLY="moderate heavy"] bash llf_confirm_launch.sh
# Waves: RUNGS_ONLY=moderate is 28 jobs (2 roots x 2 windows x 7 groups), then RUNGS_ONLY=heavy the other 28.
# DRY=1 (the default) prints every sbatch and submits nothing. A real submit needs DRY=0 and refuses to take the queue over 44 (counts the batch first).
# Job groups per (root, window, rung): static_s<N> = the four static e40 arms at seed N (3 jobs); refs_s<N> = the four v6 fslabel references at seed N (3 jobs); bar = cd_ib0_inflightfull_k100 seed 0.
set -euo pipefail
PIN=${PIN:?set PIN}; NODES=${NODES:?set NODES (space separated; at most 2 jobs per node are placed)}; DRY=${DRY:-1}
U=/share/nikola.lukic/uncut_groups; WTP=$HOME/gnn-herosim-wt/s6_lk_${PIN:0:8}
[[ $(git -C "$WTP" rev-parse HEAD) == "$PIN" ]] || { echo "FAIL LOUD: worktree $WTP is not at $PIN"; exit 2; }
declare -A INDIR=([main]=$U/xs/in_llf_conf [r1]=$U/xs/in_llf_conf_r1)
declare -A TOPOS=([main]=30285,30286,30287,30289,30290,30291,30292,30293,30294,30295,30296 [r1]=30297)
STATIC="ra_gnn_eng_combfl_linkstatic_e40_nosplit_k100,ra_twin_eng_combfl_linkstatic_e40_nosplit_k100,ra_mlp_same_combfl_linkstatic_e40_nosplit_k100,ra_mlp_wide_combfl_linkstatic_e40_nosplit_k100"
REFS="ra_gnn_eng_fslabel_combfl_nosplit_k100,ra_twin_eng_fslabel_combfl_nosplit_k100,ra_mlp_same_fslabel_combfl_nosplit_k100,ra_mlp_wide_fslabel_combfl_nosplit_k100"
BAR="cd_ib0_inflightfull_k100"
declare -A PLACED
read -r -a NODEARR <<< "$NODES"
jobs=()   # root|win|rung|group|arms|seeds|split
for root in ${ROOTS_ONLY:-main r1}; do for win in g0 g1; do for rung in ${RUNGS_ONLY:-moderate heavy}; do
  for s in 1 2 3; do
    [[ " ${GROUPS_ONLY:-static refs bar} " == *" static "* ]] && jobs+=("$root|$win|$rung|static_s$s|$STATIC|$s|uncut_link_split.json")
    [[ " ${GROUPS_ONLY:-static refs bar} " == *" refs "* ]] && jobs+=("$root|$win|$rung|refs_s$s|$REFS|$s|uncut_arm1_split.json")
  done
  [[ " ${GROUPS_ONLY:-static refs bar} " == *" bar "* ]] && jobs+=("$root|$win|$rung|bar|$BAR|0|uncut_arm1_split.json")
done; done; done
if [[ "$DRY" != 1 ]]; then
  QUEUED=$(squeue -u "$USER" -h | wc -l)
  (( QUEUED + ${#jobs[@]} <= 44 )) || { echo "FAIL LOUD: queue $QUEUED + ${#jobs[@]} jobs exceeds the 44-job cap (use ROOTS_ONLY / GROUPS_ONLY to launch in waves)"; exit 1; }
fi
i=0
for j in "${jobs[@]}"; do
  IFS='|' read -r root win rung group arms seeds split <<< "$j"
  node=""
  for k in $(seq 0 $(( ${#NODEARR[@]} - 1 ))); do n=${NODEARR[$(( (i + k) % ${#NODEARR[@]} ))]}; if (( ${PLACED[$n]:-0} < 2 )); then node=$n; break; fi; done
  [[ -n "$node" ]] || { echo "FAIL LOUD: no node with fewer than 2 of my jobs left among: $NODES"; exit 1; }
  PLACED[$node]=$(( ${PLACED[$node]:-0} + 1 )); i=$(( i + 1 ))
  sub=${root}_${win}${rung}_${group}; out=$U/anomaly/fairshare/d_conf/$sub
  [[ -e $out && "$DRY" != 1 ]] && { echo "FAIL LOUD: $out exists"; exit 2; }
  cmd=(env WT=$WTP EXPECT_HEAD=${PIN:0:8} IN=${INDIR[$root]} OUT=$out SELECT=$U/arm1_selected.json R1A_ARMS=$arms R1A_SEEDS=$seeds R1A_TOPOS=${TOPOS[$root]} R1A_WINDOWS=$win WF1_RUNGS=$rung
       PAR=8 CELL_MEM=16G LIMIT=14400 R1A_SPLIT=$split R1A_SLATE=declared_pruning_v1_uncapped R1A_EXCHANGE_TCP=1 R1A_EXCHANGE_FAIRSHARE=1
       sbatch --parsable --export=ALL -p CPU-amd -w $node --cpus-per-task=16 --mem=120G -t 12:00:00 -J llfc-$sub -o $U/anomaly/fairshare/d_conf/logs/llfc-%x-%j.out $WTP/scripts_cosim/datalab/r1a_gate.sbatch)
  if [[ "$DRY" == 1 ]]; then echo "${cmd[*]}" | sed "s#$U/##g; s#$HOME/##g"; else
    mkdir -p "$out" "$U/anomaly/fairshare/d_conf/logs"; ( unset R1A_EXCHANGE_FAIRSHARE_PARTNERS HEROSIM_ACCESS_CELLULAR_OUT_MBPS; "${cmd[@]}" ); fi
done
echo "jobs: ${#jobs[@]}"
