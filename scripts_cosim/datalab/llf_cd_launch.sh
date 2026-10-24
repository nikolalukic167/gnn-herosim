#!/bin/bash
# live_link_features_v1: generic cell launcher for the descriptive / follow-up waves on sealed id sets (S4). Same recipe as llf_confirm_launch.sh (option D: TCP=1, FAIRSHARE=1 serial, default cellular uplink; cap200
# inputs; nosplit k100; r1a_gate.sbatch at a pinned worktree), but the id sets, windows, rungs and arms come from the environment and a roots file, and the jobs are UNPINNED (the scheduler places them; EXCL keeps
# nodes out). One job per (root, window, rung, group); DRY=1 (the default) prints and submits nothing; a real submit refuses to take the queue over 44 and refuses an existing output dir.
#   ROOTS_FILE=<file>   lines "<name> <IN dir> <comma-separated topology ids>"; IN holds wf1_heavy, wf1_moderate, uncut_arm1_split.json, uncut_link_split.json and models/ (xs/in_llf_conf ... or llf_cd_in.sh output)
#   OUT_ROOT=<dir>      outputs <OUT_ROOT>/<name>_<window><rung>_<group>, logs <OUT_ROOT>/logs
#   PIN=<full sha>      the pin of the s0 group (bar/reactive/random: the d_conf bar's code); PIN_STATIC=<full sha> the pin of the static groups (default PIN). Worktrees ~/gnn-herosim-wt/s6_lk_<sha8> must exist.
#   GROUPS_ONLY="s0 static_s1 static_s2 static_s3" (default: all four)   S0_ARMS (default cd_ib0_inflightfull_k100,reactive,random; seed 0)   STATIC_ARMS (default the static e40 gnn, mlp, wide; seeds 1-3, one job per seed)
#   WINDOWS="g0 g1 g2 g3"   RUNGS="moderate heavy"   LIMIT=43200 (per-cell watchdog)   JN=<job-name prefix>   EXCL=<sbatch --exclude list>   CPUS=16
set -euo pipefail
ROOTS_FILE=${ROOTS_FILE:?}; OUT_ROOT=${OUT_ROOT:?}; PIN=${PIN:?}; PIN_STATIC=${PIN_STATIC:-$PIN}; DRY=${DRY:-1}; JN=${JN:-llfx}; LIMIT=${LIMIT:-43200}; CPUS=${CPUS:-16}
WINDOWS=${WINDOWS:-g0 g1 g2 g3}; RUNGS=${RUNGS:-moderate heavy}; GR=${GROUPS_ONLY:-s0 static_s1 static_s2 static_s3}
S0_ARMS=${S0_ARMS:-cd_ib0_inflightfull_k100,reactive,random}
STATIC_ARMS=${STATIC_ARMS:-ra_gnn_eng_combfl_linkstatic_e40_nosplit_k100,ra_mlp_same_combfl_linkstatic_e40_nosplit_k100,ra_mlp_wide_combfl_linkstatic_e40_nosplit_k100}
U=/share/nikola.lukic/uncut_groups
for p in $PIN $PIN_STATIC; do [[ $(git -C "$HOME/gnn-herosim-wt/s6_lk_${p:0:8}" rev-parse HEAD) == "$p" ]] || { echo "FAIL LOUD: worktree s6_lk_${p:0:8} is not at $p"; exit 2; }; done
jobs=()
while read -r name indir ids; do
  [[ -z "$name" || "$name" == \#* ]] && continue
  [[ -d $indir/wf1_heavy && -d $indir/wf1_moderate && -f $indir/uncut_arm1_split.json && -f $indir/uncut_link_split.json ]] || { echo "FAIL LOUD: $indir is not a complete IN dir"; exit 2; }
  for win in $WINDOWS; do for rung in $RUNGS; do for g in $GR; do
    case $g in
      s0) jobs+=("$name|$indir|$ids|$win|$rung|$g|$S0_ARMS|0|uncut_arm1_split.json|$PIN");;
      static_s[123]) jobs+=("$name|$indir|$ids|$win|$rung|$g|$STATIC_ARMS|${g#static_s}|uncut_link_split.json|$PIN_STATIC");;
      *) echo "FAIL LOUD: group $g"; exit 2;;
    esac
  done; done; done
done < "$ROOTS_FILE"
if [[ "$DRY" != 1 ]]; then
  QUEUED=$(squeue -u "$USER" -h | wc -l)
  (( QUEUED + ${#jobs[@]} <= 44 )) || { echo "FAIL LOUD: queue $QUEUED + ${#jobs[@]} jobs exceeds the 44-job cap"; exit 1; }
fi
for j in "${jobs[@]}"; do
  IFS='|' read -r name indir ids win rung g arms seeds split pin <<< "$j"
  WTP=$HOME/gnn-herosim-wt/s6_lk_${pin:0:8}; sub=${name}_${win}${rung}_${g}; out=$OUT_ROOT/$sub
  [[ -e $out && "$DRY" != 1 ]] && { echo "FAIL LOUD: $out exists"; exit 2; }
  cmd=(env WT=$WTP EXPECT_HEAD=${pin:0:8} IN=$indir OUT=$out SELECT=$U/arm1_selected.json R1A_ARMS=$arms R1A_SEEDS=$seeds R1A_TOPOS=$ids R1A_WINDOWS=$win WF1_RUNGS=$rung
       PAR=8 CELL_MEM=16G LIMIT=$LIMIT R1A_SPLIT=$split R1A_SLATE=declared_pruning_v1_uncapped R1A_EXCHANGE_TCP=1 R1A_EXCHANGE_FAIRSHARE=1
       sbatch --parsable --export=ALL -p CPU-amd ${EXCL:+--exclude=$EXCL} --cpus-per-task=$CPUS --mem=120G -t 24:00:00 -J $JN-$sub -o $OUT_ROOT/logs/$JN-%x-%j.out $WTP/scripts_cosim/datalab/r1a_gate.sbatch)
  if [[ "$DRY" == 1 ]]; then echo "${cmd[*]}" | sed "s#$U/##g; s#$HOME/##g"; else
    mkdir -p "$out" "$OUT_ROOT/logs"; ( unset R1A_EXCHANGE_FAIRSHARE_PARTNERS HEROSIM_ACCESS_CELLULAR_OUT_MBPS; "${cmd[@]}" ); fi
done
echo "jobs: ${#jobs[@]}"
