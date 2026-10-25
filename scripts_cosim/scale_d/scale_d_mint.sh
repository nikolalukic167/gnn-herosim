#!/bin/bash
# scale_d_v1 mint (docs/lineages/scale_d_v1.md, v3). Runs only after S5 has filed the ids; nothing is submitted unless GO=1.
# Recipe of llf_mint.sh (llf_cd_confirm_v1): x1 mint at 224dd02e, group inputs, then every peer-exchange payload clipped at 200 MB (cellular unchanged).
# Changed from the 24-server recipe, and only these: NC and NS, and M = the 24-server M x NS/24.
#   A: 64 servers, 427 clients, 18 ids     B: 128 servers, 853 clients, 6 ids     SA / SB: the one smoke id of A / B (outside the study ids)
#   QA / QB: the disposable quick-look ids (4 at 64 servers, 2 at 128), never study ids
# usage: [GO=1] scale_d_mint.sh <A|B|SA|SB|QA|QB> "<ids, space separated>" <sha256 of the comma-joined list>
set -euo pipefail
SC=${1:?A|B|SA|SB}; IDS=${2:?ids}; WANT=${3:?sha256}; GO=${GO:-0}
case $SC in
  A) NS=64; NC=427; N=18;;
  B) NS=128; NC=853; N=6;;
  SA) NS=64; NC=427; N=1;;
  SB) NS=128; NC=853; N=1;;
  QA) NS=64; NC=427; N=4;;
  QB) NS=128; NC=853; N=2;;
  *) echo "FAIL LOUD: scale $SC"; exit 2;;
esac
SHA=224dd02e; U=/share/nikola.lukic/uncut_groups; X=$U/scale_d/$SC; WT=$HOME/gnn-herosim-wt/uh_xs_${SHA}
[ "$(echo $IDS | wc -w)" = $N ] || { echo "FAIL LOUD: scale $SC needs $N ids"; exit 2; }
GOT=$(echo -n $IDS | tr ' ' ',' | sha256sum | cut -c1-64); [ "$GOT" = "$WANT" ] || { echo "FAIL LOUD: id list sha256 $GOT != $WANT"; exit 2; }
for t in $IDS; do [[ $t =~ ^[0-9]{5}$ ]] && [ $t -gt 30350 ] || { echo "FAIL LOUD: id $t is not a 5-digit id > 30350"; exit 2; }; done
[ ! -e $X ] || { echo "FAIL LOUD: $X exists"; exit 2; }
[[ $(git -C $WT rev-parse HEAD) == $SHA* && -z "$(git -C $WT status --porcelain)" ]] || { echo "FAIL LOUD: worktree $WT not clean at $SHA"; exit 2; }
[ "$(md5sum < $WT/scripts_cosim/datalab/big_groups_inputs.sbatch | cut -c1-32)" = db71ec73a67a67bf665dfc664979b19c ] || { echo "FAIL LOUD: big_groups_inputs.sbatch md5"; exit 2; }
[ "$(md5sum < $WT/scripts_cosim/uncut_groups_group_inputs.py | cut -c1-32)" = 9efb1da057cb85a2f9eececd0f5b684c ] || { echo "FAIL LOUD: uncut_groups_group_inputs.py md5"; exit 2; }
MM=$(python3 -c "print(repr(11.6139*$NS/24))"); MH=$(python3 -c "print(repr(27.622665025860204*$NS/24))")
echo "scale $SC: NS=$NS NC=$NC ids=$N M_MODERATE=$MM M_HEAVY=$MH root=$X"
if [ "$GO" != 1 ]; then echo "DRY (GO!=1): nothing submitted"; exit 0; fi
[ $(( $(squeue -u "$USER" -h | wc -l) + 3 )) -le 44 ] || { echo "FAIL LOUD: 3 more jobs would exceed 44"; exit 2; }
mkdir -p $X/prep $X/logs; echo $IDS > $X/ids.txt
ROOT=$X/gate_inputs_x1
J=$(env WT=$WT EXPECT_HEAD=$SHA ROOT=$ROOT X1=$U/x1_uncut TOPOS="$IDS" NC=$NC NS=$NS P=0.6 M_MODERATE=$MM M_HEAVY=$MH REPLICA_RULE=fastest_compatible BASECFG=/share/nikola.lukic/accel_nosplit/inputs/cfg_src/cc40s16345.json sbatch --parsable --export=ALL -t 04:00:00 --mem=128G -o $X/logs/mint-%j.out $WT/scripts_cosim/datalab/big_groups_inputs.sbatch)
cp $(dirname "$0")/scale_d_group.sbatch.tmpl $X/prep/group.sbatch
cp $(dirname "$0")/scale_d_cap200.sbatch.tmpl $X/prep/cap200.sbatch
sed -i "s#__X__#$X#g; s#__WT__#$WT#g; s#__ROOT__#$ROOT#g; s#__IDS__#$IDS#g; s#__NC__#$NC#g; s#__NS__#$NS#g" $X/prep/group.sbatch $X/prep/cap200.sbatch
K=$(sbatch --parsable --dependency=afterok:$J $X/prep/group.sbatch)
L=$(sbatch --parsable --dependency=afterok:$K $X/prep/cap200.sbatch)
echo "mint job $J group job $K cap200 job $L; root $X"
