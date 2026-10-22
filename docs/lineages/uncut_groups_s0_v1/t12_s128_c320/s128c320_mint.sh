#!/bin/bash
# 128 x 320 test: mint 12 FRESH topologies, recipe = the ladder's s128_c320 root (big_groups_inputs.sbatch at 224dd02e, md5 db71ec73..., PAYLOAD_* unset), p 0.6, single origin x1_uncut,
# REPLICA_RULE fastest_compatible, BASECFG cc40s16345.json, M_MODERATE = 11.6139 x 128/24, M_HEAVY = 27.622665025860204 x 128/24. NOT RUN until SCIENTIST signs and the expert says go.
# usage: s128c320_mint.sh "<12 ids>" <sha256 of the comma-joined list>
set -euo pipefail
IDS=$1; WANT=$2
U=/share/nikola.lukic/uncut_groups; ROOT=$U/gate_inputs_s128_c320_t12; WT=$HOME/gnn-herosim-wt/uh_xs_224dd02e
[ "$(echo $IDS | wc -w)" = 12 ] || { echo "FAIL LOUD: need 12 ids"; exit 2; }
GOT=$(echo -n $IDS | tr ' ' ',' | sha256sum | cut -c1-64); [ "$GOT" = "$WANT" ] || { echo "FAIL LOUD: id list sha256 $GOT != $WANT"; exit 2; }
for t in $IDS; do [[ $t =~ ^[0-9]{5}$ ]] && [ $t -ge 30253 ] || { echo "FAIL LOUD: id $t is not a 5-digit id >= 30253"; exit 2; }; done
[ ! -e $ROOT ] || { echo "FAIL LOUD: $ROOT exists"; exit 2; }
[[ $(git -C $WT rev-parse HEAD) == 224dd02e* && -z "$(git -C $WT status --porcelain)" ]] || { echo "FAIL LOUD: worktree $WT not clean at 224dd02e"; exit 2; }
[ "$(md5sum < $WT/scripts_cosim/datalab/big_groups_inputs.sbatch | cut -c1-32)" = db71ec73a67a67bf665dfc664979b19c ] || { echo "FAIL LOUD: big_groups_inputs.sbatch md5"; exit 2; }
[ "$(md5sum < $WT/scripts_cosim/uncut_groups_group_inputs.py | cut -c1-32)" = 9efb1da057cb85a2f9eececd0f5b684c ] || { echo "FAIL LOUD: uncut_groups_group_inputs.py md5"; exit 2; }
[ $(( $(squeue -u nikola.lukic -h | wc -l) + 2 )) -le 44 ] || { echo "FAIL LOUD: 2 more jobs would exceed 44"; exit 2; }
MM=$(python3 -c "print(repr(11.6139*128/24))"); MH=$(python3 -c "print(repr(27.622665025860204*128/24))")
echo "NS=128 NC=320 M_MODERATE=$MM M_HEAVY=$MH"
J=$(WT=$WT EXPECT_HEAD=224dd02e ROOT=$ROOT X1=$U/x1_uncut TOPOS="$IDS" NC=320 NS=128 P=0.6 M_MODERATE=$MM M_HEAVY=$MH REPLICA_RULE=fastest_compatible BASECFG=/share/nikola.lukic/accel_nosplit/inputs/cfg_src/cc40s16345.json sbatch --parsable --export=ALL $WT/scripts_cosim/datalab/big_groups_inputs.sbatch)
echo "mint job $J"
mkdir -p $U/s128c320_prep
cat > $U/s128c320_prep/group.sbatch <<G
#!/bin/bash
#SBATCH -J s5-t12-grp
#SBATCH -p CPU-amd
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:20:00
#SBATCH -o $U/s128c320_prep/group-%j.out
set -euo pipefail
eval "\$(micromamba shell hook --shell bash)" && micromamba activate gnn
cd $WT
python3 scripts_cosim/uncut_groups_group_inputs.py --src $ROOT/inputs --dst $ROOT/inputs_group
(cd $ROOT/inputs && md5sum -c MD5SUMS | grep -vc OK || true)
(cd $ROOT/inputs_group && md5sum -c MD5SUMS | grep -vc OK || true)
echo "inputs MD5SUMS \$(md5sum < $ROOT/inputs/MD5SUMS | cut -c1-32) inputs_group MD5SUMS \$(md5sum < $ROOT/inputs_group/MD5SUMS | cut -c1-32)"
python3 - $ROOT "$IDS" <<'P'
import json, sys
root, ids = sys.argv[1], sys.argv[2].split()
for tag in ("moderate", "heavy"):
    for t in ids:
        c = json.load(open(f"{root}/inputs_group/wf1_{tag}/cfg/cc40s{t}.json"))
        assert (c["nodes"]["client_nodes"]["count"], c["nodes"]["server_nodes"]["count"]) == (320, 128), (tag, t)
        assert str(c["network"]["topology"]["seed"]) == t, (tag, t, "topology seed")
print("cfgs OK: 320 x 128 in 24 cfgs, topology seed = id")
P
echo done
G
K=$(sbatch --parsable --dependency=afterok:$J $U/s128c320_prep/group.sbatch)
echo "group job $K (afterok $J)"
