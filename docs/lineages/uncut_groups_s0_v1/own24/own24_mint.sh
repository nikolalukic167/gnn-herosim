#!/bin/bash
# Own-state confirmation: mint 24 FRESH 24-server topologies (recipe of the conf2 / fl3 mint). NOT RUN until SCIENTIST signs the id list and the expert says go.
# 160 clients x 24 servers, p = 0.6, single origin x1_uncut, REPLICA_RULE=fastest_compatible, BASECFG cc40s16345.json, M_MODERATE 11.6139, M_HEAVY 27.622665025860204 (unscaled),
# big_groups_inputs.sbatch at f8814905, then uncut_groups_group_inputs.py for the inputs_group tree; ids = the literal-rule scan's lowest free 24-run (scan10).
# usage: fl3_mint.sh "<24 ids, space separated>" <sha256 of the comma-joined list>
set -euo pipefail
IDS=$1; WANT=$2
U=/share/nikola.lukic/uncut_groups; ROOT=$U/gate_inputs_own24
[ "$(echo $IDS | wc -w)" = 24 ] || { echo "FAIL LOUD: need 24 ids"; exit 2; }
GOT=$(echo -n $IDS | tr ' ' ',' | sha256sum | cut -c1-64); [ "$GOT" = "$WANT" ] || { echo "FAIL LOUD: id list sha256 $GOT != $WANT"; exit 2; }
for t in $IDS; do [[ $t =~ ^[0-9]{5}$ ]] && [ $t -ge 20508 ] || { echo "FAIL LOUD: id $t not a 5-digit id from 20508"; exit 2; }; done
[ ! -e $ROOT ] || { echo "FAIL LOUD: $ROOT exists"; exit 2; }
cd ~/gnn-herosim-wt/uh_f8814905
[ "$(git rev-parse HEAD | cut -c1-8)" = f8814905 ] || { echo "FAIL LOUD: mint worktree not at f8814905"; exit 2; }
[ -z "$(git status --porcelain)" ] || { echo "FAIL LOUD: mint worktree dirty"; exit 2; }
[ "$(md5sum < scripts_cosim/datalab/big_groups_inputs.sbatch | cut -c1-32)" = cf6e4f426e8122187233d0c9c6844670 ] || { echo "FAIL LOUD: big_groups_inputs.sbatch md5"; exit 2; }
[ "$(md5sum < scripts_cosim/uncut_groups_group_inputs.py | cut -c1-32)" = 9efb1da057cb85a2f9eececd0f5b684c ] || { echo "FAIL LOUD: uncut_groups_group_inputs.py md5"; exit 2; }
echo "NS=24 M_MODERATE=11.6139 M_HEAVY=27.622665025860204"
J=$(WT=$PWD EXPECT_HEAD=f8814905 ROOT=$ROOT X1=$U/x1_uncut TOPOS="$IDS" NC=160 NS=24 P=0.6 M_MODERATE=11.6139 M_HEAVY=27.622665025860204 REPLICA_RULE=fastest_compatible BASECFG=/share/nikola.lukic/accel_nosplit/inputs/cfg_src/cc40s16345.json sbatch --parsable --export=ALL scripts_cosim/datalab/big_groups_inputs.sbatch)
echo "mint job $J"
mkdir -p $U/own24_prep
cat > $U/own24_prep/group.sbatch <<G
#!/bin/bash
#SBATCH -J s5-own24-grp
#SBATCH -p CPU-amd
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:20:00
#SBATCH -o $U/own24_prep/group-%j.out
set -euo pipefail
eval "\$(micromamba shell hook --shell bash)" && micromamba activate gnn
cd ~/gnn-herosim-wt/uh_f8814905
python3 scripts_cosim/uncut_groups_group_inputs.py --src $ROOT/inputs --dst $ROOT/inputs_group
(cd $ROOT/inputs && md5sum -c MD5SUMS | grep -vc OK || true)
(cd $ROOT/inputs_group && md5sum -c MD5SUMS | grep -vc OK || true)
md5sum $ROOT/inputs/MD5SUMS $ROOT/inputs_group/MD5SUMS
python3 - $ROOT "$IDS" <<'P'
import json, sys
root, ids = sys.argv[1], sys.argv[2].split()
for tag in ("moderate", "heavy"):
    for t in ids:
        c = json.load(open(f"{root}/inputs_group/wf1_{tag}/cfg/cc40s{t}.json"))
        assert (c["nodes"]["client_nodes"]["count"], c["nodes"]["server_nodes"]["count"]) == (160, 24), (tag, t)
print("cfgs OK: 160 x 24")
P
echo done
G
K=$(sbatch --parsable --dependency=afterok:$J $U/own24_prep/group.sbatch)
echo "group job $K (afterok $J)"
