#!/bin/bash
# Server-count ladder (descriptive, no label, no registration): mint 18 fresh topologies in 6 roots, recipe = the size pilot's / conf2's (big_groups_inputs.sbatch at 224dd02e, PAYLOAD_* unset),
# p 0.6, single origin x1_uncut, REPLICA_RULE fastest_compatible, BASECFG cc40s16345.json; M_MODERATE = 11.6139 x servers/24, M_HEAVY = 27.622665025860204 x servers/24 (the group arrival RATE scales with servers, never the group size).
# roots (3 ids each, in order): s96_c160, s128_c160, s192_c160, s256_c160, s128_c320, s256_c320.
# usage: ladder_mint.sh "<18 ids>" <sha256 of the comma-joined list>
set -euo pipefail
IDS=$1; WANT=$2
U=/share/nikola.lukic/uncut_groups; L=$U/topo_explore/ladder; WT=$HOME/gnn-herosim-wt/uh_xs_224dd02e
[ "$(echo $IDS | wc -w)" = 18 ] || { echo "FAIL LOUD: need 18 ids"; exit 2; }
GOT=$(echo -n $IDS | tr ' ' ',' | sha256sum | cut -c1-64); [ "$GOT" = "$WANT" ] || { echo "FAIL LOUD: id list sha256 $GOT != $WANT"; exit 2; }
for t in $IDS; do [[ $t =~ ^[0-9]{5}$ ]] && [ $t -ge 30200 ] || { echo "FAIL LOUD: id $t is not a 5-digit id >= 30200"; exit 2; }; done
[ ! -e $L/gate_inputs_s96_c160 ] || { echo "FAIL LOUD: ladder roots exist"; exit 2; }
[[ $(git -C $WT rev-parse HEAD) == 224dd02e* && -z "$(git -C $WT status --porcelain)" ]] || { echo "FAIL LOUD: worktree $WT not clean at 224dd02e"; exit 2; }
[ "$(md5sum < $WT/scripts_cosim/datalab/big_groups_inputs.sbatch | cut -c1-32)" = db71ec73a67a67bf665dfc664979b19c ] || { echo "FAIL LOUD: big_groups_inputs.sbatch md5"; exit 2; }
[ "$(md5sum < $WT/scripts_cosim/uncut_groups_group_inputs.py | cut -c1-32)" = 9efb1da057cb85a2f9eececd0f5b684c ] || { echo "FAIL LOUD: uncut_groups_group_inputs.py md5"; exit 2; }
mkdir -p $L/prep; set -- $IDS; ids=($IDS)
roots=("96 160" "128 160" "192 160" "256 160" "128 320" "256 320")
for i in 0 1 2 3 4 5; do
  read NS NC <<< "${roots[$i]}"; T="${ids[$((3*i))]} ${ids[$((3*i+1))]} ${ids[$((3*i+2))]}"; ROOT=$L/gate_inputs_s${NS}_c${NC}
  MM=$(python3 -c "print(repr(11.6139*$NS/24))"); MH=$(python3 -c "print(repr(27.622665025860204*$NS/24))")
  J=$(WT=$WT EXPECT_HEAD=224dd02e ROOT=$ROOT X1=$U/x1_uncut TOPOS="$T" NC=$NC NS=$NS P=0.6 M_MODERATE=$MM M_HEAVY=$MH REPLICA_RULE=fastest_compatible BASECFG=/share/nikola.lukic/accel_nosplit/inputs/cfg_src/cc40s16345.json sbatch --parsable --export=ALL $WT/scripts_cosim/datalab/big_groups_inputs.sbatch)
  cat > $L/prep/group_s${NS}_c${NC}.sbatch <<G
#!/bin/bash
#SBATCH -J lad-grp-s${NS}-c${NC}
#SBATCH -p CPU-amd
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:20:00
#SBATCH -o $L/prep/group-s${NS}_c${NC}-%j.out
set -euo pipefail
eval "\$(micromamba shell hook --shell bash)" && micromamba activate gnn
cd $WT
python3 scripts_cosim/uncut_groups_group_inputs.py --src $ROOT/inputs --dst $ROOT/inputs_group
(cd $ROOT/inputs && md5sum -c MD5SUMS | grep -vc OK || true)
(cd $ROOT/inputs_group && md5sum -c MD5SUMS | grep -vc OK || true)
echo "s${NS}_c${NC} inputs MD5SUMS \$(md5sum < $ROOT/inputs/MD5SUMS | cut -c1-32) inputs_group MD5SUMS \$(md5sum < $ROOT/inputs_group/MD5SUMS | cut -c1-32)"
python3 - $ROOT "$T" <<'P'
import json, sys
root, ids = sys.argv[1], sys.argv[2].split()
got = set()
for tag in ("moderate", "heavy"):
    for t in ids:
        c = json.load(open(f"{root}/inputs_group/wf1_{tag}/cfg/cc40s{t}.json"))
        got.add((c["nodes"]["client_nodes"]["count"], c["nodes"]["server_nodes"]["count"]))
print("counted from the minted cfgs (clients, servers):", sorted(got), "over", len(ids), "topologies x 2 rungs")
P
echo done
G
  K=$(sbatch --parsable --dependency=afterok:$J $L/prep/group_s${NS}_c${NC}.sbatch)
  echo "s${NS}_c${NC} ids $T: M_MODERATE=$MM M_HEAVY=$MH mint job $J group job $K"
done
