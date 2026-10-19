#!/bin/bash
# Exchange-model sensitivity study: mint the 12 fresh 24-server topologies once per payload variant (7 roots), at the merged sampler commit. NOT RUN until the sampler change is merged,
# the ids are rescanned just before, and the expert says go.  Recipe = conf2 / fl3: 160 clients x 24 servers, p 0.6, single origin x1_uncut, REPLICA_RULE fastest_compatible, BASECFG cc40s16345.json,
# M_MODERATE 11.6139, M_HEAVY 27.622665025860204 (unscaled), big_groups_inputs.sbatch + uncut_groups_group_inputs.py; variants only through the sampler env (PAYLOAD_*):
#   x1 (default, wf1_v1 as filed) | x0 (PAYLOAD_SCALE=0: ZERO PAYLOAD; latency and rendezvous remain) | x0p1 (0.1) | x0p3 (0.3) | x3 (3) | tier0 (PAYLOAD_HEAVY_SHARE=0) | azblob (PAYLOAD_SAMPLER=azblob; per-access blob size as a proxy, not function-to-function exchange)
# usage: xs_mint.sh "<12 ids, space separated>" <sha256 of the comma-joined list> <merged commit sha with the sampler change>
set -euo pipefail
IDS=$1; WANT=$2; SHA=$3
U=/share/nikola.lukic/uncut_groups; X=$U/xs; WT=$HOME/gnn-herosim-wt/uh_xs_${SHA:0:8}
[ "$(echo $IDS | wc -w)" = 12 ] || { echo "FAIL LOUD: need 12 ids"; exit 2; }
GOT=$(echo -n $IDS | tr ' ' ',' | sha256sum | cut -c1-64); [ "$GOT" = "$WANT" ] || { echo "FAIL LOUD: id list sha256 $GOT != $WANT"; exit 2; }
for t in $IDS; do [[ $t =~ ^[0-9]{5}$ ]] && [ $t -ge 30176 ] || { echo "FAIL LOUD: id $t is not a 5-digit id >= 30176"; exit 2; }; done
for v in x1 x0 x0p1 x0p3 x3 tier0 azblob; do [ ! -e $X/gate_inputs_xs_$v ] || { echo "FAIL LOUD: $X/gate_inputs_xs_$v exists"; exit 2; }; done
[ -d $WT ] || git -C $HOME/gnn-herosim worktree add -q --detach $WT $SHA
[[ $(git -C $WT rev-parse HEAD) == $SHA* && -z "$(git -C $WT status --porcelain)" ]] || { echo "FAIL LOUD: worktree $WT not clean at $SHA"; exit 2; }
grep -q "PAYLOAD_SCALE" $WT/scripts_cosim/datalab/big_groups_inputs.sbatch || { echo "FAIL LOUD: $SHA lacks the sampler change"; exit 2; }
[ "$(md5sum < $WT/scripts_cosim/uncut_groups_group_inputs.py | cut -c1-32)" = 9efb1da057cb85a2f9eececd0f5b684c ] || { echo "FAIL LOUD: uncut_groups_group_inputs.py md5"; exit 2; }
TABLE=$WT/src/placement/data/azblob_bytes_quantiles_v1.json
[ "$(sha256sum < $TABLE | cut -c1-64)" = c21a74c45fbf70c9d1c3ceeb3158158d4f5b1a86781b3887c9ae2018f44589c7 ] || { echo "FAIL LOUD: azblob table sha256"; exit 2; }
mkdir -p $X/prep
declare -A VENV=( [x1]="" [x0]="PAYLOAD_SCALE=0" [x0p1]="PAYLOAD_SCALE=0.1" [x0p3]="PAYLOAD_SCALE=0.3" [x3]="PAYLOAD_SCALE=3" [tier0]="PAYLOAD_HEAVY_SHARE=0" [azblob]="PAYLOAD_SAMPLER=azblob PAYLOAD_AZBLOB_TABLE=$TABLE" )
mj=(); gj=()
for v in x1 x0 x0p1 x0p3 x3 tier0 azblob; do
  ROOT=$X/gate_inputs_xs_$v
  J=$(env ${VENV[$v]} WT=$WT EXPECT_HEAD=${SHA:0:8} ROOT=$ROOT X1=$U/x1_uncut TOPOS="$IDS" NC=160 NS=24 P=0.6 M_MODERATE=11.6139 M_HEAVY=27.622665025860204 REPLICA_RULE=fastest_compatible BASECFG=/share/nikola.lukic/accel_nosplit/inputs/cfg_src/cc40s16345.json sbatch --parsable --export=ALL $WT/scripts_cosim/datalab/big_groups_inputs.sbatch)
  cat > $X/prep/group_$v.sbatch <<G
#!/bin/bash
#SBATCH -J xs-grp-$v
#SBATCH -p CPU-amd
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:20:00
#SBATCH -o $X/prep/group-$v-%j.out
set -euo pipefail
eval "\$(micromamba shell hook --shell bash)" && micromamba activate gnn
cd $WT
python3 scripts_cosim/uncut_groups_group_inputs.py --src $ROOT/inputs --dst $ROOT/inputs_group
(cd $ROOT/inputs && md5sum -c MD5SUMS | grep -vc OK || true)
(cd $ROOT/inputs_group && md5sum -c MD5SUMS | grep -vc OK || true)
echo "$v inputs MD5SUMS \$(md5sum < $ROOT/inputs/MD5SUMS | cut -c1-32) inputs_group MD5SUMS \$(md5sum < $ROOT/inputs_group/MD5SUMS | cut -c1-32)"
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
  K=$(sbatch --parsable --dependency=afterok:$J $X/prep/group_$v.sbatch)
  echo "$v mint job $J group job $K"; mj+=($J); gj+=($K)
done
cat > $X/prep/check.sbatch <<G
#!/bin/bash
#SBATCH -J xs-mint-check
#SBATCH -p CPU-amd
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:20:00
#SBATCH -o $X/prep/check-%j.out
set -uo pipefail
cd $X
echo "cfg md5 over inputs_group cfg files, per root (must be one value):"
for v in x1 x0 x0p1 x0p3 x3 tier0 azblob; do echo "\$v \$( (cd gate_inputs_xs_\$v/inputs_group && cat wf1_moderate/cfg/*.json wf1_heavy/cfg/*.json | md5sum | cut -c1-32))"; done
echo "x1 g0 heavy wl md5 \$(md5sum < gate_inputs_xs_x1/inputs_group/wf1_heavy/wl/grounded_g0_n50000.json | cut -c1-32)  (fl3: e08f33d3fa2b49e7341796e32071d0fd)"
G
sbatch --parsable --dependency=afterok:$(IFS=:; echo "${gj[*]}") $X/prep/check.sbatch
