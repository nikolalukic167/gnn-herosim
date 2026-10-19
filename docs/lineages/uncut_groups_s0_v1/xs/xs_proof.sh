#!/bin/bash
# Proof for the sampler change (commit 96abb284): re-mint the 12 fl3 ids into scratch roots at the new commit. The default re-mint must reproduce gate_inputs_fl3's MD5SUMS exactly;
# the variants are minted for the cross-check against the post-mint glue and for the azblob end-to-end run. Scratch only; not the study's mint.
set -euo pipefail
U=/share/nikola.lukic/uncut_groups; P=$U/xs_proof; WT=$HOME/gnn-herosim-wt/uh_xs_96abb284; IDS="$(seq -s' ' 30149 30160)"
[ ! -e $P ] || { echo "FAIL LOUD: $P exists"; exit 2; }
[[ $(git -C $WT rev-parse HEAD) == 96abb284* && -z "$(git -C $WT status --porcelain)" ]] || { echo "FAIL LOUD: worktree"; exit 2; }
mkdir -p $P
T=$WT/src/placement/data/azblob_bytes_quantiles_v1.json
run() { # name, extra env...
  name=$1; shift
  env WT=$WT EXPECT_HEAD=96abb284 ROOT=$P/$name X1=$U/x1_uncut TOPOS="$IDS" NC=160 NS=24 P=0.6 M_MODERATE=11.6139 M_HEAVY=27.622665025860204 REPLICA_RULE=fastest_compatible \
    BASECFG=/share/nikola.lukic/accel_nosplit/inputs/cfg_src/cc40s16345.json "$@" sbatch --parsable --export=ALL $WT/scripts_cosim/datalab/big_groups_inputs.sbatch
}
a=$(run default); echo "default mint $a"
b=$(run x0p3 PAYLOAD_SCALE=0.3); echo "x0p3 mint $b"
c=$(run tier0 PAYLOAD_HEAVY_SHARE=0); echo "tier0 mint $c"
d=$(run x0 PAYLOAD_SCALE=0); echo "x0 mint $d"
e=$(run azblob PAYLOAD_SAMPLER=azblob PAYLOAD_AZBLOB_TABLE=$T); echo "azblob mint $e"
cat > $P/group.sbatch <<G
#!/bin/bash
#SBATCH -J xs-proof-grp
#SBATCH -p CPU-amd
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:20:00
#SBATCH -o $P/group-%j.out
set -euo pipefail
eval "\$(micromamba shell hook --shell bash)" && micromamba activate gnn
cd $WT
python3 scripts_cosim/uncut_groups_group_inputs.py --src $P/default/inputs --dst $P/default/inputs_group
(cd $P/default/inputs && md5sum -c MD5SUMS | grep -vc OK || true)
(cd $P/default/inputs_group && md5sum -c MD5SUMS | grep -vc OK || true)
echo "default inputs MD5SUMS      \$(md5sum < $P/default/inputs/MD5SUMS | cut -c1-32)   fl3 \$(md5sum < $U/gate_inputs_fl3/inputs/MD5SUMS | cut -c1-32)"
echo "default inputs_group MD5SUMS \$(md5sum < $P/default/inputs_group/MD5SUMS | cut -c1-32)   fl3 \$(md5sum < $U/gate_inputs_fl3/inputs_group/MD5SUMS | cut -c1-32)"
cmp $P/default/inputs/MD5SUMS $U/gate_inputs_fl3/inputs/MD5SUMS && echo "inputs MD5SUMS IDENTICAL"
cmp $P/default/inputs_group/MD5SUMS $U/gate_inputs_fl3/inputs_group/MD5SUMS && echo "inputs_group MD5SUMS IDENTICAL"
for v in default x0p3 tier0 x0 azblob; do echo "\$v g0 heavy wl \$(md5sum < $P/\$v/inputs/wf1_heavy/wl/grounded_g0_n50000.json | cut -c1-32)  cfg+manifest md5 \$( (cd $P/\$v/inputs/wf1_heavy && md5sum cfg/*.json | md5sum | cut -c1-32))"; done
echo "glue (post-mint) g0 heavy: x0 082288f2  x0p3 ad875724  tier0 731f0ba9  x1 e08f33d3"
echo done
G
k=$(sbatch --parsable --dependency=afterok:$a:$b:$c:$d:$e $P/group.sbatch); echo "group+compare job $k"
