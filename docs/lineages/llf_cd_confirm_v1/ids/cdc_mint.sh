#!/bin/bash
# live_link_features_v1 confirmation topologies: 24 fresh 24-server topologies (llf_cd_confirm_v1, amendment A1), xs x1 recipe, then every peer-exchange payload clipped at 200 MB (cap200_build.sbatch recipe, md5 e05fbf3d1647037564bd9fe6e90532aa), cellular 4 MB/s unchanged.
# usage: llf_mint.sh "<24 ids>" <sha256 of the comma-joined list>      (recipe commit 224dd02e, worktree uh_xs_224dd02e, as the xs set)
set -euo pipefail
IDS=$1; WANT=$2; SHA=224dd02e
U=/share/nikola.lukic/uncut_groups; X=$U/llf_cd_conf; WT=$HOME/gnn-herosim-wt/uh_xs_${SHA:0:8}
[ "$(echo $IDS | wc -w)" = 24 ] || { echo "FAIL LOUD: need 24 ids"; exit 2; }
GOT=$(echo -n $IDS | tr ' ' ',' | sha256sum | cut -c1-64); [ "$GOT" = "$WANT" ] || { echo "FAIL LOUD: id list sha256 $GOT != $WANT"; exit 2; }
for t in $IDS; do [[ $t =~ ^[0-9]{5}$ ]] && [ $t -gt 30303 ] || { echo "FAIL LOUD: id $t is not a 5-digit id >= 30304"; exit 2; }; done
[ ! -e $X ] || { echo "FAIL LOUD: $X exists"; exit 2; }
[[ $(git -C $WT rev-parse HEAD) == $SHA* && -z "$(git -C $WT status --porcelain)" ]] || { echo "FAIL LOUD: worktree $WT not clean at $SHA"; exit 2; }
[ "$(md5sum < $WT/scripts_cosim/datalab/big_groups_inputs.sbatch | cut -c1-32)" = db71ec73a67a67bf665dfc664979b19c ] || { echo "FAIL LOUD: big_groups_inputs.sbatch md5"; exit 2; }
[ "$(md5sum < $WT/scripts_cosim/uncut_groups_group_inputs.py | cut -c1-32)" = 9efb1da057cb85a2f9eececd0f5b684c ] || { echo "FAIL LOUD: uncut_groups_group_inputs.py md5"; exit 2; }
[ $(( $(squeue -u nikola.lukic -h | wc -l) + 3 )) -le 44 ] || { echo "FAIL LOUD: 3 more jobs would exceed 44"; exit 2; }
mkdir -p $X/prep $X/logs; echo $IDS > $X/ids.txt
ROOT=$X/gate_inputs_x1
J=$(env WT=$WT EXPECT_HEAD=${SHA:0:8} ROOT=$ROOT X1=$U/x1_uncut TOPOS="$IDS" NC=160 NS=24 P=0.6 M_MODERATE=11.6139 M_HEAVY=27.622665025860204 REPLICA_RULE=fastest_compatible BASECFG=/share/nikola.lukic/accel_nosplit/inputs/cfg_src/cc40s16345.json sbatch --parsable --export=ALL -o $X/logs/mint-%j.out $WT/scripts_cosim/datalab/big_groups_inputs.sbatch)
cat > $X/prep/group.sbatch <<G
#!/bin/bash
#SBATCH -J llf-grp
#SBATCH -p CPU-amd
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:20:00
#SBATCH -o $X/logs/group-%j.out
set -euo pipefail
eval "\$(micromamba shell hook --shell bash)" && micromamba activate gnn
cd $WT
python3 scripts_cosim/uncut_groups_group_inputs.py --src $ROOT/inputs --dst $ROOT/inputs_group
(cd $ROOT/inputs && md5sum -c MD5SUMS | grep -vc OK || true)
(cd $ROOT/inputs_group && md5sum -c MD5SUMS | grep -vc OK || true)
echo "x1 inputs MD5SUMS \$(md5sum < $ROOT/inputs/MD5SUMS | cut -c1-32) inputs_group MD5SUMS \$(md5sum < $ROOT/inputs_group/MD5SUMS | cut -c1-32)"
python3 - $ROOT "$IDS" <<'P'
import json, sys
root, ids = sys.argv[1], sys.argv[2].split()
for tag in ("moderate", "heavy"):
    for t in ids:
        c = json.load(open(f"{root}/inputs_group/wf1_{tag}/cfg/cc40s{t}.json"))
        assert (c["nodes"]["client_nodes"]["count"], c["nodes"]["server_nodes"]["count"]) == (160, 24), (tag, t)
        assert c["network"]["topology"]["seed"] == int(t), (tag, t)
print("cfgs OK: 160 x 24, topology seed = id")
P
echo done
G
K=$(sbatch --parsable --dependency=afterok:$J $X/prep/group.sbatch)
cat > $X/prep/cap200.sbatch <<'C'
#!/bin/bash
#SBATCH -J llf-cap200
#SBATCH -p CPU-amd
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 00:30:00
#SBATCH -o /share/nikola.lukic/uncut_groups/llf_cd_conf/logs/cap200-%j.out
set -euo pipefail
eval "$(micromamba shell hook --shell bash)" && micromamba activate gnn
X=/share/nikola.lukic/uncut_groups/llf_cd_conf; U=/share/nikola.lukic/uncut_groups; SRC=$X/gate_inputs_x1/inputs_group; DST=$X/gate_inputs_cap200/inputs_group
[[ ! -e $X/gate_inputs_cap200 ]] || { echo "FAIL LOUD: gate_inputs_cap200 exists"; exit 2; }
mkdir -p $DST
cp $SRC/manifest_*.json $DST/
for r in moderate heavy; do mkdir -p $DST/wf1_$r/wl; cp -r $SRC/wf1_$r/cfg $DST/wf1_$r/cfg; done
python3 - "$SRC" "$DST" <<'EOF'
import glob, hashlib, json, os, sys
src, dst = sys.argv[1], sys.argv[2]
CAP = 200e6
for f in sorted(glob.glob(os.path.join(src, "wf1_*/wl/*.json"))):
    raw = open(f, "rb").read()
    d = json.loads(raw)
    pe = d["peer_exchange"]
    n_clip = sum(1 for p in pe if p[2] > CAP)
    before = sum(p[2] for p in pe)
    for p in pe:
        if p[2] > CAP:
            p[2] = CAP
    after = sum(p[2] for p in pe)
    d["payload_cap_v1"] = {"cap_bytes": CAP, "rule": "min(bytes, cap) per pair; nothing else changed",
                           "source_sha256": hashlib.sha256(raw).hexdigest(), "pairs": len(pe), "pairs_clipped": n_clip,
                           "bytes_before": before, "bytes_after": after}
    out = os.path.join(dst, os.path.relpath(f, src))
    with open(out + ".tmp", "w") as fh:
        json.dump(d, fh)
    os.replace(out + ".tmp", out)
    print(os.path.relpath(f, src), "pairs", len(pe), "clipped", n_clip, "bytes kept %.3f" % (after / before))
EOF
( cd $DST && find . -type f ! -name MD5SUMS | sort | xargs md5sum > MD5SUMS && md5sum -c --quiet MD5SUMS && echo "MD5SUMS ok $(wc -l < MD5SUMS) files" )
echo "cap200 inputs_group MD5SUMS $(md5sum < $DST/MD5SUMS | cut -c1-32)"
echo "cfg md5 x1 vs cap200: $(cat $SRC/wf1_moderate/cfg/*.json $SRC/wf1_heavy/cfg/*.json | md5sum | cut -c1-32) $(cat $DST/wf1_moderate/cfg/*.json $DST/wf1_heavy/cfg/*.json | md5sum | cut -c1-32)"
mkdir -p $X/in_cap200/models
cp -p $U/xs/in_cap200/uncut_arm1_split.json $X/in_cap200/
cp -p $U/xs/in_cap200/models/* $X/in_cap200/models/
for r in moderate heavy; do ln -s $DST/wf1_$r $X/in_cap200/wf1_$r; done
echo DONE
C
L=$(sbatch --parsable --dependency=afterok:$K $X/prep/cap200.sbatch)
echo "mint job $J group job $K cap200 job $L"
