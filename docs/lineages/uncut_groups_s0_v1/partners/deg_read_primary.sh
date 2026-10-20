#!/bin/bash
#SBATCH --job-name=deg-read-primary
#SBATCH --partition=CPU-amd
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --exclude=os-cpu-slurm-21,os-cpu-slurm-15
#SBATCH --output=/share/nikola.lukic/uncut_groups/partners/reads/deg-read-primary-%j.out
# Partner structure, the PRIMARY read (--confirm-degree-trend; frozen P11: s6/reader-s128-deg bd00b187, blob b87ca2d9). Reads
# --deg-root partners/deg_view (a read-only symlink view mapping S5's tree onto the reader's k1/k2/k4/k8 layout; no file copied or
# changed). After size-128's secondary Record line (0c850494) is on origin/reference-physics. Written once.
set -uo pipefail
eval "$(micromamba shell hook --shell bash)" && micromamba activate gnn
U=/share/nikola.lukic/uncut_groups; P=$U/partners; V=$P/deg_view; REPO=$HOME/gnn-herosim
RD=$HOME/gnn-herosim-wt/s6_reader_bd00b187; cd $RD
[[ $(git rev-parse HEAD) == bd00b187* && -z "$(git status --porcelain)" ]] || { echo "FAIL LOUD: reader HEAD"; exit 1; }
[[ $(git rev-parse HEAD:scripts_cosim/uncut_arm_gate_read.py) == b87ca2d9* ]] || { echo "FAIL LOUD: reader blob"; exit 1; }
[[ ! -e $P/reads/read_deg_primary.json ]] || { echo "FAIL LOUD: a primary read exists; one read only"; exit 1; }
git -C $REPO fetch -q origin || { echo "FAIL LOUD: fetch"; exit 1; }
git -C $REPO merge-base --is-ancestor 0c8504949109d69d38ba0ac8c77912b47a879f7d origin/reference-physics || { echo "FAIL LOUD: size-128 secondary Record line not merged yet"; exit 1; }
[[ $(cd $V && ls -lR --time-style=+ . | md5sum | cut -c1-32) == ecce0bb1ed8cdecf191763ceee1d9bed ]] || { echo "FAIL LOUD: deg_view listing md5"; exit 1; }
T=$(seq -s, 30176 30187); H=dd359919d7fc42c5e0bec1f40d934edb6192b0993fa542cd08608b56e59cceb7
for L in k1 k2 k4 k8; do echo "$L: learned $(ls $V/$L/cells_learned/ | grep -c summary.json) bar $(ls $V/$L/cells_bar/ | grep -c summary.json) reactive $(ls $V/$L/cells_reactive/ | grep -c summary.json); failed $(ls $V/$L/cells_learned/ $V/$L/cells_reactive/ | grep -c failed.json)"; done
python3 scripts_cosim/uncut_arm_gate_read.py --gate $P/unused --topos $T --topos-sha256 $H --repo $REPO \
  --pin a459dd98 --gate-change 095948ea,f6818d09 --w-pins 74ddf70e --bar cd_ib0_inflightfull \
  --primary ra_gnn_eng_combfl_nosplit --mlp ra_mlp_same_combfl_nosplit --checkpoint-md5 $U/comb/md5_comball.json --seeds 1,2 --suffix _k100 \
  --confirm-degree-trend --deg-root $V --stage primary \
  --out $P/reads/read_deg_primary.json > $P/reads/read_deg_primary.stdout 2>&1
rc=$?
if [[ $rc != 0 || ! -s $P/reads/read_deg_primary.json ]]; then
  echo "PRIMARY READ FAILED rc=$rc"; tail -5 $P/reads/read_deg_primary.stdout; exit 1
fi
echo "[primary] read_deg_primary.json md5 $(md5sum < $P/reads/read_deg_primary.json | cut -c1-32)"
echo "[primary] stdout md5 $(md5sum < $P/reads/read_deg_primary.stdout | cut -c1-32)"
cat $P/reads/read_deg_primary.stdout
