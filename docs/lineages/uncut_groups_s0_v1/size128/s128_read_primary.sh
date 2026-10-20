#!/bin/bash
#SBATCH --job-name=s128-read-primary
#SBATCH --partition=CPU-amd
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --exclude=os-cpu-slurm-21,os-cpu-slurm-15
#SBATCH --output=/share/nikola.lukic/uncut_groups/size128/reads/s128-read-primary-%j.out
# FL size-128, the PRIMARY read (F1; frozen P11: s6/reader-s128-deg bd00b187, blob b87ca2d9, frozen at reference-physics f4b1a4d2).
# Registered flags only. --gate is the bar and FL gnn cells (576); cells_desc, mlp_cells and cells_reactive stay sealed. Written once;
# its md5 is filed before the secondary opens anything.
set -uo pipefail
eval "$(micromamba shell hook --shell bash)" && micromamba activate gnn
U=/share/nikola.lukic/uncut_groups; R=$U/size128; REPO=$HOME/gnn-herosim
RD=$HOME/gnn-herosim-wt/s6_reader_bd00b187; cd $RD
[[ $(git rev-parse HEAD) == bd00b187* && -z "$(git status --porcelain)" ]] || { echo "FAIL LOUD: reader HEAD"; exit 1; }
[[ $(git rev-parse HEAD:scripts_cosim/uncut_arm_gate_read.py) == b87ca2d9* ]] || { echo "FAIL LOUD: reader blob"; exit 1; }
[[ ! -e $R/reads/read_s128_primary.json ]] || { echo "FAIL LOUD: a primary read exists; one read only"; exit 1; }
git -C $REPO fetch -q origin || { echo "FAIL LOUD: fetch"; exit 1; }
git -C $REPO merge-base --is-ancestor 3ee081248c5c1f6e1539e0f399e7c5a7caad8d69 origin/reference-physics || { echo "FAIL LOUD: the twin secondary Record line is not merged yet"; exit 1; }
T=$(seq -s, 30241 30252); H=b3fa6bee1a6092df70fcf825bc83126fe05864707ab89352a967c58ea3ef3be8
echo "cells: bar $(ls $R/cells_bar/*.summary.json | wc -l) learned $(ls $R/cells_learned/*.summary.json | wc -l) summaries; failed $(ls $R/cells_bar/*.failed.json $R/cells_learned/*.failed.json 2>/dev/null | wc -l)"
python3 scripts_cosim/uncut_arm_gate_read.py --gate $R/cells_bar,$R/cells_learned --topos $T --topos-sha256 $H --repo $REPO \
  --pin a459dd98 --gate-change 095948ea,f6818d09 --w-pins 74ddf70e --bar cd_ib0_inflightfull --cotenancy required \
  --primary ra_gnn_eng_combfl_nosplit --checkpoint-md5 $U/comb/md5_comball.json --seeds 1,2 --suffix _k100 \
  --confirm-size128 --stage primary \
  --out $R/reads/read_s128_primary.json > $R/reads/read_s128_primary.stdout 2>&1
rc=$?
if [[ $rc != 0 || ! -s $R/reads/read_s128_primary.json ]]; then
  echo "PRIMARY READ FAILED rc=$rc"; tail -5 $R/reads/read_s128_primary.stdout; exit 1
fi
echo "[primary] read_s128_primary.json md5 $(md5sum < $R/reads/read_s128_primary.json | cut -c1-32)"
echo "[primary] stdout md5 $(md5sum < $R/reads/read_s128_primary.stdout | cut -c1-32)"
cat $R/reads/read_s128_primary.stdout
