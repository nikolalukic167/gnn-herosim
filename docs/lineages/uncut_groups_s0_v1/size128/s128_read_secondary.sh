#!/bin/bash
#SBATCH --job-name=s128-read-secondary
#SBATCH --partition=CPU-amd
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --exclude=os-cpu-slurm-21,os-cpu-slurm-15
#SBATCH --output=/share/nikola.lukic/uncut_groups/size128/reads/s128-read-secondary-%j.out
# FL size-128, the SECONDARY read (F2 + decision-charge rows + descriptive; frozen P11 bd00b187, blob b87ca2d9). After F1's md5
# (dfd3a5ca, Record 4e486119) is on origin/reference-physics. Opens mlp_cells (384), cells_desc (288) and cells_react (96). Written once.
set -uo pipefail
eval "$(micromamba shell hook --shell bash)" && micromamba activate gnn
U=/share/nikola.lukic/uncut_groups; R=$U/size128; REPO=$HOME/gnn-herosim
RD=$HOME/gnn-herosim-wt/s6_reader_bd00b187; cd $RD
[[ $(git rev-parse HEAD) == bd00b187* && -z "$(git status --porcelain)" ]] || { echo "FAIL LOUD: reader HEAD"; exit 1; }
[[ $(git rev-parse HEAD:scripts_cosim/uncut_arm_gate_read.py) == b87ca2d9* ]] || { echo "FAIL LOUD: reader blob"; exit 1; }
[[ ! -e $R/reads/read_s128_secondary.json ]] || { echo "FAIL LOUD: a secondary read exists; one read only"; exit 1; }
[[ $(md5sum < $R/reads/read_s128_primary.json | cut -c1-32) == dfd3a5ca2bd3d6eb0a40b45b03790bf8 ]] || { echo "FAIL LOUD: F1 md5"; exit 1; }
git -C $REPO fetch -q origin || { echo "FAIL LOUD: fetch"; exit 1; }
REC=4e486119ac847ed17cb0246e0c2dc787a420619f
git -C $REPO merge-base --is-ancestor $REC origin/reference-physics || { echo "FAIL LOUD: the F1 Record line is not merged yet"; exit 1; }
T=$(seq -s, 30241 30252); H=b3fa6bee1a6092df70fcf825bc83126fe05864707ab89352a967c58ea3ef3be8
echo "cells: mlp $(ls $R/mlp_cells/*.summary.json | wc -l) desc $(ls $R/cells_desc/*.summary.json | wc -l) react $(ls $R/cells_react/*.summary.json | wc -l) summaries; failed $(ls $R/mlp_cells/*.failed.json $R/cells_desc/*.failed.json $R/cells_react/*.failed.json 2>/dev/null | wc -l)"
python3 scripts_cosim/uncut_arm_gate_read.py --gate $R/cells_bar,$R/cells_learned --topos $T --topos-sha256 $H --repo $REPO \
  --pin a459dd98 --gate-change 095948ea,f6818d09 --w-pins 74ddf70e --bar cd_ib0_inflightfull --cotenancy required \
  --primary ra_gnn_eng_combfl_nosplit --checkpoint-md5 $U/comb/md5_comball.json --seeds 1,2 --suffix _k100 \
  --confirm-size128 --stage secondary --mlp ra_mlp_same_combfl_nosplit --mlp-gate $R/mlp_cells --desc-gate $R/cells_desc \
  --reactive-gate $R/cells_react --primary-read $R/reads/read_s128_primary.json --record $REC \
  --out $R/reads/read_s128_secondary.json > $R/reads/read_s128_secondary.stdout 2>&1
rc=$?
if [[ $rc != 0 || ! -s $R/reads/read_s128_secondary.json ]]; then
  echo "SECONDARY READ FAILED rc=$rc"; tail -5 $R/reads/read_s128_secondary.stdout; exit 1
fi
echo "[secondary] read_s128_secondary.json md5 $(md5sum < $R/reads/read_s128_secondary.json | cut -c1-32)"
echo "[secondary] stdout md5 $(md5sum < $R/reads/read_s128_secondary.stdout | cut -c1-32)"
cat $R/reads/read_s128_secondary.stdout
