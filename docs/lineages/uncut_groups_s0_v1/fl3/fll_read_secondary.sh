#!/bin/bash
#SBATCH --job-name=fll-read-secondary
#SBATCH --partition=CPU-amd
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --exclude=os-cpu-slurm-21,os-cpu-slurm-15
#SBATCH --output=/share/nikola.lukic/uncut_groups/fl3/reads/fll-read-secondary-%j.out
# FL + lambda, the descriptive SECONDARY read (frozen P8.1: s6/reader-fllambda-p81 34c3297c, blob bfbf124e). After the primary
# (md5 e6cbb39d) is filed at s6/fll-primary b74ec353 on origin/reference-physics. Opens cells_desc (768) only; cells_twin and
# cells_react stay sealed. Written once.
set -uo pipefail
eval "$(micromamba shell hook --shell bash)" && micromamba activate gnn
U=/share/nikola.lukic/uncut_groups; F=$U/fl3; REPO=$HOME/gnn-herosim
RD=$HOME/gnn-herosim-wt/s6_reader_34c3297c; cd $RD
[[ $(git rev-parse HEAD) == 34c3297c* && -z "$(git status --porcelain)" ]] || { echo "FAIL LOUD: reader HEAD"; exit 1; }
[[ $(git rev-parse HEAD:scripts_cosim/uncut_arm_gate_read.py) == bfbf124e02d6b3f5fa1844cef8c3fb1469b91595 ]] || { echo "FAIL LOUD: reader blob"; exit 1; }
[[ ! -e $F/reads/read_fll_secondary.json ]] || { echo "FAIL LOUD: a secondary read exists; one read only"; exit 1; }
[[ $(md5sum < $F/reads/read_fll_primary.json | cut -c1-32) == e6cbb39db912e4705a27852bb28d844e ]] || { echo "FAIL LOUD: primary read md5"; exit 1; }
[[ $(md5sum < $F/md5_fl3.json | cut -c1-32) == 61ee96d42d960d951008d2add1e62a57 ]] || { echo "FAIL LOUD: md5_fl3.json"; exit 1; }
git -C $REPO fetch -q origin || { echo "FAIL LOUD: fetch"; exit 1; }
REC=b74ec353ba28a533892af0e5bd03cd290a5bded6
git -C $REPO merge-base --is-ancestor $REC origin/reference-physics || { echo "FAIL LOUD: the primary Record line is not on origin/reference-physics"; exit 1; }
T=$(seq -s, 30149 30160); H=f6c5af0a8f74a8caa83ba45a97903ec47798ae8898e8e226b3ff22291bb59379
echo "cells: desc $(ls $F/cells_desc/*.summary.json | wc -l) summaries; failed $(ls $F/cells_desc/*.failed.json 2>/dev/null | wc -l)"
python3 scripts_cosim/uncut_arm_gate_read.py --gate $F/cells_learned,$F/cells_bar --topos $T --topos-sha256 $H --repo $REPO \
  --pin a459dd98 --gate-change 095948ea,f6818d09,74ddf70e,b8909f58,05ec45c1 --w-pins 91269ec0 \
  --bar cd_ib0_inflightfull --cotenancy required --primary ra_gnn_eng_flreg_nosplit \
  --checkpoint-md5 $F/md5_fl3.json,$U/comb/md5_comball.json --seeds 1,2 --suffix _k100 \
  --confirm-fl-lambda --stage secondary --desc-gate $F/cells_desc --mlp ra_mlp_same_flreg_nosplit \
  --primary-read $F/reads/read_fll_primary.json --record $REC \
  --out $F/reads/read_fll_secondary.json > $F/reads/read_fll_secondary.stdout 2>&1
rc=$?
if [[ $rc != 0 || ! -s $F/reads/read_fll_secondary.json ]]; then
  echo "SECONDARY READ FAILED rc=$rc"; tail -5 $F/reads/read_fll_secondary.stdout; exit 1
fi
echo "[secondary] read_fll_secondary.json md5 $(md5sum < $F/reads/read_fll_secondary.json | cut -c1-32)"
echo "[secondary] stdout md5 $(md5sum < $F/reads/read_fll_secondary.stdout | cut -c1-32)"
cat $F/reads/read_fll_secondary.stdout
