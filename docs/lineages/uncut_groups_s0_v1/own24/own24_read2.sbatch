#!/bin/bash
#SBATCH -J s5-own24-read
#SBATCH -p CPU-amd
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH -o /share/nikola.lukic/uncut_groups/own24/logs/read-%j.out
# Own-state round primary read: the frozen --confirm-own-state reader at reference-physics b9963f51 (blob e3921965 (pin 6bab474c)), run once, from /share/nikola.lukic/uncut_groups.
set -euo pipefail
eval "$(micromamba shell hook --shell bash)" && micromamba activate gnn
RW=$HOME/gnn-herosim-wt/s5_reader_6bab474c
[[ $(git -C $RW rev-parse HEAD) == 6bab474c49051de5af2ce03d4da13df055a8fbb5 && $(git -C $RW status --porcelain | wc -l) == 0 ]] || { echo "FAIL LOUD: reader worktree"; exit 2; }
[[ $(git -C $RW rev-parse HEAD:scripts_cosim/uncut_arm_gate_read.py) == e3921965* ]] || { echo "FAIL LOUD: reader blob"; exit 2; }
cd /share/nikola.lukic/uncut_groups
mkdir -p own24/reads; [ -e own24/reads/read_own_primary.json ] && { echo "FAIL LOUD: read exists (written once)"; exit 2; }
python3 $RW/scripts_cosim/uncut_arm_gate_read.py --confirm-own-state --stage primary --pin a459dd98 --primary ra_gnn_eng_combfl_dgu_nosplit --twin ra_twin_eng_combfl_dgu_nosplit --bar cd_ib0_inflightfull --seeds 1,2,3,4,5,6 --suffix _k100 --cotenancy required \
  --topos 20508,20509,20510,20511,20512,20513,20514,20515,20516,20517,20518,20519,20520,20521,20522,20523,20524,20525,20526,20527,20528,20529,20530,20531 --topos-sha256 d4b1fb268c14af65a7c201ef27a97fbfb3a3033a964218cfbf443f060032ac37 \
  --repo $RW --gate own24/cells_new_gnn,own24/cells_new_twin --old-gate own24/cells_old --bar-gate own24/cells_bar --react-gate own24/cells_react --ident-gate own24/ident --ident-ref fl3/cells_bar,fl3/cells_learned,fl3/cells_twin \
  --ckpt-snapshot own24/ckpt_snapshot/md5_dgu.json --checkpoint-md5 own24/ckpt_snapshot/md5_dgu.json,comb/md5_comball.json,fl3/md5_twin.json \
  --inputs-group gate_inputs_own24/inputs_group --cell-inputs own24/in_new,own24/in_old --out own24/reads/read_own_primary.json
md5sum own24/reads/read_own_primary.json
