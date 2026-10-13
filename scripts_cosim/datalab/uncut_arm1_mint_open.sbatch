#!/bin/bash
# uncut_groups_s0_v1 Arm 1: opening-only corpus windows from the UNUSED tail of each library (after window B), SCIENTIST's ruling 2026-10-13: NO group range overlaps A or B.
# Per shard, N_OPEN windows of 5,500 tasks (a heavy cell's 360 s opening is ~4,600 tasks; moderate ~1,900), each starting 500 groups after the previous window's last group.
#   sbatch --export=ALL,WT=<pin>,EXPECT_HEAD=<sha>,MINT=/share/nikola.lukic/uncut_arm1/mint,N_OPEN=2 uncut_arm1_mint_open.sbatch
#SBATCH --job-name=arm1-mintopen
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/arm1-mintopen-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
set -euo pipefail
WT=${WT:?FAIL LOUD: set WT}; EXPECT_HEAD=${EXPECT_HEAD:?}; MINT=${MINT:?}; N_OPEN=${N_OPEN:-2}
LIBS=/share/nikola.lukic/uncut_groups/libs/uncut; GATE=/share/nikola.lukic/uncut_groups/x1_uncut
BASEWL=/home/nikola.lukic/gnn-herosim/simulation_data/peer_affinity_live_gate/workloads
cd "$WT"
for v in $(compgen -e | grep -E '^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|NEAR_RTT_|PARTIAL_STATE|WF1_)'); do unset "$v"; done
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"; micromamba activate gnn
export PYTHONPATH="$WT" HEROSIM_PY=python3
HEAD=$(git rev-parse HEAD); [[ "$HEAD" == "$EXPECT_HEAD"* ]] || { echo "FAIL LOUD: worktree HEAD $HEAD is not $EXPECT_HEAD"; exit 1; }
i=0
for row in 0:drainable_f4000_n50000 50:drainable_w1_n50000 100:drainable_w2_n50000 130:drainable_w3_n50000; do
  IFS=: read -r K BW <<< "$row"
  END=$(python3 -c "import json;m=json.load(open('$MINT/x1B/grounded_g${i}_n50000.json'))['grounded_workload_v1'];print(m['start_group']+m['n_groups'])")
  EXCL="$GATE/mint_g$i.json $MINT/x1A/grounded_g${i}_n50000.json $MINT/x1B/grounded_g${i}_n50000.json"
  for k in $(seq 1 "$N_OPEN"); do
    mkdir -p "$MINT/x1C$k"
    S=$((END + 500))
    python3 scripts_cosim/grounded_workload_v1_mint.py --lib "$LIBS/fanout_lib_$K.json" --base "$BASEWL/$BW.json" --seed $((7420 + 10 * k + i)) --n-tasks 5500 --start-group "$S" \
      --exclude-mint $EXCL --out "$MINT/x1C$k/grounded_g${i}_n50000.json" >/dev/null
    NG=$(python3 -c "import json;m=json.load(open('$MINT/x1C$k/grounded_g${i}_n50000.json'))['grounded_workload_v1'];print(m['n_groups'])")
    echo "[mint] shard g$i opening window C$k: start_group $S n_groups $NG"
    EXCL="$EXCL $MINT/x1C$k/grounded_g${i}_n50000.json"
    END=$((S + NG))
  done
  i=$((i + 1))
done
echo done
