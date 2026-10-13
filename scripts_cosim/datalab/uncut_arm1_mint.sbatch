#!/bin/bash
# uncut_groups_s0_v1 Arm 1 corpus: mint the corpus windows from the uncut libraries, option (a): library groups from G0 to the library's end
# (the gate consumes groups below G0; the mint asserts zero overlap with the gate's own mint). Two windows per shard: A at G0, B after A's last group + 500.
#   sbatch --export=ALL,WT=<pin>,EXPECT_HEAD=<sha>,OUT=<dir that does not exist> uncut_arm1_mint.sbatch
#SBATCH --job-name=arm1-mint
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/arm1-mint-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
set -euo pipefail
WT=${WT:?FAIL LOUD: set WT}; EXPECT_HEAD=${EXPECT_HEAD:?}; OUT=${OUT:?}
LIBS=/share/nikola.lukic/uncut_groups/libs/uncut; GATE=/share/nikola.lukic/uncut_groups/x1_uncut
BASEWL=/home/nikola.lukic/gnn-herosim/simulation_data/peer_affinity_live_gate/workloads
cd "$WT"
for v in $(compgen -e | grep -E '^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|NEAR_RTT_|PARTIAL_STATE|WF1_)'); do unset "$v"; done
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"; micromamba activate gnn
export PYTHONPATH="$WT" HEROSIM_PY=python3
HEAD=$(git rev-parse HEAD); echo "HEAD $HEAD"; [[ "$HEAD" == "$EXPECT_HEAD"* ]] || { echo "FAIL LOUD: worktree HEAD $HEAD is not $EXPECT_HEAD"; exit 1; }
[[ ! -e "$OUT" ]] || { echo "FAIL LOUD: $OUT exists"; exit 1; }
mkdir -p "$OUT/x1A" "$OUT/x1B"
i=0
for row in 0:drainable_f4000_n50000:12720 50:drainable_w1_n50000:13295 100:drainable_w2_n50000:12954 130:drainable_w3_n50000:13271; do
  IFS=: read -r K BW G0 <<< "$row"
  python3 scripts_cosim/grounded_workload_v1_mint.py --lib "$LIBS/fanout_lib_$K.json" --base "$BASEWL/$BW.json" --seed $((7400 + i)) --start-group "$G0" \
    --exclude-mint "$GATE/mint_g$i.json" --out "$OUT/x1A/grounded_g${i}_n50000.json" >/dev/null
  NG=$(python3 -c "import json;print(json.load(open('$OUT/x1A/grounded_g${i}_n50000.json'))['grounded_workload_v1']['n_groups'])")
  GB=$((G0 + NG + 500))
  python3 scripts_cosim/grounded_workload_v1_mint.py --lib "$LIBS/fanout_lib_$K.json" --base "$BASEWL/$BW.json" --seed $((7410 + i)) --start-group "$GB" \
    --exclude-mint "$GATE/mint_g$i.json" "$OUT/x1A/grounded_g${i}_n50000.json" --out "$OUT/x1B/grounded_g${i}_n50000.json" >/dev/null
  python3 - "$OUT" "$i" "$G0" "$GB" <<'PY'
import json, sys
out, i, g0, gb = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
for w in ("A", "B"):
    m = json.load(open(f"{out}/x1{w}/grounded_g{i}_n50000.json"))["grounded_workload_v1"]
    print(f"[mint] shard g{i} window {w}: start_group {m.get('start_group', 0)} n_groups {m['n_groups']} n_tasks {m['n_tasks']} rate {m['rate_per_s']:.5f}/s (base {m['base_rate_per_s']:.5f}) mean group {m['mean_group_size']:.2f}")
PY
  i=$((i + 1))
done
echo "minted: $OUT/x1A $OUT/x1B"
