#!/bin/bash
# r1_attribution_v1 gate: identity of the existing kinds after a code change, and a servesmoke of a new kind.
# Re-runs ARMS (default: every classical arm and ra_gnn_eng / ra_gnn_eng_cdapply at seed 1) for TOPO x g0 x 3 rungs from WT and compares each cell
# with BASE (a gate directory from the earlier commit) on every simulated field; then runs NEW (default cd_random_seed, seeds 1 and 2) once.
#   sbatch --export=ALL,WT=...,EXPECT_HEAD=...,IN=<inputs>,BASE=<earlier gate dir>,OUT=<new dir> r1a_gate_identity.sbatch
#SBATCH --job-name=r1a-id
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/r1a-id-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=03:00:00
set -euo pipefail
WT=${WT:?}; EXPECT_HEAD=${EXPECT_HEAD:?}; IN=${IN:?}; BASE=${BASE:?}; OUT=${OUT:?}; TOPO=${TOPO:-9483}
ARMS=${ARMS:-cd,cd_declared,locality,batched,selfpredict,reactive,ra_gnn_eng,ra_gnn_eng_cdapply}
NEW=${NEW:-cd_random_seed}
D=/home/nikola.lukic/gnn-herosim/simulation_data
SMOKE=$D/workload_fix_v1/r1a_arms_servesmoke_inputs
cd "$WT"
for v in $(compgen -e | grep -E '^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|NEAR_RTT_|PARTIAL_STATE|WF1_|R1A_)'); do unset "$v"; done
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT" PYTHONUNBUFFERED=1
export HEROSIM_RAW_DIR=/tmp/herosim_raw_${SLURM_JOB_ID:-$$}; mkdir -p "$HEROSIM_RAW_DIR"; trap 'rm -rf "$HEROSIM_RAW_DIR"' EXIT
HEAD=$(git rev-parse HEAD); echo "HEAD $HEAD node $(hostname) OUT=$OUT BASE=$BASE"
[[ "$HEAD" == "$EXPECT_HEAD"* ]] || { echo "FAIL LOUD: worktree HEAD $HEAD is not $EXPECT_HEAD"; exit 1; }
[[ -z "$(git status --porcelain -- src scripts_cosim experiments run_experiment.py)" ]] || { echo "FAIL LOUD: worktree dirty"; exit 1; }
[[ ! -e "$OUT" ]] || { echo "FAIL LOUD: $OUT exists"; exit 1; }
mkdir -p "$OUT/in"
for r in light moderate heavy; do ln -s "$IN/wf1_$r" "$OUT/in/wf1_$r"; done
ln -s "$SMOKE/models" "$OUT/in/models"; cp "$SMOKE/r1_attribution_v1_split.json" "$OUT/in/"
SEL=$D/small_batch_confirm_v1/inputs/selected.json
export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa GATE_FIXED_POLICY_TIME_SCALE=1.0 HEROSIM_SHARED_AUTOSCALER=0
export WF1_RUNGS=light,moderate,heavy R1A_TOPOS=$TOPO R1A_WINDOWS=g0
set +e
R1A_ARMS=$ARMS R1A_SEEDS=1 python3 scripts_cosim/fresh_topo_burst_v1_gate.py r1a --inputs "$OUT/in" --out "$OUT/gate" --selection "$SEL" --parallel 24 --mem 8G --timeout 8100 --no-scope | grep -v "^\[skip" &
R1A_ARMS=$NEW R1A_SEEDS=1,2 python3 scripts_cosim/fresh_topo_burst_v1_gate.py r1a --inputs "$OUT/in" --out "$OUT/new" --selection "$SEL" --parallel 6 --mem 8G --timeout 8100 --no-scope | grep -v "^\[skip" &
wait
set -e
echo "== identity vs $BASE =="
python3 scripts_cosim/r1a_gate_identity.py "$BASE" "$OUT/gate" || echo "IDENTITY-FAILED"
echo "== new kind =="
ls "$OUT"/new/*.failed.json 2>/dev/null && cat "$OUT"/new/*.failed.json
python3 scripts_cosim/r1_attribution_v1_read.py --gate "$OUT/new" | head -12
