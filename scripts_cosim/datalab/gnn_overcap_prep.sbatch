#!/bin/bash
# gnn_overcap_v1 prep, one job: tests, topology-name check, the G = 8 and current-group-size inputs for 19206-19217, the rate check against the G = 8 gate, then the smoke and
# identity cells (the smoke cells of the gate arms are gate cells in the gate's own OUT and are not rerun).
#   sbatch --export=ALL,WT=<pinned worktree>,EXPECT_HEAD=<sha> gnn_overcap_prep.sbatch
#SBATCH --job-name=oc-prep
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/oc-prep-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=10:00:00
set -euo pipefail
WT=${WT:?FAIL LOUD: set WT}; EXPECT_HEAD=${EXPECT_HEAD:?FAIL LOUD: set EXPECT_HEAD}
cd "$WT"
for v in $(compgen -e | grep -E '^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|NEAR_RTT_|PARTIAL_STATE|WF1_)'); do unset "$v"; done
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"; micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT" PYTHONUNBUFFERED=1
HEAD=$(git rev-parse HEAD); echo "HEAD $HEAD node $(hostname) job=${SLURM_JOB_ID:-}"
[[ "$HEAD" == "$EXPECT_HEAD"* ]] || { echo "FAIL LOUD: worktree HEAD $HEAD is not $EXPECT_HEAD"; exit 1; }
L=/share/nikola.lukic; O=$L/gnn_overcap; B=$L/big_groups
TOPOS="19206 19207 19208 19209 19210 19211 19212 19213 19214 19215 19216 19217"

echo "== 1. tests"
python3 -m pytest tests/test_gnn_overcap.py tests/test_gnn_overcap_gate_read.py tests/test_gnn_overcap_batches_read.py tests/test_cd_expansion.py tests/test_prune_g8_gate_read.py -q 2>&1 | tail -4
python3 -m pytest tests/test_record_hygiene.py -q 2>&1 | tail -6 || echo "NOTE: record hygiene reported failures above (read them; the cells below still run)"

echo "== 2. topology names"
hits=$(find $L -maxdepth 7 \( -name 'cc40s1920[6-9]*' -o -name 'cc40s1921[0-7]*' \) -not -path "$O/*" 2>/dev/null | head -5)
[[ -z "$hits" ]] || { echo "FAIL LOUD: topology names already used: $hits"; exit 1; }
[[ ! -e $O/inputs_G8 && ! -e $O/inputs_G0 ]] || { echo "FAIL LOUD: $O/inputs_* exist"; exit 1; }
mkdir -p $O
python3 - <<P
import json
s = json.load(open("$B/selected_gate.json")); s["topologies"] = [int(t) for t in "$TOPOS".split()]; s["spares"] = []
s["lineage"] = "gnn_overcap_v1 live gate"; s["note"] = "12 fresh ids 19206-19217; token scan 2026-10-11 (docs/lineages/gnn_overcap_v1.md)"
json.dump(s, open("$O/selected.json", "w"), indent=1)
P

echo "== 3. inputs"
export M_MODERATE=11.6139 M_HEAVY=27.622665025860204 REPLICA_RULE=fastest_compatible
for G in 8 0; do
  WT=$WT EXPECT_HEAD=$EXPECT_HEAD ROOT=$O/inputs_G$G X1=$B/x1_G$G TOPOS="$TOPOS" bash scripts_cosim/datalab/big_groups_inputs.sbatch 2>&1 | tee $O/inputs_G$G.log | grep -E "^\[(rate|check)\]|inputs ready"
  cp -a /share/nikola.lukic/prune_g8/gate_inputs/inputs/models $O/inputs_G$G/inputs/models
  ln -s /share/nikola.lukic/accel_gate/inputs_41ef627b/inputs/accel_replica_v1_split.json $O/inputs_G$G/inputs/accel_replica_v1_split.json
  ls $O/inputs_G$G/inputs/models | grep -c '\.pt$' | xargs -I{} echo "models(.pt): {}"
done
python3 - <<'P'
import re, statistics as st
want = {"moderate": 5.3470, "heavy": 12.7173}
for g in (8,):
    rates = {}
    for l in open("/share/nikola.lukic/gnn_overcap/inputs_G8.log"):
        m = re.match(r"\[rate\] (\w+) (g\d): ([\d.]+) tasks/s, groups (\d+), mean group size ([\d.]+)", l)
        if m: rates.setdefault(m.group(1), []).append((float(m.group(3)), int(m.group(4)), float(m.group(5))))
    for tag, w in want.items():
        r = st.mean(x[0] for x in rates[tag]); print(f"[ratecheck] G8 {tag}: mean {r:.4f} tasks/s vs gate {w} ({100*(r/w-1):+.2f} %), groups {[x[1] for x in rates[tag]]}, mean group size {[x[2] for x in rates[tag]]}")
        assert abs(r / w - 1) <= 0.02, "FAIL LOUD: rate differs from the G=8 gate by more than 2 %"
P

echo "== 4. smoke and identity cells"
export R1A_SEEDS=1 R1A_STEM=accel-replica-v1 R1A_SPLIT=accel_replica_v1_split.json CELL_MEM=8G LIMIT=20000
export HEROSIM_RAW_DIR=/tmp/herosim_raw_${SLURM_JOB_ID:-$$}; mkdir -p "$HEROSIM_RAW_DIR"; trap 'rm -rf "$HEROSIM_RAW_DIR"' EXIT
export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa GATE_FIXED_POLICY_TIME_SCALE=1.0 HEROSIM_SHARED_AUTOSCALER=0
GATE=$O/gate_G8; SM=$O/smoke; ID=$O/ident
mkdir -p $GATE $SM $ID $O/gate_raw $O/gate_log $O/smoke_raw $O/smoke_log
run() {  # run <label> <inputs> <select> <out> <arms> <topo> <rung> <par> [extra env k=v ...]
  local label=$1 in=$2 sel=$3 out=$4 arms=$5 topo=$6 rung=$7 par=$8; shift 8
  ( env "$@" R1A_ARMS=$arms R1A_TOPOS=$topo R1A_WINDOWS=g0 WF1_RUNGS=$rung SELECT=$sel \
      python3 scripts_cosim/fresh_topo_burst_v1_gate.py r1a --inputs $in --out $out --selection $sel --parallel $par --mem 8G --timeout $LIMIT --no-scope 2>&1 | grep -v '^\[skip' > $O/smoke_$label.log; echo "$label rc=${PIPESTATUS[0]}" ) &
}
run gate    $O/inputs_G8/inputs $O/selected.json $GATE "ra_gnn_eng_ovg,cd_exactS" 19206 heavy 2 KEEP_RAW_DIR=$O/gate_raw OVERCAP_LOG_DIR=$O/gate_log
run dry     $O/inputs_G8/inputs $O/selected.json $SM   "ra_gnn_eng_ovd,ra_gnn_eng_ovo" 19206 heavy 2 KEEP_RAW_DIR=$O/smoke_raw OVERCAP_LOG_DIR=$O/smoke_log
run ident   $B/gate_inputs_G8/inputs $B/selected_gate.json $ID "cd_exactS" 16900 moderate 1
wait
echo "== 5. summaries"; ls $GATE/*.summary.json $SM/*.summary.json $ID/*.summary.json 2>&1; ls $GATE/*.failed.json $SM/*.failed.json $ID/*.failed.json 2>/dev/null || true
echo "prep done"
