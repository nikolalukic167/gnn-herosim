#!/bin/bash
#SBATCH --job-name=kpa-trace
#SBATCH --partition=CPU-amd
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --exclude=os-cpu-slurm-21,os-cpu-slurm-15
#SBATCH --output=/share/nikola.lukic/uncut_groups/kpa_conf/kpa-trace-%x-%j.out
# KPA conformance (ii): re-run ONE filed G8 bar cell (cc40s19537 / g0<RUNG> / cd_ib0_inflightfull_k100 s0) at its pin a459dd98 with the
# physics-audit trace on, assert it equals the filed summary, keep only the kpa (+ header) rows. Body = r1a_gate.sbatch at a459dd98 with one
# added export (HEROSIM_AUDIT_TRACE) after its env wipe. RUNG=moderate|heavy via --export.
set -euo pipefail
RUNG=${RUNG:?}; U=/share/nikola.lukic/uncut_groups; K=$U/kpa_conf
WT=$HOME/gnn-herosim-wt/s6_g8proof_a459dd98; EXPECT_HEAD=a459dd98; IN=$U/comb/in_comball; OUT=$K/cells_$RUNG; SELECT=$U/arm1_selected.json
FILED=$U/comb/cells_bar_g8/cc40s19537__g0${RUNG}__cd_ib0_inflightfull_k100_s0.summary.json
[ -e $OUT ] && { echo "FAIL LOUD: $OUT exists"; exit 2; }
[ -e $K/trace_g0$RUNG.kpa.jsonl ] && { echo "FAIL LOUD: trace exists"; exit 2; }
cd "$WT"
for v in $(compgen -e | grep -E '^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|NEAR_RTT_|PARTIAL_STATE|WF1_)'); do unset "$v"; done
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT" PYTHONUNBUFFERED=1
export HEROSIM_RAW_DIR=/tmp/herosim_raw_${SLURM_JOB_ID:-$$}; mkdir -p "$HEROSIM_RAW_DIR"; trap 'rm -rf "$HEROSIM_RAW_DIR"' EXIT
HEAD=$(git rev-parse HEAD); echo "HEAD $HEAD node $(hostname) job=${SLURM_JOB_ID:-}"
[[ "$HEAD" == "$EXPECT_HEAD"* ]] || { echo "FAIL LOUD: worktree HEAD"; exit 1; }
[[ -z "$(git status --porcelain -- src scripts_cosim experiments)" ]] || { echo "FAIL LOUD: worktree dirty"; exit 1; }
mkdir -p "$OUT"
export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa GATE_FIXED_POLICY_TIME_SCALE=1.0 HEROSIM_SHARED_AUTOSCALER=0
export WF1_RUNGS=$RUNG R1A_TOPOS=19537 R1A_WINDOWS=g0 R1A_ARMS=cd_ib0_inflightfull_k100 R1A_SEEDS=1,2 R1A_SPLIT=uncut_arm1_split.json R1A_SLATE=declared_pruning_v1_uncapped
export HEROSIM_AUDIT_TRACE=$HEROSIM_RAW_DIR/trace.jsonl
python3 scripts_cosim/fresh_topo_burst_v1_gate.py r1a --inputs "$IN" --out "$OUT" --selection "$SELECT" --parallel 1 --mem 16G \
  --timeout 14400 --no-scope | grep -v "^\[skip"
NEW=$OUT/cc40s19537__g0${RUNG}__cd_ib0_inflightfull_k100_s0.summary.json
[ -s "$NEW" ] && [ -s $HEROSIM_AUDIT_TRACE ] || { echo "FAIL LOUD: no summary or no trace"; ls -la $OUT $HEROSIM_RAW_DIR; exit 1; }
python3 - "$FILED" "$NEW" "$HEROSIM_AUDIT_TRACE" "$K/trace_g0$RUNG.kpa.jsonl" <<'PY'
import json, sys, hashlib, os
filed, new = (json.load(open(p)) for p in sys.argv[1:3])
for k in ("total_rtt", "endTime", "num_tasks", "averageElapsedTime"):
    assert filed[k] == new[k], f"FAIL LOUD: {k} filed {filed[k]!r} re-run {new[k]!r}"
d = os.path.dirname(sys.argv[2]); base = os.path.basename(sys.argv[2])[:-len(".summary.json")]
fd = os.path.dirname(sys.argv[1])
for ext in (".tasks.npy", ".tasks_bsz.npy", ".tasks_lock.npy"):
    a, b = (hashlib.md5(open(os.path.join(x, base + ext), "rb").read()).hexdigest() for x in (fd, d))
    assert a == b, f"FAIL LOUD: {ext} differs"
print("[identity] total_rtt, endTime, num_tasks, averageElapsedTime and the three per-task arrays equal the filed cell")
n = {"header": 0, "kpa": 0}
with open(sys.argv[3]) as fh, open(sys.argv[4] + ".partial", "w") as out:
    for line in fh:
        k = json.loads(line)["k"]
        if k in n:
            out.write(line); n[k] += 1
assert n["header"] == 1 and n["kpa"] > 0, n
os.replace(sys.argv[4] + ".partial", sys.argv[4])
print("[trace] rows", n)
PY
echo "[trace] md5 $(md5sum < $K/trace_g0$RUNG.kpa.jsonl | cut -c1-32)  full-trace md5 $(md5sum < $HEROSIM_AUDIT_TRACE | cut -c1-32)"
