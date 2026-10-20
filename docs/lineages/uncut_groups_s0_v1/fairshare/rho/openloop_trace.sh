#!/bin/bash
#SBATCH --job-name=s6-openloop-trace
#SBATCH --partition=CPU-amd
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --exclude=os-cpu-slurm-[21,15,13,14,16,10,17]
#SBATCH --output=/share/nikola.lukic/uncut_groups/anomaly/fairshare/logs/s6-openloop-trace-%j.out
# usage: sbatch --export=ALL,PIN=<sha> openloop_trace.sh
# SCIENTIST's open-loop monotonicity test (a): the filed heavy bar cell cc40s19537 / g0 (24 servers, switch off) re-run at PIN with the audit trace, asserted equal
# to the filed cell, keeping only its remote peer exchanges (time, task, src, dst, bytes, route, latency) in emission order.
set -euo pipefail
PIN=${PIN:?}; U=/share/nikola.lukic/uncut_groups; R=$U/anomaly/fairshare/openloop; mkdir -p $R $U/anomaly/fairshare/logs
OUTC=$R/cell; [ -e $OUTC ] && { echo "FAIL LOUD: exists"; exit 2; }
WTP=$HOME/gnn-herosim-wt/s6_fs_${PIN:0:8}; [[ $(git -C $WTP rev-parse HEAD) == $PIN* ]] || { echo "FAIL LOUD: worktree"; exit 2; }
sed 's#^python3 scripts_cosim/fresh_topo_burst_v1_gate.py r1a#export HEROSIM_AUDIT_TRACE="$TRACE_OUT"\n&#' $WTP/scripts_cosim/datalab/r1a_gate.sbatch > /tmp/ol_gate_$$.sbatch
grep -q 'HEROSIM_AUDIT_TRACE="$TRACE_OUT"' /tmp/ol_gate_$$.sbatch || { echo "FAIL LOUD: patch"; exit 2; }
eval "$(micromamba shell hook --shell bash)" && micromamba activate gnn
export TRACE_OUT=/tmp/ol_trace_${SLURM_JOB_ID:-$$}.jsonl; rm -f $TRACE_OUT
( export WT=$WTP EXPECT_HEAD=${PIN:0:8} IN=$U/comb/in_comball OUT=$OUTC SELECT=$U/arm1_selected.json R1A_ARMS=cd_ib0_inflightfull_k100 R1A_SEEDS=0 R1A_TOPOS=19537 R1A_WINDOWS=g0 WF1_RUNGS=heavy PAR=1 CELL_MEM=16G LIMIT=14400 R1A_SPLIT=uncut_arm1_split.json R1A_SLATE=declared_pruning_v1_uncapped
  unset R1A_EXCHANGE_TCP R1A_EXCHANGE_FAIRSHARE R1A_EXCHANGE_FAIRSHARE_PARTNERS
  bash /tmp/ol_gate_$$.sbatch ) > $R/log.txt 2>&1
N=cc40s19537__g0heavy__cd_ib0_inflightfull_k100_s0
python3 - $TRACE_OUT $OUTC/$N.summary.json $U/comb/cells_bar_g8/$N.summary.json $R/xfers.jsonl <<'PY'
import json, sys
tr, new_p, filed_p, out = sys.argv[1:5]
new, filed = json.load(open(new_p)), json.load(open(filed_p))
for k in ("endTime", "total_rtt", "num_tasks", "averageElapsedTime"):
    assert new[k] == filed[k], f"FAIL LOUD: {k}"
n = 0
with open(tr) as fh, open(out + ".partial", "w") as o:
    for line in fh:
        if '"xfer"' not in line:
            continue
        r = json.loads(line)
        if r.get("k") == "xfer" and r.get("kind") == "peer":
            o.write(json.dumps({"t": r["t"], "task": r["task"], "src": r["src"], "dst": r["dst"], "bytes": r["bytes"], "route": r["route"], "latency": r["latency"]}) + "\n"); n += 1
want = sum(v["transfers"] for v in new["peerExchangeByAccessClass"].values())
assert n == want, (n, want)
import os; os.replace(out + ".partial", out); print("[openloop] transfers", n, "identity ok")
PY
rm -f $TRACE_OUT; echo "[openloop] xfers md5 $(md5sum < $R/xfers.jsonl | cut -c1-32)"
