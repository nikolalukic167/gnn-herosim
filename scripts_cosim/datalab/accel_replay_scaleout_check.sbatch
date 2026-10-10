#!/bin/bash
# accel_replica_v1: does the I11 replay APPLY the replica rule? A passing replay says nothing if no replica is created inside the label horizon, so this
# replays the SAME captured states (a CD capture on the calibrated cell) under both rules and compares the replicas the replay created and the platform
# mix it ended with. Truth is the continuing live run (no isolated runs), so only the replay is exercised.
#   sbatch --export=ALL,WT=..,CELLDIR=<calib heavy_root>,TOPO=9905,RUNG=m27p6227,N=40,OUT=<new dir> accel_replay_scaleout_check.sbatch
#SBATCH --job-name=accel-rs
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/accel-rs-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
set -euo pipefail
WT=${WT:?}; CELLDIR=${CELLDIR:?}; TOPO=${TOPO:?}; RUNG=${RUNG:?}; N=${N:-40}; OUT=${OUT:?}
cd "$WT"; eval "$(micromamba shell hook --shell bash)"; micromamba activate gnn
for v in $(compgen -e | grep -E '^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|DECLARED_SLATE)'); do unset "$v"; done
mkdir -p "$OUT/in_fast/grounded_$RUNG/cfg" "$OUT/in_first/grounded_$RUNG/cfg"
for k in fast first; do ln -sfn "$CELLDIR/inputs/wf1_$RUNG/wl" "$OUT/in_$k/grounded_$RUNG/wl"; done
python3 - "$CELLDIR/inputs/wf1_$RUNG/cfg/cc40s$TOPO.json" "$OUT" "$RUNG" "$TOPO" <<'PY'
import json, sys
src, out, rung, topo = sys.argv[1:5]
c = json.load(open(src)); assert c["preinit"]["replica_placement_rule"] == "fastest_compatible"
json.dump(c, open(f"{out}/in_fast/grounded_{rung}/cfg/cc40s{topo}.json", "w"))
c["preinit"].pop("replica_placement_rule"); json.dump(c, open(f"{out}/in_first/grounded_{rung}/cfg/cc40s{topo}.json", "w"))
PY
export HEROSIM_PY=python3 PYTHONPATH=$WT TS=1.0 HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa GATE_FIXED_POLICY_TIME_SCALE=1.0 HEROSIM_SHARED_AUTOSCALER=0
W=$OUT/work; mkdir -p "$W"
# capture once under the rule the cell was minted with
HEROSIM_AUDIT_INPUTS=$OUT/in_fast bash scripts_cosim/physics_audit/capture.sh $TOPO $RUNG 50000 "$W/${TOPO}_$RUNG"
for k in fast first; do
  C=$W/${TOPO}_$RUNG
  ( export GNN_DECODE_MODE=masked_topo GNN_BATCH_BY_PEER_GROUP=1 HEROSIM_PEER_EXCHANGE=1 HEROSIM_SERVER_ONLY_REPLICAS=1 HEROSIM_WARMTH_PHYSICS=node_disk_v2 PYTHONHASHSEED=0 SIM_FORCE_FULL_STATS=1 OMP_NUM_THREADS=1 GNN_CAPTURE_DATASET_STATE=0 HEROSIM_SNAPSHOT_FIDELITY=1 HEROSIM_POLICY_TIME_SCALE=1.0 I11_SCALE=1
    python3 scripts_cosim/physics_audit/i11_replay.py --cfg "$OUT/in_$k/grounded_$RUNG/cfg/cc40s$TOPO.json" --workload "$CELLDIR/inputs/wf1_$RUNG/wl/grounded_g0_n50000.json" \
      --trace $C/t.jsonl --snapshots $C/snap.jsonl --out $OUT/replay_$k.jsonl --n $N --tmin 360 --params live --truth continuing --cell ${TOPO}_$RUNG > $OUT/replay_$k.log 2>&1 )
done
python3 - "$OUT" <<'PY'
import json, sys
o = sys.argv[1]
a = [json.loads(l) for l in open(f"{o}/replay_fast.jsonl")]; b = [json.loads(l) for l in open(f"{o}/replay_first.jsonl")]
assert [r["batch"] for r in a] == [r["batch"] for r in b], "the two replays did not replay the same states"
n_up = sum(1 for r in a if r.get("replay_scale_ups")); n_up_b = sum(1 for r in b if r.get("replay_scale_ups"))
diff_end = [r["t"] for r, q in zip(a, b) if r.get("replay_replicas_end") != q.get("replay_replicas_end")]
diff_val = [r["t"] for r, q in zip(a, b) if r.get("replay") is not None and q.get("replay") is not None and abs(r["replay"] - q["replay"]) > 1e-9]
print(f"states {len(a)}; with a replica created inside the replay horizon: fastest {n_up}, first {n_up_b}")
print(f"states whose end-of-replay replica platform mix differs between the rules: {len(diff_end)}; whose replayed latency differs: {len(diff_val)}")
for r, q in zip(a, b):
    if r.get("replay_replicas_end") != q.get("replay_replicas_end"):
        print(f"  t={r['t']:.1f} fastest {json.dumps(r['replay_replicas_end'])[:260]}\n           first   {json.dumps(q['replay_replicas_end'])[:260]}"); break
PY
