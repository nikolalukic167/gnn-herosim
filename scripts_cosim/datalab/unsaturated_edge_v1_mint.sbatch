#!/bin/bash
# unsaturated_edge_v1 -- mint the study's environments. docs/lineages/unsaturated_edge_v1.md.
#
# unsaturated_scale_v2 showed that a 4-environment read on this apparatus inflates every effect
# 4-10x, because all four environments were the same, burstiest arrival window. The standing
# -20.8 % headline (gnnedge0 vs reactive, 6 servers, 40 clients) is exactly such a read. This
# lineage gives the 6-server CLIENT rungs the v2 treatment: 16 (topology, window) environments.
#
# This mints 8 more 6-server topologies (seeds 9101-9108) at 40 and 80 clients, giving 12
# candidates per rung with the existing cc40/cc80 s9001/9002/9003/9005. Seed 9004 is NOT used --
# it hangs on every policy (peer_only_v1). The four arrival windows already exist: w0 is
# drainable_f4000_n50000 (the 6-server rung's own workload) and w1-w3 were minted by
# unsaturated_scale_v2 rate-matched to w0 at 0.4604 arrivals/s, so they carry over unchanged.
#
# Nothing here is selected: the saturation screen picks the study's 4 topologies per rung by a
# rule registered before it runs.
#
#   sbatch scripts_cosim/datalab/unsaturated_edge_v1_mint.sbatch
#
#SBATCH --job-name=ue1-mint
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/ue1-mint-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/ue1-mint-%j.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
echo "[CODE] $(git describe --dirty --always) on $(git rev-parse --abbrev-ref HEAD)"

GATE=simulation_data/peer_affinity_live_gate
BASE="$GATE/configs/cs6s9001.json"          # the 6-server base peer_only_v1 B7 minted from
MANIFEST=simulation_data/unsaturated_edge_v1/cells.json
mkdir -p simulation_data/unsaturated_edge_v1
[[ -f "$BASE" ]] || { echo "FAIL LOUD: base cell missing: $BASE" >&2; exit 1; }

# --- 1. topologies ------------------------------------------------------------------------
NEW_SEEDS=9101,9102,9103,9104,9105,9106,9107,9108
if [[ -f "$MANIFEST" ]]; then
  echo "[mint] $MANIFEST exists, not re-minting"
else
  python3 scripts_cosim/cluster_scale_v1_cells.py --base "$BASE" --sim-inputs data/nofs-ids \
    --vary clients --rungs 40,80 --seeds "$NEW_SEEDS" --out-dir "$GATE/configs" \
    --manifest "$MANIFEST.partial" --prefix cc
  mv "$MANIFEST.partial" "$MANIFEST"
fi

# Every candidate cell must exist, declare 6 servers and the rung's client count, and hold
# every policy constant of the base.
python3 - "$GATE/configs" "$BASE" <<'PY'
import json, sys
cfgdir, base = sys.argv[1], json.load(open(sys.argv[2]))
seeds = [9001, 9002, 9003, 9005, 9101, 9102, 9103, 9104, 9105, 9106, 9107, 9108]
for nc in (40, 80):
    for s in seeds:
        p = f"{cfgdir}/cc{nc}s{s}.json"
        cfg = json.load(open(p))
        if int(cfg["nodes"]["server_nodes"]["count"]) != int(base["nodes"]["server_nodes"]["count"]):
            raise SystemExit(f"FAIL LOUD: {p} moved the server count -- this axis is CLIENTS")
        if int(cfg["nodes"]["client_nodes"]["count"]) != nc:
            raise SystemExit(f"FAIL LOUD: {p} declares {cfg['nodes']['client_nodes']['count']} clients, rung is {nc}")
        if float(cfg["scheduler"]["batch_timeout"]) != 16.0:
            raise SystemExit(f"FAIL LOUD: {p} batch_timeout moved; every policy constant is held fixed")
print(f"[cells] 2 rungs x {len(seeds)} candidate topologies at 6 servers, batch_timeout 16.0")
PY

# --- 2. arrival windows: already minted by unsaturated_scale_v2, verified not rebuilt ---------
python3 - "$GATE/workloads" <<'PY'
import json, sys
wd = sys.argv[1]
base = json.load(open(f"{wd}/drainable_f4000_n50000.json"))
target = float(base["arrival_rescale"]["rate_after_per_s"])
for w in (0, 1, 2, 3):
    p = f"{wd}/drainable_f4000_n50000.json" if w == 0 else f"{wd}/drainable_w{w}_n50000.json"
    d = json.load(open(p)); r = d["arrival_rescale"]; t = r["truncation"]
    if len(d["events"]) != 50000:
        raise SystemExit(f"FAIL LOUD: {p} has {len(d['events'])} events, the ladder is n50000")
    if t.get("skip_events", 0) != w * 100000:
        raise SystemExit(f"FAIL LOUD: {p} skip_events={t.get('skip_events', 0)}, expected {w*100000}")
    rate = float(r["rate_after_per_s"])
    if abs(rate / target - 1.0) > 0.005:
        raise SystemExit(f"FAIL LOUD: {p} delivers {rate:.4f}/s, w0 {target:.4f}/s -- a LOAD axis")
    print(f"[trace] w{w}: factor {float(r['factor']):.1f}, {rate:.4f} arrivals/s, "
          f"span {r['span_after_s']/3600:.2f} h")
print("[trace] 4 windows carried over from unsaturated_scale_v2, rate matched to w0")
PY
echo "=== mint complete ==="
