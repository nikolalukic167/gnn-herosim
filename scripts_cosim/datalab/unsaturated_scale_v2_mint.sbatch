#!/bin/bash
# unsaturated_scale_v2 -- mint the study's environments. docs/lineages/unsaturated_scale_v2.md.
#
# An ENVIRONMENT is a (topology, arrival window) pair. v1 had four of them -- four topologies on
# ONE arrival window -- and that is the measured weakness: the checkpoint statistic's sd scales
# as exactly 1/sqrt(cells), so with four it is 14-29 pp against a 5-8 pp effect, and the median
# result swings 19 pp between the two halves of the single window.
#
# This mints:
#   * 8 more 80-server topologies (seeds 9101-9108), giving 12 candidates with the existing
#     9001/9002/9003/9005. Seed 9004 is NOT used -- it hangs on every policy (peer_only_v1).
#   * 3 more arrival windows out of the SAME production trace: events [100k,150k), [200k,250k),
#     [300k,350k) of 450,729. Real arrival data, disjoint from the window every gate in this
#     programme has used, at the same factor 4000 and the same 50,000 events.
#
# Nothing here is selected: the saturation screen picks the study's 4 topologies by a rule
# registered before it runs. Minting more than the study uses is the point -- it is what makes
# the selection a screen rather than a choice.
#
#   sbatch scripts_cosim/datalab/unsaturated_scale_v2_mint.sbatch
#
#SBATCH --job-name=us2-mint
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/us2-mint-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/us2-mint-%j.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00

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
BASE="$GATE/configs/cell_s9001_f4000_pg16.json"
MANIFEST=simulation_data/unsaturated_scale_v2/cells.json
SRC="$GATE/workloads/workload-150-150-peer_p2_x200.json"
mkdir -p simulation_data/unsaturated_scale_v2
[[ -f "$BASE" ]] || { echo "FAIL LOUD: base cell missing: $BASE" >&2; exit 1; }
[[ -f "$SRC" ]]  || { echo "FAIL LOUD: source trace missing: $SRC" >&2; exit 1; }

# --- 1. topologies ------------------------------------------------------------------------
NEW_SEEDS=9101,9102,9103,9104,9105,9106,9107,9108
if [[ -f "$MANIFEST" ]]; then
  echo "[mint] $MANIFEST exists, not re-minting"
else
  python3 scripts_cosim/cluster_scale_v1_cells.py --base "$BASE" --sim-inputs data/nofs-ids \
    --rungs 80 --seeds "$NEW_SEEDS" --out-dir "$GATE/configs" \
    --manifest "$MANIFEST.partial" --prefix cs
  mv "$MANIFEST.partial" "$MANIFEST"
fi

# Every candidate topology must exist and declare 80 servers with the policy constants held.
python3 - "$GATE/configs" <<'PY'
import json, sys
cfgdir = sys.argv[1]
seeds = [9001, 9002, 9003, 9005, 9101, 9102, 9103, 9104, 9105, 9106, 9107, 9108]
for s in seeds:
    p = f"{cfgdir}/cs80s{s}.json"
    cfg = json.load(open(p))
    n = int(cfg["nodes"]["server_nodes"]["count"])
    bt = float(cfg["scheduler"]["batch_timeout"])
    if n != 80:
        raise SystemExit(f"FAIL LOUD: {p} declares {n} servers, the rung is 80")
    if bt != 16.0:
        raise SystemExit(f"FAIL LOUD: {p} batch_timeout={bt}; every policy constant is held fixed")
print(f"[cells] {len(seeds)} candidate topologies at 80 servers, batch_timeout 16.0")
PY

# --- 2. arrival windows -------------------------------------------------------------------
# w0 is the existing drainable_f4000_n50000.json (skip 0) and is NOT rebuilt -- runs already
# depend on it, and --skip-events 0 is byte-identical to how it was built (tested).
for W in 1 2 3; do
  SKIP=$(( W * 100000 ))
  OUT="$GATE/workloads/drainable_f4000_n50000_w${W}.json"
  if [[ -f "$OUT" ]]; then echo "[skip] exists: $OUT"; continue; fi
  python3 scripts_cosim/rescale_workload_arrivals.py --workload "$SRC" --factor 4000 \
    --max-events 50000 --skip-events "$SKIP" --group-size 10 --output "$OUT.partial"
  mv "$OUT.partial" "$OUT"
done

python3 - "$GATE/workloads" <<'PY'
import json, sys
wd = sys.argv[1]
base = json.load(open(f"{wd}/drainable_f4000_n50000.json"))
span0 = base["arrival_rescale"]["span_after_s"]
first0 = base["events"][0]["timestamp"]
seen = [{id(None)}]
sigs = []
for w in (0, 1, 2, 3):
    p = f"{wd}/drainable_f4000_n50000.json" if w == 0 else f"{wd}/drainable_f4000_n50000_w{w}.json"
    d = json.load(open(p)); r = d["arrival_rescale"]; t = r["truncation"]
    if len(d["events"]) != 50000 or t["max_events"] != 50000:
        raise SystemExit(f"FAIL LOUD: {p} has {len(d['events'])} events, the ladder is n50000")
    if float(r["factor"]) != 4000.0:
        raise SystemExit(f"FAIL LOUD: {p} declares factor {r['factor']}, expected 4000")
    if t.get("skip_events", 0) != w * 100000:
        raise SystemExit(f"FAIL LOUD: {p} skip_events={t.get('skip_events', 0)}, expected {w*100000}")
    # Every window must START where window 0 starts, or a policy idles through the skipped span.
    if abs(d["events"][0]["timestamp"] - first0) > 1e-6:
        raise SystemExit(f"FAIL LOUD: {p} starts at {d['events'][0]['timestamp']}, w0 starts at {first0}")
    # ...and every peer pair must land inside the window.
    if d["peer_exchange"] and max(max(p2[0], p2[1]) for p2 in d["peer_exchange"]) >= 50000:
        raise SystemExit(f"FAIL LOUD: {p} has a peer pair reaching past the window")
    sigs.append(tuple(e["application"]["demand_scale"][next(iter(e["application"]["dag"]))]
                      for e in d["events"][:50]))
    print(f"[trace] w{w}: {len(d['events'])} events, {r['rate_after_per_s']:.3f} arrivals/s, "
          f"span {r['span_after_s']:.0f} s ({r['span_after_s']/3600:.1f} h), "
          f"peer pairs {t['peer_pairs_after']}")
# The windows must be DIFFERENT arrival data, not four copies of one window.
if len(set(sigs)) != 4:
    raise SystemExit("FAIL LOUD: two windows carry identical leading events -- not disjoint")
print("[trace] 4 windows, disjoint, all rebased to a common start")
PY
echo "=== mint complete ==="
