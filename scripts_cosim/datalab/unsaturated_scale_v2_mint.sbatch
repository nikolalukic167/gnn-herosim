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
#
# THE FACTOR IS PER WINDOW, and that is not a detail. Measured 2026-09-19: at a common factor
# of 4000 the windows deliver 0.460 / 0.747 / 0.753 / 0.757 arrivals/s -- window 0, the only
# window every gate in this programme has ever used, is 1.6x SLOWER than the rest of the trace.
# Holding the FACTOR fixed would therefore make the window axis a load axis in disguise, and at
# 0.75/s reactive on 80 servers sits between its 0.66 share at 0.46/s and its 0.955 at 0.92/s --
# i.e. the comparison would confound arrival pattern with saturation, which is the exact defect
# unsaturated_scale_v1 exists to have fixed. Each window instead gets the factor that matches
# w0's REALISED rate, so all four deliver 50,000 tasks at 0.460/s over 30.2 h and the only
# moving part is the arrival PATTERN.
python3 - "$SRC" "$GATE/workloads" <<'PY'
import json, subprocess, sys
src, wd = sys.argv[1], sys.argv[2]
w0 = json.load(open(f"{wd}/drainable_f4000_n50000.json"))
target = float(w0["arrival_rescale"]["rate_after_per_s"])
print(f"[trace] target rate = w0's realised {target:.6f} arrivals/s")
ev = json.load(open(src))["events"]
for w in (1, 2, 3):
    skip, n = w * 100000, 50000
    native = ev[skip + n - 1]["timestamp"] - ev[skip]["timestamp"]
    factor = n / (target * native)
    out = f"{wd}/drainable_w{w}_n50000.json"
    print(f"[trace] w{w}: native span {native:.3f} s -> factor {factor:.2f}")
    subprocess.run([sys.executable, "scripts_cosim/rescale_workload_arrivals.py",
                    "--workload", src, "--factor", f"{factor:.6f}", "--max-events", str(n),
                    "--skip-events", str(skip), "--group-size", "10",
                    "--output", out + ".partial"], check=True)
    subprocess.run(["mv", out + ".partial", out], check=True)
PY

python3 - "$GATE/workloads" <<'PY'
import json, sys
wd = sys.argv[1]
base = json.load(open(f"{wd}/drainable_f4000_n50000.json"))
target = float(base["arrival_rescale"]["rate_after_per_s"])
first0 = base["events"][0]["timestamp"]
sigs = []
for w in (0, 1, 2, 3):
    p = f"{wd}/drainable_f4000_n50000.json" if w == 0 else f"{wd}/drainable_w{w}_n50000.json"
    d = json.load(open(p)); r = d["arrival_rescale"]; t = r["truncation"]
    if len(d["events"]) != 50000 or t["max_events"] != 50000:
        raise SystemExit(f"FAIL LOUD: {p} has {len(d['events'])} events, the ladder is n50000")
    if t.get("skip_events", 0) != w * 100000:
        raise SystemExit(f"FAIL LOUD: {p} skip_events={t.get('skip_events', 0)}, expected {w*100000}")
    # THE bar for this design: every window delivers the SAME arrival RATE, so the window axis
    # is arrival PATTERN and not load. 0.5 % is well inside the band that separates this rung
    # (0.46/s, share 0.66) from the next one up (0.92/s, share 0.955).
    rate = float(r["rate_after_per_s"])
    if abs(rate / target - 1.0) > 0.005:
        raise SystemExit(f"FAIL LOUD: {p} delivers {rate:.4f} arrivals/s, w0 delivers "
                         f"{target:.4f}; the window axis would be a LOAD axis")
    # No window may idle through the skipped span before its first arrival. The windows are
    # rebased in NATIVE time and then scaled, and the factors differ by ~1.6x, so their scaled
    # starts differ by ~18 s -- 0.02 % of a 108,600 s span, which is what this bar allows. An
    # un-rebased window would start ~100,000 native seconds in and fail by five orders.
    t0, span = float(d["events"][0]["timestamp"]), float(r["span_after_s"])
    if t0 > 0.001 * span:
        raise SystemExit(f"FAIL LOUD: {p} starts at {t0:.1f} s of a {span:.0f} s span "
                         f"({100*t0/span:.3f} %); it was not rebased and every policy would "
                         "idle through the skipped span")
    # ...and every peer pair must land inside the window.
    if d["peer_exchange"] and max(max(p2[0], p2[1]) for p2 in d["peer_exchange"]) >= 50000:
        raise SystemExit(f"FAIL LOUD: {p} has a peer pair reaching past the window")
    sigs.append(tuple(e["application"]["demand_scale"][next(iter(e["application"]["dag"]))]
                      for e in d["events"][:50]))
    print(f"[trace] w{w}: {len(d['events'])} events, factor {float(r['factor']):.1f}, "
          f"{rate:.4f} arrivals/s, span {r['span_after_s']/3600:.2f} h, "
          f"peer pairs {t['peer_pairs_after']}")
# The windows must be DIFFERENT arrival data, not four copies of one window.
if len(set(sigs)) != 4:
    raise SystemExit("FAIL LOUD: two windows carry identical leading events -- not disjoint")
print("[trace] 4 windows: disjoint arrival PATTERNS, common start, rate matched to w0")
PY
echo "=== mint complete ==="
