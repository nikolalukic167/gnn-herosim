#!/bin/bash
# workload_fix_v1 W2+W3 rerun on R1.1 (docs/lineages/workload_fix_v1.md, "Identity result and amendment R1.1-T"):
# the 19 test topologies x 4 windows x 5 classical arms x 2 provisional rungs, batching window fixed at 1 s (WB2), W2 payloads, W3 access classes (mix 40/40/20).
# Inputs: stage_w23_wb2/inputs (W2-WB2 inputs + access_classes; rungs lo = x0.2666, hi = x11.6139, batch_timeout 1 s); their window files are verified against the manifests before the first run.
#   WT and EXPECT_HEAD name the pinned detached worktree and its commit (no default; never move the worktree while it runs).
#   OUT defaults to simulation_data/workload_fix_v1/stage_w23_r11/gate and must not exist.
#SBATCH --job-name=wf1w23r11
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/wf1w23r11-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=24:00:00
set -euo pipefail
WT=${WT:?FAIL LOUD: set WT (pinned detached worktree)}
EXPECT_HEAD=${EXPECT_HEAD:?FAIL LOUD: set EXPECT_HEAD}
D=/home/nikola.lukic/gnn-herosim/simulation_data
W=$D/workload_fix_v1
IN=${WF1_IN:-$W/stage_w23_wb2/inputs}
SELECT=$D/small_batch_confirm_v1/inputs/selected.json
OUT=${OUT:-$W/stage_w23_r11/gate}
P=${PAR:-$(( ${SLURM_CPUS_PER_TASK:-64} - 2 ))}
RUNGS=lo,hi
EXPECTED=760   # 19 topologies x 2 rungs x 4 windows x 5 arms
cd "$WT"
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT"
export HEROSIM_RAW_DIR=/tmp/herosim_raw_${SLURM_JOB_ID:-$$}
mkdir -p "$HEROSIM_RAW_DIR"; trap 'rm -rf "$HEROSIM_RAW_DIR"' EXIT
python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"
HEAD=$(git rev-parse HEAD)
echo "HEAD $HEAD  node $(hostname)  OUT=$OUT"
[[ "$HEAD" == "$EXPECT_HEAD"* ]] || { echo "FAIL LOUD: worktree HEAD $HEAD is not $EXPECT_HEAD"; exit 1; }
[[ -z "$(git status --porcelain -- src scripts_cosim experiments run_experiment.py)" ]] || { echo "FAIL LOUD: worktree dirty"; exit 1; }
[[ -f src/placement/request_timeout.py ]] || { echo "FAIL LOUD: this worktree has no R1.1 request timeout"; exit 1; }
[[ ! -e "$OUT" ]] || { echo "FAIL LOUD: $OUT exists; a stage directory is not overwritten"; exit 1; }
# R1 environment (the driver refuses anything else); R1.1 is R1 + the code at EXPECT_HEAD
export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa GATE_FIXED_POLICY_TIME_SCALE=1.0
export HEROSIM_SHARED_AUTOSCALER=0 WF1_ARMS=

# pre-flight: inputs are the WB2 files, the cell list is the registered 760, no infeasible topology is selected
python3 - "$IN" "$SELECT" "$EXPECTED" <<'EOF'
import hashlib, json, os, sys
inp, sel, expected = sys.argv[1], sys.argv[2], int(sys.argv[3])
topos = json.load(open(sel))["topologies"]
assert len(topos) == 19 and not {9603, 9604, 9605, 9606} & set(topos), topos
for tag, mult in (("lo", 0.2666), ("hi", 11.6139)):
    m = json.load(open(f"{inp}/manifest_{tag}.json"))
    assert m["payload_sampler"] == "wf1_v1" and m["batch_timeout_fixed_s"] == 1.0 and m["topologies"] == topos, tag
    assert abs(m["rungs"][tag]["multiplier"] - mult) < 1e-9, (tag, m["rungs"][tag]["multiplier"])
    for name, want in m["rungs"][tag]["wl_sha256"].items():
        h = hashlib.sha256(open(f"{inp}/wf1_{tag}/wl/{name}", "rb").read()).hexdigest()
        assert h == want, f"{tag}/{name}: window file changed since it was built"
    for t in topos:
        c = json.load(open(f"{inp}/wf1_{tag}/cfg/cc40s{t}.json"))
        assert c["scheduler"]["batch_timeout"] == 1.0 and c["workload_fix_v1_batch_timeout"]["fixed_s"] == 1.0, (tag, t)
        assert c["network"]["backbone"]["access_classes"] == {"mix": {"wired": 0.4, "wifi": 0.4, "cellular": 0.2}}, (tag, t)
print(f"[preflight] inputs ok: 2 rungs x 4 windows x {len(topos)} topologies; expected runs {expected} = 19*2*4*5")
EOF

[[ "$OUT" == */stage_w23_r11/* || -n "${OUT_OVERRIDE_OK:-}" ]] || { echo "FAIL LOUD: OUT is outside stage_w23_r11"; exit 1; }
mkdir -p "$OUT"
set +e
WF1_RUNGS=$RUNGS python3 scripts_cosim/fresh_topo_burst_v1_gate.py wf1 --inputs "$IN" --out "$OUT" --selection "$SELECT" \
  --parallel "$P" --mem 4G --timeout "${TIMEOUT:-2700}" --no-scope | grep -v "^\[skip"
rc=${PIPESTATUS[0]}
# registered failure rule (transfer_physics_v1): one rerun of a failed run at 3x the timeout; a second failure stands
if ls "$OUT"/*.failed.json >/dev/null 2>&1; then
  echo "=== first pass: $(ls "$OUT"/*.failed.json | wc -l) failed; rerunning once at 3x timeout ==="
  mkdir -p "$OUT/first_pass_failed" && mv "$OUT"/*.failed.json "$OUT/first_pass_failed/"
  WF1_RUNGS=$RUNGS python3 scripts_cosim/fresh_topo_burst_v1_gate.py wf1 --inputs "$IN" --out "$OUT" --selection "$SELECT" \
    --parallel "$P" --mem 4G --timeout "$(( ${TIMEOUT:-2700} * 3 ))" --no-scope | grep -v "^\[skip"
  rc=${PIPESTATUS[0]}
fi
set -e
echo "=== w23_r11: $(ls "$OUT"/*.summary.json 2>/dev/null | wc -l) of $EXPECTED summaries, $(ls "$OUT"/*.failed.json 2>/dev/null | wc -l) failed ==="
echo "driver rc=$rc"

# request failures per arm and rung next to the latencies (R1.1: a timed-out task is a failure, reported, never dropped)
python3 - "$OUT" <<'EOF'
import glob, json, sys, collections
tot = collections.defaultdict(lambda: [0, 0, 0])
for f in glob.glob(sys.argv[1] + "/*.summary.json"):
    r = json.load(open(f))
    arm, win = r["arm"].split("__")[2].rsplit("_s", 1)[0], r["window"]
    key = (win[2:], arm)
    n = r.get("requestFailures") or 0
    n = n if isinstance(n, (int, float)) else len(n)
    tot[key][0] += 1; tot[key][1] += n; tot[key][2] += n > 0
print("rung arm: runs, request failures, runs with a failure")
for (rung, arm), (runs, fails, cells) in sorted(tot.items()):
    print(f"  {rung:3s} {arm:12s} {runs:4d} {fails:6d} {cells:4d}")
json.dump({f"{k[0]}/{k[1]}": v for k, v in tot.items()}, open(sys.argv[1] + "/request_failures.json", "w"), indent=1)
EOF

