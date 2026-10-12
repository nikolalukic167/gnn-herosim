#!/bin/bash
# scale_160_v1 live gate, inputs (docs/lineages/scale_160_v1.md): the 12 fresh test topologies at 160 clients x 24 servers, p = 0.6,
# at the moderate and heavy rungs, as WF1 (W2 payloads + W3 access classes + W4 mix/repair, 1 s window), single origin spread over the
# 160 clients (scale_probe_remap_clients.py, disclosed), windows g0-g3. Nothing is built until the calibrated multipliers are given.
#   M_MODERATE, M_HEAVY   final arrival-rate multipliers over the x1 window (the window timestamps are divided by them, as r1a_gate_inputs:
#                         factor = 1 / m). They are NOT the 40 x 6 multipliers times servers/6: the calibration (S5) signs these numbers.
#   TOPOS                 space-separated seeds; default the selection scripts_cosim/scale_160_v1_selected.json (16001-16012)
#   sbatch --export=ALL,WT=<pinned worktree>,EXPECT_HEAD=<sha>,ROOT=<dir>,M_MODERATE=<m>,M_HEAVY=<m> s160_gate_inputs.sbatch
#SBATCH --job-name=s160-in
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/s160-in-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
set -euo pipefail
WT=${WT:?FAIL LOUD: set WT (pinned detached worktree)}
EXPECT_HEAD=${EXPECT_HEAD:?FAIL LOUD: set EXPECT_HEAD}
ROOT=${ROOT:?FAIL LOUD: set ROOT}
M_MODERATE=${M_MODERATE:?FAIL LOUD: set M_MODERATE (the signed moderate multiplier for 160 x 24)}
M_HEAVY=${M_HEAVY:?FAIL LOUD: set M_HEAVY (the signed heavy multiplier for 160 x 24)}
NC=${NC:-160}; NS=${NS:-24}; P=${P:-0.6}
D=/home/nikola.lukic/gnn-herosim/simulation_data
X1=$D/small_batch_confirm_v1/inputs/grounded/wl
BASECFG=${BASECFG:-$D/workload_fix_v1/corpus_prod_prov/cfg_src/cc40s9101.json}  # accel_replica_v1: S7's fastest_compatible cell cfg
SELECTION=${SELECTION:-scripts_cosim/scale_160_v1_selected.json}
REPLICA_RULE=${REPLICA_RULE:-}  # accel_replica_v1: every minted cfg must carry preinit.replica_placement_rule = this
cd "$WT"
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT"
HEAD=$(git rev-parse HEAD)
echo "HEAD $HEAD node $(hostname) ROOT=$ROOT"
[[ "$HEAD" == "$EXPECT_HEAD"* ]] || { echo "FAIL LOUD: worktree HEAD $HEAD is not $EXPECT_HEAD"; exit 1; }
rc=0; timeout 300 git diff --quiet HEAD -- src scripts_cosim experiments run_experiment.py || rc=$?
[[ $rc == 0 ]] || echo "WARN: worktree check rc=$rc (1 = tracked files differ from HEAD, 124 = timed out after 300 s)"
[[ ! -e "$ROOT" ]] || { echo "FAIL LOUD: $ROOT exists"; exit 1; }
TOPOS=${TOPOS:-$(python3 -c "import json;print(*json.load(open('$SELECTION'))['topologies'])")}
export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa GATE_FIXED_POLICY_TIME_SCALE=1.0 HEROSIM_SHARED_AUTOSCALER=0
mkdir -p "$ROOT"
for t in $TOPOS; do python3 scripts_cosim/scale_probe_cfg.py "$BASECFG" "$ROOT/cfg_src" --clients "$NC" --servers "$NS" --p "$P" --seed "$t" >/dev/null; done
TAGS=""
for spec in moderate=$M_MODERATE heavy=$M_HEAVY; do
  tag=${spec%%=*}; m=${spec##*=}
  python3 scripts_cosim/workload_fix_v1_build.py --grounded-wl "$X1" --cfg-dir "$ROOT/cfg_src" --topologies $TOPOS \
    --rung "$tag=$(python3 -c "print(repr(1.0/$m))")" --payload-sampler wf1_v1 --task-mix wf1_v1 --batch-timeout-fixed 1.0 --out "$ROOT/_b_$tag"
  mkdir -p "$ROOT/inputs_w24"; mv "$ROOT/_b_$tag/wf1_$tag" "$ROOT/inputs_w24/wf1_$tag"; cp "$ROOT/_b_$tag/manifest.json" "$ROOT/inputs_w24/manifest_$tag.json"; rm -rf "$ROOT/_b_$tag"
  TAGS="$TAGS $tag"
done
python3 scripts_cosim/workload_fix_v1_w3_build.py --w2-inputs "$ROOT/inputs_w24" --tags $TAGS --out "$ROOT/inputs" --diff-out "$ROOT/w3_vs_w2_diff.json"
for tag in $TAGS; do for W in g0 g1 g2 g3; do
  f="$ROOT/inputs/wf1_$tag/wl/grounded_${W}_n50000.json"
  python3 scripts_cosim/scale_probe_remap_clients.py "$f" "$f.tmp" --clients "$NC" && mv "$f.tmp" "$f"
done; done
REPLICA_RULE="$REPLICA_RULE" python3 - "$ROOT/inputs" "$NC" "$NS" "$P" $TOPOS <<'PY'
import json, os, sys
inp, nc, ns, p, topos = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), sys.argv[5:]
rule = os.environ.get("REPLICA_RULE") or None
for tag in ("moderate", "heavy"):
    for t in topos:
        c = json.load(open(f"{inp}/wf1_{tag}/cfg/cc40s{t}.json"))
        got = (c["nodes"]["client_nodes"]["count"], c["nodes"]["server_nodes"]["count"], c["network"]["topology"]["connection_probability"], c["network"]["topology"]["seed"])
        assert got == (nc, ns, p, int(t)), f"{tag} {t}: {got}"
        assert (c.get("preinit") or {}).get("replica_placement_rule") == rule, f"{tag} {t}: replica_placement_rule {(c.get('preinit') or {}).get('replica_placement_rule')!r}, want {rule!r}"
print(f"[check] {len(topos)} topologies x moderate/heavy: {nc} clients, {ns} servers, p={p}, topology seed = id, replica rule {rule}")
PY
echo "inputs ready: $ROOT/inputs (rungs moderate, heavy)"
