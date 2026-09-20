#!/bin/bash
# joint_burst_v2 -- train gnnedge0 and its MP-OFF twin UNCAPPED on the v2 burst corpus, 16 seeds each.
# docs/lineages/joint_burst_v2.md. The bipartite_edge_v1 train recipe verbatim (cache, split,
# checkpoint names swapped). 32 tasks; T = arm_index*16 + seed-1.
#
#   sbatch --array=0-31 scripts_cosim/datalab/joint_burst_v2_train.sbatch
#
#SBATCH --job-name=jb2-train
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/jb2-train-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/jb2-train-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-31

set -euo pipefail
ulimit -n 65536 2>/dev/null || true
cd /home/nikola.lukic/gnn-herosim
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1
export WANDB_ENTITY=nikolalukic167-tu-wien
export OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 OPENBLAS_NUM_THREADS=32 CUDA_VISIBLE_DEVICES=""
python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__,'cuda',torch.cuda.is_available())"
echo "[CODE] $(git describe --dirty --always) on $(git rev-parse --abbrev-ref HEAD)"
ARMS=(gnnedge0 mpoff)
T=${SLURM_ARRAY_TASK_ID}
ARM=${ARMS[$(( T / 16 ))]}; SEED=$(( T % 16 + 1 ))
CFG=experiments/joint_burst_v2_${ARM}.yaml
CACHE=simulation_data/graphs_cache_joint_burst_v2_psv3_inf
SPLIT=experiments/joint_burst_v2_split.json
CK="models/joint-burst-v2-${ARM}-lr2e3-seed${SEED}.pt"
echo "[TASK $T] arm=$ARM seed=$SEED cfg=$CFG -> $CK"
[[ -f "$CFG" ]] || { echo "FAIL LOUD: $CFG missing (git pull?)" >&2; exit 1; }
[[ -f "$CACHE/metadata.json" && -f "$CACHE/valid_combos_near_rtt_capped.pkl" && -f "$CACHE/rtt_chunk_0.pkl" ]] || { echo "FAIL LOUD: $CACHE incomplete" >&2; exit 1; }
[[ -f "$SPLIT" ]] || { echo "FAIL LOUD: split artifact $SPLIT missing" >&2; exit 1; }
python3 - "$CACHE" "$ARM" <<'EOF2'
import json, pickle, sys
m = json.load(open(sys.argv[1] + "/metadata.json"))
for k, v in (("partial_state_contract", "partial_state_v3"), ("dag_primary_alpha_key", "inf"), ("peer_exchange_block", True), ("platform_feature_dim", 14)):
    if m.get(k) != v:
        raise SystemExit(f"FAIL LOUD: cache metadata {k}={m.get(k)!r}, this recipe requires {v!r}")
got = m.get("label_objective") or "rtt"
if not got.startswith("rtt_drift:1") or "clock=measured" not in got:
    raise SystemExit(f"FAIL LOUD: cache label_objective={got!r}")
if sys.argv[2] == "gnnedge0":
    g = pickle.load(open(sys.argv[1] + "/graphs.pkl", "rb"))[0]
    ea = getattr(g, "edge_attr", None)
    if ea is None or ea.numel() == 0 or int(ea.shape[1]) != 5 or int(ea.shape[0]) != int(g.edge_index.shape[1]):
        raise SystemExit("FAIL LOUD: cache edge_attr is not the 5-column bipartite block the conv needs")
print("[cache] contract ok, label =", got, "| datasets =", m.get("num_datasets"))
EOF2
mkdir -p logs models
if [[ -f "$CK" ]]; then echo "[skip] $CK exists"; exit 0; fi
python3 -u run_experiment.py "$CFG" --seed "$SEED"
RUN_NAME=$(python3 -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['wandb']['run_name'])" "$CFG")
PRODUCED="models/${RUN_NAME}-seed${SEED}.pt"
if [[ -f "$PRODUCED" && ! -f "$CK" ]]; then
  mv "$PRODUCED" "$CK"
  [[ -f "${PRODUCED%.pt}.contract.json" ]] && mv "${PRODUCED%.pt}.contract.json" "${CK%.pt}.contract.json"
  [[ -f "${PRODUCED%.pt}-final.pt" ]] && mv "${PRODUCED%.pt}-final.pt" "${CK%.pt}-final.pt"
  [[ -f "${PRODUCED%.pt}-final.contract.json" ]] && mv "${PRODUCED%.pt}-final.contract.json" "${CK%.pt}-final.contract.json"
fi
[[ -f "$CK" ]] || { echo "FAIL LOUD: $CFG seed $SEED produced no $CK (looked for $PRODUCED)" >&2; exit 1; }
SC="${CK%.pt}.contract.json"
[[ -f "$SC" ]] || { echo "FAIL LOUD: no sidecar $SC" >&2; exit 1; }
python3 - "$SC" "$ARM" "$SPLIT" "$CK" <<'EOF2'
import hashlib, json, sys
import torch
sc, arm, split, ck = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3], sys.argv[4]
checks = {"disable_message_passing": arm == "mpoff", "mp_peer_edges": True, "mp_residual": False,
          "partial_state_contract": "partial_state_v3", "partial_state_feature_dim": 22,
          "dag_alpha_key": "inf", "decode_replica_reuse": True, "decode_relax_on_stuck": True, "peer_mass": True}
if arm == "gnnedge0":
    checks.update(mp_bipartite_edge_conv=True, mp_bipartite_edge_attr_zero=True, mp_platform_edges=True)
for k, want in checks.items():
    if sc.get(k) != want:
        raise SystemExit(f"FAIL LOUD: sidecar {k}={sc.get(k)!r}, arm {arm} expects {want!r}")
es = sc.get("early_stopping") or {}
if (es.get("patience"), es.get("min_epochs")) != (60, 100):
    raise SystemExit(f"FAIL LOUD: sidecar early_stopping={es!r}")
sha = hashlib.sha256(open(split, "rb").read()).hexdigest()
if (sc.get("split_artifact") or {}).get("sha256") != sha:
    raise SystemExit("FAIL LOUD: sidecar split sha != the corpus split")
if arm == "gnnedge0":
    sd = torch.load(ck, map_location="cpu")
    if not any(k.startswith("bip_convs.") for k in sd):
        raise SystemExit("FAIL LOUD: no bip_convs.* weights in a gnnedge0 checkpoint")
print(f"[sidecar] ok: arm={arm}")
EOF2
echo "=== task $T done: $CK ==="
