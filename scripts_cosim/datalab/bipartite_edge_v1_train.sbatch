#!/bin/bash
# bipartite_edge_v1 -- training. docs/lineages/bipartite_edge_v1.md.
#
# Task table (the shell SPECS below is the single source of truth):
#   D1/D2  tasks  0-15   gnnedge   BipartiteEdgeConv on the real 5-column edge_attr
#          tasks 16-31   gnnedge0  the SAME conv with edge_attr zeroed -- the control
#
# Same corpus, split artifact, objective, schedule, learning rate and 16 seeds as
# peer_only_v1's `gnn`, so gnn / gnnres / gnnedge / gnnedge0 are four points differing in the
# bipartite stage and nothing else. Recipe inherited from partial_state_v3 (T1b, lr 2e-3 not
# re-tuned, CE-only, teacher-forced masked_topo, patience 60 / floor 100).
#
# Checkpoints: models/bipartite-edge-v1-1670-{arm}-lr2e3-seed{n}.pt
#
#   sbatch --array=0-31 scripts_cosim/datalab/bipartite_edge_v1_train.sbatch   # D1/D2
#   sbatch --array=32-47 scripts_cosim/datalab/bipartite_edge_v1_train.sbatch  # E2b seeds 17-32
#
# 32 tasks fits under MaxSubmit=50 in one submission; an array counts EVERY task against it.
#
#SBATCH --job-name=be-train
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/be-train-%A_%a.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/be-train-%A_%a.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --array=0-47

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
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=""
python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__,'cuda',torch.cuda.is_available())"
echo "[CODE] $(git describe --dirty --always) on $(git rev-parse --abbrev-ref HEAD)"

# bipartite_aggr_v1 AMENDMENT 1 (2026-09-18): seeds 17-32 of gnnedge0 at tasks 32-47, so
# E2 can be re-read at n = 32. APPENDED -- tasks 0-31 keep their arms and their checkpoints.
SPECS=("gnnedge 1 16" "gnnedge0 1 16" "gnnedge0 17 32")
ARMS=(); SEEDS=()
# Spec format is "arm FIRSTSEED LASTSEED" so a later block can add seeds to an arm that
# already has some, WITHOUT moving any existing task index (appending a bare count could not).
for spec in "${SPECS[@]}"; do
  read -r arm s0 s1 <<< "$spec"
  for s in $(seq "$s0" "$s1"); do ARMS+=("$arm"); SEEDS+=("$s"); done
done
NTASKS=${#ARMS[@]}
T=${SLURM_ARRAY_TASK_ID}
if (( T >= NTASKS )); then echo "[TASK $T] beyond the $NTASKS declared arms"; exit 0; fi
ARM=${ARMS[$T]}; SEED=${SEEDS[$T]}

CFG=experiments/bipartite_edge_v1_1670_${ARM}.yaml
CACHE=simulation_data/graphs_cache_peer_only_v1_1670_psv3
SPLIT=experiments/peer_only_v1_1670_split.json
GUARD=simulation_data/peer_only_v1/b0.json
CK="models/bipartite-edge-v1-1670-${ARM}-lr2e3-seed${SEED}.pt"
echo "[TASK $T/$NTASKS] arm=$ARM seed=$SEED cfg=$CFG cache=$CACHE -> $CK"

[[ -f "$CFG" ]] || { echo "FAIL LOUD: $CFG missing (git pull?)" >&2; exit 1; }
[[ -f "$CACHE/metadata.json" ]] || { echo "FAIL LOUD: $CACHE has no metadata.json" >&2; exit 1; }
[[ -f "$CACHE/valid_combos_near_rtt_capped.pkl" ]] || { echo "FAIL LOUD: $CACHE lacks the near-RTT sidecar" >&2; exit 1; }
[[ -f "$CACHE/rtt_chunk_0.pkl" ]] || { echo "FAIL LOUD: $CACHE lacks rtt_chunk_*.pkl" >&2; exit 1; }
[[ -f "$SPLIT" ]] || { echo "FAIL LOUD: split artifact $SPLIT missing" >&2; exit 1; }
[[ -f "$GUARD" ]] || { echo "FAIL LOUD: $GUARD missing -- the instrument bar for this cache has not been read" >&2; exit 1; }
python3 - "$GUARD" <<'EOF2'
import json, sys
g = json.load(open(sys.argv[1]))
if g.get("verdict") != "INSTRUMENT-PASS":
    raise SystemExit(f"FAIL LOUD: {sys.argv[1]} verdict is {g.get('verdict')!r}; nothing trains on a cache that failed its instrument bar")
print("[guard] INSTRUMENT-PASS from", sys.argv[1])
EOF2
python3 - "$CACHE" <<'EOF2'
import json, sys
m = json.load(open(sys.argv[1] + "/metadata.json"))
for k, v in (("partial_state_contract", "partial_state_v3"), ("dag_primary_alpha_key", "2.5"),
             ("peer_exchange_block", True), ("platform_feature_dim", 14)):
    if m.get(k) != v:
        raise SystemExit(f"FAIL LOUD: cache metadata {k}={m.get(k)!r}, this recipe requires {v!r}")
got = m.get("label_objective") or "rtt"
if not got.startswith("rtt_drift:1") or "clock=measured" not in got:
    raise SystemExit(f"FAIL LOUD: cache label_objective={got!r}")
print("[cache] contract ok, label =", got, "| datasets =", m.get("num_datasets"))
EOF2
# The conv is conditioned on data.edge_attr, so a cache without one would train a model whose
# message passing is fed nothing. Cheaper to assert here than to discover it in a gate read.
python3 - "$CACHE" <<'EOF2'
import pickle, sys
graphs = pickle.load(open(sys.argv[1] + "/graphs.pkl", "rb"))
g = graphs[0]
ea = getattr(g, "edge_attr", None)
if ea is None or ea.numel() == 0 or int(ea.shape[1]) != 5:
    raise SystemExit(f"FAIL LOUD: cache graph 0 edge_attr is {None if ea is None else tuple(ea.shape)}; the bipartite conv needs 5 columns")
if int(ea.shape[0]) != int(g.edge_index.shape[1]):
    raise SystemExit(f"FAIL LOUD: edge_attr {int(ea.shape[0])} rows vs edge_index {int(g.edge_index.shape[1])} cols -- not aligned")
print("[cache] edge_attr", tuple(ea.shape), "aligned with edge_index; nonzero cols =",
      int((ea.abs().sum(0) > 0).sum()))
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
checks = {"disable_message_passing": False, "mp_peer_edges": True, "mp_platform_edges": True,
          "mp_residual": False,
          # The whole lineage rests on these two. mp_bipartite_edge_conv is pinned TRUE on
          # BOTH arms -- checking only one would let a silently-GIN arm through -- and the
          # zero flag is what separates the treatment from its control.
          "mp_bipartite_edge_conv": True,
          "mp_bipartite_edge_attr_zero": arm == "gnnedge0",
          "partial_state_contract": "partial_state_v3", "partial_state_feature_dim": 22,
          "dag_alpha_key": "2.5", "decode_replica_reuse": True, "decode_relax_on_stuck": True,
          "peer_mass": True}
for k, want in checks.items():
    if sc.get(k) != want:
        raise SystemExit(f"FAIL LOUD: sidecar {k}={sc.get(k)!r}, arm {arm} expects {want!r}")
es = sc.get("early_stopping") or {}
if (es.get("patience"), es.get("min_epochs")) != (60, 100):
    raise SystemExit(f"FAIL LOUD: sidecar early_stopping={es!r}")
sa = sc.get("split_artifact") or {}
sha = hashlib.sha256(open(split, "rb").read()).hexdigest()
if sa.get("sha256") != sha:
    raise SystemExit(f"FAIL LOUD: sidecar split sha {sa.get('sha256')} != {split} {sha}")
# The sidecar is a claim; the weights are the fact. mp_bipartite_edge_conv is weight-VISIBLE,
# so verify it against the state dict rather than trusting the trainer to have recorded it.
sd = torch.load(ck, map_location="cpu")
n_bip = sum(1 for k in sd if k.startswith("bip_convs."))
if n_bip == 0:
    raise SystemExit("FAIL LOUD: sidecar says mp_bipartite_edge_conv=True but the checkpoint has no bip_convs.* weights")
print(f"[sidecar] ok: arm={arm} bip_convs weights={n_bip} attr_zero={sc.get('mp_bipartite_edge_attr_zero')}")
EOF2
echo "=== task $T done: $CK ==="
