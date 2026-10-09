#!/bin/bash
# scale_160_v1 PIPELINE dry run on S6's dry corpus (one topology, wrong rungs, a handful of batches): cache build -> a TOY split by dataset (flagged dry_run;
# the real split is by topology) -> gnn_eng and gnn_eng_physmp g3 for a few epochs through run_experiment.py -> sidecheck, curves, cache size, peak RSS, epoch cost.
# Nothing here is evidence; checkpoints are deleted with the run. Reads the corpus through a symlink view (nothing is written into it).
#   sbatch --export=ALL,WT=<pinned worktree>,OUT=<new dir>[,CORPUS=<dir>,SUF=dry,EPOCHS=4] scale160_dry.sbatch
#SBATCH --job-name=s160-dry
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/s160-dry-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=100G
#SBATCH --time=06:00:00
set -uo pipefail
WT=${WT:?}; OUT=${OUT:?}; CP=${CORPUS:-/share/nikola.lukic/workload_fix_v1/scale160_dry}; SUF=${SUF:-dry}; EPOCHS=${EPOCHS:-4}
cd "$WT"
for v in $(compgen -e | grep -E '^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|NEAR_RTT_|PARTIAL_STATE)'); do unset "$v"; done
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT:$WT/src/notebooks" PYTHONUNBUFFERED=1 WANDB_ENTITY=nikolalukic167-tu-wien OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 CUDA_VISIBLE_DEVICES=""
export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa HEROSIM_SHARED_AUTOSCALER=0 GATE_FIXED_POLICY_TIME_SCALE=1.0
export DAG_PRIMARY_ALPHA_KEY=inf PARTIAL_STATE_CONTRACT=partial_state_v5 NEAR_RTT_LABEL_BACKLOG_CLOCK=seeded
echo "HEAD $(git rev-parse HEAD) node $(hostname)"
[[ ! -e "$OUT" ]] || { echo "FAIL LOUD: $OUT exists"; exit 1; }
[[ -L models ]] || { echo "FAIL LOUD: models/ must be the symlink"; exit 1; }
mkdir -p "$OUT/view" "$OUT/cfg" "$OUT/ckpt"
SRC=$(ls -d "$CP"/gnn_datasets_wf1_*_"$SUF" | head -1); echo "corpus $SRC: $(ls "$SRC" | grep -c '^ds_') datasets"
T0=$(date +%s)
python3 - "$SRC" "$OUT/view" <<'PY'
import os, shutil, sys
from pathlib import Path
src, view = Path(sys.argv[1]), Path(sys.argv[2])
dst = view / src.name; dst.mkdir()
n = 0
for d in sorted(src.glob("ds_*")):
    t = dst / d.name; t.mkdir()
    for f in d.iterdir():
        (shutil.copyfile if f.name == "optimal_result.json" else os.symlink)(f, t / f.name)
    n += 1
print(f"view {dst.name}: {n} datasets")
PY
for B in "$OUT"/view/*/; do python3 scripts_cosim/refresh_optimal_full_stats.py --base-dir "$B" --rewrite-ssc | tail -1; done
T1=$(date +%s)
python3 -u src/notebooks/prepare_graphs_cache.py --base-dirs "$OUT"/view/*/ --cache-dir "$OUT/cache" --dag-partial-state --platform-feature-dim 14 > "$OUT/cache.log" 2>&1 || { tail -30 "$OUT/cache.log"; echo "FAIL LOUD: cache build failed"; exit 1; }
T2=$(date +%s)
echo "view+ssc ${T1}-${T0} = $((T1-T0)) s; cache build $((T2-T1)) s"
grep -vE "it/s\]|^\s*$" "$OUT/cache.log" | tail -12 | cut -c1-220
du -sh "$OUT/cache" | cut -f1 | sed 's/^/cache size: /'; ls -l "$OUT/cache/graphs.pkl" | awk '{print "graphs.pkl bytes:", $5}'
python3 - "$OUT" <<'PY'
import json, pickle, statistics, sys
from pathlib import Path
out = Path(sys.argv[1]); cache = out / "cache"
m = json.loads((cache / "metadata.json").read_text()); g = pickle.load(open(cache / "graphs.pkl", "rb"))
ids = m["dataset_ids"]
print(f"graphs {len(g)} rtt rows {m.get('num_rtt_combo_rows')} physics_env {m.get('physics_env')} slate {m.get('candidate_slate')} contract {m.get('partial_state_contract')} pf_dim {m.get('platform_feature_dim')}")
def n_tasks(h): return int(getattr(h, "num_tasks", None) or h.task_features.shape[0])
print("per graph: tasks", [n_tasks(h) for h in g], "platform nodes", [int(h.platform_features.shape[0]) for h in g], "edges", [int(h.edge_index.shape[1]) for h in g])
need = {"physics_env", "candidate_slate", "partial_state_contract"}
assert need <= set(m), f"FAIL LOUD: metadata lacks {need - set(m)}"
assert len(g) >= 5, f"FAIL LOUD: only {len(g)} graphs; the toy split needs 5"
# TOY split by dataset (dry only: the production split is by topology)
from non_unique_lib.training_contract import SPLIT_ARTIFACT_SCHEMA, canonical_parent_id
parents = sorted({canonical_parent_id((m.get("parent_dataset_ids") or ids)[k]) for k in range(len(ids))})
sp = {"schema": SPLIT_ARTIFACT_SCHEMA, "cache_dir": str(cache), "n_parents": len(parents), "random_state": 0, "split_by": "dataset_TOY", "dry_run": True,
      "heldout_as_val": True, "test_placeholder": True, "topologies": {"train": ["16101"], "val": ["16101"], "test": ["16101"]},
      "train": parents[:-2], "val": parents[-2:-1], "test": parents[-1:]}
(out / "split.json").write_text(json.dumps(sp, sort_keys=True, separators=(",", ":")) + "\n")
print("toy split", {k: len(sp[k]) for k in ("train", "val", "test")})
PY
[[ -f "$OUT/split.json" ]] || { echo "FAIL LOUD: no toy split"; exit 1; }
fail=0
for ARM in gnn_eng gnn_eng_physmp; do
  python3 - "$ARM" "$OUT" "$EPOCHS" <<'PY'
import sys, yaml
arm, out, ep = sys.argv[1], sys.argv[2], int(sys.argv[3])
cfg = yaml.safe_load(open(f"experiments/scale_160_v1_{arm}_g3.yaml"))
cfg["cache_dir"] = f"{out}/cache"; cfg["env"]["NEAR_RTT_SPLIT_ARTIFACT"] = f"{out}/split.json"
cfg["args"].update({"epochs": ep, "patience": ep, "min-epochs": 1})
cfg["wandb"]["run_name"] += "-DRYRUN"; cfg["wandb"]["tags"] = cfg["wandb"]["tags"] + ["dryrun"]
yaml.safe_dump(cfg, open(f"{out}/cfg/{arm}.yaml", "w"), sort_keys=False)
PY
  RN=$(python3 -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['wandb']['run_name'])" "$OUT/cfg/$ARM.yaml")-seed1
  rm -f models/$RN.pt models/$RN.contract.json models/$RN.val.json models/$RN-final.pt models/$RN-final.contract.json
  T3=$(date +%s.%N)
  /usr/bin/time -v python3 -u run_experiment.py "$OUT/cfg/$ARM.yaml" --seed 1 2> "$OUT/ckpt/$ARM.time" | python3 -c "
import sys, time
for l in sys.stdin:
    sys.stdout.write(f'{time.time():.2f} ' + l); sys.stdout.flush()" > "$OUT/ckpt/$ARM.log"
  rc=${PIPESTATUS[0]}
  T4=$(date +%s.%N)
  grep -E "Maximum resident|Elapsed \(wall" "$OUT/ckpt/$ARM.time" | sed "s/^/[$ARM] /"
  if [[ $rc -ne 0 || ! -f models/$RN.contract.json ]]; then echo "[$ARM] FAILED rc=$rc"; tail -15 "$OUT/ckpt/$ARM.log" | cut -c1-250; fail=$((fail+1)); continue; fi
  python3 scripts_cosim/joint_burst_v2_sidecheck.py models/$RN.contract.json "ra_$ARM" "$OUT/split.json" inf && SC=ok || SC=FAIL
  echo "[$ARM] rc=$rc wall $(python3 -c "print(round($T4-$T3,1))")s sidecheck=$SC"
  grep -E "^[0-9.]+ Epoch +[0-9]+/" "$OUT/ckpt/$ARM.log" | cut -c1-230
  cat models/$RN.val.json 2>/dev/null; echo
  rm -f models/$RN.pt models/$RN.contract.json models/$RN.val.json models/$RN-final.pt models/$RN-final.contract.json
  [[ "$SC" == ok ]] || fail=$((fail+1))
done
echo "=== arms failed: $fail"
exit $fail
