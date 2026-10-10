#!/bin/bash
# r1_attribution_v1 downstream dry run, step 2: every learned arm trained for 2 epochs through run_experiment.py on the dry-run cache, with W&B logging.
# Per arm: the _smoke config with the cache and split swapped (absolute paths), timestamps on every epoch line, the sidecar checked with the
# arm's sidecheck kind, the curves read with read_training_curves.py. Checkpoints are copied to $RUN/ckpt_$DEVICE; delete them with the cache.
#   sbatch --export=ALL,WT=<pinned worktree>,RUN=<dry-run dir with cache/ and split.json>,DEVICE=cpu|gpu [--partition=GPU-a40 --gres=gpu:1] r1a_dryrun_train.sbatch
#SBATCH --job-name=r1a-dtrain
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/r1a-dtrain-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=12:00:00
set -uo pipefail
ulimit -n 65536 2>/dev/null || true
WT=${WT:?}; RUN=${RUN:?}; DEVICE=${DEVICE:-cpu}
ARMS=${ARMS:-"gnn_eng twin_eng mlp_same gnn_raw twin_raw gnn_eng_physmp set_transformer"}
cd "$WT"
for v in $(compgen -e | grep -E '^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|NEAR_RTT_|PARTIAL_STATE)'); do unset "$v"; done
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT:$WT/src/notebooks" PYTHONUNBUFFERED=1 WANDB_ENTITY=nikolalukic167-tu-wien OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
[[ "$DEVICE" == cpu ]] && export CUDA_VISIBLE_DEVICES=""
python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__,'cuda',torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
echo "HEAD $(git rev-parse HEAD) node $(hostname) device $DEVICE"
export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa HEROSIM_SHARED_AUTOSCALER=0 GATE_FIXED_POLICY_TIME_SCALE=1.0
CKPT=$RUN/ckpt_$DEVICE; CFG=$RUN/cfg_$DEVICE; mkdir -p "$CKPT" "$CFG"
SPLIT=$RUN/split.json; CACHE=$RUN/cache
[[ -L models ]] || { echo "FAIL LOUD: models/ must be the symlink"; exit 1; }
fail=0
for ARM in $ARMS; do
  python3 - "$ARM" "$CACHE" "$SPLIT" "$CFG" "$DEVICE" <<'PY'
import sys, yaml
arm, cache, split, cfgdir, dev = sys.argv[1:6]
cfg = yaml.safe_load(open(f"experiments/r1_attribution_v1_{arm}_smoke.yaml"))
cfg["cache_dir"] = cache; cfg["env"]["NEAR_RTT_SPLIT_ARTIFACT"] = split
cfg["wandb"]["run_name"] += f"-dryrun-{dev}"; cfg["wandb"]["tags"] = cfg["wandb"]["tags"] + ["dryrun", dev]
yaml.safe_dump(cfg, open(f"{cfgdir}/{arm}.yaml", "w"), sort_keys=False)
PY
  RUN_NAME=$(python3 -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['wandb']['run_name'])" "$CFG/$ARM.yaml")-seed1
  rm -f "models/$RUN_NAME.pt" "models/$RUN_NAME.contract.json" "models/$RUN_NAME-final.pt" "models/$RUN_NAME-final.contract.json"
  T0=$(date +%s.%N)
  python3 -u run_experiment.py "$CFG/$ARM.yaml" --seed 1 2>&1 | python3 -c "
import sys, time
for l in sys.stdin:
    sys.stdout.write(f'{time.time():.2f} ' + l); sys.stdout.flush()" > "$CKPT/$ARM.log"
  rc=${PIPESTATUS[0]}
  T1=$(date +%s.%N)
  if [[ $rc -ne 0 || ! -f "models/$RUN_NAME.contract.json" ]]; then echo "[$ARM] FAILED rc=$rc"; tail -15 "$CKPT/$ARM.log" | cut -c1-250; fail=$((fail+1)); continue; fi
  cp "models/$RUN_NAME.pt" "models/$RUN_NAME.contract.json" "$CKPT/"
  python3 scripts_cosim/joint_burst_v2_sidecheck.py "models/$RUN_NAME.contract.json" "ra_$ARM" "$SPLIT" inf && SC=ok || SC=FAIL
  WB=$(ls -dt wandb/run-* 2>/dev/null | head -1)
  echo "[$ARM] rc=$rc wall $(python3 -c "print(round($T1-$T0,1))")s sidecheck=$SC wandb=$WB"
  grep -E "^[0-9.]+ Epoch +[0-9]+/" "$CKPT/$ARM.log" | cut -c1-230
  python3 scripts_cosim/read_training_curves.py "$WB" 2>&1 | head -40 | cut -c1-200
  [[ "$SC" == ok ]] || fail=$((fail+1))
done
echo "=== arms failed: $fail"
exit $fail
