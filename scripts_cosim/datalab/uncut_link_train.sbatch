#!/bin/bash
# live_link_features_v1 development training (fair-share-label recipe: from scratch, 20 epochs, lr 1e-3, partial_state_v7). One job = one (arm, seed); the launcher (uncut_link_launch.sh) submits the 12 (4 arms x seeds 1-3) with --nodelist spread over 4 nodes.
# Needs in the environment: WT, EXPECT_HEAD, CFGN (uncut_link_{gnn,twin,mlp,mlp_wide}[_e100] on CPU; uncut_linkconv_{gnn,gnn_rep,twin}[_e100] on a GPU, submitted with --partition=GPU-a40,GPU-l40s --gres=gpu:1), S (1-3), SPLIT_SHA, LABEL_SHA, CACHE_SHA (sha256 of <cache>/metadata.json), EXPECT_CPU (the lscpu model name: all 12 run on one device class). The three shas are the SAME for all four arms (one shared corpus, split and label set).
#SBATCH --job-name=link-train
#SBATCH --output=/share/nikola.lukic/uncut_groups/flowcap/link_corpus/logs/train-%x-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=150G
#SBATCH --time=2-00:00:00
#SBATCH --exclude=os-cpu-slurm-[21,15]
set -uo pipefail
ulimit -n 65536 2>/dev/null || true
WT=${WT:?FAIL LOUD: set WT}; EXPECT_HEAD=${EXPECT_HEAD:?FAIL LOUD: set EXPECT_HEAD}
CFGN=${CFGN:?FAIL LOUD: set CFGN}; S=${S:?FAIL LOUD: set S}
# CFGN -> the run name, the gate sidecheck arm, the venue and the epochs; _e100 is the A2/A3 100-epoch twin of the 20-epoch config (run name suffix -e100)
BASE=${CFGN%_e100}; BASE=${BASE%_e40}; EP=20; [[ "$CFGN" == *_e100 ]] && EP=100; [[ "$CFGN" == *_e40 ]] && EP=40  # _e40: the A6 40-epoch runs of the selected variant (run name suffix -e40)
case "$BASE" in
  uncut_link_gnn) WANT=uncut-link-gnn-eng; ARMCHK=ra_gnn_eng_v7; VENUE=cpu;;
  uncut_link_twin) WANT=uncut-link-twin; ARMCHK=ra_twin_eng_v7; VENUE=cpu;;
  uncut_link_mlp) WANT=uncut-link-mlp-same; ARMCHK=ra_mlp_same_v7; VENUE=cpu;;
  uncut_link_mlp_wide) WANT=uncut-link-mlp-wide; ARMCHK=ra_mlp_same_v7; VENUE=cpu;;
  uncut_linkstatic_gnn) WANT=uncut-linkstatic-gnn-eng; ARMCHK=ra_gnn_eng_v8; VENUE=cpu;;
  uncut_linkstatic_twin) WANT=uncut-linkstatic-twin; ARMCHK=ra_twin_eng_v8; VENUE=cpu;;
  uncut_linkstatic_mlp) WANT=uncut-linkstatic-mlp-same; ARMCHK=ra_mlp_same_v8; VENUE=cpu;;
  uncut_linkstatic_mlp_wide) WANT=uncut-linkstatic-mlp-wide; ARMCHK=ra_mlp_same_v8; VENUE=cpu;;
  uncut_linkconv_gnn) WANT=uncut-linkconv-gnn-eng; ARMCHK=ra_gnn_eng_v7conv; VENUE=gpu;;
  uncut_linkconv_gnn_rep) WANT=uncut-linkconv-gnn-eng-rep; ARMCHK=ra_gnn_eng_v7conv; VENUE=gpu;;
  uncut_linkconv_twin) WANT=uncut-link-twin-gpu; ARMCHK=ra_twin_eng_v7; VENUE=gpu;;
  *) echo "FAIL LOUD: CFGN $CFGN"; exit 1;;
esac
[[ $EP != 20 ]] && WANT=$WANT-e$EP
[[ "$S" =~ ^[1-3]$ ]] || { echo "FAIL LOUD: seed $S"; exit 1; }
cd "$WT"
for v in $(compgen -e | grep -E '^(HEROSIM_|GNN_|COSIM_|LIVE_AUDIT_|GATE_|SIM_|QUEUE_FEATURE|MAX_PLACEMENT|NEAR_RTT_|PARTIAL_STATE|R1A_|TRAIN_INIT)'); do unset "$v"; done
if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3 PYTHONPATH="$WT:$WT/src/notebooks" PYTHONUNBUFFERED=1 WANDB_ENTITY=nikolalukic167-tu-wien OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
export CUBLAS_WORKSPACE_CONFIG=:4096:8
[[ "$VENUE" == cpu ]] && export CUDA_VISIBLE_DEVICES=""
export HEROSIM_TRANSFER_MODEL=pipelined HEROSIM_REPLICA_RELEASE=1 HEROSIM_SCALEOUT=kpa HEROSIM_SHARED_AUTOSCALER=0 GATE_FIXED_POLICY_TIME_SCALE=1.0
export R1A_SLATE=declared_pruning_v1_uncapped
HEAD=$(git rev-parse HEAD)
echo "cfg $CFGN seed $S | HEAD $HEAD node $(hostname) job ${SLURM_JOB_ID:-} cpus $(nproc) smt $(cat /sys/devices/system/cpu/smt/control 2>/dev/null) ulimit-n $(ulimit -n)"
[[ "$HEAD" == "$EXPECT_HEAD"* ]] || { echo "FAIL LOUD: worktree HEAD $HEAD is not $EXPECT_HEAD"; exit 1; }
CPU=$(lscpu | sed -n "s/^Model name: *//p" | head -1); echo "venue $VENUE cpu model: $CPU"
if [[ "$VENUE" == cpu ]]; then
  [[ "$CPU" == "${EXPECT_CPU:?FAIL LOUD: set EXPECT_CPU}" ]] || { echo "FAIL LOUD: cpu model $CPU is not $EXPECT_CPU"; exit 1; }
  [[ "$(nproc)" -ge 6 ]] || { echo "FAIL LOUD: fewer than 6 cores in the cpuset"; exit 1; }
else
  echo "gpu: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 | head -1)"
  python3 -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "FAIL LOUD: a GPU venue without a visible GPU"; exit 1; }
fi
[[ "$(ulimit -n)" -ge 65536 ]] || { echo "FAIL LOUD: ulimit -n $(ulimit -n) < 65536"; exit 1; }
[[ -z "$(git status --porcelain -- src scripts_cosim experiments run_experiment.py)" ]] || { echo "FAIL LOUD: worktree dirty"; exit 1; }
CFG=experiments/$CFGN.yaml
grep -q TBD "$CFG" && { echo "FAIL LOUD: $CFG still has a TBD"; exit 1; }
# determinism test first (no GPU). A failure or error is fatal. The only skips allowed are the route_b smoke caches that are absent on the cluster and the ARM2 peer cache that is not set (the 900763 reasons: 16 passed, 6 skipped); any other skip reason is fatal. The skip lines go to the log.
DET=$(CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 python3 -m pytest tests/test_trainer_determinism.py -q -rs 2>&1 | tail -30); echo "$DET" | grep -E "SKIPPED|passed|failed|error" | cut -c1-260
echo "$DET" | grep -qE ' passed' && ! echo "$DET" | grep -qE ' (failed|error)' || { echo "FAIL LOUD: determinism test did not pass"; exit 1; }
BADSKIP=$(echo "$DET" | grep -E '^SKIPPED' | grep -vE 'cache not present at .*/graphs_cache_route_b_smoke_s_dag|no --dag-partial-state cache at .*/graphs_cache_route_b_smoke_s_dag/graphs.pkl|ARM2_DETERMINISM_CACHE not set to a peer cache' || true)
[[ -z "$BADSKIP" ]] || { echo "FAIL LOUD: determinism test skipped for a reason other than the absent route_b/ARM2 caches:"; echo "$BADSKIP"; exit 1; }
read -r CACHE CONTRACT LO SPLITF INIT < <(python3 -c "import yaml,sys; c=yaml.safe_load(open(sys.argv[1])); e=c['env']; assert c['trainer']=='gnn' and c['args']['epochs'] in (20, 40, 100) and c['args']['min-epochs']==c['args']['epochs']==c['args']['patience'] and (c['args']['epochs']==20)==(not c['wandb']['run_name'].endswith('-e%d' % c['args']['epochs'])) and c['args']['learning-rate']==0.001, 'recipe'; print(c['cache_dir'], e['PARTIAL_STATE_CONTRACT'], e['NEAR_RTT_LABEL_OVERRIDE_JSON'], e['NEAR_RTT_SPLIT_ARTIFACT'], e.get('TRAIN_INIT_CHECKPOINT','-'))" "$CFG") || { echo "FAIL LOUD: config $CFG"; exit 1; }
[[ "$INIT" == "-" ]] || { echo "FAIL LOUD: TRAIN_INIT_CHECKPOINT set; this is from scratch"; exit 1; }
[[ -f "$CFG" && -L models && -f "$SPLITF" && -f "$CACHE/metadata.json" && -f "$LO" ]] || { echo "FAIL LOUD: config, models/ symlink, split ($SPLITF), cache ($CACHE) or labels ($LO) missing"; exit 1; }
[[ "$(sha256sum "$SPLITF" | cut -d' ' -f1)" == "${SPLIT_SHA:?FAIL LOUD: set SPLIT_SHA}" ]] || { echo "FAIL LOUD: split sha256"; exit 1; }
[[ "$(sha256sum "$LO" | cut -d' ' -f1)" == "${LABEL_SHA:?FAIL LOUD: set LABEL_SHA}" ]] || { echo "FAIL LOUD: label override $LO sha256"; exit 1; }
[[ "$(sha256sum "$CACHE/metadata.json" | cut -d' ' -f1)" == "${CACHE_SHA:?FAIL LOUD: set CACHE_SHA}" ]] || { echo "FAIL LOUD: cache metadata sha256"; exit 1; }
mkdir -p logs
RUN=$(python3 -c "import yaml,sys;print(yaml.safe_load(open(sys.argv[1]))['wandb']['run_name'])" "$CFG")
[[ "$RUN" == "$WANT" ]] || { echo "FAIL LOUD: run_name $RUN is not $WANT"; exit 1; }
NTRAIN=$(python3 -c "import json,sys;print(len(json.load(open(sys.argv[1]))['train']))" "$SPLITF")
CK=models/$RUN-seed$S
[[ ! -e "$CK.pt" && ! -e "$CK-final.pt" ]] || { echo "FAIL LOUD: $CK exists; refusing to overwrite"; exit 1; }
python3 -u run_experiment.py "$CFG" --seed "$S" > "logs/link-$RUN-seed$S.log" 2>&1
rc=$?
echo "run_experiment exit $rc"
fail=0
[[ $rc -eq 0 ]] || { tail -15 "logs/link-$RUN-seed$S.log" | cut -c1-240; fail=1; }
for f in "$CK-final.pt" "$CK-final.contract.json" "$CK.val.json"; do [[ -f "$f" ]] || { echo "FAIL LOUD: no $f"; fail=$((fail+1)); }; done
echo "label override $LO sha256 $LABEL_SHA; split $SPLITF sha256 $SPLIT_SHA; cache $CACHE metadata sha256 $CACHE_SHA"
[[ "$(tr '\r' '\n' < "logs/link-$RUN-seed$S.log" | grep -c "train label replaced on $NTRAIN graphs")" -ge 1 ]] || { echo "FAIL LOUD: the label override was not applied to all $NTRAIN train graphs"; fail=$((fail+1)); }
grep -qiE "TRAIN_INIT_CHECKPOINT|loaded (init|checkpoint)|warm.?start" "logs/link-$RUN-seed$S.log" && { echo "FAIL LOUD: log mentions an init checkpoint"; fail=$((fail+1)); }
python3 - "$CK-final.contract.json" "$CONTRACT" "${CFGN##*_}" <<'PY' || fail=$((fail+1))
import json, sys
c = json.load(open(sys.argv[1]))
def find(o, k):
    if isinstance(o, dict):
        if k in o:
            return o[k]
        for v in o.values():
            r = find(v, k)
            if r is not None:
                return r
    return None
bad = []
if find(c, "batched_teacher_forced") is not True:
    bad.append("batched_teacher_forced")
if find(c, "partial_state_contract") != sys.argv[2]:
    bad.append(f"partial_state_contract {find(c, 'partial_state_contract')!r} != {sys.argv[2]}")
print("sidecar check:", "OK" if not bad else "FAIL LOUD: " + "; ".join(bad))
sys.exit(1 if bad else 0)
PY
R1A_SLATE=declared_pruning_v1_uncapped python3 scripts_cosim/joint_burst_v2_sidecheck.py "$CK-final.contract.json" "$ARMCHK" "$SPLITF" inf || fail=$((fail+1))
md5sum "$CK-final.pt" "$CK-final.contract.json" "$CK.pt" 2>/dev/null
echo "=== $RUN seed $S: $fail failure(s)"
exit $fail
