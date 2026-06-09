#!/usr/bin/env bash
# Two-phase pipeline: Phase A CE anchor (skip if exists) -> Phase B ranking fine-tune -> 7-config sweep.
set -euo pipefail

ROOT="/root/projects/my-herosim"
cd "$ROOT"

REGRET_WEIGHT="${NEAR_RTT_REGRET_WEIGHT:-0.02}"
REGRET_TAG="r$(printf '%03d' "$(python3 -c "print(int(round(float('${REGRET_WEIGHT}') * 100)))")")"
TS="$(date +%Y%m%d_%H%M%S)"
WANDB_NAME="near-rtt-v2-dim14-ce-init-${REGRET_TAG}"

PHASE_A_MODEL="models/near-rtt-v2-dim14-ce-only.pt"
NB_MODEL="src/notebooks/models/${WANDB_NAME}.pt"
REPO_MODEL="models/near-rtt-v2-dim14-ce-init-regret-${REGRET_TAG}.pt"
REPO_MODEL_ALIAS="models/near-rtt-v2-dim14-ce-init-regret.pt"
TRAINER="src/notebooks/train_near_rtt_v2_dim14_ce_init_regret.py"
TRAIN_LOG="logs/train_near_rtt_v2_dim14_ce_init_regret_${REGRET_TAG}_${TS}.log"
PIPELINE_LOG="logs/dim14_two_phase_regret_pipeline_${REGRET_TAG}_${TS}.log"
SWEEP_OUT="simulation_data/normal_sim_sweeps/gnn_near_rtt_v2_dim14_ce_init_regret_${REGRET_TAG}_${TS}"

log() {
  echo "[$(date -Is)] $*" | tee -a "$PIPELINE_LOG"
}

mkdir -p logs models src/notebooks/models

log "=== dim14 two-phase regret pipeline start ==="
log "Regret weight: ${REGRET_WEIGHT} (${REGRET_TAG})"
log "WandB name: ${WANDB_NAME}"
log "Train log: ${TRAIN_LOG}"
log "Sweep output: ${SWEEP_OUT}"

# Phase 1: CE anchor (skip if checkpoint exists)
if [[ -f "$PHASE_A_MODEL" ]]; then
  log "Phase 1: SKIP — CE anchor exists: ${PHASE_A_MODEL}"
else
  log "Phase 1: training CE anchor (100 epochs)"
  cd "${ROOT}/src/notebooks"
  if ! pipenv run python3 "${ROOT}/src/notebooks/train_near_rtt_v2_dim14_ce_only.py" > "${ROOT}/logs/train_near_rtt_v2_dim14_ce_only_${TS}.log" 2>&1; then
    log "ERROR: Phase A training failed"
    exit 1
  fi
  cd "$ROOT"
  if [[ ! -f "src/notebooks/models/near-rtt-v2-dim14-ce-only.pt" ]]; then
    log "ERROR: Phase A checkpoint missing after training"
    exit 1
  fi
  cp -f "src/notebooks/models/near-rtt-v2-dim14-ce-only.pt" "$PHASE_A_MODEL"
  log "Phase 1: copied CE anchor -> ${PHASE_A_MODEL}"
fi

# Phase 2: ranking fine-tune from CE init
log "Phase 2: ranking fine-tune (regret=${REGRET_WEIGHT}, LR=2e-4, 40ep)"
export NEAR_RTT_REGRET_WEIGHT="${REGRET_WEIGHT}"
export WANDB_RUN_NAME="${WANDB_NAME}"
cd "${ROOT}/src/notebooks"
if ! pipenv run python3 "${ROOT}/${TRAINER}" > "${ROOT}/${TRAIN_LOG}" 2>&1; then
  log "ERROR: Phase B training failed — see ${TRAIN_LOG}"
  exit 1
fi
cd "$ROOT"

if [[ ! -f "$NB_MODEL" ]]; then
  log "ERROR: Phase B checkpoint missing: ${NB_MODEL}"
  exit 1
fi

# Phase 3: copy checkpoint to models/
cp -f "$NB_MODEL" "$REPO_MODEL"
cp -f "$NB_MODEL" "$REPO_MODEL_ALIAS"
log "Copied checkpoint -> ${REPO_MODEL} (alias: ${REPO_MODEL_ALIAS})"

# Phase 4: 7-config argmax sweep
log "Phase 4: 7-config argmax sweep (no LQB)"
if ! bash scripts_cosim/important/run_gnn_near_rtt_v2_dim14_ce_init_regret_all_configs_nohup.sh "$SWEEP_OUT" "$REPO_MODEL" >> "$PIPELINE_LOG" 2>&1; then
  log "ERROR: sweep failed — see ${PIPELINE_LOG}"
  exit 1
fi

# Phase 5: RTT comparison table
log "Phase 5: RTT comparison vs dim14-ce + dim14-full"
export REGRET_TAG="${REGRET_TAG}"
pipenv run python3 - <<'PY' | tee -a "$PIPELINE_LOG"
import json
import os
from pathlib import Path

sweep = Path("simulation_data/normal_sim_sweeps")
regret_tag = os.environ.get("REGRET_TAG", "r002")
two_phase = sorted(sweep.glob(f"gnn_near_rtt_v2_dim14_ce_init_regret_{regret_tag}_*/results"))
dim14_ce = Path("simulation_data/normal_sim_sweeps/gnn_near_rtt_v2_dim14_ce_only_20260609/results")
dim14_full = Path("simulation_data/normal_sim_sweeps/gnn_near_rtt_v2_dim14_1060_20260608/results")

def load_rtt(d: Path) -> dict[str, float]:
    out = {}
    for f in sorted(d.glob("*.json")):
        if f.name.endswith(".decode_stats.json"):
            continue
        out[f.stem] = json.load(open(f))["total_rtt"]
    return out

if not two_phase:
    print(f"No two-phase sweep found for tag {regret_tag}")
    raise SystemExit(1)

d_tp = load_rtt(two_phase[-1])
d_ce = load_rtt(dim14_ce) if dim14_ce.is_dir() else {}
d_full = load_rtt(dim14_full) if dim14_full.is_dir() else {}

rows = sorted(d_tp.keys())
wins_vs_ce = wins_vs_full = 0
sum_tp = sum_ce = sum_full = 0.0
print(f"Two-phase sweep: {two_phase[-1]}")
print(f"{'config':<28} {'two-phase':>14} {'dim14-ce':>14} {'dim14-full':>14}  winner")
for cfg in rows:
    v = d_tp.get(cfg)
    vc = d_ce.get(cfg)
    vf = d_full.get(cfg)
    if v is not None:
        sum_tp += v
    if vc is not None:
        sum_ce += vc
    if vf is not None:
        sum_full += vf
    candidates = [(v, "two-phase"), (vc, "dim14-ce"), (vf, "dim14-full")]
    candidates = [(x, n) for x, n in candidates if x is not None]
    winner = min(candidates, key=lambda t: t[0])[1] if candidates else "?"
    if vc is not None and v is not None and v < vc:
        wins_vs_ce += 1
    if vf is not None and v is not None and v < vf:
        wins_vs_full += 1
    def fmt(x):
        return f"{x:>14,.0f}" if x is not None else f"{'NA':>14}"
    print(f"{cfg:<28} {fmt(v)} {fmt(vc)} {fmt(vf)}  {winner}")
print(f"Sum RTT: two-phase {sum_tp:,.0f} | dim14-ce {sum_ce:,.0f} | dim14-full {sum_full:,.0f}")
print(f"Win counts: vs dim14-ce {wins_vs_ce}/{len(rows)}; vs dim14-full {wins_vs_full}/{len(rows)}")
PY

log "=== dim14 two-phase regret pipeline complete ==="
log "Checkpoint: ${REPO_MODEL}"
log "Sweep: ${SWEEP_OUT}/results/"
log "Train log: ${TRAIN_LOG}"
