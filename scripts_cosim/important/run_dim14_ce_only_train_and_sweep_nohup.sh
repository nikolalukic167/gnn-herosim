#!/usr/bin/env bash
# Full pipeline: train dim14 CE-only (100ep) -> copy checkpoint -> 7-config argmax sweep.
set -euo pipefail

ROOT="/root/projects/my-herosim"
cd "$ROOT"

TS="$(date +%Y%m%d)"
TRAIN_LOG="logs/train_near_rtt_v2_dim14_ce_only_${TS}.log"
PIPELINE_LOG="logs/dim14_ce_only_pipeline_${TS}.log"
SWEEP_OUT="simulation_data/normal_sim_sweeps/gnn_near_rtt_v2_dim14_ce_only_${TS}"
NB_MODEL="src/notebooks/models/near-rtt-v2-dim14-ce-only.pt"
REPO_MODEL="models/near-rtt-v2-dim14-ce-only.pt"
TRAINER="src/notebooks/train_near_rtt_v2_dim14_ce_only.py"

log() {
  echo "[$(date -Is)] $*" | tee -a "$PIPELINE_LOG"
}

mkdir -p logs models src/notebooks/models

log "=== dim14 CE-only pipeline start ==="
log "Train log: ${TRAIN_LOG}"
log "Sweep output: ${SWEEP_OUT}"

if [[ -f "$REPO_MODEL" ]]; then
  log "WARN: existing checkpoint will be overwritten after training: ${REPO_MODEL}"
fi

log "Phase 1: training (100 epochs, dim14 cache, CE-only)"
cd "${ROOT}/src/notebooks"
if ! pipenv run python3 "${ROOT}/${TRAINER}" > "${ROOT}/${TRAIN_LOG}" 2>&1; then
  log "ERROR: training failed — see ${TRAIN_LOG}"
  exit 1
fi
cd "$ROOT"

if [[ ! -f "$NB_MODEL" ]]; then
  log "ERROR: training finished but checkpoint missing: ${NB_MODEL}"
  exit 1
fi

cp -f "$NB_MODEL" "$REPO_MODEL"
log "Copied checkpoint -> ${REPO_MODEL}"

log "Phase 2: 7-config argmax sweep"
if ! bash scripts_cosim/important/run_gnn_near_rtt_v2_dim14_ce_only_all_configs_nohup.sh "$SWEEP_OUT" >> "$PIPELINE_LOG" 2>&1; then
  log "ERROR: sweep failed — see ${PIPELINE_LOG}"
  exit 1
fi

log "Phase 3: quick RTT summary"
pipenv run python3 - <<'PY' | tee -a "$PIPELINE_LOG"
import json
from pathlib import Path

sweep = Path("simulation_data/normal_sim_sweeps")
dim14_ce = sorted(sweep.glob("gnn_near_rtt_v2_dim14_ce_only_*/results"))[-1]
dim14_full = Path("simulation_data/normal_sim_sweeps/gnn_near_rtt_v2_dim14_1060_20260608/results")
ce13 = Path("simulation_data/normal_sim_sweeps/gnn_near_rtt_v2_clean_1230_ce_only_20260608/results")

def load_rtt(d: Path) -> dict[str, float]:
    out = {}
    for f in sorted(d.glob("*.json")):
        if f.name.endswith(".decode_stats.json"):
            continue
        out[f.stem] = json.load(open(f))["total_rtt"]
    return out

rows = sorted(load_rtt(dim14_ce).keys())
d_ce = load_rtt(dim14_ce)
d_full = load_rtt(dim14_full) if dim14_full.is_dir() else {}
d13 = load_rtt(ce13) if ce13.is_dir() else {}

wins_vs_13 = wins_vs_full = 0
print(f"dim14-ce-only sweep: {dim14_ce}")
print(f"{'config':<28} {'dim14-ce':>14} {'dim13-ce':>14} {'dim14-full':>14}  winner")
for cfg in rows:
    v = d_ce.get(cfg)
    v13 = d13.get(cfg)
    vf = d_full.get(cfg)
    candidates = [(v, "dim14-ce"), (v13, "dim13-ce"), (vf, "dim14-full")]
    candidates = [(x, n) for x, n in candidates if x is not None]
    winner = min(candidates, key=lambda t: t[0])[1] if candidates else "?"
    if v13 is not None and v is not None and v < v13:
        wins_vs_13 += 1
    if vf is not None and v is not None and v < vf:
        wins_vs_full += 1
    def fmt(x):
        return f"{x:>14,.0f}" if x is not None else f"{'NA':>14}"
    print(f"{cfg:<28} {fmt(v)} {fmt(v13)} {fmt(vf)}  {winner}")
print(f"Win counts: dim14-ce vs dim13-ce {wins_vs_13}/{len(rows)}; vs dim14-full {wins_vs_full}/{len(rows)}")
PY

log "=== dim14 CE-only pipeline complete ==="
log "Checkpoint: ${REPO_MODEL}"
log "Sweep: ${SWEEP_OUT}/results/"
log "Train log: ${TRAIN_LOG}"
