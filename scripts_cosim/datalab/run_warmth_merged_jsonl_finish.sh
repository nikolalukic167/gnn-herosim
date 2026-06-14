#!/usr/bin/env bash
# Finish placements.jsonl for merged warmth+sparse training corpus (824–851 ds).
# Sparse: verify only (--only-missing-jsonl). Warmth: BF repair gaps + tail without best.json.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/nikola.lukic/gnn-herosim}"
cd "$PROJECT_ROOT"

ENV_NAME="${HEROSIM_ENV_NAME:-gnn}"
WARMTH_PHYSICS="${WARMTH_PHYSICS:-node_disk_v2}"
WARMTH_SUBDIR="${WARMTH_SUBDIR:-gnn_datasets_4tasks_1060_warmth_v2}"
SPARSE_SUBDIR="${SPARSE_SUBDIR:-gnn_datasets_4tasks_sparse_warmth_v2}"
TAIL_START="${TAIL_START:-491}"
TAIL_COUNT="${TAIL_COUNT:-9}"
WORKERS="${WORKERS:-}"

export PYTHONUNBUFFERED=1
export GNN_CAPTURE_DATASET_STATE="${GNN_CAPTURE_DATASET_STATE:-0}"
export COSIM_SUPPRESS_SIM_PRINTS="${COSIM_SUPPRESS_SIM_PRINTS:-1}"

mkdir -p logs "simulation_data/${WARMTH_SUBDIR}" "simulation_data/${SPARSE_SUBDIR}"

if ! command -v micromamba >/dev/null 2>&1; then
  source "${HOME}/.bashrc"
fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate "${ENV_NAME}"

scratch_dir=""
for candidate in \
  "/scratch/${USER}_${SLURM_JOB_ID:-local}" \
  "${TMPDIR:-/tmp}/herosim_${USER}_${SLURM_JOB_ID:-local}" \
  "${PROJECT_ROOT}/.scratch/${USER}_${SLURM_JOB_ID:-local}"; do
  if mkdir -p "${candidate}" 2>/dev/null; then
    chmod 700 "${candidate}" 2>/dev/null || true
    scratch_dir="${candidate}"
    break
  fi
done
if [[ -z "${scratch_dir}" ]]; then
  echo "ERROR: could not create scratch directory" >&2
  exit 1
fi
export SLURM_SCRATCH="${scratch_dir}"

if [[ -n "${SLURM_CPUS_PER_TASK:-}" && -z "${WORKERS}" ]]; then
  WORKERS=$(( SLURM_CPUS_PER_TASK > 1 ? SLURM_CPUS_PER_TASK - 1 : 1 ))
fi
WORKERS="${WORKERS:-$(( $(nproc) - 1 ))}"
(( WORKERS >= 1 )) || WORKERS=1

LOG="logs/warmth_merged_jsonl_finish_$(date +%Y%m%d_%H%M%S).log"

echo "=== Merged corpus JSONL finish ===" | tee "${LOG}"
echo "Node: ${SLURMD_NODENAME:-unknown} · Job: ${SLURM_JOB_ID:-local} · Workers: ${WORKERS}" | tee -a "${LOG}"

echo | tee -a "${LOG}"
echo "=== [1/4] Recover JSONL from .bf_scratch (warmth + sparse) ===" | tee -a "${LOG}"
bash scripts_cosim/recover_placements_jsonl_from_scratch.sh "simulation_data/${WARMTH_SUBDIR}" \
  | tee -a "${LOG}" || true
bash scripts_cosim/recover_placements_jsonl_from_scratch.sh "simulation_data/${SPARSE_SUBDIR}" \
  | tee -a "${LOG}" || true

audit_merged() {
  python3 - <<'PY'
from pathlib import Path

def audit(label, base):
    base = Path(base)
    if not base.exists():
        print(f"{label}: MISSING")
        return 0, 0, 0
    both = best_no_jsonl = neither = 0
    for ds in sorted(base.glob("ds_*")):
        has_best = (ds / "best.json").exists()
        j = ds / "placements" / "placements.jsonl"
        has_jsonl = j.exists() and j.stat().st_size > 0
        if has_best and has_jsonl:
            both += 1
        elif has_best:
            best_no_jsonl += 1
        elif not has_jsonl:
            neither += 1
    total = len(list(base.glob("ds_*")))
    print(f"{label}: total={total} both={both} best_no_jsonl={best_no_jsonl} neither={neither}")
    return both, best_no_jsonl, neither

root = Path("simulation_data")
w = audit("warmth_v2", root / "gnn_datasets_4tasks_1060_warmth_v2")
s = audit("sparse_v2", root / "gnn_datasets_4tasks_sparse_warmth_v2")
print(f"merged_both={w[0]+s[0]} merged_gaps={w[1]+w[2]+s[1]+s[2]}")
PY
}

echo | tee -a "${LOG}"
echo "=== [2/4] Audit before BF ===" | tee -a "${LOG}"
audit_merged | tee -a "${LOG}"

echo | tee -a "${LOG}"
echo "=== [3/4] BF repair: best.json without placements.jsonl ===" | tee -a "${LOG}"
python3 -u scripts_cosim/generate_gnn_datasets_fast.py \
  --quiet \
  --grid warmth_v2 \
  --max-datasets 500 \
  --start-from 0 \
  --resume \
  --only-missing-jsonl \
  --warmth-physics "${WARMTH_PHYSICS}" \
  --output-subdir "${WARMTH_SUBDIR}" \
  --progress-log-name "progress_${WARMTH_SUBDIR}_merged_jsonl_repair.txt" \
  --workers "${WORKERS}" \
  2>&1 | tee -a "${LOG}"

python3 -u scripts_cosim/generate_gnn_datasets_fast.py \
  --quiet \
  --grid sparse_warmth_v2 \
  --max-datasets 351 \
  --start-from 0 \
  --resume \
  --only-missing-jsonl \
  --warmth-physics "${WARMTH_PHYSICS}" \
  --output-subdir "${SPARSE_SUBDIR}" \
  --progress-log-name "progress_${SPARSE_SUBDIR}_merged_jsonl_repair.txt" \
  --workers "${WORKERS}" \
  2>&1 | tee -a "${LOG}"

echo | tee -a "${LOG}"
echo "=== [3b/4] Full BF warmth tail ds_$(printf '%05d' "${TAIL_START}").. ===" | tee -a "${LOG}"
python3 -u scripts_cosim/generate_gnn_datasets_fast.py \
  --quiet \
  --grid warmth_v2 \
  --max-datasets "${TAIL_COUNT}" \
  --start-from "${TAIL_START}" \
  --resume \
  --warmth-physics "${WARMTH_PHYSICS}" \
  --output-subdir "${WARMTH_SUBDIR}" \
  --progress-log-name "progress_${WARMTH_SUBDIR}_merged_tail.txt" \
  --workers "${WORKERS}" \
  2>&1 | tee -a "${LOG}"

echo | tee -a "${LOG}"
echo "=== [4/4] Audit after BF + validate merged ===" | tee -a "${LOG}"
audit_merged | tee -a "${LOG}"

python3 - <<'PY' | tee -a "${LOG}"
import sys
from pathlib import Path

warmth = Path("simulation_data/gnn_datasets_4tasks_1060_warmth_v2")
sparse = Path("simulation_data/gnn_datasets_4tasks_sparse_warmth_v2")
bad = []
for base in (warmth, sparse):
    for ds in sorted(base.glob("ds_*")):
        best = ds / "best.json"
        jsonl = ds / "placements" / "placements.jsonl"
        if not best.exists():
            bad.append(f"{ds}: missing best.json")
        elif not jsonl.exists() or jsonl.stat().st_size == 0:
            bad.append(f"{ds}: missing placements.jsonl")
if bad:
    print("MERGED JSONL VALIDATION FAILED:", file=sys.stderr)
    for line in bad[:30]:
        print(f"  {line}", file=sys.stderr)
    if len(bad) > 30:
        print(f"  ... +{len(bad)-30} more", file=sys.stderr)
    sys.exit(1)
print(f"merged JSONL OK: warmth={len(list(warmth.glob('ds_*')))} sparse={len(list(sparse.glob('ds_*')))}")
PY

if [[ -d "${scratch_dir}" && "${scratch_dir}" == /scratch/* ]]; then
  rm -rf "${scratch_dir}" || true
fi
