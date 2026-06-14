#!/usr/bin/env bash
# warmth+sparse+skew merged MLP: seq cache -> train -> datalab hub9 MLP sweep -> wait -> results.
set -euo pipefail

ROOT="/root/projects/my-herosim"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${ROOT}/src/notebooks"

WARMTH_DIR="simulation_data/gnn_datasets_4tasks_1060_warmth_v2"
SPARSE_DIR="simulation_data/gnn_datasets_4tasks_sparse_warmth_v2"
SKEW_DIR="simulation_data/gnn_datasets_4tasks_skew_warmth_v2"
SEQ_CACHE="simulation_data/graphs_cache_warmth_v2_sparse_skew_merged_seq"
MLP_MODEL="models/tabular/batch_edge_mlp_warmth_sparse_skew_merged_ce_reduced.pt"
GNN_MODEL="models/near-rtt-v2-warmth-sparse-skew-merged-ce-reduced.pt"
SWEEP_DIR="simulation_data/normal_sim_sweeps/warmth_sparse_skew_merged_ce_reduced_hub9_20260612"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/nikolalukic167}"
REMOTE="${REMOTE:-nikola.lukic@cluster.datalab.tuwien.ac.at}"
REPO="${REPO:-/home/nikola.lukic/gnn-herosim}"
RSYNC_SSH="ssh -i ${SSH_KEY}"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="logs/warmth_sparse_skew_merged_mlp_train_hub9_${TS}.log"

mkdir -p logs models/tabular

exec > >(tee -a "$LOG") 2>&1
echo "=== skew-merged MLP train + hub9 datalab pipeline ${TS} ==="

skew_done=$(find "$SKEW_DIR" -name best.json 2>/dev/null | wc -l)
if [[ "$skew_done" -lt 1 ]]; then
  echo "ERROR: skew dir has ${skew_done} best.json" >&2
  exit 1
fi

echo "=== Phase 1: sequential graph cache (warmth + sparse + skew) ==="
pipenv run python3 -u src/notebooks/prepare_graphs_cache_seq.py \
  --base-dirs "$WARMTH_DIR" "$SPARSE_DIR" "$SKEW_DIR" \
  --cache-dir "$SEQ_CACHE"

n_graphs=$(pipenv run python3 -c "import pickle; print(len(pickle.load(open('${SEQ_CACHE}/graphs.pkl','rb'))))")
echo "Seq cache: ${SEQ_CACHE} (${n_graphs} graphs)"

echo "=== Phase 2: CE-reduced MLP train (100ep) ==="
export MLP_SEQ_CACHE_DIR="$(pwd)/${SEQ_CACHE}"
export MLP_MODEL_PATH="$(pwd)/${MLP_MODEL}"
pipenv run python3 -u src/notebooks/train_mlp_v2_warmth_sparse_skew_merged_ce_reduced.py

if [[ ! -f "$MLP_MODEL" ]]; then
  echo "ERROR: MLP checkpoint missing: ${MLP_MODEL}" >&2
  exit 1
fi
echo "MLP trained: ${MLP_MODEL}"

echo "=== Phase 3: hub9 sweep prep (MLP-only manifest) ==="
export SWEEP_DIR GNN_MODEL MLP_MODEL
bash scripts_cosim/important/prepare_warmth_sparse_skew_merged_hub9_sweep.sh

{
  head -1 "${SWEEP_DIR}/configs/jobs_smcr_hub9.tsv"
  grep '^mlp' "${SWEEP_DIR}/configs/jobs_smcr_hub9.tsv"
} > "${SWEEP_DIR}/configs/jobs_smcr_hub9_mlp.tsv"
mlp_jobs=$(($(wc -l < "${SWEEP_DIR}/configs/jobs_smcr_hub9_mlp.tsv") - 1))
if [[ "$mlp_jobs" -ne 9 ]]; then
  echo "ERROR: expected 9 MLP jobs, got ${mlp_jobs}" >&2
  exit 1
fi

pipenv run python3 - <<PY
import json
from pathlib import Path
meta_path = Path("${SWEEP_DIR}") / "configs" / "sweep_meta.json"
meta = json.loads(meta_path.read_text(encoding="utf-8"))
meta["mlp_model"] = "${MLP_MODEL}"
meta["mlp_retrain"] = "warmth_sparse_skew_merged_seq"
meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
PY

echo "=== Phase 4: rsync to datalab ==="
ssh -i "$SSH_KEY" "$REMOTE" "mkdir -p ${REPO}/models/tabular ${REPO}/${SWEEP_DIR}/configs ${REPO}/${SWEEP_DIR}/results ${REPO}/scripts_cosim/datalab ${REPO}/logs"

rsync -avP -e "$RSYNC_SSH" "$MLP_MODEL" "${REMOTE}:${REPO}/models/tabular/"
rsync -avP -e "$RSYNC_SSH" "${MLP_MODEL}.meta.json" "${REMOTE}:${REPO}/models/tabular/"
rsync -avP -e "$RSYNC_SSH" "${SWEEP_DIR}/configs/jobs_smcr_hub9_mlp.tsv" "${REMOTE}:${REPO}/${SWEEP_DIR}/configs/"
rsync -avP -e "$RSYNC_SSH" \
  scripts_cosim/datalab/submit_warmth_sparse_skew_merged_ce_reduced_hub9_mlp.sh \
  scripts_cosim/datalab/warmth_sparse_merged_ce_reduced_hub9_gpu.sbatch \
  scripts_cosim/datalab/run_warmth_sparse_merged_ce_reduced_hub9_one.sh \
  "${REMOTE}:${REPO}/scripts_cosim/datalab/"

ssh -i "$SSH_KEY" "$REMOTE" "chmod +x ${REPO}/scripts_cosim/datalab/*.sh && sed -i 's/\r$//' ${REPO}/scripts_cosim/datalab/*.sh ${REPO}/scripts_cosim/datalab/*.sbatch"

echo "=== Phase 5: submit MLP hub9 on datalab ==="
submit_out=$(ssh -i "$SSH_KEY" "$REMOTE" "cd ${REPO} && \
  export SWEEP_DIR='${SWEEP_DIR}' && \
  export MLP_MODEL='${MLP_MODEL}' && \
  export GNN_MODEL='${GNN_MODEL}' && \
  bash scripts_cosim/datalab/submit_warmth_sparse_skew_merged_ce_reduced_hub9_mlp.sh")
echo "$submit_out"

job_id=$(echo "$submit_out" | sed -n 's/^JOB_ID=//p')
if [[ -z "$job_id" ]]; then
  echo "ERROR: submit did not return JOB_ID" >&2
  exit 1
fi

echo "=== Phase 6: wait for datalab job ${job_id} ==="
while true; do
  state=$(ssh -i "$SSH_KEY" "$REMOTE" "sacct -j ${job_id} --format=State -n -P 2>/dev/null | head -1" || echo "UNKNOWN")
  pending=$(ssh -i "$SSH_KEY" "$REMOTE" "squeue -u nikola.lukic -j ${job_id} -h 2>/dev/null | wc -l" || echo 0)
  completed=$(ssh -i "$SSH_KEY" "$REMOTE" "sacct -j ${job_id} --format=State -n -P 2>/dev/null | grep -c COMPLETED || true")
  failed=$(ssh -i "$SSH_KEY" "$REMOTE" "sacct -j ${job_id} --format=State -n -P 2>/dev/null | grep -cE 'FAILED|TIMEOUT|CANCELLED' || true")
  echo "[$(date -Is)] state=${state} queue=${pending} completed=${completed} failed=${failed}"
  if [[ "$pending" -eq 0 ]] && [[ "$completed" -ge 9 ]]; then
    break
  fi
  if [[ "$failed" -gt 0 ]]; then
    echo "ERROR: datalab job failures detected" >&2
    ssh -i "$SSH_KEY" "$REMOTE" "sacct -j ${job_id} --format=JobID,State,ExitCode -n" || true
    exit 1
  fi
  sleep 60
done

echo "=== Phase 7: results table ==="
ssh -i "$SSH_KEY" "$REMOTE" "python3 - <<'PY'
import json
from pathlib import Path
from collections import defaultdict

res_dir = Path('${REPO}/${SWEEP_DIR}/results')
by_cfg = defaultdict(dict)
for p in sorted(res_dir.glob('*.json')):
    if 'decode_stats' in p.name:
        continue
    d = json.loads(p.read_text())
    rtt = d.get('total_rtt')
    if rtt is None:
        continue
    name = p.name
    if '_gnn_' in name:
        cfg = name.replace('_gnn_sparse_merged_ce_reduced.json','')
        by_cfg[cfg]['gnn'] = float(rtt)
    elif '_mlp_' in name:
        cfg = name.replace('_mlp_sparse_merged_ce_reduced.json','')
        by_cfg[cfg]['mlp'] = float(rtt)
    elif '_knative' in name:
        cfg = name.replace('_knative.json','')
        by_cfg[cfg]['kn'] = float(rtt)

configs = sorted(by_cfg.keys())
print('config\\tgnn\\tmlp(skew)\\tknative\\tmlp_vs_kn%\\tgnn_vs_kn%\\twinner')
sums = {'gnn':0,'mlp':0,'kn':0}
wins = defaultdict(int)
for cfg in configs:
    d = by_cfg[cfg]
    g,m,k = d.get('gnn'), d.get('mlp'), d.get('kn')
    if None in (g,m,k):
        print(f'{cfg}\\tMISSING {d}')
        continue
    dm = (m-k)/k*100
    dg = (g-k)/k*100
    winner = min([('gnn',g),('mlp',m),('kn',k)], key=lambda x:x[1])[0]
    wins[winner]+=1
    sums['gnn']+=g; sums['mlp']+=m; sums['kn']+=k
    print(f'{cfg}\\t{g/1e6:.3f}M\\t{m/1e6:.3f}M\\t{k/1e6:.3f}M\\t{dm:+.1f}%\\t{dg:+.1f}%\\t{winner}')
sk = sums['kn']
print('---')
print(f'SUM\\t{sums[\"gnn\"]/1e6:.3f}M\\t{sums[\"mlp\"]/1e6:.3f}M\\t{sums[\"kn\"]/1e6:.3f}M\\t{(sums[\"mlp\"]-sk)/sk*100:+.1f}%\\t{(sums[\"gnn\"]-sk)/sk*100:+.1f}%')
print('WINS', dict(wins))
PY"

echo "=== Done ==="
echo "MLP: ${MLP_MODEL}"
echo "Log: ${LOG}"
