#!/bin/bash
#SBATCH --job-name=s7-id-ils
#SBATCH --partition=CPU-amd
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --exclude=os-cpu-slurm-21,os-cpu-slurm-15,os-cpu-slurm-13,os-cpu-slurm-14,os-cpu-slurm-16,os-cpu-slurm-10,os-cpu-slurm-17,os-cpu-slurm-18,os-cpu-slurm-19,os-cpu-slurm-20,os-cpu-slurm-22
#SBATCH --output=/share/nikola.lukic/uncut_groups/ils_ident/logs/%x-%j.out
# identity: uncut_ils.py run on one filed K1 job (heavy_g1_07, n=34, budget-limited) at the merged pin vs its filed row (rows_s0.jsonl, made at 98febf7d), same flags
set -euo pipefail
eval "$(micromamba shell hook --shell bash)"; micromamba activate gnn
WT=${WT:?}; cd $WT; export PYTHONPATH=$PWD HEROSIM_PY=python3
[[ $(git rev-parse HEAD) == ${PIN:?}* ]] || { echo "FAIL LOUD: HEAD"; exit 2; }
K=/share/nikola.lukic/uncut_ils/k1b; O=/share/nikola.lukic/uncut_groups/ils_ident/ils; mkdir -p $O
python3 - <<'P'
import json
K="/share/nikola.lukic/uncut_ils/k1b"; O="/share/nikola.lukic/uncut_groups/ils_ident/ils"
job=[json.loads(l) for l in open(K+"/jobs_s0.jsonl") if json.loads(l)["state_id"]=="heavy_g1_07"]
assert len(job)==1; open(O+"/job.jsonl","w").write(json.dumps(job[0])+"\n")
P
python3 scripts_cosim/uncut_ils.py run $O/job.jsonl $O/rows.jsonl --arm cd_ib0 --seed 0 --replay all --max-seconds 300
python3 - <<'P'
import json
K="/share/nikola.lukic/uncut_ils/k1b"; O="/share/nikola.lukic/uncut_groups/ils_ident/ils"
WALL={"search_wall","wall","replay_wall","replay_setup","replay_run_mean","s_per_call_ms"}
f=[json.loads(l) for l in open(K+"/rows_s0.jsonl") if json.loads(l)["state_id"]=="heavy_g1_07"]; r=[json.loads(l) for l in open(O+"/rows.jsonl")]
assert len(f)==1 and len(r)==1, (len(f),len(r))
strip=lambda d: json.dumps({k:v for k,v in d.items() if k not in WALL}, sort_keys=True)
a,b=strip(f[0]),strip(r[0])
bad=[k for k in sorted(set(f[0])|set(r[0])) if k not in WALL and json.dumps(f[0].get(k),sort_keys=True)!=json.dumps(r[0].get(k),sort_keys=True)]
print("keys compared", len(set(f[0])-WALL), "wall-clock keys excluded", sorted(WALL), "differing", bad)
print("PASS" if a==b else "FAIL")
raise SystemExit(0 if a==b else 1)
P
