#!/bin/bash
# One-state smoke of the fresh-100 lock-wait path: ILS label + W-horizon hand and ILS runs, then `check lockwait`.
#   sbatch --export=ALL,WT=<pinned worktree>,KEY=<state key> uncut_arm1_lockwait_smoke.sbatch
#SBATCH --job-name=a1-lwsmoke
#SBATCH --output=/share/nikola.lukic/uncut_arm1/test/lwsmoke-%j.out
#SBATCH --partition=CPU-amd
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
set -euo pipefail
WT=${WT:?FAIL LOUD: set WT}; KEY=${KEY:?FAIL LOUD: set KEY}
eval "$(micromamba shell hook --shell bash)"; micromamba activate gnn
cd "$WT"; export PYTHONPATH=$PWD HEROSIM_PY=python3; git rev-parse HEAD
OUT=/share/nikola.lukic/uncut_arm1/test/lwsmoke_${SLURM_JOB_ID}.jsonl
python3 - <<P
import json
from scripts_cosim import uncut_arm1 as A, uncut_horizon as H
st = [s for s in json.load(open("/share/nikola.lukic/uncut_arm1/arm1_state_ids.json"))["states"] if s["key"] == "$KEY"][0]
job = A.job_of(st, "/share/nikola.lukic/uncut_arm1")
lab = A.label_state(job, 0, None)
assert lab.get("r") is not None, lab["source"]
H.HORIZON_SPECS = ("W",)
hz = H.run_state(dict(job), {"hand": lab["hand"], "ils": lab["ils"]})
row = {"state_id": st["key"], "bucket": st["bucket"], "source": lab["source"], "r": lab.get("r")}
row["lock_wait"] = {n: {kk: vv["lock_wait"] for kk, vv in p["by_n"].items()} for n, p in hz["plans"].items() if p}
print(json.dumps(row))
import math
for n in ("hand", "ils"):
    for who in ("batch", "later"):
        x = row["lock_wait"][n]["W"][who]; print(n, who, x); assert math.isfinite(x["sum"]) and x["n"] > 0
open("$OUT", "w").write(json.dumps(row) + "\n")
P
python3 scripts_cosim/uncut_arm1.py check lockwait $OUT
