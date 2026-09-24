#!/bin/bash
# drainable_objective_v1 -- Phase B offline artifact: read every training run's curves
# against their chance floors, so the shaped-label runs are on the record as measured and
# not as they look on the W&B charts (CLAUDE.md step 5, docs/lessons.md "Read a finished
# run's curves against their chance floor").
#
# Not a bar and not a gate. It produces one full report per run plus a compact TSV, both
# under simulation_data/drainable_objective_v1/curve_reads/.
#
#   sbatch scripts_cosim/datalab/drainable_objective_v1_curve_read.sbatch
#   RUN_GLOB='wandb/run-20260915_1*' sbatch scripts_cosim/datalab/...
#
#SBATCH --job-name=dobj-curves
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/dobj-curves-%j.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/dobj-curves-%j.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail
cd /home/nikola.lukic/gnn-herosim

if ! command -v micromamba >/dev/null 2>&1; then source "${HOME}/.bashrc"; fi
eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PWD"
export PYTHONUNBUFFERED=1

python3 -c "import sys,torch;print('[INTERP]',sys.executable,torch.__version__)"
echo "[CODE] $(git describe --dirty --always) on $(git rev-parse --abbrev-ref HEAD)"

RUN_GLOB="${RUN_GLOB:-wandb/run-20260915_1*}"
OUT_DIR="${OUT_DIR:-simulation_data/drainable_objective_v1/curve_reads}"
mkdir -p "$OUT_DIR"

# The run's own name is not in the wandb directory name; it is in output.log.
N=0
for d in $RUN_GLOB; do
  [[ -d "$d" ]] || continue
  name=$(grep -m1 -o "drainable-objective-v1-[a-z0-9-]*" "$d/files/output.log" 2>/dev/null || true)
  [[ -n "$name" ]] || { echo "[skip] $d: no drainable-objective run name in output.log"; continue; }
  ${HEROSIM_PY} scripts_cosim/read_training_curves.py "$d" > "$OUT_DIR/${name}.txt" 2>/dev/null
  echo "[read] $name  <- $d"
  N=$((N + 1))
done
[[ $N -gt 0 ]] || { echo "FAIL LOUD: no runs matched RUN_GLOB=$RUN_GLOB" >&2; exit 1; }
echo "[read] $N runs"

${HEROSIM_PY} - "$OUT_DIR" <<'PY'
import glob, json, os, re, sys

out_dir = sys.argv[1]
rows = []
for path in sorted(glob.glob(os.path.join(out_dir, "drainable-objective-v1-*.txt"))):
    text = open(path).read()
    name = os.path.basename(path)[:-4]

    def grab(pattern, cast=float, default=None):
        m = re.search(pattern, text)
        return cast(m.group(1)) if m else default

    rows.append({
        "run": name,
        "arm": "gnn" if "-gnn-" in name else ("mpoff" if "-mpoff-" in name else "?"),
        "seed": int(re.search(r"seed(\d+)", name).group(1)),
        "epochs": grab(r"TRAINING CURVE READ -- (\d+) epochs", int),
        "selector": grab(r"checkpoint metric looks like (\S+);", str),
        "selected_epoch": grab(r"best at epoch (\d+) of", int),
        "pct_after_selected": grab(r"(\d+) epochs \(\d+%\) ran after", int),
        # selected value = the selector curve's minimum
        "selected_regret": grab(r"val/regret_masked_topo\s+[\d.,]+\s+([\d.,]+)@\d+",
                                lambda s: float(s.replace(",", ""))),
        "task_acc_best": grab(r"val/task_acc\s+start=[\d.e+-]+ best=([\d.e+-]+)", float),
        "task_acc_chance": grab(r"val/task_acc\s+start=[\d.e+-]+ best=[\d.e+-]+ chance=([\d.e+-]+)", float),
        "ce_best": grab(r"val/ce\s+start=[\d.e-]+ best=([\d.e-]+)", float),
        "ce_chance": grab(r"baseline/val_chance_ce\s+= ([\d.,]+)",
                          lambda s: float(s.replace(",", ""))),
        "ce_last": grab(r"val/ce\s+[\d.,]+\s+[\d.,]+@\d+\s+[\d.,]+@\d+\s+([\d.,]+)",
                        lambda s: float(s.replace(",", ""))),
        "memorised": "the run memorised" in text,
    })

if not rows:
    raise SystemExit("FAIL LOUD: no per-run reports to summarise")

tsv = os.path.join(out_dir, "curve_read_table.tsv")
cols = list(rows[0])
with open(tsv, "w") as fh:
    fh.write("\t".join(cols) + "\n")
    for r in sorted(rows, key=lambda r: (r["arm"], r["seed"])):
        fh.write("\t".join("" if r[c] is None else str(r[c]) for c in cols) + "\n")

json.dump(rows, open(os.path.join(out_dir, "curve_read_table.json"), "w"), indent=1)

for arm in ("gnn", "mpoff"):
    vals = sorted(r["selected_regret"] for r in rows if r["arm"] == arm and r["selected_regret"])
    eps = sorted(r["selected_epoch"] for r in rows if r["arm"] == arm and r["selected_epoch"])
    if not vals:
        continue
    mid = vals[len(vals) // 2]
    print(f"[{arm:5s}] n={len(vals):2d}  selected val/regret_masked_topo "
          f"min={vals[0]:.3f} median={mid:.3f} max={vals[-1]:.3f}   "
          f"selected epoch {eps[0]}-{eps[-1]}")
# Every run's CE floor must be the same number, or the arms are not on one corpus.
floors = {r["ce_chance"] for r in rows}
print(f"[floor] val_chance_ce = {floors}  (one value expected)")
above = [r["run"] for r in rows if r["ce_last"] and r["ce_chance"] and r["ce_last"] > r["ce_chance"]]
print(f"[floor] runs whose LAST val/ce is above chance: {len(above)}  {above[:4]}")
print(f"[wrote] {tsv}")
PY
echo "=== curve read done ==="
