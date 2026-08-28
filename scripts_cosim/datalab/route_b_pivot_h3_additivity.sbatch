#!/bin/bash
#SBATCH --job-name=rbh3-addit
#SBATCH --output=/home/nikola.lukic/gnn-herosim/logs/route-b-pivot-h3-additivity-%A.out
#SBATCH --error=/home/nikola.lukic/gnn-herosim/logs/route-b-pivot-h3-additivity-%A.err
#SBATCH --partition=CPU-amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# route_b_env_pivot_v1 — measure additivity DIRECTLY on the H3 separable control.
#
# WHY THIS AND NOT S0. The amended H2's control failed S0 by 24x and it took a session to
# attribute, because S0 reads additivity THROUGH a decoder and a cap. The direct
# measurement (regress rtt on one-hot (task, platform) over the full exhaustive sweep, no
# decoder / cap / surrogate in the path) settled it in minutes: H2 ctrl R^2 median
# 0.78/0.82 and residual 12.8%/13.8%, against H0/H1 ctrl R^2 0.99997+ and residual
# 0.16-1.0%. H3 inherits H2's grid and packs 8 tasks onto the same 2 hosting nodes, so its
# control is under the same doubt and the co-location density only goes UP.
#
# THIS JOB READS NO BAR. S0 as registered is not computed here and no S1-S4 number is
# touched. AMENDMENT 4 (which would make this measurement the S0 statistic) is DRAFTED AND
# NOT SIGNED OFF -- the number this produces is a diagnostic, not a rung verdict, whatever
# it says.
#
# Compute node, not the login node: this streams ~204 x 40,320 rows of placements.jsonl,
# roughly 13x the H2 run, and is exactly the shape the login-node watchdog kills silently
# (datalab-pitfalls #2).
#
# Generation integrity is asserted FIRST and the job dies if it fails: 204/204
# sweep_complete with the num_placements histogram, per AMENDMENT 3 section 7. SUCCESS
# counts and absent skip files are not evidence of a usable sweep.

set -euo pipefail
PROJECT_ROOT="/home/nikola.lukic/gnn-herosim"
cd "$PROJECT_ROOT"

eval "$(micromamba shell hook --shell bash)"
micromamba activate gnn
export HEROSIM_PY=python3
export PYTHONPATH="$PROJECT_ROOT"

# Prove which interpreter actually served, rather than trusting this banner
# (datalab-pitfalls #8, and herosim-provenance-beats-banners).
python3 -c "import sys, numpy; print('interpreter:', sys.executable); print('numpy:', numpy.__version__)"

CORPUS=simulation_data/gnn_datasets_route_b_pivot_h3_ctrl

echo "=== generation integrity (fails loud before anything is measured) ==="
python3 -u - "$CORPUS" <<'PYEOF'
import collections, glob, json, os, sys

corpus = sys.argv[1]
dirs = sorted(glob.glob(os.path.join(corpus, "ds_*")))
hist, bad, missing = collections.Counter(), [], []
counters = collections.Counter()
for d in dirs:
    mp = os.path.join(d, "placement_metadata.json")
    if not os.path.exists(mp):
        missing.append(os.path.basename(d))
        continue
    m = json.load(open(mp))
    if m.get("sweep_complete") is not True:
        bad.append((os.path.basename(d), m.get("rows_written"), m.get("num_placements")))
    hist[m.get("num_placements")] += 1
    for k in ("timed_out", "worker_failed", "worker_exception"):
        counters[k] += m.get(k, 0)
    counters["early_terminated"] += bool(m.get("early_terminated"))

skips = sum(os.path.exists(os.path.join(d, "skip_reason.json")) for d in dirs)
nojl = sum(not os.path.exists(os.path.join(d, "placements", "placements.jsonl")) for d in dirs)
print(f"datasets            : {len(dirs)}")
print(f"num_placements      : {dict(hist)}")
print(f"sweep_complete true : {len(dirs) - len(bad) - len(missing)}/{len(dirs)}")
print(f"skip files          : {skips}   missing placements.jsonl: {nojl}")
print(f"worker counters     : {dict(counters)}")
if bad or missing or skips or nojl:
    print("OFFENDERS:", bad[:10], missing[:10])
    raise SystemExit("FAIL LOUD: H3 control corpus is not clean; refusing to measure it")
print("integrity OK")
PYEOF

echo
echo "=== direct additivity measurement ==="
python3 -u scripts_cosim/measure_route_b_additivity.py \
  --corpus "$CORPUS" \
  --co-residency --pairwise \
  --out simulation_data/route_b_pivot_h3_ctrl_additivity.json

echo
echo "DONE. Calibration to compare against (from the frozen control corpora):"
echo "  H0 ctrl (S0 PASS)  R^2 median 0.999991 / 0.999757   residual 0.159% / 0.721%"
echo "  H1 ctrl (S0 PASS)  R^2 median 0.999975 / 0.999856   residual 0.193% / 0.999%"
echo "  H2 ctrl (S0 FAIL)  R^2 median 0.7826   / 0.8161     residual 12.822% / 13.814%"
