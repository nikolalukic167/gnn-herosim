#!/usr/bin/env python3
"""DAgger r1 smoke: dry-run label of a few states captured from a learned arm's cell (uncut_groups_s0_v1). Picks the first state of each size class from the
cell's snapshot file, labels it through uncut_arm1.label_state with check_arm = False and the given budget (PER), and prints the wall per state.
usage: dagger_smoke_label.py <stem> <cfg.json> <per> [seed]   (stem = cell path without .snapshots.jsonl / .summary.json)"""
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts_cosim import uncut_arm1 as A  # noqa: E402

stem, cfg, per = sys.argv[1], sys.argv[2], int(sys.argv[3])
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
t0 = time.time()
want, picked = {"le10": (2, 10), "11_50": (11, 50), "51_100": (51, 256)}, {}
with open(stem + ".snapshots.jsonl") as fh:
    for line in fh:
        rec = json.loads(line)
        n = int(rec["horizon"])
        for b, (lo, hi) in want.items():
            if b not in picked and lo <= n <= hi:
                picked[b] = (int(rec["trigger_task_id"]), float(rec["time"]), n)
        if len(picked) == len(want):
            break
print(f"[smoke] scan {time.time() - t0:.1f} s; picked {picked}", flush=True)
for b, (trig, tm, n) in picked.items():
    job = {"state_id": f"smoke#{trig}", "bucket": b, "group_size": n, "snapfile": stem + ".snapshots.jsonl", "trigger_task_id": trig, "time": tm,
           "cell_summary": stem + ".summary.json", "cell_config": cfg, "sim_input": str(REPO / "data" / "nofs-ids"), "role": "first", "check_arm": False}
    t1 = time.time()
    lab = A.label_state(job, seed, None, per=per)
    print(f"[smoke] {b} n={n} trigger={trig}: source={lab.get('source')} r={lab.get('r')} n_evals={lab.get('n_evals')} budget={lab.get('budget')} stopped={lab.get('stopped')} "
          f"search_wall={lab.get('search_wall')} chosen_match={lab.get('chosen_match')} per={lab.get('per')} check_arm={lab.get('check_arm')} reason={lab.get('reason')} "
          f"error={lab.get('error')} | total wall {time.time() - t1:.1f} s", flush=True)
