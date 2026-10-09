"""Forced-live isolated values for chosen plans of fidelity-replay datasets (the live truth the replay labels are compared with).
usage (from a worktree with the forced-batch hook): forced_live_plans.py OUT.json --relabel RELABEL.json --datasets LIST [--best 3 --random 3 --workers 40]
Plans are taken from a relabel_plans.py output (so every variant's label exists for them): the best `--best` by stored label plus `--random`."""
import argparse, bisect, json, os, random, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, os.getcwd())
from scripts_cosim.ranking_mapping import plan_task_ids

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--relabel", required=True); ap.add_argument("--datasets", required=True)
ap.add_argument("--best", type=int, default=3); ap.add_argument("--random", type=int, default=3); ap.add_argument("--workers", type=int, default=40)
a = ap.parse_args()
REPO = Path(".").resolve()
want = {l.strip() for l in open(a.datasets) if l.strip()}
rel = {x["ds"]: x for x in json.load(open(a.relabel)) if "error" not in x and x["ds"] in want}
jobs = []
for ds, x in rel.items():
    rows = sorted(x["rows"], key=lambda r: r["old"])
    pick = rows[: a.best]; rest = rows[a.best:]
    random.Random(sum(map(ord, Path(ds).name))).shuffle(rest)
    for r in pick + rest[: a.random]:
        jobs.append((ds, r["plan_idx"]))


def run(job):
    ds, pi = job
    d = Path(ds)
    try:
        ws = json.load(open(d / "warm_snapshot.json")); prov = ws["provenance"]
        tids = plan_task_ids(json.load(open(d / "workload.json"))["events"], ws["snapshot"]["fidelity"]["batch"])
        plans = [r for r in (json.loads(l) for l in open(d / "placements" / "placements.jsonl")) if "placement_plan" in r]
        plan = plans[pi]["placement_plan"]
        arr = [float(e["timestamp"]) for e in json.load(open(prov["trace"]))["events"]]
        n = bisect.bisect_right(arr, prov["snapshot_time"])
        out = REPO / "_fl" / f"{d.name}_{pi}.json"; out.parent.mkdir(exist_ok=True)
        env = dict(os.environ, GNN_DECODE_MODE="masked_topo", GNN_BATCH_BY_PEER_GROUP="1", HEROSIM_MAX_EVENTS=str(n),
                   HEROSIM_FORCED_PLACEMENTS=json.dumps({str(tids[int(i)]): list(v) for i, v in plan.items()}))
        for k in ("HEROSIM_AUDIT_TRACE", "LIVE_AUDIT_SNAPSHOT_PATH"):
            env.pop(k, None)
        p = subprocess.run([sys.executable, "src/executesimulation.py", "--config", prov["cell_config"], "--workload", prov["trace"],
                            "--policy", "peer_greedy_network_batch", "--output", str(out)], env=env, cwd=str(REPO), capture_output=True, text=True)
        applied = (p.stderr + p.stdout).count("FORCED_BATCH_APPLIED")
        if p.returncode != 0 or not out.exists():
            return dict(ds=ds, plan_idx=pi, error=((p.stderr or p.stdout)[-200:]).replace("\n", " "))
        res = json.load(open(out)); out.unlink()
        trs = {t["taskId"]: t for t in res["stats"]["taskResults"] if t.get("taskId", -1) >= 0}
        if applied != 1 or any(t not in trs for t in tids):
            return dict(ds=ds, plan_idx=pi, error=f"applied={applied}")
        return dict(ds=ds, plan_idx=pi, live=sum(trs[t]["doneTime"] - trs[t]["scheduledTime"] for t in tids))
    except Exception as e:  # recorded
        return dict(ds=ds, plan_idx=pi, error=f"{type(e).__name__}: {str(e)[:150]}")


with ThreadPoolExecutor(a.workers) as ex:
    res = list(ex.map(run, jobs))
json.dump(res, open(a.out, "w"))
print("forced-live plans:", len(res), "ok", sum(1 for r in res if "live" in r), "errors", sum(1 for r in res if "error" in r))
