"""Label-horizon ranking test. For each dataset: the isolated argmin plan, the next K plans by isolated label, 2 random plans; each executed
live (HEROSIM_FORCED_PLACEMENTS on the batch, capture policy, later arrivals placed by the policy) isolated (arrivals <= decision time) and
admitted (arrivals <= decision time + W, W covering the batch's completion).
usage: ranking.py OUTDIR DATASETROOT[,DATASETROOT..] [--max N] [--workers W] [--next K]
env: run from a worktree with the R1 flags exported (see ranking.sbatch)."""
import sys, os, json, bisect, random, subprocess, math, glob, argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from src.placement.sweep_status import sweep_complete

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("roots")
ap.add_argument("--select", default="", help="file with 'root/ds_id' per line; default all datasets with a full sweep")
ap.add_argument("--max", type=int, default=0); ap.add_argument("--workers", type=int, default=16)
ap.add_argument("--next", type=int, default=3); ap.add_argument("--random", type=int, default=2)
ap.add_argument("--policy", default="peer_greedy_network_batch")
ap.add_argument("--quota", default="", help="rung substring:count[,..] e.g. hi:20,mid:10; round-robin over topologies, multi-node optima first")
ap.add_argument("--gate-only", action="store_true", help="only isolated runs (gate 3)")
args = ap.parse_args()
OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
REPO = Path(".").resolve()

def load(ds):
    ds = Path(ds)
    prov = json.load(open(ds / "warm_snapshot.json"))["provenance"]
    plans = []
    for i, l in enumerate(open(ds / "placements" / "placements.jsonl")):
        r = json.loads(l)
        if "placement_plan" in r: plans.append((r["placement_plan"], r["rtt"]))
    fid = json.load(open(ds / "fidelity_replay.json"))
    return prov, plans, fid

def key(plan): return json.dumps(sorted((int(k), tuple(v)) for k, v in plan.items()))

def pick(plans, k_next, k_rand, seed):
    order = sorted(range(len(plans)), key=lambda i: plans[i][1])
    chosen = order[:1 + k_next]
    rest = order[1 + k_next:]
    random.Random(seed).shuffle(rest)
    return chosen + rest[:k_rand], order

def run(cfg, wl, task_ids, plan, cut_n, tag):
    f = OUT / "runs" / (tag + ".json"); f.parent.mkdir(exist_ok=True)
    env = dict(os.environ, HEROSIM_MAX_EVENTS=str(cut_n),
               HEROSIM_FORCED_PLACEMENTS=json.dumps({str(task_ids[int(i)]): list(v) for i, v in plan.items()}))
    for k in ("HEROSIM_AUDIT_TRACE", "LIVE_AUDIT_SNAPSHOT_PATH"): env.pop(k, None)
    p = subprocess.run([sys.executable, "src/executesimulation.py", "--config", cfg, "--workload", wl, "--policy", args.policy, "--output", str(f)],
                       env=env, cwd=str(REPO), capture_output=True, text=True)
    applied = p.stderr.count("FORCED_BATCH_APPLIED") + p.stdout.count("FORCED_BATCH_APPLIED")
    if p.returncode != 0 or not f.exists():
        return dict(error=("rc %d: " % p.returncode) + (p.stderr[-300:] or p.stdout[-300:]).replace("\n", " "), applied=applied)
    res = json.load(open(f)); f.unlink()
    trs = {t["taskId"]: t for t in res["stats"]["taskResults"] if t.get("taskId", -1) >= 0}
    return dict(applied=applied, tr={t: (trs[t]["scheduledTime"], trs[t]["doneTime"]) for t in task_ids if t in trs})

def evaluate(job):
    ds, pi, plan, label, prov, t0, fid = job
    cfg, wl, tids, t = prov["cell_config"], prov["trace"], prov["task_ids"], prov["snapshot_time"]
    arr = [float(e["timestamp"]) for e in json.load(open(wl))["events"]]
    tag = "%s_%d" % (Path(ds).name, pi)
    iso = run(cfg, wl, tids, plan, bisect.bisect_right(arr, t), tag + "_iso")
    row = dict(ds=str(ds), plan_idx=pi, label=label)
    if "error" in iso: row["iso_error"] = iso["error"]; return row
    if iso["applied"] != 1 or len(iso["tr"]) != len(tids): row["iso_error"] = "applied=%d tasks=%d" % (iso["applied"], len(iso["tr"])); return row
    if any(abs(iso["tr"][x][0] - t) > 1e-6 for x in tids): row["iso_error"] = "batch scheduled at a different instant"; return row
    row["iso"] = sum(iso["tr"][x][1] - iso["tr"][x][0] for x in tids)
    if args.gate_only: return row
    W = max(5.0, 2 * max(iso["tr"][x][1] - iso["tr"][x][0] for x in tids) + 1)
    for _ in range(4):
        adm = run(cfg, wl, tids, plan, bisect.bisect_right(arr, t + W), tag + "_adm")
        if "error" in adm: row["adm_error"] = adm["error"]; return row
        if all(adm["tr"][x][1] <= t + W for x in tids): break
        W *= 3
    else:
        row["adm_error"] = "batch not complete within W=%.0f" % W; return row
    row["adm"] = sum(adm["tr"][x][1] - adm["tr"][x][0] for x in tids); row["W"] = W
    row["extra_arrivals"] = bisect.bisect_right(arr, t + W) - bisect.bisect_right(arr, t)
    return row

# datasets
dss = []
if args.select: dss = [Path(l.strip()) for l in open(args.select) if l.strip()]
else:
    for r in args.roots.split(","): dss += sorted(Path(r).glob("ds_*"))
dss = [d for d in dss if sweep_complete(d)]   # a half-finished sweep has the wrong argmin
if args.quota:
    def info(d):
        prov, plans, _ = load(d)
        order = sorted(range(len(plans)), key=lambda i: plans[i][1])
        return prov["cell_config"], prov.get("cell_seed"), len({v[0] for v in plans[order[0]][0].values()}) == 1
    cand = {}
    for d in dss:
        try: cfgp, seed, single = info(d)
        except Exception as e: print("skip", d, e); continue
        cand.setdefault(cfgp, []).append((d, seed, single))
    picked = []
    for q in args.quota.split(","):
        sub, n = q.split(":"); n = int(n)
        pool = [x for c, xs in cand.items() if sub in c for x in xs]
        by_seed = {}
        for x in pool: by_seed.setdefault(x[1], []).append(x)
        multi = []; single = []
        for seed in sorted(by_seed):
            multi.append([x for x in by_seed[seed] if not x[2]]); single.append([x for x in by_seed[seed] if x[2]])
        def rr(lists):
            out = []; i = 0
            while any(lists):
                for l in lists:
                    if l: out.append(l.pop(0))
            return out
        m, s_ = rr(multi), rr(single); sel = []
        while len(sel) < n and (m or s_):
            if m: sel.append(m.pop(0))
            if len(sel) < n and s_: sel.append(s_.pop(0))
        print("quota", sub, "wanted", n, "got", len(sel), "multi-node", sum(1 for x in sel if not x[2]), "topologies", len({x[1] for x in sel}), flush=True)
        picked += [x[0] for x in sel]
    dss = picked
jobs = []; meta = {}
for ds in dss:
    try: prov, plans, fid = load(ds)
    except Exception as e: print("skip", ds, e); continue
    if len(plans) < 2: continue
    chosen, order = pick(plans, args.next, args.random, hash(ds.name) % 1000)
    nodes = lambda pl: len({v[0] for v in pl.values()})
    meta[str(ds)] = dict(tasks=len(prov["task_ids"]), plans=len(plans), single_node_opt=nodes(plans[order[0]][0]) == 1,
                         open_peers=sum(1 for rows in (fid.get("fidelity") or {}).get("peers", {}).values() for r in rows if r[1] is None) if isinstance(fid.get("fidelity"), dict) else None)
    for pi in chosen: jobs.append((ds, pi, plans[pi][0], plans[pi][1], prov, 0, fid))
    if args.max and len(meta) >= args.max: break
print(len(meta), "datasets,", len(jobs), "plan evaluations", flush=True)
with ThreadPoolExecutor(args.workers) as ex: rows = list(ex.map(evaluate, jobs))
json.dump(dict(meta=meta, rows=rows), open(OUT / "ranking_rows.json", "w"))

# gate 3 + report
def rel(a, b): return abs(a - b) / b if b else float("inf")
g3 = [rel(r["iso"], r["label"]) for r in rows if "iso" in r]
bad = [r for r in rows if "iso" not in r]
def q(v, p): v = sorted(v); return v[min(len(v) - 1, math.ceil(p * len(v)) - 1)] if v else float("nan")
print("GATE3: evaluated %d, failed to run/match %d, forced-iso vs label: median %.4f%% p95 %.4f%% max %.4f%%" % (len(g3), len(bad), 100 * q(g3, .5), 100 * q(g3, .95), 100 * max(g3) if g3 else float("nan")))
for r in sorted([r for r in rows if "iso" in r], key=lambda r: -rel(r["iso"], r["label"]))[:5]:
    print("   worst", Path(r["ds"]).name, r["plan_idx"], "label %.4f iso %.4f abs miss %.4f" % (r["label"], r["iso"], abs(r["iso"] - r["label"])))
for r in bad[:8]: print("   FAIL", Path(r["ds"]).name, r["plan_idx"], r.get("iso_error"))
json.dump(dict(gate3_median=q(g3, .5), gate3_p95=q(g3, .95), gate3_max=max(g3) if g3 else None, failed=len(bad)), open(OUT / "gate3.json", "w"))
if args.gate_only: sys.exit(0)

def ranks(v):
    o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]: j += 1
        for k in range(i, j + 1): r[o[k]] = (i + j) / 2 + 1
        i = j + 1
    return r
def spear(a, b):
    ra, rb = ranks(a), ranks(b); n = len(a); ma, mb = sum(ra) / n, sum(rb) / n
    sa = math.sqrt(sum((x - ma) ** 2 for x in ra)); sb = math.sqrt(sum((x - mb) ** 2 for x in rb))
    return sum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / (sa * sb) if sa and sb else float("nan")
by = {}
for r in rows: by.setdefault(r["ds"], []).append(r)
res = []
for ds, rs in by.items():
    ok = [r for r in rs if "iso" in r and "adm" in r]
    first = [r for r in rs if r["plan_idx"] == min(x["plan_idx"] for x in rs)]
    # the isolated argmin of the evaluated set, by the forced isolated value
    if len(ok) < 2: continue
    a = min(ok, key=lambda r: r["iso"]); b = min(ok, key=lambda r: r["adm"])
    res.append(dict(ds=ds, n=len(ok), agree=a["adm"] <= b["adm"] * (1 + 1e-9), regret=(a["adm"] - b["adm"]) / b["adm"],
                    rho=spear([r["iso"] for r in ok], [r["adm"] for r in ok]), single=meta[ds]["single_node_opt"], open_peers=meta[ds]["open_peers"]))
json.dump(res, open(OUT / "ranking_result.json", "w"))
def report(name, rr):
    if not rr: print(name, "n=0"); return
    reg = [x["regret"] for x in rr]; rho = [x["rho"] for x in rr if x["rho"] == x["rho"]]
    print("%s: datasets %d  argmin agreement %.1f%%  regret median %.3f%% p90 %.3f%% max %.3f%%  Spearman median %.3f (mean %.3f)  with open peers at the decision: %d" % (
        name, len(rr), 100 * sum(x["agree"] for x in rr) / len(rr), 100 * q(reg, .5), 100 * q(reg, .9), 100 * max(reg), q(rho, .5), sum(rho) / len(rho) if rho else float("nan"),
        sum(1 for x in rr if x["open_peers"])))
report("ALL", res); report("single-node optimum", [x for x in res if x["single"]]); report("multi-node optimum", [x for x in res if not x["single"]])
