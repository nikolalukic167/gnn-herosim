"""scale_160_v1: CD-steered log bisection of the arrival multiplier on scaled WF1 cells, after load_recalibration_v1.
Per step: mint the WF1 inputs at multiplier m for the calibration topologies (W2 payloads + W4 mix + single origin, W3 classes, the
probe's client spread), run CD on topologies x windows in parallel (physics_audit/run_cell.sh, full stats), take the median EFFECTIVE
queue share ((queue + lock wait) / elapsed, gate reader's lock_wait_profile) and steer: below the band -> up, above -> down.
Guards per CD cell (every cell must pass for the step to count): request failures <= 1 % of tasks, p95 latency <= 300 s, run end <= 1.25 x
last arrival, finished within the steering limit. Fallback when the band is not reached: the highest passing step below it.
usage: scale_rung_bisect.py ROOT --wt WT --seeds 9905 9906 ... --windows g0 g1 --clients 160 --servers 24 --p 0.6
       --band 0.40 0.50 --lo M --hi M --steps 8 --tag heavy [--knative]   (writes ROOT/<tag>/steps.json and ROOT/<tag>/result.json)"""
import argparse, json, math, os, statistics as st, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("root"); ap.add_argument("--wt", required=True); ap.add_argument("--seeds", type=int, nargs="+", required=True)
ap.add_argument("--windows", nargs="+", default=["g0", "g1"]); ap.add_argument("--clients", type=int, default=160)
ap.add_argument("--servers", type=int, default=24); ap.add_argument("--p", type=float, default=0.6)
ap.add_argument("--band", type=float, nargs=2, required=True); ap.add_argument("--lo", type=float, required=True); ap.add_argument("--hi", type=float, required=True)
ap.add_argument("--steps", type=int, default=8); ap.add_argument("--tag", required=True); ap.add_argument("--limit", type=int, default=2700)
ap.add_argument("--parallel", type=int, default=8); ap.add_argument("--knative", action="store_true", help="run knative_network as context at the chosen rung")
ap.add_argument("--extra-policies", nargs="*", default=[], help="descriptive arms run at the chosen rung on the same cells (e.g. peer_greedy_network_batch peer_greedy_selfpredict_network)")
a = ap.parse_args()
ROOT = Path(a.root); WT = Path(a.wt); OUT = ROOT / a.tag; OUT.mkdir(parents=True, exist_ok=True)
X1 = "/home/nikola.lukic/gnn-herosim/simulation_data/small_batch_confirm_v1/inputs/grounded/wl"
BASECFG = "/home/nikola.lukic/gnn-herosim/simulation_data/workload_fix_v1/corpus_prod_prov/cfg_src/cc40s9101.json"
sys.path.insert(0, str(WT))
from scripts_cosim.fresh_topo_burst_v1_gate import lock_wait_profile, latency_percentiles, backlog_profile  # noqa: E402

CFG = ROOT / "cfg_src"
if not CFG.is_dir():
    for s in a.seeds:
        subprocess.run([sys.executable, "scripts_cosim/scale_probe_cfg.py", BASECFG, str(CFG), "--clients", str(a.clients), "--servers", str(a.servers),
                        "--p", str(a.p), "--seed", str(s)], cwd=WT, check=True, stdout=subprocess.DEVNULL)


def mtag(m): return "m" + ("%.4f" % m).replace(".", "p")


def mint(m):
    tag = mtag(m); inp = ROOT / "inputs"; inp.mkdir(exist_ok=True)
    if (inp / f"wf1_{tag}").is_dir():
        return tag
    # ONE convention: m is the FINAL arrival multiplier on the x1 windows (factor = 1/m), the same number the gate's RUNGS take.
    # (The probe's sbatch derived its multiplier from the production rung as m_prod * servers/6; the first (c) driver applied that
    # scaling twice and ran CD at 4x the capture's load -- coordinator check 2026-10-10, retracted.)
    factor = 1.0 / m
    b = ROOT / f"_b_{tag}"; w24 = ROOT / f"_w24_{tag}"; w24.mkdir(exist_ok=True)
    subprocess.run([sys.executable, "scripts_cosim/workload_fix_v1_build.py", "--grounded-wl", X1, "--cfg-dir", str(CFG), "--topologies", *map(str, a.seeds),
                    "--rung", f"{tag}={factor!r}", "--payload-sampler", "wf1_v1", "--task-mix", "wf1_v1", "--batch-timeout-fixed", "1", "--out", str(b)],
                   cwd=WT, check=True, stdout=subprocess.DEVNULL)
    (b / f"wf1_{tag}").rename(w24 / f"wf1_{tag}"); (b / "manifest.json").replace(w24 / f"manifest_{tag}.json")
    w3 = ROOT / f"_w3_{tag}"
    subprocess.run([sys.executable, "scripts_cosim/workload_fix_v1_w3_build.py", "--w2-inputs", str(w24), "--tags", tag, "--out", str(w3),
                    "--diff-out", str(ROOT / f"w3_vs_w2_{tag}.json")], cwd=WT, check=True, stdout=subprocess.DEVNULL)
    (w3 / f"wf1_{tag}").rename(inp / f"wf1_{tag}")
    for f in sorted((inp / f"wf1_{tag}" / "wl").glob("grounded_g*_n50000.json")):
        subprocess.run([sys.executable, "scripts_cosim/scale_probe_remap_clients.py", str(f), str(f) + ".tmp", "--clients", str(a.clients)], cwd=WT, check=True, stdout=subprocess.DEVNULL)
        Path(str(f) + ".tmp").replace(f)
    (inp / f"grounded_{tag}").symlink_to(inp / f"wf1_{tag}")
    return tag


def lock_profile(tr, stt):
    """Where CD's lock waits sit: per task type and per platform type (taskResults carry the platform TYPE, not the replica), and the
    per-replica picture from stats.platformResults when the run kept it. Concentration = share of the total lock wait on the top 1/5/10."""
    def lw(r):
        if r.get("ioEndTime") is None or r.get("computeStartTime") is None:
            return 0.0
        return max(0.0, float(r["computeStartTime"]) - float(r["ioEndTime"]))
    rows = [r for r in tr if r.get("taskId") is None or int(r["taskId"]) >= 0]
    total = sum(lw(r) for r in rows)
    def name(x):
        return (x.get("shortName") or x.get("name")) if isinstance(x, dict) else str(x)
    by_task, by_plat = {}, {}
    for r in rows:
        by_task[name(r.get("taskType"))] = by_task.get(name(r.get("taskType")), 0.0) + lw(r)
        by_plat[name(r.get("platform"))] = by_plat.get(name(r.get("platform")), 0.0) + lw(r)
    out = dict(total_lock_wait_s=total, tasks_with_lock_wait=sum(1 for r in rows if lw(r) > 0), n_tasks=len(rows),
               by_task_type={k: round(v / total, 3) for k, v in by_task.items()} if total else {},
               by_platform_type={k: round(v / total, 3) for k, v in by_plat.items()} if total else {})
    pr = stt.get("platformResults") or stt.get("platforms")
    if isinstance(pr, list) and pr:
        vals = []
        for p in pr:
            v = p.get("totalLockWait") or p.get("lockWaitTime") or p.get("totalQueueTime") or p.get("queueTime")
            if v is not None:
                vals.append((float(v), p.get("nodeName") or p.get("node"), p.get("platformId"), name(p.get("platformType"))))
        vals.sort(reverse=True); s = sum(v for v, *_ in vals) or 1.0
        out["per_replica"] = dict(key="totalLockWait|lockWaitTime|totalQueueTime|queueTime (first present)", replicas=len(vals), replicas_nonzero=sum(1 for v, *_ in vals if v > 0),
                                  top1_share=round(vals[0][0] / s, 3) if vals else None, top5_share=round(sum(v for v, *_ in vals[:5]) / s, 3),
                                  top10_share=round(sum(v for v, *_ in vals[:10]) / s, 3), top10=[dict(wait=round(v, 1), node=n, platform=i, type=t) for v, n, i, t in vals[:10]])
    return out


def run_cell(tag, seed, win, policy, profile=False):
    raw = Path(os.environ.get("HEROSIM_RAW_DIR", "/tmp")) / f"{a.tag}_{tag}_{seed}_{win}_{policy}.raw.json"
    log = OUT / f"{tag}_{seed}_{win}_{policy}.log"
    env = dict(os.environ, HEROSIM_AUDIT_INPUTS=str(ROOT / "inputs"), TS="1.0", SIM_FORCE_FULL_STATS="1", HEROSIM_SHARED_AUTOSCALER="0",
               GATE_FIXED_POLICY_TIME_SCALE="1.0", OMP_NUM_THREADS="1")
    t0 = time.time()
    try:
        p = subprocess.run(["timeout", "--kill-after=60", str(a.limit), "scripts_cosim/physics_audit/run_cell.sh", str(seed), win, tag, policy, str(raw), "50000"],
                           cwd=WT, env=env, capture_output=True, text=True)
        rc = p.returncode
        log.write_text((p.stdout or "")[-4000:] + "\n" + (p.stderr or "")[-4000:])
    except Exception as e:  # noqa: BLE001
        rc = 99; log.write_text(repr(e))
    wall = time.time() - t0
    row = dict(tag=tag, seed=seed, window=win, policy=policy, rc=rc, wall_s=round(wall), finished=(rc == 0 and raw.is_file()))
    if row["finished"]:
        try:
            stt = json.load(open(raw))["stats"]
            wl = json.load(open(ROOT / "inputs" / f"wf1_{tag}" / "wl" / f"grounded_{win}_n50000.json"))
            last = max(ev["timestamp"] for ev in wl["events"])
            e = float(stt.get("averageElapsedTime") or 0.0); q = float(stt.get("averageQueueTime") or 0.0)
            tr = stt.get("taskResults") or []
            lw = lock_wait_profile(tr) if tr else None; lp = latency_percentiles(tr) if tr else None
            row.update(num_tasks=stt.get("num_tasks"), elapsed=e, plain_share=(q / e if e else None),
                       effective_share=(lw or {}).get("effective_queue_share"), lock_wait_mean=(lw or {}).get("mean"),
                       p95=(lp or {}).get("p95"), request_failures=stt.get("requestFailures") or 0, end_time=stt.get("endTime"), last_arrival=last,
                       end_over_arrival=(float(stt["endTime"]) / last if stt.get("endTime") and last else None))
            n = row["num_tasks"] or 50000
            # load_recalibration_v1 stability guard (amended form): per-task backlog = placement wait + queue + lock wait, last
            # arrival quarter's mean <= 2x the mean of the middle two; the in-system 3/4-over-1/2 ratio binds only with >= 20 tasks
            # in the system at 1/2 (below that it is reported, not applied)
            bp = backlog_profile(tr, with_lock_wait=True) if tr else None
            row["backlog"] = bp
            if bp:
                half = bp["in_system_at"]["half"]; tq = bp["in_system_at"]["three_quarter"]
                in_sys_ok = (tq / half <= 2.0) if half >= 20 else True
                backlog_ok = bp["last_over_mid"] <= 2.0 and in_sys_ok
            else:
                backlog_ok = False
            row["guards"] = dict(failures=(row["request_failures"] <= 0.01 * n), p95=(row["p95"] is not None and row["p95"] <= 300.0),
                                 end=(row["end_over_arrival"] is not None and row["end_over_arrival"] <= 1.25), backlog=backlog_ok, finished=True)
            if profile and tr:
                row["lock_profile"] = lock_profile(tr, stt)
        except Exception as ex:  # noqa: BLE001
            row.update(error=repr(ex)[:300]); row["guards"] = dict(finished=False)
        finally:
            raw.unlink(missing_ok=True)
    else:
        row["guards"] = dict(finished=False)
    row["pass"] = all(row["guards"].values())
    return row


def evaluate(m, policy="peer_greedy_network_cd", profile=False):
    tag = mint(m)
    jobs = [(tag, s, w, policy, profile) for s in a.seeds for w in a.windows]
    with ThreadPoolExecutor(max_workers=a.parallel) as ex:
        rows = list(ex.map(lambda j: run_cell(*j), jobs))
    shares = [r["effective_share"] for r in rows if r.get("effective_share") is not None]
    return dict(m=m, tag=tag, policy=policy, median_effective_share=(st.median(shares) if shares else None),
                median_plain_share=st.median([r["plain_share"] for r in rows if r.get("plain_share") is not None] or [float("nan")]),
                cells_finished=sum(r["finished"] for r in rows), cells_pass=sum(r["pass"] for r in rows), cells=rows)


steps = []; lo, hi = a.lo, a.hi; chosen = None
for i in range(a.steps):
    m = math.sqrt(lo * hi) if i else a.lo  # the first step evaluates the lower end (the bracket's seeded point), then log midpoints
    if i == 1:
        m = a.hi
    r = evaluate(m); r["step"] = i; steps.append(r)
    json.dump(steps, open(OUT / "steps.json", "w"), indent=1)
    s = r["median_effective_share"]
    ok = r["cells_pass"] == len(r["cells"])
    print(f"[{a.tag}] step {i} m={m:.4f} eff={s} plain={r['median_plain_share']:.3f} finished {r['cells_finished']}/{len(r['cells'])} pass {r['cells_pass']}", flush=True)
    if s is None or not ok:
        hi = m  # an unstable or unfinished step bounds the search from above
        continue
    if a.band[0] <= s <= a.band[1]:
        chosen = r; break
    if s < a.band[0]:
        lo = m
    else:
        hi = m
    if i >= 1 and hi / lo < 1.02:
        break
if chosen is None:
    below = [r for r in steps if r["median_effective_share"] is not None and r["cells_pass"] == len(r["cells"]) and r["median_effective_share"] < a.band[0]]
    chosen = max(below, key=lambda r: r["m"]) if below else None
    fallback = True
else:
    fallback = False
result = dict(tag=a.tag, band=a.band, bracket=[a.lo, a.hi], steps=len(steps), fallback=fallback,
              multiplier=(chosen or {}).get("m"), median_effective_share=(chosen or {}).get("median_effective_share"), chosen=chosen)
if chosen:
    result["cd_profiled"] = evaluate(chosen["m"], "peer_greedy_network_cd", profile=True)
    for pol in a.extra_policies:
        result[pol] = evaluate(chosen["m"], pol, profile=True)
    if a.knative:
        result["knative"] = evaluate(chosen["m"], "knative_network")
json.dump(result, open(OUT / "result.json", "w"), indent=1)
print(f"[{a.tag}] RESULT multiplier={result['multiplier']} eff={result['median_effective_share']} fallback={fallback}", flush=True)
