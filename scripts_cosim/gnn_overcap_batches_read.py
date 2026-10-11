"""gnn_overcap_v1 readout from kept per-task results: where H (or T) differs from cd_exactS, by whether the batch was over the exact cap, by batch size and by phase.
Descriptive; the registered tests are in gnn_overcap_gate_read.py.

A batch is a line of the arm's own over-cap log (HEROSIM_OVERCAP_LOG): its task ids, size n and whether the slate product was above the cap. Task ids are workload event
indices (checked against the dispatched times, as selfsearch_gap_batches does). Both arms are bucketed by the OVER-CAP ARM's batches, so the same task falls in the same bucket in
both: the arms follow different trajectories and cd_exactS writes no such log. Buckets: over / within the cap, and size <= 4 / 5-8 / 9+; phase = the opening quarter (first 12,500 tasks) vs
the rest. Per bucket the gap is sum(latency over-cap arm - latency cd_exactS) / all tasks, so the buckets add to the mean-latency gap; it is printed in % of cd_exactS's mean latency
(median over topologies and the pooled share of the gap). Then per batch: the share that stack (2+ tasks on one platform) and cold starts per batch, per arm and bucket.

  gnn_overcap_batches_read.py --pairs <topo>=<rawA>,<rawE>,<overcap.jsonl>,<workload.json> ... [--out f.json]
"""
import argparse
import json
import statistics as st
import sys
from collections import defaultdict

SIZES = (("le4", 1, 4), ("5to8", 5, 8), ("ge9", 9, 10 ** 9))
OPENING_TASKS = 12500


def size_bucket(n):
    return next(name for name, lo, hi in SIZES if lo <= n <= hi)


def load_tasks(path):
    doc = json.load(open(path))
    rows = (doc.get("stats") or doc)["taskResults"]
    out = {}
    for r in rows:
        tid = r.get("taskId")
        if tid is None or int(tid) < 0:
            continue
        out[int(tid)] = {"disp": float(r["dispatchedTime"]), "lat": float(r["doneTime"]) - float(r["dispatchedTime"]),
                         "cold": 1 if r.get("coldStarted") else 0, "plat": (r.get("executionNode"), r.get("executionPlatform"))}
    return out


def load_groups(workload_path):
    ev = json.load(open(workload_path))["events"]
    return {"ts": [float(e["timestamp"]) for e in ev]}


def check_alignment(tasks, groups, tol=0.5):
    """Task ids are workload event indices only if the dispatched times follow the event timestamps; fail loudly otherwise."""
    worst = max(abs(t["disp"] - groups["ts"][i]) for i, t in tasks.items())
    if len(tasks) != len(groups["ts"]) or worst > tol:
        raise SystemExit(f"FAIL LOUD: task ids do not line up with workload events ({len(tasks)} tasks vs {len(groups['ts'])} events, worst dispatched-time offset {worst:.3f} s)")
    return worst


def load_batches(path):
    out = []
    for line in open(path):
        r = json.loads(line)
        out.append({"tasks": [int(i) for i in r["tasks"]], "n": int(r["n"]), "over": bool(r["over"])})
    return out


def check_partition(batches, n_tasks):
    ids = [i for b in batches for i in b["tasks"]]
    if len(ids) != len(set(ids)):
        raise SystemExit("FAIL LOUD: a task appears in two batches of the over-cap log")
    if len(ids) != n_tasks:
        raise SystemExit(f"FAIL LOUD: the over-cap log covers {len(ids)} tasks, the cell has {n_tasks} (singleton batches are not logged: they are never over the cap and are tallied under size 1 only if the log holds them)")


def buckets(batch):
    return ("over" if batch["over"] else "within", size_bucket(batch["n"]))


def gap_by_bucket(a, b, batches):
    n = len(a)
    base = sum(t["lat"] for t in b.values()) / n
    out = defaultdict(float)
    for bt in batches:
        phase = "opening" if min(bt["tasks"]) < OPENING_TASKS else "rest"
        d = sum(a[i]["lat"] - b[i]["lat"] for i in bt["tasks"]) / n
        cap, sz = buckets(bt)
        for ph in (phase, "all"):
            for c in (cap, "all"):
                for s in (sz, "all"):
                    out[(ph, c, s)] += d
    gap = sum(a[i]["lat"] - b[i]["lat"] for i in a) / n
    return {k: v for k, v in out.items()}, base, gap


def stack_cold(tasks, batches):
    res = defaultdict(lambda: [0, 0, 0, 0, 0])
    for bt in batches:
        mem = bt["tasks"]
        counts = defaultdict(int)
        for i in mem:
            counts[tasks[i]["plat"]] += 1
        stacked = 1 if len(mem) >= 2 and max(counts.values()) >= 2 else 0
        cold = sum(tasks[i]["cold"] for i in mem)
        cap, sz = buckets(bt)
        for c in (cap, "all"):
            for s in (sz, "all"):
                r = res[(c, s)]
                r[0] += 1
                r[1] += stacked
                r[2] += cold
                r[3] += len(mem)
                r[4] += sum(1 for k in counts.values() if k >= 2)
    return dict(res)


def analyse(a, b, batches):
    gb, base, gap = gap_by_bucket(a, b, batches)
    return {"gap_s": gap, "base_s": base, "gap_pct": 100 * gap / base,
            "buckets": {"|".join(k): {"gap_s": v, "pct_of_b": 100 * v / base, "share": v / gap if gap else None} for k, v in sorted(gb.items())},
            "stack_cold_A": {"|".join(k): v for k, v in stack_cold(a, batches).items()}, "stack_cold_B": {"|".join(k): v for k, v in stack_cold(b, batches).items()}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    x = ap.parse_args()
    res = {}
    for spec in x.pairs:
        topo, rest = spec.split("=", 1)
        pa, pb, log, wl = rest.split(",")
        groups = load_groups(wl)
        a, b = load_tasks(pa), load_tasks(pb)
        check_alignment(a, groups)
        check_alignment(b, groups)
        batches = load_batches(log)
        check_partition(batches, len(a))
        res[topo] = analyse(a, b, batches)
        over_tasks = sum(len(bt["tasks"]) for bt in batches if bt["over"])
        res[topo]["over_cap_task_share"] = over_tasks / len(a)
        print(f"{topo}: gap {res[topo]['gap_s']:+.4f} s/task = {res[topo]['gap_pct']:+.2f} % of cd_exactS; over-cap batches hold {100 * over_tasks / len(a):.1f} % of the tasks", flush=True)
    keys = sorted({k for r in res.values() for k in r["buckets"]})
    tot_gap = sum(r["gap_s"] for r in res.values())
    print("\n== gap by bucket (phase|cap|size): % of cd_exactS mean latency (median over topologies | pooled share of the gap)")
    for k in keys:
        pct = [r["buckets"][k]["pct_of_b"] for r in res.values() if k in r["buckets"]]
        sh = sum(r["buckets"][k]["gap_s"] for r in res.values() if k in r["buckets"]) / tot_gap if tot_gap else float("nan")
        print(f"  {k:24s} {st.median(pct):+7.3f} pp   {100 * sh:+7.1f} %   per topology " + " ".join(f"{p:+.2f}" for p in pct))
    for label, field in (("over-cap arm (A)", "stack_cold_A"), ("cd_exactS (B)", "stack_cold_B")):
        print(f"\n== {label}: batches, share that stack, tasks per batch, cold starts per batch (pooled, by the over-cap arm's batches)")
        for k in sorted({k for r in res.values() for k in r[field]}):
            t = [sum(r[field][k][i] for r in res.values() if k in r[field]) for i in range(5)]
            print(f"  {k:14s} batches {t[0]:6d}  stack {t[1] / t[0]:.3f}  cold/batch {t[2] / t[0]:.3f}  cold/task {t[2] / t[3]:.4f}  sharing/batch {t[4] / t[0]:.3f}")
    if x.out:
        json.dump(res, open(x.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
