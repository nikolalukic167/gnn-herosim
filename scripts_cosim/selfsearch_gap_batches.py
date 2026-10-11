"""selfsearch_gap_v1 diagnostic reader: where does GNN self-search lose to cd_exactS, by batch size and phase, from kept per-task results.
Descriptive (4 topologies, one window); no bars.

Per cell pair (topology): A = ra_gnn_eng_ssx_s1 (self-search with the exact-S shadow, which serves exactly what self-search serves), B = cd_exactS_s0.
Both read the same workload, so a task has the same arrival, peer group and batch size in both. Batch = the task's peer group in the workload (checked against
the per-task dispatched time); size buckets <= 4, 5-8, 9+. Phase = the opening quarter (first 12,500 tasks in arrival order) vs the rest, and also t < 120 s.

 1. Gap by bucket: for each (phase, size) bucket, (sum of A's task latencies - sum of B's over the bucket's tasks) / all tasks, so buckets add to the mean-latency gap;
    in seconds per task, in % of B's mean latency, and as a share of the gap. Medians over topologies and the pooled value.
 2. Stacking: per batch of size >= 2, whether two or more of its tasks ran on one platform (executionNode + executionPlatform); the share of batches that stack
    and the mean number of tasks that share a platform with a batch-mate, per arm.
 3. Cold starts per batch (sum of coldStarted over the batch's tasks), per arm.
 4. Disagreement (A's log only, from the shadow): per batch, whether the exact-S plan reached from self-search's plan differs from it, how many tasks it moves, and the
    rule's S of the self-search plan minus the S of that plan (seconds per batch, and % of the refined S). A batch where the exact search fell back to the expansion
    is reported separately. The refined plan is CD passes from the self-search plan then exhaustive S over the top-5 slate, so it is cd_exactS's S-argmin up to ties and
    the over-cap fallback; it is not the plan the cd_exactS arm served in its own trajectory.

  selfsearch_gap_batches.py --pairs <topo>=<rawA>,<rawB>,<cdr.jsonl>,<workload.json> ... [--out f.json]
"""
import argparse
import json
import statistics as st
import sys
from collections import defaultdict

SIZES = (("le4", 1, 4), ("5to8", 5, 8), ("ge9", 9, 10 ** 9))
OPENING_TASKS = 12500
OPENING_SECONDS = 120.0


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
    members = defaultdict(list)
    for i, e in enumerate(ev):
        members[e["peer_group"]].append(i)
    return {"ts": [float(e["timestamp"]) for e in ev], "group": [e["peer_group"] for e in ev], "members": dict(members)}


def check_alignment(tasks, groups, tol=0.5):
    """Task ids are workload event indices only if the dispatched times follow the event timestamps; fail loudly otherwise."""
    worst = max(abs(t["disp"] - groups["ts"][i]) for i, t in tasks.items())
    if len(tasks) != len(groups["ts"]) or worst > tol:
        raise SystemExit(f"FAIL LOUD: task ids do not line up with workload events ({len(tasks)} tasks vs {len(groups['ts'])} events, worst dispatched-time offset {worst:.3f} s)")
    return worst


def gap_buckets(a, b, groups):
    n = len(a)
    size = {i: len(groups["members"][groups["group"][i]]) for i in a}
    out = defaultdict(float)
    for i in a:
        d = (a[i]["lat"] - b[i]["lat"]) / n
        sb = size_bucket(size[i])
        out[("quarter", "opening" if i < OPENING_TASKS else "rest", sb)] += d
        out[("t120", "opening" if groups["ts"][i] < OPENING_SECONDS else "rest", sb)] += d
        out[("quarter", "opening" if i < OPENING_TASKS else "rest", "all")] += d
        out[("t120", "opening" if groups["ts"][i] < OPENING_SECONDS else "rest", "all")] += d
    base = sum(t["lat"] for t in b.values()) / n
    return dict(out), base, sum(a[i]["lat"] - b[i]["lat"] for i in a) / n


def stack_cold(tasks, groups):
    """Per (phase, size) bucket of batches: batches, stacked batches, tasks sharing a platform with a batch-mate, cold starts."""
    res = defaultdict(lambda: [0, 0, 0, 0, 0])
    for g, mem in groups["members"].items():
        n = len(mem)
        if any(i not in tasks for i in mem):
            continue
        counts = defaultdict(int)
        for i in mem:
            counts[tasks[i]["plat"]] += 1
        stacked = 1 if n >= 2 and max(counts.values()) >= 2 else 0
        sharing = sum(c for c in counts.values() if c >= 2) if n >= 2 else 0
        cold = sum(tasks[i]["cold"] for i in mem)
        for phase in ("opening" if min(mem) < OPENING_TASKS else "rest", "all"):
            for sb in (size_bucket(n), "all"):
                r = res[(phase, sb)]
                r[0] += 1
                r[1] += stacked
                r[2] += sharing
                r[3] += cold
                r[4] += n
    return dict(res)


def disagreement(path, groups):
    res = defaultdict(lambda: {"batches": 0, "differ": 0, "moved": 0, "tasks": 0, "fallback": 0, "s_gap": [], "s_rel": []})
    for line in open(path):
        r = json.loads(line)
        n = len(r["tasks"])
        phase = "opening" if min(r["tasks"]) < OPENING_TASKS else "rest"
        differ = r["seed"] != r["refined"]
        moved = sum(1 for x, y in zip(r["seed"], r["refined"]) if x != y)
        for ph in (phase, "all"):
            for sb in (size_bucket(n), "all"):
                x = res[(ph, sb)]
                x["batches"] += 1
                x["differ"] += int(differ)
                x["moved"] += moved
                x["tasks"] += n
                x["fallback"] += int(r["exact_fallback"] > 0)
                x["s_gap"].append(r["s_seed"] - r["s_refined"])
                x["s_rel"].append((r["s_seed"] - r["s_refined"]) / r["s_refined"] if r["s_refined"] > 0 else 0.0)
    out = {}
    for k, x in res.items():
        g = sorted(x["s_gap"])
        out[k] = {"batches": x["batches"], "differ_share": x["differ"] / x["batches"], "tasks_moved_share": x["moved"] / x["tasks"], "fallback_share": x["fallback"] / x["batches"],
                  "s_gap_mean": sum(g) / len(g), "s_gap_median": st.median(g), "s_gap_rel_mean": sum(x["s_rel"]) / len(x["s_rel"]), "s_gap_negative_batches": sum(1 for v in g if v < -1e-9)}
    return out


def analyse(a, b, groups, cdr_path):
    gb, base, gap = gap_buckets(a, b, groups)
    return {"gap_s": gap, "base_s": base, "gap_pct": 100 * gap / base,
            "buckets": {"|".join(k): {"gap_s": v, "pct_of_b": 100 * v / base, "share": v / gap if gap else None} for k, v in sorted(gb.items())},
            "stack_cold_A": {"|".join(k): v for k, v in stack_cold(a, groups).items()}, "stack_cold_B": {"|".join(k): v for k, v in stack_cold(b, groups).items()},
            "disagreement": {"|".join(k): v for k, v in disagreement(cdr_path, groups).items()}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    x = ap.parse_args()
    res = {}
    for spec in x.pairs:
        topo, rest = spec.split("=", 1)
        pa, pb, cdr, wl = rest.split(",")
        groups = load_groups(wl)
        a, b = load_tasks(pa), load_tasks(pb)
        check_alignment(a, groups)
        check_alignment(b, groups)
        res[topo] = analyse(a, b, groups, cdr)
        print(f"topology {topo}: gap {res[topo]['gap_s']:+.4f} s/task = {res[topo]['gap_pct']:+.2f} % of B ({len(a)} tasks)", flush=True)
    keys = sorted({k for r in res.values() for k in r["buckets"]})
    print("\n== gap by bucket: % of cd_exactS mean latency (median over topologies | pooled share of the gap)")
    for k in keys:
        pct = [r["buckets"][k]["pct_of_b"] for r in res.values() if k in r["buckets"]]
        tot_gap = sum(r["gap_s"] for r in res.values())
        sh = sum(r["buckets"][k]["gap_s"] for r in res.values() if k in r["buckets"]) / tot_gap if tot_gap else float("nan")
        print(f"  {k:24s} {st.median(pct):+7.3f} pp   {100 * sh:+7.1f} %   per topology " + " ".join(f"{p:+.2f}" for p in pct))

    def arm_rows(label, field):
        print(f"\n== {label}: batches, share of batches that stack, tasks sharing a platform per batch, cold starts per batch (pooled over topologies)")
        ks = sorted({k for r in res.values() for k in r[field]})
        for k in ks:
            tot = [sum(r[field][k][i] for r in res.values() if k in r[field]) for i in range(5)]
            print(f"  {k:14s} batches {tot[0]:6d}  stack {tot[1] / tot[0]:.3f}  sharing/batch {tot[2] / tot[0]:.3f}  cold/batch {tot[3] / tot[0]:.3f}  cold/task {tot[3] / tot[4]:.4f}")

    arm_rows("self-search (A)", "stack_cold_A")
    arm_rows("cd_exactS (B)", "stack_cold_B")
    print("\n== disagreement with the exact-S plan from self-search's plan (A's batches, pooled)")
    dk = sorted({k for r in res.values() for k in r["disagreement"]})
    for k in dk:
        ds = [r["disagreement"][k] for r in res.values() if k in r["disagreement"]]
        nb = sum(d["batches"] for d in ds)
        w = lambda f: sum(d[f] * d["batches"] for d in ds) / nb
        print(f"  {k:14s} batches {nb:6d}  differ {w('differ_share'):.3f}  tasks moved {w('tasks_moved_share'):.3f}  fallback {w('fallback_share'):.3f}  "
              f"S(self-search)-S(refined) mean {w('s_gap_mean'):.3f} s, median-of-medians {st.median(d['s_gap_median'] for d in ds):.3f} s, mean rel {100 * w('s_gap_rel_mean'):.2f} %")
    if x.out:
        json.dump(res, open(x.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
