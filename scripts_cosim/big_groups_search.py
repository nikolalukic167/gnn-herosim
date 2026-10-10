#!/usr/bin/env python3
"""big_groups_s0_v1 reference search: plans scored by their replayed true batch latency (rollout_replay batch-only mode).

  big_groups_search.py check OUT.jsonl STATES.txt [--n-states 40] [--per-state 10] [--workers 31]

check: the scorer must reproduce the corpus label. For each state (a held-out fidelity dataset with a sweep) the label-optimal row
and a seeded draw of other sweep rows are replayed in ONE worker process, in sequence (the search scores a state's plans that way),
and the first plan is replayed again at the end; the row records |replayed - label| / label per plan and whether the repeat is
bit-identical. Bar (registered): within 0.1 % on the sampled sweep plans.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from multiprocessing import get_context
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts_cosim"))


_FR = {}


def _replay(ds: str):
    """The co-sim label's own replay of the dataset's state (executecosimulation's fidelity path), one per dataset per process."""
    if ds not in _FR:
        import os

        os.environ["HEROSIM_SNAPSHOT_FIDELITY"] = "1"
        prov = json.load(open(Path(ds) / "generation_provenance.json"))
        for k, v in (prov.get("physics_env") or {}).items():
            if k.startswith("HEROSIM_") or k in ("GNN_DECODE_MODE", "GNN_BATCH_BY_PEER_GROUP", "COSIM_AUTOSCALER_RECONCILE_INTERVAL"):
                os.environ.setdefault(k, str(v))
        from src.placement import fidelity_replay

        seed = json.load(open(Path(ds) / "infrastructure.json"))["live_snapshot_seed"]
        spec = dict(seed["fidelity_replay"])
        if not Path(spec["sim_input"]).exists():
            spec["sim_input"] = str(REPO / "data" / "nofs-ids")
        types = [next(iter(e["application"]["dag"])) for e in json.load(open(Path(ds) / "workload.json"))["events"]]
        offered = {t: [(sp["node_name"], sp["platform_id"]) for sp in specs if sp.get("candidate", True)]
                   for t, specs in seed["replicas_by_type"].items()}
        _FR[ds] = fidelity_replay.FidelityReplay(spec, types, offered)
    return _FR[ds]


def score(ds: str, plan) -> dict:
    """True batch latency of `plan` ([(node_id, platform_id)] in dataset task order) at the dataset's state: the label's own
    replay (exact batching, determined scheduler), not a policy run -- a policy's 1 ms batch poll shifts the decision off t0."""
    t0 = time.time()
    fr = _replay(ds)
    res = fr.cut_to_batch(fr.run({i: (int(p[0]), int(p[1])) for i, p in enumerate(plan)}))
    trs = sorted((tr for tr in res["stats"]["taskResults"] if tr.get("taskId", -1) >= 0), key=lambda tr: tr["taskId"])
    served = [int(tr["executionPlatform"]) for tr in trs]
    if served != [int(p[1]) for p in plan]:
        raise RuntimeError(f"replay ran platforms {served}, not the forced {[int(p[1]) for p in plan]}")
    return dict(q_batch=float(res["stats"]["total_rtt"]), batch_lat=[float(tr["doneTime"]) - float(tr["scheduledTime"]) for tr in trs],
                wall=time.time() - t0)


def _check_state(job):
    t0 = time.time()
    ds = job["ds"]
    try:
        rows = [json.loads(l) for l in open(Path(ds) / "placements" / "placements.jsonl") if l.strip()]
        rows = [r for r in rows if "placement_plan" in r and r.get("rtt") is not None]
        n = len(rows[0]["placement_plan"])
        best = min(rows, key=lambda r: float(r["rtt"]))
        rng = random.Random(f"{job['seed']}:{Path(ds).name}")
        others = [r for r in rows if r is not best]
        pick = [best] + rng.sample(others, min(job["per_state"] - 1, len(others)))
        out = []
        for r in pick:
            plan = [r["placement_plan"][str(i)] for i in range(n)]
            s = score(ds, plan)
            lab = float(r["rtt"])
            out.append(dict(plan=plan, label=lab, replay=s["q_batch"], rel=(s["q_batch"] - lab) / lab, wall=s["wall"]))
        again = score(ds, out[0]["plan"])["q_batch"]
        return dict(ds=ds, n_tasks=n, n_rows=len(rows), plans=out, repeat_identical=again == out[0]["replay"],
                    repeat=again, wall=time.time() - t0)
    except Exception as e:
        import traceback

        return dict(ds=ds, error=f"{type(e).__name__}: {str(e)[:300]}", tb=traceback.format_exc()[-1500:])


CAP = 100_000  # the declared pruning's plan cap: argmin-S is exact at or below it, the marked S-only local search above


def _neighbours(plan, slates):
    """Every single-task move inside the task's slate, then every pair swap that keeps both tasks inside their own slates."""
    n = len(plan)
    out = []
    for i in range(n):
        for c in slates[i]:
            if c != plan[i]:
                out.append(plan[:i] + [c] + plan[i + 1:])
    for i in range(n):
        for j in range(i + 1, n):
            if plan[i] != plan[j] and plan[j] in slates[i] and plan[i] in slates[j]:
                p = list(plan)
                p[i], p[j] = plan[j], plan[i]
                out.append(p)
    return out


def _s_search(cost, slates, start):
    """Lowest cost over the slate: exact enumeration up to CAP plans, else best-improvement local search (moves + swaps) from
    `start` until no neighbour is strictly cheaper. Returns (plan, cost, exact, evaluations)."""
    from itertools import product

    total = 1
    for s in slates:
        total *= len(s)
    if total <= CAP:
        best, bc = None, None
        for combo in product(*slates):
            c = cost(list(combo))
            if bc is None or c < bc - 1e-9:
                best, bc = list(combo), c
        return best, bc, True, total
    cur, cc, evals = list(start), cost(list(start)), 1
    while True:
        nb = _neighbours(cur, slates)
        evals += len(nb)
        scored = [(cost(p), k) for k, p in enumerate(nb)]
        c, k = min(scored)
        if c >= cc - 1e-9:
            return cur, cc, False, evals
        cur, cc = nb[k], c


def hook(sched, batch_tasks, plan_objs, system_state, ctx):
    """At the batch decision of the cd_exactS replay: the condition's slate (the dataset's task_candidates), cd_exactS's plan, and
    the S- and S'-minimal plans over that slate, all in dataset task order as (node_id, platform_id)."""
    ds = Path(ctx["job"]["ds"])
    fr = json.load(open(ds / "fidelity_replay.json"))
    local_of = ctx["local_of"]
    pos = [local_of[int(t.id)] for t in batch_tasks]  # batch position -> dataset index
    n = len(pos)
    couple = {(int(nd.id), int(p.id)): (nd, p) for nd in sched.nodes.items for p in nd.platforms.items}
    slates_ds = [[tuple(map(int, c)) for c in fr["task_candidates"][d]] for d in range(n)]
    slates = [slates_ds[pos[i]] for i in range(n)]  # batch order
    orch, nodes = sched._pg_orchestrator(), sched.nodes.items
    cd_plan = [(int(nd.id), int(p.id)) for nd, p in plan_objs]

    def cost_fn(shared):
        def cost(pl):
            sched.pg_shared_link = shared
            try:
                return sched._pg_plan_cost(batch_tasks, [couple[c] for c in pl], orch, {}, nodes)[0]
            finally:
                sched.pg_shared_link = False
        return cost

    start = [c if c in slates[i] else slates[i][0] for i, c in enumerate(cd_plan)]
    out = {"n": n, "plans_in_slate": 1, "cd_plan_in_slate": all(c in slates[i] for i, c in enumerate(cd_plan))}
    for s in slates:
        out["plans_in_slate"] *= len(s)
    to_ds = lambda pl: [list(pl[pos.index(d)]) for d in range(n)]
    out["cd_plan"] = to_ds(cd_plan)
    for name, shared in (("s", False), ("s2", True)):
        pl, c, exact, ev = _s_search(cost_fn(shared), slates, start)
        out[f"argmin_{name}"] = to_ds(pl)
        out[f"{name}_min"], out[f"{name}_exact"], out[f"{name}_evals"] = c, exact, ev
        out[f"{name}_of_cd"] = cost_fn(shared)(cd_plan)
    out["slate_sizes"] = [len(s) for s in slates_ds]
    return out


def _search_state(job):
    """One state: the decision pass (cd_exactS replay with the hook), then the replay-scored reference search to budget B."""
    t0 = time.time()
    ds = job["ds"]
    try:
        import os

        import rollout_replay

        if job.get("cd_top_k"):
            os.environ["HEROSIM_PG_CD_EXACT_TOP_K"] = str(job["cd_top_k"])
        dec = rollout_replay._one(dict(ds=ds, plan="policy", H=-1.0, continuation="cd_exacts", table=None, eps=0.0,
                                       decision_hook="big_groups_search:hook"))
        if "error" in dec:
            raise RuntimeError(dec["error"] + "\n" + dec.get("tb", ""))
        h = dec["hook"]
        if h is None:
            raise RuntimeError("the decision hook never ran (the batch was not decided whole)")
        fr = json.load(open(Path(ds) / "fidelity_replay.json"))
        slates = [[tuple(map(int, c)) for c in fr["task_candidates"][d]] for d in range(h["n"])]
        B = int(job["budget"])
        seen = {}  # plan tuple -> replayed batch latency

        def q(pl):
            key = tuple(tuple(map(int, c)) for c in pl)
            if key not in seen:
                seen[key] = score(ds, [list(c) for c in key])["q_batch"]
            return seen[key]

        # the starts (the GNN self-search plan is not available offline: flagged in the read); the S' plan is scored too, so the
        # best-known plan is the best of everything replayed
        starts = {"argmin_s": h["argmin_s"], "cd_exacts": h["cd_plan"]}
        start_q = {k: q(v) for k, v in starts.items()}
        start_q["argmin_s2"] = q(h["argmin_s2"])
        first = min(starts, key=lambda k: (start_q[k], k))
        cur = [tuple(map(int, c)) for c in starts[first]]
        cq, sweeps, converged = q(cur), 0, False
        rng = random.Random(f"{job['seed']}:{Path(ds).name}")
        while True:
            nb = _neighbours(cur, slates)
            new = [p for p in nb if tuple(p) not in seen]
            room = B - len(seen)
            if new and room <= 0:
                break
            if len(new) > room:
                rng.shuffle(new)
                new = new[:room]  # the budget ends inside this sweep: a seeded share of it is replayed
            for p in new:
                q(p)
            sweeps += 1 if new else 0
            done = [p for p in nb if tuple(p) in seen]
            best = min(done, key=lambda p: (seen[tuple(p)], tuple(p))) if done else None
            if best is not None and seen[tuple(best)] < cq - 1e-12:
                cur, cq = [tuple(c) for c in best], seen[tuple(best)]
                continue
            converged = len(done) == len(nb)  # every neighbour replayed and none strictly better: a local optimum
            break
        ranked = sorted(seen.values())
        return dict(ds=ds, condition=job.get("condition"), n=h["n"], plans_in_slate=h["plans_in_slate"], slate_sizes=h["slate_sizes"],
                    cd_plan_in_slate=h["cd_plan_in_slate"], s_exact=h["s_exact"], s2_exact=h["s2_exact"],
                    s_evals=h["s_evals"], s2_evals=h["s2_evals"], q_argmin_s=start_q["argmin_s"], q_cd_exacts=start_q["cd_exacts"],
                    q_argmin_s2=start_q["argmin_s2"], q_best=ranked[0], q_second=ranked[1] if len(ranked) > 1 else None,
                    best_plan=[list(c) for c in cur], replays=len(seen), budget=B, sweeps=sweeps, converged=converged,
                    exact_batches=dec.get("exact_batches"), exact_fallbacks=dec.get("exact_fallbacks"),
                    all_q=ranked[:50], wall=time.time() - t0)
    except Exception as e:
        import traceback

        return dict(ds=ds, condition=job.get("condition"), error=f"{type(e).__name__}: {str(e)[:300]}", tb=traceback.format_exc()[-2000:])


def search(a):
    states = [l.split() for l in open(a.states) if l.strip()]  # "DATASET CONDITION"
    jobs = [dict(ds=s[0], condition=s[1] if len(s) > 1 else None, budget=a.budget, seed=a.seed,
                 cd_top_k=(10 ** 6 if len(s) > 1 and s[1] == "none" else None)) for s in states]
    with get_context("spawn").Pool(a.workers, maxtasksperchild=1) as pool, open(a.out, "w") as fh:
        for r in pool.imap_unordered(_search_state, jobs):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
    print(f"search: {len(jobs)} states -> {a.out}")


def read(a):
    """Per (G, rung, condition): search health, M1 (argmin-S regret vs best-known, pooled; cd_exactS alongside), M2 (top-2 gap),
    M3 (plans > cap), M4 (argmin-S' regret, pooled), and the registered bar per G >= 8 (CD health is S7's read)."""
    from collections import defaultdict
    from statistics import median

    rows = [json.loads(l) for f in a.states.split(",") for l in open(f) if l.strip()]
    cells = defaultdict(list)
    errors = defaultdict(int)
    for r in rows:
        tag = json.load(open(Path(r["ds"]) / "warm_snapshot.json"))["provenance"]["source_tag"]
        _, rung, g = tag.split("_")
        if "error" in r:
            errors[(g, rung, r["condition"])] += 1
            continue
        cells[(g, rung, r["condition"])].append(r)
    pooled = lambda v, k: (sum(x[k] for x in v) - sum(x["q_best"] for x in v)) / sum(x["q_best"] for x in v)
    print(f"{len(rows)} rows, {sum(errors.values())} errors {dict(errors)}")
    res = {}
    for key in sorted(cells):
        v = cells[key]
        gap = sorted((x["q_second"] - x["q_best"]) / x["q_best"] for x in v if x["q_second"] is not None)
        conv = sum(x["converged"] for x in v) / len(v)
        m = dict(n=len(v), m1=pooled(v, "q_argmin_s"), cd=pooled(v, "q_cd_exacts"), m4=pooled(v, "q_argmin_s2"),
                 m2=median(gap) if gap else float("nan"), m2_1=sum(g <= 0.01 for g in gap) / max(1, len(gap)),
                 m2_5=sum(g <= 0.05 for g in gap) / max(1, len(gap)), m3=sum(x["plans_in_slate"] > CAP for x in v) / len(v),
                 conv=conv, s_exact=sum(x["s_exact"] for x in v) / len(v), replays=median(x["replays"] for x in v),
                 miss=sum(x["q_argmin_s"] > x["q_best"] * (1 + 1e-9) for x in v) / len(v))
        res[key] = m
        print(f"G {key[0]:8s} {key[1]:8s} {key[2]:4s} n {m['n']:3d} | M1 {100*m['m1']:6.2f} % (cd_exactS {100*m['cd']:6.2f} %, argmin-S misses "
              f"{100*m['miss']:.0f} %, S exact {100*m['s_exact']:.0f} %) | M2 med {100*m['m2']:.2f} % <=1 % {100*m['m2_1']:.0f} % <=5 % "
              f"{100*m['m2_5']:.0f} % | M3 {100*m['m3']:.0f} % | M4 {100*m['m4']:6.2f} % | converged {100*conv:.0f} %, replays med {m['replays']:.0f}"
              + ("  UNDER-SEARCH CHECK DUE (converged < 50 %)" if conv < 0.5 else ""))
    for cond in ("top5", "none"):
        for g in sorted({k[0] for k in res if k[0] != "Gcurrent"}):
            ks = [(g, r, cond) for r in ("moderate", "heavy") if (g, r, cond) in res]
            if len(ks) < 2:
                continue
            ok = all(res[k]["m1"] >= 0.12 and res[k]["m4"] >= 0.08 and res[k]["m2"] >= 0.02 for k in ks)
            print(f"bar {g} {cond}: M1 >= 12 %, M4 >= 8 %, median M2 >= 2 % at both rungs -> {'MET' if ok else 'NOT MET'} "
                  "(CD health from S7's read completes the bar)")


def check(a):
    states = [l.strip() for l in open(a.states) if l.strip()]
    rng = random.Random(a.seed)
    states = sorted(rng.sample(states, min(a.n_states, len(states))))
    jobs = [dict(ds=d, per_state=a.per_state, seed=a.seed) for d in states]
    res = []
    with get_context("spawn").Pool(a.workers, maxtasksperchild=1) as pool, open(a.out, "w") as fh:
        for r in pool.imap_unordered(_check_state, jobs):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            res.append(r)
    ok = [r for r in res if "error" not in r]
    rel = sorted(abs(p["rel"]) for r in ok for p in r["plans"])
    walls = sorted(p["wall"] for r in ok for p in r["plans"])
    print(f"check: {len(ok)}/{len(res)} states without error, {len(rel)} plans")
    if rel:
        q = lambda v, f: v[min(len(v) - 1, int(f * len(v)))]
        print(f"  |replay - label| / label: median {q(rel, .5):.2e} p95 {q(rel, .95):.2e} max {rel[-1]:.2e}; "
              f"within 0.1 %: {sum(x <= 1e-3 for x in rel)}/{len(rel)}; exact (<1e-9): {sum(x < 1e-9 for x in rel)}/{len(rel)}")
        print(f"  repeat in-process bit-identical: {sum(r['repeat_identical'] for r in ok)}/{len(ok)}")
        print(f"  wall per plan: median {q(walls, .5):.2f} s p95 {q(walls, .95):.2f} s")
        print("  PASS" if rel[-1] <= 1e-3 and all(r["repeat_identical"] for r in ok) and len(ok) == len(res) else "  FAIL")
    for r in res:
        if "error" in r:
            print("  ERROR", r["ds"], r["error"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("check", "search", "read"))
    ap.add_argument("out")
    ap.add_argument("states")
    ap.add_argument("--n-states", type=int, default=40)
    ap.add_argument("--per-state", type=int, default=10)
    ap.add_argument("--workers", type=int, default=31)
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--budget", type=int, default=600, help="replayed plans per state (registered B = 600, the same for every G)")
    a = ap.parse_args()
    {"check": check, "search": search, "read": read}[a.mode](a)


if __name__ == "__main__":
    main()
