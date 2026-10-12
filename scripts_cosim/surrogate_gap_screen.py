#!/usr/bin/env python3
"""Surrogate-gap screen: the evaluation-budget frontier on brute-forced plan tables (decision_cost_s0_v1 input).

A state is a fidelity-replay dataset (a captured batch of n tasks) whose placements.jsonl holds the replay-scored co-sim label for EVERY plan of the
FULL candidate set (declared_pruning_v1_uncapped / never pruned; <= --cap plans). An exact evaluation of a plan is one fidelity replay, which reproduces
its stored label bit for bit (extend_corpus_check re-score diff 0.0), so a method's regret after k exact evaluations is read off the table. For each state
and each k in KS the regret (label - optimum) / optimum of:
  ib0_s / cd_s   the surrogate's argmin with 0 evaluations (k = 0), and the best label among the k lowest-S plans evaluated exactly (k >= 1); ib0 = S with the
                 in-batch queue term scaled to 0 (HEROSIM_PG_BATCH_ALPHA=0, cd_exactS_ib0), cd = the registered S
  exp_ib0 / exp_cd   budgeted search from the argmin-S start on the true labels: ICM sweeps (single-task moves, all alternatives of a task evaluated, best taken),
                 then alpha-expansion (every nonempty subset of the tasks that offer a label moved to it, per label), repeated until no expansion improves;
                 the start is evaluation 1, every distinct plan evaluated counts once, the best plan seen so far is the answer when the budget ends
  gnn            the GNN plan score (sum over tasks of the logit of the task's choice with the others committed, GnnScheduler._exact_plan_scorer) over the
                 full candidate set: top-1 with 0 evaluations (k = 0), best label among the top-k by score evaluated exactly
  random         expected best label of k plans drawn without replacement (the floor a surrogate has to beat)
plus the label's own shape: plans, number of optimal ties, multi-node optimum, latency-distinct top-2 gap, rank of the optimum under each S.

  surrogate_gap_screen.py select OUT.tsv CORPUS_DIR[:source] ... [--per-n 60] [--ns 2,3,4,5,6] [--cap 100000]
  surrogate_gap_screen.py run OUT.jsonl LIST.tsv [--ckpt X.pt --gnn-max-plans N] [--workers 8]
  surrogate_gap_screen.py walls OUT.jsonl LIST.tsv [--per-cell 3 --plans 5]
  surrogate_gap_screen.py read ROWS.jsonl [--walls WALLS.jsonl] [--json OUT]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics as st
import sys
import time
from collections import defaultdict
from multiprocessing import get_context
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts_cosim"))

KS = (1, 2, 4, 8, 16, 32)
EPS = 1e-12
TIE = 1e-9
CAP = 100_000
ARMS = ("ib0_s", "cd_s", "exp_ib0", "exp_cd", "gnn", "random")


# ---- tables -------------------------------------------------------------------------------------------------------------------------

def load_table(ds):
    """(combos, rtt): one row per plan (min rtt on a duplicate), combos sorted -- the canonical row order every array in this file uses.
    A combo is a tuple of (node_id, platform_id), one per task in dataset task order."""
    best = {}
    with open(Path(ds) / "placements" / "placements.jsonl") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            if "placement_plan" not in r or r.get("rtt") is None:
                continue
            p = r["placement_plan"]
            combo = tuple((int(v[0]), int(v[1])) for _, v in sorted(p.items(), key=lambda kv: int(kv[0])))
            v = float(r["rtt"])
            if combo not in best or v < best[combo]:
                best[combo] = v
    combos = sorted(best)
    return combos, np.array([best[c] for c in combos], float)


def candidates_of(combos):
    """Per task slot, the sorted distinct couples; the table must be their full product."""
    n = len(combos[0])
    cands = [sorted({c[i] for c in combos}) for i in range(n)]
    if math.prod(len(c) for c in cands) != len(combos):
        raise RuntimeError(f"the table ({len(combos)} rows) is not the full product of its slots {[len(c) for c in cands]}")
    return cands


# ---- label shape ---------------------------------------------------------------------------------------------------------------------

def label_shape(combos, rtt, s_cd=None, s_ib0=None):
    opt = float(rtt.min())
    first = int(np.argmin(rtt))               # canonical optimum: lowest label, ties by combo (the sweep's tie-break)
    ties = int((rtt <= opt * (1 + TIE)).sum())
    above = rtt[rtt > opt * (1 + TIE)]
    out = {"plans": len(rtt), "opt": opt, "opt_ties": ties,
           "multi_node_opt": len({nd for nd, _ in combos[first]}) > 1,
           "top2_gap_rel": float((above.min() - opt) / opt) if len(above) and opt > 0 else None}
    for name, s in (("cd", s_cd), ("ib0", s_ib0)):
        if s is not None:
            order = np.argsort(s, kind="stable")
            rank = np.empty(len(s), int)
            rank[order] = np.arange(1, len(s) + 1)
            out[f"opt_rank_{name}"] = int(rank[first])
            out[f"s_ties_{name}"] = int((s <= s.min() + 1e-12).sum())
    return out


# ---- arms ----------------------------------------------------------------------------------------------------------------------------

def regret(label, opt):
    return (float(label) - opt) / opt if opt > 0 else 0.0


def ranked_curve(rtt, score, ks=KS, descending=False):
    """{0: regret of the top-1 pick, k: regret of the best label among the k best-scored plans}; score ties by the canonical row order."""
    opt = float(rtt.min())
    order = np.argsort(-score if descending else score, kind="stable")
    labels = rtt[order[:max(ks)]]
    out = {0: regret(labels[0], opt)}
    for k in ks:
        out[k] = regret(labels[:k].min(), opt)
    return out


def random_curve(rtt, ks=KS):
    """Expected regret of the best of k plans drawn without replacement: sum_j label_(j) C(N-j, k-1) / C(N, k) over the sorted labels."""
    lab = np.sort(rtt)
    N = len(lab)
    opt = float(lab[0])
    out = {0: regret(lab.mean(), opt)}
    for k in ks:
        if k >= N:
            out[k] = 0.0
            continue
        j = np.arange(1, N - k + 2)           # rank of the minimum, 1-based; plans ranked below it must not be drawn
        logp = np.array([math.lgamma(N - jj + 1) - math.lgamma(k) - math.lgamma(N - jj - k + 2) for jj in j]) \
            - (math.lgamma(N + 1) - math.lgamma(k + 1) - math.lgamma(N - k + 1))
        out[k] = regret(float((np.exp(logp) * lab[:N - k + 1]).sum()), opt)
    return out


class _Stop(Exception):
    pass


def budgeted_search(table, cands, start, kmax):
    """ICM + alpha-expansion from `start` on the true labels. Returns (best-so-far label after each distinct evaluation, evaluations used, converged)."""
    n = len(cands)
    seen, traj = {}, []

    def ev(p):
        if p in seen:
            return seen[p]
        if len(seen) >= kmax:
            raise _Stop
        v = table[p]
        seen[p] = v
        traj.append(v if not traj else min(traj[-1], v))
        return v

    labels = sorted({c for cs in cands for c in cs})
    cur = tuple(start)
    converged = False
    try:
        ev(cur)
        while True:
            moved = True
            while moved:                       # ICM to its fixpoint
                moved = False
                for i in range(n):
                    bp, bv = cur, ev(cur)
                    for c in cands[i]:
                        if c == cur[i]:
                            continue
                        p = cur[:i] + (c,) + cur[i + 1:]
                        v = ev(p)
                        if v < bv - EPS:
                            bp, bv = p, v
                    if bp != cur:
                        cur, moved = bp, True
            improved = False
            for lab in labels:                 # one expansion round
                idx = [i for i in range(n) if lab in cands[i]]
                bp, bv = cur, ev(cur)
                for mask in range(1, 1 << len(idx)):
                    p = list(cur)
                    for j, i in enumerate(idx):
                        if mask >> j & 1:
                            p[i] = lab
                    p = tuple(p)
                    if p == cur:
                        continue
                    v = ev(p)
                    if v < bv - EPS:
                        bp, bv = p, v
                if bp != cur:
                    cur, improved = bp, True
            if not improved:
                converged = True
                break
    except _Stop:
        pass
    return traj, len(seen), converged


def search_curve(table, cands, start, opt, ks=KS, kmax=5000):
    traj, used, converged = budgeted_search(table, cands, start, kmax)
    at = lambda k: traj[min(k, len(traj)) - 1]
    curve = {k: regret(at(k), opt) for k in ks}
    curve[0] = curve[1]                         # the start itself costs one evaluation; k = 0 is not defined for a search, kept equal to k = 1 for the table
    return curve, {"evals": used, "converged": converged, "regret": regret(traj[-1], opt)}


def state_row(combos, rtt, s_cd, s_ib0, gnn=None, ks=KS):
    """All arms of one state. `gnn` is None when the state was not scored by the GNN (over --gnn-max-plans)."""
    table = dict(zip(combos, rtt.tolist()))
    cands = candidates_of(combos)
    opt = float(rtt.min())
    row = label_shape(combos, rtt, s_cd, s_ib0)
    arms, conv = {}, {}
    arms["ib0_s"] = ranked_curve(rtt, s_ib0, ks)
    arms["cd_s"] = ranked_curve(rtt, s_cd, ks)
    for name, s in (("exp_ib0", s_ib0), ("exp_cd", s_cd)):
        arms[name], conv[name] = search_curve(table, cands, combos[int(np.argmin(s))], opt, ks)
    if gnn is not None:
        arms["gnn"] = ranked_curve(rtt, gnn, ks, descending=True)
    arms["random"] = random_curve(rtt, ks)
    row.update(arms=arms, conv=conv, n=len(combos[0]))
    return row


# ---- replay passes (datalab) ---------------------------------------------------------------------------------------------------------

def _plans_in_batch_order(combo, batch_tasks, local_of, couple):
    return [couple[combo[local_of[int(t.id)]]] for t in batch_tasks]


def hook_s(sched, batch_tasks, plan_objs, system_state, ctx):
    """cd_exactS decision pass: S of every table row under the registered weights and with the in-batch queue term at 0 (ib0), canonical row order."""
    t0 = time.time()
    ds = ctx["job"]["ds"]
    combos, _ = load_table(ds)
    local_of = ctx["local_of"]
    couple = {(int(n.id), int(p.id)): (n, p) for n in sched.nodes.items for p in n.platforms.items}
    orch, nodes = sched._pg_orchestrator(), sched.nodes.items
    saved = (sched.pg_alpha_on, sched.pg_drain_alpha, sched.pg_batch_alpha)
    out = {}
    try:
        for name, (on, da, ba) in (("s_cd", saved), ("s_ib0", (True, 1.0, 0.0))):
            sched.pg_alpha_on, sched.pg_drain_alpha, sched.pg_batch_alpha = on, da, ba
            memo = {}
            vals = [float(sched._pg_plan_cost(batch_tasks, _plans_in_batch_order(c, batch_tasks, local_of, couple), orch, memo, nodes)[0])
                    for c in combos]
            for c in combos[:5]:                # a shared memo must give what a fresh one gives
                pl = _plans_in_batch_order(c, batch_tasks, local_of, couple)
                if abs(float(sched._pg_plan_cost(batch_tasks, pl, orch, {}, nodes)[0]) - vals[combos.index(c)]) > 1e-9:
                    raise RuntimeError("S with a shared memo differs from S with a fresh memo")
            out[name] = vals
    finally:
        sched.pg_alpha_on, sched.pg_drain_alpha, sched.pg_batch_alpha = saved
    out["s_wall"] = time.time() - t0
    return out


def _s_pass(ds):
    import rollout_replay

    dec = rollout_replay._one(dict(ds=ds, plan="policy", H=-1.0, continuation="cd_exacts", table=None, eps=0.0,
                                   decision_hook="surrogate_gap_screen:hook_s"))
    if "error" in dec:
        raise RuntimeError(dec["error"] + "\n" + dec.get("tb", ""))
    if dec["hook"] is None:
        raise RuntimeError("the decision hook never ran (the batch was not decided whole)")
    return dec["hook"]


def _gnn_pass(ds, ckpt):
    """The GNN plan score of every table row at the decision, from the GNN scheduler's own exact plan scorer. The scores are taken at the CD refine call,
    where the host holds the decision's queue/temporal snapshot (_cd_snap)."""
    os.environ.update(GNN_MODEL_PATH=ckpt, GNN_PREFIX_ALPHA_KEY="inf", PARTIAL_STATE_CONTRACT="partial_state_v5",
                      PARTIAL_STATE_LOAD_SECONDS="1", PARTIAL_STATE_EXCHANGE_SECONDS="1")
    import torch

    import rollout_replay
    from src.placement import fidelity_replay
    from src.policy.gnn.scheduler import GNNScheduler

    combos, _ = load_table(ds)
    cands = candidates_of(combos)
    cap = {}
    orig_init, orig_refine = fidelity_replay.FidelityReplay.__init__, GNNScheduler._cd_refine

    def init(self, *a, **kw):
        orig_init(self, *a, **kw)
        cap["fr"] = self

    def refine(self, tasks, placements, system_state, mode):
        fr = cap["fr"]
        local_of = {fr.batch_local[i]: i for i in range(len(fr.batch_local))}
        if "scores" not in cap and set(local_of) <= {int(t.id) for t in tasks}:
            t0 = time.time()
            pos = [local_of[int(t.id)] for t in tasks]
            allowed = {i: set(cands[pos[i]]) for i in range(len(tasks))}
            score = self._exact_plan_scorer(tasks, system_state, allowed)
            with torch.no_grad():
                cap["scores"] = [float(score([combo[pos[i]] for i in range(len(tasks))])) for combo in combos]
            cap["wall"] = time.time() - t0
        return orig_refine(self, tasks, placements, system_state, mode)

    fidelity_replay.FidelityReplay.__init__, GNNScheduler._cd_refine = init, refine
    try:
        dec = rollout_replay._one(dict(ds=ds, plan="policy", H=-1.0, continuation="cd_exacts_prg", table=None, eps=0.0))
    finally:
        fidelity_replay.FidelityReplay.__init__, GNNScheduler._cd_refine = orig_init, orig_refine
    if "error" in dec:
        raise RuntimeError(dec["error"] + "\n" + dec.get("tb", ""))
    if "scores" not in cap:
        raise RuntimeError("the GNN refine call never saw the whole batch")
    return cap["scores"], cap["wall"]


def _state(job):
    t0 = time.time()
    ds = job["ds"]
    try:
        combos, rtt = load_table(ds)
        h = _s_pass(ds)
        s_cd, s_ib0 = np.array(h["s_cd"]), np.array(h["s_ib0"])
        gnn, gnn_wall = None, None
        if job.get("ckpt") and len(combos) <= job["gnn_max_plans"]:
            g, gnn_wall = _gnn_pass(ds, job["ckpt"])
            gnn = np.array(g)
        row = state_row(combos, rtt, s_cd, s_ib0, gnn)
        row.update(ds=ds, rung=job["rung"], source=job["source"], s_wall=h["s_wall"], gnn_wall=gnn_wall, gnn_scored=gnn is not None, wall=time.time() - t0)
        return row
    except Exception as e:
        import traceback

        return dict(ds=ds, rung=job.get("rung"), error=f"{type(e).__name__}: {str(e)[:300]}", tb=traceback.format_exc()[-1500:])


def _walls(job):
    """Wall per exact replay (the label path's own FidelityReplay, one process per state): the first call builds the replay, the rest are the steady
    cost; each replayed plan is compared with its stored label."""
    try:
        import big_groups_search as B

        combos, rtt = load_table(job["ds"])
        rng = random.Random(f"walls:{job['ds']}")
        pick = sorted(rng.sample(range(len(combos)), min(job["plans"] + 1, len(combos))))
        out = []
        for i in pick:
            s = B.score(job["ds"], [list(c) for c in combos[i]])
            out.append({"wall": s["wall"], "rel": abs(s["q_batch"] - float(rtt[i])) / float(rtt[i])})
        return dict(ds=job["ds"], rung=job["rung"], n=len(combos[0]), plans=len(combos), first=out[0]["wall"], steady=[o["wall"] for o in out[1:]],
                    max_rel=max(o["rel"] for o in out))
    except Exception as e:
        return dict(ds=job["ds"], rung=job.get("rung"), error=f"{type(e).__name__}: {str(e)[:300]}")


# ---- select / run / walls ------------------------------------------------------------------------------------------------------------

def rung_of(ds):
    cfg = json.load(open(Path(ds) / "warm_snapshot.json"))["provenance"]["cell_config"]
    return cfg.split("/inputs/")[1].split("/")[0].replace("wf1_", "")


def usable(ds, cap):
    """(rung, n, plans) when the dataset holds a complete table of the FULL candidate set within cap, else None."""
    d = Path(ds)
    try:
        meta = json.loads((d / "placement_metadata.json").read_text())
        if meta.get("sweep_complete") is not True or int(meta["num_placements"]) > cap:
            return None
        draw = json.loads((d / "warm_snapshot.json").read_text())["provenance"]["candidate_draw"]
        if draw.get("pruned") is True and draw.get("rule") != "declared_pruning_v1_uncapped":
            return None
        full = draw.get("full_plans")
        if full is not None and int(full) != int(meta["num_placements"]):
            return None
        n = len(json.loads((d / "workload.json").read_text())["trace_task_ids"])
        return rung_of(d), n, int(meta["num_placements"])
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return None


def select(a):
    """per (rung, n): --per-n states, sources in the order given (so the held-out corpus fills first), a seeded draw inside a source."""
    pools = defaultdict(lambda: defaultdict(list))
    for spec in a.corpora:
        path, _, src = spec.partition(":")
        src = src or Path(path).name
        for d in sorted(Path(path).glob("ds_*")):
            u = usable(d, a.cap)
            if u and u[1] in a.ns:
                pools[(u[0], u[1])][src].append((str(d.resolve()), u[2]))
    rng = random.Random(a.seed)
    rows = []
    for (rung, n), by_src in sorted(pools.items()):
        chosen = []
        for spec in a.corpora:
            src = (spec.partition(":")[2] or Path(spec.partition(":")[0]).name)
            pool = list(by_src.get(src, []))
            rng.shuffle(pool)
            chosen += [(d, p, src) for d, p in pool[:a.per_n - len(chosen)]]
        rows += [(d, rung, n, p, src) for d, p, src in chosen]
        print(f"{rung:9s} n={n}: {len(chosen)} states (available {sum(len(v) for v in by_src.values())}: "
              f"{ {k: len(v) for k, v in by_src.items()} })")
    with open(a.out, "w") as fh:
        for r in rows:
            fh.write("\t".join(map(str, r)) + "\n")
    print(f"select: {len(rows)} states -> {a.out}")


def _jobs(a):
    return [dict(zip(("ds", "rung", "n", "plans", "source"), l.rstrip("\n").split("\t"))) for l in open(a.states) if l.strip()]


def run(a):
    jobs = _jobs(a)
    for j in jobs:
        j.update(ckpt=a.ckpt, gnn_max_plans=a.gnn_max_plans)
    with get_context("spawn").Pool(a.workers, maxtasksperchild=1) as pool, open(a.out, "w") as fh:
        for r in pool.imap_unordered(_state, jobs):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
    print(f"run: {len(jobs)} states -> {a.out}")


def walls(a):
    by = defaultdict(list)
    for j in _jobs(a):
        by[(j["rung"], j["n"])].append(j)
    rng = random.Random(a.seed)
    jobs = []
    for _, js in sorted(by.items()):
        rng.shuffle(js)
        jobs += [dict(j, plans=a.plans) for j in js[:a.per_cell]]
    with get_context("spawn").Pool(a.workers, maxtasksperchild=1) as pool, open(a.out, "w") as fh:
        for r in pool.imap_unordered(_walls, jobs):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
    print(f"walls: {len(jobs)} states -> {a.out}")


# ---- read ----------------------------------------------------------------------------------------------------------------------------

def _q(v, f):
    s = sorted(v)
    return s[min(len(s) - 1, int(f * len(s)))]


def k_needed(curves, tol, share, ks=(0,) + KS):
    """Smallest k whose share of states within tol of the optimum is at least `share`; None when no k on the grid gets there."""
    for k in ks:
        if sum(c[k] <= tol for c in curves) / len(curves) >= share:
            return k
    return None


def aggregate(rows):
    """{(rung, n): {arm: {k: {median, mean, p90, within1, within3}}, ...}} over the rows without error; the GNN arm only over states it scored."""
    cells = defaultdict(list)
    for r in rows:
        if "error" not in r:
            cells[(r["rung"], int(r["n"]))].append(r)
    out = {}
    for key, rs in sorted(cells.items()):
        cell = {"states": len(rs), "plans_median": st.median(r["plans"] for r in rs),
                "multi_node_opt": sum(r["multi_node_opt"] for r in rs) / len(rs),
                "top2_gap_median": st.median(r["top2_gap_rel"] for r in rs if r["top2_gap_rel"] is not None),
                "opt_ties_median": st.median(r["opt_ties"] for r in rs), "arms": {}, "needed": {}, "conv": {}}
        for arm in ARMS:
            sub = [r for r in rs if arm in r["arms"]]
            if not sub:
                continue
            cur = [{int(k): v for k, v in r["arms"][arm].items()} for r in sub]
            cell["arms"][arm] = {"n": len(sub), **{k: {"median": st.median(c[k] for c in cur), "mean": st.fmean(c[k] for c in cur),
                                                       "p90": _q([c[k] for c in cur], 0.9), "within1": sum(c[k] <= 0.01 for c in cur) / len(cur),
                                                       "within3": sum(c[k] <= 0.03 for c in cur) / len(cur)} for k in (0,) + KS}}
            cell["needed"][arm] = {f"{int(100 * tol)}%/{int(100 * share)}": k_needed(cur, tol, share) for tol in (0.01, 0.03) for share in (0.5, 0.9)}
        for name in ("exp_ib0", "exp_cd"):
            cv = [r["conv"][name] for r in rs]
            cell["conv"][name] = {"evals_median": st.median(c["evals"] for c in cv), "evals_p90": _q([c["evals"] for c in cv], 0.9),
                                  "converged": sum(c["converged"] for c in cv) / len(cv), "regret_at_convergence_median": st.median(c["regret"] for c in cv),
                                  "regret_at_convergence_mean": st.fmean(c["regret"] for c in cv)}
        out[key] = cell
    return out


def read(a):
    rows = [json.loads(l) for f in a.rows for l in open(f) if l.strip()]
    errs = [r for r in rows if "error" in r]
    agg = aggregate(rows)
    print(f"{len(rows)} rows, {len(errs)} errors; sources {dict(defaultdict(int, {s: sum(1 for r in rows if r.get('source') == s) for s in {r.get('source') for r in rows}}))}")
    for e in errs[:5]:
        print("  ERROR", e["ds"], e["error"])
    hdr = "k:        " + "".join(f"{k:>8d}" for k in (0,) + KS)
    for (rung, n), c in agg.items():
        print(f"\n== {rung} n={n}: {c['states']} states, plans median {c['plans_median']:.0f}, multi-node optimum {100 * c['multi_node_opt']:.0f} %, "
              f"top-2 gap median {100 * c['top2_gap_median']:.2f} %, optimal ties median {c['opt_ties_median']:.0f}")
        print("   regret % of optimum, median (mean)   [k = 0: pick with no exact evaluation; search arms: k = 1 is the start]")
        print("  " + hdr)
        for arm, v in c["arms"].items():
            print(f"  {arm:9s} " + "".join(f"{100 * v[k]['median']:5.2f}({100 * v[k]['mean']:4.1f})"[:8].rjust(8) for k in (0,) + KS) + f"   [{v['n']} states]")
        print("   share of states within 1 % / 3 %")
        for arm, v in c["arms"].items():
            print(f"  {arm:9s} " + "".join(f"{100 * v[k]['within1']:3.0f}/{100 * v[k]['within3']:<3.0f} " for k in (0,) + KS))
        print("   smallest k on the grid for >=50 % / >=90 % of states within 1 % and within 3 % (- = not within 32)")
        for arm, nd in c["needed"].items():
            f = lambda x: "-" if x is None else str(x)
            print(f"  {arm:9s} 1%: {f(nd['1%/50'])}/{f(nd['1%/90'])}   3%: {f(nd['3%/50'])}/{f(nd['3%/90'])}")
        for name, cv in c["conv"].items():
            print(f"  {name} search to convergence: evals median {cv['evals_median']:.0f} p90 {cv['evals_p90']:.0f}, converged {100 * cv['converged']:.0f} %, "
                  f"regret there median {100 * cv['regret_at_convergence_median']:.2f} % mean {100 * cv['regret_at_convergence_mean']:.2f} %")
    walls_rows = [json.loads(l) for f in (a.walls or []) for l in open(f) if l.strip()]
    wl = {}
    if walls_rows:
        ok = [w for w in walls_rows if "error" not in w]
        print(f"\n== wall per exact replay (label path; {len(ok)}/{len(walls_rows)} states ok, max |replay - label|/label {max(w['max_rel'] for w in ok):.1e})")
        for rung in sorted({w['rung'] for w in ok}):
            for n in sorted({w['n'] for w in ok if w['rung'] == rung}):
                sel = [w for w in ok if w['rung'] == rung and w['n'] == n]
                steady = [x for w in sel for x in w['steady']]
                wl[f"{rung}|{n}"] = {"steady_median": st.median(steady), "steady_p90": _q(steady, 0.9), "first_median": st.median(w['first'] for w in sel), "states": len(sel)}
                print(f"  {rung:9s} n={n}: steady {wl[f'{rung}|{n}']['steady_median']:.2f} s (p90 {wl[f'{rung}|{n}']['steady_p90']:.2f}), first call {wl[f'{rung}|{n}']['first_median']:.2f} s [{len(sel)} states]")
    sw = [r["s_wall"] for r in rows if "error" not in r]
    gw = [r["gnn_wall"] for r in rows if "error" not in r and r.get("gnn_wall") is not None]
    if sw:
        print(f"\nS pass (both S over the table): median {st.median(sw):.1f} s per state; GNN plan scoring: " + (f"median {st.median(gw):.1f} s per state over {len(gw)} states" if gw else "not run"))
    if a.json:
        Path(a.json).write_text(json.dumps({"aggregate": {f"{k[0]}|{k[1]}": v for k, v in agg.items()}, "walls": wl, "errors": len(errs)}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("out"); s.add_argument("corpora", nargs="+")
    s.add_argument("--per-n", type=int, default=60); s.add_argument("--cap", type=int, default=CAP); s.add_argument("--seed", type=int, default=20261012)
    s.add_argument("--ns", type=lambda x: [int(i) for i in x.split(",")], default=[2, 3, 4, 5, 6])
    s.set_defaults(fn=select)
    r = sub.add_parser("run")
    r.add_argument("out"); r.add_argument("states")
    r.add_argument("--ckpt", default=None); r.add_argument("--gnn-max-plans", type=int, default=20000); r.add_argument("--workers", type=int, default=8)
    r.set_defaults(fn=run)
    w = sub.add_parser("walls")
    w.add_argument("out"); w.add_argument("states")
    w.add_argument("--per-cell", type=int, default=3); w.add_argument("--plans", type=int, default=5); w.add_argument("--workers", type=int, default=8)
    w.add_argument("--seed", type=int, default=20261012)
    w.set_defaults(fn=walls)
    d = sub.add_parser("read")
    d.add_argument("rows", nargs="+"); d.add_argument("--walls", nargs="*"); d.add_argument("--json", default=None)
    d.set_defaults(fn=read)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
