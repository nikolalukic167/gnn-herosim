#!/usr/bin/env python3
"""physics_audit_v1 -- invariant checkers I1-I10, I12 and I13, over traces written with HEROSIM_AUDIT_TRACE.

(I11 is `i11_replay.py`.) Every checker returns a dict
    {"id", "name", "status", "bar", "numbers", "detail"}
with status one of PASS, FAIL, FAIL-WITH-CAUSE (I5's pre-stated fallback), NOT-TESTED. A checker that cannot
evaluate its invariant from the data it was given returns NOT-TESTED, never PASS: unknown is not a pass.

  check_invariants.py trace  --trace T [--result R] [--name ARM]            I1-I7, I9, I10 on one run
  check_invariants.py determinism --pair A.json B.json [--pair ...]         I8 over pairs of result files
  check_invariants.py rungs --rung x20=T1 --rung x30=T2 ...                 I12 across load rungs
All print a table and, with --out, write the JSON.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

EPS = 1e-9


# ---------------------------------------------------------------------------------------------------------------
# trace access
# ---------------------------------------------------------------------------------------------------------------
class Trace:
    def __init__(self, path: str):
        self.path = path
        self.rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                self.rows[row["k"]].append(row)
        if not self.rows.get("header"):
            raise ValueError(f"{path}: no header row; was it written with HEROSIM_AUDIT_TRACE?")
        self.header = self.rows["header"][0]
        self.env = self.header["env"]

    def __getitem__(self, kind: str) -> List[Dict[str, Any]]:
        return self.rows.get(kind, [])


def result(id_: str, name: str, status: str, bar: str, numbers: Dict[str, Any], detail: str = "") -> Dict[str, Any]:
    return {"id": id_, "name": name, "status": status, "bar": bar, "numbers": numbers, "detail": detail}


def rank(values: Sequence[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    if len(x) < 3 or len(x) != len(y):
        return None
    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    if sxx == 0 or syy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / math.sqrt(sxx * syy)


# ---------------------------------------------------------------------------------------------------------------
# I1 Little's law per platform
# ---------------------------------------------------------------------------------------------------------------
def i1_littles_law(tr: Trace, tol: float = 0.05, min_tasks: int = 200) -> Dict[str, Any]:
    """L = lambda x W per platform, with "in system" = arrived on the platform and not yet completed (queued,
    in rendezvous, in transfer or executing); arrival = the task is put on the platform's queue.

    Two parts, both required.
    (a) The counter. `platform_in_flight` (queued + admitted + in flight + running) is what KPA scales on. At every
        KPA tick it is compared, platform by platform, with the number of tasks the rows say are in the system at
        that instant: it must lie between the count with ties excluded and the count with ties included (events at
        the tick's own instant are ordered arbitrarily). A counter that leaks or double counts shows here. It is
        compared pointwise, not averaged over ticks, because ticks queue behind scheduler decisions and so sample
        just after arrivals (the mean of tick samples reads ~1.5x the time average).
    (b) Little's law itself, from the rows: L is the exact time average of the number in system over the window
        [first tick, last tick]; lambda and W are the arrival rate and mean sojourn of the tasks that arrived in it.
        Within `tol` on every platform with >= `min_tasks` tasks."""
    import bisect

    enq = {r["task"]: r["t"] for r in tr["enqueue"]}
    per_q: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for r in tr["svc"]:
        if r["task"] in enq:
            per_q[r["q"]].append((enq[r["task"]], r["t"]))
    bar = f"counter == rows at every tick; L within {tol:.0%} of lambda x W on every platform with >= {min_tasks} tasks"
    ticks: Dict[float, Dict[str, int]] = defaultdict(dict)
    for r in tr["kpa"]:
        for q, n in (r.get("occ") or {}).items():
            ticks[r["t"]][q] = n
    times = sorted(ticks)
    if len(times) < 10:
        return result("I1", "Little's law per platform", "NOT-TESTED", bar, {"ticks": len(times)},
                      "fewer than 10 KPA ticks (is HEROSIM_SCALEOUT=kpa on?)")
    # (a) pointwise counter check
    starts = {q: sorted(a for a, _d in ivs) for q, ivs in per_q.items()}
    ends = {q: sorted(d for _a, d in ivs) for q, ivs in per_q.items()}
    checked = violations = 0
    examples = []
    for t in times:
        for q, n in ticks[t].items():
            if q not in starts:
                low = up = 0
            else:
                # in system at t: enq <= t < done.  lower: enq < t and done > t ; upper: enq <= t and done >= t
                low = bisect.bisect_left(starts[q], t) - bisect.bisect_right(ends[q], t)
                up = bisect.bisect_right(starts[q], t) - bisect.bisect_left(ends[q], t)
            checked += 1
            if not (low <= n <= up):
                violations += 1
                if len(examples) < 5:
                    examples.append({"t": t, "q": q, "counter": n, "rows_low": low, "rows_high": up})
    # (b) Little's law from the rows
    t0, t1 = times[0], times[-1]
    span = t1 - t0
    rows = {}
    worst = 0.0
    for q, ivs in sorted(per_q.items()):
        if len(ivs) < min_tasks:
            continue
        inside = [(a, d) for a, d in ivs if t0 <= a <= t1 and d <= t1 + 1e-9]
        if len(inside) < min_tasks // 2:
            continue
        lam = len(inside) / span
        w = sum(d - a for a, d in inside) / len(inside)
        l_exact = sum(max(0.0, min(d, t1) - max(a, t0)) for a, d in ivs) / span
        lw = lam * w
        gap = abs(l_exact - lw) / lw if lw > 0 else math.inf
        worst = max(worst, gap)
        rows[q] = {"tasks": len(ivs), "lambda": lam, "W": w, "lambda_W": lw, "L": l_exact, "rel_gap": gap}
    numbers = {"counter_samples": checked, "counter_violations": violations, "counter_examples": examples,
               "platforms_tested": len(rows), "worst_rel_gap": worst if rows else None, "per_platform": rows}
    if not rows:
        return result("I1", "Little's law per platform", "NOT-TESTED", bar,
                      dict(numbers, max_tasks_on_a_platform=max((len(v) for v in per_q.values()), default=0)),
                      f"no platform has {min_tasks} tasks in the window; use a longer run")
    ok = violations == 0 and worst <= tol
    detail = "" if ok else (f"{violations} of {checked} counter samples outside the rows' bounds" if violations
                            else "worst: " + max(rows.items(), key=lambda kv: kv[1]["rel_gap"])[0])
    return result("I1", "Little's law per platform", "PASS" if ok else "FAIL", bar, numbers, detail)


# ---------------------------------------------------------------------------------------------------------------
# I2 transfer time vs analytic, I3 store-and-forward off
# ---------------------------------------------------------------------------------------------------------------
MB = 1024.0 * 1024.0


def i2_transfer_time(tr: Trace, tol: float = 0.01, sample: int = 1000) -> Dict[str, Any]:
    """Ingress: the interval the task spent between being popped and arriving, minus its measured wait for the link
    pipes, must equal route latency + size / bottleneck (route latency from the topology, bottleneck from the
    route's own link bandwidths). That interval is read from timestamps, not from the charged value, so a charge
    that disagrees with the clock shows. Peer exchange: each transfer's charged time must equal size / bottleneck of
    its route, and the per-task sum (plus link latencies) must equal the exchange the task reports."""
    svc = {r["task"]: r for r in tr["svc"]}
    ingress = [x for x in tr["xfer"] if x["kind"] == "ingress" and x["model"] == "pipelined"
               and x["task"] in svc and x["route"] and x.get("route_latency") is not None]
    step = max(1, len(ingress) // sample)
    picked = ingress[::step][:sample]
    errs = []
    for x in picked:
        s = svc[x["task"]]
        bottleneck = min(bw for _k, bw in x["route"])
        analytic = x["route_latency"] + x["bytes"] / (bottleneck * MB)
        observed = s["arrived"] - s["pop"] - (x["wait"] or 0.0)
        errs.append(abs(observed - analytic) / analytic if analytic > 0 else abs(observed))
    peer_errs = []
    by_task: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for x in tr["xfer"]:
        if x["kind"] in ("peer", "dependency"):
            by_task[x["task"]].append(x)
            if x["route"]:
                analytic = x["bytes"] / (min(bw for _k, bw in x["route"]) * MB) if x["model"] == "pipelined" else None
                if analytic:
                    peer_errs.append(abs(x["charged"] - analytic) / analytic)
    sum_errs = []
    for task, xs in list(by_task.items())[: sample]:
        s = svc.get(task)
        if s is None or not s.get("exchange"):
            continue
        total = sum((x["charged"] or 0.0) + (x["latency"] or 0.0) for x in xs)
        sum_errs.append(abs(total - s["exchange"]) / s["exchange"])
    n = len(errs) + len(peer_errs) + len(sum_errs)
    numbers = {"ingress_checked": len(errs), "ingress_max_err": max(errs, default=None),
               "peer_checked": len(peer_errs), "peer_max_err": max(peer_errs, default=None),
               "exchange_sum_checked": len(sum_errs), "exchange_sum_max_err": max(sum_errs, default=None)}
    bar = f"within {tol:.0%}, up to {sample} transfers"
    if not errs:
        return result("I2", "Transfer time vs analytic", "NOT-TESTED", bar, numbers, "no pipelined ingress transfers with a route in the trace")
    worst = max(errs + peer_errs + sum_errs)
    return result("I2", "Transfer time vs analytic", "PASS" if worst <= tol else "FAIL", bar, dict(numbers, worst=worst))


def i3_no_store_and_forward(tr: Trace) -> Dict[str, Any]:
    """No transfer is charged hops x size anywhere (ingress, peer, parent->child): the model is pipelined, no row is
    flagged store-and-forward, and every multi-hop transfer's charge is size / bottleneck, not a multiple of it."""
    bar = "no transfer charged hops x size"
    model = tr.env.get("HEROSIM_TRANSFER_MODEL")
    bad = []
    multi = 0
    for x in tr["xfer"]:
        if x["model"] != "pipelined" or x["sf"]:
            bad.append({"task": x["task"], "kind": x["kind"], "model": x["model"], "sf": x["sf"]})
            continue
        if len(x["route"]) > 1 and x["route"]:
            multi += 1
            single = x["bytes"] / (min(bw for _k, bw in x["route"]) * MB)
            if x["charged"] and single > 0 and x["charged"] > single * 1.01 and kind_is_pure_charge(x):
                bad.append({"task": x["task"], "kind": x["kind"], "charged": x["charged"], "size_over_bottleneck": single})
    numbers = {"transfers": len(tr["xfer"]), "multi_hop_transfers": multi, "violations": len(bad), "transfer_model_env": model}
    if not tr["xfer"]:
        return result("I3", "Store-and-forward off everywhere", "NOT-TESTED", bar, numbers, "no transfers in the trace")
    if model != "pipelined":
        return result("I3", "Store-and-forward off everywhere", "FAIL", bar, numbers, f"HEROSIM_TRANSFER_MODEL={model!r}")
    return result("I3", "Store-and-forward off everywhere", "PASS" if not bad else "FAIL", bar,
                  dict(numbers, examples=bad[:5]))


def kind_is_pure_charge(x: Dict[str, Any]) -> bool:
    """`charged` is the link hold for ingress and the payload transfer for peer / dependency: both are single-pass."""
    return x["kind"] in ("ingress", "peer", "dependency")


# ---------------------------------------------------------------------------------------------------------------
# I4 released replicas
# ---------------------------------------------------------------------------------------------------------------
def i4_released_replicas(tr: Trace) -> Dict[str, Any]:
    """Compute never overlaps on one replica; I/O overlap is observed; the sandbox stays warm across overlapping
    tasks. The compute lock is held from compute_start to done (execution and output)."""
    bar = "no compute overlap; I/O overlap observed; warmth preserved"
    if tr.env.get("HEROSIM_REPLICA_RELEASE") != "1":
        return result("I4", "Released replicas", "NOT-TESTED", bar, {}, "HEROSIM_REPLICA_RELEASE is not 1 in this run")
    by_q: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in tr["svc"]:
        by_q[r["q"]].append(r)
    compute_overlaps = 0
    io_overlap_pairs = 0
    cold_while_warm = 0
    warm_mismatch = 0
    checked = 0
    for q, rs in by_q.items():
        rs_c = sorted(rs, key=lambda r: (r["compute_start"], r["done"]))
        for a, b in zip(rs_c, rs_c[1:]):
            if b["compute_start"] < a["done"] - EPS:
                compute_overlaps += 1
        rs_io = sorted((r for r in rs if r.get("io_start") is not None and r.get("io_end") is not None), key=lambda r: r["io_start"])
        horizon = -math.inf
        for r in rs_io:
            if r["io_start"] < horizon - EPS:
                io_overlap_pairs += 1
            horizon = max(horizon, r["io_end"])
        # warmth lives in one replica incarnation: a release and re-creation resets the sandbox
        incs: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
        for r in rs:
            incs[r.get("incarnation")].append(r)
        for rs_i in incs.values():
            rs_p = sorted(rs_i, key=lambda r: (r["pop"], r["task"]))
            for prev, cur in zip(rs_p, rs_p[1:]):
                checked += 1
                if prev["type"] == cur["type"] and cur["cold_started"]:
                    cold_while_warm += 1
                if (cur["prev_type"] == cur["type"]) != bool(cur["warm"]):
                    warm_mismatch += 1
    concurrent = sum(1 for r in tr["svc"] if (r.get("inflight_at_pop") or 0) > 0)
    numbers = {"tasks": len(tr["svc"]), "pops_with_other_tasks_in_flight": concurrent, "compute_overlaps": compute_overlaps,
               "io_overlapping_pairs": io_overlap_pairs, "cold_start_after_same_function_popped": cold_while_warm,
               "warm_flag_vs_previous_type_mismatches": warm_mismatch, "consecutive_pops_checked": checked}
    if concurrent == 0:
        return result("I4", "Released replicas", "NOT-TESTED", bar, numbers, "no task was popped while another was in flight; nothing to observe")
    ok = compute_overlaps == 0 and io_overlap_pairs > 0 and cold_while_warm == 0 and warm_mismatch == 0
    return result("I4", "Released replicas", "PASS" if ok else "FAIL", bar, numbers)


# ---------------------------------------------------------------------------------------------------------------
# I5 scale-out causality
# ---------------------------------------------------------------------------------------------------------------
def i5_scaleout_causality(tr: Trace, share_bar: float = 0.5, rho_bar: float = 0.5) -> Dict[str, Any]:
    """Load-caused creations > 50 % of all creations AND Spearman >= 0.5 between the number of load-caused replicas
    alive and in-flight concurrency over time (summed over functions, one point per KPA tick). If load-caused
    creations are <= 50 % because reachability-triggered creation dominates, the result is FAIL-WITH-CAUSE (the
    pre-stated fallback), not a physics bug."""
    bar = f"load share > {share_bar:.0%} and Spearman >= {rho_bar}"
    created = [r for r in tr["rep_up"] if r["cause"] != "initial"]
    by_cause: Dict[str, int] = defaultdict(int)
    for r in created:
        by_cause[r["cause"]] += 1
    total = len(created)
    if total == 0 or tr.env.get("HEROSIM_SCALEOUT") != "kpa":
        return result("I5", "Scale-out causality", "NOT-TESTED", bar, {"creations": total, "by_cause": dict(by_cause)},
                      "no creations, or HEROSIM_SCALEOUT is not kpa")
    share = by_cause.get("load", 0) / total
    series: Dict[float, List[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    for r in tr["kpa"]:
        series[r["t"]][0] += r["load_live"]
        series[r["t"]][1] += r["obs"]
        series[r["t"]][2] += r["stable"]
    ordered = sorted(series.items())
    tick_rho = spearman([v[0] for _t, v in ordered], [v[1] for _t, v in ordered])
    # "over time": equal weight per stretch of time, not per tick (ticks bunch where the scheduler is busy)
    bin_s = 10.0 * float(tr.env.get("HEROSIM_POLICY_TIME_SCALE") or 1.0)
    bins: Dict[int, List[List[float]]] = defaultdict(list)
    for t, v in ordered:
        bins[int(t // bin_s)].append(v)
    xs = [sum(v[0] for v in vs) / len(vs) for _b, vs in sorted(bins.items())]
    ys = [sum(v[1] for v in vs) / len(vs) for _b, vs in sorted(bins.items())]
    rho = spearman(xs, ys)
    # information only, NOT the verdict: the same correlation against KPA's own demand signal (the 60 s stable-window
    # average it scales on). Replicas persist after a burst, so they follow that average, not instantaneous load.
    stable_rho = spearman([v[0] for _t, v in ordered], [v[2] for _t, v in ordered])
    numbers = {"creations": total, "by_cause": dict(by_cause), "load_share": share, "spearman_binned": rho,
               "spearman_per_tick": tick_rho, "info_spearman_vs_kpa_stable_window": stable_rho,
               "bin_seconds": bin_s, "bins": len(xs), "ticks": len(ordered)}
    if rho is None:
        return result("I5", "Scale-out causality", "NOT-TESTED", bar, numbers, "load-caused replica count or concurrency is constant")
    if share <= share_bar:
        return result("I5", "Scale-out causality", "FAIL-WITH-CAUSE", bar, numbers,
                      "reachability-triggered creation dominates: replica counts are partly reachability-driven")
    ok = rho >= rho_bar
    detail = "" if ok else (f"instantaneous in-flight concurrency; vs KPA's stable-window average rho = "
                            f"{stable_rho:.2f}" if stable_rho is not None else "")
    return result("I5", "Scale-out causality", "PASS" if ok else "FAIL", bar, numbers, detail)


# ---------------------------------------------------------------------------------------------------------------
# I6 memory
# ---------------------------------------------------------------------------------------------------------------
def i6_memory(tr: Trace, run_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """No node's alive replicas ever hold more memory than the node has; free memory never goes negative; refusals
    are logged (and, given the run's result, equal its memory_cap_refusals counter)."""
    bar = "no node over its memory; refusals logged"
    node_mem = {n["node"]: n["memory"] for n in tr.header["nodes"]}
    alive: Dict[Tuple[str, str, int], float] = {}
    peak: Dict[str, float] = defaultdict(float)
    violations = []
    neg_avail = 0
    events = sorted([(r["t"], 0 if r["k"] == "rep_down" else 1, r) for k in ("rep_up", "rep_down") for r in tr[k]],
                    key=lambda e: (e[0], e[1]))
    for _t, _o, r in events:
        node = r["q"].split(":")[0]
        key = (r["fn"], r["q"], r["inc"])
        if r["k"] == "rep_up":
            alive[key] = r["mem"]
            used = sum(m for (fn, q, inc), m in alive.items() if q.split(":")[0] == node)
            peak[node] = max(peak[node], used)
            if used > node_mem[node] + 1e-9:
                violations.append({"t": r["t"], "node": node, "used": used, "memory": node_mem[node]})
        else:
            alive.pop(key, None)
        if r.get("node_avail") is not None and r["node_avail"] < -1e-9:
            neg_avail += 1
    refusals = sum(r["n"] for r in tr["mem_refuse"])
    numbers = {"replica_events": len(events), "violations": len(violations), "negative_free_memory": neg_avail,
               "refusals_logged": refusals, "peak_fraction": max((peak[n] / node_mem[n] for n in peak), default=0.0)}
    counter = None
    if run_result is not None:
        counter = (run_result.get("stats", {}).get("scaleOut") or {}).get("memory_cap_refusals")
        numbers["refusals_in_result"] = counter
    ok = not violations and neg_avail == 0 and (counter is None or counter == refusals)
    status = "PASS" if ok else "FAIL"
    detail = "" if ok else ("refusal rows != result counter" if counter is not None and counter != refusals else "memory over-commit")
    if refusals == 0 and numbers["peak_fraction"] < 0.9 and ok:
        detail = "memory never came close to binding in this run; the refusal path is untested"
    return result("I6", "Memory", status, bar, numbers, detail)


# ---------------------------------------------------------------------------------------------------------------
# I7 conservation
# ---------------------------------------------------------------------------------------------------------------
def i7_conservation(tr: Trace) -> Dict[str, Any]:
    bar = "every arrived task completes or is logged failed"
    ends = tr["end"]
    arrive = {r["task"] for r in tr["arrive"]}
    done = {r["task"] for r in tr["svc"]}
    if not ends:
        return result("I7", "Conservation", "FAIL", bar, {"arrived": len(arrive), "completed": len(done)},
                      "no end row: the run did not finish, so its tasks are neither completed nor logged failed")
    e = ends[-1]
    failed = set(e["failed_ids"])
    silent = sorted(arrive - done - failed)
    numbers = {"created": e["created"], "dispatched": e["dispatched"], "done": e["done"], "failed": e["failed"],
               "arrive_rows": len(arrive), "svc_rows": len(done), "silently_missing": len(silent),
               "undispatched": len(e["undispatched_ids"]), "not_done": len(e["not_done_ids"])}
    ok = (e["dispatched"] == e["done"] + e["failed"] and not silent and not e["undispatched_ids"] and not e["not_done_ids"]
          and e["created"] == e["dispatched"])
    return result("I7", "Conservation", "PASS" if ok else "FAIL", bar, dict(numbers, examples=silent[:5]))


# ---------------------------------------------------------------------------------------------------------------
# I13 pool conservation
# ---------------------------------------------------------------------------------------------------------------
def i13_pool_conservation(tr: Trace) -> Dict[str, Any]:
    """Every platform is free, owned by a replica, or being drained: free + owned + draining equals the node's
    platform count at every KPA tick, and the free count equals the node's own `available_platforms` counter. A
    platform that is none of them has leaked out of the pool and no starved task can ever claim it."""
    bar = "free + owned + draining = platforms on every node at every KPA tick"
    rows = tr["pool"]
    if not rows:
        return result("I13", "Pool conservation", "NOT-TESTED", bar, {"ticks": 0}, "no pool rows in the trace")
    total = {n["node"]: int(n["platforms"]) for n in tr.header["nodes"]}
    bad, first = 0, []
    for r in rows:
        seen = r["nodes"]
        for name, count in total.items():
            free, owned, draining, avail = seen.get(name, [0, 0, 0, None])
            if free + owned + draining != count or (avail is not None and avail != free):
                bad += 1
                if len(first) < 5:
                    first.append({"t": r["t"], "node": name, "free": free, "owned": owned, "draining": draining,
                                  "platforms": count, "available_platforms": avail})
    return result("I13", "Pool conservation", "PASS" if not bad else "FAIL", bar,
                  {"ticks": len(rows), "violations": bad, "first": first})


# ---------------------------------------------------------------------------------------------------------------
# I8 determinism
# ---------------------------------------------------------------------------------------------------------------
def i8_determinism(pairs: Sequence[Tuple[str, str]], expected_cells: int = 12) -> Dict[str, Any]:
    """Same seed -> identical summary, two runs. `pairs` are (run A, run B) result files of the same cell. Wall-clock
    fields (decision / scheduling time) and the run's code-dirty marker are the only masked paths; see
    replay_identity.py."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from replay_identity import compare

    bar = f"identical summary on {expected_cells} cells, two runs each"
    diffs = {}
    for a, b in pairs:
        d = compare(a, b, strict=False, limit=5)
        if d:
            diffs[f"{Path(a).name} vs {Path(b).name}"] = d
    numbers = {"cells": len(pairs), "differing": len(diffs), "examples": dict(list(diffs.items())[:3])}
    if len(pairs) < expected_cells:
        return result("I8", "Determinism", "NOT-TESTED", bar, numbers, f"{len(pairs)} cells given, {expected_cells} required")
    return result("I8", "Determinism", "PASS" if not diffs else "FAIL", bar, numbers)


# ---------------------------------------------------------------------------------------------------------------
# I9 cold-start accounting
# ---------------------------------------------------------------------------------------------------------------
def i9_cold_start_accounting(tr: Trace) -> Dict[str, Any]:
    """Cold starts are counted once per sandbox creation and warm starts are never charged. A sandbox is created when
    a popped task's function differs from the one the replica's sandbox holds. Per replica incarnation, in pop order,
    a pop is cold exactly when its function differs from the previous pop's (or is the first on a fresh sandbox)."""
    bar = "cold starts = sandbox creations; warm starts never charged"
    rows = tr["svc"]
    charged_when_warm = sum(1 for r in rows if r["warm"] and (r["cold"] or 0) > EPS)
    flag_mismatch = sum(1 for r in rows if bool(r["cold_started"]) != ((r["cold"] or 0) > EPS))
    by_inc: Dict[Tuple[str, Any], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_inc[(r["q"], r.get("incarnation"))].append(r)
    expected_cold = 0
    actual_cold = 0
    excess_examples = []
    for key, rs in by_inc.items():
        rs.sort(key=lambda r: (r["pop"], r["task"]))
        prev_fn = rs[0]["prev_type"]  # the sandbox the incarnation started with (None for a fresh replica)
        for r in rs:
            should = r["type"] != prev_fn
            expected_cold += should
            actual_cold += bool(r["cold_started"])
            if bool(r["cold_started"]) and not should and len(excess_examples) < 5:
                excess_examples.append({"task": r["task"], "q": r["q"], "type": r["type"], "prev_type_at_pop": r["prev_type"],
                                        "sandbox_by_pop_order": prev_fn})
            prev_fn = r["type"]
    cold_with_zero_cost = sum(1 for r in rows if r["cold_started"] and (r["cold"] or 0) <= EPS)
    numbers = {"tasks": len(rows), "cold_starts": actual_cold, "sandbox_creations_by_pop_order": expected_cold,
               "excess_cold_starts": actual_cold - expected_cold, "warm_but_charged": charged_when_warm,
               "cold_flag_vs_charge_mismatch": flag_mismatch, "cold_flag_with_zero_cost": cold_with_zero_cost,
               "examples": excess_examples}
    if not rows:
        return result("I9", "Cold-start accounting", "NOT-TESTED", bar, numbers, "no served tasks")
    ok = charged_when_warm == 0 and flag_mismatch == 0 and actual_cold == expected_cold
    return result("I9", "Cold-start accounting", "PASS" if ok else "FAIL", bar, numbers,
                  "" if ok else "cold starts differ from sandbox creations in pop order (see examples)")


# ---------------------------------------------------------------------------------------------------------------
# I10 decision time
# ---------------------------------------------------------------------------------------------------------------
def i10_decision_time(tr: Trace, run_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Decision time is recorded outside simulated time: every decision consumes zero simulated seconds and carries a
    wall-clock duration, and the run's RTT does not include it. A scheduler path with no decision rows is reported
    NOT-TESTED (not instrumented), not passed."""
    bar = "decision time recorded outside simulated time for every arm"
    rows = tr["decision"]
    policy = tr.header["policy"]
    numbers = {"policy": policy, "decisions": len(rows)}
    if not rows:
        return result("I10", "Decision time", "NOT-TESTED", bar, numbers, f"no decision rows for {policy}: that scheduler path is not instrumented")
    advanced = [r for r in rows if abs(r["after"] - r["before"]) > EPS]
    no_wall = [r for r in rows if r.get("wall") is None or r["wall"] < 0]
    numbers.update(advanced_sim_clock=len(advanced), missing_wall_clock=len(no_wall),
                   wall_seconds_total=sum(r["wall"] for r in rows if r.get("wall") is not None))
    rtt_ok = None
    if run_result is not None:
        stats = run_result["stats"]
        tasks = [t for t in (stats.get("taskResults") or []) if t.get("taskId", -1) >= 0]
        if tasks:
            summed = sum(t["doneTime"] - t["dispatchedTime"] for t in tasks)
            rtt_ok = abs(summed - stats["total_rtt"]) <= 1e-6 * max(1.0, abs(summed))
            numbers.update(total_rtt=stats["total_rtt"], rtt_from_task_rows=summed, rtt_excludes_inference=rtt_ok)
    ok = not advanced and not no_wall and rtt_ok is not False
    return result("I10", "Decision time", "PASS" if ok else "FAIL", bar, numbers)


# ---------------------------------------------------------------------------------------------------------------
# I12 load-rung configuration
# ---------------------------------------------------------------------------------------------------------------
def i12_rung_config(rungs: Dict[str, Trace]) -> Dict[str, Any]:
    """Policy time scale, keep-alive and reconcile interval identical across all load rungs; cold starts per task
    logged per rung."""
    bar = "policy time scale, keep-alive, reconcile interval identical across rungs"
    table = {}
    for name, tr in rungs.items():
        svc = tr["svc"]
        table[name] = {
            "policy_time_scale": tr.env.get("HEROSIM_POLICY_TIME_SCALE") or "1.0",
            "keep_alive": tr.header["keep_alive"], "reconcile_interval": tr.header["reconcile_interval"],
            "kpa_stable_window": (tr.header.get("kpa") or {}).get("stable_window_s"),
            "kpa_panic_window": (tr.header.get("kpa") or {}).get("panic_window_s"),
            "tasks": len(svc), "cold_starts": sum(1 for r in svc if r["cold_started"]),
            "cold_starts_per_task": (sum(1 for r in svc if r["cold_started"]) / len(svc)) if svc else None,
            "replica_creations": sum(1 for r in tr["rep_up"] if r["cause"] != "initial"),
        }
    keys = ("policy_time_scale", "keep_alive", "reconcile_interval")
    differing = {k: sorted({str(v[k]) for v in table.values()}) for k in keys if len({str(v[k]) for v in table.values()}) > 1}
    numbers = {"per_rung": table, "differing": differing}
    if len(rungs) < 2:
        return result("I12", "Load-rung configuration", "NOT-TESTED", bar, numbers, "needs at least two rungs")
    return result("I12", "Load-rung configuration", "PASS" if not differing else "FAIL", bar, numbers,
                  "" if not differing else "policy time constants differ across rungs: " + ", ".join(differing))


# ---------------------------------------------------------------------------------------------------------------
def run_trace_checks(trace_path: str, result_path: Optional[str] = None) -> List[Dict[str, Any]]:
    tr = Trace(trace_path)
    res = json.load(open(result_path)) if result_path else None
    return [i1_littles_law(tr), i2_transfer_time(tr), i3_no_store_and_forward(tr), i4_released_replicas(tr),
            i5_scaleout_causality(tr), i6_memory(tr, res), i7_conservation(tr), i9_cold_start_accounting(tr),
            i10_decision_time(tr, res), i13_pool_conservation(tr)]


def print_table(results: Sequence[Dict[str, Any]], title: str = "") -> None:
    if title:
        print(f"== {title}")
    for r in results:
        print(f"{r['id']:4s} {r['status']:16s} {r['name']:34s} {r['detail']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("trace")
    t.add_argument("--trace", required=True)
    t.add_argument("--result")
    t.add_argument("--name", default="")
    d = sub.add_parser("determinism")
    d.add_argument("--pair", nargs=2, action="append", required=True, metavar=("A", "B"))
    d.add_argument("--cells", type=int, default=12)
    r = sub.add_parser("rungs")
    r.add_argument("--rung", action="append", required=True, metavar="NAME=TRACE")
    for p in (t, d, r):
        p.add_argument("--out")
    args = ap.parse_args()
    if args.cmd == "trace":
        results = run_trace_checks(args.trace, args.result)
        print_table(results, args.name or args.trace)
    elif args.cmd == "determinism":
        results = [i8_determinism(args.pair, args.cells)]
        print_table(results, "determinism")
    else:
        rungs = {kv.split("=", 1)[0]: Trace(kv.split("=", 1)[1]) for kv in args.rung}
        results = [i12_rung_config(rungs)]
        print_table(results, "rungs")
        for name, row in results[0]["numbers"]["per_rung"].items():
            print(f"   {name}: {row}")
    if args.out:
        json.dump(results, open(args.out, "w"), indent=1, default=str)
    return 0 if all(x["status"] in ("PASS", "NOT-TESTED", "FAIL-WITH-CAUSE") for x in results) else 1


if __name__ == "__main__":
    sys.exit(main())
