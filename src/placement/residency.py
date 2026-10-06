"""Measured-profile FIFO placement with explicitly admitted resident containers."""
from __future__ import annotations

import itertools
import numpy as np

CONTRACT = "resident_container_fifo_v1"


def state(case, copies=1):
    return {
        "resident": np.broadcast_to(np.asarray(case["resident"], bool), (copies, 3, 6)).copy(),
        "clock": np.zeros((copies, 3)),
        "cost": np.zeros(copies),
        "cold": np.zeros(copies, int),
        "evictions": np.zeros(copies, int),
    }


def evict_list(resident, function, capacity, memory, priority):
    resident = np.asarray(resident, bool).copy()
    if resident[function]:
        return []
    if memory[function] > capacity:
        raise ValueError("function cannot fit host")
    evicted = []
    while float(np.dot(resident, memory)) + memory[function] > capacity:
        candidates = np.flatnonzero(resident)
        if not len(candidates):
            raise ValueError("infeasible memory admission")
        victim = min(candidates, key=lambda f: (priority[f], int(f)))
        resident[victim] = False
        evicted.append(int(victim))
    return evicted


def priority(case, host):
    return np.asarray(case["frequency"]) * np.asarray(case["cold_s"])[:, host] / np.asarray(case["memory_mib"])


def advance(case, current, index, hosts):
    hosts = np.asarray(hosts, int)
    rows = np.arange(len(hosts))
    f = case["requests"][index]
    memory = np.asarray(case["memory_mib"])
    selected = current["resident"][rows, hosts].copy()
    miss = ~selected[:, f]
    caps = np.asarray(case["capacity_mib"])[hosts]
    if np.any(memory[f] > caps):
        raise ValueError("infeasible action")
    priorities = np.asarray(case["cold_s"]).T[hosts] * np.asarray(case["frequency"])[None] / memory[None]
    removed = np.zeros(len(hosts), int)
    for _ in range(6):
        bad = miss & (selected @ memory + memory[f] > caps)
        if not bad.any():
            break
        victims = np.where(selected, priorities, np.inf).argmin(axis=1)
        selected[rows[bad], victims[bad]] = False
        removed[bad] += 1
    if np.any(miss & (selected @ memory + memory[f] > caps)):
        raise ValueError("memory admission did not terminate")
    selected[:, f] = True
    current["resident"][rows, hosts] = selected
    duration = np.asarray(case["exec_s"])[f, hosts] + miss * np.asarray(case["cold_s"])[f, hosts] + removed * np.asarray(case["stop_s"])[hosts]
    current["clock"][rows, hosts] += duration
    current["cost"] += current["clock"][rows, hosts]
    current["cold"] += miss
    current["evictions"] += removed


def evaluate(case, plans, return_state=False):
    plans = np.asarray(plans, int)
    if plans.ndim == 1:
        plans = plans[None]
    if plans.shape[1] != len(case["requests"]) or np.any((plans < 0) | (plans >= 3)):
        raise ValueError("invalid placement plan")
    current = state(case, len(plans))
    for i in range(plans.shape[1]):
        advance(case, current, i, plans[:, i])
    return current if return_state else current["cost"]


def all_plans(n=8):
    return np.asarray(list(itertools.product(range(3), repeat=n)), dtype=np.int8)


def features(case, prefix):
    """Same raw inputs and affordable alternative summaries for every learned arm."""
    requests = case["requests"]
    cur = state(case)
    for i, h in enumerate(prefix):
        advance(case, cur, i, [h])
    memory = np.asarray(case["memory_mib"])
    capacity = np.asarray(case["capacity_mib"])
    cold = np.asarray(case["cold_s"])
    execution = np.asarray(case["exec_s"])
    resident = cur["resident"][0]
    counts = np.bincount(requests[len(prefix):], minlength=6)
    projected = execution + cold * ~resident.T
    alternatives = np.sort(projected, axis=1)
    scarcity = alternatives[:, 1] - alternatives[:, 0]
    demand = np.zeros(3)
    for f in range(6):
        demand[int(projected[f].argmin())] += counts[f] * scarcity[f]
    time_scale = max(float(np.median(cold)), .001)
    out = np.zeros((len(requests), 3, 20), np.float32)
    for i, f in enumerate(requests):
        for h in range(3):
            evicted = evict_list(resident[h], f, capacity[h], memory, priority(case, h))
            loss = sum(case["frequency"][v] * cold[v, h] for v in evicted)
            immediate = cur["clock"][0, h] + projected[f, h] + len(evicted) * case["stop_s"][h]
            out[i, h] = [execution[f, h] / time_scale, cold[f, h] / time_scale,
                         memory[f] / capacity[h], resident[h, f],
                         cur["clock"][0, h] / time_scale, (resident[h] @ memory) / capacity[h],
                         counts[f] / 8, case["frequency"][f] / max(case["frequency"]),
                         i / 8, i == len(prefix), i < len(prefix),
                         i < len(prefix) and prefix[i] == h,
                         len(evicted) / 6, loss / time_scale,
                         immediate / time_scale, alternatives[f, 0] / time_scale,
                         scarcity[f] / time_scale, demand[h] / time_scale,
                         capacity[h] / memory.max(), counts.sum() / 8]
    return out


def greedy(case, scarcity=False):
    prefix = []
    for _ in case["requests"]:
        x = features(case, prefix)[len(prefix)]
        scores = x[:, 14] + (x[:, 17] if scarcity else 0)
        prefix.append(int(scores.argmin()))
    return prefix


def matching(case):
    from scipy.optimize import linear_sum_assignment
    requests = case["requests"]
    x = features(case, [])
    n = len(requests)
    costs = np.empty((n, 3 * n))
    for h in range(3):
        for slot in range(n):
            costs[:, h * n + slot] = x[:, h, 0] * (slot + 1) + x[:, h, 1] * (1 - x[:, h, 3])
    rows, cols = linear_sum_assignment(costs)
    plan = np.empty(n, int)
    plan[rows] = cols // n
    return plan.tolist()


def search(case):
    starts = [greedy(case), greedy(case, True), matching(case)]
    best = None
    best_cost = float("inf")
    for start in starts:
        plan = np.asarray(start, int)
        cost = float(evaluate(case, plan)[0])
        for _ in range(4):
            candidates = np.tile(plan, (len(plan) * 3, 1))
            for i in range(len(plan)):
                candidates[i * 3:i * 3 + 3, i] = np.arange(3)
            values = evaluate(case, candidates)
            j = int(values.argmin())
            if values[j] >= cost - 1e-12:
                break
            plan, cost = candidates[j].copy(), float(values[j])
        if cost < best_cost:
            best, best_cost = plan.tolist(), cost
    return best
