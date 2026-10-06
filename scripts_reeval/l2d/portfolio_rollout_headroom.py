"""Guarded receding-horizon screen with four hand-rule continuations per action."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time

import numpy as np

from l2d_bridge import load_l2d, upstream_commit
from pdr import PDRS, pdr_action, run_pdr


def finish_with_rule(rule, env, candidate, mask):
    actions = []
    while not env.done():
        action = pdr_action(rule, env, candidate, mask)
        actions.append(action)
        _, _, _, _, candidate, mask = env.step(action)
    return float(env.max_endTime), actions


def guarded_portfolio_rollout(env, data, initial_cost, initial_actions):
    _, _, candidate, mask = env.reset(data)
    suffix = list(initial_actions)
    incumbent_cost = initial_cost
    actions = []
    evaluated = 0
    improvements = 0
    while not env.done():
        feasible = sorted(int(x) for x in candidate[~mask])
        if suffix[0] not in feasible:
            raise AssertionError("incumbent action is no longer feasible")
        best_suffix = suffix
        for action in feasible:
            for rule in PDRS:
                trial = copy.deepcopy(env)
                _, _, _, _, next_candidate, next_mask = trial.step(action)
                cost, continuation = finish_with_rule(rule, trial, next_candidate, next_mask)
                evaluated += 1
                if cost < incumbent_cost - 1e-5:
                    incumbent_cost = cost
                    best_suffix = [action, *continuation]
                    improvements += 1
        chosen = best_suffix[0]
        actions.append(chosen)
        _, _, _, _, candidate, mask = env.step(chosen)
        suffix = best_suffix[1:]
    result = float(env.max_endTime)
    if len(actions) != env.number_of_tasks or not np.isclose(result, incumbent_cost):
        raise AssertionError("incumbent cost or action count drifted")
    return result, actions, evaluated, improvements


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_j", type=int, required=True)
    p.add_argument("--n_m", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--n_test", type=int, required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    L = load_l2d(a.n_j, a.n_m, "cpu")
    np.random.seed(a.seed)
    data = [L.uni_instance_gen(n_j=a.n_j, n_m=a.n_m, low=L.configs.low,
                               high=L.configs.high) for _ in range(a.n_test)]
    digest = hashlib.sha256()
    for instance in data:
        for array in instance:
            digest.update(np.ascontiguousarray(array).tobytes())
    records = []
    for index, instance in enumerate(data):
        rules = {}
        plans = {}
        for rule in PDRS:
            start = time.perf_counter_ns()
            env = L.SJSSP(n_j=a.n_j, n_m=a.n_m)
            _, _, candidate, mask = env.reset(instance)
            makespan, actions = finish_with_rule(rule, env, candidate, mask)
            wall_ms = (time.perf_counter_ns() - start) / 1e6
            if not np.isclose(makespan, run_pdr(rule, L.SJSSP(n_j=a.n_j, n_m=a.n_m), instance)):
                raise AssertionError(f"PDR objective mismatch on case {index} with {rule}")
            rules[rule] = {"makespan": makespan, "wall_ms": wall_ms}
            plans[rule] = actions
        best_rule = min(PDRS, key=lambda rule: (rules[rule]["makespan"], rule))
        hand_cost = rules[best_rule]["makespan"]
        portfolio_wall_ms = sum(v["wall_ms"] for v in rules.values())
        start = time.perf_counter_ns()
        makespan, actions, evaluated, improvements = guarded_portfolio_rollout(
            L.SJSSP(n_j=a.n_j, n_m=a.n_m), instance, hand_cost, plans[best_rule])
        rollout_wall_ms = portfolio_wall_ms + (time.perf_counter_ns() - start) / 1e6
        audit = L.SJSSP(n_j=a.n_j, n_m=a.n_m)
        audit.reset(instance)
        for action in actions:
            audit.step(action)
        if not np.isclose(makespan, audit.max_endTime) or makespan > hand_cost + 1e-5:
            raise AssertionError(f"guarded rollout failed replay or dominance on case {index}")
        records.append({"case": index, "rules": rules, "best_rule": best_rule,
                        "best_rule_makespan": hand_cost, "portfolio_wall_ms": portfolio_wall_ms,
                        "rollout": {"PORTFOLIO": {"makespan": makespan,
                                                   "wall_ms": rollout_wall_ms,
                                                   "candidate_rollouts": evaluated,
                                                   "incumbent_improvements": improvements,
                                                   "actions": actions}}})
        print(f"case {index}: hand {hand_cost:.0f}, guarded rollout {makespan:.0f}, "
              f"{rollout_wall_ms:.1f} ms", flush=True)
    output = {"n_j": a.n_j, "n_m": a.n_m, "seed": a.seed, "n_test": a.n_test,
              "upstream_commit": upstream_commit(), "test_data_sha256": digest.hexdigest(),
              "records": records}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(output, fh, indent=1)
    print(a.out)


if __name__ == "__main__":
    main()
