"""Screen whether future-action rollout has useful headroom over cheap job-shop rules."""
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
    while not env.done():
        action = pdr_action(rule, env, candidate, mask)
        _, _, _, _, candidate, mask = env.step(action)
    return float(env.max_endTime)


def rollout_plan(env, data, continuation_rule):
    _, _, candidate, mask = env.reset(data)
    actions = []
    evaluated = 0
    while not env.done():
        feasible = sorted(int(x) for x in candidate[~mask])
        if len(feasible) == 1:
            chosen = feasible[0]
        else:
            scores = []
            for action in feasible:
                trial = copy.deepcopy(env)
                _, _, _, _, next_candidate, next_mask = trial.step(action)
                scores.append((finish_with_rule(continuation_rule, trial, next_candidate, next_mask), action))
                evaluated += 1
            chosen = min(scores)[1]
        actions.append(chosen)
        _, _, _, _, candidate, mask = env.step(chosen)
    result = float(env.max_endTime)
    return result, actions, evaluated


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
        for rule in PDRS:
            start = time.perf_counter_ns()
            makespan = run_pdr(rule, L.SJSSP(n_j=a.n_j, n_m=a.n_m), instance)
            rules[rule] = {"makespan": makespan, "wall_ms": (time.perf_counter_ns() - start) / 1e6}
        portfolio = min(rules, key=lambda rule: (rules[rule]["makespan"], rule))
        rollout = {}
        for rule in ("FDD_MWKR", "MWKR"):
            start = time.perf_counter_ns()
            makespan, actions, evaluated = rollout_plan(L.SJSSP(n_j=a.n_j, n_m=a.n_m), instance, rule)
            wall_ms = (time.perf_counter_ns() - start) / 1e6
            check = L.SJSSP(n_j=a.n_j, n_m=a.n_m)
            check.reset(instance)
            for action in actions:
                check.step(action)
            if not np.isclose(makespan, check.max_endTime):
                raise AssertionError(f"rollout action replay changed makespan on case {index}")
            rollout[rule] = {"makespan": makespan,
                             "wall_ms": wall_ms,
                             "candidate_rollouts": evaluated, "actions": actions}
            if makespan > rules[rule]["makespan"] + 1e-5:
                raise AssertionError(f"policy improvement failed on case {index} with {rule}")
        records.append({"case": index, "rules": rules, "best_rule": portfolio,
                        "best_rule_makespan": rules[portfolio]["makespan"],
                        "portfolio_wall_ms": sum(v["wall_ms"] for v in rules.values()),
                        "rollout": rollout})
        print(f"case {index}: hand {rules[portfolio]['makespan']:.0f}, "
              f"FDD rollout {rollout['FDD_MWKR']['makespan']:.0f}, "
              f"MWKR rollout {rollout['MWKR']['makespan']:.0f}", flush=True)
    output = {"n_j": a.n_j, "n_m": a.n_m, "seed": a.seed, "n_test": a.n_test,
              "upstream_commit": upstream_commit(), "test_data_sha256": digest.hexdigest(),
              "records": records}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(output, fh, indent=1)
    print(a.out)


if __name__ == "__main__":
    main()
