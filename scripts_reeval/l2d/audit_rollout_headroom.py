"""Independently regenerate inputs and replay saved L2D rollout schedules."""
from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np

from l2d_bridge import load_l2d, upstream_commit
from pdr import PDRS, run_pdr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--result", required=True)
    a = p.parse_args()
    result = json.load(open(a.result))
    n_j, n_m = result["n_j"], result["n_m"]
    if result["upstream_commit"] != upstream_commit():
        raise ValueError("upstream commit mismatch")
    L = load_l2d(n_j, n_m, "cpu")
    np.random.seed(result["seed"])
    data = [L.uni_instance_gen(n_j=n_j, n_m=n_m, low=L.configs.low,
                               high=L.configs.high) for _ in range(result["n_test"])]
    digest = hashlib.sha256()
    for instance in data:
        for array in instance:
            digest.update(np.ascontiguousarray(array).tobytes())
    if digest.hexdigest() != result["test_data_sha256"]:
        raise AssertionError("regenerated dataset hash mismatch")
    if len(result["records"]) != result["n_test"]:
        raise AssertionError("incomplete result")
    replayed = 0
    for index, (instance, record) in enumerate(zip(data, result["records"])):
        if record["case"] != index or set(record["rules"]) != set(PDRS):
            raise AssertionError(f"wrong case or rule set: {index}")
        for rule in PDRS:
            expected = run_pdr(rule, L.SJSSP(n_j=n_j, n_m=n_m), instance)
            if not np.isclose(expected, record["rules"][rule]["makespan"]):
                raise AssertionError(f"hand-rule mismatch: {index} {rule}")
        best = min(PDRS, key=lambda rule: (record["rules"][rule]["makespan"], rule))
        if best != record["best_rule"] or record["best_rule_makespan"] != record["rules"][best]["makespan"]:
            raise AssertionError(f"hand portfolio mismatch: {index}")
        for rule, plan in record["rollout"].items():
            actions = plan["actions"]
            if len(actions) != n_j * n_m or len(set(actions)) != n_j * n_m:
                raise AssertionError(f"invalid action sequence: {index} {rule}")
            env = L.SJSSP(n_j=n_j, n_m=n_m)
            _, _, candidate, mask = env.reset(instance)
            for action in actions:
                if action not in candidate[~mask]:
                    raise AssertionError(f"infeasible action: {index} {rule} {action}")
                _, _, _, _, candidate, mask = env.step(action)
            if not env.done() or not np.isclose(env.max_endTime, plan["makespan"]):
                raise AssertionError(f"rollout makespan mismatch: {index} {rule}")
            if rule == "PORTFOLIO" and plan["makespan"] > record["best_rule_makespan"]:
                raise AssertionError(f"guarded portfolio worsened: {index}")
            if rule in PDRS and plan["makespan"] > record["rules"][rule]["makespan"]:
                raise AssertionError(f"single-rule rollout worsened: {index} {rule}")
            replayed += 1
    print(json.dumps({"status": "PASS", "result": a.result,
                      "n_cases": len(result["records"]), "replayed_plans": replayed,
                      "dataset_sha256": digest.hexdigest()}))


if __name__ == "__main__":
    main()
