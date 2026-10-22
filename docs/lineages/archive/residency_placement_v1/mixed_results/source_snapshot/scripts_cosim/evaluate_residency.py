"""Sealed simulation comparisons and actual adaptive Docker burst evaluations."""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from scripts_cosim.residency_study import Container, DATA, PROTOCOL, ROOT, sha, write
from src.placement.residency import CONTRACT, all_plans, evaluate, evict_list, greedy, matching, priority, search
from src.policy.residency.model import ResidencyNet, decode

OUTPUT = ROOT / os.environ.get("RESIDENCY_OUTPUT_DIR", "docs/lineages/residency_placement_v1/results")
MODELS = ROOT / os.environ.get("RESIDENCY_MODELS_DIR", "models/residency_placement_v1")
EXACT_PLANS = all_plans()


def load_model(path):
    contract = json.loads(path.with_suffix(".contract.json").read_text())
    if contract["contract"] != CONTRACT or contract["cache_metadata_sha256"] != sha(DATA / "METADATA.json"):
        raise ValueError("checkpoint/corpus contract mismatch")
    for source, expected in contract["sources"].items():
        if sha(ROOT / source) != expected:
            raise ValueError(f"checkpoint source mismatch: {source}")
    model = ResidencyNet(**contract["architecture"])
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return model.eval(), contract


def selected_models():
    selected = {}
    for arm in ["gnn", "mpoff", "hand_mlp"]:
        choices = []
        for seed in [101, 202, 303]:
            path = MODELS / f"{arm}_seed{seed}.pt"
            model, contract = load_model(path)
            choices.append((contract["validation_regret"], seed, path))
        _, seed, path = min(choices)
        selected[arm] = {"seed": seed, "checkpoint": str(path.relative_to(ROOT)), "sha256": sha(path),
                         "validation_regret": min(choices)[0]}
    path = OUTPUT / "physical_selection.json"
    if path.exists() and json.loads(path.read_text()) != selected:
        raise ValueError("physical checkpoint selection changed")
    write(path, selected)
    return selected


def choose(case, arm, model=None):
    if arm == "exact":
        return EXACT_PLANS[int(evaluate(case, EXACT_PLANS).argmin())].tolist()
    if arm == "greedy":
        return greedy(case)
    if arm == "scarcity":
        return greedy(case, True)
    if arm == "matching":
        return matching(case)
    if arm == "search":
        return search(case)
    if model is None:
        raise ValueError("learned arm missing model")
    return decode(model, case)


def simulate():
    selected_models()
    path = OUTPUT / "simulation.json"
    if path.exists():
        raise ValueError("preserve sealed evaluation")
    cases = json.loads((DATA / "test.json").read_text())
    metadata = json.loads((DATA / "METADATA.json").read_text())
    if sha(DATA / "test.json") != metadata["files"]["test.json"]:
        raise ValueError("sealed test corpus changed")
    models = {f"{arm}_seed{seed}": load_model(MODELS / f"{arm}_seed{seed}.pt")[0]
              for arm in ["gnn", "mpoff", "hand_mlp"] for seed in [101, 202, 303]}
    plans = all_plans()
    results = []
    for c in cases:
        started = time.perf_counter()
        values = evaluate(c, plans)
        row = {"id": c["id"], "attempt": c["attempt"], "oracle_cost_s": float(values.min()),
               "oracle_time_s": time.perf_counter() - started, "arms": {}}
        for arm in ["greedy", "scarcity", "matching", "search", *models]:
            started = time.perf_counter()
            plan = choose(c, arm, models.get(arm))
            elapsed = time.perf_counter() - started
            outcome = evaluate(c, plan, return_state=True)
            row["arms"][arm] = {"plan": plan, "cost_s": float(outcome["cost"][0]),
                                  "planning_s": elapsed, "charged_cost_s": float(outcome["cost"][0]) + 8 * elapsed,
                                  "cold": int(outcome["cold"][0]), "evictions": int(outcome["evictions"][0])}
        results.append(row)
        if len(results) % 12 == 0:
            print("sealed simulation", len(results), flush=True)
    write(path, {"results": results, "metadata_sha256": sha(DATA / "METADATA.json")})


def physical_run(case, arm, model, repetition):
    pools = [{} for _ in range(3)]
    profiles = json.loads((DATA / "profiles.json").read_text())["summary"]
    waves = []
    current = copy.deepcopy(case)
    try:
        for h in range(3):
            for f in np.flatnonzero(current["resident"][h]):
                pools[h][int(f)] = Container(int(f), current["cpu"][h], current["memory_mib"][f])
        for wave, window in enumerate(case["windows"]):
            current["requests"] = window["requests"]
            current["resident"] = [[f in pools[h] for f in range(6)] for h in range(3)]
            if wave:
                current["frequency"] = (np.asarray(current["frequency"]) + np.bincount(current["requests"], minlength=6)).tolist()
            snapshot = copy.deepcopy(current)
            # Withheld windows are retained in the artifact but never exposed to the policy.
            snapshot.pop("windows", None)
            release = time.perf_counter()
            plan = choose(snapshot, arm, model)
            planning_s = time.perf_counter() - release
            prediction = evaluate(snapshot, plan, return_state=True)

            def host_work(h):
                events = []
                for i, (f, chosen) in enumerate(zip(current["requests"], plan)):
                    if chosen != h:
                        continue
                    before = [k in pools[h] for k in range(6)]
                    evicted = evict_list(before, f, current["capacity_mib"][h], current["memory_mib"], priority(current, h))
                    stop_s = 0.
                    for old in evicted:
                        stop_s += pools[h].pop(old).close()
                    cold = f not in pools[h]
                    startup_s = 0.
                    if cold:
                        pools[h][f] = Container(f, current["cpu"][h], current["memory_mib"][f])
                        startup_s = pools[h][f].cold_s
                    used = sum(current["memory_mib"][k] for k in pools[h])
                    if used > current["capacity_mib"][h]:
                        raise ValueError("physical memory admission violated")
                    result = pools[h][f].call()
                    if result["result"] != profiles[f"{f}:{current['cpu'][h]}"]["expected_result"]:
                        raise ValueError("incorrect benchmark output")
                    events.append({"request": i, "function": f, "host": h, "cold": cold,
                                   "evicted": evicted, "used_mib": used, "cold_s": startup_s,
                                   "stop_s": stop_s, "rpc": result,
                                   "completion_s": time.perf_counter() - release})
                return events

            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                groups = list(executor.map(host_work, range(3)))
            events = sorted([event for group in groups for event in group], key=lambda e: e["request"])
            if [e["request"] for e in events] != list(range(8)):
                raise ValueError("missing request completion")
            waves.append({"wave": wave, "snapshot": snapshot, "plan": plan, "planning_s": planning_s,
                          "predicted_cost_s": float(prediction["cost"][0]) + 8 * planning_s,
                          "predicted_cold": int(prediction["cold"][0]),
                          "predicted_evictions": int(prediction["evictions"][0]),
                          "actual_cost_s": sum(e["completion_s"] for e in events), "events": events})
    finally:
        for pool in pools:
            for worker in pool.values():
                worker.close()
    return {"id": case["id"], "attempt": case["attempt"], "arm": arm, "repetition": repetition,
            "waves": waves, "metadata_sha256": sha(DATA / "METADATA.json")}


def physical(exact_control=False):
    selected = selected_models()
    models = {arm: load_model(ROOT / row["checkpoint"])[0] for arm, row in selected.items()}
    cases = json.loads((DATA / "physical.json").read_text())
    for case in cases:
        if case["id"] not in [104, 105]:
            continue
        for repetition in range(2):
            arms = ["exact", "gnn", "mpoff"] if exact_control else ["greedy", "search", "gnn", "mpoff", "hand_mlp"]
            if repetition:
                arms.reverse()
            for arm in arms:
                path = OUTPUT / ("exact_control" if exact_control else "physical") / f"{case['attempt']}_{case['id']}_{arm}_{repetition}.json"
                if path.exists():
                    old = json.loads(path.read_text())
                    if old["metadata_sha256"] != sha(DATA / "METADATA.json"):
                        raise ValueError("cannot resume changed corpus")
                    continue
                result = physical_run(case, arm, models.get(arm), repetition)
                write(path, result)
                print("physical", case["attempt"], case["id"], arm, repetition,
                      sum(w["actual_cost_s"] for w in result["waves"]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["simulate", "physical", "exact-control"])
    args = parser.parse_args()
    torch.set_num_threads(1)
    {"simulate": simulate, "physical": physical, "exact-control": lambda: physical(True)}[args.phase]()


if __name__ == "__main__":
    main()
