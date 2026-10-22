"""Independent event accounting and paired summaries for the residency study."""
import collections
import json
from pathlib import Path

import numpy as np
from scripts_cosim.residency_study import DATA, PROTOCOL, ROOT, sha, write
from scripts_cosim.evaluate_residency import OUTPUT, MODELS, choose, load_model
from src.placement.residency import all_plans, evaluate


def scalar_replay(case, plan):
    caches = [set(i for i, present in enumerate(row) if present) for row in case["resident"]]
    clocks = [0., 0., 0.]
    total = 0.
    cold_count, eviction_count = 0, 0
    expected = []
    for i, (f, h) in enumerate(zip(case["requests"], plan)):
        missing = f not in caches[h]
        removed = []
        while missing and sum(case["memory_mib"][k] for k in caches[h]) + case["memory_mib"][f] > case["capacity_mib"][h]:
            victim = min(caches[h], key=lambda k: (case["frequency"][k] * case["cold_s"][k][h] / case["memory_mib"][k], k))
            caches[h].remove(victim)
            removed.append(victim)
        caches[h].add(f)
        clocks[h] += case["exec_s"][f][h] + missing * case["cold_s"][f][h] + len(removed) * case["stop_s"][h]
        total += clocks[h]
        cold_count += missing
        eviction_count += len(removed)
        expected.append({"request": i, "function": f, "host": h, "cold": missing, "evicted": removed,
                         "used_mib": sum(case["memory_mib"][k] for k in caches[h])})
    return total, cold_count, eviction_count, expected, [[f in cache for f in range(6)] for cache in caches]


def bootstrap(values):
    rng = np.random.default_rng(81024)
    values = np.asarray(values)
    samples = np.median(rng.choice(values, size=(4000, len(values)), replace=True), axis=1)
    return [float(x) for x in np.quantile(samples, [.025, .975])]


def main():
    import torch
    torch.set_num_threads(1)
    protocol = json.loads(PROTOCOL.read_text())
    metadata = json.loads((DATA / "METADATA.json").read_text())
    for path, expected in metadata["files"].items():
        assert sha(DATA / path) == expected, path
    assert sha(PROTOCOL) == metadata["protocol_sha256"]
    apps = {}
    for split in ["train", "validation", "test", "physical"]:
        cases = json.loads((DATA / f"{split}.json").read_text())
        apps[split] = {tuple(a) for case in cases for a in case["apps"]}
    for a, b in __import__("itertools").combinations(apps, 2):
        assert not apps[a] & apps[b], (a, b)
    profiles = json.loads((DATA / "profiles.json").read_text())
    assert profiles["worker_sha256"] == sha(ROOT / "scripts_cosim/residency_worker.py")
    selection = json.loads((OUTPUT / "physical_selection.json").read_text())
    models = {}
    for arm, selected in selection.items():
        path = ROOT / selected["checkpoint"]
        assert sha(path) == selected["sha256"]
        models[arm] = load_model(path)[0]
    simulations = json.loads((OUTPUT / "simulation.json").read_text())["results"]
    cases = {(c["attempt"], c["id"]): c for c in json.loads((DATA / "test.json").read_text())}
    plans = all_plans()
    reference_costs = []
    for row in simulations:
        c = cases[row["attempt"], row["id"]]
        full_costs = evaluate(c, plans)
        assert np.isclose(full_costs.min(), row["oracle_cost_s"], rtol=0, atol=1e-10)
        reference_costs.append(full_costs)
        for name, outcome in row["arms"].items():
            value, cold, evictions, _, _ = scalar_replay(c, outcome["plan"])
            assert np.isclose(value, outcome["cost_s"], rtol=0, atol=1e-10)
            assert (cold, evictions) == (outcome["cold"], outcome["evictions"])
    np.savez_compressed(OUTPUT / "test_reference_sweeps.npz", plans=plans,
                        costs=np.asarray(reference_costs),
                        cases=np.asarray([f"{r['attempt']}:{r['id']}" for r in simulations]))
    physical_paths = sorted((OUTPUT / "physical").glob("*.json"))
    assert len(physical_paths) == len(protocol["attempts"]) * 20, len(physical_paths)
    physical = [json.loads(p.read_text()) for p in physical_paths]
    extra_paths = sorted((OUTPUT / "exact_control").glob("*.json"))
    assert len(extra_paths) == len(protocol["attempts"]) * 12, len(extra_paths)
    extra = [json.loads(p.read_text()) for p in extra_paths]
    events = 0
    for run in physical + extra:
        previous = None
        for wave in run["waves"]:
            c = wave["snapshot"]
            assert "windows" not in c
            if previous is not None:
                assert c["resident"] == previous
            assert choose(c, run["arm"], models.get(run["arm"])) == wave["plan"]
            value, cold, evictions, expected, previous = scalar_replay(c, wave["plan"])
            assert np.isclose(value + 8 * wave["planning_s"], wave["predicted_cost_s"], rtol=0, atol=1e-10)
            assert (cold, evictions) == (wave["predicted_cold"], wave["predicted_evictions"])
            clocks = [0., 0., 0.]
            for actual, predicted in zip(wave["events"], expected):
                for key, val in predicted.items():
                    assert actual[key] == val, (key, actual, predicted)
                f, h = actual["function"], actual["host"]
                assert actual["completion_s"] >= max(clocks[h], wave["planning_s"])
                clocks[h] = actual["completion_s"]
                assert actual["used_mib"] <= c["capacity_mib"][h]
                assert actual["rpc"]["rss_kib"] / 1024 <= c["memory_mib"][f]
                assert actual["rpc"]["result"] == profiles["summary"][f"{f}:{c['cpu'][h]}"]["expected_result"]
                events += 1
            assert len(wave["events"]) == 8
            assert np.isclose(sum(e["completion_s"] for e in wave["events"]), wave["actual_cost_s"])
    summary = {"audit_pass": True, "physical_runs": len(physical) + len(extra),
               "primary_physical_runs": len(physical), "exact_control_runs": len(extra),
               "physical_requests": events, "independent_physical_app_blocks": 2,
               "attempts": {}, "exact_control": {}}
    for attempt in protocol["attempts"]:
        name = attempt["name"]
        rows = [r for r in simulations if r["attempt"] == name]
        report = {"simulation": {}, "physical": {}}
        for arm in ["gnn", "mpoff", "hand_mlp", "greedy", "matching", "scarcity", "search"]:
            names = [f"{arm}_seed{s}" for s in protocol["training_seeds"]] if arm in protocol["arms"] else [arm]
            costs = np.asarray([np.mean([r["arms"][a]["cost_s"] for a in names]) for r in rows])
            baseline = np.asarray([r["arms"]["search"]["cost_s"] for r in rows])
            contrasts = (costs / baseline - 1) * 100
            report["simulation"][arm] = {"median_pct_vs_search": float(np.median(contrasts)),
                                         "paired_bootstrap_95": bootstrap(contrasts),
                                         "mean_cost_s": float(costs.mean()),
                                         "median_planning_ms": float(np.median([r["arms"][a]["planning_s"] * 1000 for r in rows for a in names])),
                                         "per_case_pct": contrasts.tolist()}
        oracle = np.asarray([r["oracle_cost_s"] for r in rows])
        baseline = np.asarray([r["arms"]["search"]["cost_s"] for r in rows])
        report["reference_headroom_pct"] = float(np.median((1 - oracle / baseline) * 100))
        for control in ["mpoff", "hand_mlp"]:
            g = np.asarray([np.mean([r["arms"][f"gnn_seed{s}"]["cost_s"] for s in protocol["training_seeds"]]) for r in rows])
            c = np.asarray([np.mean([r["arms"][f"{control}_seed{s}"]["cost_s"] for s in protocol["training_seeds"]]) for r in rows])
            report[f"gnn_vs_{control}_pct"] = float(np.median((g / c - 1) * 100))
        runs = [r for r in physical if r["attempt"] == name]
        grouped = collections.defaultdict(dict)
        errors = []
        for r in runs:
            grouped[(r["id"], r["repetition"])][r["arm"]] = sum(w["actual_cost_s"] for w in r["waves"])
            errors.extend(abs(w["predicted_cost_s"] / w["actual_cost_s"] - 1) * 100 for w in r["waves"])
        report["physical_prediction_median_abs_error_pct"] = float(np.median(errors))
        report["physical_prediction_p90_abs_error_pct"] = float(np.quantile(errors, .9))
        for arm in protocol["physical_arms"]:
            paired = [float(np.median([(grouped[(i, rep)][arm] / grouped[(i, rep)]["search"] - 1) * 100 for rep in [0, 1]])) for i in [104, 105]]
            report["physical"][arm] = {"median_pct_vs_search": float(np.median(paired)), "per_case_pct": paired}
        for control in ["mpoff", "hand_mlp", "greedy"]:
            paired = [float(np.median([(grouped[(i, rep)]["gnn"] / grouped[(i, rep)][control] - 1) * 100 for rep in [0, 1]])) for i in [104, 105]]
            report[f"physical_gnn_vs_{control}_pct"] = float(np.median(paired))
            report[f"physical_gnn_vs_{control}_per_case_pct"] = paired
        report["physical_cold"] = sum(e["cold"] for r in runs for w in r["waves"] for e in w["events"])
        report["physical_evictions"] = sum(len(e["evicted"]) for r in runs for w in r["waves"] for e in w["events"])
        summary["attempts"][name] = report
        exact_groups = collections.defaultdict(dict)
        for r in extra:
            if r["attempt"] == name:
                exact_groups[(r["id"], r["repetition"])][r["arm"]] = sum(w["actual_cost_s"] for w in r["waves"])
        summary["exact_control"][name] = {}
        for arm in ["gnn", "mpoff"]:
            paired = [float(np.median([(exact_groups[(i, rep)][arm] / exact_groups[(i, rep)]["exact"] - 1) * 100 for rep in [0, 1]])) for i in [104, 105]]
            summary["exact_control"][name][arm] = {"median_pct_vs_exact": float(np.median(paired)), "per_case_pct": paired}
    write(OUTPUT / "audit.json", summary)
    sources = [str(PROTOCOL.relative_to(ROOT)), "experiments/residency_exact_control_v1.json", "src/placement/residency.py",
               "src/policy/residency/model.py", "src/policy/residency/train.py",
               "scripts_cosim/residency_worker.py", "scripts_cosim/residency_study.py",
               "scripts_cosim/evaluate_residency.py", "scripts_cosim/audit_residency.py"]
    write(OUTPUT / "source_manifest.json", {p: sha(ROOT / p) for p in sources})
    for source in sources:
        destination = OUTPUT / "source_snapshot" / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / source).read_bytes())
    print(json.dumps({"audit_pass": True, "runs": len(physical) + len(extra), "requests": events,
                      "exact_control": summary["exact_control"],
                      "attempts": {k: {"headroom": v["reference_headroom_pct"],
                                       "simulation_gnn_vs_search": v["simulation"]["gnn"]["median_pct_vs_search"],
                                       "physical_gnn_vs_search": v["physical"]["gnn"]["median_pct_vs_search"],
                                       "prediction_error": v["physical_prediction_median_abs_error_pct"]}
                                   for k, v in summary["attempts"].items()}}, indent=2))


if __name__ == "__main__":
    main()
