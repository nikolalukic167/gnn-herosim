"""Audit real namespace executions, causal snapshots, and paired gate summaries."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts_cosim.shuffle_network_gate import select
from scripts_cosim.shuffle_placement_s0 import write_json


def audit(directory):
    directory = Path(directory)
    protocol = json.loads((directory / "protocol.json").read_text())
    provenance = json.loads((directory / "provenance.json").read_text())
    assert hashlib.sha256((directory / "source_network_gate.py").read_bytes()).hexdigest() == provenance["source_sha256"]
    assert hashlib.sha256(Path("scripts_cosim/shuffle_network_gate.py").read_bytes()).hexdigest() == provenance["source_sha256"]
    parent_dir = Path("docs/lineages/shuffle_placement_s0_v1/run_2026-10-01")
    assert hashlib.sha256((parent_dir / "measurements.json").read_bytes()).hexdigest() == provenance["measurements_sha256"]
    raw = json.loads((parent_dir / "measurements.json").read_text())["traces"]
    traces = [sorted([t for t in raw if t["seed"] == seed], key=lambda x: x["wall_s"])[1]
              for seed in sorted({t["seed"] for t in raw})]
    totals = {}
    for path in sorted((directory / "payloads").glob("*.bin")):
        records = np.fromfile(path, dtype=np.uint32).reshape(-1, 2)
        totals[tuple(map(int, path.stem.split("_")))] = {
            "bytes": path.stat().st_size, "rows": len(records),
            "key_sum": int(records[:, 0].sum()), "value_sum": int(records[:, 1].sum())}
    rows, calibration, replayed, changed = [], [], 0, 0
    total_flows, total_bytes = 0, 0
    for seed in protocol["configuration_seeds"]:
        folder = directory / str(seed)
        topo = json.loads((folder / "topology.json").read_text())
        measurements = json.loads((folder / "calibration.json").read_text())
        assert len(measurements) == 6 * protocol["calibration_repeats"]
        calibration.append({"seed": seed,
            "median_absolute_relative_error": float(np.median([m["absolute_relative_error"] for m in measurements])),
            "per_pattern_median_error": {case: float(np.median([m["absolute_relative_error"] for m in measurements if m["case"] == case]))
                for case in sorted({m["case"] for m in measurements})}})
        for repeat in range(protocol["repeats"]):
            for interval in protocol["intervals_seconds"]:
                for arm in protocol["arms"]:
                    record = json.loads((folder / f"{repeat}_{interval}_{arm}.json").read_text())
                    decisions = record["decisions"]
                    assert len(decisions) == protocol["jobs_per_stream"]
                    assert len(record["flows"]) == len(decisions) * 16
                    assert len(record["completed"]) == len(decisions) * 4
                    history, allowed_flows, allowed_groups, durations = [], set(), set(), []
                    for j, decision in enumerate(decisions):
                        trace = traces[j % len(traces)]
                        snapshot = decision["snapshot"]
                        assert set(snapshot["progress"]) <= allowed_flows
                        assert set(snapshot["completed"]) <= allowed_groups
                        assert decision["controller_lag_s"] >= -1e-9
                        expected, active = select(trace, topo, history, snapshot, arm, protocol["exact_forecast_budget"])
                        assert list(expected) == decision["plan"]
                        assert active == decision["observed_active_jobs"]
                        static, _ = select(trace, topo, [], snapshot, "static_route", protocol["exact_forecast_budget"])
                        changed += int(arm == "adaptive_route" and list(static) != decision["plan"])
                        replayed += 1
                        selected = decision["arrival"] + decision["controller_lag_s"] + decision["decision_s"]
                        ready = [max(selected, decision["arrival"] + t) for t in trace["ready_s"]]
                        for r in range(4):
                            key = f"job:{decision['id']}:{r}"
                            actual = record["completed"][key]
                            assert actual["ready"] <= actual["compute_start"] <= actual["end"]
                            for field in ("rows", "key_sum", "value_sum"):
                                assert actual[field] == sum(totals[trace["seed"], m, r][field] for m in range(4))
                            allowed_groups.add(key)
                            for m in range(4):
                                flow_key = f"job:{decision['id']}:{m}:{r}"
                                flow = record["flows"][flow_key]
                                assert flow["received"] == flow["bytes"] == totals[trace["seed"], m, r]["bytes"]
                                assert flow["start"] >= ready[m] - 1e-6
                                assert flow["start"] <= flow["end"] <= actual["ready"]
                                total_flows += 1
                                total_bytes += flow["bytes"]
                                allowed_flows.add(flow_key)
                        durations.append(max(record["completed"][f"job:{decision['id']}:{r}"]["end"] for r in range(4)) - decision["arrival"])
                        history.append({"id": decision["id"], "trace": trace, "plan": decision["plan"], "data_ready": ready})
                    assert np.allclose(durations, record["durations_s"], rtol=0, atol=1e-12)
                    assert abs(float(np.mean(durations)) - record["mean_job_s"]) < 1e-12
                    rows.append({k: record[k] for k in ("seed", "repeat", "arm", "interval_s", "mean_job_s", "median_decision_s")})
    paired = []
    for interval in protocol["intervals_seconds"]:
        for arm in protocol["arms"][1:]:
            by_seed = []
            for seed in protocol["configuration_seeds"]:
                differences = []
                for repeat in range(protocol["repeats"]):
                    def get(which):
                        return next(r for r in rows if (r["seed"], r["repeat"], r["interval_s"], r["arm"]) == (seed, repeat, interval, which))["mean_job_s"]
                    differences.append(100 * (get(arm) / get("static_route") - 1))
                by_seed.append({"seed": seed, "median_paired_percent": float(np.median(differences)), "repeat_percent": differences})
            paired.append({"interval_s": interval, "arm": arm, "per_configuration": by_seed,
                           "median_paired_percent": float(np.median([x["median_paired_percent"] for x in by_seed])),
                           "configurations_faster": sum(x["median_paired_percent"] < 0 for x in by_seed)})
    report = {"runs": len(rows), "jobs": replayed, "flows": total_flows, "bytes": total_bytes,
        "adaptive_route_decisions_differing_from_static": changed,
        "decision_replay_and_causality_pass": True, "byte_timing_and_aggregation_pass": True,
        "calibration": calibration, "paired": paired,
        "decision_median_s": {arm: float(np.median([r["median_decision_s"] for r in rows if r["arm"] == arm])) for arm in protocol["arms"]},
        "statistical_scope": "four network configurations, repeated runs are not independent; exploratory directions, not a significance claim",
        "gnn_training_qualified": False}
    write_json(directory / "audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    audit(parser.parse_args().directory)
