"""Independently check a completed shuffle screen without fitting or selecting models."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from scripts_cosim.shuffle_placement_s0 import replay, topology, write_json


def audit(directory):
    p = Path(directory)
    protocol = json.loads((p / "protocol.json").read_text())
    result = json.loads((p / "results.json").read_text())
    scores = json.loads((p / "exact_scores.json").read_text())
    raw = json.loads((p / "measurements.json").read_text())["traces"]
    assert len(raw) == 12 and len(scores["plans"]) == 256 and len(scores["records"]) == 64
    assert len({tuple(plan) for plan in scores["plans"]}) == 256
    assert set(protocol["development_topologies"]).isdisjoint(protocol["holdout_topologies"])
    assert all(len(row["costs"]) == 256 and len(row["features"]) == 256 for row in scores["records"])
    assert all(np.isfinite(row["costs"]).all() and min(row["costs"]) > 0 for row in scores["records"])
    paired = result["paired"]
    for row in paired:
        original = next(x for x in scores["records"] if
                        (x["topology"], x["trace_seed"]) == (row["topology"], row["trace_seed"]))
        assert row["cost_s"]["bounded_reference"] == min(original["costs"])
        order = np.argsort(original["proxy"], kind="stable")
        assert row["cost_s"]["route_search"] == min(original["costs"][i] for i in order[:16])
        for arm in ("route_search", "ridge_search", "extra_trees_search"):
            assert row["cost_s"][arm] <= row["cost_s"]["route_proxy"] + 1e-12
    witness = scores["records"][-1]
    trace = sorted([t for t in raw if t["seed"] == witness["trace_seed"]], key=lambda x: x["wall_s"])[1]
    plan = scores["plans"][int(np.argmin(witness["costs"]))]
    assert replay([(0, trace, plan)], topology(witness["topology"], protocol))[0] == min(witness["costs"])
    source = p / "source_screen.py"
    if not source.exists():
        source = Path("scripts_cosim/shuffle_placement_s0.py")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == json.loads(
        (p / "provenance_screen.json").read_text())["script_sha256"]
    if source != p / "source_screen.py":
        shutil.copyfile(source, p / "source_screen.py")
    with (p / "exact_scores.json").open("rb") as src, gzip.open(p / "exact_scores.json.gz", "wb") as dst:
        shutil.copyfileobj(src, dst)
    assert gzip.open(p / "exact_scores.json.gz", "rb").read() == (p / "exact_scores.json").read_bytes()
    gaps = [x["cost_s"]["route_search"] - x["cost_s"]["bounded_reference"] for x in paired]
    report = {
        "measurements": len(raw), "static_instances": len(scores["records"]), "placements_per_instance": 256,
        "exact_plan_scores": sum(len(row["costs"]) for row in scores["records"]),
        "heldout_instances": len(paired), "route_search_exact_matches": sum(gap < 1e-12 for gap in gaps),
        "max_route_search_gap_s": max(gaps),
        "max_route_search_regret_percent": max(100 * gap / row["cost_s"]["bounded_reference"]
                                                for gap, row in zip(gaps, paired)),
        "median_configuration_oracle_gain_percent": float(np.median(
            [x["oracle_gain_percent"] for x in result["per_topology"]])),
        "stream_runs": len(result["stream_replay"]),
        "stream_jobs": sum(len(x["durations_s"]) for x in result["stream_replay"]),
        "witness_replay_bit_identical": True, "compressed_scores_roundtrip": True,
        "closure": "ACTIVE; adaptive live policy and network validation not run",
    }
    write_json(p / "audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    audit(parser.parse_args().directory)
